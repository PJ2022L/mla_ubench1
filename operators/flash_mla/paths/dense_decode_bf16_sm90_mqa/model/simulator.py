"""Atom-only discrete-event prediction for the global dense-decode DAG."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import heapq
import json
import math
from typing import Any, Mapping

from .cost_database import CostDatabase
from .dag import DenseDecodeDAG, OperationNode
from .resources import ResourceModel
from .schema import KernelResources


EventKey = tuple[str, str]


@dataclass(frozen=True)
class EventTiming:
    start: float
    finish: float
    predecessor: EventKey | None
    duration: float


@dataclass(frozen=True)
class ScheduledNode:
    issue_start: float
    issue_finish: float
    service_start: float
    complete_finish: float
    isolated_latency: float
    initiation_interval: float
    slowdown: float
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class Prediction:
    result: Mapping[str, Any]
    dag: DenseDecodeDAG


def _event_time(events: Mapping[EventKey, EventTiming], key: EventKey) -> float:
    return events[key].finish


def _latest(candidates: list[tuple[float, EventKey | None]]) -> tuple[float, EventKey | None]:
    if not candidates:
        return 0.0, None
    return max(candidates, key=lambda item: (item[0], str(item[1])))


def _dedupe_provenance(rows: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    seen: set[str] = set()
    result: list[Mapping[str, Any]] = []
    for row in rows:
        key = json.dumps(row, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            result.append(dict(row))
    return result


def _critical_path_entry(
    dag: DenseDecodeDAG, node_id: str, internal_event: str,
) -> Mapping[str, Any]:
    """Translate scheduler-only service events into the public event schema."""
    result: dict[str, Any] = {
        "node_id": node_id,
        "event": internal_event,
        "phase": dag.nodes[node_id].phase,
    }
    if internal_event in {"issue", "complete"}:
        return result
    result["event"] = "service"
    if internal_event == "isolated_service":
        result["service_resource"] = dag.nodes[node_id].resource_class
        result["service_kind"] = "isolated_latency"
        return result
    resource, separator, encoded_unit = internal_event.rpartition("_service[")
    if not separator or not encoded_unit.endswith("]"):
        raise RuntimeError(f"unknown internal scheduler event: {internal_event}")
    unit = encoded_unit[:-1]
    result["service_resource"] = resource
    result["service_kind"] = "queue"
    result["service_unit"] = int(unit) if unit.isdigit() else unit
    return result


def _cache_order_key(node: OperationNode, placement: Any) -> tuple[Any, ...]:
    """Stable tie-break for cache accesses with the same modeled start time."""
    placement_key = (
        2 if placement is None else (0 if placement.kernel == "main" else 1),
        -1 if placement is None else placement.wave,
        -1 if placement is None else placement.sm_id,
        -1 if placement is None else placement.resident_slot,
        node.cta_id or "",
    )
    return (
        placement_key, node.actor, node.phase, node.source_anchor, node.atom_id,
        json.dumps(node.benchmark_params, sort_keys=True, default=str),
    )


def _wave_predecessors(dag: DenseDecodeDAG, resources: ResourceModel) -> dict[str, list[str]]:
    roots: dict[str, list[str]] = defaultdict(list)
    terminals: dict[str, list[str]] = defaultdict(list)
    incoming = defaultdict(int)
    outgoing = defaultdict(int)
    for edge in dag.dependencies:
        if dag.nodes[edge.src].cta_id == dag.nodes[edge.dst].cta_id:
            outgoing[edge.src] += 1
            incoming[edge.dst] += 1
    for node in dag.nodes.values():
        if not node.cta_id:
            continue
        if incoming[node.id] == 0:
            roots[node.cta_id].append(node.id)
        if outgoing[node.id] == 0:
            terminals[node.cta_id].append(node.id)
    by_slot: dict[tuple[str, int, int], list[Any]] = defaultdict(list)
    for placement in resources.placements.values():
        by_slot[(placement.kernel, placement.sm_id, placement.resident_slot)].append(placement)
    result: dict[str, list[str]] = defaultdict(list)
    for placements in by_slot.values():
        ordered = sorted(placements, key=lambda item: item.wave)
        for previous, current in zip(ordered, ordered[1:]):
            for root in roots[current.cta_id]:
                result[root].extend(terminals[previous.cta_id])

    # A dependent-grid CTA cannot execute even its pre-wait preamble until a
    # resident slot is available. Map the first combine wave to the matching
    # *last* main wave on each SM/slot. Depending only on main wave zero would
    # let a later persistent-main wave and combine occupy the same abstract
    # slot concurrently (illegal for the ~230 KiB main shared-memory footprint).
    # Later combine waves are already chained by the per-kernel slot rule.
    # Dense main normally has residency 1; modulo covers measured residency >1.
    main_residency = resources.resources.residency("main")
    last_main: dict[tuple[int, int], Any] = {}
    for placement in resources.placements.values():
        if placement.kernel != "main":
            continue
        key = (placement.sm_id, placement.resident_slot)
        prior = last_main.get(key)
        if prior is None or placement.wave > prior.wave:
            last_main[key] = placement
    for placement in resources.placements.values():
        if placement.kernel != "combine" or placement.wave != 0:
            continue
        predecessor = last_main.get(
            (placement.sm_id, placement.resident_slot % main_residency)
        )
        if predecessor is None:
            continue
        for root in roots[placement.cta_id]:
            result[root].extend(terminals[predecessor.cta_id])
    return result


def _throughput_per_cycle(value: float, unit: str, clock_mhz: float) -> float:
    lowered = unit.lower().replace(" ", "")
    if value <= 0:
        return 0.0
    if "/cycle" in lowered or "percycle" in lowered:
        return value
    if lowered.startswith("t") and "/s" in lowered:
        return value * 1_000_000.0 / clock_mhz
    # Includes Gop/s, Glane-op/s, Gsource-op/s, Gtile/s and the other
    # kernel-agnostic grid-rate units emitted by the family harnesses.
    if lowered.startswith("g") and "/s" in lowered and not lowered.startswith("gb"):
        return value * 1_000.0 / clock_mhz
    if lowered.startswith("m") and "/s" in lowered:
        return value / clock_mhz
    return 0.0


def _service_cycles(
    node: OperationNode, cost: Any, resources: KernelResources, quantile: str,
) -> tuple[float, float]:
    base = {
        "p10": cost.p10_cycles,
        "p50": cost.p50_cycles,
        "p90": cost.p90_cycles,
    }[quantile]
    amount = max(node.work_amount, 0.0)
    latency = base
    ii = cost.initiation_interval_cycles
    if node.work_unit in {"committed_group", "dependency_chain"}:
        latency = base + max(0.0, amount - 1.0) * ii
    else:
        rate = _throughput_per_cycle(
            cost.throughput_value, cost.throughput_unit, resources.sm_clock_mhz
        )
        if rate > 0:
            latency = max(base, amount / rate)
        else:
            # The fallback retains measured latency/II semantics and is
            # surfaced through the missing resource-curve warnings.
            latency = base + max(0.0, amount - 1.0) * ii
    issue = min(latency, ii if amount <= 1 else max(ii, latency - base + ii))
    return latency, issue


def _schedule_with_cache_routes(
    dag: DenseDecodeDAG,
    database: CostDatabase,
    kernel_resources: KernelResources,
    quantile: str,
    slowdown_overrides: Mapping[str, float] | None = None,
    cache_routes: Mapping[str, bool] | None = None,
) -> tuple[Prediction, dict[str, ScheduledNode], ResourceModel]:
    dag.validate()
    if dag.scheduler is not None and not dag.scheduler.source_defined:
        raise ValueError(
            "official prediction rejected a scheduler case that is undefined in the "
            f"upstream metadata kernel: {dag.scheduler.undefined_reason}"
        )
    database.ensure_coverage(node.atom_id for node in dag.nodes.values())
    cta_ids = sorted({node.cta_id for node in dag.nodes.values() if node.cta_id})
    resource_model = ResourceModel(
        database, kernel_resources, cta_ids,
        {node.resource_class for node in dag.nodes.values()},
    )
    wave_predecessors = _wave_predecessors(dag, resource_model)
    incoming_issue: dict[str, list[tuple[str, str]]] = defaultdict(list)
    incoming_complete: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for edge in dag.dependencies:
        target = incoming_issue if edge.dst_event == "issue" else incoming_complete
        target[edge.dst].append((edge.src, edge.src_event))

    events: dict[EventKey, EventTiming] = {}
    scheduled: dict[str, ScheduledNode] = {}
    resource_available: dict[tuple[str, int | str], tuple[float, EventKey | None]] = {}
    resource_busy: dict[tuple[str, int | str], float] = defaultdict(float)
    resource_keys_used: set[tuple[str, int | str]] = set()
    predecessor_nodes: dict[str, set[str]] = {
        node_id: set() for node_id in dag.nodes
    }
    successor_nodes: dict[str, set[str]] = {
        node_id: set() for node_id in dag.nodes
    }
    for edge in dag.dependencies:
        if edge.src != edge.dst:
            predecessor_nodes[edge.dst].add(edge.src)
            successor_nodes[edge.src].add(edge.dst)
    for node_id, predecessors in wave_predecessors.items():
        for predecessor in predecessors:
            if predecessor != node_id:
                predecessor_nodes[node_id].add(predecessor)
                successor_nodes[predecessor].add(node_id)

    def ready_key(candidate: str) -> tuple[float, int, str]:
        times = [
            _event_time(events, (src, event))
            for src, event in incoming_issue[candidate]
        ]
        times.extend(
            _event_time(events, (src, "complete"))
            for src in wave_predecessors.get(candidate, [])
        )
        node = dag.nodes[candidate]
        # At the same dependency-ready time, issue async work first. This
        # lets persistent next-Q launch overlap the previous epilogue and
        # avoids making construction/node-id order an implicit edge.
        return (max(times, default=0.0), 0 if node.async_issue else 1, candidate)

    remaining_predecessors = {
        node_id: len(predecessors)
        for node_id, predecessors in predecessor_nodes.items()
    }
    ready: list[tuple[float, int, str]] = [
        ready_key(node_id)
        for node_id, degree in remaining_predecessors.items()
        if degree == 0
    ]
    heapq.heapify(ready)
    processed = 0
    while ready:
        _, _, node_id = heapq.heappop(ready)
        processed += 1
        node: OperationNode = dag.nodes[node_id]
        placement = resource_model.placement(node.cta_id)
        query = resource_model.benchmark_query(
            node.benchmark_params, node.cta_id, node.atom_id
        )
        cost = database.lookup(
            node.atom_id, query,
        )
        isolated_latency, isolated_ii = _service_cycles(
            node, cost, kernel_resources, quantile
        )
        issue_candidates: list[tuple[float, EventKey | None]] = []
        for src, event in incoming_issue[node_id]:
            issue_candidates.append((_event_time(events, (src, event)), (src, event)))
        for src in wave_predecessors.get(node_id, []):
            issue_candidates.append((_event_time(events, (src, "complete")), (src, "complete")))
        plan = resource_model.access_plan(
            node.resource_class, node.atom_id, node.benchmark_params, node.cta_id,
            forced_cache_hit=(
                cache_routes[node_id]
                if cache_routes is not None and node_id in cache_routes else None
            ),
        )
        sm_id = placement.sm_id if placement else 0
        slowdown = max(
            resource_model.service_slowdown(
                node.resource_class, node.benchmark_params, node.cta_id, set()
            ),
            float((slowdown_overrides or {}).get(node_id, 1.0)),
        )
        latency = isolated_latency * slowdown
        ii = isolated_ii * slowdown

        local_keys = plan.local_keys
        if node.async_issue and not local_keys:
            # Memory-only asynchronous instructions still consume a per-SM
            # issue path; their byte service is accounted separately below.
            local_keys = (("issue", sm_id),)
        for key in local_keys:
            issue_candidates.append(resource_available.get(key, (0.0, None)))

        # Global-memory instructions have a short per-SM LSU issue interval
        # and an independently completing dependency/memory service.  Treat
        # them like other pipelined operations even when the source-level DAG
        # node is not marked async (its data consumers still depend on the
        # complete event).
        pipelined_issue = node.async_issue or bool(
            plan.direction and plan.memory_keys
        )
        if pipelined_issue:
            issue_start, issue_predecessor = _latest(issue_candidates)
            issue_finish = issue_start + ii
            events[(node_id, "issue")] = EventTiming(
                issue_start, issue_finish, issue_predecessor, ii
            )
            for key in local_keys:
                resource_available[key] = (issue_finish, (node_id, "issue"))
                resource_busy[key] += ii
                resource_keys_used.add(key)
            tail_key = (node_id, "isolated_service")
            tail_duration = max(latency - ii, 0.0)
            events[tail_key] = EventTiming(
                issue_finish, issue_finish + tail_duration,
                (node_id, "issue"), tail_duration,
            )
            completion_options: list[tuple[float, EventKey | None]] = [
                (events[tail_key].finish, tail_key)
            ]
            memory_earliest = (issue_finish, (node_id, "issue"))
            service_start = issue_start
        else:
            issue_start, issue_predecessor = _latest(issue_candidates)
            events[(node_id, "issue")] = EventTiming(
                issue_start, issue_start, issue_predecessor, 0.0
            )
            completion_options = []
            if local_keys:
                for key in local_keys:
                    local_event = (
                        node_id, f"{key[0]}_service[{key[1]}]"
                    )
                    local_finish = issue_start + latency
                    events[local_event] = EventTiming(
                        issue_start, local_finish, (node_id, "issue"), latency
                    )
                    completion_options.append((local_finish, local_event))
                    resource_available[key] = (local_finish, local_event)
                    resource_busy[key] += latency
                    resource_keys_used.add(key)
            else:
                isolated_event = (node_id, "isolated_service")
                events[isolated_event] = EventTiming(
                    issue_start, issue_start + latency,
                    (node_id, "issue"), latency,
                )
                completion_options.append((events[isolated_event].finish, isolated_event))
            memory_earliest = (issue_start, (node_id, "issue"))
            issue_finish = issue_start
            service_start = issue_start

        memory_request_cycle = memory_earliest[0]
        memory_stage_earliest = memory_earliest
        l2_service_finish: float | None = None
        for key in plan.memory_keys:
            available = resource_available.get(key, (0.0, None))
            # A miss traverses HBM then L2; a store enters L2 then drains to
            # HBM.  Stages from one request are ordered, while different
            # requests can pipeline through the independent global queues.
            memory_start, memory_predecessor = _latest(
                [memory_stage_earliest, available]
            )
            memory_cycles = resource_model.memory_service_cycles(plan, key, cost) * slowdown
            memory_event = (node_id, f"{key[0]}_service[{key[1]}]")
            memory_finish = memory_start + memory_cycles
            events[memory_event] = EventTiming(
                memory_start, memory_finish, memory_predecessor, memory_cycles
            )
            completion_options.append((memory_finish, memory_event))
            resource_available[key] = (memory_finish, memory_event)
            resource_busy[key] += memory_cycles
            resource_keys_used.add(key)
            memory_stage_earliest = (memory_finish, memory_event)
            if key[0] == "l2":
                l2_service_finish = memory_finish
        if l2_service_finish is not None:
            resource_model.record_cache_access(
                node_id, memory_request_cycle, l2_service_finish,
                _cache_order_key(node, placement), plan,
            )
        completion_options.extend(
            (_event_time(events, (src, event)), (src, event))
            for src, event in incoming_complete[node_id]
        )
        completion_finish, completion_predecessor = _latest(completion_options)
        events[(node_id, "complete")] = EventTiming(
            completion_finish, completion_finish, completion_predecessor, 0.0
        )
        scheduled[node_id] = ScheduledNode(
            issue_start, issue_finish, service_start, completion_finish, isolated_latency,
            isolated_ii, slowdown, cost.provenance,
        )

        for successor in successor_nodes[node_id]:
            remaining_predecessors[successor] -= 1
            if remaining_predecessors[successor] == 0:
                heapq.heappush(ready, ready_key(successor))

    if processed != len(dag.nodes):
        raise RuntimeError("event list scheduler found no dependency-ready DAG node")

    end_key, makespan = max(
        (((node_id, "complete"), events[(node_id, "complete")].finish)
         for node_id in dag.nodes),
        key=lambda item: item[1], default=(("", "complete"), 0.0),
    )
    critical_events: list[EventKey] = []
    current: EventKey | None = end_key if end_key[0] else None
    while current is not None:
        critical_events.append(current)
        current = events[current].predecessor
    critical_events.reverse()
    contribution: dict[str, float] = defaultdict(float)
    for node_id, event in critical_events:
        contribution[dag.nodes[node_id].phase] += events[(node_id, event)].duration
    phase_members: dict[str, list[ScheduledNode]] = defaultdict(list)
    for node_id, timing in scheduled.items():
        phase_members[dag.nodes[node_id].phase].append(timing)
    wall_spans = {}
    for phase, members in sorted(phase_members.items()):
        wall_spans[phase] = {
            "start_cycle": min(item.issue_start for item in members),
            "end_cycle": max(item.complete_finish for item in members),
            "wall_span_cycles": max(item.complete_finish for item in members)
                                - min(item.issue_start for item in members),
            "critical_path_contribution_cycles": contribution.get(phase, 0.0),
        }
    contribution_total = sum(item["critical_path_contribution_cycles"] for item in wall_spans.values())
    if not math.isclose(contribution_total, makespan, rel_tol=1e-10, abs_tol=1e-7):
        raise RuntimeError(
            f"critical-path attribution mismatch: {contribution_total} != {makespan}"
        )
    keys_by_resource: dict[str, set[tuple[str, int | str]]] = defaultdict(set)
    busy_by_resource: dict[str, float] = defaultdict(float)
    for key in resource_keys_used:
        keys_by_resource[key[0]].add(key)
        busy_by_resource[key[0]] += resource_busy[key]
    utilization: dict[str, float] = {}
    utilization_capacity: dict[str, Mapping[str, Any]] = {}
    for resource, keys in sorted(keys_by_resource.items()):
        parallel_units = len(keys)
        available_cycles = makespan * parallel_units
        value = (
            busy_by_resource[resource] / available_cycles
            if available_cycles > 0 else 0.0
        )
        utilization[resource] = max(0.0, min(1.0, value))
        active_sms = len({key[1] for key in keys if isinstance(key[1], int)})
        utilization_capacity[resource] = {
            "scope": "per_sm" if active_sms else "gpu",
            "active_sm": active_sms,
            "total_sm": kernel_resources.sm_count,
            "active_sm_fraction": (
                active_sms / kernel_resources.sm_count if active_sms else 1.0
            ),
            "parallel_units": parallel_units,
            "busy_cycles": busy_by_resource[resource],
            "available_cycles": available_cycles,
        }
    wave_summary = {}
    for kernel in ("main", "combine"):
        placements = [item for item in resource_model.placements.values()
                      if item.kernel == kernel]
        wave_count = max((item.wave for item in placements), default=-1) + 1
        tail_count = sum(item.wave == wave_count - 1 for item in placements) if wave_count else 0
        wave_summary[kernel] = {
            "cta_count": len(placements), "wave_count": wave_count,
            "tail_cta_count": tail_count,
            "residency_per_sm": kernel_resources.residency(kernel),
        }
    split_distribution: dict[str, int] = defaultdict(int)
    if dag.scheduler:
        prefix = dag.scheduler.num_splits_prefix
        for index in range(len(prefix) - 1):
            split_distribution[str(prefix[index + 1] - prefix[index])] += 1
    result = {
        "schema_version": 1,
        "case_id": dag.workload.case_id if dag.workload else None,
        "boundary": "gpu_metadata_main_combine",
        "predicted_e2e_cycles": makespan,
        "predicted_e2e_us": makespan / kernel_resources.sm_clock_mhz,
        "phase_timing": wall_spans,
        "critical_path": [
            _critical_path_entry(dag, node_id, event)
            for node_id, event in critical_events
        ],
        "resource_utilization": utilization,
        "resource_capacity": utilization_capacity,
        "memory_traffic_bytes": dict(resource_model.traffic_bytes),
        "cache_events": dict(resource_model.cache_events),
        "warnings": [
            f"no generic resource_curve for {resource}; isolated atom service rate used"
            for resource in sorted(resource_model.missing_resource_curves)
        ],
        "scheduler": dag.scheduler.to_json() if dag.scheduler else None,
        "cta_placement": resource_model.placement_json(),
        "cta_waves": wave_summary,
        "split_distribution": dict(sorted(split_distribution.items())),
        "kernel_resources": kernel_resources.to_json(),
        "atom_provenance": _dedupe_provenance([
            row
            for timing in scheduled.values()
            for row in timing.provenance.get("selected_rows", ())
        ]),
        "resource_curve_provenance": _dedupe_provenance(
            resource_model.used_resource_provenance
        ),
        "node_count": len(dag.nodes),
        "dependency_count": len(dag.dependencies),
        "calibration_used": False,
    }
    return Prediction(result, dag), scheduled, resource_model


def _schedule_once(
    dag: DenseDecodeDAG,
    database: CostDatabase,
    kernel_resources: KernelResources,
    quantile: str,
    slowdown_overrides: Mapping[str, float] | None = None,
) -> tuple[Prediction, dict[str, ScheduledNode], ResourceModel]:
    """Schedule and replay cache routing in modeled L2-service order."""
    routes: dict[str, bool] | None = None
    latest: tuple[Prediction, dict[str, ScheduledNode], ResourceModel] | None = None
    for _ in range(8):
        latest = _schedule_with_cache_routes(
            dag, database, kernel_resources, quantile,
            slowdown_overrides, routes,
        )
        prediction, _, resource_model = latest
        resolved = resource_model.resolved_cache_routes()
        if resolved == resource_model.actual_cache_routes():
            return latest
        routes = resolved
    if latest is None:
        raise RuntimeError("cache-order solver did not execute")
    prediction, scheduled, resource_model = latest
    result = dict(prediction.result)
    result["warnings"] = list(result.get("warnings", ())) + [
        "L2 cache-order fixed point did not converge after 8 iterations"
    ]
    return Prediction(result, dag), scheduled, resource_model


def _segmented_interaction_slowdowns(
    dag: DenseDecodeDAG,
    scheduled: Mapping[str, ScheduledNode],
    resource_model: ResourceModel,
    current_overrides: Mapping[str, float],
) -> dict[str, float]:
    """Integrate measured interaction factors over actual same-SM overlap segments.

    Every task participating in a measured mixed-resource segment receives the
    same segment factor. This makes the correction independent of node
    construction order and symmetric between the already-running task and the
    task that entered the overlap later.
    """
    intervals_by_sm: dict[int, list[tuple[float, float, str]]] = defaultdict(list)
    durations: dict[str, float] = {}
    for node_id, timing in scheduled.items():
        duration = max(0.0, timing.complete_finish - timing.service_start)
        if duration <= 0:
            continue
        placement = resource_model.placement(dag.nodes[node_id].cta_id)
        sm_id = placement.sm_id if placement else 0
        intervals_by_sm[sm_id].append(
            (timing.service_start, timing.complete_finish, node_id)
        )
        durations[node_id] = duration

    lost_service: dict[str, float] = defaultdict(float)
    affected_span: dict[str, float] = defaultdict(float)
    peak_factor: dict[str, float] = defaultdict(lambda: 1.0)
    for intervals in intervals_by_sm.values():
        starts: dict[float, list[str]] = defaultdict(list)
        finishes: dict[float, list[str]] = defaultdict(list)
        for start, finish, node_id in intervals:
            starts[start].append(node_id)
            finishes[finish].append(node_id)
        active: set[str] = set()
        previous: float | None = None
        for point in sorted(set(starts) | set(finishes)):
            left, right = previous, point
            previous = point
            if left is None:
                active.update(starts[point])
                continue
            if right <= left:
                active.difference_update(finishes[point])
                active.update(starts[point])
                continue
            tensor_nodes = [
                node_id for node_id in active
                if dag.nodes[node_id].resource_class == "tensor"
            ]
            if tensor_nodes:
                active_resources = {
                    dag.nodes[node_id].resource_class for node_id in active
                }
                peer_resources: set[str] = set()
                affected_resources = {"tensor"}
                if len(tensor_nodes) >= 2:
                    peer_resources.add("tensor")
                if "tma" in active_resources:
                    peer_resources.add("tma")
                    affected_resources.add("tma")
                if {"sfu", "shared"} <= active_resources:
                    peer_resources.update(("sfu", "shared"))
                    affected_resources.update(("sfu", "shared"))
                if peer_resources:
                    factor = max(
                        resource_model.interaction_slowdown(
                            "tensor", dag.nodes[node_id].benchmark_params,
                            dag.nodes[node_id].cta_id, peer_resources,
                        )
                        for node_id in tensor_nodes
                    )
                    if factor > 1.0:
                        span = right - left
                        for node_id in active:
                            if dag.nodes[node_id].resource_class in affected_resources:
                                lost_service[node_id] += span * (1.0 - 1.0 / factor)
                                affected_span[node_id] += span
                                peak_factor[node_id] = max(peak_factor[node_id], factor)
            active.difference_update(finishes[point])
            active.update(starts[point])
    result = {}
    for node_id, duration in durations.items():
        if affected_span.get(node_id, 0.0) >= duration - 1e-9:
            result[node_id] = peak_factor[node_id]
            continue
        base_duration = duration / max(current_overrides.get(node_id, 1.0), 1.0)
        result[node_id] = 1.0 + lost_service.get(node_id, 0.0) / max(
            base_duration, 1e-12
        )
    return result


def _simulate_quantile(
    dag: DenseDecodeDAG,
    database: CostDatabase,
    kernel_resources: KernelResources,
    quantile: str,
) -> Prediction:
    overrides: dict[str, float] = {}
    converged = False
    max_delta = 0.0
    iterations = 0
    prediction: Prediction | None = None
    resource_model: ResourceModel | None = None
    for iterations in range(1, 9):
        prediction, scheduled, resource_model = _schedule_once(
            dag, database, kernel_resources, quantile, overrides
        )
        target = _segmented_interaction_slowdowns(
            dag, scheduled, resource_model, overrides
        )
        keys = set(overrides) | set(target)
        max_delta = max(
            (abs(target.get(key, 1.0) - overrides.get(key, 1.0)) for key in keys),
            default=0.0,
        )
        if max_delta <= 1e-6:
            converged = True
            break
        overrides = {key: max(1.0, value) for key, value in target.items()}
    if prediction is None or resource_model is None:
        raise RuntimeError("interaction solver did not execute")
    if not converged:
        prediction, scheduled, resource_model = _schedule_once(
            dag, database, kernel_resources, quantile, overrides
        )
        _segmented_interaction_slowdowns(
            dag, scheduled, resource_model, overrides
        )
    result = dict(prediction.result)
    result["resource_curve_provenance"] = _dedupe_provenance(
        resource_model.used_resource_provenance
    )
    warnings = list(result.get("warnings", []))
    if not converged:
        warnings.append(
            "mixed-resource fixed-point did not converge after 8 iterations"
        )
    result["warnings"] = warnings
    result["interaction_solver"] = {
        "method": "same_sm_segmented_symmetric_fixed_point",
        "iterations": iterations,
        "converged": converged,
        "max_factor_delta": max_delta,
    }
    return Prediction(result, dag)


def simulate(
    dag: DenseDecodeDAG,
    database: CostDatabase,
    kernel_resources: KernelResources,
) -> Prediction:
    p50 = _simulate_quantile(dag, database, kernel_resources, "p50")
    p10 = _simulate_quantile(dag, database, kernel_resources, "p10")
    p90 = _simulate_quantile(dag, database, kernel_resources, "p90")
    result = dict(p50.result)
    cycles = {
        "p10": p10.result["predicted_e2e_cycles"],
        "p50": p50.result["predicted_e2e_cycles"],
        "p90": p90.result["predicted_e2e_cycles"],
    }
    result["predicted_e2e_cycles"] = cycles
    result["predicted_e2e_us"] = {
        key: value / kernel_resources.sm_clock_mhz for key, value in cycles.items()
    }
    return Prediction(result, dag)
