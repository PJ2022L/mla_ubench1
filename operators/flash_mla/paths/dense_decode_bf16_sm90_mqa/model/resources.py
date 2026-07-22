"""Measured service-curve, cache, and occupancy policy for DAG scheduling."""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
import heapq
import re
from typing import Any, Mapping

from .cost_database import CostDatabase, OperationCost
from .schema import KernelResources


ResourceKey = tuple[str, int | str]


@dataclass(frozen=True)
class CtaPlacement:
    cta_id: str
    kernel: str
    sm_id: int
    resident_slot: int
    wave: int


@dataclass(frozen=True)
class AccessPlan:
    local_keys: tuple[ResourceKey, ...]
    memory_keys: tuple[ResourceKey, ...]
    byte_count: float
    direction: str | None
    curve_resource: str | None
    curve_query: Mapping[str, Any]
    cache_key: tuple[Any, ...] | None = None
    requested_cache_mode: str | None = None
    fits_l2: bool = False
    cache_insert_bytes: int = 0
    cache_hit: bool | None = None


@dataclass(frozen=True)
class CacheAccess:
    node_id: str
    request_cycle: float
    fill_cycle: float
    order_key: tuple[Any, ...]
    cache_key: tuple[Any, ...] | None
    requested_cache_mode: str
    fits_l2: bool
    insert_bytes: int
    actual_hit: bool | None
    fill_only: bool = False


class ResourceModel:
    def __init__(
        self, database: CostDatabase, resources: KernelResources,
        cta_ids: list[str], active_resources: set[str],
    ) -> None:
        self.database = database
        self.resources = resources
        self.active_resources = active_resources
        self.placements = self._place(cta_ids)
        self.kernel_cta_counts = Counter(
            placement.kernel for placement in self.placements.values()
        )
        self.kernel_wave_counts = Counter(
            (placement.kernel, placement.wave)
            for placement in self.placements.values()
        )
        self.kernel_wave_sm_counts = Counter(
            (placement.kernel, placement.wave, placement.sm_id)
            for placement in self.placements.values()
        )
        wave_sms: dict[tuple[str, int], set[int]] = {}
        for placement in self.placements.values():
            wave_sms.setdefault(
                (placement.kernel, placement.wave), set()
            ).add(placement.sm_id)
        self.kernel_wave_active_sms = {
            key: len(sm_ids) for key, sm_ids in wave_sms.items()
        }
        self._l2_pages: OrderedDict[tuple[Any, ...], int] = OrderedDict()
        self._l2_resident_bytes = 0
        self.traffic_bytes = {"l2": 0.0, "hbm": 0.0}
        self.cache_events = {"l2_hit": 0, "hbm_miss": 0}
        self.missing_resource_curves: set[str] = set()
        self.used_resource_provenance: list[Mapping[str, Any]] = []
        self.cache_accesses: list[CacheAccess] = []

    def _place(self, cta_ids: list[str]) -> dict[str, CtaPlacement]:
        result: dict[str, CtaPlacement] = {}

        def launch_key(identifier: str) -> tuple[int, int, int]:
            main = re.fullmatch(r"main\.p(\d+)\.h(\d+)\.m(\d+)", identifier)
            if main:
                partition, head, m_block = map(int, main.groups())
                return partition, head, m_block
            combine = re.fullmatch(r"combine\.r(\d+)\.q(\d+)\.h(\d+)", identifier)
            if combine:
                request, q_index, head_block = map(int, combine.groups())
                return head_block, request, q_index
            return (0, 0, 0)

        by_kernel = {
            "main": sorted(
                (item for item in set(cta_ids) if item.startswith("main.")),
                key=launch_key,
            ),
            "combine": sorted(
                (item for item in set(cta_ids) if item.startswith("combine.")),
                key=launch_key,
            ),
        }
        for kernel, identifiers in by_kernel.items():
            residency = self.resources.residency(kernel)
            width = self.resources.sm_count * residency
            for index, cta_id in enumerate(identifiers):
                in_wave = index % width
                result[cta_id] = CtaPlacement(
                    cta_id, kernel, in_wave // residency, in_wave % residency,
                    index // width,
                )
        return result

    def placement(self, cta_id: str | None) -> CtaPlacement | None:
        return self.placements.get(cta_id or "")

    def benchmark_query(
        self, benchmark_params: Mapping[str, Any], cta_id: str | None,
        atom_id: str | None = None,
    ) -> dict[str, Any]:
        query = dict(benchmark_params)
        query.setdefault("device_sm_count", self.resources.sm_count)
        placement = self.placement(cta_id)
        if placement:
            # The event model schedules CTA waves explicitly.  Select service
            # curves for the node's actual wave, including a partially filled
            # last wave, instead of reusing the full-grid saturation point for
            # every tail CTA.
            blocks = self.kernel_wave_counts[(placement.kernel, placement.wave)]
            active_sm = self.kernel_wave_active_sms[
                (placement.kernel, placement.wave)
            ]
            resident = self.kernel_wave_sm_counts[
                (placement.kernel, placement.wave, placement.sm_id)
            ]
            query.setdefault("active_sm", active_sm)
            query.setdefault("resident_cta", resident)
            query.setdefault("blocks", blocks)
        else:
            query.setdefault("active_sm", 1)
            query.setdefault("resident_cta", 1)
            query.setdefault("blocks", 1)
        if atom_id is not None:
            query.setdefault("working_set_bytes", self._working_set_bytes(atom_id, query))
        return query

    def service_slowdown(
        self, resource: str, benchmark_params: Mapping[str, Any], cta_id: str | None,
        peer_resources: set[str] | None = None,
    ) -> float:
        return self.interaction_slowdown(
            resource, benchmark_params, cta_id, peer_resources or set()
        )

    def interaction_slowdown(
        self, resource: str, benchmark_params: Mapping[str, Any], cta_id: str | None,
        peer_resources: set[str],
    ) -> float:
        query = self.benchmark_query(benchmark_params, cta_id)
        peers = set(peer_resources or ())
        interaction_peers: list[str] = []
        factors: list[float] = []
        if resource == "tensor":
            if "tensor" in peers:
                interaction_peers.append("tensor")
            if "tma" in peers:
                interaction_peers.append("tma")
            if {"sfu", "shared"} <= peers:
                interaction_peers.append("sfu+shared")
        for peer in interaction_peers:
            interaction = self.database.resource_interaction(
                resource, query | {"peer_resource": peer, "actors": 2}
            )
            factors.append(interaction.slowdown)
            self.used_resource_provenance.extend(interaction.provenance)
        return max(factors, default=1.0)

    def _atom_family(self, atom_id: str) -> str:
        records = self.database.records.get(atom_id, ())
        return records[0].family if records else ""

    def _direction(self, atom_id: str) -> str | None:
        family = self._atom_family(atom_id)
        if family in {"global_load", "tma_load"}:
            return "load"
        if family in {"global_store", "tma_store", "bulk_store"}:
            return "store"
        lowered = atom_id.lower()
        if lowered.startswith("ld_"):
            return "load"
        if lowered.startswith("st_"):
            return "store"
        return None

    @staticmethod
    def _normalized_pattern(value: Any) -> str:
        pattern = str(value or "sequential").lower()
        return {
            "broadcast": "reuse",
            "contiguous": "sequential",
            "quad_broadcast": "reuse",
            "warp_broadcast": "reuse",
            "unique": "local",
        }.get(pattern, pattern)

    @staticmethod
    def _working_set_bytes(atom_id: str, params: Mapping[str, Any]) -> int:
        explicit = int(params.get("working_set_bytes", 0) or 0)
        if explicit > 0:
            return explicit
        entries = int(params.get("working_set_entries", 0) or 0)
        records = int(params.get("working_set_records", 0) or 0)
        words = int(params.get("working_set_words", 0) or 0)
        if entries:
            return entries * 4
        if words:
            return words * 4
        if records:
            if "v4_u32_32b" in atom_id:
                return records * 32
            if atom_id.endswith("u64"):
                return records * 8
            return records * 4
        rowsets = int(params.get("rowsets", 0) or 0)
        segments = int(params.get("segments", 0) or 0)
        if rowsets and segments:
            if "v4_f32" in atom_id:
                return rowsets * segments * 8 * 512 * 4
            split_stride = int(params.get("split_stride", 128) or 128)
            return rowsets * segments * split_stride * 4
        pages = int(params.get("working_set_pages", 0) or 0)
        if pages:
            return pages * 64 * 576 * 2
        tiles = int(params.get("working_set_tiles", 0) or 0)
        if tiles:
            element_bytes = 4 if "f32" in atom_id else 2
            return tiles * 64 * 512 * element_bytes
        return max(int(params.get("bytes", 0) or 0), 1)

    def _touch_l2_page(
        self, cache_key: tuple[Any, ...] | None, byte_count: int,
    ) -> bool:
        if cache_key is None or cache_key not in self._l2_pages:
            return False
        resident = self._l2_pages.pop(cache_key)
        self._l2_pages[cache_key] = resident
        return True

    def _insert_l2_page(
        self, cache_key: tuple[Any, ...] | None, byte_count: int,
    ) -> None:
        if cache_key is None:
            return
        previous = self._l2_pages.pop(cache_key, 0)
        self._l2_resident_bytes -= previous
        self._l2_pages[cache_key] = byte_count
        self._l2_resident_bytes += byte_count
        while self._l2_resident_bytes > self.resources.l2_bytes and self._l2_pages:
            _, evicted = self._l2_pages.popitem(last=False)
            self._l2_resident_bytes -= evicted

    @staticmethod
    def _cache_key(atom_id: str, params: Mapping[str, Any]) -> tuple[Any, ...] | None:
        memory_object = params.get("memory_object")
        if memory_object == "q":
            return (
                "q", int(params.get("request", 0)),
                int(params.get("kv_head", 0)), int(params.get("m_block", 0)),
            )
        if memory_object == "k":
            physical = params.get("physical_page")
            if physical is None:
                return None
            return (
                "k", int(physical), int(params.get("kv_head", 0)),
                int(params.get("tile", -1)),
            )
        if memory_object in {"partial_output", "partial_lse"}:
            # Main writes these buffers by global split/KV-head/m-block while
            # combine reads them by q-head.  Request is the common region
            # identity on both sides; working_set_bytes carries the complete
            # request-local footprint used for L2 retention.
            return (str(memory_object), int(params.get("request", 0)))
        cache_line = params.get("cache_line")
        if memory_object is not None and cache_line is not None:
            return (str(memory_object), int(cache_line))
        if params.get("physical_page") is not None:
            return (
                "k", int(params["physical_page"]),
                int(params.get("kv_head", 0)), int(params.get("tile", -1)),
            )
        return None

    def access_plan(
        self, resource: str, atom_id: str, benchmark_params: Mapping[str, Any],
        cta_id: str | None, *, forced_cache_hit: bool | None = None,
    ) -> AccessPlan:
        placement = self.placement(cta_id)
        sm = placement.sm_id if placement else 0
        byte_count = float(benchmark_params.get("bytes", 0) or 0)
        direction = self._direction(atom_id)
        query = self.benchmark_query(benchmark_params, cta_id, atom_id)
        query["pattern"] = self._normalized_pattern(query.get("pattern"))
        query.setdefault("outstanding_depth", int(query.get("depth", 1) or 1))
        query.setdefault("threads", int(query.get("threads", 256) or 256))

        if resource == "grid":
            # PREEXIT/ACQBULK are issued independently by resident CTAs.  A
            # single GPU-wide queue would serialize every launch/wait and can
            # dominate large combine grids spuriously.
            return AccessPlan((("grid", sm),), (), byte_count, direction, None, query)
        if resource in {"l2", "hbm"}:
            # Global-memory instructions still consume a per-SM LSU issue
            # path; L2/HBM byte service is modeled by the global queues below.
            local_keys: tuple[ResourceKey, ...] = (("lsu", sm),)
        else:
            local_keys = ((resource, sm),)
        if byte_count <= 0:
            return AccessPlan(local_keys, (), 0.0, direction, None, query)

        memory_keys: tuple[ResourceKey, ...]
        curve_resource: str | None = None
        requested_mode = str(benchmark_params.get("cache_mode", "l2_hot"))
        family = self._atom_family(atom_id)
        working_set_bytes = int(query.get("working_set_bytes", 0) or 0)
        fits_l2 = 0 < working_set_bytes <= self.resources.l2_bytes
        hit: bool | None = None
        if resource == "tma" and direction == "load":
            cache_key = self._cache_key(atom_id, benchmark_params)
            # A hot working set is only guaranteed resident when it fits the
            # measured L2 capacity.  Oversized and streaming sets use the LRU
            # page identity so reuse after an HBM fill can still hit.
            hit = forced_cache_hit
            if hit is None:
                hit = (
                    requested_mode == "l2_hot" and fits_l2
                ) or self._touch_l2_page(cache_key, int(byte_count))
            self.traffic_bytes["l2"] += byte_count
            if hit:
                self.cache_events["l2_hit"] += 1
                memory_keys = (("l2", "gpu"),)
                query["cache_mode"] = "l2_hot"
            else:
                self.cache_events["hbm_miss"] += 1
                self.traffic_bytes["hbm"] += byte_count
                memory_keys = (("hbm", "gpu"), ("l2", "gpu"))
                query["cache_mode"] = "hbm_stream"
                if forced_cache_hit is None:
                    self._insert_l2_page(cache_key, int(byte_count))
            curve_resource = "tma" if family == "tma_load" else None
        elif direction == "store" or resource == "hbm":
            if direction == "load":
                self.cache_events["hbm_miss"] += 1
            self.traffic_bytes["hbm"] += byte_count
            self.traffic_bytes["l2"] += byte_count
            # A global store first enters L2 and is then drained to HBM.
            memory_keys = (("l2", "gpu"), ("hbm", "gpu"))
            query["cache_mode"] = "hbm_stream"
        elif direction == "load":
            cache_key = self._cache_key(atom_id, benchmark_params)
            insert_bytes = max(int(byte_count), 128)
            hit = forced_cache_hit
            if hit is None:
                hit = (
                    requested_mode == "l2_hot" and fits_l2
                ) or self._touch_l2_page(cache_key, insert_bytes)
            self.traffic_bytes["l2"] += byte_count
            if hit:
                self.cache_events["l2_hit"] += 1
                memory_keys = (("l2", "gpu"),)
                query["cache_mode"] = "l2_hot"
            else:
                self.cache_events["hbm_miss"] += 1
                self.traffic_bytes["hbm"] += byte_count
                memory_keys = (("hbm", "gpu"), ("l2", "gpu"))
                query["cache_mode"] = "hbm_stream"
                if forced_cache_hit is None:
                    self._insert_l2_page(cache_key, insert_bytes)
        else:
            self.cache_events["l2_hit"] += 1
            self.traffic_bytes["l2"] += byte_count
            memory_keys = (("l2", "gpu"),)
            query["cache_mode"] = "l2_hot"
        cacheable_load = hit is not None
        producer_store = (
            direction == "store"
            and bool(benchmark_params.get("producer_consumer"))
            and self._cache_key(atom_id, benchmark_params) is not None
        )
        return AccessPlan(
            local_keys, memory_keys, byte_count, direction, curve_resource, query,
            self._cache_key(atom_id, benchmark_params)
            if cacheable_load or producer_store else None,
            ("producer_store" if producer_store else requested_mode)
            if cacheable_load or producer_store else None,
            fits_l2 if cacheable_load or producer_store else False,
            (
                max(working_set_bytes, int(byte_count))
                if producer_store
                else int(byte_count) if resource == "tma"
                else max(int(byte_count), 128)
            ) if cacheable_load or producer_store else 0,
            bool(hit) if cacheable_load else None,
        )

    def record_cache_access(
        self, node_id: str, request_cycle: float, fill_cycle: float,
        order_key: tuple[Any, ...], plan: AccessPlan,
    ) -> None:
        if plan.requested_cache_mode is None or plan.cache_key is None:
            return
        fill_only = plan.direction == "store"
        if not fill_only and plan.cache_hit is None:
            return
        self.cache_accesses.append(CacheAccess(
            node_id, request_cycle, fill_cycle, order_key, plan.cache_key,
            plan.requested_cache_mode, plan.fits_l2,
            plan.cache_insert_bytes, plan.cache_hit, fill_only,
        ))

    def actual_cache_routes(self) -> dict[str, bool]:
        return {
            item.node_id: item.actual_hit
            for item in self.cache_accesses
            if item.actual_hit is not None and not item.fill_only
        }

    def resolved_cache_routes(self) -> dict[str, bool]:
        """Replay L2 requests and fills on the modeled memory timeline.

        A miss becomes resident only after its HBM->L2 service completes.  In
        particular, a later request cannot hit merely because the miss node
        was visited earlier by the topological scheduler.
        """
        pages: OrderedDict[tuple[Any, ...], int] = OrderedDict()
        resident_bytes = 0
        result: dict[str, bool] = {}
        pending: list[tuple[float, tuple[Any, ...], tuple[Any, ...], int]] = []

        def insert(cache_key: tuple[Any, ...], byte_count: int) -> None:
            nonlocal resident_bytes
            previous = pages.pop(cache_key, 0)
            resident_bytes -= previous
            pages[cache_key] = byte_count
            resident_bytes += byte_count
            while resident_bytes > self.resources.l2_bytes and pages:
                _, evicted = pages.popitem(last=False)
                resident_bytes -= evicted

        for access in sorted(
            self.cache_accesses,
            key=lambda item: (item.request_cycle, item.order_key),
        ):
            while pending and pending[0][0] <= access.request_cycle:
                _, _, cache_key, byte_count = heapq.heappop(pending)
                insert(cache_key, byte_count)
            if access.fill_only:
                if access.cache_key is not None:
                    heapq.heappush(pending, (
                        access.fill_cycle, access.order_key,
                        access.cache_key, access.insert_bytes,
                    ))
                continue
            guaranteed_hot = (
                access.requested_cache_mode == "l2_hot" and access.fits_l2
            )
            hit = guaranteed_hot
            if not hit and access.cache_key is not None and access.cache_key in pages:
                resident = pages.pop(access.cache_key)
                pages[access.cache_key] = resident
                hit = True
            result[access.node_id] = hit
            if hit or access.cache_key is None:
                continue
            heapq.heappush(pending, (
                access.fill_cycle, access.order_key,
                access.cache_key, access.insert_bytes,
            ))
        return result

    def memory_service_cycles(
        self, plan: AccessPlan, key: ResourceKey, operation_cost: OperationCost,
    ) -> float:
        if plan.byte_count <= 0:
            return 0.0
        queue_resource = str(key[0])
        curve_resource = plan.curve_resource or queue_resource
        query = dict(plan.curve_query)
        # A cold TMA load traverses both global queues.  Select the measured
        # cold row for the HBM stage and the measured hot row for the L2 stage
        # instead of charging the HBM service curve twice.
        query["cache_mode"] = (
            "hbm_stream" if queue_resource == "hbm" else "l2_hot"
        )
        service = self.database.resource_service(
            curve_resource, query, direction=plan.direction
        )
        bandwidth = 0.0
        if service is not None and service.memory_bandwidth_value > 0:
            bandwidth = service.memory_bandwidth_value
            self.used_resource_provenance.extend(service.provenance)
        elif operation_cost.memory_bandwidth_value > 0:
            bandwidth = operation_cost.memory_bandwidth_value
            self.missing_resource_curves.add(
                f"{curve_resource}:{plan.direction or 'generic'}"
            )
        if bandwidth <= 0:
            self.missing_resource_curves.add(
                f"{curve_resource}:{plan.direction or 'generic'}"
            )
            return operation_cost.initiation_interval_cycles
        bytes_per_cycle = bandwidth * 1_000.0 / self.resources.sm_clock_mhz
        return plan.byte_count / max(bytes_per_cycle, 1e-12)

    def placement_json(self) -> list[dict[str, Any]]:
        return [dict(item.__dict__) for item in sorted(
            self.placements.values(), key=lambda value: value.cta_id
        )]
