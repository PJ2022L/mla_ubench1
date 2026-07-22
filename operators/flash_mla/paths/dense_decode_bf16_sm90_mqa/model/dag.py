"""Source-derived global dependency graph for FlashMLA dense decode.

Phases are labels only. Dependencies come from the kernel protocol; phase
boundaries never create scheduling edges.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import heapq
import json
from pathlib import Path
from typing import Any, Iterable, Mapping
import math

from .scheduler import RequestSlice, SchedulerResult, schedule_requests
from .schema import KernelResources, Workload


EVENTS = {"issue", "complete"}
DEPENDENCY_KINDS = {
    "program", "data", "barrier", "tma_ready", "wgmma_wait",
    "memory_visibility", "buffer_reuse", "grid_dependency",
}


@dataclass(frozen=True)
class OperationNode:
    id: str
    phase: str
    actor: str
    atom_id: str
    benchmark_params: Mapping[str, Any]
    work_amount: float
    work_unit: str
    resource_class: str
    async_issue: bool
    source_anchor: str
    cta_id: str | None = None
    sm_hint: int | None = None


@dataclass(frozen=True)
class Dependency:
    src: str
    dst: str
    src_event: str
    dst_event: str
    kind: str
    label: str | None = None


@dataclass
class DenseDecodeDAG:
    nodes: dict[str, OperationNode] = field(default_factory=dict)
    dependencies: list[Dependency] = field(default_factory=list)
    scheduler: SchedulerResult | None = None
    workload: Workload | None = None

    def add_node(self, node: OperationNode) -> str:
        if node.id in self.nodes:
            raise ValueError(f"duplicate DAG node: {node.id}")
        if node.work_amount < 0:
            raise ValueError(f"negative work amount for {node.id}")
        self.nodes[node.id] = node
        return node.id

    def add_dependency(
        self,
        src: str,
        dst: str,
        *,
        src_event: str = "complete",
        dst_event: str = "issue",
        kind: str = "data",
        label: str | None = None,
    ) -> None:
        if src not in self.nodes or dst not in self.nodes:
            raise ValueError(f"dependency references unknown node: {src} -> {dst}")
        if src_event not in EVENTS or dst_event not in EVENTS:
            raise ValueError("dependency event must be issue or complete")
        if kind not in DEPENDENCY_KINDS:
            raise ValueError(f"unknown dependency kind: {kind}")
        self.dependencies.append(Dependency(src, dst, src_event, dst_event, kind, label))

    def topological_nodes(self) -> list[str]:
        incoming = {node_id: 0 for node_id in self.nodes}
        outgoing: dict[str, set[str]] = {node_id: set() for node_id in self.nodes}
        for edge in self.dependencies:
            if edge.src not in self.nodes or edge.dst not in self.nodes:
                raise ValueError(
                    f"dependency references unknown node: {edge.src} -> {edge.dst}"
                )
            if edge.src_event not in EVENTS or edge.dst_event not in EVENTS:
                raise ValueError("dependency event must be issue or complete")
            if edge.kind not in DEPENDENCY_KINDS:
                raise ValueError(f"unknown dependency kind: {edge.kind}")
            # Node-level acyclicity is stricter than event-level acyclicity and
            # catches accidental protocol cycles early.
            if edge.src != edge.dst and edge.dst not in outgoing[edge.src]:
                incoming[edge.dst] += 1
                outgoing[edge.src].add(edge.dst)
        ready = [key for key, degree in incoming.items() if degree == 0]
        heapq.heapify(ready)
        ordered: list[str] = []
        while ready:
            current = heapq.heappop(ready)
            ordered.append(current)
            for target in outgoing[current]:
                incoming[target] -= 1
                if incoming[target] == 0:
                    heapq.heappush(ready, target)
        if len(ordered) != len(self.nodes):
            cyclic = sorted(key for key, degree in incoming.items() if degree)
            raise ValueError(f"dense-decode DAG contains a cycle: {cyclic[:8]}")
        return ordered

    def validate(self) -> None:
        self.topological_nodes()
        if any(not node.atom_id for node in self.nodes.values()):
            raise ValueError("every operation node requires an atom_id")

    def to_json(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": 1,
            "boundary": "gpu_metadata_main_combine",
            "workload": self.workload.to_json() if self.workload else None,
            "scheduler": self.scheduler.to_json() if self.scheduler else None,
            "nodes": [asdict(self.nodes[key]) for key in self.topological_nodes()],
            "dependencies": [asdict(edge) for edge in self.dependencies],
        }


class AtomMap:
    def __init__(self, mapping: Mapping[str, Any]) -> None:
        roles = mapping.get("roles", mapping)
        if not isinstance(roles, dict):
            raise ValueError("atom_map must contain a roles object")
        self._roles = roles

    @classmethod
    def load(cls, path: Path) -> "AtomMap":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def resolve(self, role: str, dtype: str) -> str:
        if role not in self._roles:
            raise KeyError(f"dense source role has no generic atom mapping: {role}")
        value = self._roles[role]
        if isinstance(value, str):
            return value.format(dtype=dtype)
        if isinstance(value, dict):
            selected = value.get(dtype, value.get("default"))
            if isinstance(selected, str):
                return selected.format(dtype=dtype)
        raise KeyError(f"atom mapping for {role} does not support dtype={dtype}")


class _Builder:
    def __init__(self, workload: Workload, resources: KernelResources, atom_map: AtomMap) -> None:
        self.workload = workload
        self.resources = resources
        self.atom_map = atom_map
        self.dag = DenseDecodeDAG(workload=workload)
        self.last_actor: dict[str, str] = {}
        # K cache identity includes the KV head: the same physical page number
        # addresses a different byte range for each head.
        self.last_physical_page: dict[tuple[int, int], int] = {}
        self.page_access_index = 0
        self.serial = 0

    @property
    def physical_page_count(self) -> int:
        return max(1, len({
            page for row in self.workload.block_table for page in row
        }))

    @property
    def block_table_entries(self) -> int:
        return max(1, self.workload.batch_size * self.block_table_stride)

    @property
    def block_table_stride(self) -> int:
        # Workload rows contain the accessed prefix.  The dense API uses one
        # rectangular tensor, so the best source-derived stride available in
        # this schema is the maximum row length, not the sum of all rows.
        return max(1, max((len(row) for row in self.workload.block_table), default=0))

    @property
    def output_tile_count(self) -> int:
        return (self.workload.batch_size * self.workload.num_heads_kv
                * self.workload.num_m_blocks)

    def request_split_count(self, request: int) -> int:
        scheduler = self.dag.scheduler
        if scheduler is None:
            return 0
        prefix = scheduler.num_splits_prefix
        count = prefix[request + 1] - prefix[request]
        return count if count > 1 else 0

    def request_partial_output_working_set_bytes(self, request: int) -> int:
        return max(
            1,
            self.request_split_count(request) * self.workload.num_heads_kv
            * self.workload.q_seq_per_hk * 512 * 4,
        )

    def request_partial_lse_working_set_bytes(self, request: int) -> int:
        return max(
            1,
            self.request_split_count(request) * self.workload.num_heads_kv
            * self.workload.q_seq_per_hk * 4,
        )

    def node(
        self,
        role: str,
        phase: str,
        actor: str,
        *,
        amount: float = 1,
        unit: str = "instruction",
        resource: str,
        async_issue: bool = False,
        anchor: str,
        params: Mapping[str, Any] | None = None,
        cta: str | None = None,
        program_order: bool = True,
    ) -> str:
        self.serial += 1
        node_id = f"n{self.serial:07d}.{role}.{actor}"
        merged = {"dtype": self.workload.dtype} | dict(params or {})
        if unit == "instruction" and resource in {"fp32", "sfu", "int32", "shuffle"}:
            # Scalar-family throughput is measured in dynamic lane operations,
            # not source instructions or logical rows.
            unit = "lane_op"
        self.dag.add_node(OperationNode(
            node_id, phase, actor, self.atom_map.resolve(role, self.workload.dtype),
            merged, float(amount), unit, resource, async_issue, anchor, cta,
        ))
        previous = self.last_actor.get(actor)
        if program_order and previous:
            previous_node = self.dag.nodes[previous]
            # Async instructions are ordered at issue. Their completion is
            # constrained only by explicit wait/readiness/reuse edges.
            self.dag.add_dependency(
                previous, node_id, kind="program",
                src_event="issue" if previous_node.async_issue else "complete",
                dst_event="issue",
            )
        if program_order:
            self.last_actor[actor] = node_id
        return node_id

    def edge(self, src: str, dst: str, kind: str, label: str | None = None,
             src_event: str = "complete", dst_event: str = "issue") -> None:
        self.dag.add_dependency(src, dst, kind=kind, label=label,
                                src_event=src_event, dst_event=dst_event)


def _metadata(builder: _Builder, scheduler: SchedulerResult) -> str | None:
    if builder.workload.metadata_mode == "reuse":
        return None
    actor = "metadata.cta"
    batch = builder.workload.batch_size
    parts = scheduler.num_sm_parts
    count = scheduler.operation_counts
    builder.node("metadata_load", "metadata", actor, amount=count["global_load"], resource="l2",
        anchor="get_decoding_sched_meta.cu:__ldg(seqlens_k)",
        params={"bytes": batch * 4, "cache_mode": builder.workload.cache_mode,
                "threads": 32, "issuers": min(batch, 32), "pattern": "sequential",
                "memory_object": "seqlens_k", "cache_line": 0,
                "working_set_entries": max(batch, 1)})
    builder.node("metadata_div", "metadata", actor, amount=count["div"], resource="int32",
        anchor="get_decoding_sched_meta.cu:block index and payload ceil_div")
    # The dense metadata source contains integer division but no independent
    # remainder expression.  Do not charge a REM atom for the quotient path.
    builder.node("metadata_iadd", "metadata", actor, amount=count["iadd"],
        resource="int32", anchor="get_decoding_sched_meta.cu:block/split arithmetic")
    builder.node("metadata_imad", "metadata", actor, amount=count["imad"],
        resource="int32", anchor="get_decoding_sched_meta.cu:shared/global addressing")
    builder.node("metadata_compare", "metadata", actor, amount=count["compare"],
        resource="int32", anchor="get_decoding_sched_meta.cu:loop and split predicates")
    for delta in (16, 8, 4, 2, 1):
        builder.node("metadata_shuffle", "metadata", actor, resource="shuffle",
            anchor="get_decoding_sched_meta.cu:warp total_num_blocks reduction",
            params={"delta": delta})
    builder.node("metadata_shared_store", "metadata", actor, amount=count["shared_store"],
        resource="shared", anchor="get_decoding_sched_meta.cu:shared work arrays",
        params={"threads": 32, "producers": min(batch, 32),
                "topology": "contiguous", "working_set_words": batch * 5 + 1})
    builder.node("metadata_shared_load", "metadata", actor, amount=count["shared_load"],
        resource="shared", anchor="get_decoding_sched_meta.cu:shared scheduler reads",
        params={"threads": 32, "pattern": "unique",
                "working_set_words": batch * 5 + 1})
    builder.node("metadata_warp_sync", "metadata", actor, amount=count["warp_sync"], resource="barrier",
        anchor="get_decoding_sched_meta.cu:two __syncwarp")
    builder.node("metadata_store", "metadata", actor, amount=count["metadata_store"], resource="hbm",
        anchor="get_decoding_sched_meta.cu:DecodingSchedMeta stores",
        params={"bytes": parts * 32, "cache_mode": "hbm_stream",
                "working_set_records": max(parts, 1), "pattern": "sequential"})
    return builder.node("metadata_split_store", "metadata", actor, amount=count["split_store"],
        resource="hbm", anchor="get_decoding_sched_meta.cu:num_splits stores",
        params={"bytes": (batch + 1) * 4, "cache_mode": "hbm_stream",
                "producers": min(batch + 1, 32),
                "working_set_records": batch + 1, "pattern": "sequential"})


def _score_page(
    builder: _Builder,
    *,
    cta: str,
    request: int,
    kv_head: int,
    local_page: int,
    logical_page: int,
    actor: str,
    phase: str,
    q_ready: str,
    q8_ready: Mapping[int, str],
    request_ready: str,
    reuse_after: Mapping[int, Iterable[str]],
    score_after: Mapping[int, Iterable[Any]],
) -> tuple[str, str, list[str]]:
    k_nodes: list[str] = [""] * 9
    table = builder.workload.block_table[request]
    physical_page = table[logical_page] if logical_page < len(table) else 0
    cache_identity = (physical_page, kv_head)
    previous_access = builder.last_physical_page.get(cache_identity)
    reuse_distance = (
        builder.page_access_index - previous_access if previous_access is not None else None
    )
    builder.last_physical_page[cache_identity] = builder.page_access_index
    builder.page_access_index += 1
    unique_pages = len({page for row in builder.workload.block_table for page in row})
    block_index = builder.node("block_table_load", phase, f"{cta}.control",
        resource="l2", anchor="splitkv_mla.cuh:__ldg(block_table_ptr)", cta=cta,
        params={"logical_page": logical_page, "physical_page": physical_page,
                "request": request, "memory_object": "block_table",
                "cache_line": (request * builder.block_table_stride + logical_page) // 32,
                "block_table_stride_entries": builder.block_table_stride,
                "bytes": 4, "cache_mode": builder.workload.cache_mode,
                "threads": 256, "issuers": 256, "pattern": "broadcast",
                "working_set_entries": builder.block_table_entries})
    builder.edge(request_ready, block_index, "data")
    launch_order = ([4, 5, 6, 7, 8, 0, 1, 2, 3]
                    if local_page & 1 else list(range(9)))
    for tile in launch_order:
        if local_page < 2:
            tma_actor = f"{cta}.tma.k{local_page & 1}.initial"
        else:
            tma_actor = f"{cta}.tma.k{local_page & 1}.{'lo' if tile < 4 else 'hi'}"
        load = builder.node("k_tma", phase, tma_actor, resource="tma", async_issue=True,
            anchor="splitkv_mla.cuh:launch_kv_tiles_copy_tma", cta=cta,
            params={"tile": tile, "bytes": 64 * 64 * 2,
                    "memory_object": "k",
                    "depth": 8, "working_set_pages": builder.physical_page_count,
                    "cache_mode": builder.workload.cache_mode,
                    "pattern": builder.workload.block_table_pattern,
                    "physical_page": physical_page,
                    "kv_head": kv_head,
                    "logical_page": logical_page,
                    "reuse_distance_pages": reuse_distance,
                    "working_set_bytes": (
                        unique_pages * builder.workload.num_heads_kv * 64 * 576 * 2
                    )})
        for prior in reuse_after.get(tile, ()):
            if isinstance(prior, tuple):
                source, event, reuse_label = prior
                builder.edge(
                    source, load, "buffer_reuse", reuse_label,
                    src_event=event,
                )
            else:
                builder.edge(
                    prior, load, "buffer_reuse", f"K{local_page % 2} overwrite"
                )
        builder.edge(block_index, load, "data", "physical page index")
        k_nodes[tile] = load
    order = (
        list(range(9)) if local_page == 0
        else ([4, 5, 6, 7, 8, 0, 1, 2, 3] if local_page & 1 else list(range(9)))
    )
    if local_page == 0:
        group = builder.node(
            "qk_ss", phase, actor, resource="tensor", async_issue=True,
            amount=1, unit="committed_group",
            anchor="splitkv_mla.cuh:warpgroup_cooperative_qkt_gemm_no_pipeline",
            cta=cta,
            params={"m": 64, "n": 64, "k": 16, "group_size": 36,
                    "depth": 1, "warpgroups": 1, "source_mode": "ss"},
        )
        builder.edge(q_ready, group, "tma_ready", "Q shared tile ready")
        for ready in k_nodes:
            builder.edge(ready, group, "tma_ready", "all first-page K tiles ready")
        return group, group, k_nodes

    groups: list[str] = []
    for tile in order:
        k_ready = k_nodes[tile]
        role = "qk_ss" if tile != 8 else "qk_rs"
        group = builder.node(role, phase, actor, resource="tensor", async_issue=True,
            amount=1, unit="committed_group", anchor="splitkv_mla.cuh:warpgroup_cooperative_qkt_gemm",
            cta=cta, params={"m": 64, "n": 64, "k": 16, "group_size": 4,
                              "depth": 4, "warpgroups": 1,
                              "source_mode": "ss" if role.endswith("ss") else "rs",
                              "tile": tile})
        builder.edge(q_ready, group, "tma_ready", "Q shared tile ready")
        if role == "qk_rs":
            builder.edge(q8_ready[local_page & 1], group, "data",
                         "Q tile8 register fragment ready")
        builder.edge(k_ready, group, "tma_ready", f"K tile {tile} ready")
        for prior in score_after.get(tile, ()):
            if isinstance(prior, tuple):
                source, event, wait_label = prior
                builder.edge(source, group, "wgmma_wait", wait_label,
                             src_event=event)
            else:
                builder.edge(prior, group, "wgmma_wait",
                             f"steady score segment tile {tile}")
        groups.append(group)
    # The selected wait boundary is already part of the WGMMA benchmark
    # protocol. Completion dependencies target the committed group directly;
    # there is no fake standalone wait latency.
    # For an even steady page, groups 0..3 are the phase-0 QK issue in
    # wg0_subroutine. The page+3 K1 low-half TMA is launched only after that
    # issue point and the intervening wait_group<4> release.
    phase0_issue = groups[3] if not (local_page & 1) else groups[0]
    return phase0_issue, groups[-1], k_nodes


def _softmax_and_local_update(
    builder: _Builder, *, cta: str, page: int, actor: str, phase: str,
    score_ready: str, tail_template_mask: bool, causal: bool, invalid_tokens: int,
) -> dict[str, str]:
    params = {
        "elements": 64,
        "tail_template_mask": tail_template_mask,
        "causal": causal,
        "invalid_tokens": invalid_tokens,
    }
    score_input = score_ready
    if tail_template_mask:
        mask_compare = builder.node(
            "softmax_token_compare", phase, actor, amount=64 * 64, resource="int32",
            anchor="splitkv_mla.cuh:tail token predicate", params=params, cta=cta,
        )
        builder.edge(score_ready, mask_compare, "data")
        mask_select = builder.node(
            "softmax_select", phase, actor, amount=64 * 64, resource="fp32",
            anchor="splitkv_mla.cuh:masked score select", params=params, cta=cta,
        )
        builder.edge(mask_compare, mask_select, "data")
        score_input = mask_select

    # Each WG owns two logical rows per lane.  Keep the two shuffle reductions
    # and their dependent FMNMX operations explicit instead of hiding them in
    # one opaque row-max cost.
    maximum = builder.node(
        "softmax_max", phase, actor, amount=64 * 64, resource="fp32",
        anchor="splitkv_mla.cuh:score pair and row max", params=params, cta=cta,
    )
    builder.edge(score_input, maximum, "data")
    for delta in (1, 2):
        builder.node("shuffle", phase, actor, amount=2 * 128, resource="shuffle",
            anchor="splitkv_mla.cuh:two rows xor delta 1/2",
            params={"delta": delta}, cta=cta)
        builder.node(
            "softmax_max", phase, actor, amount=2 * 128, resource="fp32",
            anchor="splitkv_mla.cuh:post-shuffle row max",
            params=params | {"delta": delta}, cta=cta,
        )
    builder.node("softmax_mul", phase, actor, amount=2 * 128, resource="fp32",
        anchor="splitkv_mla.cuh:softmax scaling", params=params, cta=cta)
    builder.node(
        "shared_load_u32", phase, actor, amount=2 * 128, resource="shared",
        anchor="splitkv_mla.cuh:sM row-state load", cta=cta,
        params={"threads": 128, "pattern": "quad_broadcast",
                "working_set_words": 64, "memory_object": "sM"},
    )
    state_max = builder.node(
        "softmax_max", phase, actor, amount=2 * 128, resource="fp32",
        anchor="splitkv_mla.cuh:new online max", params=params, cta=cta,
    )
    builder.node(
        "softmax_exp2", phase, actor, amount=2 * 128, resource="sfu",
        anchor="splitkv_mla.cuh:old-state rescale exp2f", params=params, cta=cta,
    )
    builder.node(
        "warp_sync", phase, actor, resource="barrier",
        anchor="splitkv_mla.cuh:sM read-before-write syncwarp", cta=cta,
    )
    scale_name = "sScale1" if actor.endswith(".wg1") else "sScale0"
    for state_name in ("sM", scale_name):
        builder.node(
            "shared_store_u32", phase, actor, amount=2 * 32, resource="shared",
            anchor=f"splitkv_mla.cuh:{state_name} online-state store", cta=cta,
            params={"threads": 128, "producers": 32,
                    "topology": "contiguous", "working_set_words": 64,
                    "memory_object": state_name},
        )

    def probability_update() -> str:
        builder.node(
            "softmax_ffma", phase, actor, amount=64 * 64, resource="fp32",
            anchor="splitkv_mla.cuh:rP scale-minus-max FFMA", params=params, cta=cta,
        )
        builder.node(
            "softmax_exp2", phase, actor, amount=64 * 64, resource="sfu",
            anchor="splitkv_mla.cuh:probability exp2f", params=params, cta=cta,
        )
        converted = builder.node(
            "probability_convert", phase, actor, amount=64 * 64, resource="fp32",
            anchor="splitkv_mla.cuh:rP conversion", params=params, cta=cta,
        )
        builder.node(
            "softmax_add", phase, actor, amount=64 * 64, resource="fp32",
            anchor="splitkv_mla.cuh:probability row sum", params=params, cta=cta,
        )
        return converted

    def output_update(*, combine_scale: bool) -> None:
        if combine_scale:
            builder.node(
                "shared_load_u32", phase, actor, amount=2 * 128,
                resource="shared", anchor="splitkv_mla.cuh:sScale0 row-state load",
                cta=cta, params={"threads": 128, "pattern": "quad_broadcast",
                                 "working_set_words": 64,
                                 "memory_object": "sScale0"},
            )
            builder.node(
                "softmax_mul", phase, actor, amount=2 * 128, resource="fp32",
                anchor="splitkv_mla.cuh:combined online O scale",
                params=params, cta=cta,
            )
        builder.node(
            "softmax_mul", phase, actor, amount=64 * 256, resource="fp32",
            anchor="splitkv_mla.cuh:online rO rescale", params=params, cta=cta,
        )

    if actor.endswith(".wg1"):
        convert = probability_update()
        output_update(combine_scale=True)
    else:
        output_update(combine_scale=False)
        convert = probability_update()
    complete = builder.node(
        "softmax_ffma", phase, actor, amount=2 * 128, resource="fp32",
        anchor="splitkv_mla.cuh:rL online update FFMA", params=params, cta=cta,
    )
    return {
        "entry": mask_compare if tail_template_mask else maximum,
        "max": state_max,
        "convert": convert,
        "complete": complete,
    }


def _empty_odd_update(
    builder: _Builder, *, cta: str, actor: str, phase: str, ready: str,
) -> str:
    """Model wg1_bunch_0<IS_BLK0_LAST> without fictitious P exp/convert."""
    params = {"empty_odd_page": True, "elements": 64}
    maximum = builder.node(
        "softmax_max", phase, actor, amount=64 * 64, resource="fp32",
        anchor="splitkv_mla.cuh:empty odd score pair and row max",
        params=params, cta=cta,
    )
    builder.edge(ready, maximum, "barrier", "sScale0Ready")
    for delta in (1, 2):
        builder.node(
            "shuffle", phase, actor, amount=2 * 128, resource="shuffle",
            anchor="splitkv_mla.cuh:empty odd row shuffle",
            params={"delta": delta}, cta=cta,
        )
        builder.node(
            "softmax_max", phase, actor, amount=2 * 128, resource="fp32",
            anchor="splitkv_mla.cuh:empty odd post-shuffle max",
            params=params | {"delta": delta}, cta=cta,
        )
    builder.node(
        "softmax_mul", phase, actor, amount=2 * 128, resource="fp32",
        anchor="splitkv_mla.cuh:empty odd softmax scaling", params=params, cta=cta,
    )
    builder.node(
        "shared_load_u32", phase, actor, amount=2 * 128, resource="shared",
        anchor="splitkv_mla.cuh:empty odd sM load", cta=cta,
        params={"threads": 128, "pattern": "quad_broadcast",
                "working_set_words": 64, "memory_object": "sM"},
    )
    builder.node(
        "softmax_max", phase, actor, amount=2 * 128, resource="fp32",
        anchor="splitkv_mla.cuh:empty odd new online max", params=params, cta=cta,
    )
    builder.node(
        "softmax_exp2", phase, actor, amount=2 * 128, resource="sfu",
        anchor="splitkv_mla.cuh:empty odd old-state rescale exp2f",
        params=params, cta=cta,
    )
    builder.node(
        "warp_sync", phase, actor, resource="barrier",
        anchor="splitkv_mla.cuh:empty odd sM syncwarp", cta=cta,
    )
    for state_name in ("sM", "sScale1"):
        builder.node(
            "shared_store_u32", phase, actor, amount=2 * 32, resource="shared",
            anchor=f"splitkv_mla.cuh:empty odd {state_name} store", cta=cta,
            params={"threads": 128, "producers": 32,
                    "topology": "contiguous", "working_set_words": 64,
                    "memory_object": state_name},
        )
    builder.node(
        "shared_load_u32", phase, actor, amount=2 * 128, resource="shared",
        anchor="splitkv_mla.cuh:empty odd sScale0 load", cta=cta,
        params={"threads": 128, "pattern": "quad_broadcast",
                "working_set_words": 64, "memory_object": "sScale0"},
    )
    builder.node(
        "softmax_mul", phase, actor, amount=2 * 128, resource="fp32",
        anchor="splitkv_mla.cuh:empty odd combined O scale", params=params, cta=cta,
    )
    builder.node(
        "softmax_mul", phase, actor, amount=64 * 256, resource="fp32",
        anchor="splitkv_mla.cuh:empty odd rO rescale", params=params, cta=cta,
    )
    return builder.node(
        "softmax_ffma", phase, actor, amount=2 * 128, resource="fp32",
        anchor="splitkv_mla.cuh:empty odd rL update FFMA", params=params, cta=cta,
    )


def _local_pv(builder: _Builder, cta: str, phase: str, actor: str, ready: str) -> str:
    local = builder.node("pv_rs", phase, actor, amount=1, unit="committed_group",
        resource="tensor", async_issue=True,
        anchor="splitkv_mla.cuh:warpgroup_cooperative_pv_gemm_localP",
        params={"m": 64, "n": 256, "k": 16, "group_size": 4,
                "depth": 4, "warpgroups": 1, "source_mode": "rs"}, cta=cta)
    builder.edge(ready, local, "data")
    return local


def _stsm(builder: _Builder, cta: str, phase: str, actor: str, ready: str) -> tuple[str, str]:
    store = builder.node("p_stmatrix", phase, actor, resource="shared",
        anchor="splitkv_mla.cuh:save_rPb_to_sP",
        params={"m": 64, "n": 64, "warpgroups": 1}, cta=cta)
    builder.edge(ready, store, "data")
    fence = builder.node("proxy_fence", phase, actor, resource="barrier",
        anchor="splitkv_mla.cuh:fence.proxy.async.shared::cta", cta=cta)
    builder.edge(store, fence, "memory_visibility")
    return store, fence


def _tail_v_ready(
    builder: _Builder, *, cta: str, phase: str, actor: str, ready: str,
    invalid_tokens: int, half: str, protocol_active: bool,
) -> str:
    if not protocol_active:
        return ready
    fence_ready = ready
    if invalid_tokens > 0:
        store = builder.node(
            "tail_v_store", phase, actor, resource="shared",
            anchor="splitkv_mla.cuh:fill_oob_V", cta=cta,
            params={"invalid_tokens": invalid_tokens, "warpgroups": 1,
                    "stores_per_thread": 1, "half": half},
        )
        builder.edge(ready, store, "data")
        fence_ready = store
    fence = builder.node(
        "proxy_fence", phase, actor, resource="barrier",
        anchor="splitkv_mla.cuh:fill_oob_V async proxy fence", cta=cta,
        params={"invalid_tokens": invalid_tokens, "half": half},
    )
    builder.edge(fence_ready, fence, "memory_visibility")
    return fence


def _barrier(builder: _Builder, cta: str, phase: str, actor: str, role: str,
             prior: str, label: str, *, src_event: str = "complete") -> str:
    node = builder.node(role, phase, actor, resource="barrier",
        anchor=f"splitkv_mla.cuh:NamedBarriers::{label}", cta=cta, program_order=False,
        params={"named_barrier": label})
    builder.edge(prior, node, "barrier", label, src_event=src_event)
    return node


def _barrier_2wg(
    builder: _Builder, cta: str, phase: str, actor: str, label: str,
    *, arrive_ready: str, wait_ready: str,
    arrive_src_event: str = "complete", wait_src_event: str = "complete",
) -> str:
    """One measured asymmetric two-warpgroup named-barrier protocol.

    ``bar_arrive_2wg`` already measures WG-arrive + peer-WG wait as one
    protocol. Its issue event is the nonblocking arrive-side continuation; its
    complete event is the wait-side release. Modeling another bar_sync node
    would charge the same wait twice and would incorrectly block the arriver.
    """
    node = builder.node(
        "barrier_2wg", phase, actor, resource="barrier", async_issue=True,
        anchor=f"splitkv_mla.cuh:NamedBarriers::{label}", cta=cta,
        program_order=False,
        params={"named_barrier": label, "protocol": "arrive_peer_wait"},
    )
    builder.edge(
        arrive_ready, node, "barrier", label,
        src_event=arrive_src_event, dst_event="issue",
    )
    builder.edge(
        wait_ready, node, "barrier", label,
        src_event=wait_src_event, dst_event="complete",
    )
    return node


def _update_phase(page: int, pages: int) -> str:
    """Assign both halves of a terminal pair to one non-overlapping phase."""
    pair_start = page - (page & 1)
    if pair_start + 1 >= pages:
        return "tail_update.single_drain"
    if pair_start + 2 == pages:
        return "tail_update.pair_drain"
    if pair_start + 2 == pages - 1:
        return "tail_update.pair_to_single"
    return f"pair_update[{pair_start // 2}]"


def _remote_pv(builder: _Builder, cta: str, phase: str, actor: str,
               *ready: str) -> str:
    remote = builder.node("pv_ss", phase, actor, amount=1, unit="committed_group",
        resource="tensor", async_issue=True,
        anchor="splitkv_mla.cuh:warpgroup_cooperative_pv_gemm_remoteP", cta=cta,
        params={"m": 64, "n": 256, "k": 16, "group_size": 4,
                "depth": 4, "warpgroups": 1, "source_mode": "ss"})
    for item in ready:
        builder.edge(item, remote, "barrier", "remote P/O ready")
    return remote


def _build_slice(
    builder: _Builder, item: RequestSlice, *, cta: str, q_ready: str,
    q8_ready: Mapping[int, str],
    kv_head: int,
    m_block: int,
    previous_done: str | None,
    has_next_request: bool,
    valid_rows: int,
    metadata_ready: str | None,
) -> tuple[str, str, str, str | None]:
    pages = item.pages
    setup_actor = f"{cta}.control"
    setup = builder.node("request_length_load", "request_setup", setup_actor,
        resource="l2", anchor="splitkv_mla.cuh:seqlens_k request load", cta=cta,
        params={"request": item.request_idx, "pages": pages, "bytes": 4,
                "memory_object": "seqlens_k", "cache_line": item.request_idx // 32,
                "cache_mode": builder.workload.cache_mode,
                "threads": 256, "issuers": 256, "pattern": "broadcast",
                "working_set_entries": builder.workload.batch_size})
    if metadata_ready:
        builder.edge(metadata_ready, setup, "data", "metadata ready")
    if previous_done:
        builder.edge(previous_done, setup, "barrier", "persistent request reuse")
        for ready in q8_ready.values():
            builder.edge(previous_done, ready, "barrier",
                         "Q tile8 consumed after prior request")
    sm_init = builder.node("shared_store_u32", "request_setup", f"{cta}.control",
        amount=64, resource="shared", anchor="splitkv_mla.cuh:sM MAX_INIT_VAL_SM init",
        cta=cta, program_order=False,
        params={"threads": 256, "producers": 64, "topology": "contiguous",
                "working_set_words": 64})
    builder.edge(setup, sm_init, "program")

    page_state: list[dict[str, str]] = []
    final_by_buffer: dict[int, list[str]] = {0: [], 1: []}
    reuse_by_buffer: dict[int, dict[int, list[Any]]] = {0: {}, 1: {}}
    score_by_buffer: dict[int, dict[int, list[Any]]] = {0: {}, 1: {}}
    for page in range(pages):
        wg = page & 1
        actor = f"{cta}.wg{wg}"
        phase = "first_score" if page == 0 else f"steady_score[{page - 1}]"
        logical_page = item.start_block_idx + page
        valid_k_tokens = max(0, min(
            64, builder.workload.seqlens_k[item.request_idx] - logical_page * 64
        ))
        invalid_tokens = 64 - valid_k_tokens
        phase0_issue, score, _ = _score_page(
            builder, cta=cta, request=item.request_idx, kv_head=kv_head,
            local_page=page,
            logical_page=logical_page,
            actor=actor, phase=phase, q_ready=q_ready, q8_ready=q8_ready,
            request_ready=setup,
            reuse_after=reuse_by_buffer[page & 1],
            score_after=score_by_buffer[page & 1],
        )
        if page >= 2 and not (page & 1):
            for tile in range(4):
                reuse_by_buffer[1].setdefault(tile, []).append((
                    phase0_issue, "issue",
                    "page+2 phase-0 score issue before page+3 K1 low TMA",
                ))
        update_phase = _update_phase(page, pages)
        initialized = None
        if page == 0:
            initialized = _barrier(
                builder, cta, phase, f"{cta}.sync", "barrier_sync_128",
                score, "sMInitialized", src_event="issue",
            )
            builder.edge(sm_init, initialized, "barrier", "sMInitialized")
        state = _softmax_and_local_update(
            builder, cta=cta, page=page, actor=actor, phase=update_phase,
            score_ready=score,
            tail_template_mask=page >= max(0, pages - 2),
            causal=builder.workload.causal,
            invalid_tokens=invalid_tokens,
        )
        if initialized:
            builder.edge(initialized, state["entry"], "barrier", "sMInitialized")
        page_state.append(state)

        if page & 1:
            even = page_state[page - 1]
            scale0 = _barrier_2wg(
                builder, cta, update_phase, f"{cta}.sync", "sScale0Ready",
                arrive_ready=even["complete"], wait_ready=score,
            )
            builder.edge(scale0, state["entry"], "barrier", "sScale0Ready")
            even_local = _local_pv(
                builder, cta, update_phase, f"{cta}.wg0", even["complete"]
            )
            builder.edge(
                scale0, even_local, "barrier", "sScale0Ready WG0 arrive",
                src_event="issue",
            )
            scale1 = _barrier_2wg(
                builder, cta, update_phase, f"{cta}.sync", "sScale1Ready",
                arrive_ready=state["complete"], wait_ready=even_local,
            )
            odd_stsm = builder.node(
                "p_stmatrix", update_phase, f"{cta}.wg1", resource="shared",
                anchor="splitkv_mla.cuh:save_rPb_to_sP WG1", cta=cta,
                params={"m": 64, "n": 64, "warpgroups": 1},
            )
            builder.edge(
                scale1, odd_stsm, "barrier", "sScale1Ready arrive continuation",
                src_event="issue",
            )
            if page == pages - 1:
                # IS_BLK1_LAST: fill and fence precede the local-P WGMMA;
                # there is no second post-issue fence in this template.
                odd_fence = _tail_v_ready(
                    builder, cta=cta, phase=update_phase, actor=f"{cta}.wg1",
                    ready=odd_stsm, invalid_tokens=invalid_tokens, half="right",
                    protocol_active=True,
                )
                odd_local = _local_pv(
                    builder, cta, update_phase, f"{cta}.wg1", odd_fence
                )
                odd_wait_ready, odd_wait_event = odd_local, "issue"
            else:
                # Normal steady template: STSM, issue local-P WGMMA, then make
                # sP1 visible to the async proxy before WG1 reaches sP0Ready.
                odd_local = _local_pv(
                    builder, cta, update_phase, f"{cta}.wg1", odd_stsm
                )
                odd_fence = builder.node(
                    "proxy_fence", update_phase, f"{cta}.wg1",
                    resource="barrier",
                    anchor="splitkv_mla.cuh:sP1 post-local-P proxy fence",
                    cta=cta, program_order=False,
                )
                builder.edge(
                    odd_local, odd_fence, "memory_visibility",
                    "sP1 visible after local-P issue", src_event="issue",
                )
                odd_wait_ready, odd_wait_event = odd_fence, "complete"
            scale1_p_load = builder.node(
                "shared_load_u32", update_phase, f"{cta}.wg0",
                amount=2 * 128, resource="shared",
                anchor="splitkv_mla.cuh:wg0_scale_rP0 sScale1 load",
                cta=cta, program_order=False,
                params={"threads": 128, "pattern": "quad_broadcast",
                        "working_set_words": 64, "memory_object": "sScale1"},
            )
            builder.edge(scale1, scale1_p_load, "barrier", "sScale1Ready")
            rescale = builder.node("softmax_mul", update_phase, f"{cta}.wg0",
                amount=64 * 64, resource="fp32", anchor="splitkv_mla.cuh:wg0_scale_rP0",
                cta=cta, program_order=False)
            builder.edge(scale1_p_load, rescale, "data")
            rescale_convert = builder.node(
                "probability_convert", update_phase, f"{cta}.wg0",
                amount=64 * 64, resource="fp32",
                anchor="splitkv_mla.cuh:wg0_scale_rP0 conversion",
                cta=cta, program_order=False,
            )
            builder.edge(rescale, rescale_convert, "data")
            even_stsm, even_fence = _stsm(
                builder, cta, update_phase, f"{cta}.wg0", rescale_convert
            )
            p0 = _barrier_2wg(
                builder, cta, update_phase, f"{cta}.sync", "sP0Ready",
                arrive_ready=even_fence, wait_ready=odd_wait_ready,
                wait_src_event=odd_wait_event,
            )
            remote1 = _remote_pv(
                builder, cta, update_phase, f"{cta}.wg1", p0
            )
            issued = _barrier_2wg(
                builder, cta, update_phase, f"{cta}.sync",
                "rO1sP0sV0RIssued",
                arrive_ready=remote1, wait_ready=p0,
                arrive_src_event="issue", wait_src_event="issue",
            )
            scale1_o_load = builder.node(
                "shared_load_u32", update_phase, f"{cta}.wg0",
                amount=2 * 128, resource="shared",
                anchor="splitkv_mla.cuh:wg0_rescale_rO0 sScale1 load",
                cta=cta, program_order=False,
                params={"threads": 128, "pattern": "quad_broadcast",
                        "working_set_words": 64, "memory_object": "sScale1"},
            )
            builder.edge(
                issued, scale1_o_load, "barrier", "rO1sP0sV0RIssued"
            )
            rescale_o = builder.node("softmax_mul", update_phase, f"{cta}.wg0",
                amount=64 * 256 + 2 * 128, resource="fp32", anchor="splitkv_mla.cuh:wg0_rescale_rO0",
                cta=cta, program_order=False)
            builder.edge(scale1_o_load, rescale_o, "data")
            remote0_ready = rescale_o
            if page == pages - 1:
                remote0_ready = _tail_v_ready(
                    builder, cta=cta, phase=update_phase, actor=f"{cta}.wg0",
                    ready=rescale_o, invalid_tokens=invalid_tokens, half="left",
                    protocol_active=True,
                )
            remote0 = _remote_pv(
                builder, cta, update_phase, f"{cta}.wg0",
                odd_fence, remote0_ready,
            )
            final_by_buffer[0] = [even_local, remote0, remote1]
            final_by_buffer[1] = [odd_local, remote0, remote1]
            # Source lines 788-795/923-932 reuse the two 64-column
            # halves independently instead of waiting for the full page.
            reuse_by_buffer[0] = {
                **{tile: [scale1] for tile in range(4)},
                **{tile: [remote1] for tile in range(4, 9)},
            }
            reuse_by_buffer[1] = {
                **{
                    tile: [(
                        remote0, "complete",
                        "wait_group<4> release before K1 low TMA",
                    )]
                    for tile in range(4)
                },
                **{tile: [odd_local] for tile in range(4, 9)},
            }
            # WG0 issues phase-0 (tiles 0..3), waits for the remote-odd
            # progress point, then issues phase-2 (tiles 4..8).
            wait_group_event = "complete" if page + 2 < pages else "issue"
            score_by_buffer[0] = {
                tile: [(remote0, wait_group_event, "wait_group<4> release")]
                for tile in range(4, 9)
            }
            score_by_buffer[1] = {tile: [remote1] for tile in range(9)}
        elif page == pages - 1:
            scale0 = _barrier_2wg(
                builder, cta, update_phase, f"{cta}.sync", "sScale0Ready",
                arrive_ready=state["complete"], wait_ready=q8_ready[1],
            )
            empty_complete = _empty_odd_update(
                builder, cta=cta, actor=f"{cta}.wg1",
                phase=update_phase, ready=scale0,
            )
            local_ready = _tail_v_ready(
                builder, cta=cta, phase=update_phase, actor=actor,
                ready=state["complete"], invalid_tokens=invalid_tokens, half="left",
                protocol_active=True,
            )
            local = _local_pv(builder, cta, update_phase, actor, local_ready)
            builder.edge(
                scale0, local, "barrier", "sScale0Ready WG0 arrive",
                src_event="issue",
            )
            scale1 = _barrier_2wg(
                builder, cta, update_phase, f"{cta}.sync", "sScale1Ready",
                arrive_ready=empty_complete, wait_ready=local,
            )
            scale1_p_load = builder.node(
                "shared_load_u32", update_phase, actor,
                amount=2 * 128, resource="shared",
                anchor="splitkv_mla.cuh:wg0_scale_rP0 single sScale1 load",
                cta=cta, program_order=False,
                params={"threads": 128, "pattern": "quad_broadcast",
                        "working_set_words": 64, "memory_object": "sScale1"},
            )
            builder.edge(scale1, scale1_p_load, "barrier", "sScale1Ready")
            rescale = builder.node("softmax_mul", update_phase, actor,
                amount=64 * 64, resource="fp32", anchor="splitkv_mla.cuh:wg0_scale_rP0 single",
                cta=cta, program_order=False)
            builder.edge(scale1_p_load, rescale, "data")
            rescale_convert = builder.node(
                "probability_convert", update_phase, actor,
                amount=64 * 64, resource="fp32",
                anchor="splitkv_mla.cuh:wg0_scale_rP0 single conversion",
                cta=cta, program_order=False,
            )
            builder.edge(rescale, rescale_convert, "data")
            _, fence = _stsm(builder, cta, update_phase, actor, rescale_convert)
            p_ready = _barrier_2wg(
                builder, cta, update_phase, f"{cta}.sync", "sP0Ready",
                arrive_ready=fence, wait_ready=scale1, wait_src_event="issue",
            )
            remote_ready = _tail_v_ready(
                builder, cta=cta, phase=update_phase, actor=f"{cta}.wg1",
                ready=p_ready, invalid_tokens=invalid_tokens, half="right",
                protocol_active=True,
            )
            remote = _remote_pv(
                builder, cta, update_phase, f"{cta}.wg1", fence, remote_ready
            )
            final_by_buffer[page & 1] = [local, remote]

    all_final = [node for values in final_by_buffer.values() for node in values]
    if pages == 0:
        # Upstream maps the empty request to a dummy K page, executes the
        # guarded first 36-SS QK, then skips both WG update loops.
        _, dummy_score, _ = _score_page(
            builder, cta=cta, request=item.request_idx, kv_head=kv_head, local_page=0,
            logical_page=item.start_block_idx,
            actor=f"{cta}.wg0", phase="first_score", q_ready=q_ready, q8_ready=q8_ready,
            request_ready=setup, reuse_after={}, score_after={},
        )
        initialized = _barrier(
            builder, cta, "first_score", f"{cta}.sync", "barrier_sync_128",
            dummy_score, "sMInitialized", src_event="issue",
        )
        builder.edge(sm_init, initialized, "barrier", "sMInitialized")
        all_final.extend((dummy_score, initialized))
    reduce_actor = f"{cta}.reduce"
    first_shfl0 = first_shfl1 = None
    add0 = add1 = None
    for delta in (1, 2):
        current0 = builder.node("shuffle", "l_reduction", f"{reduce_actor}.wg0",
            amount=2 * 128, resource="shuffle", anchor="splitkv_mla.cuh:rL xor 1/2 WG0",
            cta=cta, program_order=False, params={"delta": delta})
        current1 = builder.node("shuffle", "l_reduction", f"{reduce_actor}.wg1",
            amount=2 * 128, resource="shuffle", anchor="splitkv_mla.cuh:rL xor 1/2 WG1",
            cta=cta, program_order=False, params={"delta": delta})
        if add0:
            builder.edge(add0, current0, "data")
            builder.edge(add1, current1, "data")
        else:
            first_shfl0, first_shfl1 = current0, current1
        add0 = builder.node(
            "softmax_add", "l_reduction", f"{reduce_actor}.wg0",
            amount=2 * 128, resource="fp32",
            anchor="splitkv_mla.cuh:rL post-shuffle add WG0", cta=cta,
            program_order=False, params={"delta": delta},
        )
        add1 = builder.node(
            "softmax_add", "l_reduction", f"{reduce_actor}.wg1",
            amount=2 * 128, resource="fp32",
            anchor="splitkv_mla.cuh:rL post-shuffle add WG1", cta=cta,
            program_order=False, params={"delta": delta},
        )
        builder.edge(current0, add0, "data")
        builder.edge(current1, add1, "data")
    assert first_shfl0 and first_shfl1 and add0 and add1
    for final in all_final:
        builder.edge(final, first_shfl0, "wgmma_wait")
        builder.edge(final, first_shfl1, "wgmma_wait")
    store0 = builder.node("shared_store_u32", "l_reduction", f"{reduce_actor}.wg0",
        amount=64, resource="shared", anchor="splitkv_mla.cuh:sL_reduction_wksp WG0 store",
        cta=cta, program_order=False,
        params={"threads": 128, "producers": 32, "topology": "contiguous",
                "working_set_words": 128})
    store1 = builder.node("shared_store_u32", "l_reduction", f"{reduce_actor}.wg1",
        amount=64, resource="shared", anchor="splitkv_mla.cuh:sL_reduction_wksp WG1 store",
        cta=cta, program_order=False,
        params={"threads": 128, "producers": 32, "topology": "contiguous",
                "working_set_words": 128})
    builder.edge(add0, store0, "data")
    builder.edge(add1, store1, "data")
    builder.edge(setup, store0, "program")
    builder.edge(setup, store1, "program")
    reduction_sync = builder.node("barrier_sync", "l_reduction", reduce_actor,
        resource="barrier", anchor="splitkv_mla.cuh:__syncthreads L reduction", cta=cta,
        program_order=False)
    builder.edge(store0, reduction_sync, "barrier")
    builder.edge(store1, reduction_sync, "barrier")
    load0 = builder.node("shared_load_u32", "l_reduction", f"{reduce_actor}.wg0",
        amount=256, resource="shared", anchor="splitkv_mla.cuh:WG0 remote L loads",
        cta=cta, program_order=False,
        params={"threads": 128, "pattern": "quad_broadcast",
                "working_set_words": 128})
    rmw_load = builder.node("shared_load_u32", "l_reduction", f"{reduce_actor}.wg1",
        amount=2 * 32, resource="shared", anchor="splitkv_mla.cuh:WG1 L RMW load",
        cta=cta, program_order=False,
        params={"threads": 128, "pattern": "unique",
                "working_set_words": 128})
    builder.edge(reduction_sync, load0, "barrier")
    builder.edge(reduction_sync, rmw_load, "barrier")
    add_l = builder.node("softmax_add", "l_reduction", f"{reduce_actor}.wg0",
        amount=256, resource="fp32", anchor="splitkv_mla.cuh:cross-WG rL add",
        cta=cta, program_order=False)
    builder.edge(load0, add_l, "data")
    rmw_add = builder.node(
        "softmax_add", "l_reduction", f"{reduce_actor}.wg1",
        amount=2 * 32, resource="fp32",
        anchor="splitkv_mla.cuh:WG1 L RMW add", cta=cta,
        program_order=False,
    )
    builder.edge(rmw_load, rmw_add, "data")
    rmw_store = builder.node(
        "shared_store_u32", "l_reduction", f"{reduce_actor}.wg1",
        amount=2 * 32, resource="shared",
        anchor="splitkv_mla.cuh:WG1 L RMW store", cta=cta,
        program_order=False,
        params={"threads": 128, "producers": 32, "topology": "contiguous",
                "working_set_words": 128},
    )
    builder.edge(rmw_add, rmw_store, "data")
    warp = builder.node("warp_sync", "l_reduction", f"{reduce_actor}.wg1",
        resource="barrier", anchor="splitkv_mla.cuh:WG1 __syncwarp", cta=cta,
        program_order=False)
    builder.edge(rmw_store, warp, "data")
    load1 = builder.node("shared_load_u32", "l_reduction", f"{reduce_actor}.wg1",
        amount=256, resource="shared", anchor="splitkv_mla.cuh:WG1 L readback",
        cta=cta, program_order=False,
        params={"threads": 128, "pattern": "quad_broadcast",
                "working_set_words": 128})
    builder.edge(warp, load1, "data")
    prune_cmp = builder.node("softmax_compare", "l_reduction", reduce_actor,
        amount=1024, resource="fp32", anchor="splitkv_mla.cuh:rL zero/NaN prune compare",
        cta=cta, program_order=False)
    builder.edge(add_l, prune_cmp, "data")
    builder.edge(load1, prune_cmp, "data")
    reduction = builder.node("softmax_select", "l_reduction", reduce_actor,
        amount=512, resource="fp32", anchor="splitkv_mla.cuh:rL prune select", cta=cta,
        program_order=False)
    builder.edge(prune_cmp, reduction, "data")
    split = not item.is_no_split
    split_prefix = builder.dag.scheduler.num_splits_prefix
    global_split_idx = (
        split_prefix[item.request_idx] + item.split_idx_within_request
    )
    epilogue_phase = "epilogue_split" if split else "epilogue_nosplit"
    normalize = builder.node("normalize", epilogue_phase, f"{cta}.epilogue",
        amount=2 * 256, resource="sfu", anchor="splitkv_mla.cuh:store_o reciprocal normalization", cta=cta)
    builder.edge(reduction, normalize, "data")
    output_mul = builder.node("softmax_mul", epilogue_phase, f"{cta}.epilogue",
        amount=64 * 512, resource="fp32", anchor="splitkv_mla.cuh:rO / rL",
        cta=cta)
    if split:
        staging = builder.node("split_shared_store", epilogue_phase, f"{cta}.store",
            amount=64 * 256, resource="shared",
            anchor="splitkv_mla.cuh:stride-520 split staging", cta=cta,
            params={"warpgroups": 2, "stores_per_thread": 64,
                    "invalid_tokens": 1})
        builder.edge(output_mul, staging, "data")
        fence = builder.node("proxy_fence", epilogue_phase, f"{cta}.store",
            resource="barrier", anchor="splitkv_mla.cuh:split staging proxy fence", cta=cta)
        sync = builder.node("barrier_sync", epilogue_phase, f"{cta}.store",
            resource="barrier", anchor="splitkv_mla.cuh:split staging CTA sync", cta=cta)
        store = builder.node("partial_output_store", epilogue_phase, f"{cta}.store",
            resource="hbm", async_issue=True, anchor="splitkv_mla.cuh:bulk S2G partial O",
            cta=cta, params={"bytes": valid_rows * 512 * 4,
                             "cache_mode": "hbm_stream", "pattern": "sequential",
                             "working_set_tiles": (
                                 builder.request_split_count(item.request_idx)
                                 * builder.workload.num_heads_kv
                                 * builder.workload.num_m_blocks
                             ),
                             "memory_object": "partial_output",
                             "producer_consumer": True,
                             "request": item.request_idx,
                             "split": item.split_idx_within_request,
                             "global_split_idx": global_split_idx,
                             "kv_head": kv_head,
                             "m_block": m_block,
                             "q_row_start": m_block * 64,
                             "q_row_count": valid_rows,
                             "chunk_start": 0,
                             "chunk_count": 4,
                             "head_dim": 512,
                             "layout": "split_kvhead_qrow_head_dim",
                             "working_set_bytes": (
                                 builder.request_partial_output_working_set_bytes(
                                     item.request_idx
                                 )
                             )})
    else:
        output_convert = builder.node("output_convert", epilogue_phase, f"{cta}.epilogue",
            amount=64 * 512, resource="fp32",
            anchor="splitkv_mla.cuh:output conversion", cta=cta,
            program_order=False)
        builder.edge(output_mul, output_convert, "data")
        stmatrix = builder.node("o_stmatrix", epilogue_phase, f"{cta}.store",
            amount=1, resource="shared", anchor="splitkv_mla.cuh:O STSM 64x512",
            cta=cta, params={"warpgroups": 2})
        builder.edge(output_convert, stmatrix, "data")
        fence = builder.node("proxy_fence", epilogue_phase, f"{cta}.store",
            resource="barrier", anchor="splitkv_mla.cuh:O STSM proxy fence", cta=cta)
        sync = builder.node("barrier_sync", epilogue_phase, f"{cta}.store",
            resource="barrier", anchor="splitkv_mla.cuh:O store CTA sync", cta=cta)
        store = builder.node("output_store", epilogue_phase, f"{cta}.store",
            resource="tma", async_issue=True, anchor="splitkv_mla.cuh:TMA O store",
            cta=cta, params={"bytes": valid_rows * 512 * 2,
                             "cache_mode": "hbm_stream", "depth": 1,
                             "working_set_tiles": builder.output_tile_count})
    builder.edge(fence, sync, "memory_visibility")
    builder.edge(sync, store, "barrier")

    # store_o issues the asynchronous output store before the source computes
    # and writes LSE. Keep that issue/completion overlap visible in the DAG.
    lse = builder.node("lse", epilogue_phase, f"{cta}.epilogue", amount=valid_rows,
        resource="sfu", anchor="splitkv_mla.cuh:logf/log2f LSE", cta=cta,
        program_order=False)
    builder.edge(store, lse, "program", src_event="issue")
    lse_add = builder.node("softmax_add", epilogue_phase, f"{cta}.epilogue",
        amount=valid_rows, resource="fp32", anchor="splitkv_mla.cuh:LSE + sM", cta=cta)
    builder.edge(lse, lse_add, "data")
    lse_tail = lse_add
    if not split:
        lse_tail = builder.node("softmax_mul", epilogue_phase, f"{cta}.epilogue",
            amount=valid_rows, resource="fp32",
            anchor="splitkv_mla.cuh:natural-log output scale", cta=cta)
    lse_select = builder.node("softmax_select", epilogue_phase, f"{cta}.epilogue",
        amount=valid_rows, resource="fp32", anchor="splitkv_mla.cuh:LSE zero/NaN select", cta=cta)
    builder.edge(lse_tail, lse_select, "data")
    lse_params: dict[str, Any] = {
        "bytes": valid_rows * 4,
        "lane_mode": "width64",
        "working_set_records": (builder.workload.batch_size
                                * builder.workload.num_heads_kv
                                * builder.workload.q_seq_per_hk),
        "pattern": "sequential",
    }
    if split:
        lse_params |= {
            "cache_mode": "hbm_stream",
            "memory_object": "partial_lse",
            "producer_consumer": True,
            "request": item.request_idx,
            "split": item.split_idx_within_request,
            "global_split_idx": global_split_idx,
            "kv_head": kv_head,
            "m_block": m_block,
            "q_row_start": m_block * 64,
            "q_row_count": valid_rows,
            "layout": "split_kvhead_qrow",
            "working_set_bytes": builder.request_partial_lse_working_set_bytes(
                item.request_idx
            ),
        }
    else:
        lse_params["memory_object"] = "final_lse"
    lse_store = builder.node("lse_store", epilogue_phase, f"{cta}.store", amount=valid_rows,
        resource="hbm", async_issue=True,
        anchor="splitkv_mla.cuh:gSoftmaxLse", cta=cta,
        params=lse_params)
    builder.edge(lse_select, lse_store, "data")
    request_done = None
    if has_next_request:
        request_done = builder.node("barrier_sync", epilogue_phase, f"{cta}.control",
            resource="barrier", anchor="splitkv_mla.cuh:inter-request __syncthreads",
            cta=cta, program_order=False)
        builder.edge(store, request_done, "barrier")
        builder.edge(lse_store, request_done, "barrier")
    return reduction, store, lse_store, request_done


def _causal_slice(workload: Workload, item: RequestSlice, m_block: int) -> RequestSlice:
    if not workload.causal:
        return item
    last_global_q = m_block * 64 + 63
    if last_global_q >= workload.q_seq_per_hk:
        common_mask = 0
    else:
        q_head_per_kv = workload.num_heads_q // workload.num_heads_kv
        s_q_idx = last_global_q // q_head_per_kv
        common_mask = workload.seqlen_q - s_q_idx - 1
    sequence = workload.seqlens_k[item.request_idx]
    last_block = math.ceil(max(0, sequence - common_mask) / workload.page_size)
    return RequestSlice(
        item.request_idx, item.partition_idx, item.start_block_idx,
        min(item.end_block_idx, last_block), item.split_idx_within_request,
        item.is_no_split,
    )


def build_dense_decode_dag(
    workload: Workload,
    resources: KernelResources,
    atom_map: AtomMap,
) -> DenseDecodeDAG:
    builder = _Builder(workload, resources, atom_map)
    scheduler = schedule_requests(
        workload.seqlens_k, sm_count=resources.sm_count, seqlen_q=workload.seqlen_q,
        num_heads_q=workload.num_heads_q, num_heads_kv=workload.num_heads_kv,
        page_size=workload.page_size,
    )
    builder.dag.scheduler = scheduler
    metadata_ready = _metadata(builder, scheduler)
    slice_by_partition: dict[int, list[RequestSlice]] = {}
    for item in scheduler.slices:
        slice_by_partition.setdefault(item.partition_idx, []).append(item)

    request_stores: dict[int, list[str]] = {index: [] for index in range(workload.batch_size)}
    pdl_ready_events: list[tuple[str, str]] = []
    main_grid_terminals: list[str] = []
    for head in range(workload.num_heads_kv):
        for m_block in range(workload.num_m_blocks):
            for partition in range(scheduler.num_sm_parts):
                cta = f"main.p{partition}.h{head}.m{m_block}"
                control = f"{cta}.control"
                prefetch = builder.node("descriptor_prefetch", "cta_init", control,
                    resource="l2", anchor="splitkv_mla.cuh:prefetch_tma_descriptor", cta=cta,
                    params={"mode": "all", "working_pages": builder.physical_page_count,
                            "working_tiles": builder.output_tile_count})
                if metadata_ready:
                    builder.edge(metadata_ready, prefetch, "data", "metadata kernel completion")
                init = builder.node("mbarrier_init", "cta_init", control, amount=19,
                    resource="barrier", anchor="splitkv_mla.cuh:barrier initialization", cta=cta)
                if metadata_ready:
                    builder.edge(metadata_ready, init, "data")
                init_fence = builder.node("proxy_fence", "cta_init", control,
                    resource="barrier", anchor="splitkv_mla.cuh:barrier init proxy fence", cta=cta)
                init_sync = builder.node("barrier_sync", "cta_init", control,
                    resource="barrier", anchor="splitkv_mla.cuh:__syncthreads after init", cta=cta)
                schedule_load = builder.node(
                    "scheduler_record_load", "cta_init", control,
                    resource="l2", anchor="splitkv_mla.cuh:tile_scheduler_metadata_ptr",
                    cta=cta, params={"bytes": 32, "cache_mode": (
                                         "l2_hot" if workload.metadata_mode == "generate"
                                         else workload.cache_mode
                                     ),
                                     "memory_object": "scheduler_metadata",
                                     "cache_line": partition // 4,
                                     "threads": 256, "issuers": 256,
                                     "pattern": "broadcast",
                                     "working_set_records": scheduler.num_sm_parts},
                )
                if metadata_ready:
                    builder.edge(metadata_ready, schedule_load, "data")
                previous_reduction: str | None = None
                previous_done: str | None = None
                final_output_store: str | None = None
                final_lse_store: str | None = None
                partition_slices = slice_by_partition.get(partition, [])
                for slice_index, scheduled_item in enumerate(partition_slices):
                    item = _causal_slice(workload, scheduled_item, m_block)
                    q = builder.node("q_tma", "request_setup", f"{cta}.tma", resource="tma",
                        async_issue=True, anchor="splitkv_mla.cuh:launch_q_copy", cta=cta,
                        params={"bytes": 64 * 576 * 2, "cache_mode": workload.cache_mode,
                                "memory_object": "q", "request": item.request_idx,
                                "kv_head": head, "m_block": m_block,
                                "depth": 1, "pattern": "sequential",
                                "working_set_pages": builder.output_tile_count})
                    builder.edge(init, q, "program")
                    builder.edge(prefetch, q, "program")
                    builder.edge(init_sync, q, "program")
                    builder.edge(schedule_load, q, "data")
                    q8 = {}
                    for wg in (0, 1):
                        q8[wg] = builder.node(
                            "q8_ldmatrix", "request_setup", f"{cta}.wg{wg}",
                            resource="shared",
                            anchor="splitkv_mla.cuh:retrieve_rP_from_sP Q tile8",
                            cta=cta, params={"m": 64, "n": 64,
                                             "warpgroup": wg, "warpgroups": 1})
                        builder.edge(q, q8[wg], "tma_ready")
                    if previous_reduction:
                        # Source launches the next request's Q copy before the
                        # previous epilogue/store completes.
                        builder.edge(previous_reduction, q, "buffer_reuse", "persistent Q buffer")
                    reduction, output_store, lse_store, request_done = _build_slice(
                        builder, item, cta=cta, q_ready=q, q8_ready=q8,
                        kv_head=head, m_block=m_block,
                        previous_done=previous_done,
                        has_next_request=slice_index + 1 < len(partition_slices),
                        valid_rows=min(64, workload.q_seq_per_hk - m_block * 64),
                        metadata_ready=schedule_load,
                    )
                    previous_reduction = reduction
                    previous_done = request_done
                    final_output_store = output_store
                    final_lse_store = lse_store
                    request_stores[item.request_idx].extend((output_store, lse_store))
                if previous_reduction:
                    assert final_output_store is not None and final_lse_store is not None
                    trigger = builder.node("pdl_launch", "combine_dispatch", control,
                        resource="grid", async_issue=True,
                        anchor="splitkv_mla.cuh:griddepcontrol.launch_dependents", cta=cta)
                    builder.edge(previous_reduction, trigger, "grid_dependency")
                    pdl_ready_events.append((trigger, "issue"))
                    main_grid_terminals.extend((final_output_store, final_lse_store))
                else:
                    # An out-of-range persistent CTA returns after reading its
                    # scheduler record and executes no launch-dependents opcode.
                    pdl_ready_events.append((schedule_load, "complete"))
                    main_grid_terminals.append(schedule_load)

    if scheduler.num_sm_parts > 160:
        raise ValueError("dense combine supports at most 160 scheduler partitions")
    max_splits = next(
        bucket for bucket in (32, 64, 96, 128, 160)
        if scheduler.num_sm_parts <= bucket
    )
    for request in range(workload.batch_size):
        split_count = scheduler.num_splits_prefix[request + 1] - scheduler.num_splits_prefix[request]
        for s_q_idx in range(workload.seqlen_q):
            for head_block in range((workload.num_heads_q + 7) // 8):
                valid_heads = min(8, workload.num_heads_q - head_block * 8)
                q_head_begin = head_block * 8
                q_heads = tuple(range(q_head_begin, q_head_begin + valid_heads))
                q_heads_per_kv = workload.num_heads_q // workload.num_heads_kv
                kv_heads = tuple(sorted({head // q_heads_per_kv for head in q_heads}))
                q_rows_in_kv = tuple(
                    s_q_idx * q_heads_per_kv + head % q_heads_per_kv
                    for head in q_heads
                )
                global_split_begin = scheduler.num_splits_prefix[request]
                partial_output_working_set = (
                    builder.request_partial_output_working_set_bytes(request)
                )
                partial_lse_working_set = (
                    builder.request_partial_lse_working_set_bytes(request)
                )
                rowsets = (
                    workload.batch_size * workload.seqlen_q
                    * math.ceil(workload.num_heads_q / 8)
                )

                def output_load_params(split_idx: int, chunk: int) -> dict[str, Any]:
                    return {
                        "bytes": valid_heads * 32 * 16,
                        "cache_mode": "producer_reuse",
                        "memory_object": "partial_output",
                        "producer_consumer": True,
                        "request": request,
                        "split": split_idx,
                        "global_split_idx": global_split_begin + split_idx,
                        "kv_head": kv_heads[0] if len(kv_heads) == 1 else -1,
                        "kv_heads": kv_heads,
                        "head_block": head_block,
                        "q_row": s_q_idx,
                        "q_rows_in_kv": q_rows_in_kv,
                        "chunk": chunk,
                        "segments": 1,
                        "rowsets": rowsets,
                        "warps": valid_heads,
                        "vectors_per_thread": 1,
                        "pattern": "sequential",
                        "working_set_bytes": partial_output_working_set,
                    }

                cta = f"combine.r{request}.q{s_q_idx}.h{head_block}"
                actor = f"{cta}.warpgroup"
                split_load = builder.node("combine_split_load", "combine_dispatch", actor,
                    amount=2 * valid_heads * 32, resource="l2", anchor="combine.cu:num_splits prefix loads",
                    cta=cta, params={"bytes": 8, "cache_mode": (
                                         "l2_hot" if workload.metadata_mode == "generate"
                                         else workload.cache_mode
                                     ),
                                     "request": request, "memory_object": "num_splits",
                                     "cache_line": request // 32,
                                     "threads": valid_heads * 32,
                                     "issuers": valid_heads * 32,
                                     "pattern": "broadcast",
                                     "working_set_entries": workload.batch_size + 1})
                split_sub = builder.node("combine_split_sub", "combine_dispatch", actor,
                    amount=valid_heads * 32, resource="int32",
                    anchor="combine.cu:end_split-start_split", cta=cta)
                dispatch = builder.node("combine_dispatch", "combine_dispatch", actor,
                    amount=valid_heads * 32, resource="int32",
                    anchor="combine.cu:no-op split test", cta=cta,
                    params={"num_splits": split_count, "valid_heads": valid_heads})
                for ready, event in pdl_ready_events:
                    builder.edge(ready, split_load, "grid_dependency",
                                 src_event=event, dst_event="issue")
                if split_count <= 1:
                    # Every grid CTA performs this test and returns before wait.
                    continue
                wait = builder.node("pdl_wait", "combine_dispatch", actor,
                    resource="grid", async_issue=True,
                    anchor="combine.cu:griddepcontrol.wait", cta=cta)
                builder.edge(dispatch, wait, "program")
                for producer in main_grid_terminals:
                    builder.edge(
                        producer, wait, "grid_dependency",
                        "producer grid completion", dst_event="complete",
                    )
                first_loads: list[str] = []
                for chunk in range(4):
                    first_load = builder.node(
                        "combine_output_load", "combine_accumulate", actor,
                        amount=valid_heads * 32, resource="l2", async_issue=True,
                        anchor="combine.cu:first-split prefetch", cta=cta,
                        params=output_load_params(0, chunk),
                    )
                    builder.edge(wait, first_load, "grid_dependency")
                    for store in request_stores[request]:
                        # The grid wait can issue before producer stores finish;
                        # each first-split chunk observes the completed producer.
                        builder.edge(store, first_load, "grid_dependency",
                                     "partial stores visible")
                    first_loads.append(first_load)
                lse_load = builder.node("combine_lse_load", "combine_lse", actor,
                    amount=max_splits * valid_heads, resource="l2",
                    anchor="combine.cu:gLseAccum predicated loads", cta=cta,
                    params={"bytes": split_count * valid_heads * 4,
                            "cache_mode": "producer_reuse",
                            "memory_object": "partial_lse",
                            "producer_consumer": True,
                            "request": request,
                            "split": "all",
                            "global_split_begin": global_split_begin,
                            "global_split_end": global_split_begin + split_count,
                            "kv_head": kv_heads[0] if len(kv_heads) == 1 else -1,
                            "kv_heads": kv_heads,
                            "head_block": head_block,
                            "q_row": s_q_idx,
                            "q_rows_in_kv": q_rows_in_kv,
                            "max_splits": max_splits, "segments": max_splits,
                            "split_stride": (workload.num_heads_kv
                                             * workload.q_seq_per_hk),
                            "rowsets": rowsets,
                            "warps": valid_heads, "pattern": "sequential",
                            "working_set_bytes": partial_lse_working_set})
                maximum = builder.node("softmax_max", "combine_lse", actor,
                    amount=(max_splits + 5 * 32) * valid_heads, resource="fp32",
                    anchor="combine.cu:LSE max bucket", cta=cta)
                max_shuffle = None
                for delta in (16, 8, 4, 2, 1):
                    max_shuffle = builder.node("shuffle", "combine_lse", actor,
                        amount=valid_heads * 32, resource="shuffle",
                        anchor="combine.cu:max LSE warp shuffle", cta=cta,
                        params={"delta": delta})
                exponent = builder.node("softmax_exp2", "combine_lse", actor,
                    amount=max_splits * valid_heads, resource="sfu",
                    anchor="combine.cu:exp2f LSE sum", cta=cta)
                lse_sum = builder.node("softmax_add", "combine_lse", actor,
                    amount=(max_splits + 5 * 32) * valid_heads, resource="fp32",
                    anchor="combine.cu:LSE sum and shuffle reduction", cta=cta)
                sum_shuffle = None
                for delta in (16, 8, 4, 2, 1):
                    sum_shuffle = builder.node("shuffle", "combine_lse", actor,
                        amount=valid_heads * 32, resource="shuffle",
                        anchor="combine.cu:sum LSE warp shuffle", cta=cta,
                        params={"delta": delta})
                lse_log = builder.node("lse", "combine_lse", actor,
                    amount=valid_heads * 32, resource="sfu", anchor="combine.cu:log2f(sum_lse)",
                    cta=cta)
                global_lse_add = builder.node("softmax_add", "combine_lse", actor,
                    amount=valid_heads * 32, resource="fp32",
                    anchor="combine.cu:log2f(sum_lse) + max_lse", cta=cta)
                lse_output_scale = builder.node("softmax_mul", "combine_lse", actor,
                    amount=valid_heads, resource="fp32",
                    anchor="combine.cu:global LSE natural-log conversion", cta=cta)
                lse_store = builder.node("lse_store", "combine_store", actor,
                    amount=valid_heads, resource="hbm", async_issue=True,
                    anchor="combine.cu:LSE store",
                    cta=cta, params={"bytes": valid_heads * 4,
                                     "lane_mode": "width8",
                                     "working_set_records": (workload.batch_size
                                                             * workload.seqlen_q
                                                             * workload.num_heads_q),
                                     "pattern": "sequential"})
                scale_exp = builder.node("softmax_exp2", "combine_lse", actor,
                    amount=max_splits * valid_heads, resource="sfu",
                    anchor="combine.cu:exp2f normalized scales", cta=cta)
                scale_store = builder.node("shared_store_u32", "combine_lse", actor,
                    amount=max_splits * valid_heads, resource="shared",
                    anchor="combine.cu:smem_buf scale store", cta=cta,
                    params={"threads": valid_heads * 32,
                            "producers": valid_heads * 32,
                            "topology": "contiguous",
                            "working_set_words": max_splits * valid_heads})
                scale_sync = builder.node("warp_sync", "combine_lse", actor,
                    resource="barrier", anchor="combine.cu:__syncwarp scales", cta=cta)
                previous_accumulate: str | None = None
                current_loads = first_loads
                for split_idx in range(split_count):
                    scale_load = builder.node(
                        "shared_load_u32", "combine_accumulate", actor,
                        amount=valid_heads * 32, resource="shared",
                        anchor="combine.cu:smem_buf scale load", cta=cta,
                        params={"split": split_idx, "threads": valid_heads * 32,
                                "pattern": "warp_broadcast",
                                "working_set_words": max_splits * valid_heads},
                    )
                    builder.edge(scale_sync, scale_load, "barrier")
                    next_loads: list[str] = []
                    for chunk in range(4):
                        accumulate = builder.node(
                            "combine_ffma", "combine_accumulate", actor,
                            amount=valid_heads * 128, resource="fp32",
                            anchor="combine.cu:weighted float4 accumulation chunk",
                            cta=cta, params={"split": split_idx, "chunk": chunk},
                        )
                        builder.edge(scale_load, accumulate, "data")
                        builder.edge(current_loads[chunk], accumulate, "data")
                        if previous_accumulate:
                            builder.edge(previous_accumulate, accumulate, "data")
                        previous_accumulate = accumulate
                        if split_idx + 1 < split_count:
                            next_load = builder.node(
                                "combine_output_load", "combine_accumulate", actor,
                                amount=valid_heads * 32, resource="l2", async_issue=True,
                                anchor="combine.cu:interleaved next-split float4 load",
                                cta=cta,
                                params=output_load_params(split_idx + 1, chunk),
                            )
                            builder.edge(accumulate, next_load, "program")
                            next_loads.append(next_load)
                    if next_loads:
                        current_loads = next_loads
                convert = builder.node("output_convert", "combine_store", actor,
                    amount=valid_heads * 512, resource="fp32",
                    anchor="combine.cu:output conversion", cta=cta)
                output = builder.node("combine_output_store", "combine_store", actor,
                    amount=valid_heads * 128, resource="hbm",
                    anchor="combine.cu:uint64 vector output store", cta=cta,
                    params={"bytes": valid_heads * 512 * 2,
                            "warps": valid_heads, "vectors_per_thread": 4,
                            "working_set_records": (workload.batch_size
                                                    * workload.seqlen_q
                                                    * math.ceil(workload.num_heads_q / 8)),
                            "pattern": "sequential"})
                builder.edge(previous_accumulate or scale_load, convert, "data")
                builder.edge(convert, output, "data")

    builder.dag.validate()
    return builder.dag
