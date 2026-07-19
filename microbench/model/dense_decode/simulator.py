"""Discrete-event CTA/resource simulation for FlashMLA dense decode."""

from __future__ import annotations

from collections import defaultdict, deque
import copy
from dataclasses import asdict, dataclass, field
import math
import random
from typing import Any, Iterable, Mapping

from .profile import ProfileError, ProfileLookup
from .scheduler import RequestSlice, SchedulerResult, schedule_requests
from .schema import Workload


LOCAL_RESOURCES = {
    "tensor",
    "tma",
    "sfu",
    "fp32",
    "int32",
    "shuffle",
    "shared",
    "barrier",
    "issue",
}
GLOBAL_RESOURCES = {"l2", "hbm"}


@dataclass
class AtomTask:
    name: str
    isolated_cycles: float
    demand: dict[str, float]
    provenance: list[Mapping[str, Any]] = field(default_factory=list)
    remaining_cycles: float = 0.0

    def __post_init__(self) -> None:
        if self.isolated_cycles < 0 or not math.isfinite(self.isolated_cycles):
            raise ValueError(f"invalid task cycles for {self.name}")
        self.remaining_cycles = self.isolated_cycles


@dataclass
class Phase:
    name: str
    tasks: list[AtomTask]


@dataclass
class CtaJob:
    job_id: str
    kind: str
    phases: deque[Phase]
    sm_id: int | None = None
    start_cycle: float | None = None
    stop_cycle: float | None = None
    active_tasks: list[AtomTask] = field(default_factory=list)

    def start_next_phase(self) -> None:
        self.active_tasks = self.phases.popleft().tasks if self.phases else []

    @property
    def complete(self) -> bool:
        return not self.phases and not self.active_tasks


@dataclass(frozen=True)
class Prediction:
    result: Mapping[str, Any]
    scheduler: SchedulerResult


class AtomCosts:
    def __init__(self, lookup: ProfileLookup, workload: Workload) -> None:
        self.lookup = lookup
        self.workload = workload
        self.provenance: list[Mapping[str, Any]] = []

    def hbm_fraction(self, label: str | None) -> float:
        if self.workload.cache_mode == "cold":
            return 1.0
        lowered = (label or "").lower()
        if "k_" in lowered or "k_page" in lowered:
            working_bytes = self.workload.unique_k_pages * 64 * 576 * 2
            if working_bytes <= self.lookup.l2_bytes:
                return 0.03
            overflow = 1.0 - self.lookup.l2_bytes / max(working_bytes, 1)
            locality = 1.0 - 0.6 * self.workload.page_reuse_ratio
            if self.workload.block_table_pattern == "random":
                locality = min(1.0, locality * 1.25)
            return min(1.0, max(0.03, overflow * locality))
        if "q_" in lowered:
            q_bytes = self.workload.batch_size * self.workload.q_seq_per_hk * 576 * 2
            return 0.03 if q_bytes <= self.lookup.l2_bytes else 0.35
        return 0.10

    def latency(
        self,
        names: Iterable[str],
        *,
        count: float,
        resource: str,
        params: Mapping[str, Any] | None = None,
        demand: Mapping[str, float] | None = None,
        label: str | None = None,
    ) -> AtomTask:
        query = {
            "dtype": self.workload.dtype,
            "blocks": 1,
            "cache_mode": self.workload.cache_mode,
            "pattern": self.workload.block_table_pattern,
        }
        if params:
            query.update(params)
        value, unit, provenance = self.lookup.interpolate(names, "latency", query)
        if "cycle" not in unit.lower():
            raise ProfileError(f"latency unit for {list(names)} is not cycle based: {unit}")
        self.provenance.extend(provenance)
        resources = dict(demand or {resource: 1.0})
        resources.setdefault("issue", 0.05)
        return AtomTask(label or next(iter(names)), value * count, resources, provenance)

    def memory(
        self,
        names: Iterable[str],
        *,
        byte_count: float,
        resource: str,
        params: Mapping[str, Any] | None = None,
        label: str | None = None,
    ) -> AtomTask:
        query = {
            "dtype": self.workload.dtype,
            "blocks": 1,
            "cache_mode": self.workload.cache_mode,
            "pattern": self.workload.block_table_pattern,
        }
        if params:
            query.update(params)
        bandwidth, unit, provenance = self.lookup.interpolate(
            names, "memory_bandwidth", query
        )
        if unit.lower() != "gb/s" or bandwidth <= 0:
            raise ProfileError(f"memory bandwidth for {list(names)} must be positive GB/s")
        seconds = byte_count / (bandwidth * 1.0e9)
        cycles = seconds * self.lookup.sm_clock_mhz * 1.0e6
        self.provenance.extend(provenance)
        demand = {"issue": 0.03}
        if resource in LOCAL_RESOURCES:
            demand[resource] = 1.0
        elif resource == "l2":
            demand["l2"] = min(bandwidth / self.lookup.l2_gbps, 1.0)
            demand["hbm"] = min(
                bandwidth / self.lookup.hbm_gbps, 1.0
            ) * self.hbm_fraction(label)
        if resource == "tma":
            demand["l2"] = min(bandwidth / self.lookup.l2_gbps, 1.0)
            demand["hbm"] = min(
                bandwidth / self.lookup.hbm_gbps, 1.0
            ) * self.hbm_fraction(label)
        return AtomTask(label or next(iter(names)), cycles, demand, provenance)


RESULT_ALIASES: dict[str, tuple[str, ...]] = {
    "exp2_f32": ("exp2_approx_ftz_f32",),
    "lg2_f32": ("lg2_approx_ftz_f32",),
    "rcp_f32": ("rcp_approx_ftz_f32",),
    "ffma_f32": ("ffma_rn_ftz_f32",),
    "fadd_f32": ("fadd_rn_ftz_f32",),
    "fsel_f32": ("fsel_f32", "fsetp_lt_ftz_f32"),
    "shfl_bfly_b32": ("shfl_sync_bfly_b32",),
    "barrier_sync": ("bar_sync_128", "bar_sync_256"),
    "isetp_u32": ("isetp_lt_u32",),
    "proxy_fence_async_shared": ("fence_proxy_async_shared_cta",),
    "stmatrix_p_b16": ("stmatrix_p_b16",),
    "stmatrix_o_b16": ("stmatrix_o_b16",),
    "tma_load_k_bf16_rank4": ("tma_load_k_64x64_bf16_rank4",),
    "tma_load_q_bf16_rank4": ("tma_load_q_64x576_bf16_rank4",),
    "tma_store_o_bf16_rank4": ("tma_store_o_64x512_bf16_rank4",),
    "bulk_store_oaccum_f32": ("bulk_store_oaccum_64x512_f32",),
    "shared_store_f32x2_stride520": ("shared_store_u64_dense",),
    "global_load_combine_float4": ("global_load_float4_oaccum",),
    "global_store_output_bf16x4": ("global_store_u64_output",),
    "global_store_output_fp16x4": ("global_store_u64_output",),
}


def _atom_names(base: str, dtype: str | None = None) -> tuple[str, ...]:
    stems: list[str] = []
    suffix = f"_{dtype}" if dtype else ""
    stems.extend((base + suffix, base))
    for alias in RESULT_ALIASES.get(base, ()):
        stems.extend((alias + suffix if dtype else alias, alias))
    unique = tuple(dict.fromkeys(stems))
    return tuple(f"dense_decode.{stem}" for stem in unique) + unique


def _softmax_phase(costs: AtomCosts, role: str) -> Phase:
    try:
        task = costs.latency(
            _atom_names("calibration.softmax_stage", costs.workload.dtype),
            count=1,
            resource="sfu",
            params={"blocks": 1},
            demand={
                "sfu": 1.0,
                "fp32": 0.75,
                "shuffle": 0.35,
                "issue": 0.45,
            },
            label=f"softmax_{role}.calibration",
        )
        return Phase(f"softmax_{role}_calibrated", [task])
    except ProfileError:
        pass

    # 64x64 page: 4,096 score exponentials plus max/sum reductions and
    # scale/update arithmetic. Independent atom service demands overlap on the
    # issue pipeline instead of being added into one opaque constant.
    tasks = [
        costs.latency(
            _atom_names("exp2_f32"), count=32, resource="sfu",
            params={"chains": 8}, label=f"softmax_{role}.exp2"
        ),
        costs.latency(
            _atom_names("shfl_bfly_b32"), count=4, resource="shuffle",
            params={"delta": 1}, label=f"softmax_{role}.shuffle"
        ),
        costs.latency(
            _atom_names("ffma_f32"), count=72, resource="fp32",
            label=f"softmax_{role}.ffma"
        ),
        costs.latency(
            _atom_names("fmax_f32", None) + _atom_names("fsel_f32"),
            count=18, resource="fp32", label=f"softmax_{role}.max_select"
        ),
    ]
    return Phase(f"softmax_{role}", tasks)


def _calibrated_epilogue_phase(
    costs: AtomCosts, request_slice: RequestSlice
) -> Phase | None:
    no_split = request_slice.is_no_split
    calibration = (
        "calibration.epilogue_nosplit_b16"
        if no_split else "calibration.epilogue_split_f32"
    )
    output_tiles = max(
        1,
        costs.workload.batch_size
        * costs.workload.num_heads_kv
        * costs.workload.num_m_blocks,
    )
    pattern = (
        "random" if costs.workload.block_table_pattern == "random"
        else "sequential"
    )
    try:
        task = costs.latency(
            _atom_names(calibration),
            count=1,
            resource="tma",
            params={
                "blocks": 1,
                "working_set_tiles": output_tiles,
                "pattern": pattern,
            },
            demand={
                "tma": 1.0,
                "shared": 0.75 if no_split else 0.90,
                "barrier": 0.20,
                "l2": 0.55 if no_split else 0.75,
                "hbm": (
                    0.45 if no_split else 0.60
                ) if costs.workload.cache_mode == "cold" else (
                    0.10 if no_split else 0.15
                ),
                "issue": 0.30 if no_split else 0.40,
            },
            label=calibration,
        )
    except ProfileError:
        return None
    return Phase(
        "nosplit_epilogue_calibrated" if no_split
        else "split_epilogue_calibrated",
        [task],
    )


def _page_kq_phase(costs: AtomCosts, first: bool) -> Phase:
    dtype = costs.workload.dtype
    composite = "calibration.kq_first_page" if first else "calibration.kq_steady_page"
    try:
        task = costs.latency(
            _atom_names(composite, dtype), count=1, resource="tensor",
            params={"warpgroups": 2},
            demand={"tensor": 1.0, "tma": 0.8, "l2": 0.55, "hbm": 0.45},
            label=composite,
        )
        return Phase(composite, [task])
    except ProfileError:
        ss_count = 36 if first else 32
        tasks = [
            costs.latency(
                _atom_names("wgmma_qk_ss", dtype), count=ss_count,
                resource="tensor", params={"group_size": 4}, label="qk_ss"
            )
        ]
        if not first:
            tasks.append(
                costs.latency(
                    _atom_names("wgmma_qk_rs", dtype), count=4,
                    resource="tensor", params={"group_size": 4}, label="qk_rs"
                )
            )
        tasks.append(
            costs.memory(
                _atom_names("tma_load_k_bf16_rank4"),
                byte_count=64 * 576 * 2,
                resource="tma",
                params={"depth": 9},
                label="k_page_tma",
            )
        )
        return Phase("kq_first_fallback" if first else "kq_steady_fallback", tasks)


def _pv_task(costs: AtomCosts, remote: bool, role: str) -> AtomTask:
    mode = "ss" if remote else "rs"
    return costs.latency(
        _atom_names(f"wgmma_pv_{mode}", costs.workload.dtype),
        count=4,
        resource="tensor",
        params={"group_size": 4},
        label=f"pv_{role}_{mode}",
    )


def _stmatrix_p(costs: AtomCosts, role: str) -> AtomTask:
    return costs.latency(
        _atom_names("stmatrix_p_b16"), count=1, resource="shared",
        label=f"stmatrix_p_{role}",
        demand={"shared": 1.0, "issue": 0.15},
    )


def _steady_page_pair_phase(costs: AtomCosts, pair: int) -> Phase | None:
    try:
        transition = costs.latency(
            _atom_names("page_pair_transition", costs.workload.dtype),
            count=1,
            resource="tensor",
            params={"warpgroups": 2, "blocks": 1},
            demand={
                "tensor": 1.0,
                "sfu": 0.30,
                "fp32": 0.25,
                "shared": 0.45,
                "barrier": 0.20,
                "issue": 0.35,
            },
            label=f"page_pair_transition_{pair}",
        )
    except ProfileError:
        return None
    k_pages = costs.memory(
        _atom_names("tma_load_k_bf16_rank4"),
        byte_count=2 * 64 * 576 * 2,
        resource="tma",
        params={"depth": 9},
        label=f"k_page_pair_tma_{pair}",
    )
    return Phase(f"steady_page_pair_{pair}", [transition, k_pages])


def _segment_phases(costs: AtomCosts, request_slice: RequestSlice) -> deque[Phase]:
    pages = request_slice.pages
    phases: deque[Phase] = deque()
    qload = costs.memory(
        _atom_names("tma_load_q_bf16_rank4"),
        byte_count=64 * 576 * 2,
        resource="tma",
        params={"depth": 1},
        label="q_tma",
    )
    phases.append(Phase("prologue_loads", [qload]))
    if pages == 0:
        phases.append(Phase("empty_request", []))
    else:
        phases.append(_page_kq_phase(costs, first=True))

    pairs = (pages + 1) // 2
    for pair in range(pairs):
        even_page = pair * 2
        odd_page = even_page + 1
        if pair > 0 and odd_page < pages:
            calibrated = _steady_page_pair_phase(costs, pair)
            if calibrated is not None:
                phases.append(calibrated)
                continue
        if even_page > 0:
            phases.append(_page_kq_phase(costs, first=False))
        phases.append(_softmax_phase(costs, f"even_{pair}"))
        if odd_page < pages:
            phases.append(
                Phase(
                    f"even_local_vs_odd_qk_{pair}",
                    [_pv_task(costs, False, f"even_{pair}")] +
                    _page_kq_phase(costs, first=False).tasks,
                )
            )
            phases.append(_softmax_phase(costs, f"odd_{pair}"))
            phases.append(
                Phase(
                    f"p_exchange_and_remote_pv_{pair}",
                    [
                        _stmatrix_p(costs, f"odd_{pair}"),
                        _stmatrix_p(costs, f"even_{pair}"),
                        _pv_task(costs, False, f"odd_{pair}"),
                        _pv_task(costs, True, f"even_{pair}"),
                        _pv_task(costs, True, f"odd_{pair}"),
                    ],
                )
            )
        else:
            phases.append(
                Phase(
                    "single_tail_pv",
                    [
                        _pv_task(costs, False, "single_tail"),
                        _stmatrix_p(costs, "single_tail"),
                        _pv_task(costs, True, "single_tail"),
                    ],
                )
            )

    phases.append(
        Phase(
            "l_reduction",
            [
                costs.latency(
                    _atom_names("shfl_bfly_b32"), count=4,
                    resource="shuffle", params={"delta": 1}, label="l_reduce_shuffle"
                ),
                costs.latency(
                    _atom_names("barrier_sync"), count=1,
                    resource="barrier", params={"participants": 256}, label="l_reduce_cta_barrier"
                ),
                costs.latency(
                    _atom_names("fadd_f32"), count=4,
                    resource="fp32", label="l_reduce_add"
                ),
            ],
        )
    )
    calibrated_epilogue = _calibrated_epilogue_phase(costs, request_slice)
    if calibrated_epilogue is not None:
        phases.append(calibrated_epilogue)
    elif request_slice.is_no_split:
        phases.append(
            Phase(
                "nosplit_epilogue_stage",
                [
                    costs.latency(
                        _atom_names("stmatrix_o_b16"), count=2,
                        resource="shared", label="o_stmatrix"
                    )
                ],
            )
        )
        phases.append(
            Phase(
                "nosplit_epilogue_fence",
                [
                    costs.latency(
                        _atom_names("proxy_fence_async_shared"), count=1,
                        resource="barrier", label="o_proxy_fence"
                    )
                ],
            )
        )
        phases.append(
            Phase(
                "nosplit_epilogue_store",
                [
                    costs.memory(
                        _atom_names("tma_store_o_bf16_rank4"),
                        byte_count=64 * 512 * 2,
                        resource="tma",
                        params={"depth": 1},
                        label="o_tma_store",
                    ),
                ],
            )
        )
    else:
        phases.append(
            Phase(
                "split_epilogue_stage",
                [
                    costs.latency(
                        _atom_names("shared_store_f32x2_stride520"), count=64,
                        resource="shared", label="oaccum_shared_stage"
                    )
                ],
            )
        )
        phases.append(
            Phase(
                "split_epilogue_fence",
                [
                    costs.latency(
                        _atom_names("proxy_fence_async_shared"), count=1,
                        resource="barrier", label="oaccum_proxy_fence"
                    )
                ],
            )
        )
        phases.append(
            Phase(
                "split_epilogue_store",
                [
                    costs.memory(
                        _atom_names("bulk_store_oaccum_f32"),
                        byte_count=64 * 512 * 4,
                        resource="tma",
                        params={"depth": 1},
                        label="oaccum_bulk_store",
                    ),
                ],
            )
        )
    return phases


def _make_main_jobs(
    costs: AtomCosts,
    workload: Workload,
    scheduler: SchedulerResult,
) -> list[CtaJob]:
    slices_by_partition: dict[int, list[RequestSlice]] = defaultdict(list)
    for item in scheduler.slices:
        slices_by_partition[item.partition_idx].append(item)
    jobs: list[CtaJob] = []
    for m_block in range(workload.num_m_blocks):
        for kv_head in range(workload.num_heads_kv):
            for partition in range(scheduler.num_sm_parts):
                phases: deque[Phase] = deque()
                for item in slices_by_partition.get(partition, []):
                    phases.extend(_segment_phases(costs, item))
                if phases:
                    jobs.append(
                        CtaJob(
                            job_id=f"main.m{m_block}.h{kv_head}.p{partition}",
                            kind="main",
                            phases=phases,
                        )
                    )
    return jobs


def _make_combine_jobs(costs: AtomCosts, workload: Workload, scheduler: SchedulerResult) -> list[CtaJob]:
    jobs: list[CtaJob] = []
    split_counts = [
        scheduler.num_splits_prefix[index + 1] - scheduler.num_splits_prefix[index]
        for index in range(workload.batch_size)
    ]
    head_blocks = math.ceil(workload.num_heads_q / 8)
    for request, split_count in enumerate(split_counts):
        for query in range(workload.seqlen_q):
            for head_block in range(head_blocks):
                if split_count <= 1:
                    try:
                        split_load = costs.memory(
                            _atom_names("global_load_i32_ordinary"),
                            byte_count=8,
                            resource="l2",
                            params={"issuers": 1, "pattern": "broadcast"},
                            label="combine_num_splits_load",
                        )
                    except ProfileError:
                        split_load = costs.latency(
                            _atom_names("iadd3_u32"), count=2,
                            resource="int32", label="combine_num_splits_load_fallback"
                        )
                    try:
                        branch = costs.latency(
                            _atom_names("isetp_u32"), count=1,
                            resource="int32", label="combine_noop_branch"
                        )
                    except ProfileError:
                        branch = costs.latency(
                            _atom_names("iadd3_u32"), count=1,
                            resource="int32", label="combine_noop_branch_fallback"
                        )
                    phases = deque(
                        [
                            Phase(
                                "combine_noop",
                                [split_load, branch],
                            )
                        ]
                    )
                    jobs.append(
                        CtaJob(
                            job_id=f"combine_noop.r{request}.q{query}.h{head_block}",
                            kind="combine_noop",
                            phases=phases,
                        )
                    )
                    continue
                byte_count = split_count * 8 * 512 * 4
                try:
                    stage = costs.latency(
                        _atom_names("calibration.combine_stage", workload.dtype),
                        count=1,
                        resource="l2",
                        params={"num_splits": split_count, "blocks": 1},
                        demand={
                            "l2": 0.75,
                            "hbm": 0.50 if workload.cache_mode == "cold" else 0.10,
                            "sfu": 0.20,
                            "fp32": 0.65,
                            "shared": 0.30,
                            "issue": 0.35,
                        },
                        label="combine_stage",
                    )
                    phases = deque([Phase("combine_stage", [stage])])
                    jobs.append(
                        CtaJob(
                            job_id=f"combine.r{request}.q{query}.h{head_block}",
                            kind="combine",
                            phases=phases,
                        )
                    )
                    continue
                except ProfileError:
                    pass
                phases = deque(
                    [
                        Phase(
                            "combine_load",
                            [
                                costs.memory(
                                    _atom_names("global_load_combine_float4"),
                                    byte_count=byte_count,
                                    resource="l2",
                                    params={"num_splits": split_count},
                                    label="combine_oaccum_load",
                                )
                            ],
                        ),
                        Phase(
                            "combine_lse",
                            [
                                costs.latency(_atom_names("exp2_f32"), count=split_count,
                                              resource="sfu", label="combine_exp2"),
                                costs.latency(_atom_names("lg2_f32"), count=1,
                                              resource="sfu", label="combine_lg2"),
                                costs.latency(_atom_names("shfl_bfly_b32"), count=10,
                                              resource="shuffle", params={"delta": 16},
                                              label="combine_shuffle"),
                            ],
                        ),
                        Phase(
                            "combine_accumulate_store",
                            [
                                costs.latency(_atom_names("ffma_f32"), count=split_count * 512,
                                              resource="fp32", label="combine_ffma"),
                                costs.memory(
                                    _atom_names(f"global_store_output_{workload.dtype}x4"),
                                    byte_count=8 * 512 * 2,
                                    resource="l2",
                                    label="combine_output_store",
                                ),
                            ],
                        ),
                    ]
                )
                jobs.append(
                    CtaJob(
                        job_id=f"combine.r{request}.q{query}.h{head_block}",
                        kind="combine",
                        phases=phases,
                    )
                )
    return jobs


def _resource_rates(
    jobs: Iterable[CtaJob], sm_count: int
) -> tuple[dict[int, dict[str, float]], dict[str, float]]:
    local_totals: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    global_totals: dict[str, float] = defaultdict(float)
    for job in jobs:
        if job.sm_id is None:
            continue
        for task in job.active_tasks:
            for resource, demand in task.demand.items():
                if resource in GLOBAL_RESOURCES:
                    global_totals[resource] += demand
                else:
                    local_totals[job.sm_id][resource] += demand

    rates: dict[int, dict[str, float]] = defaultdict(dict)
    for job in jobs:
        if job.sm_id is None:
            continue
        for task in job.active_tasks:
            slowdown = 1.0
            for resource, demand in task.demand.items():
                if demand <= 0:
                    continue
                total = (
                    global_totals[resource]
                    if resource in GLOBAL_RESOURCES
                    else local_totals[job.sm_id][resource]
                )
                slowdown = max(slowdown, total)
            rates[id(task)]["rate"] = 1.0 / slowdown
    utilization: dict[str, float] = {}
    for resource in LOCAL_RESOURCES:
        utilization[resource] = sum(
            min(local_totals[sm_id].get(resource, 0.0), 1.0)
            for sm_id in range(sm_count)
        ) / sm_count
    for resource in GLOBAL_RESOURCES:
        utilization[resource] = min(global_totals.get(resource, 0.0), 1.0)
    return rates, utilization


def _simulate_jobs(
    jobs: list[CtaJob],
    *,
    sm_count: int,
    residency: int,
) -> tuple[float, dict[str, Any]]:
    if not jobs:
        return 0.0, {"jobs": 0, "waves": 0, "tail_cycles": 0.0}
    pending = deque(jobs)
    active: list[CtaJob] = []
    slots = [residency] * sm_count
    now = 0.0
    completed: list[CtaJob] = []
    utilization_area: dict[str, float] = defaultdict(float)

    def admit() -> None:
        nonlocal pending
        while pending:
            choices = [index for index, count in enumerate(slots) if count > 0]
            if not choices:
                return
            sm_id = max(choices, key=lambda index: slots[index])
            job = pending.popleft()
            job.sm_id = sm_id
            job.start_cycle = now
            slots[sm_id] -= 1
            job.start_next_phase()
            active.append(job)

    admit()
    while active:
        for job in list(active):
            if not job.active_tasks and job.phases:
                job.start_next_phase()
        rates, interval_utilization = _resource_rates(active, sm_count)
        completion_times = []
        for job in active:
            for task in job.active_tasks:
                rate = rates[id(task)]["rate"]
                completion_times.append(task.remaining_cycles / rate)
        if not completion_times:
            for job in list(active):
                if job.complete:
                    job.stop_cycle = now
                    slots[job.sm_id or 0] += 1
                    active.remove(job)
                    completed.append(job)
            admit()
            continue
        delta = min(completion_times)
        if not math.isfinite(delta) or delta < 0:
            raise RuntimeError("event simulator produced an invalid time step")
        now += delta
        for resource, value in interval_utilization.items():
            utilization_area[resource] += value * delta
        for job in active:
            survivors = []
            for task in job.active_tasks:
                task.remaining_cycles -= delta * rates[id(task)]["rate"]
                if task.remaining_cycles > 1.0e-9:
                    survivors.append(task)
            job.active_tasks = survivors
        for job in list(active):
            if not job.active_tasks:
                if job.phases:
                    job.start_next_phase()
                else:
                    job.stop_cycle = now
                    slots[job.sm_id or 0] += 1
                    active.remove(job)
                    completed.append(job)
        admit()

    starts = [job.start_cycle or 0.0 for job in completed]
    stops = [job.stop_cycle or 0.0 for job in completed]
    first_finish = min(stops)
    return now, {
        "jobs": len(completed),
        "waves": math.ceil(len(completed) / (sm_count * residency)),
        "first_finish_cycle": first_finish,
        "tail_cycles": max(stops) - first_finish,
        "mean_job_cycles": sum(stop - start for start, stop in zip(starts, stops)) / len(stops),
        "resource_utilization": {
            resource: area / now if now > 0 else 0.0
            for resource, area in sorted(utilization_area.items())
        },
    }


def _metadata_cycles(costs: AtomCosts, workload: Workload, parts: int) -> float:
    if workload.metadata_mode == "reuse":
        return 0.0
    try:
        task = costs.latency(
            _atom_names("calibration.metadata_stage"),
            count=1,
            resource="int32",
            params={"batch": workload.batch_size, "num_sm_parts": parts},
        )
        return task.isolated_cycles
    except ProfileError:
        shfl = costs.latency(
            _atom_names("shfl_bfly_b32"), count=5,
            resource="shuffle", params={"delta": 16}
        ).isolated_cycles
        integer = costs.latency(
            _atom_names("iadd3_u32"), count=workload.batch_size + parts * 8,
            resource="int32"
        ).isolated_cycles
        return shfl + integer


def _launch_cycles(lookup: ProfileLookup, name: str, default_us: float) -> float:
    try:
        value, unit, _ = lookup.interpolate(_atom_names(name), "latency", {})
        if "cycle" in unit.lower():
            return value
        if unit.lower() in {"us", "microsecond", "microseconds"}:
            return value * lookup.sm_clock_mhz
    except ProfileError:
        pass
    return default_us * lookup.sm_clock_mhz


def predict(
    profile: Mapping[str, Any],
    workload: Workload,
    *,
    bootstrap: int = 0,
    _include_sensitivity: bool = True,
) -> Prediction:
    lookup = ProfileLookup(profile)
    scheduler = schedule_requests(
        workload.seqlens_k,
        sm_count=lookup.sm_count,
        seqlen_q=workload.seqlen_q,
        num_heads_q=workload.num_heads_q,
        num_heads_kv=workload.num_heads_kv,
        page_size=workload.page_size,
    )
    costs = AtomCosts(lookup, workload)
    metadata_cycles = _metadata_cycles(costs, workload, scheduler.num_sm_parts)
    main_jobs = _make_main_jobs(costs, workload, scheduler)
    main_cycles, main_stats = _simulate_jobs(
        main_jobs, sm_count=lookup.sm_count, residency=lookup.main_cta_residency
    )
    combine_jobs = _make_combine_jobs(costs, workload, scheduler)
    combine_cycles, combine_stats = _simulate_jobs(
        combine_jobs, sm_count=lookup.sm_count, residency=lookup.combine_cta_residency
    )
    launch_cycles = (
        _launch_cycles(lookup, "launch_metadata", 3.0)
        if workload.metadata_mode == "generate"
        else 0.0
    )
    launch_cycles += _launch_cycles(lookup, "launch_main", 3.0)
    launch_cycles += _launch_cycles(lookup, "launch_combine", 3.0)

    # PDL can overlap combine launch/readiness with the tail of main. A measured
    # calibration record supplies the credit. Without it, the model makes no
    # speculative overlap and records zero credit.
    pdl_overlap = 0.0
    pdl_calibrated = False
    try:
        pdl_overlap, unit, pdl_prov = lookup.interpolate(
            _atom_names("calibration.pdl_overlap"), "latency",
            {"producer_blocks": len(main_jobs), "consumer_blocks": len(combine_jobs)}
        )
        if "cycle" in unit.lower():
            pass
        elif unit.lower() in {"us", "microsecond", "microseconds"}:
            pdl_overlap *= lookup.sm_clock_mhz
        else:
            raise ProfileError("PDL overlap metric must use cycles or microseconds")
        costs.provenance.extend(pdl_prov)
        pdl_overlap = min(pdl_overlap, main_cycles, combine_cycles)
        pdl_calibrated = True
    except ProfileError:
        pdl_overlap = 0.0

    total_cycles = metadata_cycles + main_cycles + combine_cycles - pdl_overlap + launch_cycles
    total_us = total_cycles / lookup.sm_clock_mhz
    stage_cycles = {
        "metadata": metadata_cycles,
        "main": main_cycles,
        "combine": combine_cycles,
        "launch": launch_cycles,
    }
    dominant_stage = max(stage_cycles, key=stage_cycles.get)
    resource_values: dict[str, float] = defaultdict(float)
    for stats in (main_stats, combine_stats):
        for resource, value in stats.get("resource_utilization", {}).items():
            resource_values[resource] = max(resource_values[resource], float(value))
    dominant_resource = max(resource_values, key=resource_values.get) if resource_values else None
    unique_provenance = []
    seen = set()
    for item in costs.provenance:
        key = (item.get("source_file"), item.get("record_index"))
        if key not in seen:
            seen.add(key)
            unique_provenance.append(dict(item))

    main_multiplier = workload.num_m_blocks * workload.num_heads_kv
    q_bytes = len(scheduler.slices) * main_multiplier * 64 * 576 * 2
    k_bytes = sum(item.pages for item in scheduler.slices) * main_multiplier * 64 * 576 * 2
    nosplit_count = sum(item.is_no_split for item in scheduler.slices) * main_multiplier
    split_count = sum(not item.is_no_split for item in scheduler.slices) * main_multiplier
    output_bytes = nosplit_count * 64 * 512 * 2
    partial_bytes = split_count * 64 * 512 * 4
    split_counts = [
        scheduler.num_splits_prefix[index + 1] - scheduler.num_splits_prefix[index]
        for index in range(workload.batch_size)
    ]
    combine_read_bytes = sum(
        count * workload.seqlen_q * workload.num_heads_q * 512 * 4
        for count in split_counts if count > 1
    )
    combine_write_bytes = sum(
        workload.seqlen_q * workload.num_heads_q * 512 * 2
        for count in split_counts if count > 1
    )
    unique_k_bytes = workload.unique_k_pages * main_multiplier * 64 * 576 * 2
    if workload.cache_mode == "cold":
        estimated_k_hbm_bytes = k_bytes
    elif unique_k_bytes <= lookup.l2_bytes:
        estimated_k_hbm_bytes = unique_k_bytes * 0.03
    else:
        resident_fraction = lookup.l2_bytes / max(unique_k_bytes, 1)
        estimated_k_hbm_bytes = unique_k_bytes * max(0.03, 1.0 - resident_fraction)

    warnings = []
    if not scheduler.source_defined:
        warnings.append(str(scheduler.undefined_reason))
    if not pdl_calibrated:
        warnings.append(
            "PDL overlap calibration is absent; main and combine are conservatively serialized"
        )
    occupancy = profile.get("occupancy", {})
    fallback_occupancy = [
        kind for kind in ("main", "combine")
        if isinstance(occupancy, dict)
        and isinstance(occupancy.get(kind), dict)
        and occupancy[kind].get("source") == "planning_fallback"
    ]
    if fallback_occupancy:
        warnings.append(
            "CTA residency uses planning fallback for: " + ", ".join(fallback_occupancy)
        )

    result: dict[str, Any] = {
        "schema_version": 1,
        "model_kind": "discrete_event_resource",
        "profile_id": profile.get("profile_id"),
        "case_id": workload.case_id,
        "workload": workload.to_json(),
        "predicted_e2e_us": {
            "p50": total_us,
            "p10": total_us,
            "p90": total_us,
        },
        "breakdown_us": {
            "metadata": metadata_cycles / lookup.sm_clock_mhz,
            "main": main_cycles / lookup.sm_clock_mhz,
            "combine": combine_cycles / lookup.sm_clock_mhz,
            "pdl_overlap_credit": pdl_overlap / lookup.sm_clock_mhz,
            "launch": launch_cycles / lookup.sm_clock_mhz,
        },
        "scheduler": scheduler.to_json(),
        "cta": {"main": main_stats, "combine": combine_stats},
        "memory": {
            "cache_mode": workload.cache_mode,
            "block_table_pattern": workload.block_table_pattern,
            "block_table_source": workload.block_table_source,
            "logical_k_pages": workload.logical_k_pages,
            "unique_k_pages": workload.unique_k_pages,
            "page_reuse_ratio": workload.page_reuse_ratio,
            "l2_bytes": lookup.l2_bytes,
            "hbm_peak_gbps": lookup.hbm_gbps,
            "l2_reference_gbps": lookup.l2_gbps,
            "requested_bytes": {
                "q_tma": q_bytes,
                "k_tma": k_bytes,
                "nosplit_output": output_bytes,
                "split_partial_output": partial_bytes,
                "combine_read": combine_read_bytes,
                "combine_write": combine_write_bytes,
            },
            "estimated_hbm_bytes": {
                "k": estimated_k_hbm_bytes,
                "other": (
                    q_bytes + output_bytes + partial_bytes +
                    combine_read_bytes + combine_write_bytes
                ) * (1.0 if workload.cache_mode == "cold" else 0.10),
            },
        },
        "resource_model": {
            "local_resources": sorted(LOCAL_RESOURCES),
            "global_resources": sorted(GLOBAL_RESOURCES),
            "sharing": "event-driven bottleneck weighted fair sharing",
            "main_cta_residency": lookup.main_cta_residency,
            "combine_cta_residency": lookup.combine_cta_residency,
            "occupancy": profile.get("occupancy", {}),
        },
        "critical_path": {
            "dominant_stage": dominant_stage,
            "dominant_resource": dominant_resource,
            "dominant_resource_utilization": (
                resource_values[dominant_resource] if dominant_resource else None
            ),
        },
        "provenance": unique_provenance,
        "profile_record_count": len(profile.get("records", [])),
        "static_artifact_hashes": profile.get("static_artifact_hashes", {}),
        "incomplete_calibration": bool(warnings),
        "warnings": warnings,
    }

    groups = {
        "tensor_and_page_pipeline": ("wgmma", "calibration.kq", "page_pair"),
        "memory_path": ("tma", "bulk_store", "global_", "combine_stage"),
        "softmax_stage": ("softmax_stage",),
        "epilogue_protocol": ("epilogue_nosplit", "epilogue_split"),
        "sfu": ("exp2", "lg2", "rcp"),
        "fp32_alu": ("ffma", "fadd", "fmul", "fsel", "fsetp"),
        "synchronization": ("bar_", "mbarrier", "fence_proxy"),
    }
    sensitivity = []
    if _include_sensitivity:
        for group, tokens in groups.items():
            perturbed = copy.deepcopy(profile)
            changed = 0
            for record in perturbed.get("records", []):
                name = str(record.get("name", ""))
                if not any(token in name for token in tokens):
                    continue
                metric_value = record.get("latency", {}).get("value")
                if isinstance(metric_value, (int, float)) and not isinstance(metric_value, bool):
                    record["latency"]["value"] = float(metric_value) * 1.10
                    changed += 1
                bandwidth_value = record.get("memory_bandwidth", {}).get("value")
                if group == "memory_path" and isinstance(
                    bandwidth_value, (int, float)
                ) and not isinstance(bandwidth_value, bool) and bandwidth_value > 0:
                    record["memory_bandwidth"]["value"] = float(bandwidth_value) / 1.10
            if changed == 0:
                continue
            slower_us = float(
                predict(
                    perturbed, workload, bootstrap=0, _include_sensitivity=False
                ).result["predicted_e2e_us"]["p50"]
            )
            sensitivity.append(
                {
                    "parameter_group": group,
                    "perturbation": "10% slower calibrated service",
                    "affected_records": changed,
                    "delta_us": slower_us - total_us,
                    "delta_ratio": (slower_us - total_us) / total_us if total_us else 0.0,
                }
            )
        result["parameter_sensitivity"] = sorted(
            sensitivity, key=lambda item: abs(item["delta_us"]), reverse=True
        )

    if bootstrap > 0:
        rng = random.Random(0)
        samples: list[float] = []
        sensitivity_by_group = {
            item["parameter_group"]: item for item in sensitivity
        }
        empirical: dict[str, list[list[float]]] = defaultdict(list)
        for record in profile.get("records", []):
            name = str(record.get("name", ""))
            raw = record.get("raw_samples", {})
            if not isinstance(raw, dict):
                continue
            for group, tokens in groups.items():
                if group not in sensitivity_by_group or not any(
                    token in name for token in tokens
                ):
                    continue
                metric_name = "memory_bandwidth" if group == "memory_path" else "latency"
                base_metric = record.get(metric_name, {}).get("value")
                values = raw.get(metric_name)
                if (
                    not isinstance(base_metric, (int, float))
                    or isinstance(base_metric, bool)
                    or base_metric <= 0
                    or not isinstance(values, list)
                ):
                    continue
                finite = [
                    float(value) for value in values
                    if isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and float(value) > 0
                ]
                if not finite:
                    continue
                if metric_name == "memory_bandwidth":
                    empirical[group].append(
                        [float(base_metric) / value for value in finite]
                    )
                else:
                    empirical[group].append(
                        [value / float(base_metric) for value in finite]
                    )
        for _ in range(bootstrap):
            sampled_us = total_us
            for group, record_samples in empirical.items():
                ratios = sorted(rng.choice(values) for values in record_samples)
                ratio = ratios[len(ratios) // 2]
                delta_for_ten_percent = sensitivity_by_group[group]["delta_us"]
                sampled_us += (ratio - 1.0) * delta_for_ten_percent / 0.10
            samples.append(max(sampled_us, 0.0))
        if empirical and samples:
            samples.sort()
            result["predicted_e2e_us"] = {
                "p10": samples[max(0, int(0.10 * (len(samples) - 1)))],
                "p50": samples[int(0.50 * (len(samples) - 1))],
                "p90": samples[int(0.90 * (len(samples) - 1))],
            }
            result["uncertainty"] = {
                "method": "empirical microbenchmark bootstrap",
                "samples": len(samples),
            }
        else:
            result["warnings"].append(
                "raw_samples are absent; p10/p90 uncertainty cannot be estimated"
            )
            result["uncertainty"] = {"method": "unavailable", "samples": 0}
    return Prediction(result=result, scheduler=scheduler)
