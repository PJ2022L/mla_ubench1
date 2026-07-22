"""Validated inputs for the SM90 dense-decode atom-DAG model."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Mapping


@dataclass(frozen=True)
class Workload:
    case_id: str
    dtype: str
    batch_size: int
    seqlen_q: int
    num_heads_q: int
    num_heads_kv: int
    head_dim_qk: int
    head_dim_v: int
    page_size: int
    seqlens_k: tuple[int, ...]
    block_table: tuple[tuple[int, ...], ...]
    causal: bool
    metadata_mode: str
    cache_mode: str
    block_table_pattern: str

    @property
    def q_seq_per_hk(self) -> int:
        return self.seqlen_q * (self.num_heads_q // self.num_heads_kv)

    @property
    def num_m_blocks(self) -> int:
        return math.ceil(self.q_seq_per_hk / 64)

    def to_json(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "seqlen_q": self.seqlen_q,
            "num_heads_q": self.num_heads_q,
            "num_heads_kv": self.num_heads_kv,
            "head_dim_qk": self.head_dim_qk,
            "head_dim_v": self.head_dim_v,
            "page_size": self.page_size,
            "seqlens_k": list(self.seqlens_k),
            "block_table": [list(row) for row in self.block_table],
            "causal": self.causal,
            "metadata_mode": self.metadata_mode,
            "cache_mode": self.cache_mode,
            "block_table_pattern": self.block_table_pattern,
        }


@dataclass(frozen=True)
class KernelResources:
    sm_count: int = 132
    sm_clock_mhz: float = 1_620.0
    l2_bytes: int = 52_428_800
    main_registers_per_thread: int = 232
    main_shared_bytes: int = 230_400
    main_threads: int = 256
    main_min_blocks_per_sm: int = 1
    combine_registers_per_thread: int = 64
    combine_shared_bytes: int = 0
    combine_threads: int = 256
    combine_min_blocks_per_sm: int = 0
    registers_per_sm: int = 65_536
    shared_bytes_per_sm: int = 232_448
    max_threads_per_sm: int = 2_048
    max_ctas_per_sm: int = 16

    def residency(self, kernel: str) -> int:
        prefix = "main" if kernel == "main" else "combine"
        threads = getattr(self, f"{prefix}_threads")
        registers = getattr(self, f"{prefix}_registers_per_thread")
        shared = getattr(self, f"{prefix}_shared_bytes")
        limits = [self.max_ctas_per_sm, self.max_threads_per_sm // max(threads, 1)]
        if registers:
            limits.append(self.registers_per_sm // max(registers * threads, 1))
        if shared:
            limits.append(self.shared_bytes_per_sm // shared)
        residency = max(1, min(limits))
        minimum = getattr(self, f"{prefix}_min_blocks_per_sm")
        if minimum and residency < minimum:
            raise ValueError(
                f"{kernel} cubin resources permit {residency} CTA/SM, below the "
                f"__launch_bounds__ compiler target of {minimum}"
            )
        return residency

    def to_json(self) -> dict[str, Any]:
        return dict(self.__dict__) | {
            "main_residency": self.residency("main"),
            "combine_residency": self.residency("combine"),
        }


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _lengths(value: Mapping[str, Any], batch: int) -> tuple[int, ...]:
    explicit = value.get("seqlens_k")
    distribution = value.get("seqlens_k_distribution")
    if (explicit is None) == (distribution is None):
        raise ValueError("provide exactly one of seqlens_k or seqlens_k_distribution")
    if explicit is not None:
        if not isinstance(explicit, list) or len(explicit) != batch:
            raise ValueError("seqlens_k must contain batch_size entries")
        result = tuple(int(item) for item in explicit)
    else:
        if not isinstance(distribution, dict):
            raise ValueError("seqlens_k_distribution must be an object")
        rng = random.Random(int(distribution.get("seed", 0)))
        kind = str(distribution.get("kind", "fixed"))
        if kind == "fixed":
            result = (int(distribution["value"]),) * batch
        elif kind == "uniform":
            low, high = int(distribution["low"]), int(distribution["high"])
            result = tuple(rng.randint(low, high) for _ in range(batch))
        elif kind == "choice":
            choices = tuple(int(item) for item in distribution["values"])
            result = tuple(rng.choice(choices) for _ in range(batch))
        else:
            raise ValueError("seqlens_k_distribution.kind must be fixed, uniform, or choice")
    if any(item < 0 for item in result):
        raise ValueError("seqlens_k entries must be non-negative")
    return result


def _block_table(
    value: Mapping[str, Any], lengths: tuple[int, ...], page_size: int, pattern: str
) -> tuple[tuple[int, ...], ...]:
    required = tuple(math.ceil(length / page_size) for length in lengths)
    explicit = value.get("block_table")
    if explicit is not None:
        if not isinstance(explicit, list) or len(explicit) != len(lengths):
            raise ValueError("block_table must contain one row per request")
        rows = []
        for index, (row, count) in enumerate(zip(explicit, required)):
            if not isinstance(row, list) or len(row) < count:
                raise ValueError(f"block_table[{index}] requires at least {count} pages")
            parsed = tuple(int(item) for item in row[:count])
            if any(item < 0 for item in parsed):
                raise ValueError("physical page ids must be non-negative")
            rows.append(parsed)
        return tuple(rows)
    config = value.get("block_table_distribution", {})
    config = config if isinstance(config, dict) else {}
    kind = str(config.get("kind", pattern))
    rng = random.Random(int(config.get("seed", 0)))
    pool = max(1, int(config.get("pool_pages", sum(required) or 1)))
    rows: list[tuple[int, ...]] = []
    cursor = 0
    for count in required:
        if kind == "contiguous":
            row = tuple(range(cursor, cursor + count))
            cursor += count
        elif kind == "random":
            row = tuple(rng.randrange(pool) for _ in range(count))
        elif kind == "reuse":
            window = max(1, int(config.get("reuse_window_pages", min(pool, 64))))
            row = tuple(index % window for index in range(count))
        else:
            raise ValueError("block-table pattern must be contiguous, random, or reuse")
        rows.append(row)
    return tuple(rows)


def load_workload(value: Mapping[str, Any]) -> Workload:
    batch = _positive_int(value.get("batch_size"), "batch_size")
    dtype = str(value.get("dtype", "bf16")).lower()
    if dtype not in {"bf16", "fp16"}:
        raise ValueError("dtype must be bf16 or fp16")
    seqlen_q = _positive_int(value.get("seqlen_q", value.get("s_q", 1)), "seqlen_q")
    heads_q = _positive_int(value.get("num_heads_q", 128), "num_heads_q")
    heads_kv = _positive_int(value.get("num_heads_kv", 1), "num_heads_kv")
    if heads_q % heads_kv:
        raise ValueError("num_heads_kv must divide num_heads_q")
    qk = _positive_int(value.get("head_dim_qk", 576), "head_dim_qk")
    v = _positive_int(value.get("head_dim_v", 512), "head_dim_v")
    page_size = _positive_int(value.get("page_size", 64), "page_size")
    if (qk, v, page_size) != (576, 512, 64):
        raise ValueError("this SM90 path fixes head_dim_qk=576, head_dim_v=512, page_size=64")
    metadata_mode = str(value.get("metadata_mode", "generate"))
    if metadata_mode not in {"generate", "reuse"}:
        raise ValueError("metadata_mode must be generate or reuse")
    cache_mode = str(value.get("cache_mode", "l2_hot"))
    if cache_mode == "warm":
        cache_mode = "l2_hot"
    elif cache_mode == "cold":
        cache_mode = "hbm_stream"
    if cache_mode not in {"l2_hot", "hbm_stream"}:
        raise ValueError("cache_mode must be l2_hot or hbm_stream")
    pattern = str(value.get("block_table_pattern", "contiguous"))
    lengths = _lengths(value, batch)
    return Workload(
        case_id=str(value.get("case_id", "case-0")),
        dtype=dtype,
        batch_size=batch,
        seqlen_q=seqlen_q,
        num_heads_q=heads_q,
        num_heads_kv=heads_kv,
        head_dim_qk=qk,
        head_dim_v=v,
        page_size=page_size,
        seqlens_k=lengths,
        block_table=_block_table(value, lengths, page_size, pattern),
        causal=bool(value.get("causal", False)) and seqlen_q > 1,
        metadata_mode=metadata_mode,
        cache_mode=cache_mode,
        block_table_pattern=pattern,
    )


def load_kernel_resources(value: Mapping[str, Any] | None) -> KernelResources:
    if not value:
        return KernelResources()
    aliases = {
        "main_launch_bound": "main_min_blocks_per_sm",
        "combine_launch_bound": "combine_min_blocks_per_sm",
    }
    normalized = dict(value)
    for old, new in aliases.items():
        if old in normalized:
            if new in normalized and normalized[new] != normalized[old]:
                raise ValueError(f"conflicting kernel resource fields: {old} and {new}")
            normalized[new] = normalized[old]
    known = KernelResources.__dataclass_fields__
    payload = {key: normalized[key] for key in known if key in normalized}
    result = KernelResources(**payload)
    if result.sm_count <= 0 or result.sm_clock_mhz <= 0:
        raise ValueError("kernel resources require positive sm_count and sm_clock_mhz")
    return result
