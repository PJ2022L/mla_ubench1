"""Input schemas for dense-decode prediction."""

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
    block_table_source: str
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

    @property
    def logical_k_pages(self) -> int:
        return sum(len(row) for row in self.block_table)

    @property
    def unique_k_pages(self) -> int:
        return len({page for row in self.block_table for page in row})

    @property
    def page_reuse_ratio(self) -> float:
        if self.logical_k_pages == 0:
            return 0.0
        return 1.0 - self.unique_k_pages / self.logical_k_pages

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
            "block_table_source": self.block_table_source,
            "causal": self.causal,
            "metadata_mode": self.metadata_mode,
            "cache_mode": self.cache_mode,
            "block_table_pattern": self.block_table_pattern,
        }


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _resolve_lengths(value: Mapping[str, Any], batch_size: int) -> tuple[int, ...]:
    explicit = value.get("seqlens_k")
    distribution = value.get("seqlens_k_distribution")
    if (explicit is None) == (distribution is None):
        raise ValueError("provide exactly one of seqlens_k or seqlens_k_distribution")
    if explicit is not None:
        if not isinstance(explicit, list) or len(explicit) != batch_size:
            raise ValueError("seqlens_k must be an array with batch_size entries")
        result = tuple(int(item) for item in explicit)
        if any(item < 0 for item in result):
            raise ValueError("seqlens_k entries must be non-negative")
        return result
    if not isinstance(distribution, dict):
        raise ValueError("seqlens_k_distribution must be an object")
    kind = distribution.get("kind", "fixed")
    seed = int(distribution.get("seed", 0))
    rng = random.Random(seed)
    if kind == "fixed":
        length = _positive_int(distribution.get("value"), "distribution.value")
        return (length,) * batch_size
    if kind == "uniform":
        low = _positive_int(distribution.get("low"), "distribution.low")
        high = _positive_int(distribution.get("high"), "distribution.high")
        if high < low:
            raise ValueError("distribution.high must be >= low")
        return tuple(rng.randint(low, high) for _ in range(batch_size))
    if kind == "choice":
        choices = distribution.get("values")
        if not isinstance(choices, list) or not choices:
            raise ValueError("choice distribution requires non-empty values")
        parsed = tuple(_positive_int(item, "distribution.values") for item in choices)
        return tuple(rng.choice(parsed) for _ in range(batch_size))
    raise ValueError("distribution.kind must be fixed, uniform, or choice")


def _resolve_block_table(
    value: Mapping[str, Any],
    seqlens_k: tuple[int, ...],
    page_size: int,
    pattern: str,
) -> tuple[tuple[tuple[int, ...], ...], str]:
    required = tuple(math.ceil(length / page_size) for length in seqlens_k)
    explicit = value.get("block_table")
    distribution = value.get("block_table_distribution")
    if explicit is not None and distribution is not None:
        raise ValueError("provide at most one of block_table or block_table_distribution")
    if explicit is not None:
        if not isinstance(explicit, list) or len(explicit) != len(seqlens_k):
            raise ValueError("block_table must have one row per batch entry")
        rows: list[tuple[int, ...]] = []
        for request, (row, pages) in enumerate(zip(explicit, required)):
            if not isinstance(row, list) or len(row) < pages:
                raise ValueError(
                    f"block_table[{request}] must contain at least {pages} page ids"
                )
            parsed = tuple(int(item) for item in row[:pages])
            if any(item < 0 for item in parsed):
                raise ValueError("block_table page ids must be non-negative")
            rows.append(parsed)
        return tuple(rows), "explicit"

    config = distribution if isinstance(distribution, dict) else {}
    kind = str(config.get("kind", pattern))
    if kind not in {"contiguous", "random", "reuse"}:
        raise ValueError("block_table_distribution.kind must be contiguous, random, or reuse")
    seed = int(config.get("seed", 0))
    rng = random.Random(seed)
    total_pages = sum(required)
    pool_pages = int(config.get("pool_pages", max(total_pages, 1)))
    if pool_pages <= 0:
        raise ValueError("block_table_distribution.pool_pages must be positive")
    rows = []
    cursor = 0
    for pages in required:
        if kind == "contiguous":
            row = tuple(range(cursor, cursor + pages))
            cursor += pages
        elif kind == "random":
            row = tuple(rng.randrange(pool_pages) for _ in range(pages))
        else:
            reuse_window = max(1, int(config.get("reuse_window_pages", min(pool_pages, 64))))
            row = tuple((page % reuse_window) for page in range(pages))
        rows.append(row)
    return tuple(rows), f"generated:{kind}"


def load_workload(value: Mapping[str, Any]) -> Workload:
    batch = _positive_int(value.get("batch_size"), "batch_size")
    dtype = str(value.get("dtype", "bf16")).lower()
    if dtype not in {"bf16", "fp16"}:
        raise ValueError("dtype must be bf16 or fp16")
    seqlen_q = _positive_int(value.get("seqlen_q", 1), "seqlen_q")
    heads_q = _positive_int(value.get("num_heads_q", 128), "num_heads_q")
    heads_kv = _positive_int(value.get("num_heads_kv", 1), "num_heads_kv")
    if heads_q % heads_kv:
        raise ValueError("num_heads_kv must divide num_heads_q")
    head_dim_qk = _positive_int(value.get("head_dim_qk", 576), "head_dim_qk")
    head_dim_v = _positive_int(value.get("head_dim_v", 512), "head_dim_v")
    page_size = _positive_int(value.get("page_size", 64), "page_size")
    if (head_dim_qk, head_dim_v, page_size) != (576, 512, 64):
        raise ValueError("SM90 dense decode fixes head_dim_qk=576, head_dim_v=512, page_size=64")
    metadata_mode = str(value.get("metadata_mode", "generate"))
    if metadata_mode not in {"generate", "reuse"}:
        raise ValueError("metadata_mode must be generate or reuse")
    cache_mode = str(value.get("cache_mode", "warm"))
    if cache_mode not in {"cold", "warm"}:
        raise ValueError("cache_mode must be cold or warm")
    block_pattern = str(value.get("block_table_pattern", "contiguous"))
    if block_pattern not in {"contiguous", "random", "reuse"}:
        raise ValueError("block_table_pattern must be contiguous, random, or reuse")
    seqlens_k = _resolve_lengths(value, batch)
    block_table, block_table_source = _resolve_block_table(
        value, seqlens_k, page_size, block_pattern
    )
    return Workload(
        case_id=str(value.get("case_id", "case-0")),
        dtype=dtype,
        batch_size=batch,
        seqlen_q=seqlen_q,
        num_heads_q=heads_q,
        num_heads_kv=heads_kv,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        page_size=page_size,
        seqlens_k=seqlens_k,
        block_table=block_table,
        block_table_source=block_table_source,
        causal=bool(value.get("causal", False)) and seqlen_q > 1,
        metadata_mode=metadata_mode,
        cache_mode=cache_mode,
        block_table_pattern=block_pattern,
    )
