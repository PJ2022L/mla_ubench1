"""CPU replica of FlashMLA's dense decoding scheduler metadata kernel."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class SchedulerMetadata:
    begin_req_idx: int
    begin_block_idx: int
    begin_split_idx: int
    is_first_req_splitted: bool
    end_req_idx: int
    end_block_idx: int
    is_last_req_splitted: bool


@dataclass(frozen=True)
class RequestSlice:
    request_idx: int
    partition_idx: int
    start_block_idx: int
    end_block_idx: int
    split_idx_within_request: int
    is_no_split: bool

    @property
    def pages(self) -> int:
        return max(0, self.end_block_idx - self.start_block_idx)


@dataclass(frozen=True)
class SchedulerResult:
    num_sm_parts: int
    payload_blocks: int
    metadata: tuple[SchedulerMetadata, ...]
    num_splits_prefix: tuple[int, ...]
    slices: tuple[RequestSlice, ...]
    source_defined: bool = True
    undefined_reason: str | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "num_sm_parts": self.num_sm_parts,
            "payload_blocks": self.payload_blocks,
            "metadata": [asdict(item) for item in self.metadata],
            "num_splits_prefix": list(self.num_splits_prefix),
            "slices": [asdict(item) | {"pages": item.pages} for item in self.slices],
            "source_defined": self.source_defined,
            "undefined_reason": self.undefined_reason,
        }


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def resolve_num_sm_parts(
    sm_count: int,
    seqlen_q: int,
    num_heads_q: int,
    num_heads_kv: int,
    block_size_m: int = 64,
) -> int:
    if min(sm_count, seqlen_q, num_heads_q, num_heads_kv, block_size_m) <= 0:
        raise ValueError("scheduler dimensions must be positive")
    if num_heads_q % num_heads_kv:
        raise ValueError("num_heads_kv must divide num_heads_q")
    q_seq_per_hk = seqlen_q * (num_heads_q // num_heads_kv)
    num_m_blocks = _ceil_div(q_seq_per_hk, block_size_m)
    return max(sm_count // num_heads_kv // num_m_blocks, 1)


def schedule_requests(
    seqlens_k: Iterable[int],
    *,
    sm_count: int,
    seqlen_q: int,
    num_heads_q: int,
    num_heads_kv: int,
    page_size: int = 64,
    fixed_overhead_blocks: int = 5,
    num_sm_parts: int | None = None,
) -> SchedulerResult:
    lengths = tuple(int(value) for value in seqlens_k)
    if not lengths:
        raise ValueError("seqlens_k must not be empty")
    if any(value < 0 for value in lengths):
        raise ValueError("seqlens_k values must be non-negative")
    if page_size <= 0 or fixed_overhead_blocks < 0:
        raise ValueError("page_size must be positive and overhead non-negative")
    parts = num_sm_parts or resolve_num_sm_parts(
        sm_count, seqlen_q, num_heads_q, num_heads_kv
    )
    if parts <= 0:
        raise ValueError("num_sm_parts must be positive")

    # The device scheduler represents a zero-length request with one temporary
    # block while constructing metadata. The main kernel later corrects the end.
    num_blocks = [max(_ceil_div(value, page_size), 1) for value in lengths]
    total_blocks = sum(value + fixed_overhead_blocks for value in num_blocks)
    payload = _ceil_div(total_blocks, parts) + fixed_overhead_blocks

    metadata: list[SchedulerMetadata] = []
    split_prefix = [0] * (len(lengths) + 1)
    now_req = 0
    now_block = 0
    now_split = 0
    cumulative_splits = 0
    source_defined = True
    undefined_reason = None

    for _partition in range(parts):
        if now_req >= len(lengths):
            source_defined = False
            undefined_reason = (
                "target get_mla_metadata_kernel would index request arrays after "
                "all requests were consumed; empty partitions are represented as no-op "
                "metadata only for diagnostic prediction"
            )
            metadata.append(
                SchedulerMetadata(
                    begin_req_idx=len(lengths),
                    begin_block_idx=0,
                    begin_split_idx=0,
                    is_first_req_splitted=False,
                    end_req_idx=len(lengths) - 1,
                    end_block_idx=0,
                    is_last_req_splitted=False,
                )
            )
            continue

        begin_req = now_req
        begin_block = now_block
        begin_split = now_split
        first_split = now_block != 0
        remaining = payload

        while now_req < len(lengths):
            request_blocks = num_blocks[now_req]
            remaining_blocks = request_blocks - now_block
            if remaining >= remaining_blocks + fixed_overhead_blocks:
                cumulative_splits += now_split + 1
                split_prefix[now_req + 1] = cumulative_splits
                remaining -= remaining_blocks + fixed_overhead_blocks
                now_req += 1
                now_block = 0
                now_split = 0
            else:
                available = remaining - fixed_overhead_blocks
                if available > 0:
                    now_block += available
                    now_split += 1
                break

        end_req = now_req if now_block > 0 else now_req - 1
        if now_block > 0:
            end_block = now_block
        else:
            end_block = 0 if lengths[end_req] == 0 else num_blocks[end_req]
        last_split = lengths[end_req] != 0 and end_block != num_blocks[end_req]
        if begin_req == end_req:
            first_split = last_split = first_split or last_split
        metadata.append(
            SchedulerMetadata(
                begin_req_idx=begin_req,
                begin_block_idx=begin_block,
                begin_split_idx=begin_split,
                is_first_req_splitted=first_split,
                end_req_idx=end_req,
                end_block_idx=end_block,
                is_last_req_splitted=last_split,
            )
        )

    if now_req != len(lengths) or now_block != 0 or now_split != 0:
        raise RuntimeError("scheduler failed to consume all requests")

    slices: list[RequestSlice] = []
    for partition_idx, item in enumerate(metadata):
        if item.begin_req_idx >= len(lengths):
            continue
        for request_idx in range(item.begin_req_idx, item.end_req_idx + 1):
            start = item.begin_block_idx if request_idx == item.begin_req_idx else 0
            end = item.end_block_idx if request_idx == item.end_req_idx else num_blocks[request_idx]
            if lengths[request_idx] == 0:
                end = 0
            split_idx = item.begin_split_idx if request_idx == item.begin_req_idx else 0
            split_count = split_prefix[request_idx + 1] - split_prefix[request_idx]
            slices.append(
                RequestSlice(
                    request_idx=request_idx,
                    partition_idx=partition_idx,
                    start_block_idx=start,
                    end_block_idx=end,
                    split_idx_within_request=split_idx,
                    is_no_split=split_count == 1,
                )
            )

    return SchedulerResult(
        num_sm_parts=parts,
        payload_blocks=payload,
        metadata=tuple(metadata),
        num_splits_prefix=tuple(split_prefix),
        slices=tuple(slices),
        source_defined=source_defined,
        undefined_reason=undefined_reason,
    )
