"""CPU replica of FlashMLA dense-decode metadata partitioning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    operation_counts: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "num_sm_parts": self.num_sm_parts,
            "payload_blocks": self.payload_blocks,
            "metadata": [asdict(item) for item in self.metadata],
            "num_splits_prefix": list(self.num_splits_prefix),
            "slices": [asdict(item) | {"pages": item.pages} for item in self.slices],
            "source_defined": self.source_defined,
            "undefined_reason": self.undefined_reason,
            "operation_counts": dict(self.operation_counts),
        }


def resolve_num_sm_parts(
    sm_count: int, seqlen_q: int, num_heads_q: int, num_heads_kv: int
) -> int:
    q_seq_per_hk = seqlen_q * (num_heads_q // num_heads_kv)
    return max(sm_count // num_heads_kv // math.ceil(q_seq_per_hk / 64), 1)


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
    lengths = tuple(int(item) for item in seqlens_k)
    if not lengths or any(item < 0 for item in lengths):
        raise ValueError("seqlens_k must be a non-empty sequence of non-negative values")
    parts = num_sm_parts or resolve_num_sm_parts(
        sm_count, seqlen_q, num_heads_q, num_heads_kv
    )
    blocks = [max(math.ceil(length / page_size), 1) for length in lengths]
    payload = math.ceil(sum(item + fixed_overhead_blocks for item in blocks) / parts)
    payload += fixed_overhead_blocks
    metadata: list[SchedulerMetadata] = []
    split_prefix = [0] * (len(lengths) + 1)
    request = block = split = cumulative = 0
    source_defined = True
    undefined_reason = None
    while_iterations = 0
    for _ in range(parts):
        if request >= len(lengths):
            source_defined = False
            undefined_reason = (
                "upstream get_mla_metadata_kernel would read first_block_idx_shared[batch_size] "
                "when num_sm_parts exceeds useful partitions; diagnostic no-op metadata was "
                "inserted, so this case is not source-defined"
            )
            metadata.append(SchedulerMetadata(len(lengths), 0, 0, False, len(lengths)-1, 0, False))
            continue
        begin_request, begin_block, begin_split = request, block, split
        first_split = block != 0
        remaining = payload
        while request < len(lengths):
            while_iterations += 1
            left = blocks[request] - block
            if remaining >= left + fixed_overhead_blocks:
                cumulative += split + 1
                split_prefix[request + 1] = cumulative
                remaining -= left + fixed_overhead_blocks
                request += 1
                block = split = 0
            else:
                available = remaining - fixed_overhead_blocks
                if available > 0:
                    block += available
                    split += 1
                break
        end_request = request if block else request - 1
        end_block = block if block else (0 if lengths[end_request] == 0 else blocks[end_request])
        last_split = lengths[end_request] != 0 and end_block != blocks[end_request]
        if begin_request == end_request:
            first_split = last_split = first_split or last_split
        metadata.append(SchedulerMetadata(
            begin_request, begin_block, begin_split, first_split,
            end_request, end_block, last_split,
        ))
    if request != len(lengths) or block or split:
        raise RuntimeError("scheduler failed to consume every request")
    slices: list[RequestSlice] = []
    for partition, item in enumerate(metadata):
        if item.begin_req_idx >= len(lengths):
            continue
        for request_idx in range(item.begin_req_idx, item.end_req_idx + 1):
            start = item.begin_block_idx if request_idx == item.begin_req_idx else 0
            end = item.end_block_idx if request_idx == item.end_req_idx else blocks[request_idx]
            if lengths[request_idx] == 0:
                end = 0
            split_idx = item.begin_split_idx if request_idx == item.begin_req_idx else 0
            count = split_prefix[request_idx + 1] - split_prefix[request_idx]
            slices.append(RequestSlice(request_idx, partition, start, end, split_idx, count == 1))
    operation_counts = {
        "global_load": len(lengths),
        "div": len(lengths) + 1,
        "iadd": 6 * len(lengths) + 5 * parts + 6 * while_iterations,
        "imad": 5 * len(lengths) + 4 * parts + 2 * while_iterations,
        "compare": 4 * len(lengths) + 4 * parts + 2 * while_iterations,
        "shuffle": 5,
        "shared_store": 5 * len(lengths) + 1,
        "shared_load": len(lengths) + 5 * parts + 2 * while_iterations,
        "warp_sync": 2,
        "metadata_store": parts,
        "split_store": len(lengths) + 1,
        "while_iterations": while_iterations,
    }
    return SchedulerResult(
        parts, payload, tuple(metadata), tuple(split_prefix), tuple(slices),
        source_defined, undefined_reason, operation_counts,
    )
