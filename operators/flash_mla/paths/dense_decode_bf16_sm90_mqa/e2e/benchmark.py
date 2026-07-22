#!/usr/bin/env python3
"""Remote-H800 held-out runner for the public dense FlashMLA decode API."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", default="dense-heldout")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--s-q", type=int, default=1)
    parser.add_argument("--s-k", type=int, default=4096)
    parser.add_argument(
        "--seqlens-k",
        help="comma-separated exact lengths; overrides --s-k and must match --batch",
    )
    parser.add_argument("--h-q", type=int, default=128)
    parser.add_argument("--h-kv", type=int, default=1)
    parser.add_argument("--d-qk", type=int, default=576)
    parser.add_argument("--d-v", type=int, default=512)
    parser.add_argument("--page", type=int, default=64)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument(
        "--metadata-mode", choices=("generate", "reuse"), default="reuse"
    )
    parser.add_argument(
        "--block-pattern", choices=("contiguous", "random", "reuse"),
        default="contiguous",
    )
    parser.add_argument(
        "--cache-mode", choices=("l2_hot", "hbm_stream"), default="l2_hot"
    )
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eviction-bytes", type=int, default=128 << 20)
    parser.add_argument("--check-correctness", action="store_true")
    parser.add_argument(
        "--validate-only", action="store_true",
        help="validate and print the case without importing CUDA dependencies",
    )
    return parser


def _parse_lengths(args: argparse.Namespace) -> list[int]:
    if args.seqlens_k:
        values = [int(value.strip()) for value in args.seqlens_k.split(",")]
        if len(values) != args.batch:
            raise ValueError("--seqlens-k must contain exactly --batch entries")
    else:
        values = [args.s_k] * args.batch
    if any(value < 0 for value in values):
        raise ValueError("K sequence lengths must be non-negative")
    return values


def validate_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch <= 0 or args.s_q <= 0 or args.h_q <= 0 or args.h_kv <= 0:
        raise ValueError("batch, s-q, h-q, and h-kv must be positive")
    if args.h_q % args.h_kv:
        raise ValueError("h-kv must divide h-q")
    if (args.d_qk, args.d_v, args.page) != (576, 512, 64):
        raise ValueError("this path fixes d-qk=576, d-v=512, page=64")
    if args.samples <= 0 or args.replays <= 0 or args.warmup < 0:
        raise ValueError("samples/replays must be positive and warmup non-negative")
    if args.eviction_bytes <= 0:
        raise ValueError("eviction-bytes must be positive")
    lengths = _parse_lengths(args)
    pages = [math.ceil(value / args.page) for value in lengths]
    return {
        "case_id": args.case_id,
        "dtype": args.dtype,
        "batch_size": args.batch,
        "seqlen_q": args.s_q,
        "seqlens_k": lengths,
        "num_heads_q": args.h_q,
        "num_heads_kv": args.h_kv,
        "head_dim_qk": args.d_qk,
        "head_dim_v": args.d_v,
        "page_size": args.page,
        "pages_per_request": pages,
        "causal_requested": args.causal,
        "causal_effective": args.causal and args.s_q > 1,
        "causal": args.causal and args.s_q > 1,
        "metadata_mode": args.metadata_mode,
        "block_table_pattern": args.block_pattern,
        "block_table_distribution": {
            "kind": args.block_pattern,
            "seed": args.seed,
            "pool_pages": max(1, sum(pages)),
            "reuse_window_pages": max(1, min(max(1, sum(pages)), 64)),
        },
        "cache_mode": args.cache_mode,
        "samples": args.samples,
        "replays_per_sample": 1 if args.cache_mode == "hbm_stream" else args.replays,
        "warmup": args.warmup,
        "seed": args.seed,
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _gpu_provenance(device: int) -> dict[str, Any]:
    command = [
        "nvidia-smi", f"--id={device}",
        "--query-gpu=name,uuid,clocks.sm,clocks.mem,pstate,power.draw,power.limit",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=True)
    fields = [field.strip() for field in completed.stdout.strip().split(",")]
    if len(fields) != 7:
        raise RuntimeError("unexpected nvidia-smi provenance output")
    return {
        "gpu_name": fields[0], "gpu_uuid": fields[1],
        "sm_clock_mhz": float(fields[2]), "memory_clock_mhz": float(fields[3]),
        "pstate": fields[4], "power_draw_w": float(fields[5]),
        "power_limit_w": float(fields[6]),
    }


def _make_block_table(torch: Any, case: dict[str, Any], device: str):
    batch = case["batch_size"]
    max_pages = max(1, max(case["pages_per_request"], default=0))
    total_pages = case["block_table_distribution"]["pool_pages"]
    table = torch.zeros(batch, max_pages, dtype=torch.int32)
    cursor = 0
    rng = random.Random(case["seed"])
    reuse_window = case["block_table_distribution"]["reuse_window_pages"]
    for request, count in enumerate(case["pages_per_request"]):
        if case["block_table_pattern"] == "contiguous":
            row = list(range(cursor, cursor + count))
            cursor += count
        elif case["block_table_pattern"] == "random":
            row = [rng.randrange(total_pages) for _ in range(count)]
        else:
            row = [index % reuse_window for index in range(count)]
        if row:
            table[request, :count] = torch.tensor(row, dtype=torch.int32)
    digest = hashlib.sha256(
        ",".join(str(int(value)) for value in table.flatten()).encode("ascii")
    ).hexdigest()
    return table.to(device), table, total_pages, digest


def _scheduler_validation(
    case: dict[str, Any], sm_count: int, actual_tensor: Any, split_prefix: list[int]
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[5]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from operators.flash_mla.paths.dense_decode_bf16_sm90_mqa.model.scheduler import (
        schedule_requests,
    )

    expected = schedule_requests(
        case["seqlens_k"],
        sm_count=sm_count,
        seqlen_q=case["seqlen_q"],
        num_heads_q=case["num_heads_q"],
        num_heads_kv=case["num_heads_kv"],
        page_size=case["page_size"],
    )
    actual_rows = [
        [int(value) for value in row[:7]]
        for row in actual_tensor.detach().cpu().tolist()
    ]
    expected_rows = [
        [
            item.begin_req_idx,
            item.end_req_idx,
            item.begin_block_idx,
            item.end_block_idx,
            item.begin_split_idx,
            int(item.is_first_req_splitted),
            int(item.is_last_req_splitted),
        ]
        for item in expected.metadata
    ]

    def digest(value: Any) -> str:
        encoded = json.dumps(value, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(encoded.encode("ascii")).hexdigest()

    mismatches = []
    for index in range(max(len(actual_rows), len(expected_rows))):
        actual = actual_rows[index] if index < len(actual_rows) else None
        wanted = expected_rows[index] if index < len(expected_rows) else None
        if actual != wanted:
            mismatches.append({"index": index, "expected": wanted, "actual": actual})
    expected_prefix = list(expected.num_splits_prefix)
    prefix_matches = split_prefix == expected_prefix
    passed = expected.source_defined and not mismatches and prefix_matches
    return {
        "source_defined": expected.source_defined,
        "undefined_reason": expected.undefined_reason,
        "official_case_eligible": passed,
        "passed": passed,
        "metadata_row_count": len(actual_rows),
        "expected_metadata_row_count": len(expected_rows),
        "metadata_sha256": digest(actual_rows),
        "expected_metadata_sha256": digest(expected_rows),
        "num_splits_matches": prefix_matches,
        "expected_num_splits_prefix": expected_prefix,
        "actual_num_splits_prefix": split_prefix,
        "metadata_matches": not mismatches,
        "mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:8],
    }


def _acceptance_gate(
    scheduler_validation: dict[str, Any], correctness: dict[str, Any]
) -> dict[str, Any]:
    reasons: list[str] = []
    if not scheduler_validation.get("source_defined", False):
        reasons.append("scheduler_source_undefined")
    if not scheduler_validation.get("metadata_matches", False):
        reasons.append("tile_scheduler_metadata_mismatch")
    if not scheduler_validation.get("num_splits_matches", False):
        reasons.append("num_splits_prefix_mismatch")
    if correctness.get("passed") is not True:
        reasons.append("correctness_not_checked" if not correctness.get("checked")
                       else "correctness_failed")
    return {
        "passed": not reasons,
        "exit_code": 0 if not reasons else 2,
        "rejection_reasons": reasons,
        "policy": "all scheduler and correctness gates are mandatory for acceptance",
    }


def _reference(torch: Any, q: Any, kv: Any, table_cpu: Any,
               case: dict[str, Any]) -> tuple[Any, Any]:
    batch = case["batch_size"]
    h_q = case["num_heads_q"]
    h_kv = case["num_heads_kv"]
    s_q = case["seqlen_q"]
    d_v = case["head_dim_v"]
    page = case["page_size"]
    outputs = []
    lses = []
    for batch_index in range(batch):
        length = case["seqlens_k"][batch_index]
        page_count = math.ceil(length / page)
        if page_count:
            ids = table_cpu[batch_index, :page_count].to(q.device, dtype=torch.long)
            values = kv.index_select(0, ids).reshape(-1, h_kv, case["head_dim_qk"])
            values = values[:length].transpose(0, 1).float()
        else:
            values = torch.empty(
                h_kv, 0, case["head_dim_qk"], device=q.device, dtype=torch.float32
            )
        query = q[batch_index].transpose(0, 1).float()
        if h_kv != h_q:
            values = values.repeat_interleave(h_q // h_kv, dim=0)
        scores = query @ values.transpose(-2, -1)
        scores *= case["head_dim_qk"] ** -0.5
        if case["causal_effective"] and length:
            mask = torch.ones(s_q, length, dtype=torch.bool, device=q.device)
            mask = mask.tril(diagonal=length - s_q)
            scores = scores.masked_fill(~mask, float("-inf"))
        lse = scores.logsumexp(dim=-1)
        if length:
            probability = torch.softmax(scores, dim=-1, dtype=torch.float32)
            output = probability @ values[..., :d_v]
        else:
            output = torch.zeros(h_q, s_q, d_v, device=q.device)
        empty = lse == float("-inf")
        output = output.masked_fill(empty.unsqueeze(-1), 0.0)
        lse = lse.masked_fill(empty, float("inf"))
        outputs.append(output.transpose(0, 1))
        lses.append(lse)
    return torch.stack(outputs), torch.stack(lses)


def _correctness(torch: Any, output: Any, lse: Any, q: Any, kv: Any,
                 table_cpu: Any, case: dict[str, Any]) -> dict[str, Any]:
    ref_output, ref_lse = _reference(torch, q, kv, table_cpu, case)
    out_f32 = output.float()
    lse_f32 = lse.float()
    atol = 8e-4 if case["dtype"] == "bf16" else 4e-4
    rtol = 2.01 / 128
    output_ok = torch.allclose(out_f32, ref_output, atol=atol, rtol=rtol, equal_nan=True)
    lse_ok = torch.allclose(lse_f32, ref_lse, atol=1e-6, rtol=8.01 / 65536,
                            equal_nan=True)
    finite_output = torch.isfinite(out_f32) & torch.isfinite(ref_output)
    finite_lse = torch.isfinite(lse_f32) & torch.isfinite(ref_lse)
    out_error = (out_f32 - ref_output).abs()
    lse_error = (lse_f32 - ref_lse).abs()
    return {
        "checked": True,
        "passed": bool(output_ok and lse_ok),
        "output_max_abs_error": float(out_error[finite_output].max().item())
            if finite_output.any() else 0.0,
        "lse_max_abs_error": float(lse_error[finite_lse].max().item())
            if finite_lse.any() else 0.0,
        "output_atol": atol,
        "output_rtol": rtol,
    }


def run_benchmark(args: argparse.Namespace, case: dict[str, Any]) -> int:
    import flash_mla
    import torch

    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    if (properties.major, properties.minor) != (9, 0) or "H800" not in properties.name:
        raise RuntimeError("formal measurements require the remote NVIDIA H800/SM90a")
    torch.manual_seed(case["seed"])
    torch.cuda.manual_seed_all(case["seed"])
    dtype = torch.bfloat16 if case["dtype"] == "bf16" else torch.float16
    device = "cuda"
    table, table_cpu, total_pages, table_sha256 = _make_block_table(
        torch, case, device
    )
    q = (torch.randn(
        case["batch_size"], case["seqlen_q"], case["num_heads_q"],
        case["head_dim_qk"], device=device, dtype=dtype
    ) / 10).clamp_(-1, 1)
    kv = (torch.randn(
        total_pages, case["page_size"], case["num_heads_kv"],
        case["head_dim_qk"], device=device, dtype=dtype
    ) / 10).clamp_(-1, 1)
    seqlens = torch.tensor(case["seqlens_k"], device=device, dtype=torch.int32)

    def call(metadata: Any):
        return flash_mla.flash_mla_with_kvcache(
            q, kv, table, seqlens, case["head_dim_v"], metadata, None,
            causal=case["causal_effective"],
        )

    metadata, _ = flash_mla.get_mla_metadata()
    if case["metadata_mode"] == "reuse":
        call(metadata)
        torch.cuda.synchronize()

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(capture_stream):
        with torch.cuda.graph(graph, stream=capture_stream):
            captured_output, captured_lse = call(metadata)
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()

    eviction = torch.empty(args.eviction_bytes // 4, device=device, dtype=torch.int32)

    def prepare_cache() -> None:
        if case["cache_mode"] == "l2_hot":
            graph.replay()
        else:
            eviction.add_(1)

    with torch.cuda.stream(capture_stream):
        for _ in range(case["warmup"]):
            prepare_cache()
            graph.replay()
    capture_stream.synchronize()

    latency_samples_us: list[float] = []
    replays = case["replays_per_sample"]
    for _ in range(case["samples"]):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(capture_stream):
            prepare_cache()
            start.record(capture_stream)
            for _ in range(replays):
                graph.replay()
            stop.record(capture_stream)
        stop.synchronize()
        latency_samples_us.append(start.elapsed_time(stop) * 1000.0 / replays)

    with torch.cuda.stream(capture_stream):
        graph.replay()
    capture_stream.synchronize()
    correctness = (
        _correctness(torch, captured_output, captured_lse, q, kv, table_cpu, case)
        if args.check_correctness else {"checked": False, "passed": None}
    )
    split_prefix = metadata.num_splits.detach().cpu().tolist()
    split_counts = [split_prefix[index + 1] - split_prefix[index]
                    for index in range(case["batch_size"])]
    scheduler_validation = _scheduler_validation(
        case,
        properties.multi_processor_count,
        metadata.tile_scheduler_metadata,
        split_prefix,
    )
    acceptance_gate = _acceptance_gate(scheduler_validation, correctness)
    provenance = _gpu_provenance(device_index)
    result = {
        "schema_version": 1,
        "case": case,
        "boundary": "cuda_graph_gpu_metadata_main_combine"
            if case["metadata_mode"] == "generate"
            else "cuda_graph_gpu_main_combine_metadata_reused",
        "latency_us": {
            "p10": _percentile(latency_samples_us, 0.10),
            "p50": statistics.median(latency_samples_us),
            "p90": _percentile(latency_samples_us, 0.90),
            "sample_count": len(latency_samples_us),
        },
        "num_splits_prefix": split_prefix,
        "split_counts": split_counts,
        "split_distribution": {
            str(value): split_counts.count(value) for value in sorted(set(split_counts))
        },
        "scheduler_validation": scheduler_validation,
        "acceptance_gate": acceptance_gate,
        "block_table_sha256": table_sha256,
        "correctness": correctness,
        "gpu": provenance | {"sm_count": properties.multi_processor_count},
    }
    print(json.dumps(result, sort_keys=True))
    return int(acceptance_gate["exit_code"])


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        case = validate_args(args)
        if args.validate_only:
            print(json.dumps({"validation": "ok", "case": case}, indent=2, sort_keys=True))
            return 0
        return run_benchmark(args, case)
    except (RuntimeError, ValueError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
