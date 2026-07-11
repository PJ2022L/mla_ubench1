#!/usr/bin/env python3
"""Measure steady-state dense FlashMLA decode (main + optional combine) with CUDA events."""

from __future__ import annotations

import argparse
import json
import math


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--s-q", type=int, default=1)
    p.add_argument("--s-k", type=int, default=4096)
    p.add_argument("--h-q", type=int, default=128)
    p.add_argument("--h-kv", type=int, default=1)
    p.add_argument("--d-qk", type=int, default=576)
    p.add_argument("--d-v", type=int, default=512)
    p.add_argument("--page", type=int, default=64)
    p.add_argument("--causal", action="store_true")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="validate and print the resolved shape without importing CUDA dependencies",
    )
    return p


def validate_args(args: argparse.Namespace) -> dict[str, int | bool]:
    positive = (
        args.batch,
        args.s_q,
        args.s_k,
        args.h_q,
        args.h_kv,
        args.d_qk,
        args.d_v,
        args.page,
        args.iters,
    )
    if min(positive) < 1:
        raise ValueError("batch, shape dimensions, page, and iters must be positive")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.h_q % args.h_kv:
        raise ValueError("h_kv must divide h_q")
    if (args.h_kv, args.d_qk, args.d_v, args.page) != (1, 576, 512, 64):
        raise ValueError("this path fixes h_kv=1, d_qk=576, d_v=512, page=64")

    return {
        "batch": args.batch,
        "s_q": args.s_q,
        "s_k": args.s_k,
        "h_q": args.h_q,
        "h_kv": args.h_kv,
        "d_qk": args.d_qk,
        "d_v": args.d_v,
        "page": args.page,
        "pages_per_request": math.ceil(args.s_k / args.page),
        "causal_requested": args.causal,
        "causal_effective": args.causal and args.s_q > 1,
        "warmup": args.warmup,
        "iters": args.iters,
    }


def run_benchmark(args: argparse.Namespace, resolved: dict[str, int | bool]) -> None:
    import flash_mla
    import torch

    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("dense_decode_bf16_sm90_mqa requires an SM90a Hopper GPU")

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    pages_per_request = int(resolved["pages_per_request"])
    num_pages = args.batch * pages_per_request
    q = torch.randn(args.batch, args.s_q, args.h_q, args.d_qk, device=device, dtype=dtype) / 10
    kv = torch.randn(num_pages, args.page, args.h_kv, args.d_qk, device=device, dtype=dtype) / 10
    block_table = torch.arange(num_pages, device=device, dtype=torch.int32).view(args.batch, pages_per_request)
    seqlens = torch.full((args.batch,), args.s_k, device=device, dtype=torch.int32)
    sched_meta, _ = flash_mla.get_mla_metadata()

    effective_causal = bool(resolved["causal_effective"])

    @torch.inference_mode()
    def run():
        return flash_mla.flash_mla_with_kvcache(
            q, kv, block_table, seqlens, args.d_v, sched_meta, None, causal=effective_causal
        )

    run()  # initialize and cache scheduler metadata
    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        run()
    stop.record()
    stop.synchronize()
    latency_ms = start.elapsed_time(stop) / args.iters

    flops = args.batch * args.s_q * args.h_q * (2 * args.d_qk * args.s_k + 2 * args.s_k * args.d_v)
    bytes_moved = args.batch * (
        args.s_q * args.h_q * args.d_qk * 2
        + args.s_k * args.h_kv * args.d_qk * 2
        + args.s_q * args.h_q * args.d_v * 2
    )
    seconds = latency_ms / 1e3
    print(json.dumps({
        "batch": args.batch, "s_q": args.s_q, "s_k": args.s_k,
        "h_q": args.h_q, "h_kv": args.h_kv, "d_qk": args.d_qk, "d_v": args.d_v,
        "page": args.page, "warmup": args.warmup, "iters": args.iters,
        "pages_per_request": pages_per_request,
        "causal_requested": args.causal, "causal_effective": effective_causal,
        "latency_ms": latency_ms,
        "effective_tflops": flops / seconds / 1e12,
        "effective_gbps": bytes_moved / seconds / 1e9,
    }, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        resolved = validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.validate_only:
        print(json.dumps({"validation": "ok", **resolved}, indent=2, sort_keys=True))
        return
    run_benchmark(args, resolved)


if __name__ == "__main__":
    main()
