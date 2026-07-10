#!/usr/bin/env python3
"""Measure steady-state dense FlashMLA decode (main + optional combine) with CUDA events."""

from __future__ import annotations

import argparse
import json
import math

import torch
import flash_mla


def main() -> None:
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
    args = p.parse_args()

    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("dense_decode_bf16_sm90_mqa requires SM90")
    if args.h_q % args.h_kv or args.s_k % args.page:
        raise ValueError("h_q must divide by h_kv and s_k must divide by page")

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    pages_per_request = args.s_k // args.page
    num_pages = args.batch * pages_per_request
    q = torch.randn(args.batch, args.s_q, args.h_q, args.d_qk, device=device, dtype=dtype) / 10
    kv = torch.randn(num_pages, args.page, args.h_kv, args.d_qk, device=device, dtype=dtype) / 10
    block_table = torch.arange(num_pages, device=device, dtype=torch.int32).view(args.batch, pages_per_request)
    seqlens = torch.full((args.batch,), args.s_k, device=device, dtype=torch.int32)
    sched_meta, _ = flash_mla.get_mla_metadata()

    @torch.inference_mode()
    def run():
        return flash_mla.flash_mla_with_kvcache(
            q, kv, block_table, seqlens, args.d_v, sched_meta, None, causal=args.causal
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
        "causal": args.causal, "latency_ms": latency_ms,
        "tflops": flops / seconds / 1e12, "gbps": bytes_moved / seconds / 1e9,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
