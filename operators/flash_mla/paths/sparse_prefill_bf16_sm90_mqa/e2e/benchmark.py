#!/usr/bin/env python3
"""Measure FlashMLA sparse BF16 prefill forward e2e with CUDA events."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

TARGET_TESTS = Path(__file__).resolve().parents[3] / "target" / "tests"
sys.path.insert(0, str(TARGET_TESTS))
import lib  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s-q", type=int, default=4096)
    ap.add_argument("--s-kv", type=int, default=8192)
    ap.add_argument("--topk", type=int, default=2048)
    ap.add_argument("--h-q", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("sparse_prefill_bf16_sm90 requires SM90")

    param = lib.TestParam(
        args.s_q, args.s_kv, args.topk, h_q=args.h_q, h_kv=1,
        d_qk=576, d_v=512, seed=0, check_correctness=False, num_runs=0,
    )
    testcase = lib.generate_testcase(param)

    @torch.inference_mode()
    def run():
        return lib.run_flash_mla_sparse_fwd(param, testcase, False)

    run()
    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        run()
    stop.record(); stop.synchronize()
    latency_ms = start.elapsed_time(stop) / args.iters
    stats = lib.count_flop_and_mem_vol(param, testcase)
    seconds = latency_ms / 1e3
    print(json.dumps({
        "s_q": args.s_q, "s_kv": args.s_kv, "topk": args.topk, "h_q": args.h_q,
        "latency_ms": latency_ms,
        "tflops": stats.fwd_flop / seconds / 1e12,
        "gbps": stats.fwd_mem_vol / seconds / 1e9,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
