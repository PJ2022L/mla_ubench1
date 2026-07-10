#!/usr/bin/env python3
"""Measure V3.2 sparse FP8 decode e2e using FlashMLA's canonical test generator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import flash_mla

TARGET_TESTS = Path(__file__).resolve().parents[3] / "target" / "tests"
sys.path.insert(0, str(TARGET_TESTS))
import lib  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--s-q", type=int, default=2)
    ap.add_argument("--s-k", type=int, default=32768)
    ap.add_argument("--topk", type=int, default=2048)
    ap.add_argument("--h-q", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("sparse_decode_fp8_sm90 requires SM90")

    raw = lib.RawTestParamForDecode(
        args.batch, args.h_q, args.s_q, 1, args.s_k, True, args.topk,
        d_qk=576, d_v=512, check_correctness=False, num_runs=0, enable_attn_sink=False,
    )
    param = raw.to_test_param()
    param.seed = 0
    testcase = lib.generate_testcase_for_decode(param)
    sched_meta, _ = flash_mla.get_mla_metadata()

    @torch.inference_mode()
    def run():
        return lib.run_flash_mla_decode(param, testcase, sched_meta, None)

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
    stats = lib.count_flop_and_mem_vol_for_decode(param, testcase)
    seconds = latency_ms / 1e3
    print(json.dumps({
        "batch": args.batch, "s_q": args.s_q, "s_k": args.s_k, "topk": args.topk,
        "h_q": args.h_q, "latency_ms": latency_ms,
        "tflops": stats.flop / seconds / 1e12,
        "gbps": stats.mem_vol / seconds / 1e9,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
