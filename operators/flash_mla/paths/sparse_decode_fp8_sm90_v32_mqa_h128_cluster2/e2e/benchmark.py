#!/usr/bin/env python3
"""Measure V3.2 sparse FP8 decode e2e using FlashMLA's canonical test generator."""

from __future__ import annotations

import argparse
import json


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--s-q", type=int, default=2)
    ap.add_argument("--s-k", type=int, default=32768)
    ap.add_argument("--topk", type=int, default=2048)
    ap.add_argument("--h-q", type=int, default=128)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--check", action="store_true", help="run the PyTorch reference; use a small shape")
    ap.add_argument(
        "--validate-only",
        action="store_true",
        help="validate and print the resolved shape without importing CUDA dependencies",
    )
    return ap


def validate_args(args: argparse.Namespace) -> dict[str, int | bool]:
    if min(args.batch, args.s_q, args.s_k, args.topk, args.h_q, args.iters) < 1:
        raise ValueError("batch, sequence/head dimensions, topk, and iters must be positive")
    if args.device < 0:
        raise ValueError("device must be non-negative")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.h_q != 128:
        raise ValueError("this V3.2 path fixes h_q=128, h_kv=1, d_qk=576, d_v=512")
    return {
        "batch": args.batch,
        "s_q": args.s_q,
        "s_k": args.s_k,
        "topk": args.topk,
        "h_q": args.h_q,
        "h_kv": 1,
        "d_qk": 576,
        "d_v": 512,
        "device": args.device,
        "warmup": args.warmup,
        "iters": args.iters,
        "correctness_check": args.check,
    }


def run_benchmark(args: argparse.Namespace) -> None:
    import sys
    from pathlib import Path

    import flash_mla
    import torch

    target_tests = Path(__file__).resolve().parents[3] / "target" / "tests"
    sys.path.insert(0, str(target_tests))
    import lib
    import ref

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise RuntimeError("sparse_decode_fp8_sm90 requires SM90")

    raw = lib.RawTestParamForDecode(
        args.batch, args.h_q, args.s_q, 1, args.s_k, True, args.topk,
        d_qk=576, d_v=512, check_correctness=args.check, num_runs=0, enable_attn_sink=False,
    )
    param = raw.to_test_param()
    param.seed = 0
    testcase = lib.generate_testcase_for_decode(param)
    sched_meta, _ = flash_mla.get_mla_metadata()

    @torch.inference_mode()
    def run():
        return lib.run_flash_mla_decode(param, testcase, sched_meta, None)

    first_out = run()
    correctness = None
    if args.check:
        ref_out, ref_lse = ref.ref_sparse_attn_decode(param, testcase)
        correctness = all((
            lib.kk.check_is_allclose("out", first_out[0], ref_out, abs_tol=1e-3, rel_tol=2.01/128, cos_diff_tol=5e-6),
            lib.kk.check_is_allclose("lse", first_out[1], ref_lse, abs_tol=1e-6, rel_tol=8.01/65536),
        ))
        if not correctness:
            raise RuntimeError("sparse decode correctness check failed")
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
        "device": torch.cuda.get_device_name(device),
        "batch": args.batch, "s_q": args.s_q, "s_k": args.s_k, "topk": args.topk,
        "h_q": args.h_q, "latency_ms": latency_ms,
        "tflops": stats.flop / seconds / 1e12,
        "gbps": stats.mem_vol / seconds / 1e9,
        "correctness_checked": args.check,
        "correct": correctness,
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
    run_benchmark(args)


if __name__ == "__main__":
    main()
