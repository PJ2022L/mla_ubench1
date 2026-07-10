#!/usr/bin/env python3
"""
Benchmark Blackwell HBM-resident elementwise throughput with CUDA events and ncu.

The default sweep mirrors the Trainium VectorEngine benchmark operations:
exp, tanh, rsqrt, add, multiply, and relu.  Each kernel loads fp32 values from
HBM, applies one operation, and stores fp32 values back to HBM.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_OPS = ["exp", "tanh", "rsqrt", "add", "multiply", "relu"]

NCU_REQUIRED_METRICS = [
    "dram__bytes_read.sum.per_second",
    "dram__bytes_write.sum.per_second",
    "sm__cycles_elapsed.avg",
]
NCU_OPTIONAL_METRICS = [
    "smsp__sass_thread_inst_executed_op_fp32_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_special_pred_on.sum",
]

CSV_FIELDS = [
    "Op",
    "Elements",
    "BytesRead",
    "BytesWritten",
    "EventTimeMs",
    "EffectiveGBps",
    "EffectiveGOps",
    "CTAsPerSM",
    "ThreadsPerBlock",
    "InnerRepeats",
    "WarmupIters",
    "TimedIters",
    "DRAMReadBps",
    "DRAMWriteBps",
    "SMCyclesAvg",
    "FP32ThreadInst",
    "SpecialThreadInst",
    "NCUMetricsMode",
]


def parse_ops(spec: str | None) -> list[str]:
    if spec is None:
        return DEFAULT_OPS
    ops = [part.strip() for part in spec.split(",") if part.strip()]
    invalid = [op for op in ops if op not in DEFAULT_OPS]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unsupported ops {invalid}; valid ops are {DEFAULT_OPS}"
        )
    return ops


def find_ncu() -> str | None:
    result = subprocess.run(["which", "ncu"], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    for path in ["/usr/local/cuda/bin/ncu", "/opt/nvidia/nsight-compute/ncu"]:
        if os.path.exists(path):
            return path
    return None


def parse_ncu_csv(output: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in output.splitlines():
        if not line.startswith('"'):
            continue
        parts = line.split('","')
        if len(parts) < 3:
            continue
        metric_name = parts[-3].strip('"')
        metric_value = parts[-1].strip('"').replace(",", "")
        if metric_name in NCU_REQUIRED_METRICS or metric_name in NCU_OPTIONAL_METRICS:
            try:
                metrics[metric_name] = float(metric_value)
            except ValueError:
                pass
    return metrics


def parse_result_line(output: str) -> dict[str, str]:
    for line in output.splitlines():
        if not line.startswith("RESULT "):
            continue
        fields: dict[str, str] = {}
        for part in line.split()[1:]:
            key, value = part.split("=", 1)
            fields[key] = value
        return fields
    raise ValueError(f"benchmark output did not contain a RESULT line:\n{output}")


def build(op: str, ctas_per_sm: int, threads: int, inner_repeats: int) -> None:
    subprocess.run(["make", "clean"], capture_output=True, check=True)
    result = subprocess.run(
        [
            "make",
            "elementwise_tput.out",
            f"OP={op}",
            f"CTAS_PER_SM={ctas_per_sm}",
            f"THREADS_PER_BLOCK={threads}",
            f"INNER_REPEATS={inner_repeats}",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"build failed for op={op}")


def run_once(
    *,
    op: str,
    ncu_path: str | None,
    elements: int,
    warmup: int,
    timed: int,
    use_ncu: bool,
    verbose: bool,
) -> tuple[dict[str, str], dict[str, float], str]:
    exe = "./elementwise_tput.out"
    app_cmd = [
        exe,
        "--elements",
        str(elements),
        "--warmup",
        str(warmup),
        "--iters",
        str(timed),
    ]

    if not use_ncu:
        result = subprocess.run(app_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            raise RuntimeError(f"run failed for op={op}")
        return parse_result_line(result.stdout), {}, "event-only"

    if ncu_path is None:
        raise RuntimeError("ncu not found; install Nsight Compute or use --no-ncu")

    metric_sets = [
        (NCU_REQUIRED_METRICS + NCU_OPTIONAL_METRICS, "full"),
        (NCU_REQUIRED_METRICS, "required-only"),
    ]
    last_stderr = ""
    for metrics, mode in metric_sets:
        ncu_cmd = [
            "sudo",
            ncu_path,
            "--clock-control",
            "none",
            "--csv",
            "--launch-skip",
            str(warmup),
            "--launch-count",
            "1",
            "--metrics",
            ",".join(metrics),
            *app_cmd,
        ]
        result = subprocess.run(ncu_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            parsed = parse_ncu_csv(result.stdout)
            return parse_result_line(result.stdout), parsed, mode
        last_stderr = result.stderr
        if verbose:
            print(result.stderr, file=sys.stderr)

    raise RuntimeError(f"ncu failed for op={op}: {last_stderr}")


def result_row(
    *,
    result: dict[str, str],
    metrics: dict[str, float],
    ctas_per_sm: int,
    threads: int,
    inner_repeats: int,
    warmup: int,
    timed: int,
    metrics_mode: str,
) -> dict[str, object]:
    return {
        "Op": result["op"],
        "Elements": int(result["elements"]),
        "BytesRead": int(result["bytes_read"]),
        "BytesWritten": int(result["bytes_written"]),
        "EventTimeMs": float(result["event_ms"]),
        "EffectiveGBps": float(result["effective_GBps"]),
        "EffectiveGOps": float(result["effective_GOps"]),
        "CTAsPerSM": ctas_per_sm,
        "ThreadsPerBlock": threads,
        "InnerRepeats": inner_repeats,
        "WarmupIters": warmup,
        "TimedIters": timed,
        "DRAMReadBps": metrics.get("dram__bytes_read.sum.per_second", ""),
        "DRAMWriteBps": metrics.get("dram__bytes_write.sum.per_second", ""),
        "SMCyclesAvg": metrics.get("sm__cycles_elapsed.avg", ""),
        "FP32ThreadInst": metrics.get(
            "smsp__sass_thread_inst_executed_op_fp32_pred_on.sum", ""
        ),
        "SpecialThreadInst": metrics.get(
            "smsp__sass_thread_inst_executed_op_special_pred_on.sum", ""
        ),
        "NCUMetricsMode": metrics_mode,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Blackwell elementwise throughput")
    parser.add_argument("-o", "--output", default="elementwise_tput_results.csv")
    parser.add_argument("--ops", type=parse_ops, default=DEFAULT_OPS)
    parser.add_argument("--elements", type=int, default=1 << 28)
    parser.add_argument("--ctas-per-sm", type=int, default=4)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--inner-repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-ncu", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    ncu_path = None if args.no_ncu else find_ncu()
    if args.no_ncu:
        print("Running without ncu; CSV will contain CUDA event metrics only.", file=sys.stderr)
    elif ncu_path is None:
        print("Error: ncu not found. Install Nsight Compute or use --no-ncu.", file=sys.stderr)
        return 1
    else:
        print(f"Using ncu: {ncu_path}", file=sys.stderr)

    output = Path(args.output)
    file_exists = output.exists() and not args.overwrite
    with output.open("a" if file_exists else "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()

        for index, op in enumerate(args.ops, 1):
            print(f"[{index}/{len(args.ops)}] op={op} build", file=sys.stderr)
            build(op, args.ctas_per_sm, args.threads, args.inner_repeats)
            result, metrics, metrics_mode = run_once(
                op=op,
                ncu_path=ncu_path,
                elements=args.elements,
                warmup=args.warmup,
                timed=args.iters,
                use_ncu=not args.no_ncu,
                verbose=args.verbose,
            )
            row = result_row(
                result=result,
                metrics=metrics,
                ctas_per_sm=args.ctas_per_sm,
                threads=args.threads,
                inner_repeats=args.inner_repeats,
                warmup=args.warmup,
                timed=args.iters,
                metrics_mode=metrics_mode,
            )
            writer.writerow(row)
            csv_file.flush()
            print(
                f"[{index}/{len(args.ops)}] op={op} "
                f"{row['EffectiveGBps']:.1f} GB/s {row['EffectiveGOps']:.1f} GOps/s",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
