#!/usr/bin/env python3
"""
Benchmark TMA 2D load latency across configurations.
Each CTA issues exactly 1 TMA load and waits serially — no pipeline.
The kernel uses clock64() to measure per-CTA round-trip latency (thread 0),
then reports min/median/max cycles across all CTAs via stdout.
Each launch reads a distinct DRAM region, and an explicit L2 flush is
performed between iterations to ensure cold-cache (DRAM) latency.
"""

import subprocess
import csv
import sys
import os
import argparse

# Default sweep ranges
DEFAULT_CTAS_PER_SM = [1]
DEFAULT_SMEM_WIDTH = 128
DEFAULT_SMEM_HEIGHTS = [1, 2, 4, 8, 16, 32, 64, 128, 256]

CSV_FIELDS = [
    'CTAsPerSM', 'SmemWidth', 'SmemHeight', 'TileBytes',
    'LatencyCyclesMedian', 'LatencyCyclesMin', 'LatencyCyclesMax',
    'LatencyNsMedian',
]

B200_CLOCK_GHZ = 1.965  # B200 SM clock ~1.965 GHz


def run_benchmark(ctas, smem_w, smem_h, verbose=False):
    """Compile and run benchmark for given configuration."""
    label = f"CTAs={ctas}, h={smem_h}"
    tile_bytes = smem_w * smem_h * 4  # float32

    clean_cmd = ["make", "clean"]
    build_cmd = [
        "make", "tma2d_lat.out",
        f"CTAS_PER_SM={ctas}",
        f"SMEM_WIDTH={smem_w}",
        f"SMEM_HEIGHT={smem_h}",
    ]
    run_cmd = ["./tma2d_lat.out"]

    try:
        subprocess.run(clean_cmd, capture_output=True, check=True)

        if verbose:
            print(f"Building {label}...", file=sys.stderr)
        result = subprocess.run(build_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Build failed for {label}:", file=sys.stderr)
            if verbose:
                print(result.stderr, file=sys.stderr)
            return None

        if verbose:
            print(f"Running {label}...", file=sys.stderr)
        result = subprocess.run(run_cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"Run failed for {label}:", file=sys.stderr)
            if verbose:
                print(result.stderr, file=sys.stderr)
            return None

        # Parse SUMMARY line from stdout: SUMMARY,<median>,<min>,<max>
        for line in result.stdout.strip().split('\n'):
            if line.startswith('SUMMARY,'):
                parts = line.split(',')
                lat_cyc_med = float(parts[1])
                lat_cyc_min = float(parts[2])
                lat_cyc_max = float(parts[3])
                break
        else:
            print(f"No SUMMARY line in output for {label}", file=sys.stderr)
            if verbose:
                print(f"stdout: {result.stdout[:500]}", file=sys.stderr)
                print(f"stderr: {result.stderr[:500]}", file=sys.stderr)
            return None

        lat_ns_med = lat_cyc_med / B200_CLOCK_GHZ

        return {
            'CTAsPerSM': ctas,
            'SmemWidth': smem_w,
            'SmemHeight': smem_h,
            'TileBytes': tile_bytes,
            'LatencyCyclesMedian': lat_cyc_med,
            'LatencyCyclesMin': lat_cyc_min,
            'LatencyCyclesMax': lat_cyc_max,
            'LatencyNsMedian': lat_ns_med,
        }

    except subprocess.TimeoutExpired:
        print(f"Timeout for {label}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error for {label}: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description='Benchmark TMA 2D load latency')
    parser.add_argument('-o', '--output', default='tma2d_lat_results.csv',
                        help='Output CSV file (default: tma2d_lat_results.csv)')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite CSV instead of appending')
    parser.add_argument('--ctas', nargs='+', type=int, default=None,
                        help=f'CTAs per SM (default: {DEFAULT_CTAS_PER_SM})')
    parser.add_argument('--heights', nargs='+', type=int, default=None,
                        help=f'SMEM heights (default: {DEFAULT_SMEM_HEIGHTS})')
    args = parser.parse_args()

    ctas_list = args.ctas or DEFAULT_CTAS_PER_SM
    heights_list = args.heights or DEFAULT_SMEM_HEIGHTS
    smem_w = DEFAULT_SMEM_WIDTH

    configs = [
        (ctas, smem_w, h)
        for ctas in ctas_list
        for h in heights_list
    ]

    total_runs = len(configs)
    print(f"Running {total_runs} configurations...", file=sys.stderr)

    # Setup CSV
    file_exists = os.path.exists(args.output) and not args.overwrite
    csv_file = open(args.output, 'a' if file_exists else 'w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if not file_exists:
        writer.writeheader()
        csv_file.flush()

    result_count = 0
    for run_idx, (ctas, w, h) in enumerate(configs, 1):
        result = run_benchmark(ctas, w, h, verbose=args.verbose)
        tile_kib = w * h * 4 / 1024
        if result:
            writer.writerow(result)
            csv_file.flush()
            result_count += 1
            lat_cyc = result['LatencyCyclesMedian']
            lat_ns = result['LatencyNsMedian']
            print(f"[{run_idx}/{total_runs}] CTAs={ctas}, h={h} ({tile_kib:.1f} KiB): "
                  f"{lat_cyc:.1f} cyc  {lat_ns:.1f} ns",
                  file=sys.stderr)
        else:
            print(f"[{run_idx}/{total_runs}] FAILED: CTAs={ctas}, h={h}",
                  file=sys.stderr)

    csv_file.close()

    if result_count > 0:
        print(f"\nSaved {result_count} results to {args.output}", file=sys.stderr)
    else:
        print("No successful results", file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
