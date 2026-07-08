#!/usr/bin/env python3
"""
Benchmark unified tcgen05 MMA latency across configurations.
Sweeps formats, AB layouts (SS/TS), CTA groups (1SM/2SM), and dimensions.
Measures single-MMA end-to-end latency (median of 100 samples).
Outputs results to CSV.
"""

import subprocess
import csv
import sys
import os
import argparse

# MMA formats (MMA_FORMAT macro):
# Dense types (0-3): AB_LAYOUT can be SS or TS
# MX types (4-5): require M >= 128
MMA_FORMATS = {
    0: {'name': 'BF16', 'k': 16},
    1: {'name': 'E4M3', 'k': 32},
    2: {'name': 'S8', 'k': 32},
    3: {'name': 'F4', 'k': 64},
    4: {'name': 'MXF8', 'k': 32, 'mx': True},
    5: {'name': 'MXF4', 'k': 64, 'mx': True},
}

# (M, N) configs per format class and CTA group
CONFIGS_DENSE_1SM = [
    (64, 64), (64, 80), (64, 96), (64, 112), (64, 128), (64, 256),
    (128, 64), (128, 80), (128, 96), (128, 112), (128, 128), (128, 256),
]

CONFIGS_DENSE_2SM = [
    (128, 64), (128, 80), (128, 96), (128, 112), (128, 128), (128, 256),
    (256, 64), (256, 80), (256, 96), (256, 112), (256, 128), (256, 256),
]

CONFIGS_MX_1SM = [
    (128, 64), (128, 80), (128, 96), (128, 112), (128, 128), (128, 256),
]

CONFIGS_MX_2SM = [
    (128, 64), (128, 80), (128, 96), (128, 112), (128, 128), (128, 256),
    (256, 64), (256, 80), (256, 96), (256, 112), (256, 128), (256, 256),
]

CSV_FIELDS = [
    'Format', 'ABLayout', 'CTAGroup', 'M', 'N', 'K', 'MedianCycles',
]


def get_mn_configs(fmt_info, cta_group, n_sweep=None):
    """Get (M, N) configs for given format and CTA group."""
    is_mx = fmt_info.get('mx', False)

    if n_sweep:
        n_start, n_stop, n_step = n_sweep
        n_values = list(range(n_start, n_stop + 1, n_step))
        if is_mx:
            ms = [128] if cta_group == 1 else [128, 256]
        else:
            ms = [64, 128] if cta_group == 1 else [128, 256]
        return [(m, n) for m in ms for n in n_values]

    if is_mx:
        return CONFIGS_MX_1SM if cta_group == 1 else CONFIGS_MX_2SM
    return CONFIGS_DENSE_1SM if cta_group == 1 else CONFIGS_DENSE_2SM


def run_benchmark(m, n, mma_format, cta_group, ab_layout, verbose=False):
    """Compile and run latency benchmark for given configuration."""
    fmt_info = MMA_FORMATS[mma_format]
    k = fmt_info['k']
    fmt_name = fmt_info['name']
    mode = 'TS' if ab_layout == 1 else 'SS'
    label = f"{fmt_name} {mode} {cta_group}SM M={m}, N={n}"

    clean_cmd = ["make", "clean"]
    build_cmd = [
        "make", "umma_lat.out",
        f"MMA_FORMAT={mma_format}",
        f"MMA_M={m}",
        f"MMA_N={n}",
        f"MMA_K={k}",
        f"CTA_GROUP={cta_group}",
        f"AB_LAYOUT={ab_layout}",
    ]

    try:
        subprocess.run(clean_cmd, capture_output=True, check=True)

        if verbose:
            print(f"Building {label}...", file=sys.stderr)
        result = subprocess.run(build_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Build failed for {label}:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            return None

        if verbose:
            print(f"Running {label}...", file=sys.stderr)
        result = subprocess.run(["./umma_lat.out"], capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            print(f"Run failed for {label}:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            return None

        for line in result.stdout.split('\n'):
            if line.startswith('RESULT,'):
                parts = line.split(',')
                if len(parts) >= 4:
                    M, N, K = int(parts[1]), int(parts[2]), int(parts[3])
                    median_cycles = int(parts[4])
                    return {
                        'Format': fmt_name,
                        'ABLayout': mode,
                        'CTAGroup': cta_group,
                        'M': M,
                        'N': N,
                        'K': K,
                        'MedianCycles': median_cycles,
                    }

        print(f"Could not parse output for {label}", file=sys.stderr)
        print(f"stdout: {result.stdout}", file=sys.stderr)
        return None

    except subprocess.TimeoutExpired:
        print(f"Timeout for {label}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error for {label}: {e}", file=sys.stderr)
        return None


def main():
    fmt_help = ', '.join(f"{k}={v['name']}" for k, v in MMA_FORMATS.items())

    parser = argparse.ArgumentParser(description='Benchmark unified tcgen05 MMA latency')
    parser.add_argument('formats', nargs='+', type=int, metavar='FORMAT',
                        help=f'MMA format IDs: {fmt_help}')
    parser.add_argument('-o', '--output', default='lat_results.csv',
                        help='Output CSV file')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite CSV instead of appending')
    parser.add_argument('--cta-group', type=int, choices=[1, 2], default=None,
                        help='CTA group (1=1SM, 2=2SM). Default: both.')
    parser.add_argument('--mode', choices=['ss', 'ts', 'all'], default='ss',
                        help='AB layout mode: ss, ts, or all.')
    parser.add_argument('--n-sweep', type=str, default=None,
                        help='Sweep N: "start:stop:step" (e.g., "32:256:8")')
    args = parser.parse_args()

    # Validate formats
    for f in args.formats:
        if f not in MMA_FORMATS:
            print(f"Error: Invalid format {f}. Valid: {list(MMA_FORMATS.keys())}", file=sys.stderr)
            return 1

    selected_fmts = {k: v for k, v in MMA_FORMATS.items() if k in args.formats}

    # Parse n-sweep
    n_sweep = None
    if args.n_sweep:
        parts = args.n_sweep.split(':')
        if len(parts) != 3:
            print("Error: --n-sweep must be 'start:stop:step'", file=sys.stderr)
            return 1
        n_sweep = (int(parts[0]), int(parts[1]), int(parts[2]))

    # Build (ab_layout, cta_group) combinations
    ab_layouts = {'ss': [0], 'ts': [1], 'all': [0, 1]}[args.mode]
    cta_groups = [args.cta_group] if args.cta_group else [1, 2]

    sweep = []
    for ab_layout in ab_layouts:
        for cta_group in cta_groups:
            sweep.append((ab_layout, cta_group))

    # Count total runs
    total_runs = 0
    for fmt_info in selected_fmts.values():
        for ab_layout, cta_group in sweep:
            mn_configs = get_mn_configs(fmt_info, cta_group, n_sweep)
            total_runs += len(mn_configs)

    print(f"Running {total_runs} configurations...", file=sys.stderr)

    # Setup CSV
    file_exists = os.path.exists(args.output) and not args.overwrite
    csv_file = open(args.output, 'a' if file_exists else 'w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if not file_exists:
        writer.writeheader()
        csv_file.flush()

    result_count = 0
    for fmt_id, fmt_info in selected_fmts.items():
        fmt_name = fmt_info['name']
        k = fmt_info['k']

        for ab_layout, cta_group in sweep:
            mode = 'TS' if ab_layout == 1 else 'SS'
            mn_configs = get_mn_configs(fmt_info, cta_group, n_sweep)
            print(f"\n=== {fmt_name} {mode} {cta_group}SM (K={k}) ===", file=sys.stderr)

            for m, n in mn_configs:
                result = run_benchmark(m, n, mma_format=fmt_id,
                                       cta_group=cta_group, ab_layout=ab_layout,
                                       verbose=args.verbose)
                if result:
                    writer.writerow(result)
                    csv_file.flush()
                    result_count += 1
                    print(f"{fmt_name} {mode} {cta_group}SM M={m:3d}, N={n:3d}: "
                          f"{result['MedianCycles']} cycles", file=sys.stderr)

    csv_file.close()

    if result_count > 0:
        print(f"\nSaved {result_count} results to {args.output}", file=sys.stderr)
    else:
        print("No successful results", file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
