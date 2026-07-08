#!/usr/bin/env python3
"""Visualize TMA 2D latency benchmark results: latency vs bytes-in-flight."""

import argparse
import pandas as pd
import matplotlib.pyplot as plt

COLOR = '#0072B2'
GPU_CLOCK_GHZ = 1.965


def main():
    parser = argparse.ArgumentParser(description='Plot TMA 2D latency results')
    parser.add_argument('input', help='Input CSV file')
    parser.add_argument('output', help='Output PNG file')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df['tile_kib'] = df['TileBytes'] / 1024
    df = df.sort_values('tile_kib')

    sorted_kib = sorted(df['tile_kib'].unique())
    kib_labels = [f'{int(k) if k == int(k) else k}' for k in sorted_kib]
    kib_to_x = {k: i for i, k in enumerate(sorted_kib)}
    df['x'] = df['tile_kib'].map(kib_to_x)

    med_ns  = df['LatencyCyclesMedian'] / GPU_CLOCK_GHZ
    min_ns  = df['LatencyCyclesMin']    / GPU_CLOCK_GHZ
    max_ns  = df['LatencyCyclesMax']    / GPU_CLOCK_GHZ
    err_lo  = (med_ns - min_ns).clip(lower=0)
    err_hi  = (max_ns - med_ns).clip(lower=0)

    fig, ax = plt.subplots(figsize=(9, 6))

    # Line through medians
    ax.plot(df['x'], med_ns, color=COLOR, linewidth=1.5, zorder=2)

    # Min–max whiskers with median marker
    ax.errorbar(df['x'], med_ns,
                yerr=[err_lo, err_hi],
                fmt='o', color=COLOR,
                markersize=7, linewidth=0,
                elinewidth=1, capsize=8, capthick=1,
                label='Median Latency (Error Bars: Min/Max)')

    # Data point labels: "WxH"
    for _, row in df.iterrows():
        label = f"{int(row['SmemWidth'])}x{int(row['SmemHeight'])}"
        ax.annotate(label,
                    xy=(row['x'], row['LatencyCyclesMax'] / GPU_CLOCK_GHZ),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8, color='#333333')

    ax.set_xticks(range(len(sorted_kib)))
    ax.set_xticklabels(kib_labels, rotation=45, ha='right')
    ax.set_xlabel('Bytes-in-Flight per SM (KiB)', fontsize=12)
    ax.set_ylabel('Latency (Nanoseconds)', fontsize=12)
    ax.set_title('TMA 2D Latency vs Bytes-in-Flight', fontsize=13)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)

    ax_cyc = ax.secondary_yaxis(
        'right',
        functions=(lambda ns: ns * GPU_CLOCK_GHZ, lambda cyc: cyc / GPU_CLOCK_GHZ)
    )
    ax_cyc.set_ylabel('Latency (Cycles)', fontsize=12)

    plt.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
