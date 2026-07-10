#!/usr/bin/env python3
"""Plot TMA 2D throughput: scatter of all configs + max-envelope line."""

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Plot TMA 2D benchmark results')
    parser.add_argument('input', help='Input CSV file')
    parser.add_argument('output', help='Output PNG file')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df['bif_kib'] = df['BytesInFlightPerSM'] / 1024
    df['throughput_gbps'] = df['DRAMBandwidthBps'] / 1e9

    sorted_bif = sorted(df['bif_kib'].unique())
    bif_to_x = {b: i for i, b in enumerate(sorted_bif)}
    bif_labels = [f'{int(b) if b == int(b) else b}' for b in sorted_bif]
    df['x'] = df['bif_kib'].map(bif_to_x)

    fig, ax = plt.subplots(figsize=(11, 6.5))

    ax.scatter(df['x'], df['throughput_gbps'],
               color='#0072B2', alpha=0.35, s=30, edgecolors='none', zorder=2)

    envelope = (
        df.groupby('x')['throughput_gbps']
        .max()
        .reset_index()
        .sort_values('x')
    )
    ax.plot(envelope['x'], envelope['throughput_gbps'],
            color='#0072B2', linewidth=2, marker='o', markersize=5,
            label='Max throughput', zorder=3)

    ax.set_xticks(range(len(sorted_bif)))
    ax.set_xticklabels(bif_labels, rotation=45, ha='right')
    ax.set_xlabel('Bytes-in-Flight per SM (KiB)', fontsize=12)
    ax.set_ylabel('DRAM Bandwidth (GB/s)', fontsize=12)
    ax.set_title('TMA 2D Throughput vs Bytes-in-Flight', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
