#!/usr/bin/env python3
"""Compare latency across benchmark result CSVs, plotted against bytes-in-flight."""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
GPU_CLOCK_GHZ = 1.965


def compute_bif(df):
    if 'BytesInFlightPerSM' in df.columns:
        return df['BytesInFlightPerSM']
    elif 'TileBytes' in df.columns:
        return df['CTAsPerSM'] * df['TileBytes']
    elif 'LoadBytes' in df.columns and 'ThreadsPerBlock' in df.columns:
        return df['CTAsPerSM'] * df['ThreadsPerBlock'] * df['LoadBytes']
    else:
        raise ValueError("Cannot compute BytesInFlightPerSM from available columns")


def load_data(path):
    df = pd.read_csv(path)
    df['bif_kib'] = compute_bif(df) / 1024
    return df


def main():
    parser = argparse.ArgumentParser(description='Compare latency across benchmark CSVs')
    parser.add_argument('inputs', nargs='+', help='Input CSV files')
    parser.add_argument('--labels', nargs='+', help='Labels for each CSV (default: filename stem)')
    parser.add_argument('--output', required=True, help='Output PNG file')
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.inputs):
        parser.error('Number of --labels must match number of input files')

    labels = args.labels or [Path(p).stem for p in args.inputs]
    datasets = [(label, load_data(path)) for label, path in zip(labels, args.inputs)]

    all_bif = sorted(set(bif for _, df in datasets for bif in df['bif_kib'].unique()))
    bif_to_x = {b: i for i, b in enumerate(all_bif)}
    bif_labels = [f'{int(b) if b == int(b) else b}' for b in all_bif]

    y_col = 'LatencyCyclesMedian'
    title = 'Latency vs Bytes-in-Flight'

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (label, df) in enumerate(datasets):
        color = COLORS[i % len(COLORS)]
        df['x'] = df['bif_kib'].map(bif_to_x)
        df['lat_ns'] = df[y_col] / GPU_CLOCK_GHZ

        ax.scatter(df['x'], df['lat_ns'],
                   color=color, alpha=0.3, s=30, zorder=2)

        env = df.groupby('x')['lat_ns'].median().reset_index()
        env = env.sort_values('x')
        ax.plot(env['x'], env['lat_ns'],
                color=color, linewidth=2, marker='o', markersize=5, label=label, zorder=3)

    ax.set_xticks(range(len(all_bif)))
    ax.set_xticklabels(bif_labels, rotation=45, ha='right')
    ax.set_xlabel('Bytes-in-Flight per SM (KiB)', fontsize=12)
    ax.set_ylabel(f'Latency (ns)  [clock = {int(GPU_CLOCK_GHZ * 1000)} MHz]', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax2 = ax.secondary_yaxis(
        'right',
        functions=(lambda ns: ns * GPU_CLOCK_GHZ, lambda cyc: cyc / GPU_CLOCK_GHZ),
    )
    ax2.set_ylabel('Latency (SM cycles)', fontsize=12)

    plt.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
