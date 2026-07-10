#!/usr/bin/env python3
"""
Plot LDG/STS throughput: scatter of all configs + max-envelope lines per LoadType.
X-axis: Bytes-in-Flight per SM (KiB)
Y-axis: Throughput (TB/s)
"""

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Plot LDGSTS throughput')
    parser.add_argument('input', help='Input CSV file')
    parser.add_argument('output', help='Output PNG file')
    args = parser.parse_args()

    # ---------- Load data ----------
    df = pd.read_csv(args.input)

    # ---------- Derived columns ----------
    df["BytesInFlight_KiB"] = df["CTAsPerSM"] * df["NumStages"] * df["ThreadsPerBlock"] * df["LoadBytes"] / 1024.0
    df["Throughput_TBs"] = df["DRAMBandwidthBps"] / 1e12

    # ---------- Color / label mapping ----------
    load_type_meta = {
        "float":  {"label": "float (4 B)",  "color": "#1f77b4"},  # blue
        "float2": {"label": "float2 (8 B)", "color": "#ff7f0e"},  # orange
        "float4": {"label": "float4 (16 B)","color": "#2ca02c"},  # green
    }

    # ---------- Figure setup ----------
    fig, ax = plt.subplots(figsize=(11, 6.5))

    for load_type, meta in load_type_meta.items():
        subset = df[df["LoadType"] == load_type].copy()
        color = meta["color"]
        label = meta["label"]

        # 1) Scatter all data points (semi-transparent so overlaps are visible)
        ax.scatter(
            subset["BytesInFlight_KiB"],
            subset["Throughput_TBs"],
            color=color,
            alpha=0.35,
            s=30,
            edgecolors="none",
            zorder=2,
        )

        # 2) Max-envelope line: group by bytes-in-flight, take max throughput
        envelope = (
            subset.groupby("BytesInFlight_KiB")["Throughput_TBs"]
            .max()
            .reset_index()
            .sort_values("BytesInFlight_KiB")
        )

        ax.plot(
            envelope["BytesInFlight_KiB"],
            envelope["Throughput_TBs"],
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=5,
            label=label,
            zorder=3,
        )

    # ---------- Axes and labels ----------
    ax.set_xlabel("Bytes-in-Flight per SM (KiB)", fontsize=12)
    ax.set_ylabel("Throughput (TB/s)", fontsize=12)
    ax.set_title("LDGSTS DRAM Throughput vs. Bytes-in-Flight per SM", fontsize=14)
    ax.tick_params(axis="both", labelsize=10)

    # Minor ticks on y-axis (one between each major tick)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.tick_params(axis='y', which='minor', length=4)

    # Theoretical max throughput
    ax.axhline(y=8, color="black", linestyle="--", linewidth=1.5, zorder=1,
               label="Theoretical Max (8 TB/s)")

    # Light grid for readability
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    # Legend
    ax.legend(fontsize=11, framealpha=0.9)

    # ---------- Annotate key data points ----------
    # For each load type, find the envelope and label specific points
    for load_type, meta in load_type_meta.items():
        subset = df[df["LoadType"] == load_type].copy()
        subset["Warps"] = subset["ThreadsPerBlock"] // 32
        envelope = (
            subset.groupby("BytesInFlight_KiB", as_index=False)
            .apply(lambda g: g.loc[g["Throughput_TBs"].idxmax()])
            .sort_values("BytesInFlight_KiB")
        )

        if load_type in ("float", "float2"):
            # Label the max throughput point
            row = envelope.loc[envelope["Throughput_TBs"].idxmax()]
            ax.annotate(
                f"{row['CTAsPerSM']:.0f} CTA, {row['NumStages']:.0f} stages, {row['Warps']:.0f} warps",
                xy=(row["BytesInFlight_KiB"], row["Throughput_TBs"]),
                xytext=(12, -18), textcoords="offset points",
                fontsize=8, color=meta["color"],
                arrowprops=dict(arrowstyle="->", color=meta["color"], lw=1.0),
            )
        elif load_type == "float4":
            # Label the point at 32 KiB bytes-in-flight
            row = envelope[envelope["BytesInFlight_KiB"] == 32.0].iloc[0]
            ax.annotate(
                f"{row['CTAsPerSM']:.0f} CTA, {row['NumStages']:.0f} stages, {row['Warps']:.0f} warps",
                xy=(row["BytesInFlight_KiB"], row["Throughput_TBs"]),
                xytext=(12, -18), textcoords="offset points",
                fontsize=8, color=meta["color"],
                arrowprops=dict(arrowstyle="->", color=meta["color"], lw=1.0),
            )

    # Custom x-axis ticks at bytes-in-flight cluster values
    ax.set_xticks([1, 4, 8, 12, 16, 24, 32, 48, 64])
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # ---------- Save ----------
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to: {args.output}")


if __name__ == '__main__':
    main()
