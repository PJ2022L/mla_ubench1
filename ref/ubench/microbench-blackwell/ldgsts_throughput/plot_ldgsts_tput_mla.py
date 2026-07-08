#!/usr/bin/env python3
"""
Plot MLA LDGSTS throughput: two subplots.
Left: lines grouped by num stages, right: lines grouped by threads per CTA.
X-axis: Bytes-in-Flight per SM (KiB)
Y-axis: Throughput (TB/s)
"""

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Plot MLA LDGSTS throughput')
    parser.add_argument('input', help='Input CSV file')
    parser.add_argument('output', help='Output PNG file')
    args = parser.parse_args()

    # ---------- Load data ----------
    df = pd.read_csv(args.input)

    # ---------- Derived columns ----------
    df["BytesInFlight_KiB"] = df["CTAsPerSM"] * df["NumStages"] * df["ThreadsPerBlock"] * df["LoadBytes"] / 1024.0
    df["Throughput_TBs"] = df["DRAMBandwidthBps"] / 1e12

    # ---------- Figure setup ----------
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    colors_stages = {4: "#1f77b4", 8: "#ff7f0e", 12: "#2ca02c", 16: "#d62728"}
    colors_threads = {64: "#1f77b4", 128: "#ff7f0e", 256: "#2ca02c"}

    # ---------- Left: lines by num stages ----------
    for stages, color in colors_stages.items():
        subset = df[df["NumStages"] == stages].sort_values("BytesInFlight_KiB")
        if subset.empty:
            continue

        # Max envelope per bytes-in-flight
        envelope = (
            subset.groupby("BytesInFlight_KiB", as_index=False)["Throughput_TBs"]
            .max()
            .sort_values("BytesInFlight_KiB")
        )

        ax_left.scatter(subset["BytesInFlight_KiB"], subset["Throughput_TBs"],
                        color=color, alpha=0.35, s=30, edgecolors="none", zorder=2)
        ax_left.plot(envelope["BytesInFlight_KiB"], envelope["Throughput_TBs"],
                     color=color, linewidth=2.0, marker="o", markersize=5,
                     label=f"{stages} stages", zorder=3)

    ax_left.set_title("By Pipeline Stages", fontsize=13)
    ax_left.set_xlabel("Bytes-in-Flight per SM (KiB)", fontsize=11)
    ax_left.set_ylabel("Throughput (TB/s)", fontsize=11)
    ax_left.legend(fontsize=10, framealpha=0.9)

    # ---------- Right: lines by threads per CTA ----------
    for threads, color in colors_threads.items():
        subset = df[df["ThreadsPerBlock"] == threads].sort_values("BytesInFlight_KiB")
        if subset.empty:
            continue

        envelope = (
            subset.groupby("BytesInFlight_KiB", as_index=False)["Throughput_TBs"]
            .max()
            .sort_values("BytesInFlight_KiB")
        )

        ax_right.scatter(subset["BytesInFlight_KiB"], subset["Throughput_TBs"],
                         color=color, alpha=0.35, s=30, edgecolors="none", zorder=2)
        ax_right.plot(envelope["BytesInFlight_KiB"], envelope["Throughput_TBs"],
                      color=color, linewidth=2.0, marker="o", markersize=5,
                      label=f"{threads // 32} warps ({threads} threads)", zorder=3)

    ax_right.set_title("By Load Warps", fontsize=13)
    ax_right.set_xlabel("Bytes-in-Flight per SM (KiB)", fontsize=11)
    ax_right.set_ylabel("Throughput (TB/s)", fontsize=11)
    ax_right.legend(fontsize=10, framealpha=0.9)

    # ---------- Shared formatting ----------
    all_bif = sorted(df["BytesInFlight_KiB"].unique())
    for ax in (ax_left, ax_right):
        ax.set_xticks(all_bif)
        ax.set_xticklabels([f"{x:.0f}" if x == int(x) else f"{x:.1f}" for x in all_bif], fontsize=8, rotation=45)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.tick_params(axis='y', which='minor', length=4)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)

    # ---------- Annotate MLA kernel config: 1 CTA, 12 stages, 64 threads (2 warps), float4 ----------
    mla_row = df[(df["NumStages"] == 12) & (df["ThreadsPerBlock"] == 64)].iloc[0]
    mla_x = mla_row["BytesInFlight_KiB"]
    mla_y = mla_row["Throughput_TBs"]

    for ax in (ax_left, ax_right):
        ax.annotate(
            "MLA config",
            xy=(mla_x, mla_y),
            xytext=(20, 15), textcoords="offset points",
            fontsize=9, fontweight="bold", color="#333333",
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
            zorder=5,
        )

    fig.suptitle("MLA LDGSTS Throughput (1 CTA/SM, float4)", fontsize=14)
    fig.tight_layout()

    # ---------- Save ----------
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to: {args.output}")


if __name__ == '__main__':
    main()
