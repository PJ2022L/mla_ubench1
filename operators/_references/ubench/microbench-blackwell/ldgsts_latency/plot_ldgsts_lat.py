#!/usr/bin/env python3
"""
Plot actual measured LDGSTS memory latency vs bytes-in-flight per SM.

Data comes from ldgsts_lat_results.csv which contains direct latency
measurements (min/median/max cycles) for different (CTAsPerSM, ThreadsPerBlock)
configurations.

BytesInFlight per SM = CTAsPerSM * ThreadsPerBlock * LoadBytes
"""

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# ---------- Constants ----------
GPU_CLOCK_MHZ = 1965
GPU_CLOCK_GHZ = GPU_CLOCK_MHZ / 1000.0  # 1.965 GHz


def main():
    parser = argparse.ArgumentParser(description='Plot LDGSTS measured latency')
    parser.add_argument('input', help='Input CSV file')
    parser.add_argument('output', help='Output PNG file')
    args = parser.parse_args()

    # ---------- Load data ----------
    df = pd.read_csv(args.input)

    # ---------- Derived columns ----------
    df["BytesInFlight"] = df["CTAsPerSM"] * df["ThreadsPerBlock"] * df["LoadBytes"]
    df["BytesInFlight_KiB"] = df["BytesInFlight"] / 1024.0
    df["LatencyNsMedian"] = df["LatencyCyclesMedian"] / GPU_CLOCK_GHZ

    # ---------- Categorical x-axis mapping ----------
    sorted_bif = sorted(df["BytesInFlight_KiB"].unique())
    bif_to_x = {b: i for i, b in enumerate(sorted_bif)}
    bif_labels = [f"{v:g}" for v in sorted_bif]
    df["x"] = df["BytesInFlight_KiB"].map(bif_to_x)

    # ---------- Min-median-latency envelope ----------
    envelope = (
        df.loc[df.groupby("BytesInFlight_KiB")["LatencyCyclesMedian"].idxmin()]
        .sort_values("BytesInFlight_KiB")
        .reset_index(drop=True)
    )
    envelope["x"] = envelope["BytesInFlight_KiB"].map(bif_to_x)
    envelope["LatencyNsMedian"] = envelope["LatencyCyclesMedian"] / GPU_CLOCK_GHZ

    # ---------- Figure setup ----------
    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Scatter all data points (median latency)
    ax.scatter(
        df["x"],
        df["LatencyNsMedian"],
        color="#aaaaaa",
        s=30,
        zorder=2,
        label="All configs",
        alpha=0.7,
    )

    # Min-median-latency envelope line
    ax.plot(
        envelope["x"],
        envelope["LatencyNsMedian"],
        color="#2077B4",
        linewidth=2.2,
        marker="o",
        markersize=6,
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=3,
        label="Min median envelope",
    )

    # ---------- Annotate selected envelope points ----------
    n_env = len(envelope)
    annotate_indices = [0, n_env - 1]
    # Add a midpoint and the start of the uptick
    mid = n_env // 2
    if mid not in annotate_indices:
        annotate_indices.append(mid)
    # Add 8 KiB point and the start of the uptick
    for i in range(1, n_env - 1):
        if envelope.iloc[i]["BytesInFlight_KiB"] == 8.0:
            annotate_indices.append(i)
    flat_min = envelope["LatencyNsMedian"].iloc[1:n_env-1].min()
    for i in range(1, n_env - 1):
        if envelope.iloc[i]["LatencyNsMedian"] > flat_min * 1.5:
            annotate_indices.append(i)
            break
    annotate_indices = sorted(set(annotate_indices))

    # Custom offsets per index to avoid overlap
    offsets = {
        annotate_indices[0]: (-10, 18),      # first: above-left
        annotate_indices[-1]: (-60, -25),     # last: below-left
    }

    for idx in annotate_indices:
        row = envelope.iloc[idx]
        cycles = row["LatencyCyclesMedian"]
        ns = row["LatencyNsMedian"]
        bif = row["BytesInFlight_KiB"]
        x_off, y_off = offsets.get(idx, (12, 14))

        ax.annotate(
            f"{ns:.0f} ns / {cycles:.0f} cyc",
            xy=(bif_to_x[bif], ns),
            xytext=(x_off, y_off),
            textcoords="offset points",
            fontsize=8,
            color="#333333",
            ha="left",
            va="bottom",
            arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9),
        )

    # ---------- Secondary y-axis for cycles ----------
    ax2 = ax.secondary_yaxis(
        "right",
        functions=(
            lambda ns: ns * GPU_CLOCK_GHZ,   # ns -> cycles
            lambda cyc: cyc / GPU_CLOCK_GHZ  # cycles -> ns
        ),
    )
    ax2.set_ylabel("Latency (SM cycles)", fontsize=12)
    ax2.tick_params(axis="y", labelsize=10)

    # ---------- Axes and labels ----------
    ax.set_xlabel("Bytes-in-Flight per SM (KiB)", fontsize=12)
    ax.set_ylabel(f"Latency (ns)  [clock = {GPU_CLOCK_MHZ} MHz]", fontsize=12)
    ax.set_title(
        "LDGSTS Memory Latency vs Bytes-in-Flight",
        fontsize=13,
        pad=12,
    )
    ax.tick_params(axis="both", labelsize=10)

    # Categorical x-axis ticks
    ax.set_xticks(range(len(sorted_bif)))
    ax.set_xticklabels(bif_labels, fontsize=8.5, rotation=45, ha="right")

    y_min = min(df["LatencyNsMedian"].min(), envelope["LatencyNsMedian"].min())
    y_max = max(df["LatencyNsMedian"].max(), envelope["LatencyNsMedian"].max())
    y_pad = (y_max - y_min) * 0.15
    ax.set_ylim(bottom=y_min - y_pad, top=y_max + y_pad)

    # Grid
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    # Legend
    ax.legend(fontsize=11, loc="upper left", framealpha=0.9)

    # ---------- Save ----------
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to: {args.output}")

    # ---------- Print envelope table ----------
    print("\nMin-median-latency envelope:")
    print(f"{'BiF (KiB)':>10}  {'CTAsPerSM':>9}  {'Threads':>7}  {'Lat (cyc)':>10}  {'Lat (ns)':>10}")
    for _, row in envelope.iterrows():
        print(
            f"{row['BytesInFlight_KiB']:10.3f}  "
            f"{row['CTAsPerSM']:9.0f}  "
            f"{row['ThreadsPerBlock']:7.0f}  "
            f"{row['LatencyCyclesMedian']:10.1f}  "
            f"{row['LatencyNsMedian']:10.1f}"
        )


if __name__ == '__main__':
    main()
