#!/usr/bin/env python3
"""Plot 1SM vs 2SM scaling: weak scaling and strong scaling subplots.

Weak scaling: same M per SM (1SM M=128 vs 2SM M=256)
Strong scaling: same total M (1SM M=128 vs 2SM M=128)

Bar chart grouped by N. Each format = overlapped bars: 1SM (narrow, inside) + 2SM (wide, outside).
Generates separate plots for SS and TS modes.
"""

import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Read max-depth results
with open("tput_results_max.csv") as f:
    rows = list(csv.DictReader(f))

# Collect best FLOPsPerCycle for each (Format, ABLayout, CTAGroup, M, N)
data = {}
for row in rows:
    fmt = row["Format"]
    layout = row["ABLayout"]
    cta = int(row["CTAGroup"])
    m, n = int(row["M"]), int(row["N"])
    flops = float(row["FLOPsPerCycle"])
    key = (fmt, layout, cta, m, n)
    if key not in data or flops > data[key]:
        data[key] = flops

formats = ["BF16", "E4M3", "S8", "F4", "MXF8", "MXF4"]
fmt_colors = {
    "BF16": "#2D6A8F", "E4M3": "#CC5555", "S8": "#44AA66",
    "F4": "#DD8833", "MXF8": "#8855BB", "MXF4": "#CC6699",
}
fmt_colors_light = {
    "BF16": "#8BBDE0", "E4M3": "#E8A0A0", "S8": "#90D4A8",
    "F4": "#F0C88A", "MXF8": "#C4A8DD", "MXF4": "#E8B0CC",
}


def plot_1sm_vs_2sm(layout, output_file):
    n_formats = len(formats)
    bar_width_outer = 0.14
    bar_width_inner = 0.07

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    panels = [
        ("Weak Scaling: 1SM m128 vs 2SM m256", 128, 256),
        ("Strong Scaling: 1SM m128 vs 2SM m128", 128, 128),
    ]

    for ax, (title, m1sm, m2sm) in zip(axes, panels):
        all_ns = sorted(set(
            n for (f, l, c, m, n) in data
            if l == layout and c == 1 and m == m1sm
        ) & set(
            n for (f, l, c, m, n) in data
            if l == layout and c == 2 and m == m2sm
        ))

        group_positions = np.arange(len(all_ns))
        offsets = np.linspace(-(n_formats - 1) / 2 * bar_width_outer,
                               (n_formats - 1) / 2 * bar_width_outer, n_formats)

        for i, fmt in enumerate(formats):
            xs_2sm, ys_2sm = [], []
            xs_1sm, ys_1sm = [], []
            for j, n in enumerate(all_ns):
                k1 = (fmt, layout, 1, m1sm, n)
                k2 = (fmt, layout, 2, m2sm, n)
                if k1 in data and k2 in data:
                    x = group_positions[j] + offsets[i]
                    xs_2sm.append(x)
                    ys_2sm.append(data[k2])
                    xs_1sm.append(x)
                    ys_1sm.append(data[k1])

            # 2SM: outer (wider) bar
            ax.bar(xs_2sm, ys_2sm, width=bar_width_outer,
                   color=fmt_colors[fmt], edgecolor="white", linewidth=0.3)
            # 1SM: inner (narrower) bar
            ax.bar(xs_1sm, ys_1sm, width=bar_width_inner,
                   color=fmt_colors_light[fmt], edgecolor="white", linewidth=0.3)
            # Speedup label on top of each 2SM bar
            for x, y1, y2 in zip(xs_2sm, ys_1sm, ys_2sm):
                speedup = y2 / y1
                ax.text(x, y2 + 300, f"{speedup:.2f}x", ha="center", va="bottom",
                        fontsize=6.5, fontweight="bold", color=fmt_colors[fmt])

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("N", fontsize=11)
        ax.set_xticks(group_positions)
        ax.set_xticklabels([str(n) for n in all_ns], fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Achieved Throughput (FLOPs/cycle)", fontsize=11)
    axes[1].set_ylabel("Achieved Throughput (FLOPs/cycle)", fontsize=11)

    # Legend
    handles = []
    for fmt in formats:
        handles.append(mpatches.Patch(facecolor=fmt_colors[fmt], label=fmt))
    handles.append(mpatches.Patch(facecolor="none", label=""))
    handles.append(mpatches.Patch(facecolor="#888888", label="2SM (outer)"))
    handles.append(mpatches.Patch(facecolor="#CCCCCC", label="1SM (inner)"))
    for ax in axes:
        ax.legend(handles=handles, loc="upper left", fontsize=9)

    fig.suptitle(f"1SM vs 2SM Scaling ({layout} Mode)", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(hspace=0.3)
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved {output_file}")
    plt.close(fig)


plot_1sm_vs_2sm("SS", "1sm_vs_2sm_ss.png")
plot_1sm_vs_2sm("TS", "1sm_vs_2sm_ts.png")
