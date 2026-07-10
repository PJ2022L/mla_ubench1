#!/usr/bin/env python3
"""Compare SS vs TS actual throughput at M=128.

Two panels: 1SM (CTA_GROUP=1) and 2SM (CTA_GROUP=2).
SS = dashed lines, TS = solid lines. Color = format.
"""

import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

FORMAT_K = {
    "BF16": 16, "E4M3": 32, "S8": 32, "F4": 64, "MXF8": 32, "MXF4": 64,
}

def hw_peak(fmt, cta_group):
    k = FORMAT_K[fmt]
    return 512 * k if cta_group == 1 else 1024 * k

# Read max-depth results
with open("tput_results_max.csv") as f:
    rows = list(csv.DictReader(f))

# Collect FLOPsPerCycle for each (Format, ABLayout, CTAGroup, N) at M=128
data = {}
for row in rows:
    fmt = row["Format"]
    layout = row["ABLayout"]
    cta = int(row["CTAGroup"])
    m, n = int(row["M"]), int(row["N"])
    if m != 128:
        continue
    flops = float(row["FLOPsPerCycle"])
    key = (fmt, layout, cta, n)
    if key not in data or flops > data[key]:
        data[key] = flops

formats = ["BF16", "E4M3", "S8", "F4", "MXF8", "MXF4"]

# Styles
fmt_colors = {
    "BF16": "#2D6A8F", "E4M3": "#CC5555", "S8": "#44AA66",
    "F4": "#DD8833", "MXF8": "#8855BB", "MXF4": "#CC6699",
}
fmt_markers = {
    "BF16": "o", "E4M3": "s", "S8": "D", "F4": "^", "MXF8": "v", "MXF4": "P",
}
layout_linestyles = {"SS": "--", "TS": "-"}

fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=False)

for ax, cta in zip(axes, [1, 2]):
    cta_label = "1SM" if cta == 1 else "2SM"

    # Build series for this CTA group
    series = defaultdict(list)
    for (fmt, layout, c, n), flops in data.items():
        if c != cta:
            continue
        series[(fmt, layout)].append((n, flops))
    for key in series:
        series[key].sort()

    # Collect all N values
    all_n = sorted(set(n for (f, l) in series for n, _ in series[(f, l)]))
    if not all_n:
        continue
    n_to_x = {n: i for i, n in enumerate(all_n)}
    x_pos = np.arange(len(all_n))

    # Draw peak throughput lines per format
    peak_drawn = set()
    for fmt in formats:
        peak = hw_peak(fmt, cta)
        if peak not in peak_drawn:
            ax.axhline(y=peak, color=fmt_colors[fmt], linewidth=1, linestyle=":", alpha=0.5)
            ax.text(x_pos[0] - 0.15, peak, f"{peak:,}", fontsize=9, color=fmt_colors[fmt],
                    va="bottom", ha="right", fontweight="bold")
            peak_drawn.add(peak)

    for fmt in formats:
        for layout in ["SS", "TS"]:
            if (fmt, layout) not in series:
                continue
            pts = series[(fmt, layout)]
            xs = [n_to_x[n] for n, _ in pts]
            ys = [flops for _, flops in pts]
            ax.plot(xs, ys, marker=fmt_markers[fmt], linestyle=layout_linestyles[layout],
                    color=fmt_colors[fmt], markersize=6, linewidth=1.8, alpha=0.85)

    ax.set_title(f"{cta_label} (CTA_GROUP={cta})", fontsize=12)
    ax.set_xlabel("N", fontsize=11)
    ax.set_ylabel("FLOPs / Cycle", fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(n) for n in all_n], fontsize=10)
    ax.set_xlim(x_pos[0] - 0.2, x_pos[-1] + 0.2)
    ax.grid(alpha=0.3)

    # Legend
    panel_handles = []
    for fmt in formats:
        if not any((fmt, l) in series for l in ["SS", "TS"]):
            continue
        h = mlines.Line2D([], [], color=fmt_colors[fmt], marker=fmt_markers[fmt],
                           linestyle="-", markersize=5, linewidth=1.5, label=fmt)
        panel_handles.append(h)

    panel_handles.append(mlines.Line2D([], [], color="none", label=""))
    panel_handles.append(mlines.Line2D([], [], color="#666666", linestyle="-",
                                        linewidth=1.8, label="TS"))
    panel_handles.append(mlines.Line2D([], [], color="#666666", linestyle="--",
                                        linewidth=1.8, label="SS"))

    ax.legend(handles=panel_handles, loc="upper right" if cta == 1 else "right", fontsize=9)

fig.suptitle("SS vs TS Throughput (M=128)", fontsize=13)
fig.tight_layout()
fig.savefig("ss_vs_ts.png", dpi=150, bbox_inches="tight")
print("Saved ss_vs_ts.png")
plt.show()
