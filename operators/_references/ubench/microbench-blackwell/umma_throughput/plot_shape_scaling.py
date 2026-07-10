#!/usr/bin/env python3
"""Plot % of hardware peak vs N, faceted by CTAGroup.

Each line = (Format, M). X-axis = N.
Two panels: 1SM and 2SM.
Color = format, linestyle = M.
Legend includes hardware peak FLOPs/cycle.
"""

import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

# Hardware theoretical peak: 512*K (1SM), 1024*K (2SM)
FORMAT_K = {
    "BF16": 16, "E4M3": 32, "S8": 32, "F4": 64, "MXF8": 32, "MXF4": 64,
}

def hw_peak(fmt, cta_group):
    k = FORMAT_K[fmt]
    return 512 * k if cta_group == 1 else 1024 * k

# Read max-depth results
with open("tput_results_max.csv") as f:
    rows = list(csv.DictReader(f))

# For each (Format, CTAGroup, M, N), take best ABLayout
best = {}
for row in rows:
    fmt = row["Format"]
    cta = int(row["CTAGroup"])
    m, n = int(row["M"]), int(row["N"])
    flops_per_cycle = float(row["FLOPsPerCycle"])
    peak = hw_peak(fmt, cta)
    pct = flops_per_cycle / peak * 100
    key = (fmt, cta, m, n)
    if key not in best or pct > best[key]:
        best[key] = pct

# Build series per (Format, CTAGroup, M) -> list of (N, pct)
formats = ["BF16", "E4M3", "S8", "F4", "MXF8", "MXF4"]
series = defaultdict(list)
for (fmt, cta, m, n), pct in best.items():
    series[(fmt, cta, m)].append((n, pct))

for key in series:
    series[key].sort()

# Format colors
fmt_colors = {
    "BF16": "#2D6A8F", "E4M3": "#CC5555", "S8": "#44AA66",
    "F4": "#DD8833", "MXF8": "#8855BB", "MXF4": "#CC6699",
}
fmt_markers = {
    "BF16": "o", "E4M3": "s", "S8": "D", "F4": "^", "MXF8": "v", "MXF4": "P",
}
# M line styles
m_linestyles = {64: ":", 128: "--", 256: "-"}

fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

for ax, cta in zip(axes, [1, 2]):
    cta_label = "1SM" if cta == 1 else "2SM"
    ax.set_title(f"{cta_label} (CTA_GROUP={cta})", fontsize=12)

    # Collect all N values for this panel
    all_n = sorted(set(n for (fmt, c, m) in series if c == cta
                       for n, _ in series[(fmt, c, m)]))
    n_to_x = {n: i for i, n in enumerate(all_n)}
    x_pos = np.arange(len(all_n))

    for fmt in formats:
        for m_val in sorted(set(m for (f, c, m) in series if f == fmt and c == cta)):
            if (fmt, cta, m_val) not in series:
                continue
            data = series[(fmt, cta, m_val)]
            xs = [n_to_x[n] for n, _ in data]
            ys = [pct for _, pct in data]
            ax.plot(xs, ys, marker=fmt_markers[fmt], linestyle=m_linestyles[m_val],
                    color=fmt_colors[fmt], markersize=7, linewidth=1.8, alpha=0.85)

    ax.axhline(y=100, color="#888888", linewidth=1, linestyle=":", alpha=0.5)
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 110, 10))
    ax.set_xlabel("N", fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(n) for n in all_n], fontsize=10)
    ax.grid(alpha=0.3)

axes[0].set_ylabel("Peak Hardware FLOPs (%)", fontsize=11)

# Per-panel legend
for ax, cta in zip(axes, [1, 2]):
    panel_handles = []
    for fmt in formats:
        h = mlines.Line2D([], [], color=fmt_colors[fmt], marker=fmt_markers[fmt],
                           linestyle="-", markersize=5, linewidth=1.5, label=fmt)
        panel_handles.append(h)

    # Spacer
    panel_handles.append(mlines.Line2D([], [], color="none", label=""))

    # Only M values that appear in this panel
    panel_m_values = sorted(set(m for (f, c, m) in series if c == cta))
    for m_val in panel_m_values:
        h = mlines.Line2D([], [], color="#666666", linestyle=m_linestyles[m_val],
                           linewidth=1.8, label=f"M={m_val}")
        panel_handles.append(h)

    ax.legend(handles=panel_handles, loc="lower right", fontsize=10)

fig.suptitle("MMA Efficiency vs N (by Format and M)", fontsize=13)

fig.tight_layout()
fig.savefig("shape_scaling.png", dpi=150, bbox_inches="tight")
print("Saved shape_scaling.png")
plt.show()
