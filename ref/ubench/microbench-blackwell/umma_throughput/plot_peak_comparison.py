#!/usr/bin/env python3
"""Plot max achieved FLOPs/cycle vs theoretical peak for each (Format, CTAGroup).

Generates a bullet chart showing achieved throughput overlaid on theoretical peak.
"""

import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
import numpy as np

# Theoretical peak: 512*K (1SM), 1024*K (2SM)
FORMAT_K = {
    "BF16": 16, "E4M3": 32, "S8": 32, "F4": 64, "MXF8": 32, "MXF4": 64,
}

def theoretical_peak(fmt, cta_group):
    k = FORMAT_K[fmt]
    return 512 * k if cta_group == 1 else 1024 * k

# Read CSV and find max FLOPsPerCycle per (Format, CTAGroup)
best = {}
with open("tput_results_max.csv") as f:
    for row in csv.DictReader(f):
        key = (row["Format"], int(row["CTAGroup"]))
        flops = float(row["FLOPsPerCycle"])
        if key not in best or flops > best[key]:
            best[key] = flops

# Order: all formats x [1SM, 2SM]
formats = ["BF16", "E4M3", "S8", "F4", "MXF8", "MXF4"]
groups = []
for fmt in formats:
    for cta in [1, 2]:
        if (fmt, cta) in best:
            groups.append((fmt, cta))

labels = [f"{fmt} {'1SM' if c == 1 else '2SM'}" for fmt, c in groups]
achieved = np.array([best[g] for g in groups])
peaks = np.array([theoretical_peak(fmt, c) for fmt, c in groups])
pcts = achieved / peaks * 100

y = np.arange(len(groups))

# Bullet chart
fig, ax = plt.subplots(figsize=(14, 7))

bar_h_bg = 0.5
bar_h_fg = 0.25

ax.barh(y, peaks, height=bar_h_bg, color="#D0D8E4", edgecolor="#9BB0C8", label="Theoretical Peak")
ax.barh(y, achieved, height=bar_h_fg, color="#2D6A8F", edgecolor="#1A4A6A", label="Max Achieved")

# Marker line at peak
for i in range(len(groups)):
    ax.plot([peaks[i], peaks[i]], [y[i] - bar_h_bg / 2, y[i] + bar_h_bg / 2],
            color="#AA3333", linewidth=2, zorder=5)

# Annotate with achieved value, peak value (red), and percentage
for i in range(len(groups)):
    offset = peaks[i] + peaks.max() * 0.01
    txt = f"{achieved[i]:.0f} / "
    txt_peak = f"{peaks[i]:.0f}"
    txt_pct = f"  ({pcts[i]:.1f}%)"
    ta1 = TextArea(txt, textprops=dict(fontsize=9, fontweight="bold", color="#2D6A8F"))
    ta2 = TextArea(txt_peak, textprops=dict(fontsize=9, fontweight="bold", color="#6688AA"))
    ta3 = TextArea(txt_pct, textprops=dict(fontsize=9, fontweight="bold", color="#AA3333"))
    packed = HPacker(children=[ta1, ta2, ta3], pad=0, sep=0, align="center")
    box = AnchoredOffsetbox(loc="center left", child=packed, frameon=False,
                            bbox_to_anchor=(offset, y[i]), bbox_transform=ax.transData)
    ax.add_artist(box)

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel("FLOPs / Cycle", fontsize=11)
ax.set_title("UMMA Throughput: Bullet Chart (Achieved vs Peak)", fontsize=13)
ax.set_xlim(0, peaks.max() * 1.28)
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.3)

peak_line = plt.Line2D([0], [0], color="#AA3333", linewidth=2, label="Peak marker")
ax.legend(handles=[
    mpatches.Patch(facecolor="#D0D8E4", edgecolor="#9BB0C8", label="Theoretical Peak"),
    mpatches.Patch(facecolor="#2D6A8F", edgecolor="#1A4A6A", label="Max Achieved"),
    peak_line,
], loc="upper right", fontsize=10)

fig.tight_layout()
fig.savefig("peak_comparison_bullet.png", dpi=150)
print("Saved peak_comparison_bullet.png")
plt.show()
