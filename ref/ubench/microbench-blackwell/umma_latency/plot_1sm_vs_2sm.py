#!/usr/bin/env python3
"""Plot 2SM D_sat is roughly 2x of 1SM D_sat, all formats.

Single plot: 1SM M=128 vs 2SM M=128 vs 2SM M=256.
Colors = config, Shapes = format.
"""

import csv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CSV_FILE = 'pipeline_depth.csv'

# Read data (TS only)
data = {}
with open(CSV_FILE) as f:
    for row in csv.DictReader(f):
        if row['ABLayout'] != 'TS':
            continue
        fmt = row['Format']
        cta = int(row['CTAGroup'])
        M = int(row['M'])
        N = int(row['N'])
        d_sat = int(row['SaturatedPipelineDepth'])
        data[(fmt, cta, M, N)] = d_sat

fmt_list = ['BF16', 'E4M3', 'S8', 'F4', 'MXF8', 'MXF4']
fmt_markers = {
    'BF16': 'o', 'E4M3': 's', 'S8': '^',
    'F4': 'D', 'MXF8': 'v', 'MXF4': 'P',
}
fmt_jitter = {fmt: (i - 2.5) * 0.06 for i, fmt in enumerate(fmt_list)}

configs = [
    (1, 128, 'tab:blue',   -0.15, '1SM (M=128)'),
    (2, 128, 'tab:orange',  0.0,  '2SM (M=128)'),
    (2, 256, 'tab:green',   0.15, '2SM (M=256)'),
]

n_values = [64, 80, 96, 112, 128, 256]
n_to_x = {n: i for i, n in enumerate(n_values)}

fig, ax = plt.subplots(figsize=(8, 5))

for fmt in fmt_list:
    for cta, M, color, offset, label in configs:
        xs, ds = [], []
        for N in n_values:
            k = (fmt, cta, M, N)
            if k in data:
                xs.append(n_to_x[N] + fmt_jitter[fmt] + offset)
                ds.append(data[k])
        ax.plot(xs, ds, ls='-', marker=fmt_markers[fmt],
                color=color, markersize=5, alpha=0.8)

ax.set_xlabel('N')
ax.set_ylabel('Saturated Pipeline Depth (D_sat)')
ax.set_xticks(range(len(n_values)))
ax.set_xticklabels([str(n) for n in n_values])
ax.set_yticks(range(1, 11))
ax.grid(True, alpha=0.3)

# Config legend (top)
config_handles = [Line2D([0], [0], color=c, lw=2, label=l)
                  for _, _, c, _, l in configs]
config_legend = fig.legend(handles=config_handles, fontsize=8, ncol=3,
                           loc='upper center', bbox_to_anchor=(0.5, 1.02))

# Format legend (below config legend)
fmt_handles = [Line2D([0], [0], color='gray', ls='', marker=fmt_markers[fmt],
                      markersize=6, label=fmt) for fmt in fmt_list]
fig.legend(handles=fmt_handles, fontsize=8, ncol=6,
           loc='upper center', bbox_to_anchor=(0.5, 0.97))
fig.add_artist(config_legend)

fig.suptitle('2SM Requires ~2x Pipeline Depth (TS mode, all formats)', y=1.12, fontsize=13)

plt.tight_layout()
plt.savefig('1sm_vs_2sm.png', dpi=150, bbox_inches='tight')
print('Saved 1sm_vs_2sm.png')
