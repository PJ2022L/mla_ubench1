#!/usr/bin/env python3
"""Plot saturated pipeline depth vs N: D_sat decreases with N, all formats.

Two subplots: 1SM (left) and 2SM (right).
Colors = M, Shapes = format.
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
        config = (fmt, cta, M)
        if config not in data:
            data[config] = ([], [])
        data[config][0].append(N)
        data[config][1].append(d_sat)

# Sort by N
for config in data:
    ns, ds = data[config]
    paired = sorted(zip(ns, ds))
    data[config] = ([p[0] for p in paired], [p[1] for p in paired])

# N values as evenly spaced labels
n_values = [64, 80, 96, 112, 128, 256]
n_to_x = {n: i for i, n in enumerate(n_values)}

# Style mapping: color = M, shape = format
m_colors = {64: 'tab:blue', 128: 'tab:orange', 256: 'tab:green'}
fmt_list = ['BF16', 'E4M3', 'S8', 'F4', 'MXF8', 'MXF4']
fmt_markers = {
    'BF16': 'o', 'E4M3': 's', 'S8': '^',
    'F4': 'D', 'MXF8': 'v', 'MXF4': 'P',
}
fmt_jitter = {fmt: (i - 2.5) * 0.06 for i, fmt in enumerate(fmt_list)}
m_offset = {64: -0.2, 128: 0.0, 256: 0.2}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, cta, title in [(ax1, 1, '1SM'), (ax2, 2, '2SM')]:
    for (fmt, c, M), (ns, ds) in sorted(data.items()):
        if c != cta:
            continue
        xs = [n_to_x[n] + fmt_jitter[fmt] + m_offset[M] for n in ns]
        ax.plot(xs, ds,
                ls='-', marker=fmt_markers[fmt],
                color=m_colors[M], markersize=6, alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel('N')
    ax.set_xticks(range(len(n_values)))
    ax.set_xticklabels([str(n) for n in n_values])
    ax.set_yticks(range(1, 11))
    ax.grid(True, alpha=0.3)

ax1.set_ylabel('Saturated Pipeline Depth (D_sat)')

# M legend (top)
m_handles = []
for M in sorted(m_colors.keys()):
    m_handles.append(Line2D([0], [0], color=m_colors[M], lw=2, label=f'M={M}'))
m_legend = fig.legend(handles=m_handles, fontsize=8, ncol=3, loc='upper center',
                      bbox_to_anchor=(0.5, 1.02))

# Format legend (below M legend)
fmt_handles = []
for fmt in fmt_list:
    fmt_handles.append(Line2D([0], [0], color='gray', ls='', marker=fmt_markers[fmt],
                              markersize=6, label=fmt))
fig.legend(handles=fmt_handles, fontsize=8, ncol=6, loc='upper center',
           bbox_to_anchor=(0.5, 0.97))
fig.add_artist(m_legend)

fig.suptitle('D_sat Decreases with N (TS mode, all formats)', y=1.12, fontsize=13)

plt.tight_layout()
plt.savefig('pipeline_depth.png', dpi=150, bbox_inches='tight')
print('Saved pipeline_depth.png')
