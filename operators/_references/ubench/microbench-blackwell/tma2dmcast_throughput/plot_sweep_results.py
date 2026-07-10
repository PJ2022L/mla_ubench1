#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

NUM_WARPS = 4

df = pd.read_csv('tma2dmcast_sweep_full.csv')

# bif = NUM_WARPS * tile_bytes / 1024
df['bif_KiB'] = NUM_WARPS * df['tile_bytes'] / 1024

# SMEM fill throughput (GB/s) = tma_ld_bytes / gpu_time_ns
df['throughput_GBs'] = df['tma_ld_bytes'] / df['gpu_time_ns']

# L2 sectors per SMEM byte = (lts_sectors_tex_read * 32) / tma_ld_bytes
df['l2_per_smem'] = (df['lts_sectors_tex_read'] * 32) / df['tma_ld_bytes']

# Mode labels: 0 = implicit, 1 = explicit
mode_labels = {0: 'implicit', 1: 'explicit'}
colors = {
    (0, 1): '#d62728', (0, 2): '#1f77b4', (0, 4): '#ff7f0e',
    (1, 1): '#d62728', (1, 2): '#1f77b4', (1, 4): '#ff7f0e',
}
markers = {1: 'x', 2: 'o', 4: 's'}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

# CL=1 baseline (use mode 0 data, shown as solid)
sub = df[(df['mode'] == 0) & (df['cluster_size'] == 1)]
best = sub.loc[sub.groupby('bif_KiB')['throughput_GBs'].idxmax()].sort_values('bif_KiB')
ax1.plot(best['bif_KiB'], best['throughput_GBs'], color=colors[(0, 1)],
         marker=markers[1], label='cluster=1 (baseline)', linewidth=2, markersize=5,
         linestyle='-', alpha=0.85)
ax2.plot(best['bif_KiB'], best['l2_per_smem'], color=colors[(0, 1)],
         marker=markers[1], label='cluster=1 (baseline)', linewidth=2, markersize=5,
         linestyle='-', alpha=0.85)

# CL=2,4 with implicit (dashed) and explicit (solid)
for mode in [0, 1]:
    mdf = df[df['mode'] == mode]
    linestyle = '--' if mode == 0 else '-'
    for cl in [2]:
        sub = mdf[mdf['cluster_size'] == cl]
        if sub.empty:
            continue
        best = sub.loc[sub.groupby('bif_KiB')['throughput_GBs'].idxmax()]
        best = best.sort_values('bif_KiB')
        label = f'cluster={cl} {mode_labels[mode]}'
        ax1.plot(best['bif_KiB'], best['throughput_GBs'], color=colors[(mode, cl)],
                 marker=markers[cl], label=label, linewidth=2, markersize=5,
                 linestyle=linestyle, alpha=0.85)
        ax2.plot(best['bif_KiB'], best['l2_per_smem'], color=colors[(mode, cl)],
                 marker=markers[cl], label=label, linewidth=2, markersize=5,
                 linestyle=linestyle, alpha=0.85)

ax1.set_ylabel('SMEM Fill Throughput (GB/s)', fontsize=12)
ax1.set_title('TMA Multicast - Explicit vs. Implicit', fontsize=13)
ax1.legend(fontsize=9, ncol=2, handlelength=3.5)
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Bytes-in-flight per SM (KiB)', fontsize=12)
ax2.set_ylabel('L2 bytes per SMEM byte', fontsize=12)
ax2.legend(fontsize=9, ncol=2, handlelength=3.5)
ax2.grid(True, alpha=0.3)

x_ticks = sorted(df['bif_KiB'].unique())
if len(x_ticks) > 30:
    x_ticks_show = [x for x in x_ticks if x % 16 == 0 or x <= 16]
else:
    x_ticks_show = x_ticks
ax2.set_xticks(x_ticks_show)
ax2.set_xticklabels([f'{int(x)}' for x in x_ticks_show], rotation=45, ha='right', fontsize=9)

plt.tight_layout()
plt.savefig('tma2dmcast_sweep_plot.png', dpi=150)
print("Saved plot to tma2dmcast_sweep_plot.png")
