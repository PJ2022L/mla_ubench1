#!/usr/bin/env python3
"""
Plot CUTLASS GEMM mainloop benchmark results.

Single-CTA benchmark: M=TILE_M, N=TILE_N, K=131072 (vocab-sized).
Measures one SM's mainloop throughput in isolation.

Per-SM peak TFLOPS at 1300 MHz for B200:
  BF16: 1 * 1300e6 * 8192  FLOPs/SM/cycle = 10.65 TFLOPS
  FP8:  1 * 1300e6 * 16384 FLOPs/SM/cycle = 21.30 TFLOPS
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import sys
from pathlib import Path

# ---- Per-SM hardware peak at 1300 MHz ----
CLOCK_MHZ = 1300
# FLOPs per SM per cycle (from B200 spec: 2250 TFLOPS BF16 dense @ ~1860 MHz boost / 148 SMs)
FLOPS_PER_SM_PER_CYCLE = {"BF16": 8192, "FP8_E4M3": 16384}
PEAK_TFLOPS = {
    dt: CLOCK_MHZ * 1e6 * fpc / 1e12
    for dt, fpc in FLOPS_PER_SM_PER_CYCLE.items()
}
print(f"Per-SM peak TFLOPS @ {CLOCK_MHZ} MHz: BF16={PEAK_TFLOPS['BF16']:.2f}, FP8={PEAK_TFLOPS['FP8_E4M3']:.2f}")

# ---- Load data ----
csv_path = Path(__file__).parent / "results.csv"
if len(sys.argv) > 1:
    csv_path = Path(sys.argv[1])

lines = [l for l in csv_path.read_text().splitlines() if not l.startswith("#") and l.strip()]
from io import StringIO
df = pd.read_csv(StringIO("\n".join(lines)))

# Computed columns
df["tile_label"] = df.apply(lambda r: f"{int(r.TILE_M)}x{int(r.TILE_N)}x{int(r.TILE_K)}", axis=1)
df["mn_label"] = df.apply(lambda r: f"{int(r.TILE_M)}x{int(r.TILE_N)}", axis=1)
df["peak"] = df["dtype"].map(PEAK_TFLOPS)
df["pct_sol"] = 100.0 * df["TFLOPS"] / df["peak"]

# ---- Color palette ----
COLORS = plt.cm.tab10.colors

def make_tile_colormap(tiles):
    return {t: COLORS[i % len(COLORS)] for i, t in enumerate(sorted(tiles))}

# MxN tiles to show in %SOL-vs-stages and TILE_K sweep plots
FOCUS_MN = [(128, 128), (128, 192), (128, 256), (64, 128), (64, 64), (64, 32)]


# ===========================================================================
# Plot 1: TFLOPS vs Stages, one subplot per dtype, lines per tile shape
#          (only default TILE_K: 64 for BF16, 128 for FP8)
# ===========================================================================
def plot_tflops_vs_stages():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=False)

    for ax, (dtype, default_tk) in zip(axes, [("BF16", 64), ("FP8_E4M3", 128)]):
        sub = df[(df["dtype"] == dtype) & (df["TILE_K"] == default_tk)].copy()
        if sub.empty:
            continue

        tiles = sorted(sub["tile_label"].unique())
        cmap = make_tile_colormap(tiles)

        for tile in tiles:
            td = sub[sub["tile_label"] == tile].sort_values("STAGES")
            ax.plot(td["STAGES"], td["TFLOPS"], "o-", label=tile, color=cmap[tile], markersize=5)

        ax.axhline(PEAK_TFLOPS[dtype], color="red", linestyle="--", alpha=0.7, label=f"Peak ({PEAK_TFLOPS[dtype]:.0f} TF)")
        ax.set_xlabel("Pipeline Stages (0 = auto)")
        ax.set_ylabel("TFLOPS")
        ax.set_title(f"{dtype} — TFLOPS vs Pipeline Stages (TILE_K={default_tk})")
        ax.legend(fontsize=7, ncol=2, title="MxNxK tile")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sorted(sub["STAGES"].unique()))

    fig.tight_layout()
    fig.savefig(csv_path.parent / "plot_tflops_vs_stages.png", dpi=150)
    print("Saved plot_tflops_vs_stages.png")


# ===========================================================================
# Plot 2: %SOL vs Stages (same structure as plot 1)
# ===========================================================================
def plot_sol_vs_stages():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=False)

    for ax, (dtype, default_tk) in zip(axes, [("BF16", 64), ("FP8_E4M3", 128)]):
        sub = df[(df["dtype"] == dtype) & (df["TILE_K"] == default_tk) & (df["STAGES"] != 0)].copy()
        sub = sub[sub.apply(lambda r: (int(r.TILE_M), int(r.TILE_N)) in FOCUS_MN, axis=1)]
        if sub.empty:
            continue

        tiles = sorted(sub["tile_label"].unique())
        cmap = make_tile_colormap(tiles)

        for tile in tiles:
            td = sub[sub["tile_label"] == tile].sort_values("STAGES")
            ax.plot(td["STAGES"], td["pct_sol"], "o-", label=tile, color=cmap[tile], markersize=5)

        ax.axhline(100, color="red", linestyle="--", alpha=0.7, label="100% SOL")
        ax.set_xlabel("Pipeline Stages")
        ax.set_ylabel("% Speed of Light")
        ax.set_title(f"Pipeline %SOL by Stage Num ({dtype}, MMA_K={default_tk})")
        ax.legend(fontsize=7, ncol=2, title="MxNxK tile")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sorted(sub["STAGES"].unique()))
        ax.set_ylim(0, 105)

    fig.tight_layout()
    fig.savefig(csv_path.parent / "plot_sol_vs_stages.png", dpi=150)
    print("Saved plot_sol_vs_stages.png")


# ===========================================================================
# Plot 3: Best TFLOPS per MxN tile (best across all TILE_K and stages)
# ===========================================================================
def plot_best_per_tile():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, dtype in zip(axes, ["BF16", "FP8_E4M3"]):
        sub = df[df["dtype"] == dtype].copy()
        if sub.empty:
            continue

        # Best across all TILE_K and stages for each (M,N)
        best = sub.loc[sub.groupby("mn_label")["TFLOPS"].idxmax()].sort_values("TFLOPS", ascending=True)

        bars = ax.barh(range(len(best)), best["TFLOPS"], color=COLORS[0], edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(len(best)))
        ax.set_yticklabels(best["mn_label"], fontsize=8)
        ax.axvline(PEAK_TFLOPS[dtype], color="red", linestyle="--", alpha=0.7, label=f"Peak ({PEAK_TFLOPS[dtype]:.0f} TF)")

        # Annotate bars with %SOL and winning config
        for i, (_, row) in enumerate(best.iterrows()):
            pct = 100.0 * row["TFLOPS"] / PEAK_TFLOPS[dtype]
            ax.text(row["TFLOPS"] + PEAK_TFLOPS[dtype] * 0.01, i,
                    f'{row["TFLOPS"]:.1f} ({pct:.1f}%) k={int(row["TILE_K"])} s={int(row["STAGES"])}',
                    va="center", fontsize=7)

        ax.set_xlabel("TFLOPS")
        ax.set_title(f"{dtype} — Best TFLOPS per M\u00d7N Tile (best TILE_K & stages)")
        ax.legend()
        ax.grid(True, axis="x", alpha=0.3)
        ax.set_xlim(0, PEAK_TFLOPS[dtype] * 1.15)

    fig.tight_layout()
    fig.savefig(csv_path.parent / "plot_best_per_tile.png", dpi=150)
    print("Saved plot_best_per_tile.png")


# ===========================================================================
# Plot 4: TILE_K sweep — TFLOPS vs TILE_K for key (M,N) combos
# ===========================================================================
def plot_tile_k_sweep():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, dtype in zip(axes, ["BF16", "FP8_E4M3"]):
        sub = df[df["dtype"] == dtype].copy()
        if sub.empty:
            continue

        # For each (TILE_M, TILE_N), we need multiple TILE_K values
        grouped = sub.groupby(["TILE_M", "TILE_N"])["TILE_K"].nunique()
        multi_k = [mn for mn in grouped[grouped > 1].index.tolist() if mn in FOCUS_MN]

        if not multi_k:
            ax.text(0.5, 0.5, "No TILE_K sweep data", transform=ax.transAxes, ha="center")
            continue

        mn_labels = [f"{m}x{n}" for m, n in multi_k]
        cmap = {l: COLORS[i % len(COLORS)] for i, l in enumerate(mn_labels)}

        for (tm, tn), label in zip(multi_k, mn_labels):
            tile_sub = sub[(sub["TILE_M"] == tm) & (sub["TILE_N"] == tn)]
            # Best stages per TILE_K
            best_k = tile_sub.loc[tile_sub.groupby("TILE_K")["TFLOPS"].idxmax()].sort_values("TILE_K")
            ax.plot(best_k["TILE_K"], best_k["TFLOPS"], "o-", label=f"{label} (best stage)", color=cmap[label], markersize=6)

        ax.axhline(PEAK_TFLOPS[dtype], color="red", linestyle="--", alpha=0.7, label=f"Peak ({PEAK_TFLOPS[dtype]:.0f} TF)")
        ax.set_xlabel("TILE_K")
        ax.set_ylabel("TFLOPS (best across stages)")
        ax.set_title(f"{dtype} — TFLOPS vs TILE_K")
        ax.legend(fontsize=8, title="TILE_M x TILE_N")
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

    fig.tight_layout()
    fig.savefig(csv_path.parent / "plot_tile_k_sweep.png", dpi=150)
    print("Saved plot_tile_k_sweep.png")


# ===========================================================================
# Plot 5: Heatmap — best %SOL per (TILE_M x TILE_N), best across all TILE_K
# ===========================================================================
def plot_heatmap():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, dtype in zip(axes, ["BF16", "FP8_E4M3"]):
        sub = df[df["dtype"] == dtype].copy()
        if sub.empty:
            continue

        # Best %SOL across all TILE_K and stages for each (TILE_M, TILE_N)
        best = sub.groupby(["TILE_M", "TILE_N"])["pct_sol"].max().reset_index()
        # Also find which TILE_K achieved the best
        best_rows = sub.loc[sub.groupby(["TILE_M", "TILE_N"])["pct_sol"].idxmax()]

        pivot_sol = best.pivot(index="TILE_M", columns="TILE_N", values="pct_sol")
        pivot_sol = pivot_sol.sort_index(ascending=False)

        pivot_k = best_rows.pivot(index="TILE_M", columns="TILE_N", values="TILE_K")
        pivot_k = pivot_k.sort_index(ascending=False)

        im = ax.imshow(pivot_sol.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
        ax.set_xticks(range(len(pivot_sol.columns)))
        ax.set_xticklabels(pivot_sol.columns.astype(int), fontsize=8)
        ax.set_yticks(range(len(pivot_sol.index)))
        ax.set_yticklabels(pivot_sol.index.astype(int), fontsize=8)
        ax.set_xlabel("TILE_N")
        ax.set_ylabel("TILE_M")
        ax.set_title(f"{dtype} — Best %SOL (best TILE_K)")

        # Annotate cells with %SOL and winning TILE_K
        for i in range(len(pivot_sol.index)):
            for j in range(len(pivot_sol.columns)):
                val = pivot_sol.values[i, j]
                tk = pivot_k.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.1f}%\nk={int(tk)}", ha="center", va="center", fontsize=7,
                            color="white" if val > 60 else "black")

        fig.colorbar(im, ax=ax, label="%SOL", shrink=0.8)

    fig.tight_layout()
    fig.savefig(csv_path.parent / "plot_heatmap_sol.png", dpi=150)
    print("Saved plot_heatmap_sol.png")


# ---- Run all plots ----
if __name__ == "__main__":
    plot_tflops_vs_stages()
    plot_sol_vs_stages()
    plot_best_per_tile()
    plot_tile_k_sweep()
    plot_heatmap()
    print("\nAll plots saved.")
