#!/usr/bin/env python3
"""Roofline plot: arithmetic intensity vs throughput for UMMA.

Shows FP8 (E4M3) SS 1SM M=128 data with SMEM bandwidth = 128 B/cycle as the
memory-bound slope.
"""

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np

SMEM_BW = 128  # bytes/cycle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input CSV file")
    parser.add_argument("output", help="Output plot filename")
    args = parser.parse_args()

    # Read E4M3 SS 1SM M=128 data
    points = []
    with open(args.input) as f:
        for row in csv.DictReader(f):
            if row["Format"] != "E4M3":
                continue
            if row["ABLayout"] != "SS" or int(row["CTAGroup"]) != 1:
                continue
            if int(row["M"]) != 128:
                continue
            m, n, k = int(row["M"]), int(row["N"]), int(row["K"])
            tput = float(row["FLOPsPerCycle"])
            # FP8: elem_bytes = 1, SS reads A+B from SMEM
            smem_bytes = (m + n) * k  # * 1 byte per element
            ai = (2 * m * n * k) / smem_bytes
            points.append((ai, tput, n))

    points.sort()

    peak = 4 * 128 * 32  # 16384 FLOPs/cycle for E4M3 1SM M=128
    ridge_ai = peak / SMEM_BW  # 128

    fig, ax = plt.subplots(figsize=(10, 6))

    # Roofline lines
    max_ai = max(p[0] for p in points) * 1.15
    ax.plot([0, ridge_ai], [0, peak], "k-", linewidth=2, zorder=3)
    ax.plot([ridge_ai, max_ai], [peak, peak], "k-", linewidth=2, zorder=3)
    ax.plot(ridge_ai, peak, "ko", markersize=6, zorder=4)

    # Data points
    xs, ys, ns = zip(*points)
    ax.scatter(xs, ys, s=80, c="#2D6A8F", edgecolors="#1A4A6A", zorder=5,
               label="E4M3 SS 1SM M=128")

    # Label each point with N value
    for x, y, n in points:
        ax.annotate(f"N={n}", (x, y), textcoords="offset points",
                    xytext=(8, -12), fontsize=9)

    # Bandwidth slope label — compute rotation to match the line angle
    ax.set_xlim(0, max_ai)
    ax.set_ylim(0, peak * 1.15)
    fig.tight_layout()

    bbox = ax.get_position()
    fig_w, fig_h = fig.get_size_inches()
    ax_w, ax_h = bbox.width * fig_w, bbox.height * fig_h
    x_range = ax.get_xlim()[1]
    y_range = ax.get_ylim()[1]
    dx_inch = (ridge_ai / x_range) * ax_w
    dy_inch = (peak / y_range) * ax_h
    angle = np.degrees(np.arctan2(dy_inch, dx_inch))

    ax.annotate(f"SMEM BW = {SMEM_BW} B/cycle", xy=(ridge_ai * 0.4, peak * 0.4),
                fontsize=11, fontweight="bold", rotation=angle,
                ha="center", va="bottom", rotation_mode="anchor")

    ax.annotate(f"Peak = {peak:,} FLOPs/cycle", xy=(max_ai * 0.85, peak),
                fontsize=11, fontweight="bold", ha="center", va="bottom",
                xytext=(0, 6), textcoords="offset points")

    ax.set_xlabel("Arithmetic Intensity (FLOPs / SMEM byte)", fontsize=12)
    ax.set_ylabel("Throughput (FLOPs / cycle)", fontsize=12)
    ax.set_title("UMMA Roofline: FP8 (E4M3) 1SM M=128", fontsize=13)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.3)

    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
