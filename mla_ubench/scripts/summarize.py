#!/usr/bin/env python3
"""汇总原子上限表 + 组合模型对拍 + 瓶颈饼图（H800）。SCAFFOLD。
方法见 docs/03-composition-model.md, docs/04-attribution.md。

产出:
  report/atom_limits.csv   每原子: cycles, 吞吐(flop|byte/clk/SM), 主维度, ncu 隔离验证是否通过
  report/model.csv         T_producer, T_consumer, T_block, T_fused(pred), T_measured, η, 瓶颈
  report/bottleneck.png    producer vs consumer 各原子占比堆叠图
"""
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atoms-dir", default="atoms")
    ap.add_argument("--e2e-log", default="e2e/log")
    ap.add_argument("--out", default="report/")
    args = ap.parse_args()

    # TODO(impl):
    #   1) 解析 atoms/*/log → 每原子吞吐 + cycles，写 atom_limits.csv。
    #   2) 合并 model/compose.py 的输出 → model.csv（T_fused pred vs e2e measured, η）。
    #   3) 画 producer/consumer 堆叠柱状（各原子 cycles 占比）→ bottleneck.png。
    #   4) DSM crossover A/B: A2-full vs (A2-half+A3) 的 producer 时间对比。
    print(f"[TODO] summarize atoms in {args.atoms_dir} + calibrate vs {args.e2e_log} -> {args.out}")

if __name__ == "__main__":
    main()
