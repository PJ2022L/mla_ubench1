#!/usr/bin/env bash
# 编译 + 跑所有原子 + e2e，收集 log。SCAFFOLD。H800(sm_90a)。
# 用法: ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

ATOMS="a1_kv_gather a2_dequant a3_dsm_crossover a4_qk_gemm a5_pv_gemm a6_softmax a7_tma_store a8_combine"

echo "== build + run atoms =="
for a in $ATOMS; do
  echo "--- $a ---"
  make -C "atoms/$a" run || echo "[TODO] $a not implemented yet"
done

echo "== build + run e2e baseline =="
make -C e2e run || echo "[TODO] e2e not implemented yet"

echo "== compose model + calibrate =="
python model/compose.py --atoms-dir atoms --e2e-log e2e/log

echo "== summarize =="
python scripts/summarize.py --atoms-dir atoms --e2e-log e2e/log --out report/
