#!/usr/bin/env bash
# Build and run every reusable SM90 benchmark. Operator e2e/modeling is separate.
set -euo pipefail
cd "$(dirname "$0")/.."

BENCHMARKS=(
  memory/global_load/128b_nc_l2_sm90
  memory/shared_store/128b_sm90
  memory/dsm_store/128b_cluster2_sm90
  memory/tma_load/tile64x64_bf16_sm90
  memory/tma_load/tile64x576_bf16_sm90
  memory/tma_store/tile64x512_bf16_2d_sm90
  memory/tma_store/tile64x512_bf16_5d_sm90
  memory/stmatrix/m64n64_b16_x4_sm90
  memory/stmatrix/m64n256_b16_x4_sm90
  memory/splitkv_reduce/dv512_f32_sm90
  compute/convert/fp8x8_to_bf16x8_sm90
  compute/wgmma/m64n64k16_bf16_rs_ss_sm90
  compute/wgmma/m64n256k16_bf16_rs_ss_sm90
  compute/softmax/online_m64n64_exp2_shfl_sm90
)

for benchmark in "${BENCHMARKS[@]}"; do
  echo "--- $benchmark ---"
  make -C "microbench/$benchmark" run
done
