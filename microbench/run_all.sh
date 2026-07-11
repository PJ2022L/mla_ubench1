#!/usr/bin/env bash
# Apply one explicit build action to every reusable SM90 benchmark.
set -euo pipefail
cd "$(dirname "$0")/.."

usage() {
  cat <<'EOF'
Usage: microbench/run_all.sh [compile|static|run|clean]

The default action is compile. The run action must be explicit and is only for
the remote SM90/H800 environment. RUN_ID and ARGS may be supplied through the
environment; otherwise run generates one shared RUN_ID for the whole batch.
EOF
}

if (( $# > 1 )); then
  usage >&2
  exit 2
fi

ACTION="${1:-compile}"
case "$ACTION" in
  compile|static|run|clean) ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

BENCHMARKS=(
  memory/global_load/128b_nc_l2_sm90
  memory/cp_async_g2s/gather64x576_bf16_sm90
  memory/shared_store/128b_sm90
  memory/dsm_store/128b_cluster2_sm90
  memory/tma_load/tile64x64_bf16_sm90
  memory/tma_load/tile64x576_bf16_sm90
  memory/tma_store/tile64x512_bf16_2d_sm90
  memory/tma_store/tile64x64_bf16_3d_sm90
  memory/tma_store/tile64x512_bf16_4d_sm90
  memory/tma_store/tile64x512_bf16_5d_sm90
  memory/bulk_store/tile64x512_f32_sm90
  memory/stmatrix/m64n64_b16_x4_sm90
  memory/stmatrix/m64n256_b16_x4_sm90
  memory/splitkv_reduce/dv512_f32_sm90
  compute/convert/fp8x8_to_bf16x8_sm90
  compute/wgmma/m64n64k16_bf16_rs_ss_sm90
  compute/wgmma/m64n256k16_bf16_rs_ss_sm90
  compute/softmax/online_m64n64_exp2_shfl_sm90
)

MAKE_ARGS=("$ACTION")
if [[ "$ACTION" == "run" ]]; then
  if [[ -n "${RUN_ID:-}" ]]; then
    if [[ "$RUN_ID" == "." || "$RUN_ID" == ".." ||
          ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
      echo "error: invalid RUN_ID" >&2
      exit 2
    fi
  else
    HOST_ID="$(hostname -s 2>/dev/null || hostname)"
    HOST_ID="${HOST_ID//[^[:alnum:]._-]/_}"
    RANDOM_ID="$(od -An -N4 -tx1 /dev/urandom | tr -d '[:space:]')"
    RUN_ID="$(date +%Y%m%d-%H%M%S)_${HOST_ID}_${RANDOM_ID}"
  fi
  MAKE_ARGS+=("RUN_ID=$RUN_ID")
  if [[ -n "${ARGS:-}" ]]; then
    MAKE_ARGS+=("ARGS=$ARGS")
  fi
  echo "Batch RUN_ID: $RUN_ID"
fi

for benchmark in "${BENCHMARKS[@]}"; do
  echo "--- $benchmark ---"
  make -C "microbench/$benchmark" "${MAKE_ARGS[@]}"
done
