# dv512_f32_sm90

- **Target**: FlashMLA combine's FP32 partial O/LSE reads, LSE rescaling, and
  `float4` reduction. This is a tightly coupled loop, not a single instruction.
- **Geometry**: 256 threads, eight warps/heads, `D_V=512`, and four `float4`
  values per thread. `--num-splits` supports 2 through 160 using the same
  32/64/96/128/160 template buckets as the source kernel.
- **Working set**: `--working-set l2|hbm`, `--working-set-mib`, and multiple
  rowsets control input residency. `--pattern sequential|random` controls
  rowset traversal.
- **Timed body**: initial partial prefetch, warp max/sum LSE reduction,
  lane-0 LSE output, `exp2f` scale staging, next-partial prefetch, FP32 FMAs,
  and BF16 output.
- **Metrics**: cycles per CTA iteration/row, requested bytes per clock, and
  effective FMA per clock. Reported bandwidth is per median CTA timing, not an
  aggregate whole-grid event measurement.
- **Validation**: the remote run checks two heads, six output columns, and LSE
  against a CPU reference for the final rowset.
- **Execution**: build locally if needed, but run only on an H800/SM90a host.

## H800 Results

| Accepted run | Variant | Median cycles | p10 / p90 | Primary metric | Correctness |
|---|---|---:|---|---|---|
| 尚无 accepted H800 run | - | - | - | - | - |

完整数据保存在本目录 `result/runs/<run_id>/`，`result/summary.csv` 由所有不可变 runs 重建。验收 H800 数据后，在本表链接 accepted run 并填写 num-splits、working set、pattern 和 CTA count；失败或 parse-error run 不得登记。
