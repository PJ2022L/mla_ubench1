# tile64x512_f32_sm90

- **Target**: CUTLASS `SM90_BULK_COPY_S2G`, emitted as
  `cp.async.bulk.global.shared::cta.bulk_group`.
- **Geometry**: 256 threads and eight warp leaders. Each leader stores eight
  2,048-byte rows from the source-like `[64,520]` FP32 shared layout. A tile is
  exactly 64 bulk-copy calls and 131,072 useful bytes.
- **Timed body**: row stores, per-issuing-lane
  `cp.async.bulk.commit_group`, and `cp.async.bulk.wait_group.read 0`. FP32
  shared staging is outside the timed region. Completion depth is fixed at 1,
  matching the source epilogue.
- **Working set**: `--working-set l2|hbm`, `--working-set-mib`, and
  `--pattern sequential|local|random` control global destination reuse.
- **Metrics**: cycle/tile, cycle/row-store, byte/clk/SM.
- **Validation**: the remote run checks six coordinates spanning the first and
  last rows of the final destination tile against the exact shared source.
- **Consumers**: dense and sparse decode split-KV main-kernel epilogues.
- **Execution**: build locally if needed, but run only on an H800/SM90a host.

## H800 Results

| Accepted run | Variant | Median cycles | p10 / p90 | Primary metric | Correctness |
|---|---|---:|---|---|---|
| 尚无 accepted H800 run | - | - | - | - | - |

完整数据保存在本目录 `result/runs/<run_id>/`，`result/summary.csv` 由所有不可变 runs 重建。验收 H800 数据后，在本表链接 accepted run 并填写 working set、pattern 和 CTA count；失败或 parse-error run 不得登记。
