# gather64x576_bf16_sm90

- **Target**: 16-byte `cp.async.cg.shared.global.L2::cache_hint.L2::256B`.
- **Geometry**: one 128-thread producer warpgroup gathers one `[64,576]` BF16
  block. Each thread issues 36 copies; the block issues 4,608 copies and moves
  73,728 useful bytes.
- **Addressing**: sequential/local/random token indices with the same 8-thread
  group mapping used by sparse prefill.
- **Timed body**: repeated index load/address generation, exact copy issue,
  `cpasync_barrier_arrive_noinc`, mbarrier wait, and the required CTA reuse
  boundary. Cache-policy construction and input generation are outside timing.
- **Modes**: full block and the real `4/5/5/4` two-block segment schedule.
- **Metrics**: cycle/block, cycle/copy, copy/clk/SM, byte/clk/SM.
- **CLI**: `--mode block|pair --pattern sequential|local|random
  --working-set-tokens N --index-blocks N --repeat N --warmup N --samples N`.
- **Consumers**: SM90 sparse BF16 prefill K/V producer.

## H800 Results

| Accepted run | Variant | Median cycles | p10 / p90 | Primary metric | Correctness |
|---|---|---:|---|---|---|
| 尚无 accepted H800 run | - | - | - | - | - |

完整数据保存在本目录 `result/runs/<run_id>/`，`result/summary.csv` 由所有不可变 runs 重建。验收 H800 数据后，在本表链接 accepted run 并填写 mode、pattern 和 working set；失败或 parse-error run 不得登记。
