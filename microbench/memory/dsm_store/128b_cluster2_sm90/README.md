# 128b_cluster2_sm90

- **Target**: exact
  `st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.s64` used by
  FlashMLA sparse FP8 decode.
- **Geometry**: cluster `(2,1,1)`, 128 threads per CTA, 18 stores per thread.
  One CTA moves a 32-by-576 BF16 K-tile slice (36,864 bytes); the cluster moves
  the source 64-token block (73,728 bytes).
- **Modes**: `--mode peer` performs the symmetric CTA exchange; `local` maps
  the destination and transaction barrier to the issuing CTA as a control.
- **Timed body**: barrier expectation, cluster synchronization, async stores,
  and required completion wait. Global loads and FP8 conversion are excluded.
- **Metrics**: cycles per source block/store and bytes per clock per cluster.
- **Execution**: build locally if needed, but run only on an H800/SM90a host.

## H800 Results

| Accepted run | Variant | Median cycles | p10 / p90 | Primary metric | Correctness |
|---|---|---:|---|---|---|
| 尚无 accepted H800 run | - | - | - | - | - |

完整数据保存在本目录 `result/runs/<run_id>/`，`result/summary.csv` 由所有不可变 runs 重建。验收 H800 数据后，在本表链接 accepted run 并填写 local/peer mode；失败或 parse-error run 不得登记。
