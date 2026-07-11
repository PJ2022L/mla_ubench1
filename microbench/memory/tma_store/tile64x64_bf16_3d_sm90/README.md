# tile64x64_bf16_3d_sm90

- **Target**：SW128 rank-3 `SM90_TMA_STORE_3D`。
- **Geometry**：每次写一块 `[64,64]` BF16（8,192 B）；source-like `[64,512]` CTA output 由 8 次此原子组成。
- **Fixed execution**：单 CTA/32 threads；timed loop 依次选择 8 个 64-column output slices。
- **CLI**：`--depth 1..8 --working-set-tiles N --repeat N --warmup N --samples N --device N`。
- **Timed body**：TMA issue、bulk-group commit/completion、stage/tile loop 和地址控制；shared 初始化、async-proxy fence、descriptor setup 与 correctness validation 不计时。
- **Validation**：独立 kernel 写首 tile 的首 slice 和末 tile 的末 slice，host 做加权 multiset checksum，并确认其他区域为零；不证明 transaction 内 SW128 物理排列。
- **Metrics**：cycle/tile、transaction/clk/SM、requested byte/clk/SM 和原始 cycle 分位数。
- **Static evidence**：timed kernel 为 `UTMASTG.3D`，12 registers，STACK/LOCAL 0。
- **Consumers**：SM90 sparse BF16 prefill epilogue。
- **Source/config**：共享 `../benchmark.cu`；本配置固定 rank 3。

## H800 Results

| Accepted run | Variant | Median cycles | p10 / p90 | Primary metric | Correctness |
|---|---|---:|---|---|---|
| 尚无 accepted H800 run | - | - | - | - | - |

完整数据保存在本目录 `result/runs/<run_id>/`，`result/summary.csv` 由所有不可变 runs 重建。验收 H800 数据后，在本表链接 accepted run 并填写 depth 和 working set；失败或 parse-error run 不得登记。
