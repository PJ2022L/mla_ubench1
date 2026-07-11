# tile64x512_bf16_2d_sm90

- **Target**：TMA-only rank-2 shared-to-global BF16 store control。
- **Geometry**：logical `[64,512]` tile，由 8 条 SW128 `[64,64]` transactions 组成。
- **Fixed execution**：单 CTA/32 threads；descriptor rank 与 SW128 固定，不扫描 resident CTA 或 cache state。
- **CLI**：`--depth 1..8 --working-set-tiles N --repeat N --warmup N --samples N --device N`；实际 depth 受 opt-in shared-memory guard 限制，且 `working-set-tiles >= min(depth,repeat)`，避免 outstanding WAW alias。
- **Timed body**：TMA issue、bulk-group commit/completion、stage/tile loop 和地址控制；shared 初始化、async-proxy fence、descriptor setup 与 correctness validation 不计时。
- **Validation**：独立 kernel 写首/末 working-set tile，host 对 8 个 transaction 分别做加权 multiset checksum，并确认未触达区域仍为零；不证明 transaction 内 SW128 物理排列。
- **Metrics**：cycle/tile、transaction/clk/SM、requested byte/clk/SM 和原始 cycle 分位数。
- **Static evidence**：timed kernel 为 `UTMASTG.2D`，20 registers，STACK/LOCAL 0。
- **Consumers**：rank-control；FlashMLA production path 分别使用 rank 4/5 配置。
- **Source/config**：共享 `../benchmark.cu`；本配置固定 rank 2。

## H800 Results

| Accepted run | Variant | Median cycles | p10 / p90 | Primary metric | Correctness |
|---|---|---:|---|---|---|
| 尚无 accepted H800 run | - | - | - | - | - |

完整数据保存在本目录 `result/runs/<run_id>/`，`result/summary.csv` 由所有不可变 runs 重建。验收 H800 数据后，在本表链接 accepted run 并填写 depth 和 working set；失败或 parse-error run 不得登记。
