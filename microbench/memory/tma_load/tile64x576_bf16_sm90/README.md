# tile64x576_bf16_sm90

- **Target**：`SM90_TMA_LOAD`，GMEM BF16 tile → CTA shared。
- **Geometry**：Q tile `[64,576]` BF16，由 9 条 SW128 `[64,64]` TMA transactions 组成。
- **Rank mapping**：`--rank 4`（默认）对应 dense decode 和 sparse decode Q；`--rank 3` 对应 sparse prefill Q；`--rank 2` 是 descriptor-rank control。
- **Fixed execution**：SW128、`EVICT_FIRST`、单 CTA/32 threads；当前不扫描 swizzle、cache hint 或 resident CTA 数。
- **CLI**：`--rank 2|3|4 --depth 1..8 --working-set-tiles N --repeat N --warmup N --samples N --device N`。实际 depth 受 opt-in shared-memory guard 限制；H800 对 73,728 B/tile 预计最多为 3。
- **Timed body**：TMA issue、mbarrier arrive/wait、stage/tile/9-transaction loops 和地址控制；descriptor encode、allocation、input initialization 和 correctness validation 不计时。
- **Dynamic count**：每个 logical tile 发出 9 条 rank 对应的 TMA transactions；总数为 `9×repeat`。
- **Metrics**：cycle/tile、transaction/clk/SM、requested byte/clk/SM。
- **Validation**：独立 kernel 消费首/末 working-set tile 的 9 个 transactions 的全部元素，以 per-transaction 权重累加非均匀确定性输入并复用 mbarrier phase；host checksum 可发现漏传、错 tile 或跨 transaction 错位，但不区分同一 transaction 内的元素排列，不能单独证明 SW128 物理顺序。
- **Static evidence**：rank 2/3/4 timed kernels 分别为 `UTMALDG.2D/3D/4D`，均使用 22 registers，STACK/LOCAL 0。
- **Consumers**：dense/sparse decode Q prologue（rank 4）、sparse prefill Q prologue（rank 3）和 rank-2 control。
- **Source/config**：共享 `../benchmark.cu`；本配置固定 `64×576` BF16。

## H800 Results

| Accepted run | Variant | Median cycles | p10 / p90 | Primary metric | Correctness |
|---|---|---:|---|---|---|
| 尚无 accepted H800 run | - | - | - | - | - |

完整数据保存在本目录 `result/runs/<run_id>/`，`result/summary.csv` 由所有不可变 runs 重建。验收 H800 数据后，在本表链接 accepted run 并填写 rank、depth 和 working set；失败或 parse-error run 不得登记。
