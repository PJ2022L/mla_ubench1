# fp8x8_to_bf16x8_sm90

- **Target**：FlashMLA helper 的 FP8 e4m3×8 + scale → BF16×8 转换序列。
- **Geometry**：一个 128-thread warpgroup，每 thread 保留 16 组独立 FP8x8 register 输入。
- **Timed body**：只含完整转换序列和零指令 compiler dependency；BF16x8 结果在计时结束后写 global sink，计时区不含 checksum/global/shared memory 操作。
- **Metrics**：cycle/cvt、cvt/clk/SM、element/clk/SM。
- **Consumers**：FlashMLA SM90 sparse FP8 decode；其他 Hopper FP8 cache path。
- **Evidence caveat**：helper 是 source-level contract，必须用 SASS 记录实际展开序列。
- **Source/config**：共享 `../benchmark.cu`；本配置固定 8 个 FP8 输入与 8 个 BF16 输出。

## H800 Results

| Accepted run | Variant | Median cycles | p10 / p90 | Primary metric | Correctness |
|---|---|---:|---|---|---|
| 尚无 accepted H800 run | - | - | - | - | - |

完整数据保存在本目录 `result/runs/<run_id>/`，`result/summary.csv` 由所有不可变 runs 重建。验收 H800 数据后，在本表链接 accepted run 并填写对应 variant；失败或 parse-error run 不得登记。
