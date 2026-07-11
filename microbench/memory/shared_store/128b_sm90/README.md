# 128b_sm90

- **Target**：register → CTA shared 的 128-bit store。
- **Geometry**：128 threads，每轮每 thread 18 个 `st.shared.v4.u32`，默认地址精确复现 sparse-decode WG2 的 16 个 NoPE + 2 个 RoPE K-major stores。
- **Timed body**：128-bit register value、目标 shared store 与最小循环依赖；sink/readback 在计时外，不含 convert/DSM。
- **Metrics**：cycle/store、store/clk/SM、byte/clk/SM、bank-conflict counters。
- **CLI**：`--pattern kmajor|linear|hot --working-set-bytes N --repeat N --warmup N --samples N`。
- **Consumers**：FlashMLA sparse decode producer 和其他 WGMMA operand staging path。
- **Source/config**：共享 `../benchmark.cu`；本配置固定 128-bit shared store。

## H800 Results

| Accepted run | Variant | Median cycles | p10 / p90 | Primary metric | Correctness |
|---|---|---:|---|---|---|
| 尚无 accepted H800 run | - | - | - | - | - |

完整数据保存在本目录 `result/runs/<run_id>/`，`result/summary.csv` 由所有不可变 runs 重建。验收 H800 数据后，在本表链接 accepted run 并填写 pattern 和 working set；失败或 parse-error run 不得登记。
