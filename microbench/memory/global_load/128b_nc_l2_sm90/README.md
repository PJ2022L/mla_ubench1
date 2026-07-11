# 128b_nc_l2_sm90

- **Target**：128-bit non-coherent global vector load，扫描 L1 eviction/L2 prefetch hint。
- **Geometry**：一个 128-thread producer WG；每轮每 thread 执行源码一致的 `1 scale + 8 NoPE + 2 RoPE` 共 11 个 load，CTA 共 1,408 个 source calls。
- **Timed body**：index load、真实 token/维度地址生成、`EVICT_LAST` 的 B128/B256 128-bit load 与 register sink；不包含 FP8 convert/shared store。
- **Metrics**：cycle/load、load/clk/SM、unique byte/clk/SM、L2 hit rate。
- **CLI**：`--pattern sequential|local|random --working-set-tokens N --index-blocks N --repeat N --warmup N --samples N`。
- **Consumers**：FlashMLA sparse FP8 decode 的 indexed register gather；BF16 prefill 使用单独的 `cp_async_g2s` family。
- **Source/config**：共享 `../benchmark.cu`；本配置固定 128-bit vector load。

## H800 Results

| Accepted run | Variant | Median cycles | p10 / p90 | Primary metric | Correctness |
|---|---|---:|---|---|---|
| 尚无 accepted H800 run | - | - | - | - | - |

完整数据保存在本目录 `result/runs/<run_id>/`，`result/summary.csv` 由所有不可变 runs 重建。验收 H800 数据后，在本表链接 accepted run 并填写 pattern、working set 和 index blocks；失败或 parse-error run 不得登记。
