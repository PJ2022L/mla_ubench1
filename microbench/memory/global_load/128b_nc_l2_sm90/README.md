# 128b_nc_l2_sm90

- **Target**：128-bit non-coherent global vector load，扫描 L1 eviction/L2 prefetch hint。
- **Geometry**：128 threads；支持 sequential/local/random indices 和可控 working set。
- **Timed body**：地址读取 + 128-bit load + 链式 sink；不包含 FP8 convert/shared store。
- **Metrics**：cycle/load、load/clk/SM、unique byte/clk/SM、L2 hit rate。
- **Consumers**：FlashMLA sparse decode/prefill 的 indexed KV gather。
- **Source/config**：共享 `../benchmark.cu`；本配置固定 128-bit vector load。
