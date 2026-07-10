# 128b_sm90

- **Target**：register → CTA shared 的 128-bit store。
- **Geometry**：128 threads，地址布局复现 K-major/swizzled operand tile；扫描 conflict pattern。
- **Timed body**：依赖链生成 128-bit value 后只执行 shared store；不含 convert/DSM。
- **Metrics**：cycle/store、store/clk/SM、byte/clk/SM、bank-conflict counters。
- **Consumers**：FlashMLA sparse decode producer 和其他 WGMMA operand staging path。
- **Source/config**：共享 `../benchmark.cu`；本配置固定 128-bit shared store。
