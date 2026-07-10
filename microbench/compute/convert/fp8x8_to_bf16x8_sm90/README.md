# fp8x8_to_bf16x8_sm90

- **Target**：FlashMLA helper 的 FP8 e4m3×8 + scale → BF16×8 转换序列。
- **Geometry**：128 threads，输入预置 register/shared；不在计时区写 shared memory。
- **Timed body**：转换序列和寄存器依赖 sink。
- **Metrics**：cycle/cvt、cvt/clk/SM、element/clk/SM。
- **Consumers**：FlashMLA SM90 sparse FP8 decode；其他 Hopper FP8 cache path。
- **Evidence caveat**：helper 是 source-level contract，必须用 SASS 记录实际展开序列。
- **Source/config**：共享 `../benchmark.cu`；本配置固定 8 个 FP8 输入与 8 个 BF16 输出。
