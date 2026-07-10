# tile64x64_bf16_sm90

- **Target**：`SM90_TMA_LOAD`，GMEM BF16 `[64,64]` tile → CTA shared。
- **Independent variables**：swizzle、cache hint、pipeline depth、resident CTA 数。
- **Timed body**：TMA issue + transaction-barrier completion；descriptor/host setup 不计时。
- **Metrics**：cycle/tile、byte/clk/SM，多 CTA 时 GB/s。
- **Consumers**：FlashMLA dense decode K tile 和其他 SM90 attention/GEMM producer。
- **Source/config**：共享 `../benchmark.cu`；本配置固定 `64×64` BF16。
