# tile64x576_bf16_sm90

- **Target**：`SM90_TMA_LOAD`，GMEM BF16 tile → CTA shared。
- **Geometry**：Q tile `[64,576]` BF16；扫描 swizzle、cache hint 和 pipeline depth。
- **Timed body**：TMA issue + transaction barrier completion；descriptor 创建和 host setup 不计时。
- **Metrics**：cycle/tile、byte/clk/SM，多 CTA 时 GB/s。
- **Consumers**：FlashMLA decode 的 Q prologue，以及其他 SM90 attention/GEMM producer。
- **Source/config**：共享 `../benchmark.cu`；本配置固定 `64×576` BF16。
