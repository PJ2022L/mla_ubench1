# tile64x512_bf16_5d_sm90

- **Target**：shared staging 后的 SM90 tensor TMA store。
- **Geometry**：`[64,512]` BF16，sparse output 的 5D tensor descriptor。
- **Independent variables**：pipeline depth、resident CTA 数、cache state。
- **Timed body**：TMA store + 必需 completion；descriptor setup 和输入生成不计时。
- **Metrics**：cycle/tile、byte/clk/SM，多 CTA时 GB/s。
- **Consumers**：FlashMLA sparse decode output epilogue。
- **Source/config**：共享 `../benchmark.cu`；本配置固定 rank 5。
