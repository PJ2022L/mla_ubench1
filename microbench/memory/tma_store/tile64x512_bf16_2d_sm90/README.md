# tile64x512_bf16_2d_sm90

- **Target**：register→shared staging 后的 SM90 tensor TMA store。
- **Geometry**：`[64,512]` BF16，dense 2D tensor descriptor。
- **Timed body**：STSM staging + TMA store + 必需 completion；descriptor setup 不计时。
- **Metrics**：cycle/tile、byte/clk/SM、多 CTA GB/s；2D/5D 分开报告。
- **Consumers**：FlashMLA dense/sparse decode output epilogue。
- **Source/config**：共享 `../benchmark.cu`；本配置固定 rank 2。
