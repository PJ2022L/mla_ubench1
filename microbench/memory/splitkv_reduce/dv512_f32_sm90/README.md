# dv512_f32_sm90

- **Target**：读取 FP32 partial O/LSE，执行 LSE rescale 和 float4 reduction。
- **Geometry**：`D_V=512`，扫描 `num_splits`；这是 HBM 主导的紧耦合指令簇，不宣称是单条指令。
- **Timed body**：partial loads + exp2 scale + FMA accumulation + output sink。
- **Metrics**：cycle/row、byte/clk/SM、GB/s、effective FMA rate。
- **Consumers**：所有 split-KV attention combine path。
- **Source/config**：共享 `../benchmark.cu`；本配置固定 `D_V=512`、FP32 partial。
