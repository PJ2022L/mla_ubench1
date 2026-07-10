# m64n64_b16_x4_sm90

- **Target**：CUTLASS `SM90_U32x4_STSM_N` / emitted `stmatrix` family for B16 fragments。
- **Geometry**：一个 128-thread warpgroup；保存 `[64,64]` B16 P tile。
- **Timed body**：register fragment → swizzled shared tile + required async-proxy fence；不含 softmax/PV。
- **Metrics**：cycle/tile、cycle/instruction、byte/clk/SM、shared bank-conflict counters。
- **Consumers**：dense decode 的 P cross-WG exchange；dense/sparse output staging。
- **Evidence caveat**：必须保存 PTX/SASS，确认 CUTLASS atom 实际 emitted form。
- **Source/config**：共享 `../benchmark.cu`；本配置固定 `M=64,N=64,x4`。
