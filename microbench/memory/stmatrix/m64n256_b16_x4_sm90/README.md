# m64n256_b16_x4_sm90

- **Target**：CUTLASS `SM90_U32x4_STSM_N` / emitted `stmatrix` family for B16 fragments。
- **Geometry**：一个 128-thread warpgroup；保存 `[64,256]` B16 output half。
- **Independent variables**：repeat、地址 swizzle、warm/cold shared state。
- **Timed body**：register fragment → swizzled shared tile + required async-proxy fence。
- **Metrics**：cycle/tile、cycle/instruction、byte/clk/SM、shared bank-conflict counters。
- **Consumers**：dense/sparse decode output staging。
- **Source/config**：共享 `../benchmark.cu`；本配置固定 `M=64,N=256,x4`。
