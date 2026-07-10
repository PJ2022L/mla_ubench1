# m64n256k16_bf16_rs_ss_sm90

- **Target**：`MMA_64x256x16_F32BF16BF16_RS` 与 `..._SS` 两种 operand mode。
- **Geometry**：一个 128-thread warpgroup；RS/SS 分开运行、分开报告。
- **Timed body**：WGMMA issue/commit/wait 和 accumulator sink，不含 softmax/shared handoff。
- **Metrics**：cycle/instruction、instruction/clk/SM、flop/clk/SM。
- **Consumers**：FlashMLA sparse decode local/remote PV（每支每 block 4 次）。
- **Source/config**：共享 `../benchmark.cu`；本目录 Makefile 固定 `M=64,N=256,K=16`。
