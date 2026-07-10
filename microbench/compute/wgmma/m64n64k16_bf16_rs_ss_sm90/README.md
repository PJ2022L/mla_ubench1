# m64n64k16_bf16_rs_ss_sm90

- **Target**：`MMA_64x64x16_F32BF16BF16_SS` 和对应 RS operand mode，必须分开运行/报告。
- **Geometry**：一个 128-thread warpgroup；shared/register operand 预置，不计 TMA/load。
- **Timed body**：fence/commit、可控 depth 的 WGMMA issue、wait、accumulator sink。
- **Metrics**：cycle/instruction、instruction/clk/SM、flop/clk/SM。
- **Consumers**：FlashMLA sparse decode QK（36 SS/page）；dense decode steady QK（32 SS+4 RS/page）。
- **Source/config**：共享 `../benchmark.cu`；本目录 Makefile 固定 `M=64,N=64,K=16`。
