# SM90 micro-benchmark index

这是新增/复用 micro-benchmark 的唯一入口。分析新 operator path 时，**先查本表，再写 `02-atom-decomposition.md`**。

## 去重与新增流程

1. 用 `family + operand mode + shape + dtype + arch` 查找精确配置。
2. 精确匹配时直接复用已有结果；不得按 operator 名再复制一份。
3. 指令与数据路径相同、只有参数不同：在现有 family 下添加配置叶子，并复用 family 的 `benchmark.cu`。
4. 只有指令族、operand path 或测量边界确实不同，才新建 family。
5. 新配置必须先登记本表，再被 operator 文档引用；`scaffold` 不能作为实测数据。

## Registry

本表不复制任何实验数值。`Configuration` 链接进入对应 leaf README；实测值和 accepted run 链接只写在其 `H800 Results` 章节，完整原始数据保存在该配置的 `result/runs/<run_id>/`。状态仍以本表为准。

| Category | Family | Configuration | Operand/shape | Independent variables | Outputs | Status | Used by |
|---|---|---|---|---|---|---|---|
| compute | [wgmma](compute/wgmma/) | [m64n64k16_bf16_rs_ss_sm90](compute/wgmma/m64n64k16_bf16_rs_ss_sm90/) | RS、SS 分开；64×64×16 BF16 | mode、issue depth、repeat、resident WG | cycle/inst, inst/clk/SM, flop/clk/SM | validated | dense/sparse decode and sparse prefill QK |
| compute | [wgmma](compute/wgmma/) | [m64n256k16_bf16_rs_ss_sm90](compute/wgmma/m64n256k16_bf16_rs_ss_sm90/) | RS、SS 分开；64×256×16 BF16 | mode、issue depth、repeat、resident WG | cycle/inst, inst/clk/SM, flop/clk/SM | validated | dense/sparse decode and sparse prefill PV |
| compute | [convert](compute/convert/) | [fp8x8_to_bf16x8_sm90](compute/convert/fp8x8_to_bf16x8_sm90/) | FP8 e4m3×8 + scale → BF16×8 | repeat | cycle/cvt, element/clk/SM | validated | sparse FP8 decode |
| compute | [softmax](compute/softmax/) | [online_m64n64_exp2_shfl_sm90](compute/softmax/online_m64n64_exp2_shfl_sm90/) | online softmax 64×64 | local/shared state、repeat | cycle/tile, exp2 element/clk/SM | validated | dense/sparse decode and sparse prefill |
| memory | [global_load](memory/global_load/) | [128b_nc_l2_sm90](memory/global_load/128b_nc_l2_sm90/) | 128-bit non-coherent load | pattern、working-set tokens、index blocks、repeat | cycle/load, requested/physical byte/clk/SM | validated | sparse decode gather |
| memory | [cp_async_g2s](memory/cp_async_g2s/) | [gather64x576_bf16_sm90](memory/cp_async_g2s/gather64x576_bf16_sm90/) | 16B indexed G2S, BF16 64x576 | block/pair schedule、pattern、working set、repeat | cycle/block, cycle/copy, byte/clk/SM | validated | sparse prefill gather |
| memory | [shared_store](memory/shared_store/) | [128b_sm90](memory/shared_store/128b_sm90/) | 128-bit CTA shared store | kmajor/linear/hot pattern、working-set bytes、repeat | cycle/store, byte/clk/SM | validated | sparse producer, WGMMA staging |
| memory | [dsm_store](memory/dsm_store/) | [128b_cluster2_sm90](memory/dsm_store/128b_cluster2_sm90/) | 128-bit peer store, cluster=2 | local/peer、repeat | cycle/store, byte/clk/cluster | validated | sparse cluster2 crossover |
| memory | [tma_load](memory/tma_load/) | [tile64x64_bf16_sm90](memory/tma_load/tile64x64_bf16_sm90/) | GMEM→SMEM, BF16 64×64；rank 2/3/4 | rank（默认 4）、depth、working set、repeat | cycle/tile, transaction/clk/SM, byte/clk/SM | validated | dense decode K（rank 4）；rank 2/3 controls |
| memory | [tma_load](memory/tma_load/) | [tile64x576_bf16_sm90](memory/tma_load/tile64x576_bf16_sm90/) | GMEM→SMEM, BF16 64×576；rank 2/3/4 | rank（默认 4）、depth、working set、repeat | cycle/tile, transaction/clk/SM, byte/clk/SM | validated | dense/sparse decode Q（rank 4）；sparse prefill Q（rank 3）；rank 2 control |
| memory | [tma_store](memory/tma_store/) | [tile64x512_bf16_2d_sm90](memory/tma_store/tile64x512_bf16_2d_sm90/) | SMEM→GMEM, BF16 64×512, rank 2 | depth、working set、repeat | cycle/tile, transaction/clk/SM, byte/clk/SM | validated | rank-control / generic comparison |
| memory | [tma_store](memory/tma_store/) | [tile64x64_bf16_3d_sm90](memory/tma_store/tile64x64_bf16_3d_sm90/) | SMEM→GMEM, BF16 64×64, rank 3 | depth、working set、repeat | cycle/tile, transaction/clk/SM, byte/clk/SM | validated | sparse prefill epilogue |
| memory | [tma_store](memory/tma_store/) | [tile64x512_bf16_4d_sm90](memory/tma_store/tile64x512_bf16_4d_sm90/) | SMEM→GMEM, BF16 64×512, rank 4 | depth、working set、repeat | cycle/tile, transaction/clk/SM, byte/clk/SM | validated | dense decode epilogue |
| memory | [tma_store](memory/tma_store/) | [tile64x512_bf16_5d_sm90](memory/tma_store/tile64x512_bf16_5d_sm90/) | SMEM→GMEM, BF16 64×512, rank 5 | depth、working set、repeat | cycle/tile, transaction/clk/SM, byte/clk/SM | validated | sparse decode epilogue |
| memory | [bulk_store](memory/bulk_store/) | [tile64x512_f32_sm90](memory/bulk_store/tile64x512_f32_sm90/) | SMEM→GMEM bulk, FP32 64×512 | pattern、working set、CTA count、repeat；completion depth=1 | cycle/tile, cycle/row-store, byte/clk/SM | validated | dense/sparse split main epilogue |
| memory | [stmatrix](memory/stmatrix/) | [m64n64_b16_x4_sm90](memory/stmatrix/m64n64_b16_x4_sm90/) | register→SMEM, B16 64×64, x4 | fence、repeat | cycle/tile, cycle/warp-instruction, byte/clk/SM | validated | dense score exchange and sparse prefill score handoff |
| memory | [stmatrix](memory/stmatrix/) | [m64n256_b16_x4_sm90](memory/stmatrix/m64n256_b16_x4_sm90/) | register→SMEM, B16 64×256, x4 | fence、repeat | cycle/tile, cycle/warp-instruction, byte/clk/SM | validated | decode/prefill output staging |
| memory | [splitkv_reduce](memory/splitkv_reduce/) | [dv512_f32_sm90](memory/splitkv_reduce/dv512_f32_sm90/) | FP32 partial, D_V=512 | num_splits、working set、pattern、CTA count | cycle/CTA、cycle/row、byte/clk/SM | validated | split-KV combine |

`Status` 只允许以下三种值：

- `scaffold`：核心实现、计数口径或静态证据仍不完整，不能作为性能输入。
- `validated`：SM90a 编译、目标 PTX/SASS opcode、源码动态计数、`%clock64` 和 register/spill 边界已核验，但没有本地运行或 H800 性能数据。证据汇总见 [`static-validation.md`](static-validation.md)。
- `measured`：已在远端 H800 的配置内 `result/runs/<run_id>/` 保存 GPU/时钟策略、完整 CLI、warmup/sample、规范化 JSONL、完整 log 和正确性结果，且 leaf README 已链接 accepted run；只有此状态可用于定量性能结论。
