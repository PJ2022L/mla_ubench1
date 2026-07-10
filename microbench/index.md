# SM90 micro-benchmark index

这是新增/复用 micro-benchmark 的唯一入口。分析新 operator path 时，**先查本表，再写 `02-atom-decomposition.md`**。

## 去重与新增流程

1. 用 `family + operand mode + shape + dtype + arch` 查找精确配置。
2. 精确匹配时直接复用已有结果；不得按 operator 名再复制一份。
3. 指令与数据路径相同、只有参数不同：在现有 family 下添加配置叶子，并复用 family 的 `benchmark.cu`。
4. 只有指令族、operand path 或测量边界确实不同，才新建 family。
5. 新配置必须先登记本表，再被 operator 文档引用；`scaffold` 不能作为实测数据。

## Registry

| Category | Family | Configuration | Operand/shape | Independent variables | Outputs | Status | Used by |
|---|---|---|---|---|---|---|---|
| compute | [wgmma](compute/wgmma/) | [m64n64k16_bf16_rs_ss_sm90](compute/wgmma/m64n64k16_bf16_rs_ss_sm90/) | RS、SS 分开；64×64×16 BF16 | mode、issue depth、repeat、resident WG | cycle/inst, inst/clk/SM, flop/clk/SM | scaffold | dense decode, sparse decode QK |
| compute | [wgmma](compute/wgmma/) | [m64n256k16_bf16_rs_ss_sm90](compute/wgmma/m64n256k16_bf16_rs_ss_sm90/) | RS、SS 分开；64×256×16 BF16 | mode、issue depth、repeat、resident WG | cycle/inst, inst/clk/SM, flop/clk/SM | scaffold | dense/sparse decode PV |
| compute | [convert](compute/convert/) | [fp8x8_to_bf16x8_sm90](compute/convert/fp8x8_to_bf16x8_sm90/) | FP8 e4m3×8 + scale → BF16×8 | repeat、input distribution | cycle/cvt, element/clk/SM | scaffold | sparse FP8 decode/prefill |
| compute | [softmax](compute/softmax/) | [online_m64n64_exp2_shfl_sm90](compute/softmax/online_m64n64_exp2_shfl_sm90/) | online softmax 64×64 | local/shared state、repeat | cycle/tile, exp2 element/clk/SM | scaffold | dense/sparse decode |
| memory | [global_load](memory/global_load/) | [128b_nc_l2_sm90](memory/global_load/128b_nc_l2_sm90/) | 128-bit non-coherent load | access pattern、working set、L2 hint | cycle/load, byte/clk/SM, L2 hit | scaffold | sparse decode/prefill gather |
| memory | [shared_store](memory/shared_store/) | [128b_sm90](memory/shared_store/128b_sm90/) | 128-bit CTA shared store | swizzle、bank pattern、repeat | cycle/store, byte/clk/SM | scaffold | sparse producer, WGMMA staging |
| memory | [dsm_store](memory/dsm_store/) | [128b_cluster2_sm90](memory/dsm_store/128b_cluster2_sm90/) | 128-bit peer store, cluster=2 | local/peer、repeat、resident cluster | cycle/store, byte/clk/cluster | scaffold | sparse cluster2 crossover |
| memory | [tma_load](memory/tma_load/) | [tile64x64_bf16_sm90](memory/tma_load/tile64x64_bf16_sm90/) | GMEM→SMEM, BF16 64×64 | swizzle、cache hint、depth、CTA count | cycle/tile, byte/clk/SM, GB/s | scaffold | dense decode K tile |
| memory | [tma_load](memory/tma_load/) | [tile64x576_bf16_sm90](memory/tma_load/tile64x576_bf16_sm90/) | GMEM→SMEM, BF16 64×576 | swizzle、cache hint、depth、CTA count | cycle/tile, byte/clk/SM, GB/s | scaffold | decode Q prologue |
| memory | [tma_store](memory/tma_store/) | [tile64x512_bf16_2d_sm90](memory/tma_store/tile64x512_bf16_2d_sm90/) | SMEM→GMEM, BF16 64×512, rank 2 | depth、CTA count、cache state | cycle/tile, byte/clk/SM, GB/s | scaffold | dense decode epilogue |
| memory | [tma_store](memory/tma_store/) | [tile64x512_bf16_5d_sm90](memory/tma_store/tile64x512_bf16_5d_sm90/) | SMEM→GMEM, BF16 64×512, rank 5 | depth、CTA count、cache state | cycle/tile, byte/clk/SM, GB/s | scaffold | sparse decode epilogue |
| memory | [stmatrix](memory/stmatrix/) | [m64n64_b16_x4_sm90](memory/stmatrix/m64n64_b16_x4_sm90/) | register→SMEM, B16 64×64, x4 | swizzle、repeat | cycle/tile, byte/clk/SM, bank conflict | scaffold | dense P exchange |
| memory | [stmatrix](memory/stmatrix/) | [m64n256_b16_x4_sm90](memory/stmatrix/m64n256_b16_x4_sm90/) | register→SMEM, B16 64×256, x4 | swizzle、repeat | cycle/tile, byte/clk/SM, bank conflict | scaffold | decode output staging |
| memory | [splitkv_reduce](memory/splitkv_reduce/) | [dv512_f32_sm90](memory/splitkv_reduce/dv512_f32_sm90/) | FP32 partial, D_V=512 | num_splits、CTA count | cycle/row, byte/clk/SM, GB/s | scaffold | split-KV combine |

`Status` 只允许 `scaffold`、`validated`、`measured`。改为 `validated` 前必须保存 PTX/SASS 核验；改为 `measured` 前必须记录 SM90 GPU、时钟策略、warmup、样本统计和原始日志。
