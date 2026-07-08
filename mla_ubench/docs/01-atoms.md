# 01 — 8 个原子（独立、纯操作）

## 原则

每个原子是一个**完全独立的 kernel**，只做一种操作，**不保留** FlashMLA 原 kernel 的任何东西：
- ❌ 不保留 producer/consumer 的 NamedBarrier、warpgroup 专用化、cluster 同步、双缓冲流水。
- ✅ 只复用**指令级封装**（`cvt_fp8x8_bf16x8`、`GMMA::MMA_*`、`st_async_128b`、`SM90_TMA_*`），保证指令保真。
- 输入是**合成数据**（随机值，形状真实），预置到 reg/smem/HBM。
- 计时只用 2 个 bracket `bar.sync`（ref_ubench 范式），不是原 kernel 的同步。

**为什么要真独立**：原位 stub 测到的是「该操作 + 它被其他 warpgroup 拖累/等待」的时间。要得到该硬件能力的**干净上限**，必须剥离所有跨-warpgroup 依赖，单独测。overlap 由组合模型（doc 03）显式建模，而非隐含在原子里。

## FlashMLA decode 主循环结构（已核实）

源：`target_op/FlashMLA/csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh`
- **WG2 producer**（每 block）：gather FP8 → dequant → DSM crossover 到 peer CTA。
- **WG0 consumer**：QK gemm → `scale_softmax` → PV gemm(local) → store。
- **WG1 consumer**：等 sS → PV gemm(remote) → store。
- 跨 block 双缓冲：producer(block i+1) 与 consumer(block i) 重叠。

## 8 原子

| 原子 | 目录 | 主维度 | 纯操作 | 真实指令来源 | 计量 |
|---|---|---|---|---|---|
| **A1 kv_gather** | `atoms/a1_kv_gather` | HBM_BW | index→addr + `cp.async.cg.L2::256B` 拉 656B/token 进 smem | WG2 前半 | byte/clk/SM |
| **A2 dequant** | `atoms/a2_dequant` | SFU+Smem | `cvt_fp8x8_bf16x8`(e4m3→bf16 ×scale) 写 K-major smem | `components/dequant.h` | token/clk, smem byte/clk |
| **A3 dsm_crossover** | `atoms/a3_dsm_crossover` | Smem(DSM) | `st.async.weak.shared::cluster` 分发到 peer CTA | WG2 后半 | DSM byte/clk |
| **A4 qk_gemm** | `atoms/a4_qk_gemm` | TensorCore | `GMMA::MMA_64x64x16_F32BF16BF16_SS`，K=576/9 tile | `gemm<true>` | flop/clk/SM |
| **A5 pv_gemm** | `atoms/a5_pv_gemm` | TensorCore | `MMA_64x256x16_F32BF16BF16_RS/SS`，O_L/O_R | `gemm<false>` | flop/clk/SM |
| **A6 softmax** | `atoms/a6_softmax` | SFU | rowwise max(`shfl_xor`)+`exp2f`+rescale | `scale_softmax` | cycle/softmax |
| **A7 tma_store** | `atoms/a7_tma_store` | HBM_BW | `STSM`→smem→`SM90_TMA_STORE` 写 O | `store_o` | byte/clk/SM |
| **A8 combine** | `atoms/a8_combine` | HBM_BW+FPU | LSE rescale + float4 累加 | `smxx/combine/combine.cu` | byte/clk/SM |

## 映射到 4 个硬件维度

| 维度 | 主原子 | 佐证 |
|---|---|---|
| HBM 带宽 | A1, A7, A8 | — |
| Tensor Core | A4, A5 | — |
| Smem 带宽 | A2, A3 | A4/A5 操作数读 |
| L2 命中率 | A1（扫 index 分布/topk） | A8 |
| （SFU） | A6, A2 | — |

> L2 命中率不是单独原子，而是 **A1 的参数扫描**（topk × index 分布 × cache hint，让 working-set 跨越 L2 容量）。见 doc 02 的参数扫描。

下一步：`02-methodology.md`（ref_ubench 怎么测每个原子）。
