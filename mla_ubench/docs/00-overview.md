# 00 — 总览：目标、4 维度、Roofline

## 为什么做拆解 + 建模

自研 GPU 在 cmodel 上要用**真实 LLM 负载**验证 4 个硬件维度：**HBM 带宽 / Tensor Core 吞吐 / L2 命中率 / Smem 带宽**。直接跑整个 FlashMLA 端到端只得到一个**混合数字**，无法回答：
- 是哪一块硬件能力决定了这个算子的性能？
- 各能力如何拼装成融合 kernel（谁和谁重叠、谁是瓶颈）？

所以采用**「独立原子 + 组合建模」**：
1. 把 decode 主循环拆成 **8 个完全独立的原子 kernel**，每个只跑一种操作、**不保留原 kernel 的任何 barrier / warp 专用化**（否则测的是被同步拖累后的时间，非纯硬件能力）。用 `../ref/ubench` 的 **clock-cycle 单 SM** 范式测每个原子的周期 `T_i`。
2. 用**重叠感知解析式** `T_fused ≈ max(T_producer, T_consumer) + tail` 把原子时间组合成融合 kernel 预测，与端到端实测对拍，得 overlap 效率 η 与瓶颈归因。

> **本阶段只在 H800(SM90/Hopper) 跑通**，B200(SM100) 之后再做（届时加 `atoms_sm100/`，复用同一组合模型）。

## 为什么锚定 sparse FP8 decode

FlashMLA 有多个 kernel。锚定 **sparse FP8 decode（DSA）** 因为它是 DeepSeek-V3.2 的生产解码路径，且**同时存在于 SM90/SM100**（本阶段先做 SM90，将来能做同-kernel 跨架构对比）。

| Kernel | H800 (SM90) | B200 (SM100) |
|---|---|---|
| Dense decode (BF16) | ✅ | ❌ |
| **Sparse decode (FP8/DSA)** | ✅ ← 本阶段锚点 | ✅ (later) |
| Dense prefill (MHA) | ❌ | ✅ |
| Sparse prefill (DSA) | ✅ | ✅ |

## 算子形状（MODEL_TYPE::V32）

- MQA：`h_q=128, h_kv=1`；`s_q=1`（关闭 MTP）或 `2`（开 MTP/投机解码）。
- `head_dim_k (d_qk) = 576 = 512 NoPE(latent) + 64 RoPE`；`head_dim_v = 512`。
- **FP8 KV cache 每 token 656 B** = `512×fp8_e4m3(NoPE) + 4×fp32(scale) + 64×bf16(RoPE)`；`QUANT_TILE_SIZE=128`（每 128 fp8 共享一个 fp32 scale）。
- 稀疏：每个 q 用 `topk` 个 index 从 KV cache **gather**（`indices` 张量，`-1` 表无效）。

计算流程：`P = Q·gather(KV)ᵀ · scale → softmax → O = P·gather(V)`，K 与 V 是同一份 latent 的不同切片（MLA 特性）。

## Roofline（H800，来自 FlashMLA deep-dive）

- compute/mem 比 ≈ `2·h_q·s_q`；当 `h_q·s_q ≥ 128` 时 **compute-bound**（生产配置 h_q=128 即是）。
- H800 峰值 3.35 TB/s、~990 TFLOPS（throttle 到 ~865）。
- 实测：dense decode **660 TFLOPS / 3 TB·s**；sparse FP8 decode **410 TFLOPS on H800**。

## ⚠️ 核心洞察（供 later B200 阶段）：端到端会误导

FlashMLA README：B200 sparse decode 仅 ~350 TFLOPS（vs H800 410），"not really optimized yet"。端到端一个数字会误判成「B200 不行」，而**原子级**能看出 B200 硬件更强、只是软件没跟上。本阶段先在 H800 建立原子上限 + 组合模型基线，为将来做这个区分打底。

## FP8 反量化瓶颈（Hopper 特有，A2/A3 要量化）

H800 无法直接 `e4m3→bf16`，需 4 步（f8→half→f32→bf16→×scale），每 token ~50 cycle > MMA 的 34 cycle → 可能 **dequant-bound**。Hopper 用 **DSM/cluster "crossover"**（2 CTA 各反量化一半，经 `st.async.weak.shared::cluster` 互换）把每 CTA 反量化量减半。→ 原子 A2(dequant) 与 A3(dsm) 分开测，A2-full vs A2-half+A3 的对照即量化 crossover 收益。

## 下一步

看 `01-atoms.md`：8 个原子的纯操作定义与真实指令来源。
