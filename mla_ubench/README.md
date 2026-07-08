# mla_ubench — FlashMLA 原子拆解 + 组合建模 Micro-Benchmark

把 **FlashMLA sparse FP8 decode（DeepSeek Sparse Attention）** 主循环拆成 **8 个完全独立的原子 kernel**，用 `../ref/ubench` 的 **clock-cycle 单 SM** 范式测每个原子的硬件能力上限，再用**重叠感知解析式** `T_fused ≈ max(T_producer,T_consumer)+tail` 组合出融合 kernel 预测、与端到端实测对拍，定位瓶颈与 overlap 效率。

> **当前状态：脚手架（scaffold）**。目录、文档、每个原子的 ref_ubench 风格骨架（clock 计时框架就位、纯操作 body 为 `// TODO`）、Makefile、common helper、组合模型脚本已就位；**尚不能编译/运行**。
> **本阶段只针对 H800(SM90/Hopper)**；B200(SM100) 之后再加 `atoms_sm100/`。

---

## 8 个原子 ↔ 4 个硬件维度

| 原子 | 维度 | 纯操作（去掉所有原 barrier） |
|---|---|---|
| `a1_kv_gather` | HBM 带宽 | `cp.async.cg.L2::256B` gather 656B/token |
| `a2_dequant` | Smem+SFU | `cvt_fp8x8_bf16x8`(e4m3→bf16 ×scale) 写 smem |
| `a3_dsm_crossover` | Smem(DSM) | `st.async.weak.shared::cluster` 分发到 peer CTA |
| `a4_qk_gemm` | Tensor Core | `GMMA::MMA_64x64x16` (Q·Kᵀ) |
| `a5_pv_gemm` | Tensor Core | `MMA_64x256x16` (P·V) |
| `a6_softmax` | SFU | max(`shfl_xor`)+`exp2f`+rescale |
| `a7_tma_store` | HBM 带宽 | `STSM`→`SM90_TMA_STORE` 写 O |
| `a8_combine` | HBM+FPU | LSE rescale + float4 累加 |

**独立**：每个原子不保留 FlashMLA 的 barrier / warpgroup 专用化，只复用指令级封装。重叠由组合模型显式建模。

---

## 目录

```
mla_ubench/
  docs/        方法论（先读这里）
    00-overview  01-atoms  02-methodology  03-composition-model  04-attribution
    05-sm90-sparse-decode-model  ← ★ 精细化计算过程 + 算子图(mermaid) + max/+/× 建模
  common/      ref_ubench 复用的 helper：clock.cuh / measure.hpp / tma_util.cuh / mla_shapes.h / common.mk
  atoms/       8 个独立原子，每个 <atom>.cu + Makefile
  e2e/         端到端基线（调真实 FlashMLA launcher），用于标定
  model/       compose.py — 组合原子时间预测 T_fused、对拍 e2e
  scripts/     run_all.sh / summarize.py
```

复用 `../target_op/FlashMLA/csrc` 与 `../ref/ubench/NVIDIA-Hopper-Benchmark` 的头文件（不改原码）。

---

## 快速开始（实现完成后，H800）

```bash
# 单个原子（ref_ubench 惯例：每目录一个 Makefile）
make -C atoms/a4_qk_gemm run        # 输出 flop/clk/SM

# 全部原子 + e2e + 组合模型
bash scripts/run_all.sh             # 编译跑所有原子 → compose.py → summarize.py

# 组合模型对拍
python model/compose.py --atoms-dir atoms --e2e-log e2e/log
```

---

## 组合模型（一句话）

```
T_block  = max( T_A1+T_A2+T_A3 ,  T_A4+T_A6+T_A5 )   # producer ∥ consumer 双缓冲重叠
T_fused ≈ T_prologue + (num_blocks-1)*T_block + (T_A7+T_A8)
η       = T_fused / T_measured(e2e)                   # overlap 效率
```
瓶颈 = producer(memory/dequant) 还是 consumer(compute)。详见 `docs/03-composition-model.md`。

完整方法论见 plan：`~/.claude/plans/llm-micro-benchmark-gpu-cmodel-4-hbm-ten-enumerated-perlis.md`。
