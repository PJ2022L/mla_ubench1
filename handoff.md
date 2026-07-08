# Handoff — FlashMLA µbench (SM90/H800)

交接文档。读完这一页即可接手。路由见 [AGENT.md](AGENT.md)。

---

## 1. 现在是什么状态

- **脚手架完成，未编译、未运行**。目录/文档/接口/每个原子的 ref_ubench 风格骨架 + Makefile 就位；核心操作体是 `// TODO` 占位。
- 只针对 **SM90 / H800**（`sm_90a`）。B200(SM100) 之后再加 `atoms_sm100/`，复用同一组合模型。
- 目标算子：**FlashMLA sparse FP8 decode（DeepSeek-V3.2 / `MODEL_TYPE::V32`）**，源 `target_op/FlashMLA/csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh`。

## 2. 方法（三步）

1. **拆**：把 decode 主循环拆成 **8 个完全独立的原子 kernel**（不保留原 barrier/warp 专用化）。
2. **测**：用 `ref/ubench` 的 **clock-cycle 单 SM** 范式测每个原子的周期 `T_x`（`%%clock` bracket、链式依赖防优化、`flop|byte/clk/SM`）。
3. **建模**：用重叠感知解析式 `max/+/×` 组合出融合 kernel 预测，对拍端到端实测得 `η` 与瓶颈。核心式（[docs/05](mla_ubench/docs/05-sm90-sparse-decode-model.md)）：
   ```
   T_consumer = 36·T_wgmma_qk + T_softmax + 4·T_wgmma_pv   # WG0 关键路径
   T_producer = T_prod_block                                # WG2
   T_block    = max(T_producer, T_consumer)                 # 双缓冲重叠
   T_fused    = T_prologue + (n_block−1)·T_block + T_epilogue
   ```

## 3. 已核实的关键事实（省你重读源码）

- **3 个 warpgroup**（不是参考图 `docs/mla.png` 的 2 个）：WG0 consumer 主（QK→softmax→PV 左半）、WG1 consumer 副（remote-PV 右半，基本被 WG0 掩盖）、WG2 producer（load+dequant+DSM）。
- **WGMMA 发射次数**：`gemm()`（`components/helpers.h`）手动展开 K 模 → **QK=36** 条 `m64n64k16`(576/16)、**PV=4** 条 `m64n256k16`(64/16)。这就是 `N×T_wgmma`。
- **形状**：`d_qk=576(=512 NoPE+64 RoPE)`, `d_v=512`, `BLOCK_M=TOPK_BLOCK=64`, FP8 KV **656 B/token**, `NUM_K_BUFS=2`。
- **DSM crossover**：`NUM_HEADS=128`→`CLUSTER_SIZE=2`，每 CTA 只反量化 `TOPK/2=32` token 并 `st.async` 到 peer；`NUM_HEADS=64`→无 crossover（A/B 基线）。
- **Roofline**：`h_q·s_q≥128` 时 compute-bound（生产 h_q=128）；H800 3.35 TB/s、~865 TFLOPS(throttled)；实测 sparse decode ~410 TFLOPS。

## 4. 8 个原子 ↔ 硬件维度 ↔ 目录

| 原子 | 维度 | 目录 | 纯操作 | 计量 |
|---|---|---|---|---|
| A1 kv_gather | HBM | `atoms/a1_kv_gather` | `cp.async.cg.L2::256B` gather 656B/token | byte/clk, GB/s |
| A2 dequant | Smem+SFU | `atoms/a2_dequant` | `cvt_fp8x8_bf16x8` 16 cvt/token | token/clk |
| A3 dsm_crossover | Smem(DSM) | `atoms/a3_dsm_crossover` | `st.async.weak.shared::cluster` 到 peer | byte/clk |
| A4 qk_gemm | TensorCore | `atoms/a4_qk_gemm` | 36× `wgmma m64n64k16` | flop/clk/SM |
| A5 pv_gemm | TensorCore | `atoms/a5_pv_gemm` | 4× `wgmma m64n256k16` | flop/clk/SM |
| A6 softmax | SFU | `atoms/a6_softmax` | max(`shfl`)+`exp2f`+rescale | cycle/block |
| A7 tma_store | HBM | `atoms/a7_tma_store` | `STSM`+`TMA_STORE_5D` 写 O | GB/s |
| A8 combine | HBM+FPU | `atoms/a8_combine` | LSE rescale + float4 累加 | GB/s |

## 5. 下一步（按顺序，每步有验证点）

1. **打通 `common/`**：实现 `clock.cuh`(`%%clock`+NVML `getGPUClock`)、`mla_shapes.h`(数据生成)。→ 验证：编译通过。
2. **A4 qk_gemm 先行**：用它验证计时框架（对齐 `ref/ubench/.../MaxFlops` 的 `flop/clk/SM` 输出）。→ 验证：ncu `tensor active% > 70%`。
3. **A1–A8 逐个实现**，每个跑 ncu 隔离验证（只压目标维度，见 [docs/02](mla_ubench/docs/02-methodology.md) 表）。
4. **e2e_decode 基线**：构造 `SparseAttnDecodeParams`，调真实 `run_flash_splitkv_mla_fp8_sparse_kernel<V32,128>`，复现 ~410 TFLOPS。
5. **`model/compose.py` 组合**：读各原子 log → `T_fused` → 对拍 e2e 得 `η`、瓶颈。
6. **DSM 消融**：A2-full vs A2-half+A3，量化 crossover 收益。

## 6. 坑 / 注意

- **每原子自带 Makefile**（`atoms/aN_*/Makefile`，`include ../../common/common.mk`）；单跑 `make -C atoms/a4_qk_gemm run`。
- `common.mk` 的 `FLASHMLA ?= ../../../target_op/FlashMLA` —— FlashMLA 在 `target_op/` 下，此相对路径已对应。还需补 **CUTLASS/CuTe include**（FlashMLA 的 submodule，`git submodule update --init`）。
- **e2e 依赖 FlashMLA 的 kernel 实例化**（`.cu`），或链接已 `pip install` 的 `flash_mla.cuda`；见 `e2e/Makefile` 的 TODO。
- **ncu 指标名随版本变**，首次 `ncu --query-metrics` 核对；H800 会 throttle，对比时锁频或记录 `sm__cycles_elapsed`。
- **原子独立后行为可能与融合中不同**（缓存/寄存器状态）——`η` 就是量化这部分；不必追求 `η=1`，落 [0.7,1.0] 合理。
- 单 SM cycle 测稳态吞吐，**忽略 SM 间 HBM 争用**；A1/A7/A8 建议再做多 block 版（`ref/ubench/.../NewFeatures/TMA/Throughput` 写法）。

## 7. 参考锚点

- 精细化模型 + 算子图：[mla_ubench/docs/05-sm90-sparse-decode-model.md](mla_ubench/docs/05-sm90-sparse-decode-model.md)
- 完整方法论 plan：`~/.claude/plans/llm-micro-benchmark-gpu-cmodel-4-hbm-ten-enumerated-perlis.md`
- 目标 kernel：`target_op/FlashMLA/csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh`、`components/{dequant,helpers}.h`、`config.h`
- FlashMLA deep-dive：`target_op/FlashMLA/docs/20250422-*.md`、`20250929-hopper-fp8-sparse-deep-dive.md`
- 测量范式源：`ref/ubench/NVIDIA-Hopper-Benchmark/{RegularUnits/MaxFlops,shared_bw,NewFeatures/TMA,NewFeatures/DSM}`
