# AGENT.md — 仓库路由

> FlashMLA 算子拆解 + micro-benchmark。目标：把 **FlashMLA sparse FP8 decode（SM90/H800）** 拆成独立原子 kernel，实测每个硬件能力上限，用 `max/+/×` 组合模型预测融合 kernel 并对拍。**当前状态：脚手架，未编译**。先看 [handoff.md](handoff.md)。

## 目录路由

| 路径 | 是什么 | 何时进去 |
|---|---|---|
| **[mla_ubench/](mla_ubench/)** | ★ 本项目（唯一要写的代码） | 所有开发在此 |
| [target_op/FlashMLA/](target_op/FlashMLA/) | 上游目标库（只读，`#include` 其头文件） | 查真实 kernel 指令/形状 |
| [ref/papers/](ref/papers/) | 参考论文（DeepSeek / FlashAttention / Hopper 微架构 / GPU Power） | 查算法/roofline 依据 |
| [ref/ubench/](ref/ubench/) | 参考 benchmark（NVIDIA-Hopper-Benchmark 等） | 抄测量范式 |

## 任务 → 去哪

- **理解方法论/背景** → [mla_ubench/docs/00-overview.md](mla_ubench/docs/00-overview.md)
- **精细化计算过程 + 算子图 + `max/+/×` 建模** → [mla_ubench/docs/05-sm90-sparse-decode-model.md](mla_ubench/docs/05-sm90-sparse-decode-model.md) ★核心
- **写/改某个原子 kernel** → [mla_ubench/atoms/](mla_ubench/atoms/)（`aN_*/`：`.cu` + `Makefile`）＋范式见 [docs/02-methodology.md](mla_ubench/docs/02-methodology.md)
- **公共计时/数据/形状 helper** → [mla_ubench/common/](mla_ubench/common/)（`clock.cuh` / `measure.hpp` / `tma_util.cuh` / `mla_shapes.h`）
- **组合模型 / 对拍脚本** → [mla_ubench/model/compose.py](mla_ubench/model/compose.py)
- **端到端基线** → [mla_ubench/e2e/](mla_ubench/e2e/)
- **跑全套 / 汇总** → [mla_ubench/scripts/](mla_ubench/scripts/)（`run_all.sh` / `summarize.py`）

## 关键约束

- 只针对 **SM90 / H800**（`sm_90a`）；B200 later。
- 原子必须**独立**：除计时 `bar.sync` 外，不保留 FlashMLA 的 barrier / warp 专用化。
- 复用 FlashMLA **指令级封装**（`cvt_fp8x8_bf16x8` / `GMMA::MMA_*` / `st_async_128b`），不改上游代码。
- 完整方法论见 plan：`~/.claude/plans/llm-micro-benchmark-gpu-cmodel-4-hbm-ten-enumerated-perlis.md`。
