# 04 — 结果解读 / 归因（H800 阶段）

本阶段（H800）产出三层结论。将来加 B200 时，同一套原子上限 + 组合模型可直接做跨架构对比。

## 层 1 — 原子硬件能力上限（`report/atom_limits.csv`）

每个原子给出该硬件维度在 H800 上的**干净上限**：

| 原子 | 维度 | 上限（示例单位） | 对照 spec |
|---|---|---|---|
| A1 kv_gather | HBM_BW | GB/s | vs 3.35 TB/s（H800 HBM3 峰值） |
| A4 qk_gemm / A5 pv_gemm | TensorCore | TFLOPS(bf16) | vs ~865 TFLOPS（throttled 峰值） |
| A2 dequant | SFU/Smem | token/clk | vs deep-dive ~50 cyc/token |
| A3 dsm_crossover | DSM | GB/s | — |
| A6 softmax | SFU | cycle/softmax | — |
| A7/A8 | HBM_BW | GB/s | vs 3.35 TB/s |

用途：知道每种硬件能力**单独**能跑多满（利用率 = 实测/spec）。

## 层 2 — 组合模型对拍（`report/model.csv`）

| 量 | 含义 |
|---|---|
| `T_producer`, `T_consumer` | 生产/消费链各自周期 |
| `T_block = max(…)` | 单 block 稳态周期 |
| `T_fused`(pred) vs `T_measured`(e2e) | 模型预测 vs 端到端实测 |
| `η = T_model/T_measured` | overlap 效率 |
| 瓶颈 | producer(memory/dequant) 还是 consumer(compute) |

**结论模板**：
> 「H800 上该 decode 是 **{compute\|memory\|dequant}-bound**（{consumer\|producer} 主导），overlap 效率 η={…}。瓶颈原子是 {AX}，其硬件利用率 {…}%。距峰值的差距 = {原子未打满} + {overlap 缺口 1-η}。」

## 层 3 — DSM crossover 收益（`report/bottleneck.png` + 消融）

A2-full vs A2-half+A3：量化 crossover 省下的 producer 周期，验证它是否真的把 dequant-bound 缓解到 consumer 之下。

## 与 4 个硬件维度的对应（交付给 cmodel 验证）

| 维度 | 上限来自 | cmodel 该关注 |
|---|---|---|
| HBM 带宽 | A1/A7/A8 | gather/store 的持续带宽 |
| Tensor Core | A4/A5 | WGMMA 稳态吞吐 |
| Smem 带宽 | A2/A3 | 反量化写 + DSM 分发 |
| L2 命中率 | A1 参数扫描 | 稀疏 gather 下的命中曲线 |

## later：B200 阶段怎么接

1. 加 `atoms_sm100/`：A4/A5→UMMA、A2→native FP8 直转、A1→TMA gather4、A3→（若 native 转换则可能不需要）。
2. 每个原子 H800 vs B200 上限比 → 硬件规模 vs 新特性归因。
3. 同一 `compose.py` 模型算 B200 的 η；端到端 350<410 的缺口应落在「η 低（软件不成熟）」而非原子上限低 → 证明「硬件更强、软件没跟上」。

方法细节见 plan：`~/.claude/plans/llm-micro-benchmark-gpu-cmodel-4-hbm-ten-enumerated-perlis.md`。
