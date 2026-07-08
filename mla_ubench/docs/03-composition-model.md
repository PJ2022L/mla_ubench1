# 03 — 组合模型（重叠感知解析式）

> 本文是**通用**组合模型的方法说明。**精确到 SM90 sparse FP8 decode 的实例**（每 block 精确 WGMMA/dequant 计数、`max/+/×` 展开、算子图 mermaid）见 **`05-sm90-sparse-decode-model.md`**。

## 目标

用单测原子周期 `T_i` **预测**融合 kernel 的时间 `T_fused`，再与端到端实测对拍。这既验证「原子拆得对不对」，又定位「瓶颈在谁、overlap 效率多少」。

## 融合 kernel 的真实结构

decode 是**双缓冲生产者-消费者流水**（doc 01）。每处理一个 KV block：

```
生产者(WG2) 内部串行:  T_producer = T_A1(gather) + T_A2(dequant) + T_A3(dsm)
消费者(WG0/1) 内部串行: T_consumer = T_A4(qk) + T_A6(softmax) + T_A5(pv)
```

> 精确展开（见 doc 05）：`T_consumer = 36×T_wgmma_qk + T_softmax + 4×T_wgmma_pv`（这就是「MNK 大→多次 wgmma」的 `N×T` 结构）；WG1 的 remote-PV（4×T_wgmma_pv）在独立 warpgroup 上，基本被 WG0 关键路径掩盖，故不额外计入。

生产者和消费者在**不同 warpgroup 上并行**，双缓冲让 producer(block i+1) 与 consumer(block i) 重叠：

```
T_block = max(T_producer, T_consumer)          # 重叠 → 取 max
```

整 kernel（`num_blocks = topk / TOPK_BLOCK` 个 block）：

```
T_fused ≈ T_prologue                           # 首个 producer，无可重叠对象
        + (num_blocks - 1) * T_block           # 稳态流水
        + T_epilogue                           # A7 store + A8 combine
```

其中 `T_prologue = T_producer`，`T_epilogue = T_A7 + T_A8`。

## 上下界

- **下界（理想重叠）**：`max(T_producer, T_consumer)+tail`（上式）。
- **上界（全串行）**：`ΣT_i`（producer 与 consumer 不重叠）。
- 实测 `T_measured` 应落在两者之间。位置本身即诊断：越靠下界 = overlap 越好。

## 标定

```
η = T_model / T_measured        # overlap 效率，理想 ≈ 1
```
- `η ≈ 1`：模型准，流水打满，原子拆解可信。
- `η < 1`：实测比预测慢 → 有模型没算进的同步/尾部/缓存代价（= 「其他/成熟度」损失）。
- `η > 1`：模型高估某原子（可能原子独立时缓存/寄存器状态比融合中更差），需回查该原子测法。

## 瓶颈归因

```
if T_producer > T_consumer:  bottleneck = memory/dequant-bound   # A1/A2 主导
else:                        bottleneck = compute-bound          # A4/A5 主导
```
对齐 deep-dive：h_q·s_q≥128 时应是 compute-bound（consumer 主导）；但 FP8 dequant ~50cyc/token 可能把 producer 抬起来。模型直接给出是哪种。

## DSM crossover 的价值（A2/A3 消融）

crossover 让每 CTA 只反量化一半 token。对比：
```
无 crossover:  T_producer = T_A1 + T_A2(full)          # 反量化全部 token
有 crossover:  T_producer = T_A1 + T_A2(half) + T_A3    # 反量化一半 + DSM 互换
```
若 `T_A2(half)+T_A3 < T_A2(full)` → crossover 净赚，量化省了多少周期，验证 deep-dive「dequant-bound 靠 crossover 缓解」的说法。

## 实现

`model/compose.py`：读 `atoms/*/log` 的周期 → 换算到同一 block 粒度 → `compose()` 算 T_fused → 读 `e2e/log` 的 T_measured → 输出 η + 瓶颈 + 上下界 + DSM 消融。

下一步：`04-attribution.md`（把结果读成结论）。
