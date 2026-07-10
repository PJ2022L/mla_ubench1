# 03 — Reconstructing e2e performance

模型只使用三种组合语义：

- `A + B`：B 对 A 有数据或同步依赖，串行相加。
- `max(A, B)`：A/B 在不同 WG/管线并行，慢者决定阶段完成时间。
- `N × T`：同一指令或 loop body 重复 N 次。

所有输入先归一成 `cycle / CTA / 64-token block`；不同粒度不能直接相加。

## 1. Build per-block atoms

microbench 可直接报告 block geometry 的时间，也可由单指令吞吐换算：

```text
T_x_block = N_x × t_x_instruction
```

对于 global/shared/DSM 指令，优先让 benchmark 直接复现完整 128-thread block geometry，避免把 per-thread source call 错当成硬件 issue 数。

生产者 WG2 的依赖顺序为 load → convert → local/peer store：

```text
T_P = T_ld_block
    + T_cvt_block
    + T_st_shared_block
    + I(cluster2) × T_st_dsm_block
```

这是可解释的 first-order 模型。global load 的 memory-level parallelism、convert 与 store 的硬件管线重叠会通过 e2e 标定残差体现；只有 trace 能证明时才把内部 `+` 改成 `max`。

## 2. Reconstruct the consumer branches

WG0 主链：

```text
T_QK       = 36 × t_wgmma_qk_ss
T_PV_local =  4 × t_wgmma_pv_rs
```

WG1 在 WG0 softmax 发布 `sS/sScale` 后启动 remote PV：

```text
T_PV_remote = T_handoff + T_remote_scale + 4 × t_wgmma_pv_ss
```

local/remote PV 在两个 WG 上并行，consumer block 完成时间是：

```text
T_C = T_QK
    + T_softmax
    + max(T_PV_local, T_PV_remote)
```

这里不能简单丢掉 WG1；K buffer 的最终 release 需要两条 PV 分支都推进到对应 arrive。

## 3. Two-buffer pipeline

令 `B = topk / 64`。Q 的 TMA load 在 common prologue 发起，WG2 可同时生产首个 K block，因此首个 consumer 的就绪时间为：

```text
T_first_ready = max(T_qload, T_P)
```

完整 main-kernel 模型必须包括最后一个 consumer drain：

```text
T_main = T_first_ready
       + (B - 1) × max(T_P, T_C)
       + T_C
       + T_tma_store
```

例如 `topk=2048` 时 `B=32`。原模型若只写 `T_P + 31×max(P,C)+store`，会漏掉最后一个 `T_C`；在 `B=1` 时错误尤其明显。

split-KV 时再加 combine kernel：

```text
T_e2e_model = T_main + T_splitkv_reduce(num_splits)
```

只有 trace 证明 combine 与 main kernel 通过 PDL/stream 发生有效重叠时，才把这个 `+` 改成相应的 `max` 或分段 pipeline。

## 4. Bounds and calibration

理想双缓冲下界使用上面的 `max` 模型。保守全串行上界为：

```text
T_serial = T_qload + B × (T_P + T_C) + T_tma_store + T_splitkv_reduce
```

应检查：

```text
T_model_lower <= T_measured_e2e <= T_serial
rho = T_model_lower / T_measured_e2e
```

`rho` 是模型/重叠标定比，不是纯硬件利用率。`rho<1` 表示模型漏掉同步、调度、cache 或尾部成本；`rho>1` 通常表示原子独立测量高估、粒度换算错误，或融合 kernel 出现了模型未表达的内部 overlap。

## 5. Bottleneck and DSM ablation

稳态瓶颈直接来自：

```text
T_P > T_C  -> producer / memory-convert bound
T_C >= T_P -> consumer / tensor-softmax bound
```

DSM crossover 的消融使用相同 CTA/block 粒度：

```text
T_P_cross   = T_ld(32 tok) + T_cvt(32 tok) + T_sts(32 tok) + T_dsm(32 tok)
T_P_nocross = T_ld(64 tok) + T_cvt(64 tok) + T_sts(64 tok)
gain        = T_P_nocross - T_P_cross
```

只有 `gain>0` 且 e2e 同方向改善，才能说 cluster2 crossover 有净收益。

## 6. Required validation output

最终报告至少包含：

1. 每个指令级原子的 cycle、吞吐、动态计数和 ncu 隔离证据；
2. `T_P`、`T_C`、local/remote PV 两支、`max` 的胜出分支；
3. `T_main`、split tail、全串行上界和 e2e 实测；
4. `rho`、DSM A/B，以及误差最大的未建模项。

`compose.py` 是上述公式的可执行版本；日志解析仍待各 benchmark 输出格式稳定后实现。
