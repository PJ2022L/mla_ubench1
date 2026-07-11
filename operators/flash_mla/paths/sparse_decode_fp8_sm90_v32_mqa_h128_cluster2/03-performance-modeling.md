# 03 — Reconstructing sparse-decode performance

模型分三层，禁止混用粒度：

1. 一个 CTA/cluster 对一个 scheduler request segment 的 cycles；
2. persistent `partition_idx` 的多个 segment 累积及 main-grid critical partition；
3. 独立 combine grid 与 public API e2e wall interval。

## 1. Per-block producer

统一主循环原子为 `cycle / CTA / 64-token block`。优先使用完整 128-thread
producer geometry 的直接测量：

```text
T_P = T_producer_direct
```

仅在没有 direct 值时使用分解上界：

```text
T_P = T_ld_block + T_cvt_block + T_st_shared_block + T_st_dsm_block
```

这里每 CTA 只生产 32 token，并同时写 local + peer DSM，使 cluster 两 CTA 都得到
完整 64-token block。单 instruction latency 不能直接乘 128-thread source-call 数；
应由 full-geometry benchmark 吸收 coalescing、MLP 和 instruction issue。

## 2. Per-block consumer

```text
T_QK       = 36 * t_wgmma_qk_ss
T_PV_local =  4 * t_wgmma_pv_rs
T_PV_score = T_handoff + T_score_scale + 4 * t_wgmma_pv_ss
```

两支 PV 的同步关系允许并行推进，但共享 Tensor Core。首选 dual-WG harness 的
直接 stage time：

```text
T_PV = T_pv_dual_wg
```

缺少 direct 值时，只能使用带 aggregate-throughput floor 的近似：

```text
T_PV = max(T_PV_local, T_PV_score, T_pv_aggregate_floor)
```

其中 `T_pv_aggregate_floor` 来自两个 resident WG 共 8 个 WGMMA 的 SM aggregate
throughput。若把两个单 WG latency 直接 `max`，该 floor 缺失，结果不是可信下界。

```text
T_C = T_QK + T_softmax + T_PV
```

## 3. One scheduler segment

令 `B = end_block_idx - start_block_idx`。Q TMA 与首个 K producer 可并行；双
K-buffer 的 reuse 依赖两个 PV consumer 都 release：

```text
T_first  = max(T_qload, T_P)
T_steady = max(T_P, T_C)

T_segment = T_first + (B - 1) * T_steady + T_C + T_epilogue
```

最后一个 `T_C` 是 drain，不能省略。epilogue 根据 scheduler 的 split flag 二选一：

```text
non-split: T_epilogue = T_output_store_bf16_5d
split:     T_epilogue = T_partial_store_f32_bulk
```

全串行上界：

```text
T_segment_serial = T_qload + B * (T_P + T_C) + T_epilogue
```

`compose.py` 实现这一层。`--split-kv` 会要求 `T_partial_store`，不会再误用 BF16
TMA-store cycle；它也不会把 combine 的不同 grid 粒度硬加到 CTA segment。

## 4. Persistent partitions and main grid

SM90 public dispatch 设置：

```text
num_sm_parts = max(num_sms / s_q / (h_q/64), 1)
grid = (2, s_q, num_sm_parts), cluster=(2,1,1)
```

每个 `partition_idx` 的 cluster 按 `DecodingSchedMeta` 顺序处理一个或多个 request
segment。对 partition `p`：

```text
T_partition[p] ~= sum(segment r assigned to p, T_segment[p,r])
T_main_grid     ~= max_p T_partition[p]
```

相邻 request 的 Q load 在前一 request epilogue 前发起，所以上式的简单 `sum`
偏保守；精确模型应从 metadata 导出 segment 列表，并把 `Q[r+1]` 与 request `r`
的 reduction/store 建依赖 DAG。默认 `b=128,s_q=2,h_q=128` 在 H800 上不是
“一个 CTA 处理 32 blocks”，而是每个 persistent partition 通常处理多个 requests。

## 5. Split-KV combine and PDL

public API 总会 launch combine grid；`my_num_splits==1` 的 combine CTA 立即返回。
split request 的 main kernel 先写 FP32 `o_accum/lse_accum`，combine 再读所有
partials、做 LSE scaling/accumulation并写 BF16 O。

combine 通过 `cudaLaunchKernelEx` 启用 PDL，kernel 内调用
`cudaGridDependencySynchronize()`；main 在最终 request 的 epilogue 前调用
`cudaTriggerProgrammaticLaunchCompletion()`。因此 e2e 用：

```text
T_e2e = T_main_grid + T_combine_grid - T_pdl_overlap
0 <= T_pdl_overlap <= min(T_main_grid, T_combine_grid)
```

`T_pdl_overlap` 只能由 Kineto/ncu/trace 的 wall interval 得到。保守上界取 0；
不能因为源码启用 PDL 就把 main/combine 改成 `max`。

combine microbench 的 `cycle/row` 必须按 combine grid
`(b*s_q, 1, ceil(h_q/8))`、256 threads/CTA、实际 `num_splits` 与 waves 转换成
`T_combine_grid`，再与 `T_main_grid` 相加。

## 6. Bounds and validation

每层分别对拍：

```text
T_segment_model <=? T_segment_measured <= T_segment_serial
T_main_model    <=? T_main_kernel_wall
T_e2e_model     <=? T_public_api_wall
```

问号表示 atom independence、cache contention 和 scheduler overlap 仍需验证，不把
解析式宣称成严格硬件下界。最终报告至少包含：

1. producer full geometry、QK/softmax、dual-WG PV、两种 epilogue cycles；
2. metadata 导出的每 partition request/block 分配和 critical partition；
3. main/combine 各自 kernel interval、PDL wall overlap 与完整 e2e；
4. DSM crossover A/B、WGMMA aggregate throughput、HBM/L2 与 barrier stall；
5. atom model、serial bound、实测及残差最大的未建模项。
