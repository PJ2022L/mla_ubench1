# 03 — Reconstructing sparse-prefill performance

模型固定到 `KernelTemplate<576,false>`。一个 CTA 计算 `[64 heads,512]` 输出，
一个主循环 iteration 处理两个 64-token block。令：

```text
R = topk / 128              # block-pair count; topk=2048 -> R=16
N_CTA = s_q * (h_q / 64)    # default -> 8192
```

## 1. Producer pair

首选输入是 M1 以真实 128-thread、4/5/5/4 segment 顺序直接测得的
`T_P_pair`。若只拿到分项：

```text
T_P_pair = T_index_and_addr_pair
         + T_cp_async_pair
         + T_cpasync_mbarrier_pair
```

这是依赖顺序的 first-order 上界。`cp.async` 在硬件中的 memory-level parallelism
已包含在完整 geometry benchmark 中，不能再把 9,216 个 source call 乘单指令
latency。跨 pair 的 producer 受四个 half-buffer free barrier 反压。

## 2. Consumer pair and shared Tensor Core resource

每个 pair 的 Tensor Core 动态工作是：

```text
QK:    72 * wgmma.m64n64k16.ss
PV:     8 * wgmma.m64n256k16.rs
       + 8 * wgmma.m64n256k16.ss
```

WG0/WG1 的 softmax 合并存在确认的串行边：

```text
softmax(K0) -> publish max0 -> softmax_merge(K1)
-> publish max1 / score handoff -> dependent rescale/cross PV
```

两个 WG 虽可并行 issue，但共享同一 SM 的 WGMMA throughput。正确输入应是双 WG
aggregate stage measurement：

```text
T_TC_pair  = measured dual-WG time for 72 QK + 8 RS-PV + 8 SS-PV issues
T_VEC_pair = T_softmax_local + T_softmax_merge + T_score_handoff + T_rescale
```

纯资源下界为：

```text
T_C_resource_lower = max(T_TC_pair, T_VEC_pair)
```

但源码包含 `wait_group`、max 交换和 buffer-release DAG，不能仅凭不同 WG 就声称
达到该下界。保守串行上界为：

```text
T_C_serial = T_QK_2wg + T_softmax_local + T_softmax_merge
           + T_score_handoff + T_PV_2wg + T_rescale
```

最可靠的 `T_C_pair` 是在复用 C0/C1/C2/C3 指令实现的两 WG source-schedule
harness 中直接测量；应满足 `T_C_resource_lower <= T_C_pair <= T_C_serial`。

## 3. CTA pipeline

Q TMA 与 WG2 首 pair gather 可并行。外层 producer/consumer overlap 在源码中确认，
但 buffer 是四个 half 细粒度释放，并非完整 pair ping-pong。因此下面的 pair-level
`max` 是待 H800 验证的 coarse model，不是硬件事实：

```text
T_first = max(T_qload, T_P_pair)
T_steady_pair = max(T_P_pair, T_C_pair)

T_CTA_coarse = T_first
             + (R - 1) * T_steady_pair
             + T_C_pair
             + T_epilogue
```

`T_epilogue` 包含 output normalization、两 WG STSM、8 次 3D TMA store，以及
max/LSE bulk stores。全串行上界：

```text
T_CTA_serial = T_qload + R * (T_P_pair + T_C_serial) + T_epilogue
```

若 coarse model 误差大，应把 producer 拆成 `K0L/K1R/K0R/K1L` 四段，按
`bar_k*_ready/free` 构造 DAG；不要用拟合常数掩盖 half-buffer backpressure。

## 4. CTA model to kernel latency

该 path 没有 persistent scheduler：每个 CTA 只处理一个 output tile。kernel grid
latency 仍不能直接等于单 CTA cycles。令 `C_res` 为 ptxas/ncu 确认的 resident
CTA/SM，`S` 为 SM 数：

```text
waves = ceil(N_CTA / (S * C_res))
T_kernel_ideal ~= waves * T_CTA
```

`__launch_bounds__(384,1)` 只保证编译约束，不单独证明实际 `C_res=1`；还需结合
register 和 dynamic-smem 数。默认 8192 CTA 会产生很多 waves，M1 必须提供多 CTA
带宽模式，使 `T_CTA` 已包含 HBM/L2 竞争，否则线性 wave 外推会过于乐观。

建议同时报告：

```text
T_kernel_coarse     = waves * T_CTA_coarse
T_kernel_serial     = waves * T_CTA_serial
rho = T_kernel_coarse / T_kernel_measured
```

`rho` 只是模型标定比。CTA 尾波、cache sharing、频率变化或 occupancy 估错都会改变它。

## 5. E2E boundary and validation

public `flash_mla_sparse_fwd` 在 SM90 路径只 launch `sparse_attn_fwd_kernel`；没有
split-KV/combine。CUDA-event 稳态区间包含该 kernel 的 Q/K gather、softmax、O、
max_logits 和 LSE 写回，不含 testcase/indices 生成及第一次 allocator/setup。

最终 H800 报告至少包含：

1. exact `cp.async` SASS 与 per-pair cycles/bytes、单 CTA和多 CTA吞吐；
2. WGMMA 单/双 resident WG 的 aggregate throughput，避免错误的 branch `max`；
3. local/merge softmax、score STSM 和 3D TMA store cycles；
4. ptxas resources、`C_res`、waves、kernel latency 与 atom/serial bounds；
5. ncu 的 HBM/L2、Tensor Core、barrier stall，解释最大残差。
