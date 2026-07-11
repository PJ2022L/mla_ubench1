# 03 — Dense decode performance modeling

模型粒度是 scheduler 分给一个 CTA 的一次 request/split iteration，`N_page=end_block_idx-start_block_idx`。`+` 只表示确认的数据依赖，`max` 只用于源码中两个 WG 在同一 join 前没有依赖的区间。CUTLASS async API 只证明工作可以同时 outstanding；实际 overlap、issue throughput 和资源干扰必须在 H800 上测量。

## 1. Dynamic work

由 `splitkv_mla.cuh` 的 compile-time loop 可机械推导：

```text
T_QK_first  = 36 × t_qk_ss
T_QK_steady = 32 × t_qk_ss + 4 × t_qk_rs
T_PV_local  = 4 × t_pv_rs
T_PV_remote = 4 × t_pv_ss
T_K_page    = 9 × T_tma_k_tile
```

首个 even page 等待全部 9 个 K barriers 后走 all-SS、non-pipelined QK；首个 odd page及之后的 page 才走 per-tile barrier pipeline，并在 tile8 使用 RS。每个有效 page 都有一份 local PV、一份 remote PV 和一次 `[64,64]` P STSM。

softmax 不能压成一个无差别常数：

- `T_softmax_even` 对应 `wg0_bunch_0`。
- `T_softmax_odd` 对应有效 odd page 的 `wg1_bunch_0`。
- `T_softmax_empty_odd` 对应 odd page 不存在时 WG1 仍执行的 masked control/rescale path。
- `T_rescale_p_even` 和 `T_rescale_o_even` 分别对应 `wg0_scale_rP0` 与 `wg0_rescale_rO0`；后者只在完整 pair 出现。

## 2. Source-supported overlap boundaries

| Candidate | Source evidence | Modeling rule |
|---|---|---|
| WG0 even local PV vs WG1 odd softmax | WG0 在 `sScale0Ready` 后发射 local PV；WG1 同时进入 odd softmax；WG0 同时等待 `wait_group<0>` 和 `sScale1Ready` | 可写 `max(T_PV_local,T_softmax_odd)`；并发干扰由 combined scan校准 |
| later K TMA vs earlier-tile QK | 每个 steady K tile 有独立 transaction barrier；QK 按 tile wait/issue | 使用实测 `T_kq_steady_page`；没有实测时用串行 fallback，不直接写整页 `max` |
| remote PV vs next-even QK tiles 0..3 | WG0 先发 remote PV，再发四个 QK commit groups，之后 `wait_group<4>` | 两者共享 WGMMA pipeline，只能用 combined transition 测量，不能取 standalone `max` |
| WG0 vs WG1 QK | 两个 WG 可并发 issue | 共享同一 SM tensor-core/WGMMA 资源；源码不足以支持 `max(T_WG0,T_WG1)` |
| main vs combine | PDL trigger + combine `cudaGridDependencySynchronize()` | split combine 的数据读取依赖 main；no-split combine 在 sync 前 return。kernel 时间边界需 trace |

因此，旧式 `T_pair_steady=max(2×T_K_page,max(T_WG0,T_WG1))` 没有源码依据：它忽略半 buffer 覆盖顺序、WGMMA 资源共享和 `wait_group` 的实际位置。

## 3. Schedule-level model

可解释的主模型使用复刻真实控制流的组合测量。所有输入均为 cycle/CTA schedule unit：

| Cost | Measurement boundary |
|---|---|
| `T_prologue_single` | Q/K TMA launch 到单页 `rP0` ready；包含首 page non-pipelined QK |
| `T_prologue_pair` | Q/K TMA launch 到首 pair 的 `rP0/rP1` 都 ready |
| `T_pair_transition` | 当前完整 pair 的 `rP0/rP1 ready` 到下一完整 pair 的 `rP0/rP1 ready` |
| `T_pair_to_single` | 当前完整 pair ready 到 odd single-tail 的 `rP0 ready`，保留真实 tail template flags |
| `T_pair_drain` | 最后一个完整 pair ready 到两 WG 的 O/L contributions ready |
| `T_single_drain` | single page ready 到两 WG 的 O/L contributions ready；包含 empty-odd control path |

令 `N_pair=floor(N_page/2)`：

```text
T_body = T_prologue_single + T_single_drain
                                             if N_page = 1

       = T_prologue_pair
         + (N_pair-1) × T_pair_transition
         + T_pair_drain
                                             if N_page is even

       = T_prologue_pair
         + (N_pair-1) × T_pair_transition
         + T_pair_to_single
         + T_single_drain
                                             if N_page >= 3 and odd
```

这三个分支与源码的 `IS_BLK0_LAST/IS_BLK1_LAST/IS_BLK2_LAST` 控制流对应；特别是 odd tail 的 KQ 会与前一 pair drain 穿插，不能写成 `T_pair_drain + T_KQ_single`。

```text
T_main = T_body + T_reduce_L + T_output_store_{nosplit|split}
```

no-split store 是两个 WG 的 BF16 STSM staging 加 rank-4 TMA store；split store 是 FP32 shared staging 加 per-row bulk S2G。两者不能复用同一个 cost。

`T_combine` 是另一个 grid 的时间，不能无条件加到 cycle/CTA 上。只有当它已转换成同一 critical-output/e2e 口径时，才报告诊断性的：

```text
T_e2e_additive = T_main + T_combine
```

最终仍以 CUDA event 覆盖公开 API 的 main+combine elapsed time 为准。

## 4. Atom-only fallback

在 page-pair transition 尚未实现时，`compose.py` 可读取独立原子并给出保守诊断。令 `E=ceil(N_page/2)`、`O=floor(N_page/2)`：

```text
T_KQ_all = T_kq_first_page + (N_page-1) × T_kq_steady_page

T_compute_serial = E × (T_softmax_even + T_rescale_p_even)
                 + O × (T_softmax_odd + T_rescale_o_even)
                 + I(N_page odd) × T_softmax_empty_odd
                 + N_page × (T_PV_local + T_stmatrix_P + T_PV_remote)

T_overlap_credit = O × min(T_PV_local, T_softmax_odd)
                 + I(N_page odd) × min(T_PV_local, T_softmax_empty_odd)

T_source_dag = T_qload + T_KQ_all + T_compute_serial
             - T_overlap_credit + T_epilogue
```

若没有 first/steady KQ combined 结果，fallback 分别使用 `T_K_page+T_QK_first/steady`。这个结果故意不猜测跨 pair 的 WGMMA/TMA overlap，因此只用于发现计数或数量级错误，不应称为精确预测或硬件上界。

独立原子的串行和也不是严格上界：cache 状态、occupancy、两个 WG 的资源竞争、barrier stall 和 launch overhead 都可能使真实 elapsed time落在其外。对拍时报告 `T_main_model/T_main_measured`、scheduler 的真实 `N_page/num_splits` 和 residual，不要求虚假的 `T_model <= T_measured <= T_serial`。

[`compose.py`](compose.py) 优先使用 schedule-level costs；缺少这些 keys 时退化到 atom-only fallback。CLI 推荐直接传 `--n-page`，因为完整 `seqlen_k` 在 scheduler split 后不等于单 CTA 的 page 数。
