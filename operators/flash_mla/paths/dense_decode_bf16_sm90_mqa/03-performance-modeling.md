# 03 — Dense decode performance modeling

dense kernel 以两个相邻 KV pages 为 steady-state 单元。`+` 表示依赖串行，`max` 表示不同 WG/TMA/WGMMA 流可证明重叠，`N×T` 表示动态发射次数。

## 1. Instruction costs

```text
T_QK_first = 36 × t_qk_ss
T_QK_0a    = 16 × t_qk_ss                    # WG0 tiles 0..3
T_QK_0b    = 16 × t_qk_ss + 4 × t_qk_rs     # WG0 tiles 4..8
T_QK_1     = 32 × t_qk_ss + 4 × t_qk_rs     # WG1 full steady page

T_PV_local  = 4 × t_pv_rs
T_PV_remote = 4 × t_pv_ss
T_K_page    = 9 × T_tma_k_tile
```

`T_softmax` 必须使用 shared-state 模式；`T_stmatrix_P` 是一页 P exchange。

## 2. Page-pair event DAG

WG0 softmax even page 后，WG1 才能更新 odd page 的 global max。因此两个 softmax 串行，但 local PV、P exchange、remote PV、下一 pair QK/TMA 可跨 WG overlap。

first-order steady step：

```text
T_WG0 = T_softmax
      + max(T_PV_local, T_softmax)             # WG0 local PV ∥ WG1 odd softmax
      + T_stmatrix_P
      + max(T_PV_remote, T_QK_0a)              # wait_group<4> overlap
      + T_QK_0b

T_WG1 = 2 × T_softmax
      + T_stmatrix_P
      + T_PV_local
      + T_PV_remote
      + T_QK_1

T_pair_compute = max(T_WG0, T_WG1)
T_pair_load    = 2 × T_K_page
T_pair_steady  = max(T_pair_load, T_pair_compute)
```

这是可解释的 critical-path 近似，不是 cycle-accurate simulator。`T_WG1` 中部分 P exchange/PV 与 WG0 工作仍可 overlap，残差由 combined pipeline 和 e2e 校准。

## 3. Prologue, steady state, drain

令 scheduler 分给一个 CTA 的 page 数为 `N_page`，完整 page pair 数 `N_pair=floor(N_page/2)`：

```text
T_prologue_pair = max(T_qload, 2 × T_K_page)
                + max(T_QK_first, T_QK_1)

T_drain_pair = max(
    T_softmax + max(T_PV_local, T_softmax) + T_stmatrix_P + T_PV_remote,
    2 × T_softmax + T_stmatrix_P + T_PV_local + T_PV_remote
)

T_pairs = 0                                                    if N_pair=0
        = T_prologue_pair + (N_pair-1)×T_pair_steady + T_drain_pair
                                                                  otherwise
```

若 `N_page` 为奇数，增加一个单页 tail：

```text
T_KQ_single = measured_combined_K_TMA_QK_page
             # fallback upper bound: T_K_page + T_QK_first

T_odd = T_KQ_single
      + T_softmax
      + max(T_PV_local, T_stmatrix_P + T_PV_remote)
```

当 `N_page=1` 时使用 `max(T_qload,T_KQ_single)` 作为首个计算就绪时间；Q/K TMA 争用带来的误差由 combined pipeline 实测替代 fallback。

最后：

```text
T_main = T_pairs + I(N_page odd) × T_odd + T_output_store + T_reduce_L
T_e2e  = T_main + T_combine(num_splits)
```

combine 当前通过 PDL 紧随 main；没有 trace 证明 kernel 间执行重叠前，使用 `+`。

## 4. Bounds and bottleneck

全串行上界：

```text
T_serial = T_qload
         + N_page × (T_K_page + T_QK_1 + T_softmax + T_PV_local + T_stmatrix_P + T_PV_remote)
         + T_output_store + T_combine
```

诊断：

```text
T_pair_load > T_pair_compute  -> TMA/HBM bound
T_pair_compute >= T_pair_load -> WGMMA/softmax/synchronization bound
rho = T_model / T_measured_e2e
```

要求 `T_model <= T_measured <= T_serial`。若不成立，先检查 scheduler 实际 page/split 数、TMA tile+WGMMA combined 测量、causal tail，再考虑拟合常数。

[`compose.py`](compose.py) 实现该 first-order 模型；输入统一为 cycle/CTA 或 cycle/instruction。
