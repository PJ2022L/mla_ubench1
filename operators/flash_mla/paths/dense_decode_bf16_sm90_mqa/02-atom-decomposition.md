# 02 — Instruction-level atom decomposition

主循环计数粒度为一个 CTA 处理一个有效 64-token KV page；Q/output 则按该 CTA 的一次 request/split iteration 计数。只保留重复且可能进入关键路径的指令族；整数索引、少量 mask、一次性 descriptor prefetch 不单测。

完整 registry 由 [`microbench/manifest.json`](../../../../microbench/manifest.json)
维护。下表只列 dense main/combine 的核心重复工作；metadata、同步、地址控制和
普通 global/shared 访存也已在 manifest 中分类，不按 dense 私有阶段复制 atom。

## Memory atoms

| ID | Operation | Dynamic work | Shared benchmark | Notes |
|---|---|---:|---|---|
| M0 | Q TMA load | iteration: `64×576×2=73,728 B` | [`q_bf16_rank4`](../../../../microbench/memory/tma_load/q_bf16_rank4/) | prologue；source tensor map rank=4 |
| M1 | K TMA load | page: 9 tiles × `64×64×2=8,192 B/tile` = `73,728 B/page` | [`k_bf16_rank4`](../../../../microbench/memory/tma_load/k_bf16_rank4/) | dominant paged KV traffic；source tensor map rank=4 |
| M2 | P stmatrix exchange | valid page: `[64,64]` BF16 = `8,192 B` | [`p_b16`](../../../../microbench/memory/stmatrix/p_b16/) | one P tile/page；enables remote PV |
| M3 | no-split output staging + TMA store | two WG STSM `[64,256]`，then rank-4 TMA `[64,512]` B16 | [`o_b16`](../../../../microbench/memory/stmatrix/o_b16/) + [`o_bf16_rank4`](../../../../microbench/memory/tma_store/o_bf16_rank4/) | complete ordered protocol由 `epilogue_nosplit_b16` calibration测量 |
| M4 | split FP32 partial store | stride-520 FP32 register→shared staging，then `131,072 B` bulk S2G | [`u64_dense`](../../../../microbench/memory/shared_store/u64_dense/) + [`oaccum_f32`](../../../../microbench/memory/bulk_store/oaccum_f32/) | complete ordered protocol由 `epilogue_split_f32` calibration测量 |
| M5 | split combine | per output row reads `num_splits×(512 FP32 + LSE)` and writes B16 output/LSE | [`float4_oaccum`](../../../../microbench/memory/global_load/float4_oaccum/) + [`u64_output`](../../../../microbench/memory/global_store/u64_output/) | complete CTA由 BF16/FP16 combine calibration扫描 actual `num_splits` |

Q tile8 的 `ldmatrix` 是每 request 一次，为让 `sP1` 复用 `sQ` 空间；已由
[`p_b16 LDSM`](../../../../microbench/memory/ldmatrix/p_b16/) leaf 覆盖。

## Compute atoms

| ID | Operation | Dynamic count per page | Shared benchmark | Notes |
|---|---|---:|---|---|
| C0 | QK GMMA SS `m64n64k16` | first page: 36；every later valid page: 32 | [`qk_ss_bf16`](../../../../microbench/compute/wgmma/qk_ss_bf16/) | first page uses non-pipelined all-SS path |
| C1 | QK GMMA RS `m64n64k16` | first page: 0；every later valid page: 4 | [`qk_rs_bf16`](../../../../microbench/compute/wgmma/qk_rs_bf16/) | Q tile8 lives in registers |
| C2 | WG0 even shared-state softmax | one per even-role valid page | [`softmax_stage_bf16`](../../../../microbench/model/calibration/softmax_stage_bf16/) | max/exp/L update + probability conversion + O rescale interaction |
| C3 | WG1 odd shared-state softmax | one per valid odd page；even-only tail still executes a masked control/rescale path | same benchmark, distinct dense-WG1 mode | `wg1_bunch_0` also consumes `sScale0` and rescales O/L |
| C4 | post-odd even P/O rescale | P rescale before STSM once/even page；O/L rescale once/full pair | same benchmark, dense shared-state continuation | `wg0_scale_rP0` and `wg0_rescale_rO0` must not disappear into an unmeasured constant |
| C5 | local PV GMMA RS `m64n256k16` | 4 per valid page | [`pv_rs_bf16`](../../../../microbench/compute/wgmma/pv_rs_bf16/) | owner page/output half |
| C6 | remote PV GMMA SS `m64n256k16` | 4 per valid page | [`pv_ss_bf16`](../../../../microbench/compute/wgmma/pv_ss_bf16/) | exchanged P/other output half |

## Pipeline-sensitive measurement

独立原子只能给出吞吐诊断，不能直接决定 async pipeline 的 elapsed cycle。dense decode 至少需要以下组合扫描；它们是调度验证场景，不作为新的公共硬件原子登记：

1. **First-page KQ**：9 个 K TMA 全部 barrier-ready 后，一次发射 36 SS QK，复刻 `warpgroup_cooperative_qkt_gemm_no_pipeline`。
2. **Steady-page KQ**：9 个 per-tile barrier wait，每 tile 发射 4 个 QK，其中 tile8 为 RS：

```text
for tile in 0..8:
    wait TMA[tile]
    issue 4 × WGMMA[tile]
```

3. **Two-WG page-pair transition**：从当前 `rP0/rP1 ready` 到下一 pair `rP0/rP1 ready`，保留 named barriers、P exchange、half-buffer overwrite、`wait_group<1/4/0>` 和真实 compile-time tail flags。

报告 standalone TMA/WGMMA/softmax/STSM、first/steady KQ、page-pair transition 三层结果。只有 combined 测量能给出实际 overlap；不能从 source issue 顺序直接假设 `max(T_TMA,T_WGMMA)`，也不能把两个 WG 的 WGMMA standalone cycle 直接取 `max`。

## Exclusions

- `block_table` 的一次 32-bit load/page 与页地址计算；保留在 TMA setup，不单测。
- NamedBarrier arrive/wait bookkeeping；等待和资源干扰由 page-pair transition/e2e 标定。
- causal mask 只影响最后两页的谓词与填零；作为 tail correction，不建稳态原子。
- 最终 L reduction 和 LSE 写只发生一次 request/split iteration；保留在 epilogue correction。
