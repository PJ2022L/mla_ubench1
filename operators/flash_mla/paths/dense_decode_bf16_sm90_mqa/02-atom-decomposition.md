# 02 — Instruction-level atom decomposition

主循环计数粒度为一个 CTA 处理一个有效 64-token KV page；Q/output 则按该 CTA 的一次 request/split iteration 计数。只保留重复且可能进入关键路径的指令族；整数索引、少量 mask、一次性 descriptor prefetch 不单测。

下表先按 [`microbench/index.md`](../../../../microbench/index.md) 去重，再链接到具体配置叶子；不得按 dense decode 私有阶段复制 benchmark。

## Memory atoms

| ID | Operation | Dynamic work | Shared benchmark | Notes |
|---|---|---:|---|---|
| M0 | Q TMA load | iteration: `64×576×2=73,728 B` | [`tile64x576_bf16_sm90`](../../../../microbench/memory/tma_load/tile64x576_bf16_sm90/) | prologue；source tensor map rank=4 |
| M1 | K TMA load | page: 9 tiles × `64×64×2=8,192 B/tile` = `73,728 B/page` | [`tile64x64_bf16_sm90`](../../../../microbench/memory/tma_load/tile64x64_bf16_sm90/) | dominant paged KV traffic；source tensor map rank=4 |
| M2 | P stmatrix exchange | valid page: `[64,64]` BF16 = `8,192 B` | [`m64n64_b16_x4_sm90`](../../../../microbench/memory/stmatrix/m64n64_b16_x4_sm90/) | one P tile/page；enables remote PV |
| M3 | no-split output staging + TMA store | iteration: two WG each STSM `[64,256]`，then rank-4 TMA `[64,512]` BF16 | [`m64n256_b16_x4_sm90`](../../../../microbench/memory/stmatrix/m64n256_b16_x4_sm90/) + [`tile64x512_bf16_4d_sm90`](../../../../microbench/memory/tma_store/tile64x512_bf16_4d_sm90/) | dependency serial；two WG staging must be measured at CTA occupancy |
| M4 | split FP32 partial store | split iteration: ordinary FP32 register→shared staging，then `[64,512]` = `131,072 B` bulk S2G，plus LSE | [`tile64x512_f32_sm90`](../../../../microbench/memory/bulk_store/tile64x512_f32_sm90/) | shared benchmark只测 bulk S2G；staging 必须由 split-epilogue combined scan补上，不能复用 BF16 STSM |
| M5 | split combine | per output row reads `num_splits×(512 FP32 + LSE)` and writes BF16 output/LSE | [`dv512_f32_sm90`](../../../../microbench/memory/splitkv_reduce/dv512_f32_sm90/) | separate PDL dependent kernel；scan actual `num_splits` |

Q tile8 的 `ldmatrix` 是每 request 一次，为让 `sP1` 复用 `sQ` 空间；长 KV sequence 下不是主循环热点，暂不新增 microbench。若短序列 profile 显示显著，再加入 `ldmatrix_x4_b16_sm90`。

## Compute atoms

| ID | Operation | Dynamic count per page | Shared benchmark | Notes |
|---|---|---:|---|---|
| C0 | QK GMMA SS selector, derived `m64n64k16` work | first page: 36；every later valid page: 32 | [`m64n64k16_bf16_rs_ss_sm90`](../../../../microbench/compute/wgmma/m64n64k16_bf16_rs_ss_sm90/) | first page uses non-pipelined all-SS path；exact emitted mnemonic待 SASS |
| C1 | QK GMMA RS selector, derived `m64n64k16` work | first page: 0；every later valid page: 4 | same benchmark, RS mode | Q tile8 lives in registers |
| C2 | WG0 even shared-state softmax | one per even-role valid page | [`online_m64n64_exp2_shfl_sm90`](../../../../microbench/compute/softmax/online_m64n64_exp2_shfl_sm90/) | `wg0_bunch_0` mode：max/exp/L update + local O rescale |
| C3 | WG1 odd shared-state softmax | one per valid odd page；even-only tail still executes a masked control/rescale path | same benchmark, distinct dense-WG1 mode | `wg1_bunch_0` also consumes `sScale0` and rescales O/L |
| C4 | post-odd even P/O rescale | P rescale before STSM once/even page；O/L rescale once/full pair | same benchmark, dense shared-state continuation | `wg0_scale_rP0` and `wg0_rescale_rO0` must not disappear into an unmeasured constant |
| C5 | local PV GMMA RS, derived `m64n256k16` work | 4 per valid page | [`m64n256k16_bf16_rs_ss_sm90`](../../../../microbench/compute/wgmma/m64n256k16_bf16_rs_ss_sm90/) | owner page/output half |
| C6 | remote PV GMMA SS, derived `m64n256k16` work | 4 per valid page | same benchmark, SS mode | exchanged P/other output half |

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
