# 02 — Instruction-level atom decomposition

粒度为一个 CTA 处理一个 64-token KV page。只保留随 page 重复且可能进入关键路径的指令族；整数索引、少量 mask、一次性 descriptor prefetch 不单测。

下表先按 [`microbench/index.md`](../../../../microbench/index.md) 去重，再链接到具体配置叶子；不得按 dense decode 私有阶段复制 benchmark。

## Memory atoms

| ID | Operation | Dynamic work | Shared benchmark | Notes |
|---|---|---:|---|---|
| M0 | Q TMA load | request: `64×576×2=73,728 B` | [`tile64x576_bf16_sm90`](../../../../microbench/memory/tma_load/tile64x576_bf16_sm90/) | prologue only |
| M1 | K TMA load | page: 9×`64×64×2=8,192 B` | [`tile64x64_bf16_sm90`](../../../../microbench/memory/tma_load/tile64x64_bf16_sm90/) | dominant paged KV traffic |
| M2 | P stmatrix exchange | page: `[64,64]` BF16 | [`m64n64_b16_x4_sm90`](../../../../microbench/memory/stmatrix/m64n64_b16_x4_sm90/) | enables remote PV |
| M3 | output STSM + TMA store | request: `[64,512]` BF16 | [`m64n256_b16_x4_sm90`](../../../../microbench/memory/stmatrix/m64n256_b16_x4_sm90/) + [`tile64x512_bf16_2d_sm90`](../../../../microbench/memory/tma_store/tile64x512_bf16_2d_sm90/) | no-split epilogue; `+` because TMA consumes staged tile |
| M4 | split accum + combine | request/split: FP32 `[64,512]` | [`dv512_f32_sm90`](../../../../microbench/memory/splitkv_reduce/dv512_f32_sm90/) | only split path |

Q tile8 的 `ldmatrix` 是每 request 一次，为让 `sP1` 复用 `sQ` 空间；长 KV sequence 下不是主循环热点，暂不新增 microbench。若短序列 profile 显示显著，再加入 `ldmatrix_x4_b16_sm90`。

## Compute atoms

| ID | Operation | Dynamic count per page | Shared benchmark | Notes |
|---|---|---:|---|---|
| C0 | QK WGMMA SS `m64n64k16` | first page: 36；steady: 32 | [`m64n64k16_bf16_rs_ss_sm90`](../../../../microbench/compute/wgmma/m64n64k16_bf16_rs_ss_sm90/) | Q tiles 0–7 |
| C1 | QK WGMMA RS `m64n64k16` | steady: 4 | same benchmark, RS mode | register Q tile8 |
| C2 | shared-state online softmax | 1×`[64,64]` | [`online_m64n64_exp2_shfl_sm90`](../../../../microbench/compute/softmax/online_m64n64_exp2_shfl_sm90/) | shared max/scale mode |
| C3 | local PV WGMMA RS `m64n256k16` | 4 | [`m64n256k16_bf16_rs_ss_sm90`](../../../../microbench/compute/wgmma/m64n256k16_bf16_rs_ss_sm90/) | local page/output half |
| C4 | remote PV WGMMA SS `m64n256k16` | 4 | same benchmark, SS mode | exchanged P/other page |

## Pipeline-sensitive measurement

独立原子之外，dense decode 还需要一个组合扫描：9 个 `TMA[64×64]` 与每 tile 4 个 QK WGMMA 的 depth。它不是新的硬件原子，而是 M1+C0/C1 的调度实验：

```text
for tile in 0..8:
    wait TMA[tile]
    issue 4 × WGMMA[tile]
```

报告 standalone TMA、standalone WGMMA 和 combined pipeline 三组结果，才能判断 `max(T_TMA,T_WGMMA)` 是否成立。

## Exclusions

- `block_table` 的一次 32-bit load/page 与页地址计算；保留在 TMA setup，不单测。
- NamedBarrier arrive/wait bookkeeping；等待损失由 combined pipeline/e2e 标定。
- causal mask 只影响最后两页的谓词与填零；作为 tail correction，不建稳态原子。
- 最终 L reduction 和 LSE 写只发生一次/request；保留在 epilogue correction。
