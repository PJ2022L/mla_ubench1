# 02 — Instruction-level atom decomposition

拆分固定到 `KernelTemplate<576,false>`，统一以一个 CTA 的 **128-token block pair**
为主循环计数粒度。代表 shape `topk=2048` 有 16 个 pair。计数是源码级调用数，
最终 SASS issue 数必须在 H800 编译产物中确认。

## Exact producer geometry

WG2 把 128 threads 分成 16 个 8-thread group；每组负责每个 64-row block 的
4 行，每个 thread 负责一行的连续 8 个 BF16（16 B）。对 D=576，一个 pair 的
四段 tile 数为 `4 + 5 + 5 + 4 = 18`，每段每 thread 还循环 4 行：

```text
cp.async calls / thread / pair = 4 rows * 18 tiles = 72
cp.async calls / CTA / pair    = 128 * 72 = 9,216
bytes / pair                    = 9,216 * 16 = 147,456 B
bytes / 64-token block          = 73,728 B
```

`load_token_indices()` 每 thread 每 pair 发出 `2 buffers * 4 rows = 8` 个
`__ldg(int32)` 源码调用，即 1,024 calls/pair，但只有 128 个唯一 index。
该重复 load 与地址乘法保留在真实 gather harness 中，不单建整数 ALU 原子。

## Memory atoms

| ID | Significant operation | Dynamic work | Reusable benchmark | Assessment |
|---|---|---:|---|---|
| M0 | Q TMA load | 1/CTA；`64*576*2 = 73,728 B` | [`tile64x576_bf16_sm90`](../../../../microbench/memory/tma_load/tile64x576_bf16_sm90/) | 可直接复用；需 EVICT_FIRST + real swizzle |
| M1 | indexed BF16 `cp.async` G2S | 9,216 calls / pair；147,456 B | [`gather64x576_bf16_sm90`](../../../../microbench/memory/cp_async_g2s/gather64x576_bf16_sm90/) | exact 16-B instruction、random indices、4/5/5/4 order、mbarrier |
| M2 | score STSM handoff | 2 × `[64,64]` BF16 / pair | [`m64n64_b16_x4_sm90`](../../../../microbench/memory/stmatrix/m64n64_b16_x4_sm90/) | 可复用；要测两 resident WG 和真实 swizzle |
| M3 | output STSM staging | 1 × `[64,512]` / CTA，由两 WG 各 `[64,256]` | [`m64n256_b16_x4_sm90`](../../../../microbench/memory/stmatrix/m64n256_b16_x4_sm90/) | 可复用；两 WG 并行写 union O buffer |
| M4 | 3D TMA output store | 8 × `[64,64]` / CTA；总 65,536 B | [`tile64x64_bf16_3d_sm90`](../../../../microbench/memory/tma_store/tile64x64_bf16_3d_sm90/) | 不能用 decode 的单次 5D 64x512 结果替代 |

现有 [`128b_nc_l2_sm90`](../../../../microbench/memory/global_load/128b_nc_l2_sm90/)
测的是 `ld.global.nc` 到 register，不能代表本 kernel 的 `cp.async` GMEM→SMEM
数据路径、commit/mbarrier 或 16-B granularity。

## Compute atoms

| ID | Significant operation | Dynamic count / pair | Reusable benchmark | Assessment |
|---|---|---:|---|---|
| C0 | QK WGMMA SS `m64n64k16` | 36/block，72/pair | [`m64n64k16_bf16_rs_ss_sm90`](../../../../microbench/compute/wgmma/m64n64k16_bf16_rs_ss_sm90/) SS | 必须增加 `resident_wg=2` 和 source-like 16/20、20/16 issue batch |
| C1 | two-block online softmax | WG0 local tile + WG1 merged tile | [`online_m64n64_exp2_shfl_sm90`](../../../../microbench/compute/softmax/online_m64n64_exp2_shfl_sm90/) | 需要 local 与 cross-WG shared-max/handoff 两种模式；单 WG softmax 不足 |
| C2 | PV WGMMA RS | 4/block-half，8/pair | [`m64n256k16_bf16_rs_ss_sm90`](../../../../microbench/compute/wgmma/m64n256k16_bf16_rs_ss_sm90/) RS | 两 WG 各发 4；报告 aggregate stage cycles |
| C3 | cross-score PV WGMMA SS | 4/block-half，8/pair | 同一 WGMMA family 的 SS mode | 与 RS batch、STSM handoff 交错；报告 dual-WG contention |

`resident_wg=2` 不是可选美化：WG0/WG1 同时使用同一 SM 的 Tensor Core pipeline。
若只测单 WG `cycle/inst` 后对两个分支取 `max`，会漏掉共享执行资源的争用。
benchmark 应同时给出单 WG latency 和双 WG aggregate `inst/clk/SM`。

## Explicit exclusions

- 没有 FP8 数据，所以不使用 `fp8x8_to_bf16x8`。
- cluster size 为 1，所以没有 DSM store。
- prefill 是单 kernel，无 split-KV partial/combine。
- max/LSE 的两个 256-B bulk store、一次性 descriptor prefetch 和 barrier init 先并入 epilogue；相对 16 个 pair 的主循环较小。
- index load/address arithmetic 不独立计时，保留在 M1 的真实 gather geometry 中。

## Measurement contract

- M1 同时提供单 CTA latency与足量 CTA 的 aggregate bandwidth；后者才可用于 `s_q=4096` 的 HBM/L2 contention。
- C0/C2/C3 预置 shared/register operands，分 `resident_wg=1/2` 报告 cycles、aggregate inst/clk/SM、flop/clk/SM。
- C1 的输入 fragment mapping、两 WG NamedBarrier 和 `sM/sS` handoff 必须与源码一致；不能只测连续数组 `exp2f`。
- 所有 timed loop 保留指令完成所需的 commit/wait 与防优化 sink，但不把 host setup、数据生成或首次 JIT/allocator 行为计入。
