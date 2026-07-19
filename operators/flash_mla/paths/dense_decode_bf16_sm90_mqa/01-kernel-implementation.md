# 01 — Kernel implementation

本文用 `kernel-analysis-skill` 分析 `run_flash_splitkv_mla_kernel<cutlass::bfloat16_t>`。证据分为 **Confirmed (source)**、**Derived** 和 **Inference**；逻辑时间图只表达依赖与可同时 outstanding 的区间，不代表 cycle 或已测得的硬件 overlap。

源码锚点固定在 commit `9241ae3ef9bac614dd25e45e507e089f888280e0`：

- API reshape、scheduler 参数和 main+combine：`target/csrc/api/dense_decode.h:55-220`。
- WGMMA/TMA helper 与 page-pair 子程序：`target/csrc/sm90/decode/dense/splitkv_mla.cuh:27-944`。
- kernel body 和 launcher：`splitkv_mla.cuh:959-1352`。
- tile、shared memory 和 named barrier：`target/csrc/sm90/decode/dense/traits.h:13-107`。
- BF16 显式实例化：`target/csrc/sm90/decode/dense/instantiations/bf16.cu:1-6`。

## 0. Problem definition and tensors

Dense MLA decode 对每个 query 访问完整 paged KV sequence：

```text
P = (Q · Kᵀ) × softmax_scale
S = online_softmax(P, causal_mask)
O = S · V
```

MLA 的 K/V 共用 576 维 BF16 cache；QK 使用全部 576 维，PV 使用前 512 维。

| Argument | Type / layout | Public logical shape | Kernel logical shape | Role |
|---|---|---|---|---|
| `q` | BF16, last dim contiguous | `[b,s_q,h_q,576]` | `[b,q_seq_per_hk,h_kv,576]` | query |
| `kcache` | BF16 paged | `[num_pages,64,h_kv,576]` | page tile `[64,576]` | K and latent V |
| `block_table` | int32 | `[b,max_pages]` | page index stream | logical→physical page |
| `seqlens_k` | int32 | `[b]` | dynamic page count/tail | valid KV length |
| `out` | BF16 | `[b,s_q,h_q,512]` | `[b,h_kv,q_seq_per_hk,512]` | normalized output |
| `oaccum/lseaccum` | FP32 | allocated as `[b+num_sm_parts,h_kv,q_seq_per_hk,512]` / corresponding LSE | indexed by scheduler split | partial result for combine |

API 先做：

```text
q_seq_per_hk = s_q × (h_q / h_kv)
q -> [b, q_seq_per_hk, h_kv, 576]
```

默认 MQA `h_q=128,h_kv=1`：`q_seq_per_hk=128×s_q`，所以 `s_q=1` 有两个 64-row m-block，`s_q=2` 有四个。一个 `(blockIdx.x, blockIdx.y, blockIdx.z)` CTA 在 scheduler 分配的每个 request/split 上处理其中一个 `[64,512]` 输出 tile；同一 persistent CTA 可继续处理后续 request。

## 1. CTA and warp-group partition

Launcher：

```text
grid  = (ceil(q_seq_per_hk/64), h_kv, num_sm_parts)
block = (256,1,1) = 2 warp groups × 128 threads
```

`blockIdx.z` 选择一条 persistent scheduler partition。该 CTA 可处理一个或多个 request，并且首尾 request 可能只覆盖一个 page 子区间；若 request 被 split，main kernel 写 FP32 partial，随后 combine。

| Execution unit | Threads/warps | Responsibility | Main state | Synchronization |
|---|---:|---|---|---|
| CTA | 256 / 8 | 每个 request/split iteration 产生 `[64 query rows,512 output cols]` | `sQ`, `sK0/sK1`, `sP0/sP1`, `sM/sScale`, two rO halves | TMA barriers + named barriers |
| WG0 | 128 / 4 | even page QK/softmax；output cols `0:256`；local even PV + remote odd PV | `rP0,rO0` | WG1 softmax/P exchange |
| WG1 | 128 / 4 | odd page QK/softmax；output cols `256:512`；local odd PV + remote even PV | `rP1,rO1` | WG0 softmax/P exchange |
| TMA issue thread | Q/O 用 `threadIdx.x==0`；K 用调用 WG 的 lane 0 | Q tile、9 K tiles/page、no-split O tile | per-tile transaction barrier | WG0/WG1 wait corresponding K phase |

`sK0/sK1` 分别承载 even/odd page。它们不是整页一次覆盖的普通 ping-pong buffer：`0:256` 和 `256:576` 两段只有在消费对应 V half 的 PV 完成后才被下一 page 的 TMA 覆盖。每个 page `[64,576]` 又拆为 9 个 `[64,64]` K tiles；steady QK 对每个 tile 先等待 transaction barrier，再发射 4 个 `k16` GMMA source operations。

## 2. Pipeline and overlap

### Inter-warpgroup page-pair schedule

![Dense decode inter-warpgroup pipeline](assets/warpgroup-pipeline.svg)

图的左端从当前 page pair 的 QK 开始，而不是从 softmax 截断。首个 pair 的 even QK 是 prologue 中的 36 SS non-pipelined 路径；进入 steady loop 后，图左端的 `QK[i]` 实际在上一轮 subroutine 尾部发射并完成，随后当前轮才能读取 accumulator 做 softmax。把它画在当前 pair 的输入阶段，是为了完整表达 `QK -> softmax -> PV` 数据依赖；横轴仍是源码逻辑顺序，不是实测 cycle。

WG0/WG1 不是简单 producer/consumer：两者各自计算一个 page 的 QK/softmax，同时各拥有一半输出列。online softmax 的共享 `sM` 强制 page 顺序为 even→odd：WG0 更新 even page 后发出 `sScale0Ready`，WG1 才能更新 odd page 并发出 `sScale1Ready`。随后每个有效 page 的 P 都保存到 shared，供另一个 WG 计算另一半输出列。

**Confirmed (source)** 的一个跨 WG overlap 窗口是：WG0 发射 even local PV 后等待 `wait_group<0>`，同时 WG1 在收到 `sScale0Ready` 后执行 odd softmax；WG0 必须同时等 local PV 完成和 `sScale1Ready` 才能继续。因此该 join 可以建成 `max(T_PV_local_even, T_softmax_odd)`，但并发时的资源干扰仍需 H800 实测。

其余 page-pair 顺序要按源码保留：

1. WG1 先 `STSM(P_odd)`，再发射 odd local PV；两者不是两个独立并行阶段。
2. WG0 在收到 `sScale1Ready` 后对 even P rescale、`STSM(P_even)`，发出 `sP0Ready`。
3. WG1 发射 `P_even @ V_even_right` 后立即发出 `rO1sP0sV0RIssued`；WG0 随后 rescale O-left 并发射 `P_odd @ V_odd_left`。
4. 下一 pair 的 K TMA 不是整页顺序加载，而是按 buffer-half 安全性触发：WG0 在 even local PV 后覆盖 next-even `0:256`，在 odd remote PV 后覆盖 next-odd `0:256`；WG1 在 odd local PV 后覆盖 next-odd `256:576`，在 even remote PV 后覆盖 next-even `256:576`。两个 WG 之间的实际 request interleaving 没有固定总序。

### Intra-warpgroup TMA/WGMMA issue pipeline

![Dense decode intra-warpgroup pipeline](assets/intra-warpgroup-pipeline.svg)

源码确认的是异步 issue/wait 结构，而不是某个固定 overlap cycle：

- **首个 even page 是例外。** WG0 先等待 9 个 K barriers 全部 ready，再通过 `warpgroup_cooperative_qkt_gemm_no_pipeline` 连续发射 36 个 SS source operations，并 `wait_group<0>`。
- 从首个 odd page 开始，QK 使用 per-tile barrier pipeline。tiles 0–7 使用 shared-Q/SS，tile 8 的 Q 预先用 `ldmatrix` 放入 registers，使用 RS；每个 steady page 合计 `32 SS + 4 RS`。
- WG0 的确允许 remote PV 与下一 even page 的前四个 QK commit groups 同时 outstanding，但源码顺序是“发射 remote PV → 发射 QK tiles 0..3 → `wait_group<4>`”，不是先 wait 再发 QK。
- WG1 分别在 `wait_group<1>` 和 `wait_group<0>` 后覆盖 odd/even buffer 的右半段，再发射下一 odd page QK。

TMA 与 QK 有 per-tile overlap 的源码依据；remote PV 与 QK 都占 WGMMA pipeline，两个 WG 的 QK 也共享 SM tensor-core 资源，所以不能仅凭这些 source APIs 把整段费用写成 `max`。需要保存 PTX/SASS，并用 combined pipeline 测量实际 issue depth、stall 和 throughput。

同步边：

| Boundary | Mechanism | Meaning |
|---|---|---|
| Q ready | `barrier_Q` TMA transaction barrier | 两 WG 可读 `sQ` |
| K tile ready | `barriers_K{0,1}[0..8]` | 对应 64-dim tile 可进入 QK |
| even softmax ready | named barrier `sScale0Ready` | WG1 可按最新 global max 处理 odd page |
| odd scale ready | named barrier `sScale1Ready` | WG0 可 rescale even P；该 wait 也在 WG0 local PV 完成之后 |
| even P ready | `sP0Ready` | WG1 可发射 remote PV |
| remote even-P PV issued | named barrier `rO1sP0sV0RIssued` | WG0 可读取 `sP1`，并在对应 wait 后推进 shared buffer/TMA |
| output reduction/staging | `__syncthreads()` | 两 WG 的 L reduction 和 output shared staging |

## 3. Important instructions

| Instruction / intrinsic | Evidence | Scope | Purpose | Caveat |
|---|---|---|---|---|
| `SM90_TMA_LOAD` Q | Confirmed source API | GMEM→CTA shared | `[64,576]` Q prologue | PTX/SASS 待确认 |
| `SM90_TMA_LOAD` K tile | Confirmed source API | GMEM→`sK0/sK1` | 9×`[64,64]` per page | per-tile barriers enable overlap |
| CUTLASS GMMA SS/RS selector for QK | Confirmed source selector；shape/count Derived | shared/register Q × shared K | first page `36 SS`；steady page `32 SS+4 RS` | emitted `wgmma.mma_async` mnemonic待 PTX/SASS |
| CUTLASS GMMA RS/SS selector for PV | Confirmed source selector；shape/count Derived | register/shared P × shared V | per valid page `4 local RS + 4 remote SS` | emitted mnemonic待 PTX/SASS；两 WG 共享执行资源 |
| `SM90_U32x4_STSM_N` | Confirmed source atom | registers→shared | softmax P exchange；output staging | 应单独 benchmark stmatrix |
| `SM75_U32x4_LDSM_N` | Confirmed source atom | shared→register | 保存 Q tile8 到 rQ8 | 每 request 一次，非主循环热点 |
| `exp2f` + `shfl_xor` | Confirmed source | registers/warp + shared max/scale | even/odd online softmax 与 O/L rescale | WG0/WG1 路径不同，不能只测一个 generic step |
| `SM90_TMA_STORE` | Confirmed source API | shared→GMEM, rank-4 tensor map | no-split BF16 `[64,512]` output | split path改为 per-row FP32 `SM90_BULK_COPY_S2G` |
| PDL launch attributes + trigger/sync | Confirmed source API | main→combine | main CTA 在 store 前触发 dependent launch；split combine 在读取 partial 前 `cudaGridDependencySynchronize()` | no-split combine CTA 在 dependency sync 前直接 return；实际 kernel 间时间关系待 trace |

## Correctness and open evidence

- `grid.x` 的 64-row tiles 无重叠覆盖 `q_seq_per_hk`，尾行由 `num_valid_seq_q` 屏蔽。
- WG0/WG1 分别存 256 output columns，合计 512。
- 每 page 9×64 head-dim tiles 覆盖 576；每 PV 4×16 reduction tiles 覆盖 page size 64。
- even/odd softmax 对同一 `sM` 的更新由 named barriers 串行化。
- K buffer 的每个 half 只在对应 local/remote PV wait 后覆盖；`cur_phase_K0/K1` 在完整 9-tile QK 消费后翻转。
- `wait_group<4>` 前已经发射 remote PV 和四个 QK commit groups；最后 `wait_group<0>` 才允许 accumulator 被 softmax 使用。

仍需：BF16 实例的 PTX/SASS 动态计数、H800 上 combined TMA+QK 和完整 page-pair transition 的测量、`wait_group<4>` 的实际 stall/overlap、rank-4 TMA descriptor 的影响，以及 PDL main+combine 的真实时间边界。
