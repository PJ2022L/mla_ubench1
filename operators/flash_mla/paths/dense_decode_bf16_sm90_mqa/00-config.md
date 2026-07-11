# 00 - Configuration boundary

本文定义 `dense_decode_bf16_sm90_mqa` 的配置边界：哪些参数属于当前 path 的固定实例，哪些参数可以在不改变 kernel 实现的情况下运行时扫描，以及哪些上游能力不属于本 path。

## 当前 path 的准确含义

```text
dense_decode_bf16_sm90_mqa
             |    |   |
             |    |   +-- MQA：h_kv = 1
             |    +------ SM90a/Hopper
             +----------- Q/K/KV 输入为 BF16
```

当前代表性能点是 `batch=128, s_q=1|2, h_q=128, s_k` 扫描，但这些代表值不等于全部编译期常量。

## 固定参数

| 参数 | 当前 path 固定值 | 固定层级 | 依据 |
|---|---:|---|---|
| architecture | SM90a/Hopper | path | API 拒绝非 SM90a |
| input dtype | BF16 | path/实例化 | 当前 path 路由到 `bf16.cu`；上游 FP16 是另一实例化 |
| `h_kv` | 1 | path | `mqa` 的定义；当前 e2e 也固定为 1 |
| `d_qk` / `HEAD_DIM_K` | 576 | kernel compile time | `Config::HEAD_DIM_K=576`，launcher 断言 `params.d==576` |
| `d_v` / `HEAD_DIM_V` | 512 | kernel compile time | `Config::HEAD_DIM_V=512` |
| page size | 64 tokens | kernel/API | `PAGE_BLOCK_SIZE=64`，API 显式检查 |
| CTA M tile | 64 | kernel compile time | 一个 M block 覆盖 64 个 `q_seq_per_hk` 行 |
| threads / warpgroup | 256 / 2 WG | kernel compile time | `Traits::NUM_THREADS=256` |
| K shared buffers | 2 | kernel control flow | `sK0/sK1` page-pair pipeline |
| output dtype | BF16 | 当前 BF16 实例 | no-split 输出；split partial 仍使用 FP32 accumulator |

固定 shape 见 [`config.h`](../../target/csrc/sm90/decode/dense/config.h)，运行前断言见 [`splitkv_mla.cuh`](../../target/csrc/sm90/decode/dense/splitkv_mla.cuh) 的 `run_flash_splitkv_mla_kernel()`。

## 运行时可变参数

| 参数 | 约束或语义 | 对执行形状的影响 |
|---|---|---|
| `batch` | 正整数 | 改变 request 数和 persistent scheduler 工作量 |
| `s_q` | 正整数；不限于默认的 1/2 | 参与 `q_seq_per_hk`；`s_q=1` 时 causal 被强制关闭 |
| `s_k` / `seqlens_k` | 每 request 可变，可含 page tail | 改变 page 数、split 数和 main-loop 次数 |
| `h_q` | 正整数；当前 MQA 下自动满足 `h_q % h_kv == 0` | 改变 `q_seq_per_hk` 和 M-block 数 |
| `causal` | bool；只有 `s_q>1` 才有效 | 改变尾部 mask，不改变核心 tile shape |
| `softmax_scale` | float | 数值参数，不改变 kernel 几何 |
| block table / physical page order | int32，最后一维连续 | 改变 KV page 的物理地址和 cache 行为 |
| scheduler metadata / `num_splits` | 可由 API 首次调用生成并复用 | 改变每个 SM partition 的 request/page 范围和 combine 工作 |
| `num_blocks` / maximum blocks per sequence | 由 KV cache 和 block table 决定 | 改变可寻址 working set，不改变编译期 tile |

核心运行时关系是：

```text
q_seq_per_hk = s_q * h_q / h_kv
num_m_blocks = ceil(q_seq_per_hk / 64)
grid = (num_m_blocks, h_kv, num_sm_parts)
```

因此 `h_q=128` 不是 kernel 编译期常量。例如 MQA 下 `s_q=1,h_q=64` 产生 1 个 M block，`s_q=2,h_q=128` 产生 4 个 M blocks。

## 上游支持但不属于当前 path

| 上游能力 | 当前 path 处理方式 |
|---|---|
| FP16 dense decode | 上游有 `fp16.cu`，但应作为 dtype variant，不混入 BF16 path 的结果 |
| `h_kv>1` 的 GQA/MHA | API 允许 `h_q % h_kv == 0`，但不再是 MQA；scheduler/grid 语义变化，应单独标注或建立 variant |
| `d_qk=512` | API 前置检查写着允许 512/576，但当前 dense kernel 固定 `HEAD_DIM_K=576` 并在 launcher 断言；本 commit 中不能把 512 视为有效配置 |
| page size 非 64 | API 和 kernel 都不支持；需要新的 kernel 配置，而不是运行时 sweep |

另有一个上游错误信息写成 “Only head_size_v == 576”，实际条件是 `d_v==512`；以条件和 kernel config 为准。

## 本 path 的建议扫描面

```text
fixed:  dtype=bf16, arch=sm90a, h_kv=1, d_qk=576, d_v=512, page=64
scan:   batch, s_q, h_q, s_k, causal, page tail, num_splits
default performance anchor:
        batch=128, s_q=1|2, h_q=128,
        s_k=4096|8192|16384|32768
```

改变 runtime 参数仍可留在本 path；改变 dtype、MQA/GQA 语义、head dimension 或 page size时，必须重新判断是否需要新的 path 或实例化。
