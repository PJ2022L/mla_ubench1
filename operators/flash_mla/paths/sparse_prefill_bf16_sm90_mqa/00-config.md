# 00 - Configuration boundary

本文定义 `sparse_prefill_bf16_sm90_mqa` 当前分析实例的固定参数和运行时扫描面。精确建模实例是 `sm90::fwd::KernelTemplate<576, false>`；其中 `false` 表示编译期不提供 dynamic `topk_length`。

## 当前 path 的准确含义

```text
sparse_prefill_bf16_sm90_mqa
               |    |   |
               |    |   +-- h_kv=1 (MQA)
               |    +------ SM90a/Hopper
               +----------- Q/KV 为 BF16 sparse prefill
```

目录名没有编码 `h_q=128`。当前代表性能点和 e2e 脚本固定为 128 heads，但 SM90 kernel 本身也支持 64 heads；这是工具/实验范围限制，不是模板的 head-count 编译期常量。

## 固定参数

| 参数 | 当前 path 固定值 | 固定层级 | 说明 |
|---|---:|---|---|
| architecture | SM90a/Hopper | path | 当前分析不覆盖 SM100 |
| Q/KV dtype | BF16 | API/path | 不是 FP8 cache path |
| `h_kv` | 1 | kernel/path | `KernelTemplate::run()` 断言 MQA |
| `D_QK` | 576 | template | 当前精确实例为 `KernelTemplate<576,false>` |
| `D_V` | 512 | kernel compile time | V 使用 latent KV 前 512 维 |
| `HAVE_TOPK_LENGTH` | false | template | 当前实例使用统一 `topk`，不读取 per-query length |
| head tile `B_H` | 64 | kernel compile time | 一个 CTA 处理 64 heads |
| top-k block `B_TOPK` | 64 | kernel compile time | 主循环以 64 token block、128 token pair 调度 |
| threads / warpgroup | 384 / 3 WG | kernel compile time | 两个 compute WG + 一个 producer WG |
| cluster size | 1 CTA | launch geometry | `cluster=(1,1,1)`，没有 DSM |
| KV buffers | 2 | kernel control flow | BF16 gather double buffer |
| causal | unsupported | operator semantics | sparse prefill 使用显式 indices，没有 causal 参数 |
| split/combine | absent | operator path | 一个 CTA 直接完成对应 query/head tile |

模板常量见 [`config.h`](../../target/csrc/sm90/prefill/sparse/config.h)，launch 约束见 [`phase1.cuh`](../../target/csrc/sm90/prefill/sparse/phase1.cuh) 的 `KernelTemplate::run()`。

## 运行时可变参数

| 参数 | 约束或语义 | 对执行形状的影响 |
|---|---|---|
| `s_q` | 正整数 | 每个 query/head tile 一个 CTA；可远大于 65535 |
| `s_kv` | 正整数 | KV/indices 的合法地址范围和 working set |
| `topk` | 正整数且必须是 128 的整数倍 | `topk/64` blocks，`topk/128` block pairs |
| `h_q` | public SM90 dispatch 支持 64 或 128 | CTA 数按 `h_q/64` 成比例变化 |
| `indices` | int32 `[s_q,1,topk]` | 决定 gather locality；可以包含由 valid mask 处理的 OOB index |
| `softmax_scale` | float | 数值参数，不改变 tile shape |
| `attn_sink` | optional FP32 `[h_q]` | 影响最终 normalization，不改变核心 QK/PV 几何 |
| KV/Q strides | 最后一维连续，外层 stride 可变 | 影响地址步长，不改变模板 |
| device / SM count | 运行设备 | 改变 waves 和整卡吞吐，不改变单 CTA 几何 |

grid 为：

```text
N_CTA = s_q * (h_q / 64)
grid = (N_CTA, 1, 1)
cluster = (1, 1, 1)
```

该 API 没有 batch 维；`q` 是 `[s_q,h_q,576]`，一次调用描述一个 sparse-prefill 序列集合，而不是 dense decode 的 `[batch,s_q,...]` 接口。

## 当前工具限制与 kernel 能力的区别

| 项目 | kernel/API 能力 | 当前 path 工具 |
|---|---|---|
| `h_q` | SM90 public dispatch 支持 64 或 128 | `e2e/benchmark.py` 当前只接受 128；测 64 时应扩展脚本，而不是新增 kernel template |
| `topk_length` | 上游另有 `KernelTemplate<576,true>` | 当前 path 固定 `<576,false>`，不应把 true/false 数据混在同一模型中 |
| `d_qk=512` | 上游有独立 `<512,false/true>` 实例 | MODEL1 shape/layout 不属于当前 576 path |
| `attn_sink` | 当前 `<576,false>` 支持 runtime optional | 代表 e2e 可以选择 on/off，但必须在结果中明确标注 |

## 代表值而非固定值

以下只是当前 performance anchor：

```text
s_q=4096
s_kv=8192
topk=2048
h_q=128
```

它们都不是 `KernelTemplate<576,false>` 的编译期常量。上游测试还覆盖不规则 `s_q/s_kv`、OOB indices，以及多种 128-aligned top-k。

## 本 path 的建议扫描面

```text
fixed:  arch=sm90a, dtype=bf16, h_kv=1,
        d_qk=576, d_v=512, HAVE_TOPK_LENGTH=false,
        cluster=1, CTA=3 WG
scan:   s_q, s_kv, topk(128-aligned), h_q=64|128,
        indices pattern/validity, attn_sink on/off, working set
default performance anchor:
        s_q=4096, s_kv=8192, topk=2048, h_q=128
```
