# 00 - Configuration boundary

本文定义 `sparse_decode_fp8_sm90_v32_mqa_h128_cluster2` 的固定实例和合法运行时扫描面。这个 path 比 dense path 更具体：`V32`、`h128` 和 `cluster2` 都是模板/launch 几何的一部分。

## 当前 path 的准确含义

```text
sparse_decode_fp8_sm90_v32_mqa_h128_cluster2
              |    |   |   |    |       |
              |    |   |   |    |       +-- cluster=(2,1,1)
              |    |   |   |    +---------- h_q=128
              |    |   |   +--------------- h_kv=1 (MQA)
              |    |   +------------------- DeepSeek V3.2 cache format
              |    +----------------------- SM90a/Hopper
              +---------------------------- FP8 sparse KV path
```

对应的精确实例是 `KernelTemplate<ModelType::V32, 128>`，显式实例化位于 [`v32_persistent_h128.cu`](../../target/csrc/sm90/decode/sparse_fp8/instantiations/v32_persistent_h128.cu)。

## 固定参数

| 参数 | 当前 path 固定值 | 说明 |
|---|---:|---|
| architecture | SM90a/Hopper | 当前分析不覆盖 SM100 |
| model/cache format | `ModelType::V32` | DeepSeek V3.2 layout，不是向量宽度 32 |
| Q dtype | BF16 | public API 强制 BF16 |
| KV NoPE dtype | FP8 E4M3 | 512 个 FP8 NoPE 元素/token |
| KV RoPE dtype | BF16 | 64 个 BF16 RoPE 元素/token |
| `h_q` / `NUM_HEADS` | 128 | 模板参数；不是运行时可变 |
| `h_kv` | 1 | sparse decode 当前只支持 MQA |
| `d_qk` | 576 | `512 NoPE + 64 RoPE` |
| `d_v` | 512 | V 复用 latent cache 的前 512 维 |
| quant tile / scale count | 128 / 4 | 每 128 个 FP8 元素一个 FP32 scale |
| bytes per KV token | 656 B | `512 + 4*4 + 64*2` |
| CTA head tile | 64 | `BLOCK_M=64` |
| top-k processing block | 64 tokens | `TOPK_BLOCK_SIZE=64` |
| threads / warpgroup | 384 / 3 WG | producer + 两个 consumer WG |
| cluster size | 2 CTA | `128/64=2` 个 M blocks，cluster `(2,1,1)` |
| K buffers | 2 | producer/consumer double buffer |
| dynamic `topk_length` | unsupported | V3.2 launcher 断言 `topk_length==nullptr` |
| extra KV cache | unsupported | V3.2 launcher 断言 `extra_kv==nullptr` |
| causal mode | unsupported | sparse indices 路径不是 causal dense attention |

模板常量见 [`config.h`](../../target/csrc/sm90/decode/sparse_fp8/config.h)，V3.2 特有限制见 [`splitkv_mla.cuh`](../../target/csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh) 的 `KernelTemplate::run()`。

## 运行时可变参数

| 参数 | 约束或语义 | 影响 |
|---|---|---|
| `batch` | 正整数 | persistent scheduler 可让同一 cluster 顺序处理多个 request |
| `s_q` | 正整数 | grid 的第二维，并参与 `num_sm_parts` 计算 |
| `s_k` / KV working set | 正整数；由 cache 和 indices 描述 | 改变可选 token 范围与 cache/HBM 状态 |
| `topk` | 正整数且必须是 64 的整数倍 | 主循环 block 数为 `topk/64`；当前没有 per-request dynamic length |
| `indices` | int32 `[b,s_q,topk]` | 决定实际 gather 地址和 locality/randomness |
| cache page size / number of pages | 运行时 tensor shape | 不改变 656 B/token 的连续 row 要求 |
| `softmax_scale` | float | 数值参数 |
| `attn_sink` | optional FP32 `[h_q]` | kernel 支持；当前 e2e 基线默认关闭 |
| scheduler metadata / `num_splits` | 运行时生成或复用 | 决定 persistent partition 范围和 combine 工作量 |
| SM count | 设备属性 | `num_sm_parts=max(num_sms/s_q/(h_q/64),1)` |

固定 `h_q=128` 时：

```text
NUM_M_BLOCKS = 2
CLUSTER_SIZE = 2
grid = (2, s_q, num_sm_parts)
cluster = (2, 1, 1)
```

## 上游支持但不属于当前 path

| 上游实例或 feature | 为什么不属于当前 path |
|---|---|
| `ModelType::V32, h_q=64` | 上游有独立 `v32_persistent_h64.cu`；其 cluster size 为 1，应建立独立 h64/cluster1 variant |
| `ModelType::MODEL1` | `d_qk=512`、量化 tile/scale/cache row 均不同，不是 V3.2 格式 |
| dynamic `topk_length` | generic API 有该 feature，但 V3.2 SM90 kernel 明确拒绝 |
| extra KV cache / extra top-k | generic API 可表达，V3.2 SM90 kernel 明确拒绝；主要用于 MODEL1 |
| SM100 sparse decode | dispatch、cluster/head 实现不同，当前仓库不建立 SM100 path |

KV tensor的标量 dtype 在 API 层可表现为 FP8 E4M3、int8 或 uint8 容器，但物理字节布局必须仍是 V3.2 的 656 B/token；这不代表支持三种不同量化格式。

## 本 path 的建议扫描面

```text
fixed:  arch=sm90a, model=V32, h_q=128, h_kv=1,
        d_qk=576, d_v=512, cluster=2, topk_length absent,
        extra_kv absent
scan:   batch, s_q, s_k, topk(64-aligned), indices pattern,
        cache working set, num_splits, attn_sink on/off
default performance anchor:
        batch=128, s_q=2, s_k=32768, topk=2048
```
