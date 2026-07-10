# sparse_decode_fp8_sm90_v32_mqa_h128_cluster2

这是当前主分析路径，对应 FlashMLA 的 V3.2 sparse FP8 decode、Hopper SM90、MQA、`NUM_HEADS=128`、`CLUSTER_SIZE=2` 实例化。

## Source anchors

- Launcher：[`splitkv_mla.cuh`](../../target/csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh) 的 `KernelTemplate<ModelType::V32, 128>::run()`。
- Kernel 主体：同文件的 `devfunc()`。
- 固定配置：[`config.h`](../../target/csrc/sm90/decode/sparse_fp8/config.h)。
- 实例化：[`v32_persistent_h128.cu`](../../target/csrc/sm90/decode/sparse_fp8/instantiations/v32_persistent_h128.cu)。
- FP8 转换与 DSM helper：[`components/`](../../target/csrc/sm90/decode/sparse_fp8/components/)。

## Documents

1. [`01-kernel-implementation.md`](01-kernel-implementation.md)：问题、CTA/cluster、3 个 warpgroup、同步和逻辑流水线。
2. [`02-atom-decomposition.md`](02-atom-decomposition.md)：显著的指令级 memory/compute 原子及公共 benchmark 路由。
3. [`03-performance-modeling.md`](03-performance-modeling.md)：用 `max / + / ×` 从原子重建 main kernel 和 split-KV 尾部。
4. [`e2e/`](e2e/)：V3.2 public API 的 main+combine latency、TFLOPS、GB/s。

`compose.py` 和 C++ e2e 仍含 scaffold；`e2e/benchmark.py` 复用上游 testcase generator，可在安装完整 FlashMLA 后运行。本仓库尚未取得 H800 实测数据。
