# sparse_prefill_bf16_sm90_mqa

上游源码：`../../target/csrc/sm90/prefill/sparse/`。

状态：已登记，尚未完成证据驱动分析。prefill 与 decode 的 tile、并行轴和流水深度不同，禁止直接复用 decode 结论。

- `01-kernel-implementation.md`：实现与流水线。
- `02-atom-decomposition.md`：显著指令级原子及公共 microbench 路由。
- `03-performance-modeling.md`：`max / + / ×` 性能模型。
- [`e2e/`](e2e/)：public sparse-prefill API 的 latency/TFLOPS/GB/s 基线。
