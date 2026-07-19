# sparse_decode_fp8_sm90_v32_mqa_h128_cluster2

这是当前主分析路径，对应 FlashMLA 的 V3.2 sparse FP8 decode、Hopper SM90、MQA、`NUM_HEADS=128`、`CLUSTER_SIZE=2` 实例化。

## Source anchors

- Launcher：[`splitkv_mla.cuh`](../../target/csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh) 的 `KernelTemplate<ModelType::V32, 128>::run()`。
- Kernel 主体：同文件的 `devfunc()`。
- Public dispatch：[`csrc/api/sparse_decode.h`](../../target/csrc/api/sparse_decode.h) 的 `Decode_Sm90_Impl` 与 `sparse_attn_decode_interface()`。
- 固定配置：[`config.h`](../../target/csrc/sm90/decode/sparse_fp8/config.h)。
- 实例化：[`v32_persistent_h128.cu`](../../target/csrc/sm90/decode/sparse_fp8/instantiations/v32_persistent_h128.cu)。
- FP8 转换与 DSM helper：[`components/`](../../target/csrc/sm90/decode/sparse_fp8/components/)。
- Scheduler / combine：[`get_decoding_sched_meta.cu`](../../target/csrc/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu) 与 [`combine.cu`](../../target/csrc/smxx/decode/combine/combine.cu)。

## Documents

1. [`00-config.md`](00-config.md)：固定的 V3.2/h128/cluster2 实例边界和合法运行时扫描参数。
2. [`01-kernel-implementation.md`](01-kernel-implementation.md)：问题、CTA/cluster、3 个 warpgroup、同步和逻辑流水线。
3. [`02-atom-decomposition.md`](02-atom-decomposition.md)：显著的指令级 memory/compute 原子及公共 benchmark 路由。
4. [`03-performance-modeling.md`](03-performance-modeling.md)：用 `max / + / ×` 从原子重建 main kernel 和 split-KV 尾部。
5. [`e2e/`](e2e/)：V3.2 public API 的 main+combine latency、TFLOPS、GB/s。

`compose.py` 重建一个 scheduler assignment 的 CTA/cluster 周期；完整 grid 还需按 metadata 聚合各 partition。C++ e2e 仍是 scaffold，`e2e/benchmark.py` 复用上游 testcase generator，可在安装完整 FlashMLA 后运行。本仓库尚未取得 H800 实测数据。

## Model vs Measured

| Accepted run | Case / model | Predicted cycles | Measured composite cycles | Cycle error | Predicted e2e (ms) | Measured e2e (ms) | E2E error | Provenance |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 尚无 accepted H800 run | - | - | - | - | - | - | - | - |

该 sparse path 当前只保留分析文档，不属于本次 dense-only `microbench/`
交付。已有 E2E run 仅保留 `results.jsonl` 与包含 args/runtime 的 `run.log`。

`comparison.csv` 固定列为：

```text
case_id,model_kind,n_page,num_splits,
predicted_cycles,measured_composite_cycles,cycle_error_pct,
predicted_e2e_ms,measured_e2e_ms,e2e_error_pct,
microbench_run_ids,e2e_run_id,notes
```

只有作用域和单位一致时才填写误差。per-cluster/CTA `%clock64` cycle 不能用名义 SM 时钟换成 public API 的 CUDA-event ms；没有 event-level 预测时，`predicted_e2e_ms` 和 `e2e_error_pct` 留空。accepted run 必须在本表链接 model run、micro-benchmark provenance 和对应 e2e run。

归档 model run 时必须同时提供 `cycles.json`、逐字段可验证的 `provenance.json` 和固定表头的 comparison 输入 CSV。输入表填写 case、实际测量和来源 run ID；runner 从 compose JSON 补全预测字段与同单位 signed error，成功后写入不可变 run 的 `comparison.csv`。

composite 实测 cycle 必须通过 `T_measured`（多 case 使用 `T_measured__<case_id>`）回溯到成功 micro run；e2e 实测必须通过 `e2e_run_id` 回溯到本 path 的成功 e2e `latency_ms`。多 case 输出必须显式携带唯一 `case_id`。
