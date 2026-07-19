# dense_decode_bf16_sm90_mqa

FlashMLA dense MLA decode 的 BF16、SM90a、MQA 主路径。默认性能 shape 与上游测试一致：`h_q=128, h_kv=1, d_qk=576, d_v=512, page=64`，`s_q∈{1,2}`，扫描 KV sequence length。

## Source anchors

- API/参数变换：[`dense_decode.h`](../../target/csrc/api/dense_decode.h)
- Kernel 与 launcher：[`splitkv_mla.cuh`](../../target/csrc/sm90/decode/dense/splitkv_mla.cuh)
- Tile/WG/shared-memory traits：[`traits.h`](../../target/csrc/sm90/decode/dense/traits.h)
- 固定 shape：[`config.h`](../../target/csrc/sm90/decode/dense/config.h)
- BF16 实例化：[`bf16.cu`](../../target/csrc/sm90/decode/dense/instantiations/bf16.cu)
- 上游 correctness/performance 基线：[`test_flash_mla_dense_decoding.py`](../../target/tests/test_flash_mla_dense_decoding.py)

## Documents and tools

1. [`00-config.md`](00-config.md)：固定的 dtype/head dimension/page/kernel geometry，以及可扫描的 batch、sequence、head 和 scheduler 参数。
2. [`01-kernel-implementation.md`](01-kernel-implementation.md)：2-WG CTA、首 page/steady page 差异、半页覆盖的 page-pair 调度与同步边。
3. [`02-atom-decomposition.md`](02-atom-decomposition.md)：可独立测量的公共原子、动态计数，以及必须保留的组合扫描。
4. [`03-performance-modeling.md`](03-performance-modeling.md)：以源码控制流为骨架的 prologue/transition/drain 模型；独立原子只作诊断，不把未测 overlap 写成事实。
5. [`e2e/`](e2e/)：公开 API 的稳态 main+combine 延时和 effective TFLOPS/GB/s。

分析基于上游 commit `9241ae3ef9bac614dd25e45e507e089f888280e0`。当前没有保存 PTX/SASS/ncu，因此 CUTLASS type 只作为 source evidence，不冒充已观察到的机器指令；源码允许异步工作同时 outstanding，也不等同于已经测得硬件执行 overlap。

## Model vs Measured

| Accepted run | Case / model | Predicted cycles | Measured composite cycles | Cycle error | Predicted e2e (ms) | Measured e2e (ms) | E2E error | Provenance |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 尚无 accepted H800 run | - | - | - | - | - | - | - | - |

当前正式模型与结果统一放在根目录 `microbench/results/<run_id>/`。full atom 和
calibration 的 `results.jsonl` 是 profile 输入，`run.log` 保存 args、命令和
wall time，`h800-profile.json`、held-out `results.jsonl` 与 `validation.json` 是
最终交付。不要再生成旧的 summary CSV/metadata/provenance 归档层。

`comparison.csv` 固定列为：

```text
case_id,model_kind,n_page,num_splits,
predicted_cycles,measured_composite_cycles,cycle_error_pct,
predicted_e2e_ms,measured_e2e_ms,e2e_error_pct,
microbench_run_ids,e2e_run_id,notes
```

只有作用域和单位一致时才填写误差。per-CTA `%clock64` cycle 不能用名义 SM 时钟换成 public API 的 CUDA-event ms；没有 event-level 预测时，`predicted_e2e_ms` 和 `e2e_error_pct` 留空。accepted run 必须在本表链接 model run、micro-benchmark provenance 和对应 e2e run。

归档 model run 时必须同时提供 `cycles.json`、逐字段可验证的 `provenance.json` 和固定表头的 comparison 输入 CSV。输入表填写 case、实际测量和来源 run ID；runner 从 compose JSON 补全预测字段与同单位 signed error，成功后写入不可变 run 的 `comparison.csv`。

composite 实测 cycle 必须通过 `T_measured`（多 case 使用 `T_measured__<case_id>`）回溯到成功 micro run；e2e 实测必须通过 `e2e_run_id` 回溯到本 path 的成功 e2e `latency_ms`。多 case 输出必须显式携带唯一 `case_id`。
