# sparse_prefill_bf16_sm90_mqa

该 path 固定分析 FlashMLA sparse BF16 prefill 的 Hopper 实例
`sm90::fwd::KernelTemplate<576, false>`。代表 shape 为
`s_q=4096,s_kv=8192,topk=2048,h_q=128,h_kv=1,d_qk=576,d_v=512`；
`false` 表示不提供动态 `topk_length`。上游固定在 commit
`9241ae3ef9bac614dd25e45e507e089f888280e0`。

## Source anchors

- Public API / SM90 dispatch：[`csrc/api/sparse_fwd.h`](../../target/csrc/api/sparse_fwd.h) 的 `sparse_attn_prefill_interface()` / `Fwd_Sm90_Impl::run_()`。
- Kernel 配置：[`config.h`](../../target/csrc/sm90/prefill/sparse/config.h) 的 `KernelTemplate<576, false>`。
- Kernel 与 launcher：[`phase1.cuh`](../../target/csrc/sm90/prefill/sparse/phase1.cuh) 的 `devfunc()` / `run()`。
- 显式实例化：[`phase1_k576.cu`](../../target/csrc/sm90/prefill/sparse/instantiations/phase1_k576.cu)。
- `cp.async` / TMA / WGMMA helper：[`sm90/helpers.h`](../../target/csrc/sm90/helpers.h)。
- 上游 correctness/performance cases：[`test_flash_mla_sparse_prefill.py`](../../target/tests/test_flash_mla_sparse_prefill.py)。

这个实例是单 CTA、3 warpgroup、cluster size 1；BF16 KV 由 producer WG 用
`cp.async.cg.shared.global` gather 到 shared memory。它没有 FP8 convert、DSM、
split-KV 或 combine kernel，不能复用 sparse decode 的 cluster2 producer 模型。

- [`00-config.md`](00-config.md)：固定的 `<576,false>` 实例边界，以及 `s_q/s_kv/topk/h_q` 等运行时扫描面。
- [`01-kernel-implementation.md`](01-kernel-implementation.md)：实现与流水线。
- [`02-atom-decomposition.md`](02-atom-decomposition.md)：显著指令级原子及公共 microbench 路由。
- [`03-performance-modeling.md`](03-performance-modeling.md)：`max / + / ×` 性能模型。
- [`e2e/`](e2e/)：public sparse-prefill API 的 latency/TFLOPS/GB/s 基线。

状态：源码级分析完成；尚无该实例的 PTX/SASS、ncu trace 或 H800 实测，文档中的时间轴和 overlap 均不是实测 cycle。

## Model vs Measured

| Accepted run | Case / model | Predicted cycles | Measured composite cycles | Cycle error | Predicted e2e (ms) | Measured e2e (ms) | E2E error | Provenance |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 尚无 accepted H800 run | - | - | - | - | - | - | - | - |

该 sparse path 当前只保留分析文档，不属于本次 dense-only `microbench/`
交付。已有 E2E run 仅保留 `results.jsonl` 与包含 args/runtime 的 `run.log`。
该 path 当前没有 `compose.py`，不能复用 sparse decode 的 compose。

`comparison.csv` 固定列为：

```text
case_id,model_kind,n_page,num_splits,
predicted_cycles,measured_composite_cycles,cycle_error_pct,
predicted_e2e_ms,measured_e2e_ms,e2e_error_pct,
microbench_run_ids,e2e_run_id,notes
```

只有作用域和单位一致时才填写误差。per-CTA `%clock64` cycle 不能用名义 SM 时钟换成 public API 的 CUDA-event ms；没有 event-level 预测时，`predicted_e2e_ms` 和 `e2e_error_pct` 留空。该 path 不使用 split-KV，`num_splits` 应留空。accepted run 必须在本表链接 model run、micro-benchmark provenance 和对应 e2e run。

归档 model run 时必须同时提供 `cycles.json`、逐字段可验证的 `provenance.json` 和固定表头的 comparison 输入 CSV。输入表填写 case、实际测量和来源 run ID；runner 从模型 JSON 补全预测字段与同单位 signed error，成功后写入不可变 run 的 `comparison.csv`。

composite 实测 cycle 必须通过 `T_measured`（多 case 使用 `T_measured__<case_id>`）回溯到成功 micro run；e2e 实测必须通过 `e2e_run_id` 回溯到本 path 的成功 e2e `latency_ms`。多 case 输出必须显式携带唯一 `case_id`。
