# dense_decode_bf16_sm90_mqa

FlashMLA dense MLA decode 的 SM90a MQA 主路径。目录名保留历史 BF16 名称，当前
microbenchmark、DAG 和 combine 同时支持 BF16/FP16。默认性能 shape 与上游测试
一致：`h_q=128, h_kv=1, d_qk=576, d_v=512, page=64`，`s_q∈{1,2}`。

## Source anchors

- API/参数变换：[`dense_decode.h`](../../target/csrc/api/dense_decode.h)
- Kernel 与 launcher：[`splitkv_mla.cuh`](../../target/csrc/sm90/decode/dense/splitkv_mla.cuh)
- Tile/WG/shared-memory traits：[`traits.h`](../../target/csrc/sm90/decode/dense/traits.h)
- 固定 shape：[`config.h`](../../target/csrc/sm90/decode/dense/config.h)
- BF16 实例化：[`bf16.cu`](../../target/csrc/sm90/decode/dense/instantiations/bf16.cu)
- FP16 实例化：[`fp16.cu`](../../target/csrc/sm90/decode/dense/instantiations/fp16.cu)
- 上游 correctness/performance 基线：[`test_flash_mla_dense_decoding.py`](../../target/tests/test_flash_mla_dense_decoding.py)

## Documents and tools

1. [`00-config.md`](00-config.md)：固定的 dtype/head dimension/page/kernel geometry，以及可扫描的 batch、sequence、head 和 scheduler 参数。
2. [`01-kernel-implementation.md`](01-kernel-implementation.md)：2-WG CTA、首 page/steady page 差异、半页覆盖的 page-pair 调度与同步边。
3. [`02-atom-decomposition.md`](02-atom-decomposition.md)：generic inline-PTX atom family、dense role 映射和 calibration 边界。
4. [`03-performance-modeling.md`](03-performance-modeling.md)：metadata/main/two-WG/PDL/combine 全局 Atom-DAG 与资源调度模型。
5. [`e2e/`](e2e/)：公开 API held-out 测量；`generate` 覆盖
   metadata+main+combine，`reuse` 覆盖 main+combine。
6. [`model/`](model/)：DAG、scheduler、CSV cost database、resource simulator 和 CLI。
7. [`calibration/`](calibration/)：仅用于 residual/问题定位的 operator probe。

分析基于上游 commit `9241ae3ef9bac614dd25e45e507e089f888280e0`。当前没有保存 PTX/SASS/ncu，因此 CUTLASS type 只作为 source evidence，不冒充已观察到的机器指令；源码允许异步工作同时 outstanding，也不等同于已经测得硬件执行 overlap。

## Formal data boundary

正式预测只读取 `microbench/manifest.json`、各 family 最新 accepted full-sweep
`result.csv`、真实 main/combine cubin resource contract 和 workload。Calibration
结果不属于预测输入；held-out E2E 只做验收，不反向拟合 atom 或资源参数。

当前尚无 accepted H800 full sweep。remote 执行、日志和结果保留规则见仓库根目录
[`HANDOFF.md`](../../../../HANDOFF.md)。
