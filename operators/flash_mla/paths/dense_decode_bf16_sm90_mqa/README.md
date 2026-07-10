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

1. [`01-kernel-implementation.md`](01-kernel-implementation.md)：2-WG CTA、page-pair 调度、TMA/WGMMA/softmax 流水。
2. [`02-atom-decomposition.md`](02-atom-decomposition.md)：显著的公共指令级原子与动态计数。
3. [`03-performance-modeling.md`](03-performance-modeling.md)：page-pair event DAG 与 `max / + / ×` 模型。
4. [`e2e/`](e2e/)：公开 API 的稳态 main+combine 延时、TFLOPS、GB/s。

分析基于上游 commit `9241ae3ef9bac614dd25e45e507e089f888280e0`。当前没有保存 PTX/SASS/ncu，因此 CUTLASS type 只作为 source evidence，不冒充已观察到的机器指令。
