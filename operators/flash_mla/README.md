# FlashMLA operator

本目录把 FlashMLA 的上游源码与具体 kernel path 分开管理。

## 路由

- [`target/`](target/)：上游 FlashMLA 源码快照，只读。当前 gitlink commit：`9241ae3ef9bac614dd25e45e507e089f888280e0`。
- [`paths/`](paths/)：按 kernel 实例化组织的分析、拆分和 modeling。

## 当前 SM90 paths

| Path | 上游入口 | 状态 |
|---|---|---|
| [`dense_decode_bf16_sm90_mqa`](paths/dense_decode_bf16_sm90_mqa/) | `target/csrc/sm90/decode/dense/` | 实现、拆分、model、e2e 已落地 |
| [`sparse_decode_fp8_sm90_v32_mqa_h128_cluster2`](paths/sparse_decode_fp8_sm90_v32_mqa_h128_cluster2/) | `target/csrc/sm90/decode/sparse_fp8/` | 当前主路径，文档已落地 |
| [`sparse_prefill_bf16_sm90_mqa`](paths/sparse_prefill_bf16_sm90_mqa/) | `target/csrc/sm90/prefill/sparse/` | 实现分析、拆分、model、e2e 已落地 |

当前不建立 SM100 path。新增 path 前先确认它对应独立的 kernel/模板实例化，而不是只有不同 runtime shape。
