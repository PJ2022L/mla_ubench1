# 02 - Generic atom decomposition

Dense decode 不再拥有一套以 QK、PV、scheduler 命名的私有 benchmark。所有正式
cost 都来自 [`microbench/manifest.json`](../../../../microbench/manifest.json)
登记的 kernel-agnostic operation；dense 角色到 generic atom 的映射集中在
[`model/atom_map.json`](model/atom_map.json)。

例如：

| Dense source role | Generic atom |
|---|---|
| first/steady score shared-source group | `m64n64k16_ss_<dtype>` |
| steady score register-source tile 8 | `m64n64k16_rs_<dtype>` |
| local probability/value group | `m64n256k16_rs_<dtype>` |
| remote probability/value group | `m64n256k16_ss_<dtype>` |
| Q/K TMA load | `tensor_4d_64x576_<dtype>` / `tensor_4d_64x64_<dtype>` |
| no-split output TMA store | `tensor_4d_64x512_<dtype>` |
| split partial output bulk store | `cp_async_bulk_s2g_64x512_f32` |

## Atom families

- `compute/wgmma`: 8 个固定 MNK/source-mode/dtype 组合。一个 DAG 节点表示一个
  committed group，cost 边界包含 issue、commit 和所选 wait 协议，不再将结果
  误当成裸 WGMMA latency 后乘指令数。
- `compute/{sfu,fp32_alu,convert,shuffle,integer}`: softmax、LSE、归一化、地址与
  metadata 控制所需的 inline-PTX 标量指令。
- `compute/{synchronization,ordering}`: CTA/WG/named-barrier 开销与 async proxy
  visibility fence。等待语义仍由 DAG completion edge 表达。
- `memory/{tma_load,tma_store,bulk_store,matrix_movement}`: Q/K/O、P exchange、
  split staging 的通用搬运协议。
- `memory/{shared_load,shared_store,global_load,global_store,tensormap_prefetch}`:
  metadata、block table、L reduction、combine 和 descriptor 操作。
- `resource/*`: Tensor/TMA/L2/HBM/grid service 和低层 interference 曲线。这些是
  正式预测输入，但仍保持 operator-agnostic。

每个源文件、binary、JSON `name` 和 manifest ID 完全一致。family 的
`result.csv` 只保存 remote H800 最新一次完整 full sweep；quick、失败或部分扫描
只进入 `build/logs` 与 `build/raw`。

## Dynamic DAG expansion

模型按源码动态展开，而不是用固定的每页加法公式：

- first score 等待首 page 的 9 个 K TMA，然后按源码以一个包含 36 次 SS
  WGMMA 的 committed group 发射；steady score 才是 9 个四指令 group。
- 后续 score 为 8 个 SS group 加 tile 8 的 1 个 RS group，保留逐 tile TMA-ready
  和真实 `wait_group<4/1/0>` 约束。
- 每个有效 page 建 local/remote PV group、P STSM、proxy fence 和两个 WG 的
  named-barrier rendezvous。
- `N_page=0/1/even/odd`、causal tail、pair drain、single drain、split/no-split
  epilogue 分别走源码对应分支。
- metadata、persistent CTA 跨 request、PDL 和 combine 与 main 位于同一张 DAG。

Phase 只是节点标签。节点之间只有源码支持的 program/data/barrier/TMA/WGMMA/
buffer-reuse/grid-dependency 边，phase 边界不会自动增加 barrier。

## Calibration boundary

[`calibration/`](calibration/) 中的 first score、steady score、page-pair、softmax、
store、combine 和 PDL probe 可以覆盖同一 DAG 子图，因此不能相加，也不能成为
正式 cost。每个 probe 有自己的 `probe_dag`，只用于比较 atom-only prediction 与
实测边界的 residual。Residual 只能推动源码 DAG、动态计数或 generic benchmark
coverage 修正，不能生成 correction factor、offset、倍率或 overlap credit。
