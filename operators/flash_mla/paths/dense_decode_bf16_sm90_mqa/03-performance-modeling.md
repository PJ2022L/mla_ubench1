# 03 - Atom-DAG performance model

正式预测边界是 GPU 上的 metadata kernel、persistent main、PDL 和 combine
critical path。它不包含 Python、allocator、host launch 或输入准备时间。

## 1. Global DAG

[`model/dag.py`](model/dag.py) 先复刻 metadata scheduler，再为每个 main/combine
CTA 展开源码控制流。一个 `OperationNode` 对应一个 generic microbenchmark
operation；TMA/WGMMA 同时具有 issue 与 completion 事件。边只使用以下语义：

```text
program, data, barrier, tma_ready, wgmma_wait,
memory_visibility, buffer_reuse, grid_dependency
```

Canonical phase 包括 metadata、CTA init、request setup、first score、pair update、
steady score、tail update、L reduction、split/no-split epilogue 以及 combine dispatch/
LSE/accumulate/store。每个节点只归属一个 phase，但整张 DAG 只调度一次；绝不能
分别预测 phase 后简单求和。

模型显式保留：

- 首 page 的 all-K-ready QK，后续 page 的逐 tile TMA-ready QK。
- 两个 WG 的 `sMInitialized`、`sScale0Ready`、`sScale1Ready`、`sP0Ready` 和
  `rO1sP0sV0RIssued` 关系。
- WG0 的 P 交换保留 STSM -> proxy fence -> named barrier；WG1 steady 路径保留
  STSM -> local PV issue -> proxy fence -> named barrier，再由 peer WG 发射 remote PV。
- K0/K1 半 buffer 覆盖前对 local/remote PV completion 的等待。
- persistent CTA 只允许下一 request 的 Q TMA 提前；Q8/K/compute 仍等待前一
  request store protocol 和 CTA sync。
- split partial store、PDL trigger/wait、combine no-op 和 active combine load 的
  visibility dependency。

## 2. Atom costs and resources

[`model/cost_database.py`](model/cost_database.py) 只加载 manifest 登记的
`microbench/**/result.csv`。缺少 DAG atom 或 mandatory sweep parameter 会直接
报 coverage error；prediction API 不接收 calibration 路径。

异步 operation 使用测得的 dependency latency 和 initiation interval，多个 group
按 `latency + (N-1)*II` 进入事件调度，而不是按完整 latency 串行。Tensor/TMA/SFU/
FP/shared/barrier/issue、PDL grid 和 global-memory LSU issue 使用 per-SM queue；L2/HBM
是全卡共享 byte queue。Load miss 按 HBM -> L2、store 按 L2 -> HBM 有序服务。
Cache fill 只在 modeled L2 completion 后生效，并通过 fixed-point replay 消除 DAG
构造顺序对 hit/miss 的影响。

Physical page、KV head、tile、reuse distance、working set 和 cache mode 决定 K/Q
traffic。Split partial O/LSE 通过 main store 与 combine load 的 request-local
producer/consumer identity 建模；完整 request working set 超过 L2 时不会被无条件
视为 hot。

[`model/resources.py`](model/resources.py) 使用真实 cubin 的 registers、shared
memory 和 threads 计算 main/combine residency、CTA placement、wave；`__launch_bounds__`
的第二参数只作为编译期 minimum-block consistency check，不作为 residency 上限
和 tail。generic `microbench/resource/*` 曲线提供 active SM、resident CTA、WG、
outstanding depth、working set 和 mixed-resource slowdown。每个 node 按所在 CTA 的
实际 wave 查询 `blocks/active_sm/resident_cta`；combine 首 wave 还要等待同 SM/slot
上的最后 main wave 释放 occupancy。资源 JSON 只定义结构，不允许手写 HBM
fraction 或经验 correction。

## 3. Output semantics

模型输出 `p10/p50/p90` GPU cycles/us、CTA wave/tail、split distribution、L2/HBM
traffic、资源利用率、critical path 和所有 atom/resource-curve provenance。

- `wall_span_cycles`: phase 在全局时间轴上的跨度，phase 之间允许重叠，不能相加。
- `critical_path_contribution_cycles`: critical path 按 phase 归因，所有 contribution
  之和严格等于预测 E2E cycles。

```bash
python3 -m operators.flash_mla.paths.dense_decode_bf16_sm90_mqa.model build-dag \
  --workload workload.json --kernel-resources dense-resources.h800.json \
  --output dag.json

python3 -m operators.flash_mla.paths.dense_decode_bf16_sm90_mqa.model predict \
  --microbench-root microbench --kernel-resources dense-resources.h800.json \
  --workload workload.json --output prediction.json
```

[`compose.py`](compose.py) 只是兼容入口，内部调用同一个 DAG CLI；旧的 additive
cycle dictionary、schedule-level cost 和 calibration cost 输入已明确禁用。

## 4. Calibration and held-out validation

`validate-calibration` 为每个 operator probe 构建与其计时边界一致的 probe DAG，
输出 atom prediction、measured cycles、residual、relative error、status 和 suspected
resources。Calibration 模块不会被 prediction import，修改 calibration 数据必须
保证正式预测逐位不变。

Remote H800 最终使用 public dense decode CUDA-event 区间做 held-out 验证。E2E
数据不参与基础参数拟合；目标为 MAPE <= 10%、P90 APE <= 15%，并同时核对 scheduler
split、CTA wave/tail 和 profiler 的主要瓶颈。
