# FlashMLA Dense Decode H800 Handoff

## Overview

`microbench/` 是当前唯一 benchmark 套件，只覆盖公开 FlashMLA dense decode：
metadata、SM90 两 warpgroup main、PDL combine、BF16 和 FP16。

已经完成：

- 51 个 atom：31 compute、20 memory。
- 14 个 interaction calibration。
- inline PTX + CUDA，无 CUTLASS/CUTE。
- manifest 驱动的编译、扫描、静态门禁和 E2E 离散事件模型。
- 本地 51/51 atom、14/14 calibration 静态门禁通过。
- 本地 28/28 CPU/Python 测试通过。

尚未完成：任何 H800 实测、真实 main/combine 资源提取、正式 profile 和
held-out E2E 验证。本地没有执行过 GPU binary。

## Result Layout

一次正式实验使用一个 `run_id`：

```text
microbench/results/<run-id>/
  quick/atoms/                 quick smoke，不进入 profile
  quick/calibration/           quick smoke，不进入 profile
  profile/atoms/               full atom results.jsonl + run.log
  profile/calibration/         full calibration results.jsonl + run.log
  static/                      真 kernel 资源摘要与必要 SASS hash
  h800-profile.json            最终硬件 profile
  heldout/results.jsonl        E2E 留出实测
  heldout/run.log              E2E args、命令、运行耗时
  validation.json              最终误差报告
```

扫描目录只保留 `results.jsonl` 和 `run.log`；`failures.jsonl` 仅失败时出现。
不要生成重复的 `results.json`、summary CSV 或逐条环境快照。

协调 agent 先创建一个共享 run id，并把这两个变量传给每个任务：

```bash
export RUN_ID="$(date +%Y%m%d-%H%M%S)-$(hostname)"
export RUN_ROOT="microbench/results/${RUN_ID}"
```

Task 3 和 Task 4 可以分给不同 agent，但同一张 H800 上必须串行执行，禁止两个
性能扫描同时运行。每个 agent 只写自己负责的子目录。

## Task 1: Environment And Static Gate

依赖：无。交付：环境摘要、完整编译和静态门禁通过。

```bash
pwd
git status --short
nvidia-smi -L
nvidia-smi --query-gpu=name,uuid,driver_version,pstate,clocks.current.sm,clocks.current.memory,power.draw,power.limit,temperature.gpu --format=csv
nvcc --version

make -C microbench -j8 everything
make -C microbench static
make -C microbench static-calibration

python3 -m unittest \
  microbench.tests.test_dense_decode_model \
  microbench.tests.test_profile_and_scan \
  operators.flash_mla.paths.tests.test_cpu_tools -v
```

停止条件：不是 H800/SM90、任一 target 缺目标 opcode、出现 `LDL/STL`、
`STACK/LOCAL != 0` 或编译失败。

## Task 2: Quick Correctness Smoke

依赖：Task 1。交付：quick atom/calibration 无失败。

```bash
python3 microbench/scan.py --kind atom --preset quick \
  --output-dir "${RUN_ROOT}/quick/atoms"

python3 microbench/scan.py --kind calibration --preset quick \
  --output-dir "${RUN_ROOT}/quick/calibration"
```

检查两个 `run.log` 的 `run_end.status=ok`。存在 `failures.jsonl` 就停止，先修复
正确性、资源、时钟或参数问题。

## Task 3: Formal Atom Sweep

依赖：Task 2。交付：51 个 atom 的 full 曲线。

```bash
python3 microbench/scan.py --kind atom --preset full \
  --output-dir "${RUN_ROOT}/profile/atoms"
```

不要把 quick 结果复制到 `profile/`。检查 block/WG、working set、cache pattern、
split count 和 BF16/FP16 扫描完整。

## Task 4: Formal Interaction Sweep

依赖：Task 2，可与 Task 3 顺序执行。交付：14 个 calibration 的 full 曲线。

```bash
python3 microbench/scan.py --kind calibration --preset full \
  --output-dir "${RUN_ROOT}/profile/calibration"
```

KQ、page-pair、softmax、epilogue、metadata、combine 和 PDL 都必须有结果。
模型有 composite 时会替换对应 atom fallback，不要手工把两者相加。

## Task 5: Real Kernel Resources And Profile

依赖：Task 3、4。交付：真实 occupancy contract 与 `h800-profile.json`。

在 H800 构建上游 FlashMLA，保存 dense main/combine 的资源输出和必要 SASS。
然后生成资源 contract：

```bash
python3 microbench/scripts/extract_dense_resources.py \
  --main-resources <main.resources.txt> \
  --combine-resources <combine.resources.txt> \
  --main-dynamic-shared-bytes <sizeof-SharedMemoryPlan> \
  --combine-dynamic-shared-bytes 0 \
  --output "${RUN_ROOT}/static/dense_decode_resources.json"

python3 -m microbench.model.dense_decode build-profile \
  --microbench-results "${RUN_ROOT}/profile" \
  --static-artifacts "${RUN_ROOT}/static" \
  --output "${RUN_ROOT}/h800-profile.json"
```

正式 profile 不允许 `occupancy.*.source=planning_fallback`，也不允许缺少关键
result name。

## Task 6: Held-Out E2E Validation

依赖：Task 5。交付：留出实测、预测和 `validation.json`。

case 必须覆盖 BF16/FP16、整页/tail、短长序列、均匀/偏斜 batch、不同 heads、
warm/cold cache、连续/随机/reuse block table、no-split/split、metadata 生成/复用。

先冻结 workload 和 prediction，再运行公开 API。E2E `results.jsonl` 只保留
case_id、完整 shape、correctness、latency 和 profiler 确认的阶段信息；命令行、
args 和 wall time 写入 `heldout/run.log`。

```bash
python3 -m microbench.model.dense_decode validate \
  --profile "${RUN_ROOT}/h800-profile.json" \
  --cases <heldout-cases.jsonl> \
  --e2e-results "${RUN_ROOT}/heldout/results.jsonl" \
  --output "${RUN_ROOT}/validation.json"
```

验收：MAPE <= 10%，P90 APE <= 15%，并且 split 数、CTA wave/tail、阶段分解和
profiler 主要瓶颈一致。E2E 数据只用于验证，不参与基础参数拟合。

## Known Source Boundary

`get_mla_metadata_kernel` 在部分短序列/小 batch 且 `num_sm_parts` 过大时会在
消费完 request 后继续索引数组。CPU 模型会标记 `source_defined=false`。remote
必须逐 case 对比 metadata 输出，不能用模型拟合掩盖该上游边界。
