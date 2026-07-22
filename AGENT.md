# Remote H800 Agent Guide

本仓库的目标是用 kernel-agnostic、inline-PTX microbenchmark 预测 FlashMLA
dense decode 的 GPU 区间：metadata、SM90 persistent main、PDL 和 combine。覆盖
BF16/FP16，不预测 Python、allocator、输入准备或 host launch 时间。

开始 remote 工作前先读 `HANDOFF.md`。当前没有 accepted H800 性能数据。

## 当前状态

- `microbench/manifest.json`：67 entries / 21 families，其中 60 operation、
  7 resource curve。
- dense calibration：14 probes，只做 residual，不参与正式预测。
- 21 个 microbench family `result.csv` 加 calibration `result.csv`，共 22 个，
  当前全部 header-only。
- 本地已通过 67/67 SM90a PTX/SASS/resource 静态门禁、14/14 calibration
  静态编译和 41 项 model 单测。
- 本地没有执行任何 CUDA binary。

## 不可违反的边界

- CUDA binary 只能在 remote H800 / SM90a 执行。本地只做 Python 测试、
  dry-run、NVCC 编译和 PTX/SASS/resource 静态检查。
- 不修改 `operators/flash_mla/target/` 和 `operators/_references/`。
- 正式预测只能读取 `microbench/**/result.csv`、`microbench/manifest.json`、
  workload 和真实 kernel resource contract。
- Calibration 只能输出 atom prediction 与 probe measurement 的 residual；不得提供
  correction factor、offset、倍率、HBM fraction 或 overlap credit。
- Held-out E2E 只验收，不得反向拟合 atom、DAG 或资源参数。
- 同一张 H800 上只允许一个性能 runner；microbench、calibration、E2E 和 profiler
  任务必须串行。

## 目录所有权

```text
microbench/common/                 所有 family 共用的 CLI/计时/JSON/统计
microbench/{compute,memory,resource}/<family>/
  common/                          family 私有 PTX/harness
  <generic-id>.cu                  一个 generic benchmark
  scripts/                         build/sweep 入口
  build/{bin,ptx,cubin,sass,resources,logs,raw}/
  result.csv                       最新 accepted full sweep
microbench/manifest.json           operation/resource-curve contract

operators/flash_mla/paths/dense_decode_bf16_sm90_mqa/
  model/                           DAG、scheduler、cost DB、resource simulator
  calibration/                     dense-specific 高层 residual probes
  e2e/                             public API held-out runner
```

`microbench/common/` 不得放具体 opcode 或 benchmark variant。Operator 角色只能在
`model/atom_map.json` 和 DAG 中出现，不能回流到 generic benchmark 名称。

## 结果与日志

- Quick、筛选运行或失败运行不得覆盖正式 `result.csv`。
- Full sweep 只有在该 family 的全部 case 成功后才原子替换 `result.csv`。
- 正式 CSV 只保存参数、measurement、GPU/clock provenance、source/SASS hash。
- WGMMA 正式 CSV 使用紧凑 `args` JSON；shape/source mode/dtype/layout 等固定
  契约由 benchmark `name` 和 manifest 提供，不在每一行重复。
- 命令、完整 args、UTC 时间、duration、return code 和 stderr 放
  `build/logs/`；原始 JSON 和 samples 放 `build/raw/`。
- Remote accepted full sweep 后，保留对应 full log/raw、SASS、resource usage 和
  hash provenance。不要把 wall time、命令或 profiler 文本写入正式 CSV。

## 入口

```bash
conda activate vla

python3 microbench/scripts/validate_manifest.py
python3 microbench/scripts/build_all.py --dry-run
python3 microbench/scripts/run_all.py --mode quick --dry-run --device 0
python3 microbench/scripts/run_all.py --mode full --dry-run --device 0
make -C microbench static

python3 microbench/scripts/run_all.py --mode quick --device 0
python3 microbench/scripts/run_all.py --mode full --device 0
```

遇到非 H800、时钟/功耗不稳定、GPU 上有竞争负载、manifest coverage 缺失、目标
opcode 不符、PTX local、`LDL/STL`、非零 stack/local、CSV schema 错误或 correctness/
scheduler mismatch 时，停止发布正式结果并保留失败日志。
