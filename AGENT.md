# Remote Agent Guide

本仓库当前交付目标只有 FlashMLA dense decoding，目标机器固定为 H800
SM90a。先读 `HANDOFF.md`，按其中任务顺序执行。

## 目录

```text
operators/flash_mla/target/                         上游源码快照，只读
operators/flash_mla/paths/dense_decode_bf16_sm90_mqa/  源码分析与兼容入口
operators/_references/ubench/                       Hopper 参考，只读
microbench/                                         当前 benchmark 与模型
```

旧 microbench 和根目录 `tools/` 已删除。不要恢复。`microbench/scripts/` 是
Makefile 和模型内部依赖，不是 remote agent 的日常入口。

## 硬约束

- GPU benchmark 只在 remote H800 上运行。
- 本地只允许 Python 测试、NVCC 编译、PTX/SASS/资源静态检查。
- 不引入 CUTLASS/CUTE；benchmark 使用 CUDA Runtime/Driver 与 inline PTX。
- 不修改 `operators/flash_mla/target/` 和 vendored references。
- 不把 quick、失败、held-out E2E 数据混进正式 profile 输入。
- 不根据 held-out E2E 结果拟合 atom 或 calibration 参数。

## 51 个 Atom

51 表示 51 个独立 binary/扫描 leaf，不是 51 个不同 mnemonic：

- compute 31：WGMMA 8、SFU 3、FP32 ALU 5、convert 2、shuffle 1、
  integer/control 3、同步 8、ordering 1。
- memory 20：TMA/bulk 4、STSM/LDSM 3、shared 3、global load 5、
  global store 4、tensor-map prefetch 1。

BF16/FP16、SS/RS、不同 barrier participant 数分别计数，因为它们需要独立
生成目标指令和性能曲线。14 个 calibration 是多原子交互协议，不计入 atom。

## 结果规则

每个扫描目录最多包含：

```text
results.jsonl   正式六键 benchmark 结果
run.log         JSONL 日志：runner args、每个 case 的命令和 wall time
failures.jsonl  仅失败时生成
```

`results.jsonl` 顶层固定为：

```text
name, params, latency, throughput, memory_bandwidth, hardware_utilization
```

不要把 runner 时间、时间戳、命令行或 profiler 文本塞进正式结果。原始性能
samples 保留在对应 metric 内，因为模型 bootstrap 会使用它们。

## 修改要求

- 新 atom 必须登记 `microbench/manifest.json`，一个 leaf 只测一个硬件边界。
- 组合交互放 `microbench/model/calibration/`，不得伪装成 atom。
- 修改 CUDA 后必须执行编译和静态门禁；不得用本地 GPU 运行验证。
- 修改 Python 后运行 `microbench.tests` 与 dense path CPU tests。
