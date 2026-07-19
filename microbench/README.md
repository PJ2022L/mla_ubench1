# FlashMLA Dense Decode Microbench

SM90a/H800 专用 benchmark 与 E2E 模型。只使用基础 CUDA、Driver API 和
inline PTX，不依赖 CUTLASS/CUTE。

## Contents

```text
compute/             31 compute atom leaves
memory/              20 memory atom leaves
model/calibration/   14 interaction calibration leaves
model/dense_decode/  scheduler 与离散事件 E2E 模型
common/              PTX、计时、JSON、tensor-map 公共代码
config/h800.json     quick/full 扫参
manifest.json        binary、参数、PTX/SASS 和源码归属
scan.py              统一 remote runner
scripts/             Makefile/静态检查内部依赖
```

每个 atom leaf 是一个独立 binary。51 是 31 compute + 20 memory 的独立测量
配置数，不是 51 个不同 PTX mnemonic。

## Build

以下命令只编译和反汇编，不运行 GPU：

```bash
make -C microbench -j8 everything
make -C microbench static
make -C microbench static-calibration
```

## Scan

先查看命令空间：

```bash
python3 microbench/scan.py --kind all --preset quick --dry-run
python3 microbench/scan.py --kind atom --preset full --dry-run
python3 microbench/scan.py --kind calibration --preset full --dry-run
```

只在 remote H800 运行：

```bash
python3 microbench/scan.py --kind atom --preset quick \
  --output-dir microbench/results/<run-id>/quick/atoms

python3 microbench/scan.py --kind atom --preset full \
  --output-dir microbench/results/<run-id>/profile/atoms
```

## Output

每个扫描目录最多有：

```text
results.jsonl   正式六键结果
run.log         JSONL：runner args、每个 case 的 command 和 duration
failures.jsonl  仅失败时生成
```

正式结果顶层固定为：

```text
name, params, latency, throughput, memory_bandwidth, hardware_utilization
```

`results.jsonl` 不存 runner 时间戳、wall time 或命令。metric 内的 raw samples
属于正式性能数据，需要保留给 bootstrap。`run.log` 开头只记录一次 GPU 环境，
避免每条结果重复环境快照。

## Model

```bash
python3 -m microbench.model.dense_decode build-profile \
  --microbench-results microbench/results/<run-id>/profile \
  --static-artifacts microbench/results/<run-id>/static \
  --output microbench/results/<run-id>/h800-profile.json

python3 -m microbench.model.dense_decode predict \
  --profile microbench/results/<run-id>/h800-profile.json \
  --workload microbench/model/dense_decode/workload.example.json \
  --bootstrap 1000 --output prediction.json
```

quick、失败和 held-out E2E 结果不得作为 profile 输入。完整 remote 执行顺序见
根目录 `HANDOFF.md`。
