# FlashMLA Dense Decode Remote H800 Handoff

## 交付目标

本项目用 generic CUDA + inline PTX microbenchmark 构建 FlashMLA dense decode 的
源码级 Atom-DAG，并预测以下 GPU critical path：

```text
metadata kernel -> persistent SM90 main -> PDL -> combine
```

覆盖 BF16/FP16。正式边界不包含 Python、allocator、输入准备或 host launch。

## 已完成

- `microbench/` 已重构为 21 个 family、67 个 generic entries：60 operation、
  7 resource curve。
- WGMMA first score 使用真实协议：36 条 M64N64 SS WGMMA、一次 commit、最终
  wait0；steady score 是 9 个 committed group，每组 4 条，tile 8 使用 RS。
- Dense DAG 已覆盖 metadata、first/steady/tail、两个 WG 的 named barrier、K buffer
  reuse、L reduction、split/no-split epilogue、PDL、combine 和 no-op combine。
- 资源模拟包含 CTA residency/wave、per-SM execution queues、per-SM PDL/LSU issue、
  全卡 L2/HBM queues、HBM->L2 load、L2->HBM store、fill-completion cache replay、
  request-local partial-output reuse 和 same-SM interaction curves。
- Calibration 已移到 dense path，共 14 个高层 probe，仅用于 residual。
- Held-out runner 会检查 correctness、GPU metadata、split prefix 和 CPU scheduler。
- 本地门禁已通过：67/67 microbench 静态编译、14/14 calibration 静态编译、
  41/41 model tests。未执行任何 CUDA binary。

当前 22 个正式 `result.csv` 全部只有表头，因此还不能做正式 H800 prediction。

## 正式数据边界

Prediction 只允许读取：

```text
microbench/manifest.json
microbench/**/result.csv
dense-resources.h800.json
workload.json
```

Calibration `result.csv` 不是 prediction 输入。Calibration residual 只能用于发现
DAG、计数、资源曲线或 probe boundary 问题，不得自动校正预测。Held-out E2E 同样
不能用于拟合。

## Remote 任务

### 1. 环境和唯一 GPU runner

```bash
conda activate vla
pwd
git status --short
nvidia-smi -L
nvidia-smi --query-gpu=name,uuid,driver_version,pstate,clocks.current.sm,clocks.current.memory,power.draw,power.limit,temperature.gpu --format=csv
nvcc --version
```

目标必须是 H800/SM90a。扫描期间保持 GPU 空闲、时钟和功耗策略稳定；只能有一个
agent 执行 GPU 性能任务。

### 2. 编译和静态门禁

以下命令不运行 benchmark binary：

```bash
python3 microbench/scripts/validate_manifest.py
python3 microbench/scripts/build_all.py --dry-run
python3 microbench/scripts/run_all.py --mode quick --dry-run --device 0
python3 microbench/scripts/run_all.py --mode full --dry-run --device 0
make -C microbench static

CAL=operators/flash_mla/paths/dense_decode_bf16_sm90_mqa/calibration
make -C "$CAL" check
make -C "$CAL" dry-build
make -C "$CAL" build
```

必须确认目标 opcode、WGMMA shape/source mode/dtype、TMA/barrier 协议正确，并且无
CUTLASS/CUTE microbench 依赖、PTX local、`LDL/STL`、非零 stack/local。

### 3. Generic microbenchmark 扫描

先 quick 排错，再运行完整 full sweep：

```bash
python3 microbench/scripts/run_all.py --mode quick --device 0
python3 microbench/scripts/run_all.py --mode full --device 0
python3 microbench/scripts/validate_manifest.py
```

失败时查看对应 family 的 `build/logs/` 和 `build/raw/`，修复后只重跑该 family。
不要手工拼 CSV，也不要用 quick/部分数据填正式结果。Full sweep 成功后，每个
manifest entry 必须有 accepted row，并带 GPU UUID、时钟和 source/SASS hash。
WGMMA `result.csv` 是紧凑表：扫描条件在 `args` JSON，固定 shape、dtype、SS/RS
和 layout 契约查 `microbench/manifest.json`，不要重新展开成重复列。

### 4. 真实 kernel resource contract

从 remote 实际构建的 dense main/combine 生成 `dense-resources.h800.json`。字段来源：

- registers、static shared、stack/local：`cuobjdump --dump-resource-usage`。
- main dynamic shared：launcher 的 `sizeof(T::SharedMemoryPlan)` 和 launch config，
  见 `splitkv_mla.cuh` 的 dynamic-smem 设置。
- threads、launch bounds：kernel 源码和对应实例化符号。
- SM 数、L2、实际时钟：目标 H800 查询。

不要把 `model/dense-resources.example.json` 的 planning 值直接当正式数据。

### 5. DAG、prediction 和 calibration residual

从 example 创建 workload，然后执行：

```bash
MODEL=operators.flash_mla.paths.dense_decode_bf16_sm90_mqa.model

python3 -m "$MODEL" build-dag \
  --workload workload.json \
  --kernel-resources dense-resources.h800.json \
  --output dag.json

python3 -m "$MODEL" predict \
  --microbench-root microbench \
  --kernel-resources dense-resources.h800.json \
  --workload workload.json \
  --output prediction.json

python3 "$CAL/scripts/run.py" --preset quick
python3 "$CAL/scripts/run.py" --preset full

python3 -m "$MODEL" validate-calibration \
  --microbench-root microbench \
  --calibration-root "$CAL" \
  --kernel-resources dense-resources.h800.json \
  --output calibration-report.json
```

Residual 阈值：`<=10%` pass、`10%-20%` warn、`>20%` fail。Warn/fail 时修 DAG、
operation count、resource curve 或 probe boundary；禁止生成 correction factor。

Prediction 输出中 `phase_timing.wall_span_cycles` 可以重叠，不能相加；只有
`critical_path_contribution_cycles` 的 phase contribution 之和严格等于 E2E。

### 6. Held-out E2E

先冻结 microbench results、resource contract、workload 和 prediction，再按
`operators/flash_mla/paths/dense_decode_bf16_sm90_mqa/e2e/README.md` 运行 public API。

Case 矩阵至少覆盖：BF16/FP16、metadata generate/reuse、`N_page=0/1/even/odd`、
非 64 整倍数 tail、短长和偏斜 batch、causal、split/no-split、不同 heads、
contiguous/random/reuse block table、L2-hot/HBM-stream。

Accepted case 必须同时满足：

- `--check-correctness` 已执行且通过。
- GPU `tile_scheduler_metadata` 与 CPU scheduler 逐 row 一致。
- 完整 `num_splits` prefix 一致。
- `scheduler_validation.source_defined=true`。
- `acceptance_gate.passed=true`。

验收目标：held-out MAPE `<=10%`，P90 APE `<=15%`，CTA wave/tail、split 数和主要
瓶颈与 profiler 一致。

## 日志和结果保留

```text
<family>/build/bin|ptx|cubin|sass|resources/  编译与静态证据
<family>/build/logs/                           命令、args、UTC、duration、错误
<family>/build/raw/                            原始 JSON 和 samples
<family>/result.csv                            最新 accepted full sweep
```

Remote accepted full 后保留对应 full log/raw、最新 build log、SASS/resource usage
和 hash provenance。失败日志保留到问题关闭。正式 CSV 不保存 wall time、命令、
stderr 或 profiler 文本。

最终 case 只保留高价值产物：`workload.json`、`dense-resources.h800.json`、
`dag.json`、`prediction.json`、`calibration-report.json`、accepted held-out 结果和
对应 `run.log`。

## 已知上游边界

`get_mla_metadata_kernel` 在有效 partition 少于 `num_sm_parts` 时可能读取已消费完的
request。CPU scheduler 会标记 `source_defined=false`，E2E runner 会拒绝该 case。
这类输出只能作为上游边界的诊断证据，不能进入 accepted 集合，也不能通过模型拟合
掩盖。
