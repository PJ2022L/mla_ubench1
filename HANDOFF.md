# FlashMLA SM90/H800 micro-benchmark 交接

本文面向接手远端 H800 实验的 agent。当前已经完成源码 path 分析、18 个原子 benchmark 的核心实现和 SM90a 静态验收；尚未在 GPU 上执行这些 benchmark，也没有可用于性能结论的 H800 实测数据。

远端执行顺序固定为：先完成 dense decode 的 `Plan 0`，验收通过后再进入 sparse decode `Plan 1`，最后执行 sparse prefill `Plan 2`。不要一上来运行全部 18 项并同时推进三条 path。

## 第一章 Overview（项目理解）

### 1.1 项目目标

本项目从固定版本的 FlashMLA 中选择具体 SM90 kernel 实例，把热点控制流拆成可复用的 instruction-family micro-benchmark，再用源码中的同步和依赖关系组合成 kernel/e2e 性能模型。

```text
operators/flash_mla/target/     上游 FlashMLA 源码快照，只读
operators/flash_mla/paths/      三条具体 kernel path 的实现分析、原子拆分和模型
microbench/                     跨 path 复用的 SM90 原子 benchmark
operators/_references/ubench/   外部参考和本项目总结的测量经验
```

固定版本：

- FlashMLA：`9241ae3ef9bac614dd25e45e507e089f888280e0`
- CUTLASS：`147f5673d0c1c3dcf66f78d677fd647e4a020219`
- 外层仓库基线：`66f86afe5fb535139d7c80771e2fbb52b1211b7d`，其上存在预期的 tracked 修改和 untracked 文件

先读：

- [仓库规范](AGENT.md)
- [FlashMLA operator 路由](operators/flash_mla/README.md)
- [micro-benchmark 使用说明](microbench/README.md)
- [18 项 registry](microbench/index.md)
- [静态验收表](microbench/static-validation.md)
- [Hopper micro-benchmark 设计经验](operators/_references/ubench/microbenchmark-design-notes.md)

### 1.2 三条 path 与执行优先级

| Plan | Path | 核心结构 | 关键建模边界 |
|---|---|---|---|
| Plan 0 | [dense decode BF16](operators/flash_mla/paths/dense_decode_bf16_sm90_mqa/README.md) | 2 个 WG；首 page QK 为 36 SS WGMMA；steady page 为 `32 SS + 4 RS`；page-pair 调度 | 两个 WG 共享 Tensor Core，不能把单 WG 原子简单相加或取 `max`；必须补 schedule-level composite，并用 main+combine e2e 标定 |
| Plan 1 | [sparse decode FP8 cluster2](operators/flash_mla/paths/sparse_decode_fp8_sm90_v32_mqa_h128_cluster2/README.md) | indexed load、FP8 convert、shared/DSM store、双 WG compute、persistent cluster2、split-KV combine | producer/consumer overlap、双 WG PV、cluster scheduler 和 combine 需要组合测量 |
| Plan 2 | [sparse prefill BF16](operators/flash_mla/paths/sparse_prefill_bf16_sm90_mqa/README.md) | 单 CTA、3 WG、cluster size 1；每 pair 9,216 条 16 B `cp.async`，调度为 `4/5/5/4` | 没有 DSM、FP8 convert 或 split-KV，不能套用 sparse decode 模型 |

每条 path 已有 `00-config.md`、`01-kernel-implementation.md`、`02-atom-decomposition.md`、`03-performance-modeling.md` 和 `e2e/`。dense/sparse decode 另有 `compose.py`。模型中的 overlap 只是源码允许的候选关系，不是已经测得的硬件事实。

### 1.3 micro-benchmark 的测量口径

- 短、单 CTA/warpgroup/cluster 区间使用设备 `%clock64`。每个 sample 的 cycle 是同步参与线程的 `max(stop)-min(start)`。
- 默认至少 5 次 warmup、20 次正式 sample，输出 median、p10、p90 等统计。
- `cycle_per_*` 是设备周期；`*_per_clk_sm`/`*_per_clk_cluster` 是周期归一化速率，不是 ns、GB/s 或整卡 TFLOPS。
- 真正依赖延迟要求下一次操作依赖上一次完成；多独立链或多个 outstanding group 测的是吞吐。当前不能把所有 `cycle_per_*` 都叫作单指令 latency。
- 整 grid 饱和吞吐和 e2e 使用 CUDA event。DVFS 下不要用名义 `clockRate` 把 host 时间硬换成 cycle；应锁定或记录实际 SM/memory clock、P-state、功耗和温度。
- correctness、输入初始化、descriptor 创建和 host/device 分配在 timed region 外；语义必需的 barrier、fence、commit/wait 保留在 timed region 内。
- 最终以对应 binary 的 PTX/SASS 为证，核对 opcode、operand mode、动态分母、寄存器和 spill。

### 1.4 当前完成状态

已经完成：

1. 拉取并固定 FlashMLA/CUTLASS，两个上游目录保持 clean。
2. 三条 path 的实例化、CTA/WG 分工、流水线、动态计数、原子拆分和模型文档。
3. 18 个 SM90 micro-benchmark 的 CUDA 核心代码、公共计时/统计/校验工具、Makefile、README、registry 和批处理脚本。
4. 本地静态验证：18/18 使用 `-arch=sm_90a` 生成 PTX/cubin/SASS；目标 opcode 命中；所有 kernel `STACK=0, LOCAL=0`，无 PTX local memory 和 SASS `LDL/STL`。
5. CPU/Python 单元测试 13/13 通过，三个 e2e `--validate-only` 通过。

尚未完成：

- 没有 H800 micro-benchmark 或 e2e 性能数据，registry 18 项均只能是 `validated`，不能标成 `measured`。
- dense decode 需要的 first-page KQ、steady-page KQ 和 two-WG page-pair transition composite 还没有可执行实现。
- 当前 dense [e2e/benchmark.py](operators/flash_mla/paths/dense_decode_bf16_sm90_mqa/e2e/benchmark.py) 只做参数检查和 CUDA-event 计时，没有 `--check`/PyTorch reference correctness 路径。正确性必须先复用上游 reference test，或给该脚本补上同等对拍能力。
- 当前 micro-benchmark JSON 输出统计摘要，不输出逐次 raw sample。若实验要求 raw samples，先扩展公共输出并验证，再开始正式 sweep。
- 当前多数 memory benchmark 输出 per-SM/CTA proxy，不是 CUDA-event 整卡带宽。

### 1.5 工作区保护和 `measured` 定义

外层仓库是 dirty worktree，并包含 untracked benchmark 目录。不要执行 `git reset --hard`、`git clean -fd`、`git checkout -- .` 或任何会丢弃未提交内容的命令。若 H800 是另一台机器，必须同步完整工作区快照；只 clone/submodule update 得不到本次全部实现。

只有同时满足以下条件，registry 才能从 `validated` 改为 `measured`：

1. 结果来自确认过的 H800 `(9,0)`，环境、commit/diff、完整 CLI 和时钟策略已归档。
2. warmup/sample 达标，保存 JSONL；若补了 raw sample 输出，也一并归档。
3. checksum、CPU reference、untouched-zero 等适用校验通过，进程退出码为 0。
4. 本次 binary 的 PTX/SASS opcode、operand mode、动态计数和无 spill 状态已核对。
5. 重复实验稳定，无明显热降频；cache/working-set 状态有明确标签。

## 第二章 执行层面（动手在 H800 上跑实验）

### 2.1 通用执行规则

- 远端不假定固定目录或 conda 环境。先在仓库根目录运行 `pwd` 并校验仓库标志文件，随后直接使用该环境中的 `python`；性能数字只能在远端 H800 上产生。
- 结果必须就近归档，不建立 `results/h800/...` 中央副本：micro-benchmark 写入对应配置的 `result/`，e2e 写入对应 `e2e/result/`，模型输入、预测和实测对比写入对应 path 根目录的 `result/`。
- 每个 run 使用不可变 `run_id=YYYYMMDD-HHMMSS_<hostname>_<8位ID>`。同一 phase/批次应显式复用同一个 `RUN_ID`，以便跨 micro-benchmark、e2e 和模型结果追踪；已经存在的 run 目录不得覆盖。
- 一个 phase 验收通过后再进入下一个 phase。遇到 CUDA error、checksum mismatch、SASS 不一致、spill、温度/时钟异常时，停在当前 phase 修复和重跑。
- 单项 micro-benchmark 使用 `make run ARGS='...' RUN_ID='<optional>'`；runner 在 `result/runs/<run_id>/` 保存 `metadata.json`、规范化 `result.jsonl` 和完整 `run.log`。失败/解析失败的 run 也保留，但不能 accepted。
- e2e 的 `result/runs/<run_id>/` 使用同样的 metadata/JSONL/log 契约；path 根 `result/runs/<run_id>/` 另外保存 `cycles.json`、`provenance.json`、`predictions.jsonl` 和 `comparison.csv`。
- 每层 `summary.csv` 必须能由不可变 `runs/` 重建。micro leaf README 总结 `H800 Results`，e2e README 总结完整 shape/correctness/latency/main-combine 边界，path README 总结 `Model vs Measured`；没有 accepted run 时保留明确空状态，不填示例值。
- `comparison.csv` 只比较作用域和单位一致的数据。不得用名义 SM 时钟把 per-CTA `%clock64` cycle 直接换成 CUDA-event e2e ms；没有 event-level 预测时，预测 e2e 和相应误差列留空。
- Plan 0 完成前，不执行全量 `./microbench/run_all.sh run`，也不并行推进 Plan 1/2。

模型 run 必须同时提供 compose 输入和逐字段来源，runner 会在创建不可变目录前验证 provenance 指向的 JSONL record/metric 与 cycle 值一致：

```bash
python tools/result_tool.py run \
  --result-dir operators/flash_mla/paths/<path>/result \
  --kind model \
  --cycles-json <cycles.json> \
  --provenance-json <provenance.json> \
  --comparison-csv <comparison-input.csv> -- \
  python operators/flash_mla/paths/<path>/compose.py \
    --cycles-json <cycles.json> --n-page <N>
```

model stdout 会规范化为 `predictions.jsonl`。输入 comparison CSV 使用固定表头，至少填写 `case_id`、一种实际测量值、`microbench_run_ids`，若填写 e2e 实测还必须填写 `e2e_run_id`；`model_kind`、预测值和同单位 signed error 可以留空，由 runner 根据模型输出补全并写入归档的 `comparison.csv`。

`provenance.json` 的每个 key 必须与 `cycles.json` 一一对应，值必须包含 `source_file`、`record_index`、`metric` 和 `run_id`。runner 只接受仓库内 `result/runs/<run_id>/result.jsonl` 中、metadata 标记成功的 micro record，并将路径规范化为仓库相对路径。

若 comparison 填写 `measured_composite_cycles`，单 case 的 `cycles.json` 必须包含同值的 `T_measured`，多 case 必须分别包含 `T_measured__<case_id>`，并为这些 key 提供同样严格的 micro provenance。若填写 `measured_e2e_ms`，runner 会从当前 path 的 `e2e/result/runs/<e2e_run_id>/result.jsonl` 读取成功 run，并核对其中的 `latency_ms`。多 case 模型输出必须自己携带唯一 `case_id`，runner 按 ID 关联，不依赖输出顺序。

### Plan 0：dense decode BF16

Plan 0 的完成标准不是“binary 能跑”，而是 dense 相关原子、dense-specific composite、公开 API correctness、e2e 性能和模型回填形成闭环。

#### Phase 0A：同步工作区并冻结 H800 环境

目标：确认拿到完整实现，并记录可复现实验环境。

```bash
repo_root="$(pwd)"
test -f "$repo_root/HANDOFF.md"
test -f "$repo_root/AGENT.md"

git -C "$repo_root" status --short
git -C "$repo_root" submodule status --recursive
test -f "$repo_root/microbench/static-validation.md"
test -f "$repo_root/microbench/compute/wgmma/benchmark.cu"
test -f "$repo_root/operators/flash_mla/paths/dense_decode_bf16_sm90_mqa/e2e/benchmark.py"

nvidia-smi -L
nvidia-smi --query-gpu=name,uuid,driver_version,pstate,clocks.current.sm,clocks.current.memory,clocks.max.sm,power.limit,ecc.mode.current,temperature.gpu --format=csv
nvcc --version
python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))'

git -C "$repo_root/operators/flash_mla/target" submodule update --init --recursive
FLASH_MLA_DISABLE_SM100=1 python -m pip install -v "$repo_root/operators/flash_mla/target"
```

验收门槛：GPU capability 为 `(9,0)`；CUDA 满足 FlashMLA 要求；FlashMLA/CUTLASS commit 正确；`target/` 和 CUTLASS 内部 clean；环境快照已归档。未通过时不要进入 0B。

#### Phase 0B：只构建 dense 相关 micro-benchmark 并复核静态证据

目标：使用 H800 机器上的 CUDA/toolchain 重新生成 dense 子集 binary/PTX/SASS。

Dense 子集固定为：

```text
compute/wgmma/m64n64k16_bf16_rs_ss_sm90
compute/wgmma/m64n256k16_bf16_rs_ss_sm90
compute/softmax/online_m64n64_exp2_shfl_sm90
memory/tma_load/tile64x64_bf16_sm90
memory/tma_load/tile64x576_bf16_sm90
memory/stmatrix/m64n64_b16_x4_sm90
memory/stmatrix/m64n256_b16_x4_sm90
memory/tma_store/tile64x512_bf16_4d_sm90
memory/bulk_store/tile64x512_f32_sm90
memory/splitkv_reduce/dv512_f32_sm90
```

```bash
while read -r bench; do
  make -C "microbench/$bench" compile static
done <<'EOF'
compute/wgmma/m64n64k16_bf16_rs_ss_sm90
compute/wgmma/m64n256k16_bf16_rs_ss_sm90
compute/softmax/online_m64n64_exp2_shfl_sm90
memory/tma_load/tile64x64_bf16_sm90
memory/tma_load/tile64x576_bf16_sm90
memory/stmatrix/m64n64_b16_x4_sm90
memory/stmatrix/m64n256_b16_x4_sm90
memory/tma_store/tile64x512_bf16_4d_sm90
memory/bulk_store/tile64x512_f32_sm90
memory/splitkv_reduce/dv512_f32_sm90
EOF
```

验收门槛：10/10 编译成功；与 [static-validation.md](microbench/static-validation.md) 逐项核对 opcode 和动态数量；`STACK/LOCAL=0`、无 `LDL/STL`；保存 nvcc 版本、PTX/SASS 和 hash。发现远端 toolchain 改变指令形状时，先修正 benchmark/计数，不能沿用本地静态表。

#### Phase 0C：dense 原子 correctness smoke

目标：用小规模、较短 repeat 验证每个 binary 在 H800 上可执行，checksum/CPU validation 全部通过，并确认输出字段和数量符合 README。

必跑变体：

| Family | Smoke 变体 |
|---|---|
| TMA load 64x64/64x576 | `--rank 4 --depth 1`，再各跑一个可接受的 `depth>1` |
| WGMMA 64x64/64x256 | RS/SS、latency/throughput；throughput 使用 `instructions-per-group=4`，`resident-wg=1|2` 至少各一组 |
| softmax | `--mode local` 和 dense 必需的 `--mode dense-pair` |
| STMatrix 64x64/64x256 | `--fence=true|false` |
| rank-4 TMA store | `--depth 1` 和一个 `depth>1`，满足 working-set alias guard |
| bulk store | L2/HBM working set 各一组，确认 global output validation |
| split reduce | 至少 `num-splits=2` 和一个 production-like split 数，确认 CPU reference |

验收门槛：所有进程退出码 0；无 CUDA error；所有 checksum/reference/untouched-zero 校验通过；JSON 数值有限；重复两轮没有数量级漂移。此 phase 只证明可运行和正确，不登记性能结论。

#### Phase 0D：dense 原子正式 sweep

目标：分别建立 latency/completion-cycle、per-SM throughput proxy、cache/working-set 和并发深度曲线。

按以下顺序逐 family 执行，不要把所有参数做笛卡尔积：

1. WGMMA：RS/SS 分开；latency/throughput 分开；`issue-depth=1,2,4,8`，`resident-wg=1,2`；QK/PV 均重点测 `instructions-per-group=4`。
2. TMA load：rank 固定为 dense 的 4；`depth=1` 表示串行 tile completion，随后扫描合法 depth；分别选择 L2-hot 小 working set 和超过 L2 的工作集。`64x576` 每 tile 73,728 B，H800 上实际可用 depth 以 shared-memory guard 为准。
3. softmax：以 `dense-pair` 为主，`local` 作为诊断；记录寄存器、occupancy 和是否 spill。
4. STMatrix：两个 shape 都跑 fence on/off，区分纯 store 和 async-proxy visibility 成本。
5. no-split epilogue：分别测两 WG 的 `m64n256` staging 与 rank-4 TMA store，不把二者误报为一个已经组合的成本。
6. split epilogue/combine：bulk store 扫 L2/HBM 和 CTA 数；split reduce 扫真实 `num_splits`、working set 和 CTA 数。

每个点至少保存完整 CLI、warmup/sample/repeat、JSONL、时钟前后快照和对应 SASS hash。`*_per_clk_sm` 只能叫 per-SM proxy；若需要整卡 GB/s/TFLOPS，必须增加或使用 CUDA-event grid 测量并扫描 CTA 数直到平台。

验收门槛：每条曲线有可解释趋势；同一点独立重跑可复现；没有热降频；cycle 分母与 SASS 动态指令数一致。异常点先重跑和 profile，不直接删除或用 `min` 代替 median。

#### Phase 0E：补齐 dense schedule-level composite

目标：实现独立原子无法表达的真实 TMA/WGMMA/barrier/two-WG 调度。该代码应放在 dense path 下作为 path-specific composite，不要伪装成新的通用硬件原子。

按小 phase 实现和验收：

- Phase 0E-1：first-page KQ。9 个 K TMA barrier-ready 后发射 36 SS QK，复刻 non-pipelined first page。
- Phase 0E-2：steady-page KQ。9 个 tile 逐个 `wait TMA[tile] -> issue 4 WGMMA[tile]`，tile8 使用 RS，得到 `32 SS + 4 RS`。
- Phase 0E-3：two-WG page-pair transition。保留 named barrier、P exchange、half-buffer overwrite、真实 `wait_group<1/4/0>` 和两个 WG 的共享 Tensor Core 竞争。
- Phase 0E-4：补齐 modeling 需要的 prologue/drain/tail 边界：`T_prologue_single`、`T_prologue_pair`、`T_pair_to_single`、`T_pair_drain`、`T_single_drain`。

每个小 phase 都必须先做小 shape correctness/sink 验证，再保存 SASS、动态计数和 cycle 分位数，验收后才进入下一个。不得用 `max(T_TMA,T_WGMMA)` 或两个 standalone WG 的 `max` 代替缺失的 composite。

#### Phase 0F：dense e2e correctness 闭环

目标：公开 `flash_mla_with_kvcache` 的输出和 LSE 对 PyTorch reference 正确，然后才允许测 e2e 性能。

先做 CPU-only 参数检查：

```bash
python operators/flash_mla/paths/dense_decode_bf16_sm90_mqa/e2e/benchmark.py \
  --validate-only --batch 128 --s-q 1 --s-k 4097 --causal
```

当前 e2e 脚本没有 `--check`。remote 必须二选一并留下明确产物：

1. 给 e2e 脚本补 `--check`，复用上游 testcase generator/reference/tolerance，对同一批输入比较 output 和 LSE；或
2. 在同一安装和 commit 下运行 [上游 dense reference test](operators/flash_mla/target/tests/test_flash_mla_dense_decoding.py)，并保存覆盖 Plan 0 shape 的通过记录。

至少覆盖：`s_q=1|2`、整页和非 64 整倍数 tail、causal 生效/不生效、短序列、production-like 长序列，以及会触发 split/combine 的 case。`s_q=1` 时 causal 被上游强制关闭，结果中必须区分 requested/effective causal。

验收门槛：output/LSE 均按上游 tolerance 通过；main 和需要时的 combine 都实际执行；没有 NaN/Inf 或非法内存访问。仅 `--validate-only` 通过不算 e2e correctness 完成。

#### Phase 0G：dense e2e 性能与 kernel 边界

目标：获得 public API 的 main+必要 combine 稳态延时，并用 profiler 区分 main/combine/scheduler 行为。

正式矩阵：

```text
batch = 128
s_q   = 1, 2
s_k   = 4096, 8192, 16384, 32768
tail  = 至少加入 4097 和一个接近页边界的非整倍数
causal = s_q=2 时 false/true；s_q=1 记录 effective=false
```

示例：

```bash
python operators/flash_mla/paths/dense_decode_bf16_sm90_mqa/e2e/benchmark.py \
  --batch 128 --s-q 1 --s-k 4096 --warmup 10 --iters 100
```

CUDA event 必须覆盖一次 public API 的完整 GPU 区间。对代表 case 使用 Kineto/ncu 确认 main、combine、PDL 和实际 `num_splits/N_page`；不能把单 main-kernel 时间称为 e2e。每个点独立重复多轮，保存原始 JSON、profiler 命令和时钟前后状态。

验收门槛：全部性能点已经通过 Phase 0F 对应正确性；重复结果稳定；main/combine 边界清楚；effective TFLOPS/GB/s 明确标为算法最小工作量，不冒充硬件实际 traffic。

#### Phase 0H：模型回填和 Plan 0 完成判定

目标：用实测 composite 为主、独立原子为诊断，对 dense main/e2e 做可解释回填。

将结果按 [dense modeling 文档](operators/flash_mla/paths/dense_decode_bf16_sm90_mqa/03-performance-modeling.md) 的字段写入 cycles JSON，然后运行：

```bash
python operators/flash_mla/paths/dense_decode_bf16_sm90_mqa/compose.py \
  --cycles-json <dense-cycles.json> --n-page <pages-assigned-to-one-CTA>
```

比较 `T_main_model`、e2e、实际 `N_page/num_splits` 和 residual。若偏差无法解释，优先检查动态计数、cache、双 WG WGMMA 竞争、barrier stall、epilogue 和 combine，不增加无物理含义的拟合常数。

Plan 0 只有在以下条件全部满足后才完成：

- dense 10 项原子完成 H800 correctness 和正式 sweep；
- first/steady KQ、page-pair transition、prologue/drain/tail composite 已实现并验证；
- e2e output/LSE correctness 与 public API main+combine 性能均完成；
- PTX/SASS、环境、CLI、JSON/profiler 产物可追溯；
- dense 模型已回填，结论能解释主要误差；
- 只把满足 `measured` 契约的 registry 项更新为 `measured`。

### Plan 1：sparse decode FP8 cluster2

只有 Plan 0 完成后进入。复用 Plan 0 已验证的 WGMMA/TMA/softmax/stmatrix/bulk/split-reduce 数据时，必须确认 shape、rank、operand mode、working set 和作用域完全一致。

- Phase 1A：构建并 smoke sparse decode 新增原子：global indexed load、FP8 convert、shared store、cluster2 DSM store、rank-5 TMA store；验证 cluster lifecycle、peer payload 和 output correctness。
- Phase 1B：扫描 producer 原子和 local/peer、L2/HBM、cluster/CTA 并行度；保留 requested/unique bytes 区分。
- Phase 1C：实现 producer-consumer、双 WG PV 和 persistent cluster2 schedule composite，不能用单原子相加代替。
- Phase 1D：运行 sparse decode e2e reference correctness，再测 public API main+combine；profile scheduler metadata、PDL、split 数和 combine grid。
- Phase 1E：回填 sparse decode `compose.py`，解释 cluster/persistent scheduler/combine residual，完成可追溯归档后再更新状态。

### Plan 2：sparse prefill BF16

只有 Plan 1 完成后进入。该 path 是单 CTA、3 WG、cluster size 1，不引入 DSM/FP8/split-KV。

- Phase 2A：构建并 smoke rank-3 Q TMA、`cp.async` gather、WGMMA、dense/shared-state softmax、STMatrix 和 rank-3 TMA store。
- Phase 2B：重点扫描 `cp.async` block/pair、真实 `4/5/5/4` schedule、L2/HBM working set和 outstanding depth。
- Phase 2C：实现/验证 3-WG prefill schedule composite，检查 Q/KV pipeline、barrier、WGMMA 竞争和 epilogue。
- Phase 2D：运行 sparse prefill public API reference correctness和 e2e CUDA-event 矩阵，用 profiler确认没有被错误套入 decode/split-combine 边界。
- Phase 2E：回填 sparse prefill modeling 文档和结果表；该 path 当前没有独立 `compose.py`，不要复用 sparse decode compose。

三个 plan 全部结束后，最后统一更新 registry、各 leaf README 的 accepted run、三个 e2e README 和三条 path 的 `Model vs Measured`；所有表格必须能从各自就近归档的不可变 runs 重建。
