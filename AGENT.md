# AGENT.md — GPU kernel 拆解仓库规范

本仓库把真实 GPU operator 的具体 kernel path 拆成可复用的 micro-benchmark，并用解析模型重建端到端性能。当前只分析 **NVIDIA SM90/Hopper**。

## 根目录路由

```text
operators/                 按 operator 管理上游源码和具体 kernel path
  <operator>/
    target/                上游源码快照，只读
    paths/                 该 operator 的可分析 kernel path
microbench/                跨 operator 复用的 micro-benchmark
  index.md                 强制去重与路由表；新增原子前先查
  common/                  构建、计时、TMA、shape 等公共基础设施
  memory/<family>/<config>/ 数据移动、cache、shared memory、DSM、TMA
  compute/<family>/<config>/ Tensor Core、格式转换、SFU/归约
.qoder/skills/             仓库本地分析 skill；不是业务目录
```

`operators/_references/` 不是 operator：其中 `ubench/`、论文和其他 vendored 内容只读；仓库自写的精炼指南可直接放在 `_references/` 根部。不要修改 `operators/*/target/` 或 vendored reference tree；日常分析只在 `paths/`、`microbench/`、自写指南和本文件中开发。

## Kernel path 命名

path 必须具体到一个 kernel/实例化，使用小写 snake_case，并尽量编码：

```text
<stage>_<dtype/cache>_<arch>_<mode>_<关键调度特征>
```

示例：

- `dense_decode_bf16_sm90_mqa`
- `sparse_decode_fp8_sm90_v32_mqa_h128_cluster2`
- `sparse_prefill_bf16_sm90_mqa`

不要建立含混的 `decode/`、`sm90/` 或 `new_kernel/` 目录。当前不得新增 SM100 path；先把 SM90 的分析和测量口径跑通。

## 每个 path 的固定文档

每个 `operators/<operator>/paths/<kernel-path>/` 必须包含：

0. `00-config.md`：明确 path 名对应的精确实例；分开记录编译期固定参数、path 主动限定参数、合法运行时扫描面，以及上游支持但不属于该 path 的 variants。代表性能 shape 不能冒充编译期常量。
1. `01-kernel-implementation.md`：数学问题、参数/shape、grid/CTA 映射、warp/warpgroup 分工、shared/register 所有权、同步和流水线图。
2. `02-atom-decomposition.md`：只选明显耗时的访存或计算原子，尽量下沉到 `ld`、TMA、WGMMA、格式转换等指令族；给出计数、输入布局和对应 microbench。
3. `03-performance-modeling.md`：把原子周期组合成 kernel 和 e2e 预测。固定语义：`+` 为依赖串行，`max` 为可证明重叠，`N × T` 为循环/动态指令次数。
4. `e2e/`：真实 operator API 的 correctness/performance 基线，至少报告完整 shape、稳态 latency 和 FLOPS/bytes 口径。

另放一个 `README.md` 做源码锚点、variant 元数据、文档状态和路由，不在 README 复制各文档正文。

e2e 必须明确测量边界。scheduler metadata、编译、数据生成属于 setup，不计入稳态；main+combine 等正常执行路径上的 GPU kernel 必须计入。单独 main-kernel 时间只能叫 kernel latency，不能叫 e2e。

撰写实现文档时必须使用 `.qoder/skills/kernel-analysis-skill`：先读 kernel 和 launch site，再区分 **Confirmed / Derived / Inference**。流水线 SVG 由 skill 的 `render_pipeline_svg.py` 生成；图的横轴默认是逻辑时间，不能伪装成实测 cycle。若源码没有 WG 内 overlap，明确画/写串行依赖，不套用 FlashAttention-3 的效果。

## Micro-benchmark 选择规则

设计、实现或审查 GPU micro-benchmark 时，必须使用 [`.qoder/skills/cuda-microbenchmark-skill`](.qoder/skills/cuda-microbenchmark-skill/SKILL.md)。先区分依赖延迟、串行完成周期、单 SM 吞吐和整 grid 吞吐，再按任务读取其中的计时、访存、计算、验收分册及独立示例。

只为满足下列条件的原子建 benchmark：

- 在目标 path 中随 block/tile/token 重复，动态工作量显著；
- 是潜在关键路径或会占满独立硬件管线；
- 能从 operator 中隔离，且输入布局仍与真实 kernel 一致；
- 输出能归一为 `cycle/op`、`op/clk/SM`、`byte/clk/SM` 或 `flop/clk/SM`。

不要单独测地址加法、一次性谓词、轻量 barrier bookkeeping 等小负载，除非 profiler 证明它们显著。紧密数据依赖且拆开会失真的指令可组成一个原子，例如 online softmax 的 `max + exp2 + sum + rescale`。

分类规则：

- `microbench/memory/`：`ld/st.global`、TMA、`st.shared`、DSM、cache/访存模式。
- `microbench/compute/`：WGMMA、FP8/BF16 转换、softmax/SFU、数值归约。

固定层级为：

```text
microbench/<memory|compute>/<instruction-family>/<parameter-configuration>/
```

family 表示指令大类，例如 `wgmma`、`tma_load`、`global_load`、`softmax`。配置叶子写清 shape/dtype、operand mode 和架构，例如 `compute/wgmma/m64n64k16_bf16_rs_ss_sm90/`。不得使用 `a1`、`atom4` 或某个 operator 私有名称。

family 拥有共享 `benchmark.cu`；配置叶子只用 Makefile 的 `DEFINES` 固定参数，并包含自己的 README。只有指令/数据路径确实不同才新建 family；只增加 MNK、tile、dtype 或 rank 时，在既有 family 下新增配置。

每个 family 至少有：

- `benchmark.cu`：只包含目标工作和防优化依赖；

每个配置叶子至少有：

- `README.md`：被测指令/指令簇、真实输入几何、独立变量、计时排除项、输出指标、使用它的 operator paths；
- `Makefile`：设置 `SOURCE := ../benchmark.cu`，用 `DEFINES` 固定配置，复用 `microbench/common/common.mk`。

`microbench/index.md` 是唯一 registry。新增 operator 的 `02-atom-decomposition.md` 前，按 `family + operand mode + shape + dtype + arch` 查表：精确匹配必须复用；同 family 新参数只新增配置；新配置先登记 index，再被 operator 文档引用。`scaffold` 不得当作实测结果。

## Modeling 规范

先把所有量换算到同一粒度（推荐 cycle/CTA/block）：

```text
T_chain   = T_a + T_b                         # 数据依赖，串行
T_overlap = max(T_producer, T_consumer)       # 不同执行单元可证明重叠
T_loop    = N × T_iteration                   # N 次循环/指令发射
```

两级 producer/consumer 流水必须同时包含 prologue、稳态和 drain。对 `N` 个 block，不能漏掉最后一个 consumer：

```text
T_pipeline = T_first_ready + (N-1) × max(T_P, T_C) + T_C + T_tail
```

每个 `max` 都必须能在实现文档的 WG/stream 分工和同步边上找到证据。给出理想重叠下界、全串行上界和 e2e 标定比；不要把拟合常数伪装成硬件事实。

## 工作流与验证

1. 固定 upstream commit、kernel 实例化和代表 shape。
2. 用 kernel-analysis skill 完成实现文档和流水线图。
3. 从热点循环提取动态指令计数，更新拆分文档。
4. 先查 `microbench/index.md`；精确匹配直接复用，缺少时才登记并新增最小配置。
5. 用 ncu/PTX/SASS 验证被测指令和资源隔离。
6. 运行原子、e2e、model，检查实测是否落在模型上下界之间。

开始执行前先在仓库根目录运行 `pwd`，确认当前 remote checkout，再直接使用 `python`；不要假定固定绝对路径或 conda 环境名。CUDA 语法检查使用 `nvcc -arch=sm_90a`；真实性能结论必须来自 H800/SM90 机器。
