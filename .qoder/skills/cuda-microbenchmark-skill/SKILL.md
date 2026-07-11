---
name: cuda-microbenchmark-skill
description: "设计、实现、审查和解释 CUDA GPU 硬件 micro-benchmark，覆盖指令/流水线延迟、计算吞吐、shared/L1/L2/HBM 访存延迟与带宽、TMA/cp.async/DSM/WGMMA 等 Hopper 异步机制，以及 clock64、inline PTX、CUDA Event、依赖链、并行链、cache 状态、SASS 验证和 DVFS 控制。用于用户要求‘测 GPU 硬件能力’、‘写 micro-benchmark’、‘测 latency/throughput/bandwidth/cycle’、‘怎么计时才准’或审查现有 CUDA benchmark 是否可信时。"
---

# CUDA GPU Micro-benchmark 方法

先定义测量问题，再写 kernel。禁止用同一段循环同时声称得到“单指令延迟”和“峰值吞吐”。

## 执行流程

1. 固定 GPU 架构、目标指令/数据路径、operand shape、作用域和 cache 状态。
2. 将目标归类为：依赖延迟、串行完成周期、单 SM 吞吐、整卡吞吐或组合阶段成本。
3. 选择计时器：短单 SM 区间用 `%clock64`；整 grid 饱和吞吐用同一 stream 的 CUDA Event。
4. 为延迟构造真依赖；为吞吐构造多独立链、outstanding group 和足量 resident CTA/WG。
5. 将初始化、allocation、descriptor 创建、首次 page fault 和普通数据准备移出 timed region；保留语义必需的 fence/barrier/wait。
6. 建立防优化 sink 和 correctness 校验；异步 store/load 必须验证目标内存确实更新。
7. 保存 PTX/CUBIN/SASS，核对 opcode、operand mode、vector width、cache modifier、动态指令数、寄存器和 spill。
8. 预热、重复采样、报告 median/p10/p90；记录实际时钟、功耗、温度、ECC、CUDA/driver/nvcc 和完整 CLI。
9. 用明确分母输出 `cycle/op`、`op/clk/SM`、`byte/clk/SM`、GB/s 或 TFLOPS；写清 `/thread`、`/warp`、`/warpgroup`、`/CTA`、`/SM` 或 `/cluster`。

## 按任务读取分册

在实现或审查前，完整读取与任务匹配的分册：

- 计时器、inline PTX、同步、DVFS、统计：[计时与 CUDA 特性](references/01-timing-and-cuda-features.md)
- shared/L1/L2/HBM、load/store、TMA、带宽：[访存类方法](references/02-memory-methodology.md)
- ALU/SFU/convert、Tensor Core、WGMMA：[计算类方法](references/03-compute-methodology.md)
- correctness、SASS、环境、结果契约和常见错误：[准确性与验收](references/04-validation-and-reporting.md)
- 本仓库 `operators/_references/ubench` 的对应源码位置：[参考代码路由](references/05-reference-map.md)

## 示例路由

示例只展示一种测量意图，不要互相替代：

- 单 load 真依赖延迟：[全局访存 pointer chase](examples/01-global-load-latency.md)
- 多 SM 饱和读带宽：[全局访存吞吐](examples/02-global-memory-bandwidth.md)
- 单条 FMA 依赖链延迟：[FMA latency](examples/03-fma-latency.md)
- 多 accumulator FMA 峰值吞吐：[FMA throughput](examples/04-fma-throughput.md)
- Hopper 异步完成周期与 pipeline depth：[TMA/WGMMA async](examples/05-hopper-async-pipeline.md)

## 必须先回答的五个问题

在提交代码前写出：

```text
1. 被测原子是什么，timed region 包含/排除什么？
2. latency 的下一次操作如何依赖上一次完成，或 throughput 有多少独立链？
3. 计时作用域是什么，为什么选择 clock64 或 CUDA Event？
4. 分母来自哪里，SASS 中每轮实际有多少条目标指令/多少有效字节？
5. cache、clock、occupancy、correctness 和防优化如何验证？
```

答不清时先补证据，不生成性能结论。

## 仓库约束

- 在本仓库新增公共 benchmark 前先查 `microbench/index.md`，按 family/config 去重。
- SM90 特性使用 `-arch=sm_90a`；本地只编译和静态检查，H800 性能数字只在远端运行。
- 优先复用 `microbench/common/clock.cuh`、`benchmark_utils.hpp`、`measure.hpp` 和现有 family 模式。
- 不修改 `operators/*/target/` 或 vendored `_references/ubench`；参考实现用于提炼方法，不直接复制其固定容量、SM 数或时钟常量。

## 禁止的解释

- 不把“连续发很多条后只 wait 一次”称为单条 completion latency。
- 不把一条依赖链的 `cycle/op` 取倒数称为峰值 throughput。
- 不跨 SM 拼接 `%clock64` 的最早 start/最晚 stop 作为整 grid 时间。
- 不用一次 `clockRate`/NVML 采样把 event 秒数硬换成 cycles。
- 不把 requested bytes 当作实际 DRAM transaction bytes；需要物理流量时使用 profiler counter。
- 不以源码循环代替 SASS 动态计数，不以静态无 spill 代替运行时 occupancy 验证。
