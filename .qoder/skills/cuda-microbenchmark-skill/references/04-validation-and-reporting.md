# 准确性、验收与结果报告

## 1. 三层证据

每个 benchmark 同时保留：

1. 源码证据：timed region、依赖链、动态循环数、输入布局。
2. 静态证据：PTX/SASS opcode、vector width、cache hint、寄存器、stack/local/spill。
3. 运行证据：环境、correctness、raw samples、时钟稳定性、profiler counter。

缺少任一层时，结果只能标为 scaffold/validated，不能当成可信 measured number。

## 2. 防优化

- 输入来自 runtime buffer或参数，避免编译期常量折叠。
- inline PTX 使用 `asm volatile`、正确输入/输出约束和必要的 `memory` clobber。
- 最终 accumulator/index/checksum 写入 global sink并拷回 host 检查。
- 空 asm只能建立编译器依赖，不能替代 GPU fence或 completion wait。
- 不依赖永远不执行的分支、未读取 dummy buffer或单纯 C++ `volatile`。

## 3. correctness

- 至少一个小规模 case与 CPU/reference 对拍。
- load 验证读取值；store 验证目标地址更新且未触达区域保持不变。
- TMA/cp.async/DSM 验证首尾 tile、多个 transaction和 phase复用。
- WGMMA/convert/SFU检查有限非零输出，必要时逐元素比较。
- correctness kernel可在 timed region外，但必须对应同一布局和指令语义。

## 4. 静态检查

构建并保存：

```bash
nvcc -O3 -std=c++17 -arch=sm_90a --ptx benchmark.cu -o benchmark.ptx
nvcc -O3 -std=c++17 -arch=sm_90a --cubin benchmark.cu -o benchmark.cubin
nvdisasm --print-line-info benchmark.cubin > benchmark.sass
```

检查：

- 目标 opcode是否存在，形状/dtype/mode是否正确；
- 每轮静态 opcode数量与动态循环数；
- `STACK/LOCAL`、`LDL/STL`、register count；
- 非预期 load/store、转换、branch或barrier是否进入 timed loop；
- 编译版本变化是否改变指令选择。

## 5. 环境记录

每个正式结果至少保存：

```text
timestamp, hostname, GPU name/UUID, compute capability
driver, CUDA runtime, nvcc, framework versions
SM count, ECC, power limit, persistence/P-state
SM/memory clock before and after, temperature, power draw
source commit, dirty status, build flags, binary/SASS hash
full CLI, env vars, warmup, samples, repeat
shape, dtype, operand mode, cache state, working set
raw JSONL, correctness, exit code
```

## 6. 输出字段

建议 JSONL：

```json
{
  "benchmark": "family/config",
  "gpu": "H800",
  "measurement": "latency_dependency|completion|throughput",
  "scope": "thread|warpgroup|cta|sm|cluster|grid",
  "timer": "clock64|cuda_event",
  "warmup": 5,
  "samples": 20,
  "repeat": 512,
  "median_cycles": 12345,
  "p10_cycles": 12200,
  "p90_cycles": 12600,
  "cycle_per_op": 12.3,
  "op_per_clk_sm": null,
  "requested_bytes": 0,
  "working_set_bytes": 0,
  "correct": true,
  "sass_hash": "..."
}
```

只输出适用字段；不要把 cycle 与 millisecond 放在同一个无单位 `latency` 字段。

## 7. 参考实现中需要修正的旧模式

- 32 位 `%clock` 长循环会回绕；优先 `%clock64`。
- 固定 Volta/A100 cache容量、SM数和 `clock_freq_MHZ` 不适用于 H800。
- 不跨 SM 拼接 clock64；整 grid用 Event。
- 一次预热、多次取最小值会系统性偏低；使用正式样本分布。
- 初始化 shared 后缺少 barrier会产生数据竞争。
- working set、注释和 cache modifier不一致时，不能仅凭文件名声称测到 L1/L2/DRAM。
- WGMMA/TMA 长 issue loop后只 wait一次，测的是聚合完成速率，不自动是单条 latency。

## 8. 最终审查问题

```text
目标指令是否真的存在于 SASS？
分母是否按动态 SASS 数量计算？
latency 是否有真依赖，throughput 是否已饱和？
cache state和working set是否可复现？
异步操作是否在stop前完成并被消费？
结果是否可能被compiler优化或spill污染？
DVFS、热降频和样本波动是否被记录？
```
