# 02 — 测量范式（ref_ubench 风格）

提炼自 `../ref/ubench/NVIDIA-Hopper-Benchmark`（`RegularUnits/MaxFlops`、`shared_bw`、`NewFeatures/TMA`）。

## 核心：clock-cycle，单 SM

```cuda
__global__ void atom(uint32_t* startClk, uint32_t* stopClk, /* data */, uint32_t* sink) {
    // 1) 预置数据到 reg/smem（不计时）
    register T s = data[uid];

    // 2) bracket + start clock
    asm volatile("bar.sync 0;");
    uint32_t start; asm volatile("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");

    // 3) 纯目标操作，REPEAT 次，链式依赖防优化
    #pragma unroll 1
    for (int j = 0; j < REPEAT; ++j) { s = op(s); }   // 例：s[tmp]=…;tmp=s[tmp]; 或 wgmma

    // 4) stop clock
    asm volatile("bar.sync 0;");
    uint32_t stop; asm volatile("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");

    // 5) 写回
    startClk[uid]=start; stopClk[uid]=stop; sink[uid]=s;
}
// main: cycles = max(stopClk) - min(startClk)
```

（本仓库把 2/4 步封装成 `common/clock.cuh` 的 `CLK_START/CLK_STOP` 宏。）

## 计量换算

| 类型 | 公式 | 单位 |
|---|---|---|
| 计算吞吐（A4/A5） | `ops * REPEAT / cycles` | flop/clk/SM |
| 带宽（A1/A2/A3/A7/A8） | `bytes * REPEAT / cycles` | byte/clk/SM |
| 延迟（A6/latency） | `cycles / REPEAT` | cycle/op |
| →绝对值 | `× getGPUClock()`(Hz) | GB/s 或 TFLOPS |

## 关键约定

- **单 SM**（`BLOCKS_NUM=1`）：结果与 grid/batch 无关，是纯硬件能力，可移植可比。
- **防编译器消除**：链式依赖（`s1+=s1*s2`、`tmp=s[tmp]`、rP 跨轮复用），最后把结果写 sink。
- **稳态**：`REPEAT` 足够大（512–1024）摊薄启动；`common/measure.hpp` 多次跑取中位数。
- **带宽争用**：单 SM 测的是稳态吞吐。要测**多 SM 对 HBM 的争用**，用多 block 版 + `getGPUClock()` 换 GB/s（参考 `../ref/ubench/NVIDIA-Hopper-Benchmark/NewFeatures/TMA/Throughput` 的多 block 写法）。A1/A7/A8 建议两种都测。

## 复用的 ref_ubench helper（已改写进 `common/`）

| 本仓库 | 源 | 作用 |
|---|---|---|
| `common/clock.cuh` | `MaxFlops.cu` + `gpu-clock.cuh` | `%%clock` 宏 + NVML 取 SM 频率 |
| `common/measure.hpp` | `MeasurementSeries.hpp` | 多次取稳态 |
| `common/tma_util.cuh` | `NewFeatures/TMA/util.h` | `cuTensorMapEncodeTiled` 封装（A1/A7） |
| `common/gpu_check.h` | 各 bench 的 `gpuErrchk` | 错误检查 |

## 每原子隔离验证（ncu, H800）

编译 `make -C atoms/aX run` 后跑 ncu，确认只压目标维度：

| 原子 | 通过准则 |
|---|---|
| A1/A7/A8 | `dram__throughput` 高；`tensor active%` ≈ 0 |
| A2 | `sm__inst_executed_pipe_xu`（转换）+ smem 写高 |
| A3 | cluster/DSM smem 流量高 |
| A4/A5 | `sm__pipe_tensor_op_hmma_cycles_active.pct` > 70%；HBM ≈ 0 |
| A6 | `sm__inst_executed_pipe_xu`（exp2）高 |

指标名 + SM 频率 throttle 首次在 H800 上 `ncu --query-metrics` 核对。

## 参数扫描（L2 命中率）

A1 加参数扫描即得 L2 曲线：`topk∈{64,256,1024,4096,16384} × index_dist∈{sequential,local,random} × cache_hint`，working-set = 唯一token数×656B，从 < L2 扫到 >> L2，看 `lts__t_sector_hit_rate.pct`。

下一步：`03-composition-model.md`（原子时间怎么组合）。
