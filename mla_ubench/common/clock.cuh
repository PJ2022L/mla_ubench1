#pragma once
// clock 计时宏 + SM 频率获取。SCAFFOLD。
// 范式源自 ref/ubench/NVIDIA-Hopper-Benchmark（MaxFlops.cu 的 %%clock + gpu-clock.cuh 的 NVML）。
//
// 核心思想（cycle-based，单 SM，可移植）：
//   在 kernel 内用 %%clock 读起止周期，cycles = max(stopClk)-min(startClk)；
//   吞吐 = ops(or bytes)*REPEAT / cycles → flop/clk/SM 或 byte/clk/SM；
//   延迟 = cycles / REPEAT → cycle/op。
//   若要换算成 GB/s / TFLOPS，用 getGPUClock() 拿 SM 频率(Hz)乘上去。

#include <cstdint>
#include <cuda_runtime.h>

// ---- 设备侧：clock bracket 宏（放进每个原子 kernel）----
// 用法：
//   CLK_START(start);
//   #pragma unroll 1
//   for (int j=0;j<REPEAT;++j) { /* 纯目标操作 */ }
//   CLK_STOP(stop);
#define CLK_BAR()        asm volatile("bar.sync 0;")
#define CLK_START(v)     do { CLK_BAR(); asm volatile("mov.u32 %0, %%clock;" : "=r"(v) :: "memory"); } while(0)
#define CLK_STOP(v)      do { CLK_BAR(); asm volatile("mov.u32 %0, %%clock;" : "=r"(v) :: "memory"); } while(0)

namespace mla_ubench {

// 主机侧：从 startClk/stopClk 数组算总周期（max(stop)-min(start)）。TODO(impl)。
uint32_t reduce_cycles(const uint32_t* startClk, const uint32_t* stopClk, int n);

// NVML 取 SM 时钟(MHz)，用于 cycle→time 换算。源自 gpu-clock.cuh。TODO(impl)。
unsigned int getGPUClock(int deviceId = 0);

}  // namespace mla_ubench
