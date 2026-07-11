# 示例 3：FP32 FMA 依赖延迟

## 测量目标

测同一 accumulator 上连续 FP32 FMA 的 RAW dependency latency。使用单线程、单依赖链和 `%clock64`。

## 核心结构

```cpp
__global__ void fma_latency(const float* input,
                            uint64_t* cycles,
                            float* sink,
                            int repeat) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float x = input[0];
    float a = input[1];
    float b = input[2];

    uint64_t start;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start) :: "memory");
#pragma unroll 1
    for (int i = 0; i < repeat; ++i) {
        asm volatile("fma.rn.f32 %0, %0, %1, %2;"
                     : "+f"(x) : "f"(a), "f"(b));
    }
    uint64_t stop;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop) :: "memory");

    cycles[0] = stop - start;
    sink[0] = x;
}
```

每次FMA读取上一条写出的 `x`，因此不能并行发射同一链。runtime input和global sink防止constant folding/删除。

## 指标

```text
cycle_per_fma = (cycles - baseline_loop_cycles) / repeat
```

baseline保留循环和寄存器依赖但移除FMA。检查SASS中每轮正好一条目标FFMA且无spill。

## 不能这样解释

`1 / cycle_per_fma`不是SM峰值FMA throughput。峰值吞吐需要多独立accumulator和足量线程，见示例4。

## 参考

`operators/_references/ubench/NVIDIA-Hopper-Benchmark/RegularUnits/alu_lat_float/`
