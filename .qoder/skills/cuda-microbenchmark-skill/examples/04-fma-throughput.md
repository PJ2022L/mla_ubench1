# 示例 4：FP32 FMA 吞吐

## 测量目标

使用多个独立accumulator、足量线程和CUDA Event测整卡FP32 FMA吞吐。它不测单条依赖latency。

## 核心结构

```cpp
__global__ void fma_throughput(const float* input,
                               float* sink,
                               int repeat) {
    size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    float a = input[tid * 2 + 0];
    float b = input[tid * 2 + 1];
    float x0=1, x1=2, x2=3, x3=4, x4=5, x5=6, x6=7, x7=8;

#pragma unroll 1
    for (int i = 0; i < repeat; ++i) {
        asm volatile(
            "fma.rn.f32 %0, %0, %8, %9;\n"
            "fma.rn.f32 %1, %1, %8, %9;\n"
            "fma.rn.f32 %2, %2, %8, %9;\n"
            "fma.rn.f32 %3, %3, %8, %9;\n"
            "fma.rn.f32 %4, %4, %8, %9;\n"
            "fma.rn.f32 %5, %5, %8, %9;\n"
            "fma.rn.f32 %6, %6, %8, %9;\n"
            "fma.rn.f32 %7, %7, %8, %9;"
            : "+f"(x0), "+f"(x1), "+f"(x2), "+f"(x3),
              "+f"(x4), "+f"(x5), "+f"(x6), "+f"(x7)
            : "f"(a), "f"(b));
    }
    sink[tid] = x0+x1+x2+x3+x4+x5+x6+x7;
}
```

## 指标

若SASS确认每线程每轮8条FMA：

```text
total_FLOPs = blocks * threads * repeat * 8 FMA * 2 FLOP/FMA
TFLOPS = total_FLOPs / event_seconds / 1e12
```

扫描accumulator数、threads/CTA和blocks，直到平台；寄存器过多会降低occupancy。输入load和最终sink位于循环外，repeat要足够大以摊薄它们和launch成本。

## 参考

- `operators/_references/ubench/NVIDIA-Hopper-Benchmark/RegularUnits/MaxFlops/`
- `operators/_references/ubench/gpgpu-sim/GPU_Microbenchmark/MaxFlops/`
