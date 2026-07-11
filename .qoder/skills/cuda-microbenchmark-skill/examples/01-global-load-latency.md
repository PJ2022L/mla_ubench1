# 示例 1：全局内存 load 依赖延迟

## 测量目标

测一条指定 cache modifier 的 global load 从地址已知到结果可用于下一次地址计算的依赖周期。不是带宽测试。

## 核心结构

```cpp
__device__ __forceinline__ uint64_t clock64_now() {
    uint64_t x;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(x) :: "memory");
    return x;
}

__device__ __forceinline__ uint32_t load_cg(const uint32_t* ptr) {
    uint32_t x;
    asm volatile("ld.global.cg.u32 %0, [%1];"
                 : "=r"(x) : "l"(ptr) : "memory");
    return x;
}

__global__ void load_latency(const uint32_t* next,
                             uint64_t* cycles,
                             uint32_t* sink,
                             int repeat) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    uint32_t index = load_cg(next);  // 计时外预热/建立起点
    uint64_t start = clock64_now();
#pragma unroll 1
    for (int i = 0; i < repeat; ++i) {
        index = load_cg(next + index);
    }
    uint64_t stop = clock64_now();

    cycles[0] = stop - start;
    sink[0] = index;
}
```

host 预先构造环形 permutation，使每个 `next[index]` 指向下一个合法位置。不能使用 `next[i]` 的线性循环，因为后续地址不依赖 load 结果，硬件可以并发。

## 指标

```text
cycle_per_load_raw = cycles / repeat
cycle_per_load_net = (cycles - baseline_cycles) / repeat
```

baseline 使用相同循环和寄存器依赖，但不访问 global memory。工作集和预热决定这是 L2-hot 还是 HBM/TLB 混合延迟；必须写入结果标签并用 profiler验证。

## 参考

- `operators/_references/ubench/NVIDIA-Hopper-Benchmark/RegularUnits/mem_lat/`
- `operators/_references/ubench/gpgpu-sim/GPU_Microbenchmark/shared_lat/`

不要复制其中的32位clock和固定cache容量。
