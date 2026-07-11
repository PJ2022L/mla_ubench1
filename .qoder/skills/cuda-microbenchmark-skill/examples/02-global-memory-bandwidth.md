# 示例 2：全局内存读吞吐

## 测量目标

让多个 SM、warp和独立load饱和内存系统，使用CUDA Event计算整grid requested read GB/s。它不测单次load latency。

## 核心结构

```cpp
__global__ void read_bw(const float4* input,
                        float4* sink,
                        size_t count) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    float4 a0 = {0, 0, 0, 0};
    float4 a1 = {0, 0, 0, 0};

    for (; i + stride < count; i += 2 * stride) {
        float4 x0 = input[i];
        float4 x1 = input[i + stride];
        a0.x += x0.x; a0.y += x0.y; a0.z += x0.z; a0.w += x0.w;
        a1.x += x1.x; a1.y += x1.y; a1.z += x1.z; a1.w += x1.w;
    }
    sink[static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x] = {
        a0.x + a1.x, a0.y + a1.y, a0.z + a1.z, a0.w + a1.w
    };
}
```

使用多套accumulator避免单条加法依赖成为瓶颈。检查SASS确实生成目标vector load；尾部单独处理或让count满足launch几何。

## Event计时

```cpp
cudaEventRecord(start, stream);
read_bw<<<blocks, threads, 0, stream>>>(input, sink, count);
cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&ms, start, stop);
```

## 指标与扫描

```text
requested_GB/s = actual_dynamic_load_count * 16 / seconds / 1e9
```

扫描 `blocks`、threads、独立streams、working set和访问模式，直到平台。working set小于L2时只能标L2/cache带宽；测HBM时应明显超过L2并用DRAM counter确认。sink store不计入requested read bytes，但如果报告总fabric traffic必须单独加入。

## 参考

- `operators/_references/ubench/NVIDIA-Hopper-Benchmark/RegularUnits/mem_bw/`
- `operators/_references/ubench/gpgpu-sim/GPU_Microbenchmark/l1_bw_32f_unroll/`
