// 128-bit register-to-shared store micro-benchmark. SM90 scaffold.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include "clock.cuh"

#define THREADS_PER_BLOCK 128
#define REPEAT 1024

__global__ void benchmark_st_shared_128b(uint32_t* start_clk, uint32_t* stop_clk,
                                         uint4* sink) {
    __shared__ uint4 smem[THREADS_PER_BLOCK];
    const int tid = threadIdx.x;
    uint4 value = {uint32_t(tid + 1), 2u, 3u, 4u};
    uint32_t start = 0, stop = 0;
    CLK_START(start);
    #pragma unroll 1
    for (int i = 0; i < REPEAT; ++i) {
        // TODO(impl): selectable K-major/swizzled address; emit one 128-bit shared store.
        smem[tid] = value;
        value.x += smem[(tid + i) & (THREADS_PER_BLOCK - 1)].x;
    }
    CLK_STOP(stop);
    start_clk[tid] = start;
    stop_clk[tid] = stop;
    if (tid == 0) sink[0] = value;
}

int main() {
    printf("[TODO] shared_store/128b_sm90 scaffold\n");
    return 0;
}
