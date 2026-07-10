// ld.global.nc 128-bit + L2 hint micro-benchmark. SM90 scaffold.
// Pure operation: indexed address generation + vector global load + register sink.
// **不保留** FlashMLA 的 producer/consumer barrier、warpgroup 专用化、cluster 同步。
// 范式：ref_ubench MaxFlops/tma_bw_2d —— 单 SM、%%clock 计时；带宽版可多 block + NVML。
//
// Report cycle/load, load/clk/SM, unique byte/clk/SM and L2 hit rate.

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include "clock.cuh"
#include "gpu_check.h"
#include "attention_shapes.h"
// #include "tma_util.cuh"   // 若走 TMA 路径

using namespace microbench;

#define BLOCKS_NUM 1
#define THREADS_PER_BLOCK 256   // 与 FlashMLA producer warpgroup 规模对齐
#define REPEAT 256

__global__ void a1_kv_gather(uint32_t* startClk, uint32_t* stopClk,
                             const uint8_t* __restrict__ kv,      // [topk, 656]
                             const int32_t* __restrict__ indices, // [topk]
                             uint32_t* dsink, int topk) {
    uint32_t uid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = 0, stop = 0;
    uint32_t acc = 0;

    CLK_START(start);
    #pragma unroll 1
    for (int j = 0; j < REPEAT; ++j) {
        // TODO(impl): select sequential/local/random index, emit the exact
        // load_128b_from_gmem cache-hint variant, then fold the register value into acc.
    }
    CLK_STOP(stop);

    startClk[uid] = start; stopClk[uid] = stop; dsink[uid] = acc;
}

int main() {
    // TODO(impl): malloc/gen kv+indices（gen_kv_fp8/gen_indices），cudaMalloc/memcpy，
    //   launch a1_kv_gather<<<BLOCKS_NUM,THREADS,smem>>>，拷回 clocks，
    //   cycles = reduce_cycles(...); byte/clk = topk*656.0*REPEAT/cycles;
    //   print cycle/load, load/clk/SM, unique byte/clk/SM, and cache-hit counters.
    printf("[TODO] global_load/128b_nc_l2_sm90 scaffold\n");
    return 0;
}
