// A1 — kv_gather（HBM 带宽）。SCAFFOLD / 独立原子。
// 纯操作：index→物理地址 + cp.async.cg.L2::256B（或 TMA）拉 656B/token 进 smem，然后丢弃。
// **不保留** FlashMLA 的 producer/consumer barrier、warpgroup 专用化、cluster 同步。
// 范式：ref_ubench MaxFlops/tma_bw_2d —— 单 SM、%%clock 计时；带宽版可多 block + NVML。
//
// 计量：byte/clk/SM = topk*656*REPEAT / cycles；×SM频率 → GB/s。
// 真实指令来源：splitkv_mla.cuh WG2 前半（gather）。

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include "clock.cuh"
#include "gpu_check.h"
#include "mla_shapes.h"
// #include "tma_util.cuh"   // 若走 TMA 路径

using namespace mla_ubench;

#define BLOCKS_NUM 1
#define THREADS_PER_BLOCK 256   // 与 FlashMLA producer warpgroup 规模对齐
#define REPEAT 256

__global__ void a1_kv_gather(uint32_t* startClk, uint32_t* stopClk,
                             const uint8_t* __restrict__ kv,      // [topk, 656]
                             const int32_t* __restrict__ indices, // [topk]
                             uint32_t* dsink, int topk) {
    extern __shared__ uint8_t smem[];   // 落点，仅防优化
    uint32_t uid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = 0, stop = 0;
    uint32_t acc = 0;

    CLK_START(start);
    #pragma unroll 1
    for (int j = 0; j < REPEAT; ++j) {
        // TODO(impl): for each token this thread owns:
        //   int tok = indices[k];                                  // index load
        //   const uint8_t* src = kv + tok * shape::BYTES_PER_TOKEN;// 物理地址
        //   cp.async.cg.shared.global.L2::256B  128b 宽拷贝进 smem  // 纯 gather
        //   （或 TMA: cp.async.bulk.tensor.2d，见 tma_util.cuh）
        //   acc ^= smem[...];                                      // 链式依赖防优化
    }
    CLK_STOP(stop);

    startClk[uid] = start; stopClk[uid] = stop; dsink[uid] = acc;
}

int main() {
    // TODO(impl): malloc/gen kv+indices（gen_kv_fp8/gen_indices），cudaMalloc/memcpy，
    //   launch a1_kv_gather<<<BLOCKS_NUM,THREADS,smem>>>，拷回 clocks，
    //   cycles = reduce_cycles(...); byte/clk = topk*656.0*REPEAT/cycles;
    //   printf("A1 kv_gather: %f byte/clk/SM, cycles=%u\n", ...);
    printf("[TODO] A1 kv_gather scaffold\n");
    return 0;
}
