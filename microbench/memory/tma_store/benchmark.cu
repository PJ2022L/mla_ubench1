// Tensor TMA store BF16 micro-benchmark (2D/5D modes). SM90 scaffold.
// 纯操作：STSM(register→smem) → SM90_TMA_STORE(smem→HBM) 写 O [64x512] bf16。
// **不保留** 原 epilogue barrier。O 预置 reg/smem，只跑 store REPEAT 次。
// 范式：ref_ubench mem_bw / tma_bw —— 测写带宽 byte/clk/SM（或多 block + NVML → GB/s）。
//
// 真实指令来源：splitkv_mla.cuh store_o()（SM90_U32x4_STSM_N + SM90_TMA_STORE_5D）。

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include "clock.cuh"
#include "gpu_check.h"
#include "attention_shapes.h"
// #include "tma_util.cuh"

using namespace microbench;

#define THREADS_PER_BLOCK 256
#define REPEAT 256

__global__ void a7_tma_store(uint32_t* startClk, uint32_t* stopClk,
                             uint16_t* __restrict__ gO /* [64,512] bf16 */) {
    __shared__ uint16_t sOBuf[shape::BLOCK_M * shape::D_V];
    // TODO(impl): 预填 rO 寄存器随机。
    uint32_t uid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = 0, stop = 0;

    CLK_START(start);
    #pragma unroll 1
    for (int j = 0; j < REPEAT; ++j) {
        // TODO(impl):
        //   SM90_U32x4_STSM_N::copy(rO → sOBuf);        // reg→smem
        //   fence_view_async_shared();
        //   SM90_TMA_STORE_5D::copy(&tensor_map_o, sOBuf, ...);  // smem→HBM
        //   tma_store_arrive();
    }
    CLK_STOP(stop);
    startClk[uid] = start; stopClk[uid] = stop;
}

int main() {
    // TODO(impl): byte/clk = 64*512*2*REPEAT/cycles；×freq → GB/s。
    printf("[TODO] tma_store parameterized scaffold\n");
    return 0;
}
