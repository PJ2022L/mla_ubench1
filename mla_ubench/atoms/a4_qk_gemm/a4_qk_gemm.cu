// A4 — qk_gemm（Tensor Core / WGMMA）。SCAFFOLD / 独立原子。
// 纯操作：GMMA::MMA_64x64x16_F32BF16BF16_SS，Q·Kᵀ，K=576 分 9 个 16-wide tile。
// **不保留** 原 TMA 加载 / softmax / barrier。Q/K 预置 smem（随机 bf16），只跑 MMA 主循环 REPEAT 次。
// 范式：ref_ubench MaxFlops（把 fma 换成 wgmma）—— 单 SM %%clock，测稳态 flop/clk/SM。
//
// 真实指令来源：splitkv_mla.cuh gemm<true>() + TiledMMA_QK。

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include "clock.cuh"
#include "gpu_check.h"
#include "mla_shapes.h"
// #include "sparse_fp8/config.h"   // TODO: TiledMMA_QK, SmemLayoutQ/K
// #include <cute/tensor.hpp>

using namespace mla_ubench;

#define THREADS_PER_BLOCK 128   // 1 warpgroup
#define REPEAT 512

__global__ void a4_qk_gemm(uint32_t* startClk, uint32_t* stopClk, float* dsink) {
    // TODO(impl): __shared__ bf16 sQ[64*576], sK[64*576]（SW128 swizzle 布局），预填随机。
    uint32_t uid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = 0, stop = 0;
    float acc = 0.f;

    CLK_START(start);
    #pragma unroll 1
    for (int j = 0; j < REPEAT; ++j) {
        // TODO(impl): cute::gemm(TiledMMA_QK{}, sQ, sK, rP);   // 9 个 K-tile，warpgroup_arrive/wait
        //   保持 rP 链式依赖（下一轮用上一轮 rP）防优化。
        //   acc += rP(0);
    }
    CLK_STOP(stop);
    startClk[uid] = start; stopClk[uid] = stop; dsink[uid] = acc;
}

int main() {
    // TODO(impl): launch；flop = 2*64*64*576（QK 一次）*REPEAT；flop/clk = flop/cycles。
    //   对齐 MaxFlops 输出格式：printf("A4 qk_gemm: %f flop/clk/SM\n", ...);
    printf("[TODO] A4 qk_gemm scaffold\n");
    return 0;
}
