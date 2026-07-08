// A5 — pv_gemm（Tensor Core / WGMMA）。SCAFFOLD / 独立原子。
// 纯操作：MMA_64x256x16_F32BF16BF16_RS/SS，P·V，O 分 O_L/O_R（各 64x256）。
// **不保留** 原 softmax / barrier / store。P(bf16)、V(bf16) 预置，只跑 PV 主循环 REPEAT 次。
// 范式：ref_ubench MaxFlops 风格；测 flop/clk/SM。
//
// 真实指令来源：splitkv_mla.cuh gemm<false>() + TiledMMA_PV_LocalP/RemoteP。

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include "clock.cuh"
#include "gpu_check.h"
#include "mla_shapes.h"

using namespace mla_ubench;

#define THREADS_PER_BLOCK 128
#define REPEAT 512

__global__ void a5_pv_gemm(uint32_t* startClk, uint32_t* stopClk, float* dsink) {
    // TODO(impl): __shared__ bf16 sP[64*64], sV[64*512]（V 转置布局），预填随机。
    uint32_t uid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = 0, stop = 0;
    float acc = 0.f;

    CLK_START(start);
    #pragma unroll 1
    for (int j = 0; j < REPEAT; ++j) {
        // TODO(impl): cute::gemm(TiledMMA_PV_LocalP{}, rP, sV_L, rO_L);
        //             cute::gemm(TiledMMA_PV_RemoteP{}, sP, sV_R, rO_R);
        //   rO 链式依赖防优化；acc += rO_L(0);
    }
    CLK_STOP(stop);
    startClk[uid] = start; stopClk[uid] = stop; dsink[uid] = acc;
}

int main() {
    // TODO(impl): flop = 2*64*512*64（PV 一次）*REPEAT；printf flop/clk/SM。
    printf("[TODO] A5 pv_gemm scaffold\n");
    return 0;
}
