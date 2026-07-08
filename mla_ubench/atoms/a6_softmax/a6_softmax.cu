// A6 — softmax（SFU）。SCAFFOLD / 独立原子。
// 纯操作：online softmax 一步：rowwise max(shfl_xor) → exp2f(P*scale - max) → rowwise sum → rescale。
// **不保留** 原 barrier（sScale_and_sS_ready 等）。喂合成 logits [64x64]，只跑 softmax REPEAT 次。
// 范式：ref_ubench alu_lat / MaxFlops —— 测 SFU(exp2) 吞吐 / 归约延迟（cycle/op）。
//
// 真实指令来源：splitkv_mla.cuh scale_softmax()（第 32–86 行）。

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include "clock.cuh"
#include "gpu_check.h"
#include "mla_shapes.h"

using namespace mla_ubench;

#define THREADS_PER_BLOCK 128
#define REPEAT 1024

__global__ void a6_softmax(uint32_t* startClk, uint32_t* stopClk, float* dsink) {
    float rP[8];   // 每线程持有的 P 片段（对齐 partition_fragment_C 布局）
    // TODO(impl): 预填 rP 随机。
    uint32_t uid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = 0, stop = 0;
    float acc = 0.f;

    CLK_START(start);
    #pragma unroll 1
    for (int j = 0; j < REPEAT; ++j) {
        // TODO(impl):
        //   float m = -INFINITY; for i: m = max(m, rP[i]);       // rowwise max
        //   for offset: m = max(m, __shfl_xor_sync(-1, m, offset));// warp 归约
        //   for i: rP[i] = exp2f(rP[i]*scale - m);                // SFU exp2
        //   float l = Σ rP[i]; (+ shfl 归约)                       // rowwise sum
        //   acc += rP[0] + m + l;                                 // 防优化
    }
    CLK_STOP(stop);
    startClk[uid] = start; stopClk[uid] = stop; dsink[uid] = acc;
}

int main() {
    // TODO(impl): cycle/softmax = cycles/REPEAT；报 exp2 吞吐。
    printf("[TODO] A6 softmax scaffold\n");
    return 0;
}
