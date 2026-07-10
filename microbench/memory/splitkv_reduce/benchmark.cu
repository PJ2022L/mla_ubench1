// Split-KV HBM+F32 reduction benchmark. SM90 scaffold.
// 纯操作：读 num_splits 份 partial，LSE rescale（exp2f(lse_i - lse_global)）+ float4 累加。
// **不保留** 原 PDL / 与主 kernel 的重叠。喂合成 o_accum/lse_accum，只跑 reduce 循环。
// 范式：ref_ubench mem_bw —— 主流量是读 o_accum，测 HBM 读带宽。
//
// Source pattern: operators/flash_mla/target/csrc/smxx/decode/combine/combine.cu.

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include "clock.cuh"
#include "gpu_check.h"
#include "attention_shapes.h"

using namespace microbench;

#define THREADS_PER_BLOCK 256   // 8 warps，对齐 combine BLOCK_SIZE_M=8
#define REPEAT 128

__global__ void a8_combine(uint32_t* startClk, uint32_t* stopClk,
                           const float* __restrict__ o_accum,   // [.., num_splits, D_V]
                           const float* __restrict__ lse_accum, // [.., num_splits]
                           uint16_t* __restrict__ o_out, int num_splits) {
    uint32_t uid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = 0, stop = 0;
    float4 result = {0,0,0,0};

    CLK_START(start);
    #pragma unroll 1
    for (int j = 0; j < REPEAT; ++j) {
        // TODO(impl): for split in [0,num_splits):
        //   float lse_scale = exp2f(lse_accum[split] - lse_global);
        //   float4 d = *(float4*)(o_accum + split*stride + lane*16);
        //   result += lse_scale * d;    // fused multiply-add，float4 累加
        //   （结构直接照搬 combine.cu 内层循环）
    }
    CLK_STOP(stop);
    startClk[uid] = start; stopClk[uid] = stop;
    if (uid == 0) o_out[0] = (uint16_t)result.x;   // 防优化
}

int main() {
    // TODO(impl): 扫 num_splits∈{32,64,96,128,160}；byte/clk = num_splits*D_V*4*REPEAT/cycles。
    printf("[TODO] splitkv_reduce/dv512_f32_sm90 scaffold\n");
    return 0;
}
