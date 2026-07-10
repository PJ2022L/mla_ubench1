// cvt_fp8x8_bf16x8 conversion micro-benchmark. SM90 scaffold.
// Inputs are preloaded; the timed body excludes shared/DSM stores.
// 范式：ref_ubench shared_bw + MaxFlops —— 单 SM %%clock；测反量化吞吐(token/clk) 与 smem 写带宽。
//
// Source helper: operators/flash_mla/target/csrc/sm90/decode/sparse_fp8/components/dequant.h.
// deep-dive：H800 上 e4m3→bf16 需 4 步 ~50cyc/token（可能 dequant-bound）—— 本原子量化它。

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include "clock.cuh"
#include "gpu_check.h"
#include "attention_shapes.h"
// #include "sparse_fp8/components/dequant.h"   // TODO: FlashMLA cvt_fp8x8_bf16x8

using namespace microbench;

#define BLOCKS_NUM 1
#define THREADS_PER_BLOCK 128
#define REPEAT 512

__global__ void a2_dequant(uint32_t* startClk, uint32_t* stopClk, uint32_t* dsink) {
    __shared__ uint8_t raw_fp8[shape::TOPK_BLOCK * shape::D_NOPE];   // 预置的 raw
    // TODO(impl): 预填 raw_fp8 + scales（随机）
    uint32_t uid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = 0, stop = 0, acc = 0;

    CLK_START(start);
    #pragma unroll 1
    for (int j = 0; j < REPEAT; ++j) {
        // TODO(impl): 每线程负责若干 8-elem chunk：
        //   fp8x8 in = load raw_fp8[...];
        //   bf16x8 out = cvt_fp8x8_bf16x8(in, scale_bf162);   // 纯反量化
        //   fold one output register into acc; shared/DSM stores are separate benchmarks.
    }
    CLK_STOP(stop);
    startClk[uid] = start; stopClk[uid] = stop; dsink[uid] = acc;
}

int main() {
    // TODO(impl): launch + cycles → token/clk（= TOPK_BLOCK*REPEAT/cycles）+ smem byte/clk。
    printf("[TODO] convert/fp8x8_to_bf16x8_sm90 scaffold\n");
    return 0;
}
