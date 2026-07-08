// A2 — dequant（SFU + Smem 写）。SCAFFOLD / 独立原子。
// 纯操作：cvt_fp8x8_bf16x8（e4m3→bf16 ×scale）把 raw FP8 反量化，写 K-major smem。
// **不保留** 原 barrier / DSM 分发（DSM 是 A3 单独测）。预置 raw FP8 于 smem。
// 范式：ref_ubench shared_bw + MaxFlops —— 单 SM %%clock；测反量化吞吐(token/clk) 与 smem 写带宽。
//
// 真实指令来源：target_op/FlashMLA/csrc/sm90/decode/sparse_fp8/components/dequant.h 的 cvt_fp8x8_bf16x8。
// deep-dive：H800 上 e4m3→bf16 需 4 步 ~50cyc/token（可能 dequant-bound）—— 本原子量化它。

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include "clock.cuh"
#include "gpu_check.h"
#include "mla_shapes.h"
// #include "sparse_fp8/components/dequant.h"   // TODO: FlashMLA cvt_fp8x8_bf16x8

using namespace mla_ubench;

#define BLOCKS_NUM 1
#define THREADS_PER_BLOCK 128
#define REPEAT 512

__global__ void a2_dequant(uint32_t* startClk, uint32_t* stopClk, uint32_t* dsink) {
    __shared__ uint8_t raw_fp8[shape::TOPK_BLOCK * shape::D_NOPE];   // 预置的 raw
    __shared__ uint16_t out_bf16[shape::TOPK_BLOCK * shape::D_NOPE]; // 反量化落点
    // TODO(impl): 预填 raw_fp8 + scales（随机）
    uint32_t uid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = 0, stop = 0, acc = 0;

    CLK_START(start);
    #pragma unroll 1
    for (int j = 0; j < REPEAT; ++j) {
        // TODO(impl): 每线程负责若干 8-elem chunk：
        //   fp8x8 in = load raw_fp8[...];
        //   bf16x8 out = cvt_fp8x8_bf16x8(in, scale_bf162);   // 纯反量化
        //   st.weak.shared 128b → out_bf16[...];              // smem 写
        //   acc ^= out_bf16[...];                             // 防优化
    }
    CLK_STOP(stop);
    startClk[uid] = start; stopClk[uid] = stop; dsink[uid] = acc;
}

int main() {
    // TODO(impl): launch + cycles → token/clk（= TOPK_BLOCK*REPEAT/cycles）+ smem byte/clk。
    printf("[TODO] A2 dequant scaffold\n");
    return 0;
}
