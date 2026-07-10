// Parameterized SM90 BF16 WGMMA scaffold. BENCH_M/N/K come from the
// configuration leaf Makefile. RS and SS are run and reported separately.

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include "clock.cuh"
#include "gpu_check.h"
#include "attention_shapes.h"
// #include "sparse_fp8/config.h"   // TODO: TiledMMA_QK, SmemLayoutQ/K
// #include <cute/tensor.hpp>

using namespace microbench;

#ifndef BENCH_M
#define BENCH_M 64
#endif
#ifndef BENCH_N
#define BENCH_N 64
#endif
#ifndef BENCH_K
#define BENCH_K 16
#endif

#define THREADS_PER_BLOCK 128
#define REPEAT 512

__global__ void benchmark_wgmma(uint32_t* startClk, uint32_t* stopClk, float* dsink) {
    // TODO(impl): prefill operands outside the timed region, then select the
    // exact m64nNk16 RS or SS atom from BENCH_N and an operand-mode argument.
    uint32_t uid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = 0, stop = 0;
    float acc = 0.f;

    CLK_START(start);
    #pragma unroll 1
    for (int j = 0; j < REPEAT; ++j) {
        // TODO(impl): issue WGMMA at configurable depth. Include the required
        // fence/commit/wait and retain an accumulator dependency/sink.
    }
    CLK_STOP(stop);
    startClk[uid] = start; stopClk[uid] = stop; dsink[uid] = acc;
}

int main() {
    // TODO(impl): report RS and SS independently. Per instruction:
    // FLOP = 2 * BENCH_M * BENCH_N * BENCH_K.
    printf("[TODO] wgmma m%dn%dk%d bf16 RS/SS SM90 scaffold\n",
           BENCH_M, BENCH_N, BENCH_K);
    return 0;
}
