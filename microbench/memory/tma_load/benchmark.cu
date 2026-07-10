// 2D BF16 TMA-load micro-benchmark. SM90 scaffold.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include "clock.cuh"
#include "tma_util.cuh"

int main() {
    // TODO(impl): create CUtensorMap, issue SM90_TMA_LOAD into swizzled shared memory,
    // wait on the transaction barrier, and report cycle/tile + byte/clk/SM.
    printf("[TODO] tma_load parameterized scaffold\n");
    return 0;
}
