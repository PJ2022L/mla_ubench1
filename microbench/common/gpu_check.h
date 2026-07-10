#pragma once
// gpuErrchk —— 复用 ref_ubench 惯例（每个 bench 都有这一段）。
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define gpuErrchk(ans) { microbench::gpuAssert((ans), __FILE__, __LINE__); }
namespace microbench {
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        std::fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) std::exit(code);
    }
}
}  // namespace microbench
