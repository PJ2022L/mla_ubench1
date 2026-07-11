#pragma once
// CUDA runtime error checking shared by all benchmarks.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace microbench {
inline void gpuAssert(cudaError_t code,
                      const char* file,
                      int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        std::fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) std::exit(code);
    }
}
}  // namespace microbench

#define gpuErrchk(ans)                                                     \
    do {                                                                   \
        ::microbench::gpuAssert((ans), __FILE__, __LINE__);                \
    } while (0)

#define GPU_CHECK(ans) gpuErrchk(ans)
#define GPU_CHECK_LAST() gpuErrchk(cudaGetLastError())
