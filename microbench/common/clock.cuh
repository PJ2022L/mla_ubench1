#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

namespace microbench {

// Per-SM cycle counter. Use it only for threads that execute on the same SM;
// CUDA events are the authoritative timer for whole-grid throughput.
__device__ __forceinline__ uint64_t device_clock64() {
    uint64_t value;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(value) :: "memory");
    return value;
}

__device__ __forceinline__ void block_clock_barrier() {
    asm volatile("bar.sync 0;" ::: "memory");
}

struct ClockStats {
    std::size_t count = 0;
    uint64_t min_start = 0;
    uint64_t max_start = 0;
    uint64_t min_stop = 0;
    uint64_t max_stop = 0;
    uint64_t envelope_cycles = 0;
    uint64_t min_thread_cycles = 0;
    uint64_t max_thread_cycles = 0;
    double mean_thread_cycles = 0.0;
};

// Reduces per-thread timestamps from one synchronized CTA/warpgroup. The
// envelope is max(stop) - min(start); per-thread fields expose skew directly.
inline ClockStats reduce_clocks(const uint64_t* starts,
                                const uint64_t* stops,
                                std::size_t count) {
    if (starts == nullptr || stops == nullptr) {
        throw std::invalid_argument("reduce_clocks received a null pointer");
    }
    if (count == 0) {
        throw std::invalid_argument("reduce_clocks requires at least one sample");
    }

    ClockStats result;
    result.count = count;
    result.min_start = std::numeric_limits<uint64_t>::max();
    result.min_stop = std::numeric_limits<uint64_t>::max();
    result.min_thread_cycles = std::numeric_limits<uint64_t>::max();

    long double elapsed_sum = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        if (stops[i] < starts[i]) {
            throw std::runtime_error("clock64 stop precedes start");
        }
        const uint64_t elapsed = stops[i] - starts[i];
        result.min_start = std::min(result.min_start, starts[i]);
        result.max_start = std::max(result.max_start, starts[i]);
        result.min_stop = std::min(result.min_stop, stops[i]);
        result.max_stop = std::max(result.max_stop, stops[i]);
        result.min_thread_cycles = std::min(result.min_thread_cycles, elapsed);
        result.max_thread_cycles = std::max(result.max_thread_cycles, elapsed);
        elapsed_sum += static_cast<long double>(elapsed);
    }

    result.envelope_cycles = result.max_stop - result.min_start;
    result.mean_thread_cycles =
        static_cast<double>(elapsed_sum / static_cast<long double>(count));
    return result;
}

inline uint64_t reduce_cycles(const uint64_t* starts,
                              const uint64_t* stops,
                              std::size_t count) {
    return reduce_clocks(starts, stops, count).envelope_cycles;
}

inline ClockStats reduce_clocks(const std::vector<uint64_t>& starts,
                                const std::vector<uint64_t>& stops) {
    if (starts.size() != stops.size()) {
        throw std::invalid_argument("clock timestamp vectors have different sizes");
    }
    return reduce_clocks(starts.data(), stops.data(), starts.size());
}

inline uint64_t reduce_cycles(const std::vector<uint64_t>& starts,
                              const std::vector<uint64_t>& stops) {
    return reduce_clocks(starts, stops).envelope_cycles;
}

inline int nominal_sm_clock_khz(int device = 0) {
    int clock_khz = 0;
    const cudaError_t status =
        cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device);
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
    return clock_khz;
}

}  // namespace microbench

// These compatibility macros synchronize the whole CTA. Every thread in the
// CTA must execute them. Prefer uint64_t timestamp variables.
#define CLK_BAR() ::microbench::block_clock_barrier()
#define CLK_START(v)                       \
    do {                                   \
        static_assert(sizeof(v) == sizeof(uint64_t), \
                      "CLK_START requires uint64_t storage"); \
        CLK_BAR();                         \
        (v) = ::microbench::device_clock64(); \
    } while (0)
#define CLK_STOP(v)                        \
    do {                                   \
        static_assert(sizeof(v) == sizeof(uint64_t), \
                      "CLK_STOP requires uint64_t storage"); \
        CLK_BAR();                         \
        (v) = ::microbench::device_clock64(); \
    } while (0)
