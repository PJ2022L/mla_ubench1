// FlashMLA sparse-decode 128-bit register-to-shared producer stores.

#include <cstdint>
#include <exception>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "benchmark_utils.hpp"
#include "clock.cuh"

namespace {

using microbench::CliArgs;
using microbench::DeviceBuffer;

constexpr int kThreads = 128;
constexpr int kStoresPerThread = 18;
constexpr int kRows = 64;
constexpr int kDimension = 576;
constexpr int kTileBytes = kRows * kDimension * sizeof(uint16_t);

enum class Pattern : int {
    KMajor = 0,
    Linear = 1,
    Hot = 2,
};

__device__ __forceinline__ void store_shared_128(void* destination,
                                                 const uint4& value) {
    const uint32_t address =
        static_cast<uint32_t>(__cvta_generic_to_shared(destination));
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n"
                 :
                 : "r"(address), "r"(value.x), "r"(value.y), "r"(value.z),
                   "r"(value.w)
                 : "memory");
}

__device__ __forceinline__ int kmajor_byte_offset(int tid, int store_idx) {
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int token = warp * 8 + lane % 8;
    const int dimension_lane = lane / 8;

    int dimension_base = 0;
    if (store_idx < 16) {
        const int nope_tile = store_idx / 2;
        const int half = store_idx % 2;
        dimension_base = nope_tile * 64 + dimension_lane * 16 + half * 8;
    } else {
        const int rope_tile = store_idx - 16;
        dimension_base = 512 + rope_tile * 32 + dimension_lane * 8;
    }
    const int element_offset = dimension_base * kRows + token * 8;
    return element_offset * static_cast<int>(sizeof(uint16_t));
}

__global__ void shared_store_kernel(uint64_t* __restrict__ starts,
                                    uint64_t* __restrict__ stops,
                                    uint32_t* __restrict__ sink,
                                    int repeat,
                                    int working_set_bytes,
                                    Pattern pattern) {
    extern __shared__ __align__(16) unsigned char shared_bytes[];
    const int tid = threadIdx.x;
    const int slots = working_set_bytes / 16;
    uint4 value = {
        static_cast<uint32_t>(tid + 1),
        static_cast<uint32_t>(tid * 3 + 5),
        static_cast<uint32_t>(tid * 7 + 11),
        static_cast<uint32_t>(tid * 13 + 17),
    };

    uint64_t start = 0;
    uint64_t stop = 0;
    CLK_START(start);
#pragma unroll 1
    for (int iteration = 0; iteration < repeat; ++iteration) {
#pragma unroll
        for (int store_idx = 0; store_idx < kStoresPerThread; ++store_idx) {
            int byte_offset = 0;
            if (pattern == Pattern::KMajor) {
                byte_offset = kmajor_byte_offset(tid, store_idx);
            } else if (pattern == Pattern::Linear) {
                const int slot =
                    (tid * kStoresPerThread + store_idx + iteration) % slots;
                byte_offset = slot * 16;
            } else {
                const int slot = (store_idx * 8 + tid % 8) % slots;
                byte_offset = slot * 16;
            }
            store_shared_128(shared_bytes + byte_offset, value);
        }
        value.x += 0x9e3779b9u;
        value.y ^= value.x;
    }
    CLK_STOP(stop);

    __syncthreads();
    const int sink_slot =
        pattern == Pattern::KMajor
            ? kmajor_byte_offset(tid, tid % kStoresPerThread) / 16
            : (tid * 17) % slots;
    const uint4 observed =
        reinterpret_cast<const uint4*>(shared_bytes)[sink_slot];
    starts[tid] = start;
    stops[tid] = stop;
    sink[tid] = observed.x ^ observed.y ^ observed.z ^ observed.w ^
                static_cast<uint32_t>(tid + 1);
}

Pattern parse_pattern(const std::string& value) {
    if (value == "kmajor") {
        return Pattern::KMajor;
    }
    if (value == "linear") {
        return Pattern::Linear;
    }
    if (value == "hot") {
        return Pattern::Hot;
    }
    throw std::invalid_argument("--pattern must be kmajor, linear, or hot");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const CliArgs args(argc, argv);
        const std::string pattern_name = args.get_string("pattern", "kmajor");
        const Pattern pattern = parse_pattern(pattern_name);
        int working_set_bytes =
            args.get_int("working-set-bytes", kTileBytes, 2048, 196608);
        working_set_bytes = (working_set_bytes / 16) * 16;
        if (pattern == Pattern::KMajor && working_set_bytes < kTileBytes) {
            throw std::invalid_argument(
                "kmajor pattern requires --working-set-bytes >= 73728");
        }
        const int repeat = args.get_int("repeat", 512, 1, 1 << 20);
        const int warmup = args.get_int("warmup", 5, 0, 1000);
        const int samples = args.get_int("samples", 20, 1, 10000);
        const auto device = microbench::require_sm90(args.get_int("device", 0));

        DeviceBuffer<uint64_t> starts(kThreads);
        DeviceBuffer<uint64_t> stops(kThreads);
        DeviceBuffer<uint32_t> sink(kThreads);
        microbench::throw_if_cuda_error(
            cudaFuncSetAttribute(shared_store_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 working_set_bytes),
            "shared_store_kernel dynamic shared-memory attribute");

        auto measure_once = [&]() -> double {
            shared_store_kernel<<<1, kThreads, working_set_bytes>>>(
                starts.data(), stops.data(), sink.data(), repeat,
                working_set_bytes, pattern);
            microbench::throw_if_cuda_error(cudaGetLastError(),
                                             "shared_store_kernel launch");
            microbench::throw_if_cuda_error(cudaDeviceSynchronize(),
                                             "shared_store_kernel synchronize");
            const auto host_starts = starts.copy_to_host();
            const auto host_stops = stops.copy_to_host();
            return static_cast<double>(
                microbench::reduce_cycles(host_starts, host_stops));
        };

        const auto series = microbench::run_samples(warmup, samples, measure_once);
        const auto summary = series.summary();
        const auto host_sink = sink.copy_to_host();
        const uint64_t checksum = std::accumulate(
            host_sink.begin(), host_sink.end(), uint64_t{0});
        if (checksum == 0) {
            throw std::runtime_error(
                "shared-store sink is zero; target work may have been removed");
        }

        const double stores = static_cast<double>(kThreads) *
                              kStoresPerThread * repeat;
        const double bytes = stores * 16.0;
        microbench::JsonLine json;
        json.add("benchmark", "shared_store/128b_sm90")
            .add("gpu", device.properties.name)
            .add("pattern", pattern_name)
            .add("working_set_bytes", working_set_bytes)
            .add("repeat", repeat)
            .add("threads", kThreads)
            .add("stores_per_thread_per_block", kStoresPerThread)
            .add("checksum", checksum);
        microbench::add_measurement_summary(json, summary);
        json.add("cycle_per_source_store", summary.median /
                    (static_cast<double>(repeat) * kStoresPerThread))
            .add("source_store_per_clk_sm", stores / summary.median)
            .add("byte_per_clk_sm", bytes / summary.median)
            .print();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "shared-store benchmark error: " << error.what() << '\n';
        return 1;
    }
}
