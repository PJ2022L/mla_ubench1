// FlashMLA V3.2 sparse-decode indexed 128-bit register loads.

#include <algorithm>
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
#include "sm90/decode/sparse_fp8/components/dequant.h"

namespace {

using microbench::CliArgs;
using microbench::DeviceBuffer;
using sm90::decode::sparse_fp8::L1CacheHint;
using sm90::decode::sparse_fp8::L2PrefetchHint;
using sm90::decode::sparse_fp8::load_128b_from_gmem;

constexpr int kThreads = 128;
constexpr int kTokensPerCta = 32;
constexpr int kPhysicalRowBytes = 656;
constexpr int kNopeBytes = 512;
constexpr int kScaleBytes = 16;
constexpr int kRopeOffset = kNopeBytes + kScaleBytes;
constexpr int kLoadsPerThread = 1 + 8 + 2;

__device__ __forceinline__ uint32_t fold(uint32_t value, const uint4& data) {
    value ^= data.x + 0x9e3779b9u;
    value = (value << 7) | (value >> 25);
    value += data.y ^ data.z;
    value ^= data.w + 0x85ebca6bu;
    return value;
}

__global__ void global_load_kernel(const uint8_t* __restrict__ kv,
                                   const int32_t* __restrict__ indices,
                                   uint64_t* __restrict__ starts,
                                   uint64_t* __restrict__ stops,
                                   uint32_t* __restrict__ sink,
                                   int index_blocks,
                                   int repeat) {
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int token_slot = warp * 8 + lane % 8;
    const int dimension_lane = lane / 8;
    uint32_t checksum = static_cast<uint32_t>(tid + 1);

    uint64_t start = 0;
    uint64_t stop = 0;
    CLK_START(start);
#pragma unroll 1
    for (int iteration = 0; iteration < repeat; ++iteration) {
        const int index_block = iteration % index_blocks;
        const int token = __ldg(indices + index_block * kTokensPerCta + token_slot);
        const uint8_t* row = kv + static_cast<int64_t>(token) * kPhysicalRowBytes;

        const uint4 scales = load_128b_from_gmem<
            uint4, L1CacheHint::EVICT_LAST, L2PrefetchHint::B128>(
                row + kNopeBytes);
        checksum = fold(checksum, scales);

#pragma unroll
        for (int tile = 0; tile < 8; ++tile) {
            const uint4 nope = load_128b_from_gmem<
                uint4, L1CacheHint::EVICT_LAST, L2PrefetchHint::B256>(
                    row + dimension_lane * 16 + tile * 64);
            checksum = fold(checksum, nope);
        }

#pragma unroll
        for (int tile = 0; tile < 2; ++tile) {
            const uint4 rope = load_128b_from_gmem<
                uint4, L1CacheHint::EVICT_LAST, L2PrefetchHint::B128>(
                    row + kRopeOffset + dimension_lane * 16 + tile * 64);
            checksum = fold(checksum, rope);
        }
    }
    CLK_STOP(stop);

    starts[tid] = start;
    stops[tid] = stop;
    sink[tid] = checksum;
}

std::vector<int32_t> make_indices(const std::string& pattern,
                                  int working_set_tokens,
                                  int index_blocks) {
    std::vector<int32_t> indices(
        static_cast<std::size_t>(index_blocks) * kTokensPerCta);
    uint32_t random_state = 0x1234567u;
    const int local_tokens = std::min(working_set_tokens, 256);
    for (int block = 0; block < index_blocks; ++block) {
        for (int slot = 0; slot < kTokensPerCta; ++slot) {
            int token = 0;
            if (pattern == "sequential") {
                token = (block * kTokensPerCta + slot) % working_set_tokens;
            } else if (pattern == "local") {
                token = (block * 17 + slot) % local_tokens;
            } else if (pattern == "random") {
                random_state = random_state * 1664525u + 1013904223u;
                token = static_cast<int>(random_state %
                                         static_cast<uint32_t>(working_set_tokens));
            } else {
                throw std::invalid_argument(
                    "--pattern must be sequential, local, or random");
            }
            indices[static_cast<std::size_t>(block) * kTokensPerCta + slot] =
                token;
        }
    }
    return indices;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const CliArgs args(argc, argv);
        const std::string pattern = args.get_string("pattern", "random");
        const int working_set_tokens =
            args.get_int("working-set-tokens", 32768, kTokensPerCta, 1 << 20);
        const int repeat = args.get_int("repeat", 256, 1, 1 << 20);
        const int default_index_blocks = std::max(
            1, std::min(repeat, (working_set_tokens + kTokensPerCta - 1) /
                                    kTokensPerCta));
        const int index_blocks =
            args.get_int("index-blocks", default_index_blocks, 1, 1 << 20);
        const int warmup = args.get_int("warmup", 5, 0, 1000);
        const int samples = args.get_int("samples", 20, 1, 10000);
        const auto device = microbench::require_sm90(args.get_int("device", 0));

        const std::size_t kv_bytes =
            static_cast<std::size_t>(working_set_tokens) * kPhysicalRowBytes;
        std::vector<uint8_t> host_kv(kv_bytes);
        for (std::size_t i = 0; i < host_kv.size(); ++i) {
            host_kv[i] = static_cast<uint8_t>((i * 37u + 11u) & 0xffu);
        }
        const auto host_indices =
            make_indices(pattern, working_set_tokens, index_blocks);

        DeviceBuffer<uint8_t> kv(host_kv.size());
        DeviceBuffer<int32_t> indices(host_indices.size());
        DeviceBuffer<uint64_t> starts(kThreads);
        DeviceBuffer<uint64_t> stops(kThreads);
        DeviceBuffer<uint32_t> sink(kThreads);
        kv.copy_from_host(host_kv);
        indices.copy_from_host(host_indices);

        auto measure_once = [&]() -> double {
            global_load_kernel<<<1, kThreads>>>(
                kv.data(), indices.data(), starts.data(), stops.data(), sink.data(),
                index_blocks, repeat);
            microbench::throw_if_cuda_error(cudaGetLastError(),
                                             "global_load_kernel launch");
            microbench::throw_if_cuda_error(cudaDeviceSynchronize(),
                                             "global_load_kernel synchronize");
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
                "global-load sink is zero; target work may have been removed");
        }

        const double source_calls = static_cast<double>(kThreads) *
                                    kLoadsPerThread * repeat;
        const double requested_bytes = source_calls * 16.0;
        const double physical_bytes = static_cast<double>(kTokensPerCta) *
                                      kPhysicalRowBytes * repeat;
        microbench::JsonLine json;
        json.add("benchmark", "global_load/128b_nc_l2_sm90")
            .add("gpu", device.properties.name)
            .add("pattern", pattern)
            .add("working_set_tokens", working_set_tokens)
            .add("working_set_bytes", kv_bytes)
            .add("index_blocks", index_blocks)
            .add("repeat", repeat)
            .add("threads", kThreads)
            .add("loads_per_thread_per_block", kLoadsPerThread)
            .add("checksum", checksum);
        microbench::add_measurement_summary(json, summary);
        json.add("cycle_per_source_load", summary.median /
                    (static_cast<double>(repeat) * kLoadsPerThread))
            .add("source_load_per_clk_sm", source_calls / summary.median)
            .add("requested_byte_per_clk_sm", requested_bytes / summary.median)
            .add("physical_byte_per_clk_sm", physical_bytes / summary.median)
            .print();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "global-load benchmark error: " << error.what() << '\n';
        return 1;
    }
}
