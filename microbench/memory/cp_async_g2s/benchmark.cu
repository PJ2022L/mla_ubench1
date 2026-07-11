// FlashMLA sparse-prefill indexed BF16 cp.async GMEM-to-shared gather.

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
#include "sm90/helpers.h"
#include "sm90/prefill/sparse/config.h"

namespace {

using microbench::CliArgs;
using microbench::DeviceBuffer;
using Kernel = sm90::fwd::KernelTemplate<576, false>;
using TileLayout = Kernel::SmemLayoutKTiles<1>;

constexpr int kThreads = 128;
constexpr int kRows = 64;
constexpr int kDimension = 576;
constexpr int kGroups = 16;
constexpr int kRowsPerGroup = 4;
constexpr int kBufferElements = kRows * kDimension;
constexpr int kBufferBytes = kBufferElements * sizeof(bf16);
constexpr int kBufferCount = 2;
constexpr int kBarrierCount = 4;
constexpr int kBarrierOffset = kBufferBytes * kBufferCount;
constexpr int kSharedBytes = kBarrierOffset + kBarrierCount * sizeof(uint64_t);

enum class Mode : int {
    Block = 0,
    Pair = 1,
};

__device__ __forceinline__ void commit_to_mbar(transac_bar_t& barrier) {
    cutlass::arch::cpasync_barrier_arrive_noinc(
        reinterpret_cast<uint64_t*>(&barrier));
}

__global__ void cp_async_gather_kernel(const bf16* __restrict__ kv,
                                       const int32_t* __restrict__ indices,
                                       uint64_t* __restrict__ starts,
                                       uint64_t* __restrict__ stops,
                                       uint32_t* __restrict__ sink,
                                       int index_blocks,
                                       int repeat,
                                       Mode mode) {
    extern __shared__ __align__(16) unsigned char shared_raw[];
    bf16* const buffers = reinterpret_cast<bf16*>(shared_raw);
    transac_bar_t* const barriers =
        reinterpret_cast<transac_bar_t*>(shared_raw + kBarrierOffset);

    const int tid = threadIdx.x;
    const int idx_in_group = tid % 8;
    const int group_idx = tid / 8;
    auto first_tile = cute::make_tensor(
        cute::make_smem_ptr(buffers), TileLayout{});
    bf16* const my_shared_base = &first_tile(group_idx, idx_in_group * 8);
    const int64_t cache_policy = sm90::createpolicy_evict_last();

    if (tid == 0) {
#pragma unroll
        for (int i = 0; i < kBarrierCount; ++i) {
            barriers[i].init(kThreads);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    auto copy_segment = [&](int iteration,
                            int logical_block,
                            int buffer_idx,
                            int tile_start,
                            int tile_end) {
        const int tokens_per_iteration =
            mode == Mode::Pair ? 2 * kRows : kRows;
        const int index_block = iteration % index_blocks;
        const int* const iteration_indices =
            indices + index_block * tokens_per_iteration + logical_block * kRows;
#pragma unroll
        for (int local_row = 0; local_row < kRowsPerGroup; ++local_row) {
            const int row = local_row * kGroups + group_idx;
            const int token = __ldg(iteration_indices + row);
            const bf16* const global_base =
                kv + static_cast<int64_t>(token) * kDimension + idx_in_group * 8;
#pragma unroll
            for (int tile = tile_start; tile < tile_end; ++tile) {
                bf16* const shared_dst =
                    my_shared_base + buffer_idx * kBufferElements +
                    tile * (kRows * 64) + local_row * (kGroups * 64);
                sm90::cp_async_cacheglobal_l2_prefetch_256B(
                    global_base + tile * 64, shared_dst, true, cache_policy);
            }
        }
    };

    uint32_t phase = 0;
    uint64_t start = 0;
    uint64_t stop = 0;
    CLK_START(start);
#pragma unroll 1
    for (int iteration = 0; iteration < repeat; ++iteration) {
        if (mode == Mode::Pair) {
            copy_segment(iteration, 0, 0, 0, 4);  // K0-left
            commit_to_mbar(barriers[0]);
            copy_segment(iteration, 1, 1, 4, 9);  // K1-right
            commit_to_mbar(barriers[1]);
            copy_segment(iteration, 0, 0, 4, 9);  // K0-right
            commit_to_mbar(barriers[2]);
            copy_segment(iteration, 1, 1, 0, 4);  // K1-left
            commit_to_mbar(barriers[3]);
        } else {
            copy_segment(iteration, 0, 0, 0, 9);
            commit_to_mbar(barriers[0]);
        }

        if (tid == 0) {
            barriers[0].wait(phase);
            if (mode == Mode::Pair) {
                barriers[1].wait(phase);
                barriers[2].wait(phase);
                barriers[3].wait(phase);
            }
        }
        __syncthreads();
        phase ^= 1;
    }
    CLK_STOP(stop);

    const uint32_t* const words = reinterpret_cast<const uint32_t*>(buffers);
    starts[tid] = start;
    stops[tid] = stop;
    sink[tid] = words[(tid * 17) % (kBufferBytes / sizeof(uint32_t))] ^
                static_cast<uint32_t>(tid + 1);
}

Mode parse_mode(const std::string& value) {
    if (value == "block") {
        return Mode::Block;
    }
    if (value == "pair") {
        return Mode::Pair;
    }
    throw std::invalid_argument("--mode must be block or pair");
}

std::vector<int32_t> make_indices(const std::string& pattern,
                                  int working_set_tokens,
                                  int index_blocks,
                                  int tokens_per_iteration) {
    std::vector<int32_t> result(
        static_cast<std::size_t>(index_blocks) * tokens_per_iteration);
    uint32_t random_state = 0x31415926u;
    const int local_tokens = std::min(working_set_tokens, 256);
    for (int block = 0; block < index_blocks; ++block) {
        for (int row = 0; row < tokens_per_iteration; ++row) {
            int token = 0;
            if (pattern == "sequential") {
                token = (block * tokens_per_iteration + row) % working_set_tokens;
            } else if (pattern == "local") {
                token = (block * 29 + row) % local_tokens;
            } else if (pattern == "random") {
                random_state = random_state * 1664525u + 1013904223u;
                token = static_cast<int>(random_state %
                                         static_cast<uint32_t>(working_set_tokens));
            } else {
                throw std::invalid_argument(
                    "--pattern must be sequential, local, or random");
            }
            result[static_cast<std::size_t>(block) * tokens_per_iteration + row] =
                token;
        }
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const CliArgs args(argc, argv);
        const std::string mode_name = args.get_string("mode", "pair");
        const Mode mode = parse_mode(mode_name);
        const std::string pattern = args.get_string("pattern", "random");
        const int working_set_tokens =
            args.get_int("working-set-tokens", 8192, kRows, 1 << 20);
        const int repeat = args.get_int("repeat", 32, 1, 1 << 16);
        const int tokens_per_iteration =
            mode == Mode::Pair ? 2 * kRows : kRows;
        const int default_index_blocks = std::max(
            1, std::min(repeat, (working_set_tokens + tokens_per_iteration - 1) /
                                    tokens_per_iteration));
        const int index_blocks =
            args.get_int("index-blocks", default_index_blocks, 1, 1 << 18);
        const int warmup = args.get_int("warmup", 5, 0, 1000);
        const int samples = args.get_int("samples", 20, 1, 10000);
        const auto device = microbench::require_sm90(args.get_int("device", 0));

        std::vector<uint16_t> host_kv(
            static_cast<std::size_t>(working_set_tokens) * kDimension);
        for (std::size_t i = 0; i < host_kv.size(); ++i) {
            host_kv[i] = static_cast<uint16_t>((i * 17u + 3u) & 0xffffu);
        }
        const auto host_indices = make_indices(
            pattern, working_set_tokens, index_blocks, tokens_per_iteration);

        DeviceBuffer<uint16_t> kv_bits(host_kv.size());
        DeviceBuffer<int32_t> indices(host_indices.size());
        DeviceBuffer<uint64_t> starts(kThreads);
        DeviceBuffer<uint64_t> stops(kThreads);
        DeviceBuffer<uint32_t> sink(kThreads);
        kv_bits.copy_from_host(host_kv);
        indices.copy_from_host(host_indices);

        microbench::throw_if_cuda_error(
            cudaFuncSetAttribute(cp_async_gather_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 kSharedBytes),
            "cp_async_gather_kernel dynamic shared-memory attribute");

        auto measure_once = [&]() -> double {
            cp_async_gather_kernel<<<1, kThreads, kSharedBytes>>>(
                reinterpret_cast<const bf16*>(kv_bits.data()), indices.data(),
                starts.data(), stops.data(), sink.data(), index_blocks, repeat,
                mode);
            microbench::throw_if_cuda_error(cudaGetLastError(),
                                             "cp_async_gather_kernel launch");
            microbench::throw_if_cuda_error(cudaDeviceSynchronize(),
                                             "cp_async_gather_kernel synchronize");
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
                "cp.async sink is zero; target work may have been removed");
        }

        const int calls_per_thread = mode == Mode::Pair ? 72 : 36;
        const double copies =
            static_cast<double>(kThreads) * calls_per_thread * repeat;
        const double bytes = copies * 16.0;
        microbench::JsonLine json;
        json.add("benchmark", "cp_async_g2s/gather64x576_bf16_sm90")
            .add("gpu", device.properties.name)
            .add("mode", mode_name)
            .add("pattern", pattern)
            .add("working_set_tokens", working_set_tokens)
            .add("working_set_bytes",
                 static_cast<std::size_t>(working_set_tokens) * kDimension *
                     sizeof(uint16_t))
            .add("index_blocks", index_blocks)
            .add("repeat", repeat)
            .add("threads", kThreads)
            .add("copies_per_thread_per_iteration", calls_per_thread)
            .add("checksum", checksum);
        microbench::add_measurement_summary(json, summary);
        json.add("cycle_per_block_or_pair", summary.median / repeat)
            .add("cycle_per_source_copy", summary.median /
                    (static_cast<double>(repeat) * calls_per_thread))
            .add("source_copy_per_clk_sm", copies / summary.median)
            .add("byte_per_clk_sm", bytes / summary.median)
            .print();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cp.async G2S benchmark error: " << error.what() << '\n';
        return 1;
    }
}
