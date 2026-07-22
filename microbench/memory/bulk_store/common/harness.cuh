#pragma once

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/bench.hpp"
#include "ptx.cuh"

namespace microbench::bulk_store_bench {
namespace {

constexpr int kThreads = 256;
constexpr int kRows = 64;
constexpr int kColumns = 512;
constexpr int kSharedStride = 520;
constexpr int kRowBytes = kColumns * static_cast<int>(sizeof(float));
constexpr int kTileBytes = kRows * kRowBytes;
constexpr int kSharedElements = kRows * kSharedStride;
constexpr int kSharedBytes = kSharedElements * static_cast<int>(sizeof(float));
constexpr int kMaxWorkingTiles = 1 << 19;

__host__ __device__ constexpr float row_pattern(int row) {
    return static_cast<float>(row + 1);
}

enum class Pattern : int {
    kLocal,
    kSequential,
    kRandom,
};

Pattern parse_pattern(const std::string& value) {
    if (value == "local") return Pattern::kLocal;
    if (value == "sequential") return Pattern::kSequential;
    if (value == "random") return Pattern::kRandom;
    throw std::invalid_argument(
        "--pattern must be local, sequential, or random");
}

__device__ __forceinline__ int select_tile(Pattern pattern,
                                           int iteration,
                                           int working_tiles,
                                           int block,
                                           int grid_blocks) {
    const int tiles_per_block = working_tiles / grid_blocks;
    const int block_base = block * tiles_per_block;
    if (pattern == Pattern::kLocal) {
        return block_base;
    }
    if (pattern == Pattern::kRandom) {
        uint32_t value =
            static_cast<uint32_t>(iteration + 1) * 0x85ebca6bu;
        value ^= value >> 16;
        value *= 0x7feb352du;
        value ^= value >> 15;
        const int offset = static_cast<int>(
            value % static_cast<uint32_t>(tiles_per_block));
        return block_base + offset;
    }
    return block_base + iteration % tiles_per_block;
}

template <bool Target = true>
__global__ void bulk_store_kernel(float* output,
                                  uint64_t* cycles,
                                  uint32_t* sinks,
                                  int iterations,
                                  int working_tiles,
                                  Pattern pattern) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    extern __shared__ __align__(128) float shared_tile[];
    __shared__ uint64_t issuer_warp_cycles[2];
    for (int index = threadIdx.x; index < kSharedElements;
         index += blockDim.x) {
        shared_tile[index] = row_pattern(index / kSharedStride);
    }
    __syncthreads();
    microbench::ptx::async_shared_fence();
    __syncthreads();

    int final_tile = 0;
    const uint64_t start = microbench::read_clock64();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        final_tile = select_tile(
            pattern, iteration, working_tiles,
            static_cast<int>(blockIdx.x), static_cast<int>(gridDim.x));
        if (threadIdx.x < kRows) {
            if constexpr (Target) {
                const int row = threadIdx.x;
                microbench::ptx::bulk_store_shared_to_global(
                    output +
                        (static_cast<int64_t>(final_tile) * kRows + row) *
                            kColumns,
                    shared_tile + row * kSharedStride,
                    kRowBytes);
                microbench::ptx::bulk_commit_group();
                microbench::ptx::bulk_wait_group<0>();
            } else {
                asm volatile("" : : "r"(final_tile));
            }
        }
    }
    const uint64_t stop = microbench::read_clock64();
    if (threadIdx.x < kRows && (threadIdx.x & 31) == 0) {
        issuer_warp_cycles[threadIdx.x / 32] = stop - start;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        cycles[blockIdx.x] = issuer_warp_cycles[0] > issuer_warp_cycles[1]
            ? issuer_warp_cycles[0]
            : issuer_warp_cycles[1];
        sinks[blockIdx.x] = static_cast<uint32_t>(final_tile + 1);
    }
#else
    if (threadIdx.x == 0) {
        cycles[blockIdx.x] = 0;
        sinks[blockIdx.x] = 0;
    }
#endif
}

}  // namespace

inline int run(int argc, char** argv) {
    try {
        const microbench::Args args(argc, argv);
        args.require_only({
            "iters", "warmup", "samples", "blocks", "device", "peak",
            "working-set-tiles", "pattern"});
        const auto options = microbench::parse_common_options(args, 128);
        const auto properties = microbench::require_sm90(options.device);
        const int blocks = microbench::resolve_blocks(options.blocks, properties, 1);
        if (blocks > kMaxWorkingTiles) {
            throw std::invalid_argument(
                "--blocks exceeds the supported collision-free bulk-store "
                "working set");
        }
        const int working_tiles = args.get_int(
            "working-set-tiles", std::max(2048, blocks), blocks,
            kMaxWorkingTiles);
        const int tiles_per_block = working_tiles / blocks;
        const std::string pattern_name =
            args.get_string("pattern", "sequential");
        const Pattern pattern = parse_pattern(pattern_name);
        if (kSharedBytes > static_cast<int>(properties.sharedMemPerBlockOptin)) {
            throw std::runtime_error(
                "64x520 FP32 shared tile exceeds "
                "sharedMemPerBlockOptin");
        }
        CUDA_CHECK(cudaFuncSetAttribute(
            bulk_store_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            kSharedBytes));
        CUDA_CHECK(cudaFuncSetAttribute(
            bulk_store_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            kSharedBytes));

        const std::size_t output_elements =
            static_cast<std::size_t>(working_tiles) * kRows * kColumns;
        microbench::DeviceBuffer<float> output(output_elements);
        output.zero();

        microbench::DeviceBuffer<uint64_t> latency_cycles(1);
        microbench::DeviceBuffer<uint64_t> latency_baseline_cycles(1);
        microbench::DeviceBuffer<uint32_t> latency_sinks(1);
        microbench::DeviceBuffer<uint32_t> latency_baseline_sinks(1);
        const auto latency_clock_samples =
            microbench::measure_paired_clock_cycles(
            options.warmup, options.samples, latency_cycles.data(),
            latency_baseline_cycles.data(), 1,
            [&] {
                bulk_store_kernel<true><<<1, kThreads, kSharedBytes>>>(
                    output.data(), latency_cycles.data(), latency_sinks.data(),
                    options.iters, working_tiles, pattern);
                bulk_store_kernel<false><<<1, kThreads, kSharedBytes>>>(
                    output.data(), latency_baseline_cycles.data(),
                    latency_baseline_sinks.data(), options.iters,
                    working_tiles, pattern);
            });
        std::vector<double> latency_metric_samples;
        latency_metric_samples.reserve(options.samples);
        for (int index = 0; index < options.samples; ++index) {
            latency_metric_samples.push_back(
                (latency_clock_samples.target[index] -
                 latency_clock_samples.baseline[index]) /
                options.iters);
        }
        const double completion_cycles =
            microbench::median(latency_metric_samples);

        microbench::DeviceBuffer<uint64_t> throughput_cycles(blocks);
        microbench::DeviceBuffer<uint64_t> throughput_baseline_cycles(blocks);
        microbench::DeviceBuffer<uint32_t> throughput_sinks(blocks);
        microbench::DeviceBuffer<uint32_t> throughput_baseline_sinks(blocks);
        const auto initiation_clock_samples =
            microbench::measure_paired_clock_cycles(
                options.warmup, options.samples, throughput_cycles.data(),
                throughput_baseline_cycles.data(), blocks,
                [&] {
                    bulk_store_kernel<true>
                        <<<blocks, kThreads, kSharedBytes>>>(
                            output.data(), throughput_cycles.data(),
                            throughput_sinks.data(), options.iters,
                            working_tiles, pattern);
                    bulk_store_kernel<false>
                        <<<blocks, kThreads, kSharedBytes>>>(
                            output.data(), throughput_baseline_cycles.data(),
                            throughput_baseline_sinks.data(), options.iters,
                            working_tiles, pattern);
                });
        std::vector<double> initiation_interval_samples;
        initiation_interval_samples.reserve(options.samples);
        for (int index = 0; index < options.samples; ++index) {
            initiation_interval_samples.push_back(
                (initiation_clock_samples.target[index] -
                 initiation_clock_samples.baseline[index]) /
                options.iters);
        }
        const double initiation_interval_cycles =
            microbench::median(initiation_interval_samples);
        const auto event_samples = microbench::measure_event_ms(
            options.warmup,
            options.samples,
            [&] {
                bulk_store_kernel<true><<<blocks, kThreads, kSharedBytes>>>(
                    output.data(), throughput_cycles.data(),
                    throughput_sinks.data(), options.iters, working_tiles,
                    pattern);
            });
        const double elapsed_ms = microbench::median(event_samples);
        const auto host_sinks = throughput_sinks.copy_to_host();
        for (uint32_t value : host_sinks) {
            if (value == 0) {
                throw std::runtime_error("bulk-store completion sink is zero");
            }
        }

        output.zero();
        bulk_store_kernel<true><<<1, kThreads, kSharedBytes>>>(
            output.data(), latency_cycles.data(), latency_sinks.data(), 1,
            working_tiles, Pattern::kSequential);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> validation_tile(
            static_cast<std::size_t>(kRows) * kColumns);
        CUDA_CHECK(cudaMemcpy(
            validation_tile.data(), output.data(), kTileBytes,
            cudaMemcpyDeviceToHost));
        for (int row = 0; row < kRows; ++row) {
            for (int column = 0; column < kColumns; ++column) {
                if (validation_tile[row * kColumns + column] !=
                    row_pattern(row)) {
                    throw std::runtime_error(
                        "bulk-store row-mapped full-tile validation failed");
                }
            }
        }

        const double tiles = static_cast<double>(blocks) * options.iters;
        const double row_transactions = tiles * kRows;
        const double requested_bytes = tiles * kTileBytes;
        auto throughput_metric_samples = event_samples;
        auto bandwidth_metric_samples = event_samples;
        for (double& value : throughput_metric_samples) {
            value = tiles / value / 1.0e6;
        }
        for (double& value : bandwidth_metric_samples) {
            value = requested_bytes / value / 1.0e6;
        }
        const double giga_tiles_per_second =
            microbench::median(throughput_metric_samples);
        const double bandwidth_gbps =
            microbench::median(bandwidth_metric_samples);

        microbench::JsonObject params;
        params.add("gpu", properties.name)
            .add("tile_rows", kRows)
            .add("tile_columns", kColumns)
            .add("dtype", "f32")
            .add("shared_stride", kSharedStride)
            .add("active_issuers", kRows)
            .add("bytes_per_row_transaction", kRowBytes)
            .add("row_transactions_per_tile", kRows)
            .add("pattern", pattern_name)
            .add("working_set_tiles", working_tiles)
            .add("working_set_bytes",
                 static_cast<uint64_t>(working_tiles) * kTileBytes)
            .add("tiles_per_block", tiles_per_block)
            .add("unused_tiles", working_tiles - tiles_per_block * blocks)
            .add("blocks", options.blocks)
            .add("resolved_blocks", blocks)
            .add("iters", options.iters)
            .add("warmup", options.warmup)
            .add("samples", options.samples)
            .add("device", options.device)
            .add("peak", options.peak)
            .add("initiation_interval_cycles", initiation_interval_cycles)
            .add("initiation_interval_boundary",
                 "target protocol minus matched tile-selection loop baseline")
            .add("clock_baseline",
                 "same two issuer warps, tile selection, loop, and sink; no "
                 "bulk store, commit_group, or wait_group")
            .add("correct", true)
            .add("correctness",
                 "full 64x512 tile with distinct pattern per source row")
            .add("address_mapping",
                 "per-CTA disjoint interval; collision-free at arbitrary progress");

        microbench::JsonObject latency;
        latency.add("value", completion_cycles)
            .add("unit", "cycle/32-row-issuer-warp")
            .add("timer", "clock64")
            .add("scope", "max of two issuer warps")
            .add("boundary",
                 "baseline-subtracted 32 lane stores+per-thread commit+wait0")
            .add_raw("samples",
                     microbench::json_number_array(latency_metric_samples))
            .add_raw("target_samples_cycles",
                     microbench::json_number_array(
                         latency_clock_samples.target))
            .add_raw("baseline_samples_cycles",
                     microbench::json_number_array(
                         latency_clock_samples.baseline));

        microbench::JsonObject throughput;
        throughput.add("value", giga_tiles_per_second)
            .add("unit", "Gtile/s")
            .add("timer", "cuda_event")
            .add("scope", "grid")
            .add("event_ms", elapsed_ms)
            .add("tiles", tiles)
            .add("row_transactions", row_transactions)
            .add_raw("samples",
                     microbench::json_number_array(throughput_metric_samples))
            .add_raw("event_samples_ms",
                     microbench::json_number_array(event_samples));

        microbench::JsonObject bandwidth;
        bandwidth.add("value", bandwidth_gbps)
            .add("unit", "GB/s")
            .add("kind", "requested")
            .add("bytes", requested_bytes)
            .add_raw("samples",
                     microbench::json_number_array(bandwidth_metric_samples));

        microbench::print_result(
            "cp_async_bulk_s2g_64x512_f32",
            params,
            latency,
            throughput,
            bandwidth,
            microbench::utilization(
                bandwidth_gbps, options.peak, "GB/s"));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "bulk_store: " << error.what() << '\n';
        return 1;
    }
}

}  // namespace microbench::bulk_store_bench
