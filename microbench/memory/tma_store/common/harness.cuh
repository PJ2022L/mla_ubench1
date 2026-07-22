#pragma once

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "common/bench.hpp"
#include "ptx.cuh"
#include "tensor_map.hpp"

namespace microbench::tma_store_bench {
namespace {

constexpr int kRows = 64;
constexpr int kColumns = 512;
constexpr int kTransactionColumns = 64;
constexpr int kTransactionsPerTile = kColumns / kTransactionColumns;
constexpr int kTransactionBytes = kRows * kTransactionColumns * 2;
constexpr int kTransactionElements = kRows * kTransactionColumns;
constexpr int kTileElements = kRows * kColumns;
constexpr int kTileBytes = kTileElements * 2;
constexpr int kThreads = 256;
constexpr int kMaxDepth = 3;
constexpr int kMaxWorkingTiles = 1 << 20;
constexpr uint16_t kPatternBase = 0x3a80u;

__host__ __device__ constexpr uint16_t transaction_pattern(int transaction) {
    return static_cast<uint16_t>(kPatternBase + transaction);
}

__device__ __forceinline__ int select_tile(int iteration,
                                           int working_tiles,
                                           int block,
                                           int grid_blocks) {
    const int tiles_per_block = working_tiles / grid_blocks;
    return block * tiles_per_block + iteration % tiles_per_block;
}

template <bool Target = true>
__global__ void tma_store_kernel(
        __grid_constant__ const CUtensorMap tensor_map,
        uint64_t* cycles,
        uint32_t* sinks,
        int iterations,
        int depth,
        int working_tiles) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    extern __shared__ __align__(128) unsigned char shared_storage[];
    auto* values = reinterpret_cast<uint16_t*>(shared_storage);
    const int shared_elements = depth * kTileElements;
    for (int index = threadIdx.x; index < shared_elements;
         index += blockDim.x) {
        const int index_in_stage = index % kTileElements;
        values[index] = transaction_pattern(
            index_in_stage / kTransactionElements);
    }
    __syncthreads();
    microbench::ptx::async_shared_fence();
    __syncthreads();

    if (threadIdx.x == 0) {
        int final_tile = 0;
        const uint64_t start = microbench::read_clock64();
        for (int base = 0; base < iterations; base += depth) {
            const int active = min(depth, iterations - base);
            for (int stage = 0; stage < active; ++stage) {
                const int iteration = base + stage;
                final_tile = select_tile(
                    iteration, working_tiles,
                    static_cast<int>(blockIdx.x), static_cast<int>(gridDim.x));
                if constexpr (Target) {
                    const unsigned char* stage_source =
                        shared_storage + stage * kTileBytes;
#pragma unroll
                    for (int transaction = 0;
                         transaction < kTransactionsPerTile;
                         ++transaction) {
                        microbench::ptx::tma_store_4d(
                            &tensor_map,
                            stage_source + transaction * kTransactionBytes,
                            transaction * kTransactionColumns,
                            0,
                            final_tile,
                            0);
                    }
                    microbench::ptx::bulk_commit_group();
                } else {
                    asm volatile("" : : "r"(final_tile));
                }
            }
            if constexpr (Target) {
                microbench::ptx::bulk_wait_group<0>();
            }
        }
        const uint64_t stop = microbench::read_clock64();
        cycles[blockIdx.x] = stop - start;
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
            "depth", "working-set-tiles"});
        const auto options = microbench::parse_common_options(args, 128);
        const auto properties = microbench::require_sm90(options.device);
        const int depth = args.get_int("depth", 1, 1, kMaxDepth);
        const int blocks = microbench::resolve_blocks(options.blocks, properties, 1);
        const int64_t minimum_working_tiles_64 =
            static_cast<int64_t>(blocks) * std::min(depth, options.iters);
        if (minimum_working_tiles_64 > kMaxWorkingTiles) {
            throw std::invalid_argument(
                "blocks * min(depth, iters) exceeds the supported TMA-store "
                "working set; reduce --blocks or --depth");
        }
        const int minimum_working_tiles =
            static_cast<int>(minimum_working_tiles_64);
        const int working_tiles = args.get_int(
            "working-set-tiles", std::max(4096, minimum_working_tiles),
            minimum_working_tiles, kMaxWorkingTiles);
        const int tiles_per_block = working_tiles / blocks;
        if (tiles_per_block < std::min(depth, options.iters)) {
            throw std::invalid_argument(
                "working-set-tiles must provide each CTA a private interval "
                "at least as large as min(depth, iters)");
        }
        const int shared_bytes = depth * kTileBytes;
        if (shared_bytes > static_cast<int>(properties.sharedMemPerBlockOptin)) {
            throw std::invalid_argument(
                "requested TMA-store depth exceeds sharedMemPerBlockOptin; "
                "64x512 b16 consumes 65536 bytes per stage");
        }

        const std::size_t output_elements =
            static_cast<std::size_t>(working_tiles) * kTileElements;
        microbench::DeviceBuffer<uint16_t> output(output_elements);
        output.zero();
        const CUtensorMap tensor_map =
            microbench::make_tma_store_64x512_b16_rank4_map(
                output.data(), working_tiles);
        CUDA_CHECK(cudaFuncSetAttribute(
            tma_store_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_bytes));
        CUDA_CHECK(cudaFuncSetAttribute(
            tma_store_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_bytes));

        microbench::DeviceBuffer<uint64_t> latency_cycles(1);
        microbench::DeviceBuffer<uint64_t> latency_baseline_cycles(1);
        microbench::DeviceBuffer<uint32_t> latency_sinks(1);
        microbench::DeviceBuffer<uint32_t> latency_baseline_sinks(1);
        const auto latency_clock_samples =
            microbench::measure_paired_clock_cycles(
            options.warmup, options.samples, latency_cycles.data(),
            latency_baseline_cycles.data(), 1,
            [&] {
                tma_store_kernel<true><<<1, kThreads, kTileBytes>>>(
                    tensor_map, latency_cycles.data(), latency_sinks.data(),
                    options.iters, 1, working_tiles);
                tma_store_kernel<false><<<1, kThreads, kTileBytes>>>(
                    tensor_map, latency_baseline_cycles.data(),
                    latency_baseline_sinks.data(), options.iters, 1,
                    working_tiles);
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
                    tma_store_kernel<true><<<blocks, kThreads, shared_bytes>>>(
                        tensor_map, throughput_cycles.data(),
                        throughput_sinks.data(), options.iters, depth,
                        working_tiles);
                    tma_store_kernel<false><<<blocks, kThreads, shared_bytes>>>(
                        tensor_map, throughput_baseline_cycles.data(),
                        throughput_baseline_sinks.data(), options.iters, depth,
                        working_tiles);
                });
        const auto event_samples = microbench::measure_event_ms(
            options.warmup,
            options.samples,
            [&] {
                tma_store_kernel<true><<<blocks, kThreads, shared_bytes>>>(
                    tensor_map, throughput_cycles.data(), throughput_sinks.data(),
                    options.iters, depth, working_tiles);
            });
        const double elapsed_ms = microbench::median(event_samples);
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
        const auto host_sinks = throughput_sinks.copy_to_host();
        for (uint32_t value : host_sinks) {
            if (value == 0) {
                throw std::runtime_error("TMA store completion sink is zero");
            }
        }

        output.zero();
        tma_store_kernel<true><<<1, kThreads, kTileBytes>>>(
            tensor_map, latency_cycles.data(), latency_sinks.data(), 1, 1,
            working_tiles);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<uint16_t> validation_tile(kTileElements);
        CUDA_CHECK(cudaMemcpy(
            validation_tile.data(), output.data(), kTileBytes,
            cudaMemcpyDeviceToHost));
        for (int row = 0; row < kRows; ++row) {
            for (int column = 0; column < kColumns; ++column) {
                const uint16_t expected = transaction_pattern(
                    column / kTransactionColumns);
                if (validation_tile[row * kColumns + column] != expected) {
                    throw std::runtime_error(
                        "rank-4 TMA store transaction-mapped validation failed");
                }
            }
        }

        const double tiles = static_cast<double>(blocks) * options.iters;
        const double transactions = tiles * kTransactionsPerTile;
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
            .add("dtype", microbench::kB16DtypeName)
            .add("tensor_rank", 4)
            .add("transactions_per_tile", kTransactionsPerTile)
            .add("transaction_shape", "64x64")
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
            .add("depth", depth)
            .add("initiation_interval_cycles", initiation_interval_cycles)
            .add("initiation_interval_boundary",
                 "target protocol minus matched tile-selection loop baseline")
            .add("throughput_depth", depth)
            .add("latency_depth", 1)
            .add("clock_baseline",
                 "same depth loop, tile selection, and sink; no TMA store, "
                 "commit_group, or wait_group")
            .add("correct", true)
            .add("correctness",
                 "full 64x512 tile with distinct pattern per transaction")
            .add("address_mapping",
                 "per-CTA disjoint interval; collision-free at arbitrary progress");

        microbench::JsonObject latency;
        latency.add("value", completion_cycles)
            .add("unit", "cycle/tile")
            .add("timer", "clock64")
            .add("scope", "cta")
            .add("boundary",
                 "baseline-subtracted 8x issue + 1x commit + wait0")
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
            .add("transactions", transactions)
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

        const std::string result_name =
            std::string("tensor_4d_64x512_") + microbench::kB16DtypeName;
        microbench::print_result(
            result_name,
            params,
            latency,
            throughput,
            bandwidth,
            microbench::utilization(
                bandwidth_gbps, options.peak, "GB/s"));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tma_store: " << error.what() << '\n';
        return 1;
    }
}

}  // namespace microbench::tma_store_bench
