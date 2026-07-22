#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../../../../../../microbench/common/bench.hpp"
#include "../../../../../../microbench/memory/matrix_movement/common/harness.cuh"
#include "../../../../../../microbench/memory/tma_store/common/ptx.cuh"
#include "../../../../../../microbench/memory/tma_store/common/tensor_map.hpp"

namespace microbench::epilogue_stage_bench {

enum class Protocol { kNoSplitB16, kSplitF32 };
enum class Pattern : int { kLocal, kSequential, kRandom };

constexpr int kThreads = 256;
constexpr int kRows = 64;
constexpr int kColumns = 512;
constexpr int kNoSplitSharedBytes = kRows * kColumns * 2;
constexpr int kSplitStride = 520;
constexpr int kSplitSharedBytes = kRows * kSplitStride * 4;
constexpr int kNoSplitTransactionBytes = kRows * 64 * 2;
constexpr int kSplitRowBytes = kColumns * 4;

inline Pattern parse_pattern(const std::string& value) {
    if (value == "local") return Pattern::kLocal;
    if (value == "sequential") return Pattern::kSequential;
    if (value == "random") return Pattern::kRandom;
    throw std::invalid_argument("--pattern must be local, sequential, or random");
}

__device__ __forceinline__ int select_tile(Pattern pattern,
                                            int iteration,
                                            int working_tiles,
                                            int block,
                                            int grid_blocks) {
    const int tiles_per_block = working_tiles / grid_blocks;
    const int base = block * tiles_per_block;
    if (pattern == Pattern::kLocal) return base;
    if (pattern == Pattern::kRandom) {
        uint32_t value = static_cast<uint32_t>(iteration + 1) * 0x85ebca6bu;
        value ^= value >> 16;
        value *= 0x7feb352du;
        value ^= value >> 15;
        return base + static_cast<int>(value % tiles_per_block);
    }
    return base + iteration % tiles_per_block;
}

__device__ __forceinline__ void store_float2(void* pointer,
                                              uint32_t lo,
                                              uint32_t hi) {
    const uint32_t address = shared_address(pointer);
    asm volatile("st.shared.v2.u32 [%0], {%1, %2};"
                 :: "r"(address), "r"(lo), "r"(hi) : "memory");
}

template <Protocol P>
__global__ __launch_bounds__(kThreads, 1)
void epilogue_kernel(__grid_constant__ const CUtensorMap tensor_map,
                     void* output,
                     uint64_t* cycles,
                     uint32_t* sinks,
                     uint32_t* smids,
                     int iterations,
                     int working_tiles,
                     Pattern pattern) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    extern __shared__ __align__(1024) unsigned char storage[];
    const uint32_t seed = 0x3f803f80U ^
        static_cast<uint32_t>(threadIdx.x + blockIdx.x * 257);
    int final_tile = 0;
    asm volatile("bar.sync 0;" ::: "memory");
    const uint64_t start = read_clock64();
#pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        final_tile = select_tile(
            pattern, iteration, working_tiles,
            static_cast<int>(blockIdx.x), static_cast<int>(gridDim.x));
        if constexpr (P == Protocol::kNoSplitB16) {
            const int warpgroup = static_cast<int>(threadIdx.x) / 128;
            const uint32_t base = shared_address(
                storage + warpgroup * (kNoSplitSharedBytes / 2));
#pragma unroll
            for (int instruction = 0; instruction < 16; ++instruction) {
                const uint32_t value = seed + instruction * 4U;
                matrix_movement_bench::stmatrix_x4(
                    base + matrix_movement_bench::sw128_offset(instruction),
                    value, value + 1U, value + 2U, value + 3U);
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("bar.sync 0;" ::: "memory");
            if (threadIdx.x == 0) {
#pragma unroll
                for (int transaction = 0; transaction < 8; ++transaction) {
                    ptx::tma_store_4d(
                        &tensor_map,
                        storage + transaction * kNoSplitTransactionBytes,
                        transaction * 64, 0, final_tile, 0);
                }
                ptx::bulk_commit_group();
                ptx::bulk_wait_group<0>();
            }
            asm volatile("bar.sync 0;" ::: "memory");
        } else {
#pragma unroll 1
            for (int store = 0; store < 64; ++store) {
                const int pair = static_cast<int>(threadIdx.x) + store * kThreads;
                const int row = pair / 256;
                const int column_pair = pair % 256;
                store_float2(
                    storage + (row * kSplitStride + column_pair * 2) * 4,
                    seed + store * 2U, seed + store * 2U + 1U);
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("bar.sync 0;" ::: "memory");
            if (threadIdx.x < kRows) {
                const int row = static_cast<int>(threadIdx.x);
                auto* destination = static_cast<float*>(output) +
                    (static_cast<int64_t>(final_tile) * kRows + row) * kColumns;
                ptx::bulk_store_shared_to_global(
                    destination, storage + row * kSplitStride * 4,
                    kSplitRowBytes);
                ptx::bulk_commit_group();
                ptx::bulk_wait_group<0>();
            }
            asm volatile("bar.sync 0;" ::: "memory");
        }
    }
    const uint64_t stop = read_clock64();
    if (threadIdx.x == 0) {
        cycles[blockIdx.x] = stop - start;
        sinks[blockIdx.x] = static_cast<uint32_t>(final_tile + 1);
        smids[blockIdx.x] = read_smid();
    }
#else
    if (threadIdx.x == 0) {
        cycles[blockIdx.x] = 0;
        sinks[blockIdx.x] = 0;
        smids[blockIdx.x] = 0;
    }
#endif
}

template <Protocol P>
void validate_output(const CUtensorMap& tensor_map,
                     void* output,
                     int shared_bytes) {
    DeviceBuffer<uint64_t> cycles(1);
    DeviceBuffer<uint32_t> sinks(1);
    DeviceBuffer<uint32_t> smids(1);
    epilogue_kernel<P><<<1, kThreads, shared_bytes>>>(
        tensor_map, output, cycles.data(), sinks.data(), smids.data(),
        1, 1, Pattern::kLocal);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if constexpr (P == Protocol::kNoSplitB16) {
        std::vector<uint16_t> tile(
            static_cast<std::size_t>(kRows) * kColumns);
        CUDA_CHECK(cudaMemcpy(
            tile.data(), output, tile.size() * sizeof(uint16_t),
            cudaMemcpyDeviceToHost));
        if (std::any_of(tile.begin(), tile.end(),
                        [](uint16_t value) { return value == 0; })) {
            throw std::runtime_error(
                "no-split epilogue left zero values in the validated output tile");
        }
    } else {
        std::vector<uint32_t> tile_bits(
            static_cast<std::size_t>(kRows) * kColumns);
        CUDA_CHECK(cudaMemcpy(
            tile_bits.data(), output, tile_bits.size() * sizeof(uint32_t),
            cudaMemcpyDeviceToHost));
        if (std::any_of(tile_bits.begin(), tile_bits.end(),
                        [](uint32_t value) { return value == 0; })) {
            throw std::runtime_error(
                "split epilogue left zero values in the validated output tile");
        }
    }
}

template <Protocol P>
int run(int argc, char** argv) {
    try {
        const Args args(argc, argv);
        args.require_only({"iters", "warmup", "samples", "blocks", "device",
                           "peak", "working-set-tiles", "pattern"});
        const auto options = parse_common_options(args, 16);
        const auto properties = require_sm90(options.device);
        const int blocks = resolve_blocks(options.blocks, properties, 1);
        const int working_tiles = args.get_int(
            "working-set-tiles", std::max(512, blocks), blocks, 1 << 19);
        const std::string pattern_name = args.get_string("pattern", "sequential");
        const Pattern pattern = parse_pattern(pattern_name);
        constexpr int kSharedBytes = P == Protocol::kNoSplitB16
            ? kNoSplitSharedBytes : kSplitSharedBytes;
        if (kSharedBytes > static_cast<int>(properties.sharedMemPerBlockOptin)) {
            throw std::runtime_error("epilogue shared tile exceeds opt-in limit");
        }
        CUDA_CHECK(cudaFuncSetAttribute(
            epilogue_kernel<P>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            kSharedBytes));
        DeviceBuffer<uint16_t> output_b16;
        DeviceBuffer<float> output_f32;
        CUtensorMap tensor_map{};
        void* output = nullptr;
        if constexpr (P == Protocol::kNoSplitB16) {
            output_b16.resize(
                static_cast<std::size_t>(working_tiles) * kRows * kColumns);
            output_b16.zero();
            output = output_b16.data();
            tensor_map = make_tma_store_64x512_b16_rank4_map(
                output, working_tiles);
        } else {
            output_f32.resize(
                static_cast<std::size_t>(working_tiles) * kRows * kColumns);
            output_f32.zero();
            output = output_f32.data();
        }
        validate_output<P>(tensor_map, output, kSharedBytes);
        DeviceBuffer<uint64_t> latency_cycles(1);
        DeviceBuffer<uint32_t> latency_sinks(1);
        DeviceBuffer<uint32_t> latency_smids(1);
        const auto raw_cycles = measure_clock_cycles(
            options.warmup, options.samples, latency_cycles.data(), [&] {
                epilogue_kernel<P><<<1, kThreads, kSharedBytes>>>(
                    tensor_map, output, latency_cycles.data(),
                    latency_sinks.data(), latency_smids.data(),
                    options.iters, working_tiles, pattern);
            });
        auto latency_samples = raw_cycles;
        for (double& value : latency_samples) value /= options.iters;
        DeviceBuffer<uint64_t> throughput_cycles(blocks);
        DeviceBuffer<uint32_t> throughput_sinks(blocks);
        DeviceBuffer<uint32_t> throughput_smids(blocks);
        const auto event_samples = measure_event_ms(
            options.warmup, options.samples, [&] {
                epilogue_kernel<P><<<blocks, kThreads, kSharedBytes>>>(
                    tensor_map, output, throughput_cycles.data(),
                    throughput_sinks.data(), throughput_smids.data(),
                    options.iters, working_tiles, pattern);
            });
        for (uint32_t value : throughput_sinks.copy_to_host()) {
            if (value == 0) throw std::runtime_error("epilogue sink is zero");
        }
        const double tiles = static_cast<double>(blocks) * options.iters;
        constexpr double kBytesPerTile = P == Protocol::kNoSplitB16
            ? static_cast<double>(kNoSplitSharedBytes)
            : static_cast<double>(kRows * kColumns * 4);
        auto throughput_samples = event_samples;
        auto bandwidth_samples = event_samples;
        for (double& value : throughput_samples) value = tiles / value / 1.0e6;
        for (double& value : bandwidth_samples) {
            value = tiles * kBytesPerTile / value / 1.0e6;
        }
        JsonObject params;
        params.add("gpu", properties.name)
            .add("protocol", P == Protocol::kNoSplitB16
                ? "STSM+proxy-fence+rank4-TMA-store"
                : "STS.64-stride520+proxy-fence+bulk-S2G")
            .add("working_set_tiles", working_tiles)
            .add("pattern", pattern_name).add("threads", kThreads)
            .add("shared_bytes", kSharedBytes)
            .add("iters", options.iters).add("warmup", options.warmup)
            .add("samples", options.samples).add("blocks", options.blocks)
            .add("resolved_blocks", blocks).add("device", options.device)
            .add("peak", options.peak).add("correct", true);
        const auto observed_smids = throughput_smids.copy_to_host();
        JsonObject latency;
        latency.add("value", median(latency_samples)).add("unit", "cycle/tile")
            .add("scope", "ordered store protocol; excludes normalization and LSE")
            .add_raw("samples", json_number_array(latency_samples))
            .add_raw("observed_smids", json_number_array(observed_smids));
        JsonObject throughput;
        throughput.add("value", median(throughput_samples)).add("unit", "Gtile/s")
            .add_raw("samples", json_number_array(throughput_samples))
            .add_raw("event_samples_ms", json_number_array(event_samples));
        JsonObject bandwidth;
        bandwidth.add("value", median(bandwidth_samples)).add("unit", "GB/s")
            .add("bytes_per_tile", kBytesPerTile)
            .add_raw("samples", json_number_array(bandwidth_samples));
        print_result(
            P == Protocol::kNoSplitB16
                ? "nosplit_store_protocol_b16"
                : "split_store_protocol_f32",
            params, latency, throughput, bandwidth,
            utilization(median(bandwidth_samples), options.peak, "GB/s"));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "epilogue stage: " << error.what() << '\n';
        return 1;
    }
}

}  // namespace microbench::epilogue_stage_bench
