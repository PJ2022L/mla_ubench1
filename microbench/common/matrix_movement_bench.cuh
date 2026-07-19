#pragma once

#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "bench.hpp"

namespace microbench::matrix_movement_bench {

enum class Variant { kStmatrixP, kStmatrixO, kLdmatrixP };

template <Variant V>
struct Traits;

template <>
struct Traits<Variant::kStmatrixP> {
    static constexpr bool kStore = true;
    static constexpr int kTileN = 64;
    static constexpr int kInstructions = 4;
    static constexpr const char* kCase = "stmatrix_p";
    static constexpr const char* kName = "dense_decode.stmatrix_p_b16";
    static constexpr const char* kRole = "P register-to-shared exchange";
};

template <>
struct Traits<Variant::kStmatrixO> {
    static constexpr bool kStore = true;
    static constexpr int kTileN = 256;
    static constexpr int kInstructions = 16;
    static constexpr const char* kCase = "stmatrix_o";
    static constexpr const char* kName = "dense_decode.stmatrix_o_b16";
    static constexpr const char* kRole = "O accumulator shared staging";
};

template <>
struct Traits<Variant::kLdmatrixP> {
    static constexpr bool kStore = false;
    static constexpr int kTileN = 64;
    static constexpr int kInstructions = 4;
    static constexpr const char* kCase = "ldmatrix_p";
    static constexpr const char* kName = "dense_decode.ldmatrix_p_b16";
    static constexpr const char* kRole =
        "P shared-to-register exchange and Q tile-8 load";
};

constexpr int kWarpThreads = 32;
constexpr int kWarpgroupThreads = 128;
constexpr int kMaxThreads = 256;
constexpr int kWordsPerWarpTile = 128;
constexpr int kBytesPerInstruction = 512;
constexpr int kSwizzleMask = 896;
constexpr int kBytesPerN64Group = 8192;

__device__ __forceinline__ void stmatrix_x4(uint32_t address,
                                            uint32_t x0,
                                            uint32_t x1,
                                            uint32_t x2,
                                            uint32_t x3) {
    asm volatile(
        "stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], "
        "{%1, %2, %3, %4};"
        :
        : "r"(address), "r"(x0), "r"(x1), "r"(x2), "r"(x3)
        : "memory");
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t address,
                                            uint32_t& x0,
                                            uint32_t& x1,
                                            uint32_t& x2,
                                            uint32_t& x3) {
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3)
        : "r"(address)
        : "memory");
}

__device__ __forceinline__ uint32_t dense_sw128_offset(int instruction) {
    const uint32_t lane = static_cast<uint32_t>(
        threadIdx.x % kWarpgroupThreads);
    const uint32_t lane_offset =
        (((lane << 5) & ~1023U) |
         ((lane << 6) & 960U) |
         ((lane >> 1) & 8U)) * 2U;
    const uint32_t logical = lane_offset +
        static_cast<uint32_t>(instruction / 4) * kBytesPerN64Group +
        static_cast<uint32_t>(instruction % 4) * 32U;
    return logical ^ ((logical & kSwizzleMask) >> 3);
}

template <int Threads>
__global__ __launch_bounds__(Threads)
void ldmatrix_latency_kernel(uint64_t* cycles, uint32_t* sinks, int iters) {
    __align__(128) __shared__ uint32_t
        storage[(Threads / kWarpThreads) * kWordsPerWarpTile];
    const int warp = threadIdx.x / kWarpThreads;
    const int lane = threadIdx.x % kWarpThreads;
    uint32_t* tile = storage + warp * kWordsPerWarpTile;
    const uint32_t tile_address = shared_address(tile);
    const uint32_t chain_address = tile_address + 16U;
    for (int index = lane; index < kWordsPerWarpTile; index += kWarpThreads) {
        tile[index] = chain_address;
    }
    asm volatile("bar.sync 0;" ::: "memory");
    uint32_t address = tile_address + static_cast<uint32_t>(lane) * 16U;
    uint32_t x0 = 0, x1 = 0, x2 = 0, x3 = 0;
    const uint64_t start = read_clock64();
#pragma unroll 1
    for (int iteration = 0; iteration < iters; ++iteration) {
        ldmatrix_x4(address, x0, x1, x2, x3);
        address = x0;
    }
    const uint64_t stop = read_clock64();
    sinks[threadIdx.x] = address;
    if (threadIdx.x == 0) cycles[0] = stop - start;
}

template <int Threads, int Instructions>
__global__ __launch_bounds__(Threads)
void ldmatrix_throughput_kernel(uint32_t* sinks, int iters) {
    constexpr int kWarpgroups = Threads / kWarpgroupThreads;
    constexpr int kBytesPerWarpgroup =
        Instructions * kBytesPerInstruction *
        (kWarpgroupThreads / kWarpThreads);
    __align__(1024) __shared__ uint8_t storage[kWarpgroups * kBytesPerWarpgroup];
    auto* words = reinterpret_cast<uint32_t*>(storage);
    for (int index = threadIdx.x;
         index < static_cast<int>(sizeof(storage) / sizeof(uint32_t));
         index += Threads) {
        words[index] = 0x3f803f80U ^ static_cast<uint32_t>(index);
    }
    asm volatile("bar.sync 0;" ::: "memory");
    const int warpgroup = threadIdx.x / kWarpgroupThreads;
    const uint32_t base = shared_address(
        storage + warpgroup * kBytesPerWarpgroup);
    uint32_t outputs[Instructions * 4] = {};
#pragma unroll 1
    for (int iteration = 0; iteration < iters; ++iteration) {
#pragma unroll
        for (int instruction = 0; instruction < Instructions; ++instruction) {
            ldmatrix_x4(base + dense_sw128_offset(instruction),
                        outputs[instruction * 4 + 0],
                        outputs[instruction * 4 + 1],
                        outputs[instruction * 4 + 2],
                        outputs[instruction * 4 + 3]);
        }
    }
    uint32_t checksum = 0;
#pragma unroll
    for (int index = 0; index < Instructions * 4; ++index) {
        checksum |= outputs[index];
    }
    sinks[static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x] =
        checksum;
}

template <int Threads, int Instructions>
__global__ __launch_bounds__(Threads)
void stmatrix_throughput_kernel(uint32_t* sinks, int iters) {
    constexpr int kWarpgroups = Threads / kWarpgroupThreads;
    constexpr int kBytesPerWarpgroup =
        Instructions * kBytesPerInstruction *
        (kWarpgroupThreads / kWarpThreads);
    __align__(1024) __shared__ uint8_t storage[kWarpgroups * kBytesPerWarpgroup];
    const int warpgroup = threadIdx.x / kWarpgroupThreads;
    uint8_t* warpgroup_storage = storage + warpgroup * kBytesPerWarpgroup;
    const uint32_t base = shared_address(warpgroup_storage);
    const uint32_t seed = 0x3f803f80U ^
        (static_cast<uint32_t>(blockIdx.x) << 8) ^ threadIdx.x;
#pragma unroll 1
    for (int iteration = 0; iteration < iters; ++iteration) {
#pragma unroll
        for (int instruction = 0; instruction < Instructions; ++instruction) {
            const uint32_t value = seed + instruction * 4U;
            stmatrix_x4(base + dense_sw128_offset(instruction),
                        value, value + 1U, value + 2U, value + 3U);
        }
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("bar.sync 0;" ::: "memory");
    uint32_t checksum = 0;
#pragma unroll
    for (int instruction = 0; instruction < Instructions; ++instruction) {
        checksum |= *reinterpret_cast<const uint32_t*>(
            warpgroup_storage + dense_sw128_offset(instruction));
    }
    sinks[static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x] =
        checksum;
}

inline void require_nonzero(const std::vector<uint32_t>& values,
                            const char* label) {
    for (const uint32_t value : values) {
        if (value == 0) {
            throw std::runtime_error(std::string(label) + " sink is zero");
        }
    }
}

inline void require_uniform_warps(const std::vector<uint32_t>& values) {
    for (std::size_t base = 0; base < values.size(); base += kWarpThreads) {
        if (values[base] == 0) {
            throw std::runtime_error("LDMATRIX latency sink is zero");
        }
        for (int lane = 1; lane < kWarpThreads; ++lane) {
            if (values[base + lane] != values[base]) {
                throw std::runtime_error("LDMATRIX dependency chain diverged");
            }
        }
    }
}

template <Variant V, int Threads>
inline void launch_throughput(uint32_t* sink, int blocks, int iters) {
    if constexpr (Traits<V>::kStore) {
        stmatrix_throughput_kernel<Threads, Traits<V>::kInstructions>
            <<<blocks, Threads>>>(sink, iters);
    } else {
        ldmatrix_throughput_kernel<Threads, Traits<V>::kInstructions>
            <<<blocks, Threads>>>(sink, iters);
    }
}

template <Variant V>
int run(int argc, char** argv) {
    try {
        const Args args(argc, argv);
        args.require_only({"iters", "warmup", "samples", "blocks", "peak",
                           "device", "warpgroups"});
        const auto options = parse_common_options(args);
        const int warpgroups = args.get_int("warpgroups", 2, 1, 2);
        const int threads = warpgroups * kWarpgroupThreads;
        const int warps = threads / kWarpThreads;
        const auto device = require_sm90(options.device);
        const int blocks = resolve_blocks(options.blocks, device);
        DeviceBuffer<uint64_t> cycles(1);
        DeviceBuffer<uint32_t> latency_sink(threads);
        DeviceBuffer<uint32_t> throughput_sink(
            static_cast<std::size_t>(blocks) * threads);

        double raw_cycles = 0.0;
        double cycles_per_instruction = 0.0;
        std::vector<double> latency_metric_samples;
        if constexpr (!Traits<V>::kStore) {
            latency_metric_samples = measure_clock_cycles(
                options.warmup, options.samples, cycles.data(), [&] {
                    if (threads == kWarpgroupThreads) {
                        ldmatrix_latency_kernel<kWarpgroupThreads>
                            <<<1, kWarpgroupThreads>>>(
                                cycles.data(), latency_sink.data(), options.iters);
                    } else {
                        ldmatrix_latency_kernel<kMaxThreads><<<1, kMaxThreads>>>(
                            cycles.data(), latency_sink.data(), options.iters);
                    }
                });
            raw_cycles = median(latency_metric_samples);
            for (double& value : latency_metric_samples) value /= options.iters;
            cycles_per_instruction = median(latency_metric_samples);
            require_uniform_warps(latency_sink.copy_to_host());
        }

        const auto event_samples = measure_event_ms(
            options.warmup, options.samples, [&] {
                if (threads == kWarpgroupThreads) {
                    launch_throughput<V, kWarpgroupThreads>(
                        throughput_sink.data(), blocks, options.iters);
                } else {
                    launch_throughput<V, kMaxThreads>(
                        throughput_sink.data(), blocks, options.iters);
                }
            });
        const double elapsed_ms = median(event_samples);
        require_nonzero(throughput_sink.copy_to_host(), Traits<V>::kCase);
        const double instructions = static_cast<double>(blocks) * warps *
            options.iters * Traits<V>::kInstructions;
        const double logical_bytes = instructions * kBytesPerInstruction;
        auto throughput_metric_samples = event_samples;
        auto bandwidth_metric_samples = event_samples;
        for (double& value : throughput_metric_samples) {
            value = instructions / value / 1.0e6;
        }
        for (double& value : bandwidth_metric_samples) {
            value = logical_bytes / value / 1.0e6;
        }
        const double throughput_ginst = median(throughput_metric_samples);
        const double bandwidth_gbs = median(bandwidth_metric_samples);

        JsonObject params;
        params.add("case", Traits<V>::kCase)
            .add("op", Traits<V>::kStore ? "stmatrix" : "ldmatrix")
            .add("tile_m", 64).add("tile_n", Traits<V>::kTileN)
            .add("dense_role", Traits<V>::kRole)
            .add("shape", "m8n8.x4.b16")
            .add("shared_layout", "dense_k_major_sw128")
            .add("instructions_per_warp_iteration", Traits<V>::kInstructions)
            .add("warpgroups", warpgroups).add("threads", threads)
            .add("iters", options.iters).add("warmup", options.warmup)
            .add("samples", options.samples).add("blocks", options.blocks)
            .add("resolved_blocks", blocks).add("device", options.device)
            .add("peak", options.peak);
        JsonObject latency;
        if constexpr (Traits<V>::kStore) {
            latency.add_null("value").add("unit", "cycles/instruction")
                .add_null("timer").add_null("scope")
                .add("reason", "store has no natural register dependency");
        } else {
            latency.add("value", cycles_per_instruction)
                .add("unit", "cycles/instruction").add("timer", "clock64")
                .add("scope", "single_cta")
                .add("raw_median_cycles", raw_cycles)
                .add_raw("samples", json_number_array(latency_metric_samples));
        }
        JsonObject throughput = metric(throughput_ginst, "Ginst/s");
        throughput.add("timer", "cuda_event").add("scope", "grid")
            .add("median_ms", elapsed_ms).add("instructions", instructions)
            .add_raw("samples", json_number_array(throughput_metric_samples))
            .add_raw("event_samples_ms", json_number_array(event_samples));
        JsonObject bandwidth = metric(bandwidth_gbs, "GB/s");
        bandwidth.add("kind", "logical_shared").add("bytes", logical_bytes)
            .add("direction", Traits<V>::kStore
                ? "register_to_shared" : "shared_to_register")
            .add_raw("samples", json_number_array(bandwidth_metric_samples));
        print_result(Traits<V>::kName, params, latency, throughput, bandwidth,
                     utilization(throughput_ginst, options.peak, "Ginst/s"));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << Traits<V>::kCase << ": " << error.what() << '\n';
        return 1;
    }
}

}  // namespace microbench::matrix_movement_bench
