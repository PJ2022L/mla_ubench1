#pragma once

#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/bench.hpp"

namespace microbench::matrix_movement_bench {

enum class Variant { kStmatrixM64N64, kStmatrixM64N256, kLdmatrixM64N64 };

template <Variant V>
struct Traits;

template <>
struct Traits<Variant::kStmatrixM64N64> {
    static constexpr bool kStore = true;
    static constexpr int kTileN = 64;
    static constexpr int kInstructions = 4;
    static constexpr const char* kCase = "stmatrix_m64n64_b16_x4";
    static constexpr const char* kName = "stmatrix_m64n64_b16_x4";
    static constexpr const char* kRole = "64x64 b16 register-to-shared tile";
};

template <>
struct Traits<Variant::kStmatrixM64N256> {
    static constexpr bool kStore = true;
    static constexpr int kTileN = 256;
    static constexpr int kInstructions = 16;
    static constexpr const char* kCase = "stmatrix_m64n256_b16_x4";
    static constexpr const char* kName = "stmatrix_m64n256_b16_x4";
    static constexpr const char* kRole = "64x256 b16 register-to-shared tile";
};

template <>
struct Traits<Variant::kLdmatrixM64N64> {
    static constexpr bool kStore = false;
    static constexpr int kTileN = 64;
    static constexpr int kInstructions = 4;
    static constexpr const char* kCase = "ldmatrix_m64n64_b16_x4";
    static constexpr const char* kName = "ldmatrix_m64n64_b16_x4";
    static constexpr const char* kRole =
        "64x64 b16 shared-to-register tile";
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

__device__ __forceinline__ uint32_t sw128_offset(int instruction) {
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

__device__ __forceinline__ void loop_control_baseline(
        int iterations,
        uint64_t& start,
        uint64_t& stop,
        uint32_t& sink) {
    asm volatile(
        "{\n\t"
        ".reg .pred loop_pred;\n\t"
        ".reg .u32 outer;\n\t"
        "mov.u64 %0, %%clock64;\n\t"
        "mov.u32 outer, %3;\n\t"
        "mb_matrix_control_outer:\n\t"
        "add.u32 outer, outer, -1;\n\t"
        "setp.ne.u32 loop_pred, outer, 0;\n\t"
        "@loop_pred bra.uni mb_matrix_control_outer;\n\t"
        "mov.u64 %1, %%clock64;\n\t"
        "mov.u32 %2, outer;\n\t"
        "}\n"
        : "=l"(start), "=l"(stop), "=r"(sink)
        : "r"(iterations)
        : "memory");
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
void ldmatrix_throughput_kernel(uint64_t* cycles,
                                uint64_t* baseline_cycles,
                                uint32_t* sinks,
                                int iters) {
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
    uint64_t start = 0;
    if (cycles != nullptr) start = read_clock64();
#pragma unroll 1
    for (int iteration = 0; iteration < iters; ++iteration) {
#pragma unroll
        for (int instruction = 0; instruction < Instructions; ++instruction) {
            ldmatrix_x4(base + sw128_offset(instruction),
                        outputs[instruction * 4 + 0],
                        outputs[instruction * 4 + 1],
                        outputs[instruction * 4 + 2],
                        outputs[instruction * 4 + 3]);
        }
    }
    uint64_t stop = 0;
    if (cycles != nullptr) stop = read_clock64();
    uint32_t checksum = 0;
#pragma unroll
    for (int index = 0; index < Instructions * 4; ++index) {
        checksum |= outputs[index];
    }
    if (baseline_cycles != nullptr) {
        uint64_t baseline_start = 0;
        uint64_t baseline_stop = 0;
        uint32_t baseline_sink = 0;
        loop_control_baseline(
            iters, baseline_start, baseline_stop, baseline_sink);
        checksum |= baseline_sink;
        if (threadIdx.x % kWarpgroupThreads == 0) {
            const int index = static_cast<int>(blockIdx.x) * kWarpgroups +
                warpgroup;
            baseline_cycles[index] = baseline_stop - baseline_start;
        }
    }
    if (cycles != nullptr && threadIdx.x % kWarpgroupThreads == 0) {
        const int index =
            static_cast<int>(blockIdx.x) * kWarpgroups + warpgroup;
        cycles[index] = stop - start;
    }
    sinks[static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x] =
        checksum;
}

template <int Threads, int Instructions>
__global__ __launch_bounds__(Threads)
void stmatrix_throughput_kernel(uint64_t* cycles,
                                uint64_t* baseline_cycles,
                                uint32_t* sinks,
                                int iters) {
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
    uint64_t start = 0;
    if (cycles != nullptr) start = read_clock64();
#pragma unroll 1
    for (int iteration = 0; iteration < iters; ++iteration) {
#pragma unroll
        for (int instruction = 0; instruction < Instructions; ++instruction) {
            const uint32_t value = seed + instruction * 4U;
            stmatrix_x4(base + sw128_offset(instruction),
                        value, value + 1U, value + 2U, value + 3U);
        }
    }
    uint64_t stop = 0;
    if (cycles != nullptr) stop = read_clock64();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("bar.sync 0;" ::: "memory");
    uint32_t checksum = 0;
#pragma unroll
    for (int instruction = 0; instruction < Instructions; ++instruction) {
        checksum |= *reinterpret_cast<const uint32_t*>(
            warpgroup_storage + sw128_offset(instruction));
    }
    if (baseline_cycles != nullptr) {
        uint64_t baseline_start = 0;
        uint64_t baseline_stop = 0;
        uint32_t baseline_sink = 0;
        loop_control_baseline(
            iters, baseline_start, baseline_stop, baseline_sink);
        checksum |= baseline_sink;
        if (threadIdx.x % kWarpgroupThreads == 0) {
            const int index = static_cast<int>(blockIdx.x) * kWarpgroups +
                warpgroup;
            baseline_cycles[index] = baseline_stop - baseline_start;
        }
    }
    if (cycles != nullptr && threadIdx.x % kWarpgroupThreads == 0) {
        const int index =
            static_cast<int>(blockIdx.x) * kWarpgroups + warpgroup;
        cycles[index] = stop - start;
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
inline void launch_throughput(uint64_t* cycles,
                              uint64_t* baseline_cycles,
                              uint32_t* sink,
                              int blocks,
                              int iters) {
    if constexpr (Traits<V>::kStore) {
        stmatrix_throughput_kernel<Threads, Traits<V>::kInstructions>
            <<<blocks, Threads>>>(cycles, baseline_cycles, sink, iters);
    } else {
        ldmatrix_throughput_kernel<Threads, Traits<V>::kInstructions>
            <<<blocks, Threads>>>(cycles, baseline_cycles, sink, iters);
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
        DeviceBuffer<uint64_t> dependency_cycles(1);
        DeviceBuffer<uint32_t> latency_sink(threads);
        DeviceBuffer<uint32_t> throughput_sink(
            static_cast<std::size_t>(blocks) * threads);

        double raw_dependency_cycles = 0.0;
        double dependency_cycles_per_instruction = 0.0;
        std::vector<double> dependency_metric_samples;
        if constexpr (!Traits<V>::kStore) {
            dependency_metric_samples = measure_clock_cycles(
                options.warmup, options.samples, dependency_cycles.data(), [&] {
                    if (threads == kWarpgroupThreads) {
                        ldmatrix_latency_kernel<kWarpgroupThreads>
                            <<<1, kWarpgroupThreads>>>(
                                dependency_cycles.data(), latency_sink.data(),
                                options.iters);
                    } else {
                        ldmatrix_latency_kernel<kMaxThreads><<<1, kMaxThreads>>>(
                            dependency_cycles.data(), latency_sink.data(),
                            options.iters);
                    }
                });
            raw_dependency_cycles = median(dependency_metric_samples);
            for (double& value : dependency_metric_samples) {
                value /= options.iters;
            }
            dependency_cycles_per_instruction =
                median(dependency_metric_samples);
            require_uniform_warps(latency_sink.copy_to_host());
        }

        const std::size_t cycle_count =
            static_cast<std::size_t>(blocks) * warpgroups;
        DeviceBuffer<uint64_t> tile_cycles(cycle_count);
        DeviceBuffer<uint64_t> tile_baseline_cycles(cycle_count);
        const auto tile_clock_samples = measure_paired_clock_cycles(
            options.warmup, options.samples, tile_cycles.data(),
            tile_baseline_cycles.data(), cycle_count, [&] {
                if (threads == kWarpgroupThreads) {
                    launch_throughput<V, kWarpgroupThreads>(
                        tile_cycles.data(), tile_baseline_cycles.data(),
                        throughput_sink.data(), blocks, options.iters);
                } else {
                    launch_throughput<V, kMaxThreads>(
                        tile_cycles.data(), tile_baseline_cycles.data(),
                        throughput_sink.data(), blocks, options.iters);
                }
            });
        std::vector<double> tile_interval_samples;
        tile_interval_samples.reserve(options.samples);
        for (int index = 0; index < options.samples; ++index) {
            tile_interval_samples.push_back(
                (tile_clock_samples.target[index] -
                 tile_clock_samples.baseline[index]) / options.iters);
        }
        const double tile_initiation_interval = median(tile_interval_samples);
        std::vector<double> latency_metric_samples;
        latency_metric_samples.reserve(options.samples);
        for (int index = 0; index < options.samples; ++index) {
            if constexpr (Traits<V>::kStore) {
                latency_metric_samples.push_back(tile_interval_samples[index]);
            } else {
                latency_metric_samples.push_back(
                    dependency_metric_samples[index] +
                    (Traits<V>::kInstructions - 1) *
                        tile_interval_samples[index] / Traits<V>::kInstructions);
            }
        }
        const double tile_latency = median(latency_metric_samples);

        const auto event_samples = measure_event_ms(
            options.warmup, options.samples, [&] {
                if (threads == kWarpgroupThreads) {
                    launch_throughput<V, kWarpgroupThreads>(
                        nullptr, nullptr, throughput_sink.data(), blocks,
                        options.iters);
                } else {
                    launch_throughput<V, kMaxThreads>(
                        nullptr, nullptr, throughput_sink.data(), blocks,
                        options.iters);
                }
            });
        const double elapsed_ms = median(event_samples);
        require_nonzero(throughput_sink.copy_to_host(), Traits<V>::kCase);
        const double tiles = static_cast<double>(blocks) * warpgroups *
            options.iters;
        const double instructions = static_cast<double>(blocks) * warps *
            options.iters * Traits<V>::kInstructions;
        const double logical_bytes =
            tiles * 64.0 * Traits<V>::kTileN * 2.0;
        auto throughput_metric_samples = event_samples;
        auto bandwidth_metric_samples = event_samples;
        for (double& value : throughput_metric_samples) {
            value = tiles / value / 1.0e6;
        }
        for (double& value : bandwidth_metric_samples) {
            value = logical_bytes / value / 1.0e6;
        }
        const double throughput_gtiles = median(throughput_metric_samples);
        const double bandwidth_gbs = median(bandwidth_metric_samples);

        JsonObject params;
        params.add("case", Traits<V>::kCase)
            .add("op", Traits<V>::kStore ? "stmatrix" : "ldmatrix")
            .add("tile_m", 64).add("tile_n", Traits<V>::kTileN)
            .add("access_shape", Traits<V>::kRole)
            .add("shape", "m8n8.x4.b16")
            .add("shared_layout", "k_major_sw128")
            .add("instructions_per_warp_iteration", Traits<V>::kInstructions)
            .add("warp_instructions_per_tile",
                 (kWarpgroupThreads / kWarpThreads) * Traits<V>::kInstructions)
            .add("work_unit", "m64_tile")
            .add("initiation_interval_cycles", tile_initiation_interval)
            .add("clock_baseline", "matched inline-PTX add+setp+branch loop")
            .add("warpgroups", warpgroups).add("threads", threads)
            .add("iters", options.iters).add("warmup", options.warmup)
            .add("samples", options.samples).add("blocks", options.blocks)
            .add("resolved_blocks", blocks).add("device", options.device)
            .add("peak", options.peak);
        JsonObject latency;
        if constexpr (Traits<V>::kStore) {
            latency.add("value", tile_latency).add("unit", "cycles/m64_tile")
                .add("timer", "clock64").add("scope", "grid_max_warpgroup")
                .add("boundary", "complete tile issue span; visibility is a separate atom")
                .add_raw("samples", json_number_array(latency_metric_samples));
        } else {
            latency.add("value", tile_latency)
                .add("unit", "cycles/m64_tile").add("timer", "clock64")
                .add("scope", "dependency_completion_plus_tile_issue")
                .add("dependency_cycles_per_instruction",
                     dependency_cycles_per_instruction)
                .add("raw_dependency_median_cycles", raw_dependency_cycles)
                .add_raw("samples", json_number_array(latency_metric_samples));
        }
        latency.add_raw("tile_issue_samples_cycles",
                        json_number_array(tile_interval_samples))
            .add_raw("raw_tile_samples_cycles",
                     json_number_array(tile_clock_samples.target))
            .add_raw("loop_baseline_samples_cycles",
                     json_number_array(tile_clock_samples.baseline));
        JsonObject throughput = metric(throughput_gtiles, "Gtile/s");
        throughput.add("timer", "cuda_event").add("scope", "grid")
            .add("median_ms", elapsed_ms).add("tiles", tiles)
            .add("target_warp_instructions", instructions)
            .add_raw("samples", json_number_array(throughput_metric_samples))
            .add_raw("event_samples_ms", json_number_array(event_samples));
        JsonObject bandwidth = metric(bandwidth_gbs, "GB/s");
        bandwidth.add("kind", "logical_shared").add("bytes", logical_bytes)
            .add("direction", Traits<V>::kStore
                ? "register_to_shared" : "shared_to_register")
            .add_raw("samples", json_number_array(bandwidth_metric_samples));
        print_result(Traits<V>::kName, params, latency, throughput, bandwidth,
                     utilization(throughput_gtiles, options.peak, "Gtile/s"));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << Traits<V>::kCase << ": " << error.what() << '\n';
        return 1;
    }
}

}  // namespace microbench::matrix_movement_bench
