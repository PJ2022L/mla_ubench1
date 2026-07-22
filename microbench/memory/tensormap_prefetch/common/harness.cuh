#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "common/bench.hpp"
#include "tensor_map.hpp"

namespace microbench::memory_atomic {
enum class Variant {
    kTensorMapPrefetchRank4,
};
inline JsonObject null_metric(std::string_view unit,
                              std::string_view reason) {
    JsonObject result;
    result.add_null("value")
        .add("unit", unit)
        .add("reason", reason)
        .add_raw("samples", "[]");
    return result;
}

inline JsonObject latency_metric(double cycles_per_round,
                                 std::string_view boundary,
                                 double source_ops_per_round,
                                 const std::vector<double>& samples = {}) {
    JsonObject result;
    result.add("value", cycles_per_round)
        .add("unit", "cycle/round")
        .add("timer", "clock64")
        .add("scope", "cta")
        .add("kind", "full_issue_loop")
        .add("boundary", boundary)
        .add("source_ops_per_round", source_ops_per_round)
        .add_raw("samples", json_number_array(samples));
    return result;
}

inline std::vector<double> rate_samples(double numerator,
                                        const std::vector<double>& event_ms_samples) {
    std::vector<double> samples;
    samples.reserve(event_ms_samples.size());
    for (double sample_ms : event_ms_samples) {
        samples.push_back(numerator / sample_ms / 1.0e6);
    }
    return samples;
}

inline JsonObject throughput_metric(double operations,
                                    double elapsed_ms,
                                    std::string_view operation,
                                    const std::vector<double>& event_ms_samples = {}) {
    const std::vector<double> samples = rate_samples(operations, event_ms_samples);
    const double value = samples.empty()
        ? operations / elapsed_ms / 1.0e6
        : median(samples);
    JsonObject result;
    result.add("value", value)
        .add("unit", "Gsource-op/s")
        .add("timer", "cuda_event")
        .add("scope", "grid")
        .add("operation", operation)
        .add("source_ops", operations)
        .add("event_ms", elapsed_ms)
        .add_raw("samples", json_number_array(samples));
    return result;
}

inline JsonObject bandwidth_metric(double requested_bytes,
                                   double elapsed_ms,
                                   std::string_view space,
                                   const std::vector<double>& event_ms_samples = {}) {
    const std::vector<double> samples = rate_samples(
        requested_bytes, event_ms_samples);
    const double value = samples.empty()
        ? requested_bytes / elapsed_ms / 1.0e6
        : median(samples);
    JsonObject result;
    result.add("value", value)
        .add("unit", "GB/s")
        .add("kind", "requested")
        .add("space", space)
        .add("bytes", requested_bytes)
        .add_raw("samples", json_number_array(samples));
    return result;
}

inline void add_common_params(JsonObject& params,
                              const cudaDeviceProp& properties,
                              const CommonOptions& options,
                              int resolved_blocks) {
    params.add("gpu", properties.name)
        .add("iters", options.iters)
        .add("warmup", options.warmup)
        .add("samples", options.samples)
        .add("blocks", options.blocks)
        .add("resolved_blocks", resolved_blocks)
        .add("device", options.device)
        .add("peak", options.peak);
}

inline void print_memory_result(std::string_view name,
                                const JsonObject& params,
                                double cycles_per_round,
                                std::string_view latency_boundary,
                                double source_ops_per_round,
                                double total_source_ops,
                                double requested_bytes,
                                double elapsed_ms,
                                std::string_view operation,
                                std::string_view memory_space,
                                double peak,
                                bool peak_is_applicable,
                                const std::vector<double>& latency_samples = {},
                                const std::vector<double>& event_ms_samples = {}) {
    const JsonObject latency = latency_metric(
        cycles_per_round, latency_boundary, source_ops_per_round,
        latency_samples);
    const JsonObject throughput = throughput_metric(
        total_source_ops, elapsed_ms, operation, event_ms_samples);
    const JsonObject bandwidth = bandwidth_metric(
        requested_bytes, elapsed_ms, memory_space, event_ms_samples);
    const std::vector<double> bandwidth_samples = rate_samples(
        requested_bytes, event_ms_samples);
    const double measured_bandwidth = bandwidth_samples.empty()
        ? requested_bytes / elapsed_ms / 1.0e6
        : median(bandwidth_samples);
    const JsonObject hardware = peak_is_applicable
        ? utilization(measured_bandwidth, peak, "GB/s")
        : null_metric("ratio", "no validated peak for this instruction domain");
    print_result(name, params, latency, throughput, bandwidth, hardware);
}
__device__ __forceinline__ void prefetch_tensormap(const void* pointer) {
    const uint64_t address = reinterpret_cast<uint64_t>(pointer);
    asm volatile("prefetch.tensormap [%0];"
                 :
                 : "l"(address)
                 : "memory");
}
template <Variant V, bool Target = true>
__global__ void tensormap_prefetch_kernel(
        __grid_constant__ const CUtensorMap descriptor0,
        __grid_constant__ const CUtensorMap descriptor1,
        __grid_constant__ const CUtensorMap descriptor2,
        uint64_t* cycles,
        uint32_t* sinks,
        int iterations,
        int mode) {
    static_assert(V == Variant::kTensorMapPrefetchRank4);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    if (threadIdx.x == 0) {
        const uint64_t start = read_clock64();
#pragma unroll 1
        for (int iteration = 0; iteration < iterations; ++iteration) {
            if (mode == 0 || mode == 3) {
                if constexpr (Target) {
                    prefetch_tensormap(&descriptor0);
                } else {
                    asm volatile("" : : "l"(&descriptor0));
                }
            }
            if (mode == 1 || mode == 3) {
                if constexpr (Target) {
                    prefetch_tensormap(&descriptor1);
                } else {
                    asm volatile("" : : "l"(&descriptor1));
                }
            }
            if (mode == 2 || mode == 3) {
                if constexpr (Target) {
                    prefetch_tensormap(&descriptor2);
                } else {
                    asm volatile("" : : "l"(&descriptor2));
                }
            }
        }
        cycles[blockIdx.x] = read_clock64() - start;
        sinks[blockIdx.x] = static_cast<uint32_t>(iterations + mode + 1);
    }
#else
    if (threadIdx.x == 0) {
        cycles[blockIdx.x] = 0;
        sinks[blockIdx.x] = 0;
    }
#endif
}

struct TimingResult {
    double cycles_per_round;
    double elapsed_ms;
    std::vector<double> latency_samples;
    std::vector<double> target_cycle_samples;
    std::vector<double> baseline_cycle_samples;
    std::vector<double> event_ms_samples;
};

template <typename LatencyLaunch, typename ThroughputLaunch>
inline TimingResult measure_atomic(const CommonOptions& options,
                                   uint64_t* latency_cycles,
                                   uint64_t* latency_baseline_cycles,
                                   LatencyLaunch&& latency_launch,
                                   ThroughputLaunch&& throughput_launch) {
    const auto clock_samples = measure_paired_clock_cycles(
        options.warmup, options.samples, latency_cycles,
        latency_baseline_cycles, 1,
        std::forward<LatencyLaunch>(latency_launch));
    const auto event_samples = measure_event_ms(
        options.warmup, options.samples,
        std::forward<ThroughputLaunch>(throughput_launch));
    std::vector<double> normalized_cycle_samples;
    normalized_cycle_samples.reserve(options.samples);
    for (int index = 0; index < options.samples; ++index) {
        normalized_cycle_samples.push_back(
            (clock_samples.target[index] - clock_samples.baseline[index]) /
            options.iters);
    }
    return TimingResult{median(normalized_cycle_samples),
                        median(event_samples),
                        std::move(normalized_cycle_samples),
                        clock_samples.target,
                        clock_samples.baseline,
                        event_samples};
}

inline void require_nonzero_sinks(const DeviceBuffer<uint32_t>& sinks,
                                  const char* operation) {
    const auto host = sinks.copy_to_host();
    for (uint32_t value : host) {
        if (value == 0) {
            throw std::runtime_error(std::string(operation) +
                                     " produced a zero sink");
        }
    }
}

inline int parse_tensormap_mode(const std::string& value) {
    if (value == "descriptor0") return 0;
    if (value == "descriptor1") return 1;
    if (value == "descriptor2") return 2;
    if (value == "all") return 3;
    throw std::invalid_argument("--mode must be descriptor0, descriptor1, descriptor2, or all");
}

template <Variant V>
inline int run_tensormap_prefetch(int argc, char** argv) {
    static_assert(V == Variant::kTensorMapPrefetchRank4);
    const Args args(argc, argv);
    args.require_only({"iters", "warmup", "samples", "blocks", "device",
                       "peak", "mode", "working-pages", "working-tiles"});
    const CommonOptions options = parse_common_options(args, 4096);
    const cudaDeviceProp properties = require_sm90(options.device);
    const std::string mode_name = args.get_string("mode", "all");
    const int mode = parse_tensormap_mode(mode_name);
    const int working_pages = args.get_int("working-pages", 64, 1, 8192);
    const int working_tiles = args.get_int("working-tiles", 64, 1, 8192);
    const int blocks = resolve_blocks(options.blocks, properties, 4);
    constexpr std::size_t kWideTileElements = 64 * 576;
    constexpr std::size_t kOElements = 64 * 512;

    DeviceBuffer<uint16_t> descriptor0_storage(
        static_cast<std::size_t>(working_pages) * kWideTileElements);
    DeviceBuffer<uint16_t> descriptor1_storage(
        static_cast<std::size_t>(working_pages) * kWideTileElements);
    DeviceBuffer<uint16_t> descriptor2_storage(
        static_cast<std::size_t>(working_tiles) * kOElements);
    const CUtensorMap descriptor0 = make_tma_load_64x576_b16_rank4_map(
        descriptor0_storage.data(), working_pages);
    const CUtensorMap descriptor1 = make_tma_load_64x576_b16_rank4_map(
        descriptor1_storage.data(), working_pages);
    const CUtensorMap descriptor2 = make_tma_store_64x512_b16_rank4_map(
        descriptor2_storage.data(), working_tiles);

    DeviceBuffer<uint64_t> latency_cycles(1);
    DeviceBuffer<uint64_t> latency_baseline_cycles(1);
    DeviceBuffer<uint64_t> throughput_cycles(blocks);
    DeviceBuffer<uint32_t> latency_sinks(1);
    DeviceBuffer<uint32_t> latency_baseline_sinks(1);
    DeviceBuffer<uint32_t> throughput_sinks(blocks);
    const TimingResult timing = measure_atomic(
        options, latency_cycles.data(), latency_baseline_cycles.data(),
        [&] {
            tensormap_prefetch_kernel<V, true><<<1, 32>>>(
                descriptor0, descriptor1, descriptor2, latency_cycles.data(),
                latency_sinks.data(), options.iters, mode);
            tensormap_prefetch_kernel<V, false><<<1, 32>>>(
                descriptor0, descriptor1, descriptor2,
                latency_baseline_cycles.data(), latency_baseline_sinks.data(),
                options.iters, mode);
        },
        [&] {
            tensormap_prefetch_kernel<V, true><<<blocks, 32>>>(
                descriptor0, descriptor1, descriptor2, throughput_cycles.data(),
                throughput_sinks.data(), options.iters, mode);
        });
    require_nonzero_sinks(throughput_sinks, "prefetch.tensormap");

    const int descriptors_per_round = mode == 3 ? 3 : 1;
    const double total_source_ops = static_cast<double>(blocks) * options.iters *
                                    descriptors_per_round;
    JsonObject params;
    add_common_params(params, properties, options, blocks);
    params.add("mode", mode_name)
        .add("working_pages", working_pages)
        .add("working_tiles", working_tiles)
        .add("descriptors_per_round", descriptors_per_round)
        .add("tensor_rank", 4)
        .add("q_shape", "64x576_bf16")
        .add("k_transaction_shape", "64x64_bf16_of_64x576")
        .add("o_shape", "64x512_bf16")
        .add("ptx", "prefetch.tensormap")
        .add("initiation_interval_cycles", timing.cycles_per_round)
        .add("clock_baseline",
             "same descriptor-selection branches and loop; no prefetch")
        .add("correct", true);
    JsonObject latency = latency_metric(
        timing.cycles_per_round,
        "baseline-subtracted selected descriptor prefetch round",
        descriptors_per_round, timing.latency_samples);
    latency.add_raw("target_samples_cycles",
                    json_number_array(timing.target_cycle_samples))
        .add_raw("baseline_samples_cycles",
                 json_number_array(timing.baseline_cycle_samples));
    const JsonObject throughput = throughput_metric(
        total_source_ops, timing.elapsed_ms, "prefetch.tensormap",
        timing.event_ms_samples);
    const JsonObject bandwidth = null_metric(
        "GB/s", "tensor-map prefetch has no architected payload-byte count");
    const JsonObject hardware = null_metric(
        "ratio", "no published tensor-map-prefetch peak");
    print_result("prefetch_tensormap_rank4", params,
                 latency, throughput, bandwidth, hardware);
    return 0;
}

}  // namespace microbench::memory_atomic
