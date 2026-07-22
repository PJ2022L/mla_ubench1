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
namespace microbench::memory_atomic {
enum class Variant {
    kSharedLoadU32Patterns,
};
enum class SharedLoadPattern : int {
    kUnique = 0,
    kQuadBroadcast = 1,
    kWarpBroadcast = 2,
};
inline SharedLoadPattern parse_shared_load_pattern(const std::string& value) {
    if (value == "unique") return SharedLoadPattern::kUnique;
    if (value == "quad_broadcast") {
        return SharedLoadPattern::kQuadBroadcast;
    }
    if (value == "warp_broadcast") {
        return SharedLoadPattern::kWarpBroadcast;
    }
    throw std::invalid_argument(
        "--pattern must be unique, quad_broadcast, or warp_broadcast");
}

inline bool is_power_of_two(int value) {
    return value > 0 && (value & (value - 1)) == 0;
}

inline void require_power_of_two(const char* option, int value) {
    if (!is_power_of_two(value)) {
        throw std::invalid_argument(std::string(option) +
                                    " must be a positive power of two");
    }
}

inline int require_warp_multiple(const Args& args,
                                 std::string_view name,
                                 int default_value,
                                 int minimum = 32,
                                 int maximum = 256) {
    const int value = args.get_int(name, default_value, minimum, maximum);
    if ((value & 31) != 0) {
        throw std::invalid_argument("--" + std::string(name) +
                                    " must be a multiple of 32");
    }
    return value;
}
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
                                 double target_cycles_per_round,
                                 double baseline_cycles_per_round,
                                 const std::vector<double>& samples = {},
                                 const std::vector<double>& target_samples = {},
                                 const std::vector<double>& baseline_samples = {}) {
    JsonObject result;
    result.add("value", cycles_per_round)
        .add("unit", "cycle/round")
        .add("timer", "clock64")
        .add("scope", "cta")
        .add("kind", "matched_target_minus_baseline")
        .add("boundary", boundary)
        .add("source_ops_per_round", source_ops_per_round)
        .add("target_median_cycles_per_round", target_cycles_per_round)
        .add("baseline_median_cycles_per_round", baseline_cycles_per_round)
        .add_raw("samples", json_number_array(samples))
        .add_raw("target_samples_cycles", json_number_array(target_samples))
        .add_raw("baseline_samples_cycles", json_number_array(baseline_samples));
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
        .add("timed_kernel", "target_only")
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
                                double target_cycles_per_round,
                                double baseline_cycles_per_round,
                                double total_source_ops,
                                double requested_bytes,
                                double elapsed_ms,
                                std::string_view operation,
                                std::string_view memory_space,
                                double peak,
                                bool peak_is_applicable,
                                const std::vector<double>& latency_samples = {},
                                const std::vector<double>& target_samples = {},
                                const std::vector<double>& baseline_samples = {},
                                const std::vector<double>& event_ms_samples = {}) {
    const JsonObject latency = latency_metric(
        cycles_per_round, latency_boundary, source_ops_per_round,
        target_cycles_per_round, baseline_cycles_per_round, latency_samples,
        target_samples, baseline_samples);
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

template <bool IssueTarget>
__device__ __forceinline__ uint32_t load_shared_u32_or_baseline(
        const uint32_t* pointer) {
    const uint32_t address = shared_address(pointer);
    uint32_t value = address;
    if constexpr (IssueTarget) {
        asm volatile("ld.shared.u32 %0, [%1];"
                     : "=r"(value)
                     : "r"(address)
                     : "memory");
    } else {
        // Keep the same address and checksum dataflow without issuing LDS.
        asm volatile("" : "+r"(value) : : "memory");
    }
    return value;
}

template <Variant V, bool IssueTarget>
__device__ __forceinline__ void shared_load_u32_body(
        uint64_t* cycles,
        uint32_t* sinks,
        int iterations,
        int working_words,
        SharedLoadPattern pattern) {
    static_assert(V == Variant::kSharedLoadU32Patterns);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    extern __shared__ __align__(128) uint32_t storage[];
    for (int index = threadIdx.x; index < working_words; index += blockDim.x) {
        storage[index] = static_cast<uint32_t>(index + 1);
    }
    __syncthreads();

    uint64_t start = 0;
    if (cycles != nullptr && threadIdx.x == 0) start = read_clock64();
    __syncthreads();

    uint32_t checksum = static_cast<uint32_t>(threadIdx.x + 1);
    const int mask = working_words - 1;
#pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        int group = static_cast<int>(threadIdx.x);
        if (pattern == SharedLoadPattern::kQuadBroadcast) group >>= 2;
        if (pattern == SharedLoadPattern::kWarpBroadcast) group >>= 5;
        const int index = (iteration * 32 + group) & mask;
        const uint32_t value =
            load_shared_u32_or_baseline<IssueTarget>(storage + index);
        checksum = (checksum << 5) ^ (checksum >> 2) ^ value;
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        if (cycles != nullptr) cycles[blockIdx.x] = read_clock64() - start;
        sinks[blockIdx.x] = checksum == 0 ? 1 : checksum;
    }
#else
    if (threadIdx.x == 0) {
        cycles[blockIdx.x] = 0;
        sinks[blockIdx.x] = 0;
    }
#endif
}

template <Variant V>
__global__ void shared_load_u32_target_kernel(uint64_t* cycles,
                                              uint32_t* sinks,
                                              int iterations,
                                              int working_words,
                                              SharedLoadPattern pattern) {
    shared_load_u32_body<V, true>(
        cycles, sinks, iterations, working_words, pattern);
}

template <Variant V>
__global__ void shared_load_u32_baseline_kernel(uint64_t* cycles,
                                                uint32_t* sinks,
                                                int iterations,
                                                int working_words,
                                                SharedLoadPattern pattern) {
    shared_load_u32_body<V, false>(
        cycles, sinks, iterations, working_words, pattern);
}

struct TimingResult {
    double cycles_per_round;
    double target_cycles_per_round;
    double baseline_cycles_per_round;
    double elapsed_ms;
    std::vector<double> latency_samples;
    std::vector<double> target_samples;
    std::vector<double> baseline_samples;
    std::vector<double> event_ms_samples;
};

template <typename PairedLatencyLaunch, typename ThroughputLaunch>
inline TimingResult measure_atomic(const CommonOptions& options,
                                   uint64_t* target_cycles,
                                   uint64_t* baseline_cycles,
                                   PairedLatencyLaunch&& paired_latency_launch,
                                   ThroughputLaunch&& throughput_launch) {
    const auto clock_samples = measure_paired_clock_cycles(
        options.warmup, options.samples, target_cycles, baseline_cycles, 1,
        std::forward<PairedLatencyLaunch>(paired_latency_launch));
    const auto event_samples = measure_event_ms(
        options.warmup, options.samples,
        std::forward<ThroughputLaunch>(throughput_launch));
    std::vector<double> normalized_cycle_samples;
    normalized_cycle_samples.reserve(clock_samples.target.size());
    for (std::size_t index = 0; index < clock_samples.target.size(); ++index) {
        normalized_cycle_samples.push_back(
            (clock_samples.target[index] - clock_samples.baseline[index]) /
            options.iters);
    }
    return TimingResult{median(normalized_cycle_samples),
                        median(clock_samples.target) / options.iters,
                        median(clock_samples.baseline) / options.iters,
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

template <Variant V>
inline int run_shared_load_u32(int argc, char** argv) {
    static_assert(V == Variant::kSharedLoadU32Patterns);
    const Args args(argc, argv);
    args.require_only({"iters", "warmup", "samples", "blocks", "device",
                       "peak", "threads", "pattern", "working-set-words"});
    const CommonOptions options = parse_common_options(args, 4096);
    const cudaDeviceProp properties = require_sm90(options.device);
    const int threads = require_warp_multiple(args, "threads", 128);
    const std::string pattern_name = args.get_string("pattern", "quad_broadcast");
    const SharedLoadPattern pattern = parse_shared_load_pattern(pattern_name);
    const int working_words = args.get_int(
        "working-set-words", 4096, 32, 1 << 16);
    require_power_of_two("--working-set-words", working_words);
    const int shared_bytes = working_words * static_cast<int>(sizeof(uint32_t));
    if (shared_bytes > static_cast<int>(properties.sharedMemPerBlockOptin)) {
        throw std::invalid_argument(
            "working shared-load set exceeds sharedMemPerBlockOptin");
    }
    const int blocks = resolve_blocks(options.blocks, properties, 4);

    CUDA_CHECK(cudaFuncSetAttribute(
        shared_load_u32_target_kernel<V>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
    CUDA_CHECK(cudaFuncSetAttribute(
        shared_load_u32_baseline_kernel<V>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
    DeviceBuffer<uint64_t> latency_target_cycles(1);
    DeviceBuffer<uint64_t> latency_baseline_cycles(1);
    DeviceBuffer<uint32_t> latency_target_sinks(1);
    DeviceBuffer<uint32_t> latency_baseline_sinks(1);
    DeviceBuffer<uint32_t> throughput_sinks(blocks);
    const TimingResult timing = measure_atomic(
        options, latency_target_cycles.data(), latency_baseline_cycles.data(),
        [&] {
            shared_load_u32_target_kernel<V><<<1, threads, shared_bytes>>>(
                latency_target_cycles.data(), latency_target_sinks.data(),
                options.iters, working_words, pattern);
            shared_load_u32_baseline_kernel<V><<<1, threads, shared_bytes>>>(
                latency_baseline_cycles.data(), latency_baseline_sinks.data(),
                options.iters,
                working_words, pattern);
        },
        [&] {
            shared_load_u32_target_kernel<V><<<blocks, threads, shared_bytes>>>(
                nullptr, throughput_sinks.data(), options.iters,
                working_words, pattern);
        });
    require_nonzero_sinks(latency_target_sinks, "shared ld.u32 target");
    require_nonzero_sinks(latency_baseline_sinks, "shared ld.u32 baseline");
    require_nonzero_sinks(throughput_sinks, "shared ld.u32");

    const double source_ops_per_round = static_cast<double>(threads);
    const double total_source_ops = static_cast<double>(blocks) * options.iters *
                                    source_ops_per_round;
    const double requested_bytes = total_source_ops * sizeof(uint32_t);
    JsonObject params;
    add_common_params(params, properties, options, blocks);
    params.add("threads", threads)
        .add("pattern", pattern_name)
        .add("working_set_words", working_words)
        .add("working_set_bytes", shared_bytes)
        .add("ptx", "ld.shared.u32")
        .add("clock_baseline",
             "matched address/pattern/checksum/control kernel without LDS")
        .add("throughput_protocol", "CUDA-event target kernel only")
        .add("correct", true);
    print_memory_result(
        "ld_shared_u32_patterns", params, timing.cycles_per_round,
        "target issue/dataflow loop minus matched address, pattern, checksum, "
        "loop-control, and CTA-convergence baseline",
        source_ops_per_round, timing.target_cycles_per_round,
        timing.baseline_cycles_per_round, total_source_ops, requested_bytes,
        timing.elapsed_ms, "ld.shared.u32", "shared", options.peak, false,
        timing.latency_samples, timing.target_samples, timing.baseline_samples,
        timing.event_ms_samples);
    return 0;
}

}  // namespace microbench::memory_atomic
