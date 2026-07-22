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
    kGlobalStoreF32,
    kGlobalStoreRecord32B,
    kGlobalStoreU32,
    kGlobalStoreU64,
};

enum class AccessPattern : int {
    kLocal = 0,
    kSequential = 1,
    kRandom = 2,
    kBroadcast = 3,
};
inline AccessPattern parse_access_pattern(const std::string& value,
                                          bool allow_broadcast = true) {
    if (value == "local") return AccessPattern::kLocal;
    if (value == "sequential") return AccessPattern::kSequential;
    if (value == "random") return AccessPattern::kRandom;
    if (allow_broadcast && value == "broadcast") {
        return AccessPattern::kBroadcast;
    }
    throw std::invalid_argument(
        allow_broadcast
            ? "--pattern must be local, sequential, random, or broadcast"
            : "--pattern must be local, sequential, or random");
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

template <typename T>
inline void copy_host_to_device(DeviceBuffer<T>& destination,
                                const std::vector<T>& source) {
    if (destination.size() != source.size()) {
        throw std::logic_error("host/device copy size mismatch");
    }
    if (!source.empty()) {
        CUDA_CHECK(cudaMemcpy(destination.data(), source.data(),
                              source.size() * sizeof(T),
                              cudaMemcpyHostToDevice));
    }
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
struct alignas(32) Record32B {
    uint32_t words[8];
};
static_assert(sizeof(Record32B) == 32);

struct RegisterRecord {
    uint32_t words[8];
};

__device__ __forceinline__ RegisterRecord load_global_record_32b(
        const Record32B* pointer) {
    RegisterRecord result{};
    const auto* bytes = reinterpret_cast<const unsigned char*>(pointer);
    asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%8];\n\t"
                 "ld.global.v4.u32 {%4, %5, %6, %7}, [%9];"
                 : "=r"(result.words[0]), "=r"(result.words[1]),
                   "=r"(result.words[2]), "=r"(result.words[3]),
                   "=r"(result.words[4]), "=r"(result.words[5]),
                   "=r"(result.words[6]), "=r"(result.words[7])
                 : "l"(bytes), "l"(bytes + 16)
                 : "memory");
    return result;
}

__device__ __forceinline__ void store_global_record_32b(
        Record32B* pointer,
        const RegisterRecord& value) {
    auto* bytes = reinterpret_cast<unsigned char*>(pointer);
    asm volatile("st.global.v4.u32 [%0], {%2, %3, %4, %5};\n\t"
                 "st.global.v4.u32 [%1], {%6, %7, %8, %9};"
                 :
                 : "l"(bytes), "l"(bytes + 16),
                   "r"(value.words[0]), "r"(value.words[1]),
                   "r"(value.words[2]), "r"(value.words[3]),
                   "r"(value.words[4]), "r"(value.words[5]),
                   "r"(value.words[6]), "r"(value.words[7])
                 : "memory");
}
__device__ __forceinline__ void store_global_f32(float* pointer, float value) {
    asm volatile("st.global.f32 [%0], %1;"
                 :
                 : "l"(pointer), "f"(value)
                 : "memory");
}

__device__ __forceinline__ void store_global_u32(uint32_t* pointer,
                                                  uint32_t value) {
    asm volatile("st.global.u32 [%0], %1;"
                 :
                 : "l"(pointer), "r"(value)
                 : "memory");
}

__device__ __forceinline__ void store_global_u64(uint64_t* pointer,
                                                  uint64_t value) {
    asm volatile("st.global.u64 [%0], %1;"
                 :
                 : "l"(pointer), "l"(value)
                 : "memory");
}
__device__ __forceinline__ int select_power_of_two_index(
        AccessPattern pattern,
        int iteration,
        int lane_item,
        int mask,
        int local_mask) {
    if (pattern == AccessPattern::kBroadcast) {
        return iteration & mask;
    }
    if (pattern == AccessPattern::kLocal) {
        return (lane_item + iteration * 32) & local_mask;
    }
    if (pattern == AccessPattern::kRandom) {
        uint32_t value = static_cast<uint32_t>(iteration + 1) * 0x85ebca6bu +
                         static_cast<uint32_t>(lane_item + 1) * 0x9e3779b9u;
        value ^= value >> 16;
        value *= 0x7feb352du;
        value ^= value >> 15;
        return static_cast<int>(value) & mask;
    }
    return (iteration * blockDim.x + lane_item) & mask;
}

__device__ __forceinline__ int select_private_record(
        AccessPattern pattern,
        int iteration,
        int records_per_block) {
    if (pattern == AccessPattern::kLocal) return 0;
    if (pattern == AccessPattern::kRandom) {
        uint32_t value = static_cast<uint32_t>(iteration + 1) * 0x85ebca6bu;
        value ^= value >> 16;
        value *= 0x7feb352du;
        value ^= value >> 15;
        return static_cast<int>(value) & (records_per_block - 1);
    }
    return iteration & (records_per_block - 1);
}
template <Variant V, bool Target = true>
__global__ void global_store_f32_kernel(float* output,
                                             uint64_t* cycles,
                                             uint32_t* sinks,
                                             int iterations,
                                             int records_per_block,
                                             int lane_mode,
                                             AccessPattern pattern) {
    static_assert(V == Variant::kGlobalStoreF32);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    const int values_per_record = lane_mode == 0 ? 64 : 8;
    const int producer = lane_mode == 0
        ? static_cast<int>(threadIdx.x)
        : (static_cast<int>(threadIdx.x) % 32 == 0
               ? static_cast<int>(threadIdx.x) / 32
               : -1);
    float* block_base = output +
        static_cast<int64_t>(blockIdx.x) * records_per_block * values_per_record;

    uint64_t start = 0;
    if (threadIdx.x == 0) start = read_clock64();
    __syncthreads();

#pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const int record = select_private_record(
            pattern, iteration, records_per_block);
        if (producer >= 0 && producer < values_per_record) {
            float* pointer =
                block_base + record * values_per_record + producer;
            const float value = static_cast<float>(iteration + producer + 1);
            if constexpr (Target) {
                store_global_f32(pointer, value);
            } else {
                asm volatile("" : : "l"(pointer), "f"(value) : "memory");
            }
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        cycles[blockIdx.x] = read_clock64() - start;
        sinks[blockIdx.x] = static_cast<uint32_t>(iterations + lane_mode + 1);
    }
#else
    if (threadIdx.x == 0) {
        cycles[blockIdx.x] = 0;
        sinks[blockIdx.x] = 0;
    }
#endif
}

template <Variant V, bool Target = true>
__global__ void global_store_record_32b_kernel(
        Record32B* output,
        uint64_t* cycles,
        uint32_t* sinks,
        int iterations,
        int records_per_block,
        AccessPattern pattern) {
    static_assert(V == Variant::kGlobalStoreRecord32B);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    Record32B* block_base = output +
        static_cast<int64_t>(blockIdx.x) * records_per_block;
    RegisterRecord value{};
#pragma unroll
    for (int word = 0; word < 8; ++word) {
        value.words[word] = static_cast<uint32_t>(threadIdx.x + word + 1);
    }

    uint64_t start = 0;
    if (threadIdx.x == 0) start = read_clock64();
    __syncthreads();

#pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (threadIdx.x == 0) {
            const int record = select_private_record(
                pattern, iteration, records_per_block);
            value.words[0] = static_cast<uint32_t>(iteration + 1);
            Record32B* pointer = block_base + record;
            if constexpr (Target) {
                store_global_record_32b(pointer, value);
            } else {
                asm volatile(""
                             :
                             : "l"(pointer), "r"(value.words[0]),
                               "r"(value.words[7])
                             : "memory");
            }
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        cycles[blockIdx.x] = read_clock64() - start;
        sinks[blockIdx.x] = static_cast<uint32_t>(iterations + 1);
    }
#else
    if (threadIdx.x == 0) {
        cycles[blockIdx.x] = 0;
        sinks[blockIdx.x] = 0;
    }
#endif
}

template <Variant V, bool Target = true>
__global__ void global_store_u32_kernel(
        uint32_t* output,
        uint64_t* cycles,
        uint32_t* sinks,
        int iterations,
        int records_per_block,
        int producers,
        AccessPattern pattern) {
    static_assert(V == Variant::kGlobalStoreU32);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    uint32_t* block_base = output +
        static_cast<int64_t>(blockIdx.x) * records_per_block * producers;

    uint64_t start = 0;
    if (threadIdx.x == 0) start = read_clock64();
    __syncthreads();

#pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const int record = select_private_record(
            pattern, iteration, records_per_block);
        if (threadIdx.x < producers) {
            uint32_t* pointer =
                block_base + record * producers + threadIdx.x;
            const uint32_t value =
                static_cast<uint32_t>(iteration + threadIdx.x + 1);
            if constexpr (Target) {
                store_global_u32(pointer, value);
            } else {
                asm volatile("" : : "l"(pointer), "r"(value) : "memory");
            }
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        cycles[blockIdx.x] = read_clock64() - start;
        sinks[blockIdx.x] = static_cast<uint32_t>(iterations + 1);
    }
#else
    if (threadIdx.x == 0) {
        cycles[blockIdx.x] = 0;
        sinks[blockIdx.x] = 0;
    }
#endif
}

template <Variant V, bool Target = true>
__global__ void global_store_u64_kernel(
        uint64_t* output,
        uint64_t* cycles,
        uint32_t* sinks,
        int iterations,
        int records_per_block,
        int active_warps,
        int vectors_per_thread,
        AccessPattern pattern,
        uint64_t value_pattern) {
    static_assert(V == Variant::kGlobalStoreU64);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    constexpr int kLanes = 32;
    const int warp = static_cast<int>(threadIdx.x) / kLanes;
    const int lane = static_cast<int>(threadIdx.x) & (kLanes - 1);
    const int values_per_record = active_warps * kLanes * vectors_per_thread;
    uint64_t* block_base = output +
        static_cast<int64_t>(blockIdx.x) * records_per_block * values_per_record;

    uint64_t start = 0;
    if (threadIdx.x == 0) start = read_clock64();
    __syncthreads();

#pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const int record = select_private_record(
            pattern, iteration, records_per_block);
        if (warp < active_warps) {
#pragma unroll 1
            for (int vector = 0; vector < vectors_per_thread; ++vector) {
                const int index = warp * kLanes * vectors_per_thread +
                                  vector * kLanes + lane;
                uint64_t* pointer =
                    block_base + record * values_per_record + index;
                const uint64_t value =
                    value_pattern ^ static_cast<uint64_t>(iteration + index);
                if constexpr (Target) {
                    store_global_u64(pointer, value);
                } else {
                    asm volatile(""
                                 :
                                 : "l"(pointer), "l"(value)
                                 : "memory");
                }
            }
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        cycles[blockIdx.x] = read_clock64() - start;
        sinks[blockIdx.x] = static_cast<uint32_t>(iterations + 1);
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
inline int run_global_store_f32(int argc, char** argv) {
    static_assert(V == Variant::kGlobalStoreF32);
    const Args args(argc, argv);
    args.require_only({"iters", "warmup", "samples", "blocks", "device",
                       "peak", "lane-mode", "working-set-records", "pattern"});
    const CommonOptions options = parse_common_options(args, 4096);
    const cudaDeviceProp properties = require_sm90(options.device);
    const std::string lane_mode_name =
        args.get_string("lane-mode", "width64");
    int lane_mode = -1;
    if (lane_mode_name == "width64") lane_mode = 0;
    if (lane_mode_name == "width8") lane_mode = 1;
    if (lane_mode < 0) {
        throw std::invalid_argument(
            "--lane-mode must be width64 or width8");
    }
    const int values_per_record = lane_mode == 0 ? 64 : 8;
    const int records_per_block = args.get_int(
        "working-set-records", 64, 1, 1 << 16);
    require_power_of_two("--working-set-records", records_per_block);
    const std::string pattern_name = args.get_string("pattern", "sequential");
    const AccessPattern pattern = parse_access_pattern(pattern_name, false);
    const int blocks = resolve_blocks(options.blocks, properties, 4);
    const std::size_t output_elements = static_cast<std::size_t>(blocks) *
                                        records_per_block * values_per_record;

    DeviceBuffer<float> output(output_elements);
    output.zero();
    DeviceBuffer<uint64_t> latency_target_cycles(1);
    DeviceBuffer<uint64_t> latency_baseline_cycles(1);
    DeviceBuffer<uint64_t> throughput_cycles(blocks);
    DeviceBuffer<uint32_t> latency_target_sinks(1);
    DeviceBuffer<uint32_t> latency_baseline_sinks(1);
    DeviceBuffer<uint32_t> throughput_sinks(blocks);
    const TimingResult timing = measure_atomic(
        options, latency_target_cycles.data(), latency_baseline_cycles.data(),
        [&] {
            global_store_f32_kernel<V, true><<<1, 256>>>(
                output.data(), latency_target_cycles.data(),
                latency_target_sinks.data(),
                options.iters, records_per_block, lane_mode, pattern);
            global_store_f32_kernel<V, false><<<1, 256>>>(
                output.data(), latency_baseline_cycles.data(),
                latency_baseline_sinks.data(),
                options.iters, records_per_block, lane_mode, pattern);
        },
        [&] {
            global_store_f32_kernel<V, true><<<blocks, 256>>>(
                output.data(), throughput_cycles.data(), throughput_sinks.data(),
                options.iters, records_per_block, lane_mode, pattern);
        });
    require_nonzero_sinks(latency_target_sinks, "global f32 store target");
    require_nonzero_sinks(latency_baseline_sinks,
                          "global f32 store baseline");
    require_nonzero_sinks(throughput_sinks, "global f32 store");

    output.zero();
    global_store_f32_kernel<V, true><<<1, 256>>>(
        output.data(), latency_target_cycles.data(),
        latency_target_sinks.data(), 1,
        records_per_block, lane_mode, AccessPattern::kLocal);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float validation = 0.0f;
    CUDA_CHECK(cudaMemcpy(&validation, output.data(), sizeof(validation),
                          cudaMemcpyDeviceToHost));
    if (validation == 0.0f) {
        throw std::runtime_error("global f32 store validation failed");
    }

    const double source_ops_per_round = static_cast<double>(values_per_record);
    const double total_source_ops = static_cast<double>(blocks) * options.iters *
                                    source_ops_per_round;
    const double requested_bytes = total_source_ops * sizeof(float);
    JsonObject params;
    add_common_params(params, properties, options, blocks);
    params.add("lane_mode", lane_mode_name)
        .add("working_set_records", records_per_block)
        .add("working_set_bytes",
             static_cast<uint64_t>(output_elements * sizeof(float)))
        .add("pattern", pattern_name)
        .add("producers", values_per_record)
        .add("ptx", "st.global.f32")
        .add("clock_baseline",
             "matched address/value/predicate/control kernel without STG")
        .add("throughput_protocol", "CUDA-event target kernel only")
        .add("correct", true);
    print_memory_result(
        "st_global_f32", params,
        timing.cycles_per_round,
        "target scalar f32-store loop minus matched address, value, predicate, "
        "loop-control, and CTA-convergence baseline",
        source_ops_per_round, timing.target_cycles_per_round,
        timing.baseline_cycles_per_round, total_source_ops, requested_bytes,
        timing.elapsed_ms, "st.global.f32", "global", options.peak, true,
        timing.latency_samples, timing.target_samples, timing.baseline_samples,
        timing.event_ms_samples);
    return 0;
}

template <Variant V>
inline int run_global_store_record_32b(int argc, char** argv) {
    static_assert(V == Variant::kGlobalStoreRecord32B);
    const Args args(argc, argv);
    args.require_only({"iters", "warmup", "samples", "blocks", "device",
                       "peak", "working-set-records", "pattern"});
    const CommonOptions options = parse_common_options(args, 4096);
    const cudaDeviceProp properties = require_sm90(options.device);
    const int records_per_block = args.get_int(
        "working-set-records", 64, 1, 1 << 16);
    require_power_of_two("--working-set-records", records_per_block);
    const std::string pattern_name = args.get_string("pattern", "sequential");
    const AccessPattern pattern = parse_access_pattern(pattern_name, false);
    const int blocks = resolve_blocks(options.blocks, properties, 4);
    const std::size_t output_records = static_cast<std::size_t>(blocks) *
                                       records_per_block;

    DeviceBuffer<Record32B> output(output_records);
    output.zero();
    DeviceBuffer<uint64_t> latency_target_cycles(1);
    DeviceBuffer<uint64_t> latency_baseline_cycles(1);
    DeviceBuffer<uint64_t> throughput_cycles(blocks);
    DeviceBuffer<uint32_t> latency_target_sinks(1);
    DeviceBuffer<uint32_t> latency_baseline_sinks(1);
    DeviceBuffer<uint32_t> throughput_sinks(blocks);
    const TimingResult timing = measure_atomic(
        options, latency_target_cycles.data(), latency_baseline_cycles.data(),
        [&] {
            global_store_record_32b_kernel<V, true><<<1, 32>>>(
                output.data(), latency_target_cycles.data(),
                latency_target_sinks.data(),
                options.iters, records_per_block, pattern);
            global_store_record_32b_kernel<V, false><<<1, 32>>>(
                output.data(), latency_baseline_cycles.data(),
                latency_baseline_sinks.data(),
                options.iters, records_per_block, pattern);
        },
        [&] {
            global_store_record_32b_kernel<V, true><<<blocks, 32>>>(
                output.data(), throughput_cycles.data(), throughput_sinks.data(),
                options.iters, records_per_block, pattern);
        });
    require_nonzero_sinks(latency_target_sinks, "record store target");
    require_nonzero_sinks(latency_baseline_sinks, "record store baseline");
    require_nonzero_sinks(throughput_sinks, "generic record store");

    output.zero();
    global_store_record_32b_kernel<V, true><<<1, 32>>>(
        output.data(), latency_target_cycles.data(),
        latency_target_sinks.data(), 1,
        records_per_block, AccessPattern::kLocal);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    Record32B validation{};
    CUDA_CHECK(cudaMemcpy(&validation, output.data(), sizeof(validation),
                          cudaMemcpyDeviceToHost));
    if (validation.words[0] == 0 || validation.words[7] == 0) {
        throw std::runtime_error("generic record store validation failed");
    }

    constexpr double source_ops_per_round = 2.0;
    const double total_source_ops = static_cast<double>(blocks) * options.iters *
                                    source_ops_per_round;
    const double requested_bytes = static_cast<double>(blocks) * options.iters *
                                   sizeof(Record32B);
    JsonObject params;
    add_common_params(params, properties, options, blocks);
    params.add("working_set_records", records_per_block)
        .add("working_set_bytes",
             static_cast<uint64_t>(output_records * sizeof(Record32B)))
        .add("pattern", pattern_name)
        .add("record_bytes", sizeof(Record32B))
        .add("issuers", 1)
        .add("ptx", "2x st.global.v4.u32")
        .add("clock_baseline",
             "matched address/value/predicate/control kernel without STG")
        .add("throughput_protocol", "CUDA-event target kernel only")
        .add("correct", true);
    print_memory_result(
        "st_global_v4_u32_32b", params,
        timing.cycles_per_round,
        "target two-store record loop minus matched address, value, predicate, "
        "loop-control, and CTA-convergence baseline",
        source_ops_per_round, timing.target_cycles_per_round,
        timing.baseline_cycles_per_round, total_source_ops, requested_bytes,
        timing.elapsed_ms, "st.global.v4.u32", "global", options.peak, true,
        timing.latency_samples, timing.target_samples, timing.baseline_samples,
        timing.event_ms_samples);
    return 0;
}

template <Variant V>
inline int run_global_store_u32(int argc, char** argv) {
    static_assert(V == Variant::kGlobalStoreU32);
    const Args args(argc, argv);
    args.require_only({"iters", "warmup", "samples", "blocks", "device",
                       "peak", "producers", "working-set-records", "pattern"});
    const CommonOptions options = parse_common_options(args, 4096);
    const cudaDeviceProp properties = require_sm90(options.device);
    const int producers = args.get_int("producers", 32, 1, 32);
    const int records_per_block = args.get_int(
        "working-set-records", 64, 1, 1 << 16);
    require_power_of_two("--working-set-records", records_per_block);
    const std::string pattern_name = args.get_string("pattern", "sequential");
    const AccessPattern pattern = parse_access_pattern(pattern_name, false);
    const int blocks = resolve_blocks(options.blocks, properties, 4);
    const std::size_t output_elements = static_cast<std::size_t>(blocks) *
                                        records_per_block * producers;

    DeviceBuffer<uint32_t> output(output_elements);
    output.zero();
    DeviceBuffer<uint64_t> latency_target_cycles(1);
    DeviceBuffer<uint64_t> latency_baseline_cycles(1);
    DeviceBuffer<uint64_t> throughput_cycles(blocks);
    DeviceBuffer<uint32_t> latency_target_sinks(1);
    DeviceBuffer<uint32_t> latency_baseline_sinks(1);
    DeviceBuffer<uint32_t> throughput_sinks(blocks);
    const TimingResult timing = measure_atomic(
        options, latency_target_cycles.data(), latency_baseline_cycles.data(),
        [&] {
            global_store_u32_kernel<V, true><<<1, 32>>>(
                output.data(), latency_target_cycles.data(),
                latency_target_sinks.data(),
                options.iters, records_per_block, producers, pattern);
            global_store_u32_kernel<V, false><<<1, 32>>>(
                output.data(), latency_baseline_cycles.data(),
                latency_baseline_sinks.data(),
                options.iters, records_per_block, producers, pattern);
        },
        [&] {
            global_store_u32_kernel<V, true><<<blocks, 32>>>(
                output.data(), throughput_cycles.data(), throughput_sinks.data(),
                options.iters, records_per_block, producers, pattern);
        });
    require_nonzero_sinks(latency_target_sinks, "scalar u32 store target");
    require_nonzero_sinks(latency_baseline_sinks,
                          "scalar u32 store baseline");
    require_nonzero_sinks(throughput_sinks, "scalar u32 store");

    output.zero();
    global_store_u32_kernel<V, true><<<1, 32>>>(
        output.data(), latency_target_cycles.data(),
        latency_target_sinks.data(), 1,
        records_per_block, producers, AccessPattern::kLocal);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t validation = 0;
    CUDA_CHECK(cudaMemcpy(&validation, output.data(), sizeof(validation),
                          cudaMemcpyDeviceToHost));
    if (validation == 0) {
        throw std::runtime_error("scalar u32 store validation failed");
    }

    const double source_ops_per_round = static_cast<double>(producers);
    const double total_source_ops = static_cast<double>(blocks) * options.iters *
                                    source_ops_per_round;
    const double requested_bytes = total_source_ops * sizeof(uint32_t);
    JsonObject params;
    add_common_params(params, properties, options, blocks);
    params.add("producers", producers)
        .add("working_set_records", records_per_block)
        .add("working_set_bytes",
             static_cast<uint64_t>(output_elements * sizeof(uint32_t)))
        .add("pattern", pattern_name)
        .add("ptx", "st.global.u32")
        .add("clock_baseline",
             "matched address/value/predicate/control kernel without STG")
        .add("throughput_protocol", "CUDA-event target kernel only")
        .add("correct", true);
    print_memory_result(
        "st_global_u32", params,
        timing.cycles_per_round,
        "target scalar u32-store loop minus matched address, value, predicate, "
        "loop-control, and CTA-convergence baseline",
        source_ops_per_round, timing.target_cycles_per_round,
        timing.baseline_cycles_per_round, total_source_ops, requested_bytes,
        timing.elapsed_ms, "st.global.u32", "global", options.peak, true,
        timing.latency_samples, timing.target_samples, timing.baseline_samples,
        timing.event_ms_samples);
    return 0;
}

template <Variant V>
inline int run_global_store_u64(int argc, char** argv) {
    static_assert(V == Variant::kGlobalStoreU64);
    const Args args(argc, argv);
    args.require_only({"iters", "warmup", "samples", "blocks", "device",
                       "peak", "dtype", "warps", "vectors-per-thread",
                       "working-set-records", "pattern"});
    const CommonOptions options = parse_common_options(args, 1024);
    const cudaDeviceProp properties = require_sm90(options.device);
    const std::string dtype = args.get_string("dtype", "bf16");
    uint64_t value_pattern = 0;
    if (dtype == "bf16") value_pattern = 0x3f803f803f803f80ull;
    if (dtype == "fp16") value_pattern = 0x3c003c003c003c00ull;
    if (value_pattern == 0) {
        throw std::invalid_argument("--dtype must be bf16 or fp16");
    }
    const int active_warps = args.get_int("warps", 8, 1, 8);
    const int vectors_per_thread = args.get_int(
        "vectors-per-thread", 4, 1, 4);
    const int records_per_block = args.get_int(
        "working-set-records", 64, 1, 1 << 16);
    require_power_of_two("--working-set-records", records_per_block);
    const std::string pattern_name = args.get_string("pattern", "sequential");
    const AccessPattern pattern = parse_access_pattern(pattern_name, false);
    const int blocks = resolve_blocks(options.blocks, properties, 2);
    const int values_per_record = active_warps * 32 * vectors_per_thread;
    const std::size_t output_elements = static_cast<std::size_t>(blocks) *
                                        records_per_block * values_per_record;

    DeviceBuffer<uint64_t> output(output_elements);
    output.zero();
    DeviceBuffer<uint64_t> latency_target_cycles(1);
    DeviceBuffer<uint64_t> latency_baseline_cycles(1);
    DeviceBuffer<uint64_t> throughput_cycles(blocks);
    DeviceBuffer<uint32_t> latency_target_sinks(1);
    DeviceBuffer<uint32_t> latency_baseline_sinks(1);
    DeviceBuffer<uint32_t> throughput_sinks(blocks);
    const TimingResult timing = measure_atomic(
        options, latency_target_cycles.data(), latency_baseline_cycles.data(),
        [&] {
            global_store_u64_kernel<V, true><<<1, 256>>>(
                output.data(), latency_target_cycles.data(),
                latency_target_sinks.data(),
                options.iters, records_per_block, active_warps,
                vectors_per_thread, pattern, value_pattern);
            global_store_u64_kernel<V, false><<<1, 256>>>(
                output.data(), latency_baseline_cycles.data(),
                latency_baseline_sinks.data(),
                options.iters, records_per_block, active_warps,
                vectors_per_thread, pattern, value_pattern);
        },
        [&] {
            global_store_u64_kernel<V, true><<<blocks, 256>>>(
                output.data(), throughput_cycles.data(), throughput_sinks.data(),
                options.iters, records_per_block, active_warps,
                vectors_per_thread, pattern, value_pattern);
        });
    require_nonzero_sinks(latency_target_sinks, "u64 store target");
    require_nonzero_sinks(latency_baseline_sinks, "u64 store baseline");
    require_nonzero_sinks(throughput_sinks, "BF16/FP16 output u64 store");

    output.zero();
    global_store_u64_kernel<V, true><<<1, 256>>>(
        output.data(), latency_target_cycles.data(),
        latency_target_sinks.data(), 1,
        records_per_block, active_warps, vectors_per_thread,
        AccessPattern::kLocal, value_pattern);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    uint64_t validation = 0;
    CUDA_CHECK(cudaMemcpy(&validation, output.data(), sizeof(validation),
                          cudaMemcpyDeviceToHost));
    if (validation == 0) {
        throw std::runtime_error("BF16/FP16 output u64 store validation failed");
    }

    const double source_ops_per_round = static_cast<double>(values_per_record);
    const double total_source_ops = static_cast<double>(blocks) * options.iters *
                                    source_ops_per_round;
    const double requested_bytes = total_source_ops * sizeof(uint64_t);
    JsonObject params;
    add_common_params(params, properties, options, blocks);
    params.add("dtype", dtype)
        .add("warps", active_warps)
        .add("vectors_per_thread", vectors_per_thread)
        .add("working_set_records", records_per_block)
        .add("working_set_bytes",
             static_cast<uint64_t>(output_elements * sizeof(uint64_t)))
        .add("pattern", pattern_name)
        .add("bytes_per_cta_round",
             static_cast<uint64_t>(values_per_record) * sizeof(uint64_t))
        .add("ptx", "st.global.u64")
        .add("clock_baseline",
             "matched address/value/predicate/control kernel without STG")
        .add("throughput_protocol", "CUDA-event target kernel only")
        .add("correct", true);
    print_memory_result(
        "st_global_u64", params,
        timing.cycles_per_round,
        "target packed-b16 u64-store loop minus matched address, value, "
        "predicate, loop-control, and CTA-convergence baseline",
        source_ops_per_round, timing.target_cycles_per_round,
        timing.baseline_cycles_per_round, total_source_ops, requested_bytes,
        timing.elapsed_ms, "st.global.u64", "global", options.peak, true,
        timing.latency_samples, timing.target_samples, timing.baseline_samples,
        timing.event_ms_samples);
    return 0;
}

}  // namespace microbench::memory_atomic
