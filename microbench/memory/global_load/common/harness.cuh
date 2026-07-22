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
    kGlobalLoadI32Cached,
    kGlobalLoadI32Ordinary,
    kGlobalLoadRecord32B,
    kGlobalLoadV4F32,
    kGlobalLoadF32Strided,
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
__device__ __forceinline__ uint32_t load_global_nc_u32(
        const uint32_t* pointer) {
    uint32_t value;
    asm volatile("ld.global.nc.u32 %0, [%1];"
                 : "=r"(value)
                 : "l"(pointer)
                 : "memory");
    return value;
}

__device__ __forceinline__ uint32_t load_global_u32(
        const uint32_t* pointer) {
    uint32_t value;
    asm volatile("ld.global.u32 %0, [%1];"
                 : "=r"(value)
                 : "l"(pointer)
                 : "memory");
    return value;
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

struct Float4Registers {
    float x, y, z, w;
};

__device__ __forceinline__ Float4Registers load_global_v4_f32(
        const float* pointer) {
    Float4Registers result{};
    asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(result.x), "=f"(result.y),
                   "=f"(result.z), "=f"(result.w)
                 : "l"(pointer)
                 : "memory");
    return result;
}

__device__ __forceinline__ float load_global_f32(const float* pointer) {
    float value;
    asm volatile("ld.global.f32 %0, [%1];"
                 : "=f"(value)
                 : "l"(pointer)
                 : "memory");
    return value;
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

__device__ __forceinline__ int select_rowset(AccessPattern pattern,
                                              int iteration,
                                              int block,
                                              int mask) {
    if (pattern == AccessPattern::kLocal) return block & mask;
    if (pattern == AccessPattern::kRandom) {
        uint32_t value = static_cast<uint32_t>(iteration + 1) * 0x85ebca6bu +
                         static_cast<uint32_t>(block + 1) * 0x9e3779b9u;
        value ^= value >> 16;
        value *= 0x7feb352du;
        value ^= value >> 15;
        return static_cast<int>(value) & mask;
    }
    return (iteration * gridDim.x + block) & mask;
}
template <Variant V, bool Target = true>
__global__ void global_load_i32_cached_kernel(const uint32_t* input,
                                               uint64_t* cycles,
                                               uint32_t* sinks,
                                               int iterations,
                                               int working_entries,
                                               int issuers,
                                               AccessPattern pattern) {
    static_assert(V == Variant::kGlobalLoadI32Cached ||
                  V == Variant::kGlobalLoadI32Ordinary);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    uint64_t start = 0;
    if (threadIdx.x == 0) start = read_clock64();
    __syncthreads();

    uint32_t checksum = static_cast<uint32_t>(threadIdx.x + 1);
    const int mask = working_entries - 1;
    const int local_mask = min(mask, 255);
#pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (threadIdx.x < issuers) {
            const int lane_item = static_cast<int>(
                blockIdx.x * blockDim.x + threadIdx.x);
            const int index = select_power_of_two_index(
                pattern, iteration, lane_item, mask, local_mask);
            const uint32_t* pointer = input + index;
            const uint32_t value = [&] {
                if constexpr (!Target) {
                    asm volatile("" : : "l"(pointer) : "memory");
                    return static_cast<uint32_t>(index + 1);
                } else if constexpr (V == Variant::kGlobalLoadI32Cached) {
                    return load_global_nc_u32(pointer);
                } else {
                    return load_global_u32(pointer);
                }
            }();
            checksum = (checksum << 7) ^ (checksum >> 3) ^ value;
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        cycles[blockIdx.x] = read_clock64() - start;
        sinks[blockIdx.x] = checksum == 0 ? 1 : checksum;
    }
#else
    if (threadIdx.x == 0) {
        cycles[blockIdx.x] = 0;
        sinks[blockIdx.x] = 0;
    }
#endif
}

template <Variant V, bool Target = true>
__global__ void global_load_record_32b_kernel(
        const Record32B* input,
        uint64_t* cycles,
        uint32_t* sinks,
        int iterations,
        int working_records,
        int issuers,
        AccessPattern pattern) {
    static_assert(V == Variant::kGlobalLoadRecord32B);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    uint64_t start = 0;
    if (threadIdx.x == 0) start = read_clock64();
    __syncthreads();

    uint32_t checksum = static_cast<uint32_t>(threadIdx.x + 1);
    const int mask = working_records - 1;
    const int local_mask = min(mask, 31);
#pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (threadIdx.x < issuers) {
            const int lane_item = static_cast<int>(
                blockIdx.x * blockDim.x + threadIdx.x);
            const int index = select_power_of_two_index(
                pattern, iteration, lane_item, mask, local_mask);
            const Record32B* pointer = input + index;
            RegisterRecord record{};
            if constexpr (Target) {
                record = load_global_record_32b(pointer);
            } else {
                asm volatile("" : : "l"(pointer) : "memory");
#pragma unroll
                for (int word = 0; word < 8; ++word) {
                    record.words[word] =
                        static_cast<uint32_t>(index * 17 + word + 1);
                }
            }
#pragma unroll
            for (int word = 0; word < 8; ++word) {
                checksum = (checksum << 3) ^ (checksum >> 5) ^
                           record.words[word];
            }
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        cycles[blockIdx.x] = read_clock64() - start;
        sinks[blockIdx.x] = checksum == 0 ? 1 : checksum;
    }
#else
    if (threadIdx.x == 0) {
        cycles[blockIdx.x] = 0;
        sinks[blockIdx.x] = 0;
    }
#endif
}

template <Variant V, bool Target = true>
__global__ void global_load_v4_f32_kernel(
        const float* input,
        uint64_t* cycles,
        uint32_t* sinks,
        int iterations,
        int rowsets,
        int segments,
        int active_warps,
        int vectors_per_thread,
        AccessPattern pattern) {
    static_assert(V == Variant::kGlobalLoadV4F32);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    constexpr int kHeadDimension = 512;
    constexpr int kHeads = 8;
    constexpr int kSplitStride = kHeads * kHeadDimension;
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int rowset_stride = segments * kSplitStride;
    const int rowset_mask = rowsets - 1;

    uint64_t start = 0;
    if (threadIdx.x == 0) start = read_clock64();
    __syncthreads();

    uint32_t checksum = static_cast<uint32_t>(threadIdx.x + 1);
#pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const int rowset = select_rowset(
            pattern, iteration, static_cast<int>(blockIdx.x), rowset_mask);
        if (warp < active_warps) {
            const float* head_base = input +
                static_cast<int64_t>(rowset) * rowset_stride +
                warp * kHeadDimension;
#pragma unroll 1
            for (int split = 0; split < segments; ++split) {
#pragma unroll 1
                for (int vector = 0; vector < vectors_per_thread; ++vector) {
                    const float* pointer =
                        head_base + split * kSplitStride + lane * 4 +
                        vector * 128;
                    Float4Registers values{};
                    if constexpr (Target) {
                        values = load_global_v4_f32(pointer);
                    } else {
                        asm volatile("" : : "l"(pointer) : "memory");
                        const float synthetic = static_cast<float>(
                            rowset + split + vector + lane + 1);
                        values = {synthetic, synthetic + 1.0f,
                                  synthetic + 2.0f, synthetic + 3.0f};
                    }
                    checksum ^= __float_as_uint(values.x) +
                                (__float_as_uint(values.y) << 1) +
                                (__float_as_uint(values.z) << 2) +
                                (__float_as_uint(values.w) << 3);
                }
            }
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        cycles[blockIdx.x] = read_clock64() - start;
        sinks[blockIdx.x] = checksum == 0 ? 1 : checksum;
    }
#else
    if (threadIdx.x == 0) {
        cycles[blockIdx.x] = 0;
        sinks[blockIdx.x] = 0;
    }
#endif
}

template <Variant V, bool Target = true>
__global__ void global_load_f32_strided_kernel(
        const float* input,
        uint64_t* cycles,
        uint32_t* sinks,
        int iterations,
        int rowsets,
        int segments,
        int split_stride,
        int active_warps,
        AccessPattern pattern) {
    static_assert(V == Variant::kGlobalLoadF32Strided);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int rowset_stride = segments * split_stride;
    const int rowset_mask = rowsets - 1;

    uint64_t start = 0;
    if (threadIdx.x == 0) start = read_clock64();
    __syncthreads();

    uint32_t checksum = static_cast<uint32_t>(threadIdx.x + 1);
#pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const int rowset = select_rowset(
            pattern, iteration, static_cast<int>(blockIdx.x), rowset_mask);
        if (warp < active_warps) {
            const float* rowset_base = input +
                static_cast<int64_t>(rowset) * rowset_stride;
#pragma unroll 1
            for (int split = lane; split < segments; split += 32) {
                const float* pointer =
                    rowset_base + split * split_stride + warp;
                float value;
                if constexpr (Target) {
                    value = load_global_f32(pointer);
                } else {
                    asm volatile("" : : "l"(pointer) : "memory");
                    value = static_cast<float>(rowset + split + warp + 1);
                }
                checksum = (checksum << 5) ^ (checksum >> 2) ^
                           __float_as_uint(value);
            }
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        cycles[blockIdx.x] = read_clock64() - start;
        sinks[blockIdx.x] = checksum == 0 ? 1 : checksum;
    }
#else
    if (threadIdx.x == 0) {
        cycles[blockIdx.x] = 0;
        sinks[blockIdx.x] = 0;
    }
#endif
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
inline int run_global_load_i32_cached(int argc, char** argv) {
    static_assert(V == Variant::kGlobalLoadI32Cached ||
                  V == Variant::kGlobalLoadI32Ordinary);
    constexpr bool kCached = V == Variant::kGlobalLoadI32Cached;
    const Args args(argc, argv);
    args.require_only({"iters", "warmup", "samples", "blocks", "device",
                       "peak", "threads", "issuers", "pattern",
                       "working-set-entries"});
    const CommonOptions options = parse_common_options(args, 4096);
    const cudaDeviceProp properties = require_sm90(options.device);
    const int threads = require_warp_multiple(args, "threads", 256);
    const int issuers = args.get_int("issuers", threads, 1, threads);
    const std::string pattern_name = args.get_string("pattern", "broadcast");
    const AccessPattern pattern = parse_access_pattern(pattern_name, true);
    const int working_entries = args.get_int(
        "working-set-entries", 32768, 32, 1 << 26);
    require_power_of_two("--working-set-entries", working_entries);
    const int blocks = resolve_blocks(options.blocks, properties, 4);

    std::vector<uint32_t> host_input(working_entries);
    for (int index = 0; index < working_entries; ++index) {
        host_input[index] = static_cast<uint32_t>(index * 17 + 1);
    }
    DeviceBuffer<uint32_t> input(host_input.size());
    copy_host_to_device(input, host_input);
    DeviceBuffer<uint64_t> latency_target_cycles(1);
    DeviceBuffer<uint64_t> latency_baseline_cycles(1);
    DeviceBuffer<uint64_t> throughput_cycles(blocks);
    DeviceBuffer<uint32_t> latency_target_sinks(1);
    DeviceBuffer<uint32_t> latency_baseline_sinks(1);
    DeviceBuffer<uint32_t> throughput_sinks(blocks);
    const TimingResult timing = measure_atomic(
        options, latency_target_cycles.data(), latency_baseline_cycles.data(),
        [&] {
            global_load_i32_cached_kernel<V, true><<<1, threads>>>(
                input.data(), latency_target_cycles.data(),
                latency_target_sinks.data(),
                options.iters, working_entries, issuers, pattern);
            global_load_i32_cached_kernel<V, false><<<1, threads>>>(
                input.data(), latency_baseline_cycles.data(),
                latency_baseline_sinks.data(),
                options.iters, working_entries, issuers, pattern);
        },
        [&] {
            global_load_i32_cached_kernel<V, true><<<blocks, threads>>>(
                input.data(), throughput_cycles.data(), throughput_sinks.data(),
                options.iters, working_entries, issuers, pattern);
        });
    require_nonzero_sinks(latency_target_sinks, "global scalar load target");
    require_nonzero_sinks(latency_baseline_sinks,
                          "global scalar load baseline");
    require_nonzero_sinks(throughput_sinks, "global ld.nc.u32");

    const double source_ops_per_round = static_cast<double>(issuers);
    const double total_source_ops = static_cast<double>(blocks) * options.iters *
                                    source_ops_per_round;
    const double requested_bytes = total_source_ops * sizeof(uint32_t);
    JsonObject params;
    add_common_params(params, properties, options, blocks);
    params.add("threads", threads)
        .add("issuers", issuers)
        .add("pattern", pattern_name)
        .add("working_set_entries", working_entries)
        .add("working_set_bytes",
             static_cast<uint64_t>(working_entries) * sizeof(uint32_t))
        .add("access_class", kCached
             ? "readonly scalar metadata"
             : "ordinary scalar")
        .add("ptx", kCached ? "ld.global.nc.u32" : "ld.global.u32")
        .add("clock_baseline",
             "matched address/pattern/checksum/control kernel without LDG")
        .add("throughput_protocol", "CUDA-event target kernel only")
        .add("correct", true);
    print_memory_result(
        kCached ? "ld_global_nc_u32"
                : "ld_global_u32",
        params,
        timing.cycles_per_round,
        kCached
            ? "target ld.global.nc.u32 loop minus matched address, pattern, "
              "checksum, loop-control, and CTA-convergence baseline"
            : "target ld.global.u32 loop minus matched address, pattern, "
              "checksum, loop-control, and CTA-convergence baseline",
        source_ops_per_round, timing.target_cycles_per_round,
        timing.baseline_cycles_per_round, total_source_ops, requested_bytes,
        timing.elapsed_ms,
        kCached ? "ld.global.nc.u32" : "ld.global.u32",
        "global", options.peak, true,
        timing.latency_samples, timing.target_samples, timing.baseline_samples,
        timing.event_ms_samples);
    return 0;
}

template <Variant V>
inline int run_global_load_record_32b(int argc, char** argv) {
    static_assert(V == Variant::kGlobalLoadRecord32B);
    const Args args(argc, argv);
    args.require_only({"iters", "warmup", "samples", "blocks", "device",
                       "peak", "threads", "issuers", "pattern",
                       "working-set-records"});
    const CommonOptions options = parse_common_options(args, 4096);
    const cudaDeviceProp properties = require_sm90(options.device);
    const int threads = require_warp_multiple(args, "threads", 256);
    const int issuers = args.get_int("issuers", threads, 1, threads);
    const std::string pattern_name = args.get_string("pattern", "broadcast");
    const AccessPattern pattern = parse_access_pattern(pattern_name, true);
    const int working_records = args.get_int(
        "working-set-records", 4096, 32, 1 << 22);
    require_power_of_two("--working-set-records", working_records);
    const int blocks = resolve_blocks(options.blocks, properties, 4);

    std::vector<Record32B> host_input(working_records);
    for (int record = 0; record < working_records; ++record) {
        for (int word = 0; word < 8; ++word) {
            host_input[record].words[word] =
                static_cast<uint32_t>(record * 17 + word + 1);
        }
    }
    DeviceBuffer<Record32B> input(host_input.size());
    copy_host_to_device(input, host_input);
    DeviceBuffer<uint64_t> latency_target_cycles(1);
    DeviceBuffer<uint64_t> latency_baseline_cycles(1);
    DeviceBuffer<uint64_t> throughput_cycles(blocks);
    DeviceBuffer<uint32_t> latency_target_sinks(1);
    DeviceBuffer<uint32_t> latency_baseline_sinks(1);
    DeviceBuffer<uint32_t> throughput_sinks(blocks);
    const TimingResult timing = measure_atomic(
        options, latency_target_cycles.data(), latency_baseline_cycles.data(),
        [&] {
            global_load_record_32b_kernel<V, true><<<1, threads>>>(
                input.data(), latency_target_cycles.data(),
                latency_target_sinks.data(),
                options.iters, working_records, issuers, pattern);
            global_load_record_32b_kernel<V, false><<<1, threads>>>(
                input.data(), latency_baseline_cycles.data(),
                latency_baseline_sinks.data(),
                options.iters, working_records, issuers, pattern);
        },
        [&] {
            global_load_record_32b_kernel<V, true><<<blocks, threads>>>(
                input.data(), throughput_cycles.data(), throughput_sinks.data(),
                options.iters, working_records, issuers, pattern);
        });
    require_nonzero_sinks(latency_target_sinks, "record load target");
    require_nonzero_sinks(latency_baseline_sinks, "record load baseline");
    require_nonzero_sinks(throughput_sinks, "generic record load");

    const double source_ops_per_round = static_cast<double>(issuers) * 2.0;
    const double total_source_ops = static_cast<double>(blocks) * options.iters *
                                    source_ops_per_round;
    const double requested_bytes = static_cast<double>(blocks) * options.iters *
                                   issuers * sizeof(Record32B);
    JsonObject params;
    add_common_params(params, properties, options, blocks);
    params.add("threads", threads)
        .add("issuers", issuers)
        .add("pattern", pattern_name)
        .add("working_set_records", working_records)
        .add("working_set_bytes",
             static_cast<uint64_t>(working_records) * sizeof(Record32B))
        .add("record_bytes", sizeof(Record32B))
        .add("ptx", "2x ld.global.v4.u32")
        .add("clock_baseline",
             "matched address/pattern/record/checksum/control kernel without LDG")
        .add("throughput_protocol", "CUDA-event target kernel only")
        .add("correct", true);
    print_memory_result(
        "ld_global_v4_u32_32b", params,
        timing.cycles_per_round,
        "target two-load record loop minus matched address, pattern, record, "
        "checksum, loop-control, and CTA-convergence baseline",
        source_ops_per_round, timing.target_cycles_per_round,
        timing.baseline_cycles_per_round, total_source_ops, requested_bytes,
        timing.elapsed_ms, "ld.global.v4.u32", "global", options.peak, true,
        timing.latency_samples, timing.target_samples, timing.baseline_samples,
        timing.event_ms_samples);
    return 0;
}

template <Variant V>
inline int run_global_load_v4_f32(int argc, char** argv) {
    static_assert(V == Variant::kGlobalLoadV4F32);
    const Args args(argc, argv);
    args.require_only({"iters", "warmup", "samples", "blocks", "device",
                       "peak", "segments", "rowsets", "warps",
                       "vectors-per-thread", "pattern"});
    const CommonOptions options = parse_common_options(args, 64);
    const cudaDeviceProp properties = require_sm90(options.device);
    const int segments = args.get_int("segments", 8, 1, 160);
    const int rowsets = args.get_int("rowsets", 64, 1, 1 << 16);
    require_power_of_two("--rowsets", rowsets);
    const int active_warps = args.get_int("warps", 8, 1, 8);
    const int vectors_per_thread = args.get_int(
        "vectors-per-thread", 4, 1, 4);
    const std::string pattern_name = args.get_string("pattern", "sequential");
    const AccessPattern pattern = parse_access_pattern(pattern_name, false);
    const int blocks = resolve_blocks(options.blocks, properties, 2);
    constexpr std::size_t kSplitElements = 8 * 512;
    const std::size_t input_elements = static_cast<std::size_t>(rowsets) *
                                       segments * kSplitElements;
    if (input_elements > (std::size_t{1} << 32)) {
        throw std::invalid_argument(
            "vector-data working set exceeds the 16-GiB benchmark limit");
    }

    DeviceBuffer<float> input(input_elements);
    CUDA_CHECK(cudaMemset(input.data(), 0x3f, input_elements * sizeof(float)));
    DeviceBuffer<uint64_t> latency_target_cycles(1);
    DeviceBuffer<uint64_t> latency_baseline_cycles(1);
    DeviceBuffer<uint64_t> throughput_cycles(blocks);
    DeviceBuffer<uint32_t> latency_target_sinks(1);
    DeviceBuffer<uint32_t> latency_baseline_sinks(1);
    DeviceBuffer<uint32_t> throughput_sinks(blocks);
    const TimingResult timing = measure_atomic(
        options, latency_target_cycles.data(), latency_baseline_cycles.data(),
        [&] {
            global_load_v4_f32_kernel<V, true><<<1, 256>>>(
                input.data(), latency_target_cycles.data(),
                latency_target_sinks.data(),
                options.iters, rowsets, segments, active_warps,
                vectors_per_thread, pattern);
            global_load_v4_f32_kernel<V, false><<<1, 256>>>(
                input.data(), latency_baseline_cycles.data(),
                latency_baseline_sinks.data(),
                options.iters, rowsets, segments, active_warps,
                vectors_per_thread, pattern);
        },
        [&] {
            global_load_v4_f32_kernel<V, true><<<blocks, 256>>>(
                input.data(), throughput_cycles.data(), throughput_sinks.data(),
                options.iters, rowsets, segments, active_warps,
                vectors_per_thread, pattern);
        });
    require_nonzero_sinks(latency_target_sinks, "v4.f32 load target");
    require_nonzero_sinks(latency_baseline_sinks, "v4.f32 load baseline");
    require_nonzero_sinks(throughput_sinks, "segmented float4 vector-data load");

    const double source_ops_per_round = static_cast<double>(active_warps) * 32.0 *
                                        segments * vectors_per_thread;
    const double total_source_ops = static_cast<double>(blocks) * options.iters *
                                    source_ops_per_round;
    const double requested_bytes = total_source_ops * 16.0;
    JsonObject params;
    add_common_params(params, properties, options, blocks);
    params.add("segments", segments)
        .add("rowsets", rowsets)
        .add("warps", active_warps)
        .add("vectors_per_thread", vectors_per_thread)
        .add("pattern", pattern_name)
        .add("working_set_bytes",
             static_cast<uint64_t>(input_elements * sizeof(float)))
        .add("split_stride_elements", static_cast<int>(kSplitElements))
        .add("ptx", "ld.global.v4.f32")
        .add("clock_baseline",
             "matched address/segment/value/checksum/control kernel without LDG")
        .add("throughput_protocol", "CUDA-event target kernel only")
        .add("correct", true);
    print_memory_result(
        "ld_global_v4_f32", params,
        timing.cycles_per_round,
        "target segmented v4.f32 loop minus matched address, segment, value, "
        "checksum, loop-control, and CTA-convergence baseline",
        source_ops_per_round, timing.target_cycles_per_round,
        timing.baseline_cycles_per_round, total_source_ops, requested_bytes,
        timing.elapsed_ms, "ld.global.v4.f32", "global", options.peak, true,
        timing.latency_samples, timing.target_samples, timing.baseline_samples,
        timing.event_ms_samples);
    return 0;
}

template <Variant V>
inline int run_global_load_f32_strided(int argc, char** argv) {
    static_assert(V == Variant::kGlobalLoadF32Strided);
    const Args args(argc, argv);
    args.require_only({"iters", "warmup", "samples", "blocks", "device",
                       "peak", "segments", "split-stride", "rowsets",
                       "warps", "pattern"});
    const CommonOptions options = parse_common_options(args, 256);
    const cudaDeviceProp properties = require_sm90(options.device);
    const int segments = args.get_int("segments", 32, 1, 160);
    const int split_stride = args.get_int("split-stride", 128, 8, 1 << 20);
    const int rowsets = args.get_int("rowsets", 1024, 1, 1 << 16);
    require_power_of_two("--rowsets", rowsets);
    const int active_warps = args.get_int("warps", 8, 1, 8);
    if (split_stride < active_warps) {
        throw std::invalid_argument("split-stride must cover active warps");
    }
    const std::string pattern_name = args.get_string("pattern", "sequential");
    const AccessPattern pattern = parse_access_pattern(pattern_name, false);
    const int blocks = resolve_blocks(options.blocks, properties, 4);
    const std::size_t input_elements = static_cast<std::size_t>(rowsets) *
                                       segments * split_stride;
    if (input_elements > (std::size_t{1} << 32)) {
        throw std::invalid_argument(
            "LSE working set exceeds the 16-GiB benchmark limit");
    }

    DeviceBuffer<float> input(input_elements);
    CUDA_CHECK(cudaMemset(input.data(), 0x3f, input_elements * sizeof(float)));
    DeviceBuffer<uint64_t> latency_target_cycles(1);
    DeviceBuffer<uint64_t> latency_baseline_cycles(1);
    DeviceBuffer<uint64_t> throughput_cycles(blocks);
    DeviceBuffer<uint32_t> latency_target_sinks(1);
    DeviceBuffer<uint32_t> latency_baseline_sinks(1);
    DeviceBuffer<uint32_t> throughput_sinks(blocks);
    const TimingResult timing = measure_atomic(
        options, latency_target_cycles.data(), latency_baseline_cycles.data(),
        [&] {
            global_load_f32_strided_kernel<V, true><<<1, 256>>>(
                input.data(), latency_target_cycles.data(),
                latency_target_sinks.data(),
                options.iters, rowsets, segments, split_stride, active_warps,
                pattern);
            global_load_f32_strided_kernel<V, false><<<1, 256>>>(
                input.data(), latency_baseline_cycles.data(),
                latency_baseline_sinks.data(),
                options.iters, rowsets, segments, split_stride, active_warps,
                pattern);
        },
        [&] {
            global_load_f32_strided_kernel<V, true><<<blocks, 256>>>(
                input.data(), throughput_cycles.data(), throughput_sinks.data(),
                options.iters, rowsets, segments, split_stride, active_warps,
                pattern);
        });
    require_nonzero_sinks(latency_target_sinks, "strided f32 load target");
    require_nonzero_sinks(latency_baseline_sinks,
                          "strided f32 load baseline");
    require_nonzero_sinks(throughput_sinks, "segmented strided f32 load");

    const double source_ops_per_round = static_cast<double>(active_warps) *
                                        segments;
    const double total_source_ops = static_cast<double>(blocks) * options.iters *
                                    source_ops_per_round;
    const double requested_bytes = total_source_ops * sizeof(float);
    JsonObject params;
    add_common_params(params, properties, options, blocks);
    params.add("segments", segments)
        .add("split_stride", split_stride)
        .add("rowsets", rowsets)
        .add("warps", active_warps)
        .add("pattern", pattern_name)
        .add("working_set_bytes",
             static_cast<uint64_t>(input_elements * sizeof(float)))
        .add("ptx", "ld.global.f32")
        .add("clock_baseline",
             "matched address/segment/value/checksum/control kernel without LDG")
        .add("throughput_protocol", "CUDA-event target kernel only")
        .add("correct", true);
    print_memory_result(
        "ld_global_f32_strided", params,
        timing.cycles_per_round,
        "target strided scalar-load loop minus matched address, segment, value, "
        "checksum, loop-control, and CTA-convergence baseline",
        source_ops_per_round, timing.target_cycles_per_round,
        timing.baseline_cycles_per_round, total_source_ops, requested_bytes,
        timing.elapsed_ms, "ld.global.f32", "global", options.peak, true,
        timing.latency_samples, timing.target_samples, timing.baseline_samples,
        timing.event_ms_samples);
    return 0;
}

}  // namespace microbench::memory_atomic
