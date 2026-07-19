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

#include "bench.hpp"
#include "tensor_map.hpp"

namespace microbench::memory_atomic {

enum class Variant {
    kSharedLoadU32Patterns,
    kSharedStoreU32Scalar,
    kSharedStoreU64Dense,
    kGlobalLoadI32Cached,
    kGlobalLoadI32Ordinary,
    kGlobalLoadSchedulerRecord32B,
    kGlobalLoadFloat4OAccum,
    kGlobalLoadF32LseStrided,
    kGlobalStoreF32Lse,
    kGlobalStoreSchedulerRecord32B,
    kGlobalStoreU32NumSplits,
    kGlobalStoreU64Output,
    kTensorMapPrefetchQkoRank4,
};

enum class AccessPattern : int {
    kLocal = 0,
    kSequential = 1,
    kRandom = 2,
    kBroadcast = 3,
};

enum class SharedLoadPattern : int {
    kUnique = 0,
    kQuadBroadcast = 1,
    kWarpBroadcast = 2,
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

__device__ __forceinline__ uint32_t load_shared_u32(const uint32_t* pointer) {
    uint32_t value;
    const uint32_t address = shared_address(pointer);
    asm volatile("ld.shared.u32 %0, [%1];"
                 : "=r"(value)
                 : "r"(address)
                 : "memory");
    return value;
}

__device__ __forceinline__ void store_shared_u32(uint32_t* pointer,
                                                  uint32_t value) {
    const uint32_t address = shared_address(pointer);
    asm volatile("st.shared.u32 [%0], %1;"
                 :
                 : "r"(address), "r"(value)
                 : "memory");
}

__device__ __forceinline__ void store_shared_u64(uint64_t* pointer,
                                                  uint64_t value) {
    const uint32_t address = shared_address(pointer);
    asm volatile("st.shared.u64 [%0], %1;"
                 :
                 : "r"(address), "l"(value)
                 : "memory");
}

__device__ __forceinline__ void store_shared_v2_u32(uint32_t* pointer,
                                                     uint32_t value0,
                                                     uint32_t value1) {
    const uint32_t address = shared_address(pointer);
    asm volatile("st.shared.v2.u32 [%0], {%1, %2};"
                 :
                 : "r"(address), "r"(value0), "r"(value1)
                 : "memory");
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

struct alignas(32) SchedulerRecord {
    uint32_t words[8];
};
static_assert(sizeof(SchedulerRecord) == 32);

struct RegisterRecord {
    uint32_t words[8];
};

__device__ __forceinline__ RegisterRecord load_global_record_32b(
        const SchedulerRecord* pointer) {
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
        SchedulerRecord* pointer,
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

__device__ __forceinline__ void prefetch_tensormap(const void* pointer) {
    const uint64_t address = reinterpret_cast<uint64_t>(pointer);
    asm volatile("prefetch.tensormap [%0];"
                 :
                 : "l"(address)
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

template <Variant V>
__global__ void shared_load_u32_kernel(uint64_t* cycles,
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
    if (threadIdx.x == 0) start = read_clock64();
    __syncthreads();

    uint32_t checksum = static_cast<uint32_t>(threadIdx.x + 1);
    const int mask = working_words - 1;
#pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        int group = static_cast<int>(threadIdx.x);
        if (pattern == SharedLoadPattern::kQuadBroadcast) group >>= 2;
        if (pattern == SharedLoadPattern::kWarpBroadcast) group >>= 5;
        const int index = (iteration * 32 + group) & mask;
        const uint32_t value = load_shared_u32(storage + index);
        checksum = (checksum << 5) ^ (checksum >> 2) ^ value;
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

template <Variant V>
__global__ void shared_store_u32_kernel(uint64_t* cycles,
                                        uint32_t* sinks,
                                        int iterations,
                                        int working_words,
                                        int producers,
                                        int topology) {
    static_assert(V == Variant::kSharedStoreU32Scalar);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    extern __shared__ __align__(128) uint32_t storage[];
    for (int index = threadIdx.x; index < working_words; index += blockDim.x) {
        storage[index] = 0;
    }
    __syncthreads();

    uint64_t start = 0;
    if (threadIdx.x == 0) start = read_clock64();
    __syncthreads();

    const int mask = working_words - 1;
#pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        int producer_index = static_cast<int>(threadIdx.x);
        bool participates = producer_index < producers;
        if (topology == 1) {
            participates = (threadIdx.x & 3) == 0;
            producer_index = static_cast<int>(threadIdx.x) / 4;
            participates &= producer_index < producers;
        } else if (topology == 2) {
            participates = (threadIdx.x & 31) == 0;
            producer_index = static_cast<int>(threadIdx.x) / 32;
            participates &= producer_index < producers;
        }
        if (participates) {
            const int index = (iteration * producers + producer_index) & mask;
            store_shared_u32(storage + index,
                             static_cast<uint32_t>(iteration + producer_index + 1));
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        cycles[blockIdx.x] = read_clock64() - start;
        const uint32_t value = load_shared_u32(storage);
        sinks[blockIdx.x] = value == 0 ? 1 : value;
    }
#else
    if (threadIdx.x == 0) {
        cycles[blockIdx.x] = 0;
        sinks[blockIdx.x] = 0;
    }
#endif
}

__host__ __device__ constexpr uint32_t dense_v_int64_sw128_byte_address(
        int head_group,
        int token) {
    // CUTE layout, expressed without a CUTE dependency:
    // Sw<3,4,3> o ((_16,_4),(_8,_8)):((_1,_1024),(_16,_128)).
    const uint32_t logical_element = static_cast<uint32_t>(
        (head_group & 15) + (head_group >> 4) * 1024 +
        (token & 7) * 16 + (token >> 3) * 128);
    const uint32_t physical_element =
        logical_element ^ ((logical_element & 0x380u) >> 3);
    return physical_element * static_cast<uint32_t>(sizeof(uint64_t));
}
static_assert(dense_v_int64_sw128_byte_address(0, 0) == 0);
static_assert(dense_v_int64_sw128_byte_address(0, 8) == 1152);
static_assert(dense_v_int64_sw128_byte_address(16, 0) == 8192);
static_assert(dense_v_int64_sw128_byte_address(63, 63) == 31864);

template <Variant V>
__global__ void shared_store_u64_dense_kernel(uint64_t* cycles,
                                              uint32_t* sinks,
                                              int iterations,
                                              int role,
                                              int warpgroups,
                                              int stores_per_thread,
                                              int invalid_tokens) {
    static_assert(V == Variant::kSharedStoreU64Dense);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    extern __shared__ __align__(128) unsigned char storage[];
    constexpr int kStride520Bytes = 64 * 520 * 4;
    constexpr int kTailBytes = 64 * 64 * 8;
    const int storage_bytes = role == 0 ? kStride520Bytes : kTailBytes;
    for (int index = threadIdx.x; index < storage_bytes; index += blockDim.x) {
        storage[index] = 0;
    }
    __syncthreads();

    uint64_t start = 0;
    if (threadIdx.x == 0) start = read_clock64();
    __syncthreads();

#pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (role == 0) {
            const int warpgroup = static_cast<int>(threadIdx.x) / 128;
            const int lane = static_cast<int>(threadIdx.x) & 127;
            if (warpgroup < warpgroups) {
#pragma unroll 1
                for (int operation = 0; operation < stores_per_thread;
                     ++operation) {
                    const int accumulator_index = operation * 2;
                    const int row = (lane / 32) * 16 + (lane % 32) / 4 +
                                    ((accumulator_index & 3) >= 2 ? 8 : 0);
                    const int column = warpgroup * 256 + (lane & 3) * 2 +
                                       (accumulator_index / 4) * 8;
                    auto* destination = reinterpret_cast<uint32_t*>(storage) +
                                        row * 520 + column;
                    const uint32_t value = static_cast<uint32_t>(
                        iteration + operation + threadIdx.x + 1);
                    store_shared_v2_u32(destination, value, value ^ 0x5a5a5a5au);
                }
            }
        } else if (threadIdx.x < 128) {
            const int lane = static_cast<int>(threadIdx.x);
            const int first_invalid = 64 - invalid_tokens;
            const int head_group = lane & 63;
#pragma unroll 1
            for (int token = first_invalid + lane / 64; token < 64; token += 2) {
                const uint32_t physical_address =
                    dense_v_int64_sw128_byte_address(head_group, token);
                auto* destination = reinterpret_cast<uint64_t*>(
                    storage + physical_address);
                store_shared_u64(destination, 0ull);
            }
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        cycles[blockIdx.x] = read_clock64() - start;
        sinks[blockIdx.x] = static_cast<uint32_t>(iterations + role + 1);
    }
#else
    if (threadIdx.x == 0) {
        cycles[blockIdx.x] = 0;
        sinks[blockIdx.x] = 0;
    }
#endif
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

template <Variant V>
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
            const uint32_t value = [&] {
                if constexpr (V == Variant::kGlobalLoadI32Cached) {
                    return load_global_nc_u32(input + index);
                } else {
                    return load_global_u32(input + index);
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

template <Variant V>
__global__ void global_load_scheduler_record_kernel(
        const SchedulerRecord* input,
        uint64_t* cycles,
        uint32_t* sinks,
        int iterations,
        int working_records,
        int issuers,
        AccessPattern pattern) {
    static_assert(V == Variant::kGlobalLoadSchedulerRecord32B);
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
            const RegisterRecord record = load_global_record_32b(input + index);
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

template <Variant V>
__global__ void global_load_float4_oaccum_kernel(
        const float* input,
        uint64_t* cycles,
        uint32_t* sinks,
        int iterations,
        int rowsets,
        int num_splits,
        int active_warps,
        int vectors_per_thread,
        AccessPattern pattern) {
    static_assert(V == Variant::kGlobalLoadFloat4OAccum);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    constexpr int kHeadDimension = 512;
    constexpr int kHeads = 8;
    constexpr int kSplitStride = kHeads * kHeadDimension;
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int rowset_stride = num_splits * kSplitStride;
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
            for (int split = 0; split < num_splits; ++split) {
#pragma unroll 1
                for (int vector = 0; vector < vectors_per_thread; ++vector) {
                    const Float4Registers values = load_global_v4_f32(
                        head_base + split * kSplitStride + lane * 4 +
                        vector * 128);
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

template <Variant V>
__global__ void global_load_f32_lse_strided_kernel(
        const float* input,
        uint64_t* cycles,
        uint32_t* sinks,
        int iterations,
        int rowsets,
        int num_splits,
        int split_stride,
        int active_warps,
        AccessPattern pattern) {
    static_assert(V == Variant::kGlobalLoadF32LseStrided);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int rowset_stride = num_splits * split_stride;
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
            for (int split = lane; split < num_splits; split += 32) {
                const float value = load_global_f32(
                    rowset_base + split * split_stride + warp);
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

template <Variant V>
__global__ void global_store_f32_lse_kernel(float* output,
                                             uint64_t* cycles,
                                             uint32_t* sinks,
                                             int iterations,
                                             int records_per_block,
                                             int role,
                                             AccessPattern pattern) {
    static_assert(V == Variant::kGlobalStoreF32Lse);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    const int values_per_record = role == 0 ? 64 : 8;
    const int producer = role == 0
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
            store_global_f32(
                block_base + record * values_per_record + producer,
                static_cast<float>(iteration + producer + 1));
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        cycles[blockIdx.x] = read_clock64() - start;
        sinks[blockIdx.x] = static_cast<uint32_t>(iterations + role + 1);
    }
#else
    if (threadIdx.x == 0) {
        cycles[blockIdx.x] = 0;
        sinks[blockIdx.x] = 0;
    }
#endif
}

template <Variant V>
__global__ void global_store_scheduler_record_kernel(
        SchedulerRecord* output,
        uint64_t* cycles,
        uint32_t* sinks,
        int iterations,
        int records_per_block,
        AccessPattern pattern) {
    static_assert(V == Variant::kGlobalStoreSchedulerRecord32B);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    SchedulerRecord* block_base = output +
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
            store_global_record_32b(block_base + record, value);
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

template <Variant V>
__global__ void global_store_u32_num_splits_kernel(
        uint32_t* output,
        uint64_t* cycles,
        uint32_t* sinks,
        int iterations,
        int records_per_block,
        int producers,
        AccessPattern pattern) {
    static_assert(V == Variant::kGlobalStoreU32NumSplits);
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
            store_global_u32(
                block_base + record * producers + threadIdx.x,
                static_cast<uint32_t>(iteration + threadIdx.x + 1));
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

template <Variant V>
__global__ void global_store_u64_output_kernel(
        uint64_t* output,
        uint64_t* cycles,
        uint32_t* sinks,
        int iterations,
        int records_per_block,
        int active_warps,
        int vectors_per_thread,
        AccessPattern pattern,
        uint64_t value_pattern) {
    static_assert(V == Variant::kGlobalStoreU64Output);
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
                store_global_u64(
                    block_base + record * values_per_record + index,
                    value_pattern ^ static_cast<uint64_t>(iteration + index));
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

template <Variant V>
__global__ void tensormap_prefetch_kernel(
        __grid_constant__ const CUtensorMap q_map,
        __grid_constant__ const CUtensorMap k_map,
        __grid_constant__ const CUtensorMap o_map,
        uint64_t* cycles,
        uint32_t* sinks,
        int iterations,
        int mode) {
    static_assert(V == Variant::kTensorMapPrefetchQkoRank4);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    if (threadIdx.x == 0) {
        const uint64_t start = read_clock64();
#pragma unroll 1
        for (int iteration = 0; iteration < iterations; ++iteration) {
            if (mode == 0 || mode == 3) prefetch_tensormap(&q_map);
            if (mode == 1 || mode == 3) prefetch_tensormap(&k_map);
            if (mode == 2 || mode == 3) prefetch_tensormap(&o_map);
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
    std::vector<double> event_ms_samples;
};

template <typename LatencyLaunch, typename ThroughputLaunch>
inline TimingResult measure_atomic(const CommonOptions& options,
                                   uint64_t* latency_cycles,
                                   LatencyLaunch&& latency_launch,
                                   ThroughputLaunch&& throughput_launch) {
    const auto cycle_samples = measure_clock_cycles(
        options.warmup, options.samples, latency_cycles,
        std::forward<LatencyLaunch>(latency_launch));
    const auto event_samples = measure_event_ms(
        options.warmup, options.samples,
        std::forward<ThroughputLaunch>(throughput_launch));
    std::vector<double> normalized_cycle_samples = cycle_samples;
    for (double& value : normalized_cycle_samples) {
        value /= options.iters;
    }
    return TimingResult{median(cycle_samples) / options.iters,
                        median(event_samples),
                        std::move(normalized_cycle_samples),
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
        shared_load_u32_kernel<V>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_bytes));
    DeviceBuffer<uint64_t> latency_cycles(1);
    DeviceBuffer<uint64_t> throughput_cycles(blocks);
    DeviceBuffer<uint32_t> latency_sinks(1);
    DeviceBuffer<uint32_t> throughput_sinks(blocks);
    const TimingResult timing = measure_atomic(
        options, latency_cycles.data(),
        [&] {
            shared_load_u32_kernel<V><<<1, threads, shared_bytes>>>(
                latency_cycles.data(), latency_sinks.data(), options.iters,
                working_words, pattern);
        },
        [&] {
            shared_load_u32_kernel<V><<<blocks, threads, shared_bytes>>>(
                throughput_cycles.data(), throughput_sinks.data(), options.iters,
                working_words, pattern);
        });
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
        .add("correct", true);
    print_memory_result(
        "dense_decode.shared_load_u32", params, timing.cycles_per_round,
        "one ld.shared.u32 per participating thread plus CTA convergence",
        source_ops_per_round, total_source_ops, requested_bytes,
        timing.elapsed_ms, "ld.shared.u32", "shared", options.peak, false,
        timing.latency_samples, timing.event_ms_samples);
    return 0;
}

template <Variant V>
inline int run_shared_store_u32(int argc, char** argv) {
    static_assert(V == Variant::kSharedStoreU32Scalar);
    const Args args(argc, argv);
    args.require_only({"iters", "warmup", "samples", "blocks", "device",
                       "peak", "threads", "producers", "topology",
                       "working-set-words"});
    const CommonOptions options = parse_common_options(args, 4096);
    const cudaDeviceProp properties = require_sm90(options.device);
    const int threads = require_warp_multiple(args, "threads", 256);
    const int producers = args.get_int("producers", 64, 1, threads);
    const std::string topology_name = args.get_string("topology", "contiguous");
    int topology = -1;
    if (topology_name == "contiguous") topology = 0;
    if (topology_name == "quad_leaders") topology = 1;
    if (topology_name == "warp_leaders") topology = 2;
    if (topology < 0) {
        throw std::invalid_argument(
            "--topology must be contiguous, quad_leaders, or warp_leaders");
    }
    const int maximum_producers = topology == 0
        ? threads
        : (topology == 1 ? threads / 4 : threads / 32);
    if (producers > maximum_producers) {
        throw std::invalid_argument(
            "--producers exceeds the selected topology capacity");
    }
    const int working_words = args.get_int(
        "working-set-words", 4096, 32, 1 << 16);
    require_power_of_two("--working-set-words", working_words);
    if (working_words < producers) {
        throw std::invalid_argument(
            "working-set-words must be at least producers");
    }
    const int shared_bytes = working_words * static_cast<int>(sizeof(uint32_t));
    if (shared_bytes > static_cast<int>(properties.sharedMemPerBlockOptin)) {
        throw std::invalid_argument(
            "working shared-store set exceeds sharedMemPerBlockOptin");
    }
    const int blocks = resolve_blocks(options.blocks, properties, 4);

    CUDA_CHECK(cudaFuncSetAttribute(
        shared_store_u32_kernel<V>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_bytes));
    DeviceBuffer<uint64_t> latency_cycles(1);
    DeviceBuffer<uint64_t> throughput_cycles(blocks);
    DeviceBuffer<uint32_t> latency_sinks(1);
    DeviceBuffer<uint32_t> throughput_sinks(blocks);
    const TimingResult timing = measure_atomic(
        options, latency_cycles.data(),
        [&] {
            shared_store_u32_kernel<V><<<1, threads, shared_bytes>>>(
                latency_cycles.data(), latency_sinks.data(), options.iters,
                working_words, producers, topology);
        },
        [&] {
            shared_store_u32_kernel<V><<<blocks, threads, shared_bytes>>>(
                throughput_cycles.data(), throughput_sinks.data(), options.iters,
                working_words, producers, topology);
        });
    require_nonzero_sinks(throughput_sinks, "shared st.u32");

    const double source_ops_per_round = static_cast<double>(producers);
    const double total_source_ops = static_cast<double>(blocks) * options.iters *
                                    source_ops_per_round;
    const double requested_bytes = total_source_ops * sizeof(uint32_t);
    JsonObject params;
    add_common_params(params, properties, options, blocks);
    params.add("threads", threads)
        .add("producers", producers)
        .add("topology", topology_name)
        .add("working_set_words", working_words)
        .add("working_set_bytes", shared_bytes)
        .add("roles", "sM/sScale/sL and scheduler scratch")
        .add("ptx", "st.shared.u32")
        .add("correct", true);
    print_memory_result(
        "dense_decode.shared_store_u32", params, timing.cycles_per_round,
        "one predicated st.shared.u32 per producer plus CTA convergence",
        source_ops_per_round, total_source_ops, requested_bytes,
        timing.elapsed_ms, "st.shared.u32", "shared", options.peak, false,
        timing.latency_samples, timing.event_ms_samples);
    return 0;
}

template <Variant V>
inline int run_shared_store_u64(int argc, char** argv) {
    static_assert(V == Variant::kSharedStoreU64Dense);
    const Args args(argc, argv);
    args.require_only({"iters", "warmup", "samples", "blocks", "device",
                       "peak", "role", "warpgroups", "stores-per-thread",
                       "invalid-tokens"});
    const CommonOptions options = parse_common_options(args, 1024);
    const cudaDeviceProp properties = require_sm90(options.device);
    const std::string role_name = args.get_string("role", "stride520");
    int role = -1;
    if (role_name == "stride520") role = 0;
    if (role_name == "tail_zero") role = 1;
    if (role < 0) {
        throw std::invalid_argument("--role must be stride520 or tail_zero");
    }
    const int warpgroups = args.get_int("warpgroups", 2, 1, 2);
    const int stores_per_thread = args.get_int(
        "stores-per-thread", 64, 1, 64);
    const int invalid_tokens = args.get_int("invalid-tokens", 8, 1, 64);
    const int threads = role == 0 ? warpgroups * 128 : 128;
    const int shared_bytes = role == 0 ? 64 * 520 * 4 : 64 * 64 * 8;
    if (shared_bytes > static_cast<int>(properties.sharedMemPerBlockOptin)) {
        throw std::runtime_error(
            "dense shared-store tile exceeds sharedMemPerBlockOptin");
    }
    const int blocks = resolve_blocks(options.blocks, properties, 1);

    CUDA_CHECK(cudaFuncSetAttribute(
        shared_store_u64_dense_kernel<V>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
    DeviceBuffer<uint64_t> latency_cycles(1);
    DeviceBuffer<uint64_t> throughput_cycles(blocks);
    DeviceBuffer<uint32_t> latency_sinks(1);
    DeviceBuffer<uint32_t> throughput_sinks(blocks);
    const TimingResult timing = measure_atomic(
        options, latency_cycles.data(),
        [&] {
            shared_store_u64_dense_kernel<V><<<1, threads, shared_bytes>>>(
                latency_cycles.data(), latency_sinks.data(), options.iters, role,
                warpgroups, stores_per_thread, invalid_tokens);
        },
        [&] {
            shared_store_u64_dense_kernel<V><<<blocks, threads, shared_bytes>>>(
                throughput_cycles.data(), throughput_sinks.data(), options.iters,
                role, warpgroups, stores_per_thread, invalid_tokens);
        });
    require_nonzero_sinks(throughput_sinks, "shared st.u64/v2.u32");

    const double source_ops_per_round = role == 0
        ? static_cast<double>(threads) * stores_per_thread
        : static_cast<double>(invalid_tokens) * 64.0;
    const double total_source_ops = static_cast<double>(blocks) * options.iters *
                                    source_ops_per_round;
    const double requested_bytes = total_source_ops * sizeof(uint64_t);
    JsonObject params;
    add_common_params(params, properties, options, blocks);
    params.add("role", role_name)
        .add("warpgroups", warpgroups)
        .add("stores_per_thread", stores_per_thread)
        .add("invalid_tokens", invalid_tokens)
        .add("threads", threads)
        .add("shared_bytes", shared_bytes)
        .add("ptx", role == 0 ? "st.shared.v2.u32" : "st.shared.u64")
        .add("correct", true);
    print_memory_result(
        "dense_decode.shared_store_u64_dense", params,
        timing.cycles_per_round,
        role == 0
            ? "exact 64x520 split-O float2 staging issue loop"
            : "exact 64-head-group SW128 tail-zero issue loop",
        source_ops_per_round, total_source_ops, requested_bytes,
        timing.elapsed_ms,
        role == 0 ? "st.shared.v2.u32" : "st.shared.u64",
        "shared", options.peak, false, timing.latency_samples,
        timing.event_ms_samples);
    return 0;
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
    DeviceBuffer<uint64_t> latency_cycles(1);
    DeviceBuffer<uint64_t> throughput_cycles(blocks);
    DeviceBuffer<uint32_t> latency_sinks(1);
    DeviceBuffer<uint32_t> throughput_sinks(blocks);
    const TimingResult timing = measure_atomic(
        options, latency_cycles.data(),
        [&] {
            global_load_i32_cached_kernel<V><<<1, threads>>>(
                input.data(), latency_cycles.data(), latency_sinks.data(),
                options.iters, working_entries, issuers, pattern);
        },
        [&] {
            global_load_i32_cached_kernel<V><<<blocks, threads>>>(
                input.data(), throughput_cycles.data(), throughput_sinks.data(),
                options.iters, working_entries, issuers, pattern);
        });
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
        .add("roles", kCached
             ? "seqlens_k/block_table/readonly num_splits"
             : "PDL-sensitive main-kernel num_splits")
        .add("ptx", kCached ? "ld.global.nc.u32" : "ld.global.u32")
        .add("correct", true);
    print_memory_result(
        kCached ? "dense_decode.global_load_i32_cached"
                : "dense_decode.global_load_i32_ordinary",
        params,
        timing.cycles_per_round,
        kCached
            ? "one ld.global.nc.u32 per active issuer; address generation included"
            : "one ordinary ld.global.u32 per active issuer; address generation included",
        source_ops_per_round, total_source_ops, requested_bytes,
        timing.elapsed_ms,
        kCached ? "ld.global.nc.u32" : "ld.global.u32",
        "global", options.peak, true,
        timing.latency_samples, timing.event_ms_samples);
    return 0;
}

template <Variant V>
inline int run_global_load_scheduler_record(int argc, char** argv) {
    static_assert(V == Variant::kGlobalLoadSchedulerRecord32B);
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

    std::vector<SchedulerRecord> host_input(working_records);
    for (int record = 0; record < working_records; ++record) {
        for (int word = 0; word < 8; ++word) {
            host_input[record].words[word] =
                static_cast<uint32_t>(record * 17 + word + 1);
        }
    }
    DeviceBuffer<SchedulerRecord> input(host_input.size());
    copy_host_to_device(input, host_input);
    DeviceBuffer<uint64_t> latency_cycles(1);
    DeviceBuffer<uint64_t> throughput_cycles(blocks);
    DeviceBuffer<uint32_t> latency_sinks(1);
    DeviceBuffer<uint32_t> throughput_sinks(blocks);
    const TimingResult timing = measure_atomic(
        options, latency_cycles.data(),
        [&] {
            global_load_scheduler_record_kernel<V><<<1, threads>>>(
                input.data(), latency_cycles.data(), latency_sinks.data(),
                options.iters, working_records, issuers, pattern);
        },
        [&] {
            global_load_scheduler_record_kernel<V><<<blocks, threads>>>(
                input.data(), throughput_cycles.data(), throughput_sinks.data(),
                options.iters, working_records, issuers, pattern);
        });
    require_nonzero_sinks(throughput_sinks, "scheduler record load");

    const double source_ops_per_round = static_cast<double>(issuers) * 2.0;
    const double total_source_ops = static_cast<double>(blocks) * options.iters *
                                    source_ops_per_round;
    const double requested_bytes = static_cast<double>(blocks) * options.iters *
                                   issuers * sizeof(SchedulerRecord);
    JsonObject params;
    add_common_params(params, properties, options, blocks);
    params.add("threads", threads)
        .add("issuers", issuers)
        .add("pattern", pattern_name)
        .add("working_set_records", working_records)
        .add("working_set_bytes",
             static_cast<uint64_t>(working_records) * sizeof(SchedulerRecord))
        .add("record_bytes", sizeof(SchedulerRecord))
        .add("ptx", "2x ld.global.v4.u32")
        .add("correct", true);
    print_memory_result(
        "dense_decode.global_load_scheduler_record_32b", params,
        timing.cycles_per_round,
        "two explicit 128-bit loads per 32-byte scheduler record",
        source_ops_per_round, total_source_ops, requested_bytes,
        timing.elapsed_ms, "ld.global.v4.u32", "global", options.peak, true,
        timing.latency_samples, timing.event_ms_samples);
    return 0;
}

template <Variant V>
inline int run_global_load_float4_oaccum(int argc, char** argv) {
    static_assert(V == Variant::kGlobalLoadFloat4OAccum);
    const Args args(argc, argv);
    args.require_only({"iters", "warmup", "samples", "blocks", "device",
                       "peak", "num-splits", "rowsets", "warps",
                       "vectors-per-thread", "pattern"});
    const CommonOptions options = parse_common_options(args, 64);
    const cudaDeviceProp properties = require_sm90(options.device);
    const int num_splits = args.get_int("num-splits", 8, 1, 160);
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
                                       num_splits * kSplitElements;
    if (input_elements > (std::size_t{1} << 32)) {
        throw std::invalid_argument(
            "OAccum working set exceeds the 16-GiB benchmark limit");
    }

    DeviceBuffer<float> input(input_elements);
    CUDA_CHECK(cudaMemset(input.data(), 0x3f, input_elements * sizeof(float)));
    DeviceBuffer<uint64_t> latency_cycles(1);
    DeviceBuffer<uint64_t> throughput_cycles(blocks);
    DeviceBuffer<uint32_t> latency_sinks(1);
    DeviceBuffer<uint32_t> throughput_sinks(blocks);
    const TimingResult timing = measure_atomic(
        options, latency_cycles.data(),
        [&] {
            global_load_float4_oaccum_kernel<V><<<1, 256>>>(
                input.data(), latency_cycles.data(), latency_sinks.data(),
                options.iters, rowsets, num_splits, active_warps,
                vectors_per_thread, pattern);
        },
        [&] {
            global_load_float4_oaccum_kernel<V><<<blocks, 256>>>(
                input.data(), throughput_cycles.data(), throughput_sinks.data(),
                options.iters, rowsets, num_splits, active_warps,
                vectors_per_thread, pattern);
        });
    require_nonzero_sinks(throughput_sinks, "combine float4 OAccum load");

    const double source_ops_per_round = static_cast<double>(active_warps) * 32.0 *
                                        num_splits * vectors_per_thread;
    const double total_source_ops = static_cast<double>(blocks) * options.iters *
                                    source_ops_per_round;
    const double requested_bytes = total_source_ops * 16.0;
    JsonObject params;
    add_common_params(params, properties, options, blocks);
    params.add("num_splits", num_splits)
        .add("rowsets", rowsets)
        .add("warps", active_warps)
        .add("vectors_per_thread", vectors_per_thread)
        .add("pattern", pattern_name)
        .add("working_set_bytes",
             static_cast<uint64_t>(input_elements * sizeof(float)))
        .add("split_stride_elements", static_cast<int>(kSplitElements))
        .add("ptx", "ld.global.v4.f32")
        .add("correct", true);
    print_memory_result(
        "dense_decode.global_load_float4_oaccum", params,
        timing.cycles_per_round,
        "dense combine CTA loads all selected splits and float4 vectors",
        source_ops_per_round, total_source_ops, requested_bytes,
        timing.elapsed_ms, "ld.global.v4.f32", "global", options.peak, true,
        timing.latency_samples, timing.event_ms_samples);
    return 0;
}

template <Variant V>
inline int run_global_load_f32_lse_strided(int argc, char** argv) {
    static_assert(V == Variant::kGlobalLoadF32LseStrided);
    const Args args(argc, argv);
    args.require_only({"iters", "warmup", "samples", "blocks", "device",
                       "peak", "num-splits", "split-stride", "rowsets",
                       "warps", "pattern"});
    const CommonOptions options = parse_common_options(args, 256);
    const cudaDeviceProp properties = require_sm90(options.device);
    const int num_splits = args.get_int("num-splits", 32, 1, 160);
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
                                       num_splits * split_stride;
    if (input_elements > (std::size_t{1} << 32)) {
        throw std::invalid_argument(
            "LSE working set exceeds the 16-GiB benchmark limit");
    }

    DeviceBuffer<float> input(input_elements);
    CUDA_CHECK(cudaMemset(input.data(), 0x3f, input_elements * sizeof(float)));
    DeviceBuffer<uint64_t> latency_cycles(1);
    DeviceBuffer<uint64_t> throughput_cycles(blocks);
    DeviceBuffer<uint32_t> latency_sinks(1);
    DeviceBuffer<uint32_t> throughput_sinks(blocks);
    const TimingResult timing = measure_atomic(
        options, latency_cycles.data(),
        [&] {
            global_load_f32_lse_strided_kernel<V><<<1, 256>>>(
                input.data(), latency_cycles.data(), latency_sinks.data(),
                options.iters, rowsets, num_splits, split_stride, active_warps,
                pattern);
        },
        [&] {
            global_load_f32_lse_strided_kernel<V><<<blocks, 256>>>(
                input.data(), throughput_cycles.data(), throughput_sinks.data(),
                options.iters, rowsets, num_splits, split_stride, active_warps,
                pattern);
        });
    require_nonzero_sinks(throughput_sinks, "combine strided LSE load");

    const double source_ops_per_round = static_cast<double>(active_warps) *
                                        num_splits;
    const double total_source_ops = static_cast<double>(blocks) * options.iters *
                                    source_ops_per_round;
    const double requested_bytes = total_source_ops * sizeof(float);
    JsonObject params;
    add_common_params(params, properties, options, blocks);
    params.add("num_splits", num_splits)
        .add("split_stride", split_stride)
        .add("rowsets", rowsets)
        .add("warps", active_warps)
        .add("pattern", pattern_name)
        .add("working_set_bytes",
             static_cast<uint64_t>(input_elements * sizeof(float)))
        .add("ptx", "ld.global.f32")
        .add("correct", true);
    print_memory_result(
        "dense_decode.global_load_f32_lse_strided", params,
        timing.cycles_per_round,
        "one strided LSE scalar load per active warp and split",
        source_ops_per_round, total_source_ops, requested_bytes,
        timing.elapsed_ms, "ld.global.f32", "global", options.peak, true,
        timing.latency_samples, timing.event_ms_samples);
    return 0;
}

template <Variant V>
inline int run_global_store_f32_lse(int argc, char** argv) {
    static_assert(V == Variant::kGlobalStoreF32Lse);
    const Args args(argc, argv);
    args.require_only({"iters", "warmup", "samples", "blocks", "device",
                       "peak", "role", "working-set-records", "pattern"});
    const CommonOptions options = parse_common_options(args, 4096);
    const cudaDeviceProp properties = require_sm90(options.device);
    const std::string role_name = args.get_string("role", "main");
    int role = -1;
    if (role_name == "main") role = 0;
    if (role_name == "combine") role = 1;
    if (role < 0) {
        throw std::invalid_argument("--role must be main or combine");
    }
    const int values_per_record = role == 0 ? 64 : 8;
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
    DeviceBuffer<uint64_t> latency_cycles(1);
    DeviceBuffer<uint64_t> throughput_cycles(blocks);
    DeviceBuffer<uint32_t> latency_sinks(1);
    DeviceBuffer<uint32_t> throughput_sinks(blocks);
    const TimingResult timing = measure_atomic(
        options, latency_cycles.data(),
        [&] {
            global_store_f32_lse_kernel<V><<<1, 256>>>(
                output.data(), latency_cycles.data(), latency_sinks.data(),
                options.iters, records_per_block, role, pattern);
        },
        [&] {
            global_store_f32_lse_kernel<V><<<blocks, 256>>>(
                output.data(), throughput_cycles.data(), throughput_sinks.data(),
                options.iters, records_per_block, role, pattern);
        });
    require_nonzero_sinks(throughput_sinks, "global LSE f32 store");

    output.zero();
    global_store_f32_lse_kernel<V><<<1, 256>>>(
        output.data(), latency_cycles.data(), latency_sinks.data(), 1,
        records_per_block, role, AccessPattern::kLocal);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float validation = 0.0f;
    CUDA_CHECK(cudaMemcpy(&validation, output.data(), sizeof(validation),
                          cudaMemcpyDeviceToHost));
    if (validation == 0.0f) {
        throw std::runtime_error("global LSE f32 store validation failed");
    }

    const double source_ops_per_round = static_cast<double>(values_per_record);
    const double total_source_ops = static_cast<double>(blocks) * options.iters *
                                    source_ops_per_round;
    const double requested_bytes = total_source_ops * sizeof(float);
    JsonObject params;
    add_common_params(params, properties, options, blocks);
    params.add("role", role_name)
        .add("working_set_records", records_per_block)
        .add("working_set_bytes",
             static_cast<uint64_t>(output_elements * sizeof(float)))
        .add("pattern", pattern_name)
        .add("producers", values_per_record)
        .add("ptx", "st.global.f32")
        .add("correct", true);
    print_memory_result(
        "dense_decode.global_store_f32_lse", params,
        timing.cycles_per_round,
        "main 64-thread or combine 8-lane0 scalar LSE store issue loop",
        source_ops_per_round, total_source_ops, requested_bytes,
        timing.elapsed_ms, "st.global.f32", "global", options.peak, true,
        timing.latency_samples, timing.event_ms_samples);
    return 0;
}

template <Variant V>
inline int run_global_store_scheduler_record(int argc, char** argv) {
    static_assert(V == Variant::kGlobalStoreSchedulerRecord32B);
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

    DeviceBuffer<SchedulerRecord> output(output_records);
    output.zero();
    DeviceBuffer<uint64_t> latency_cycles(1);
    DeviceBuffer<uint64_t> throughput_cycles(blocks);
    DeviceBuffer<uint32_t> latency_sinks(1);
    DeviceBuffer<uint32_t> throughput_sinks(blocks);
    const TimingResult timing = measure_atomic(
        options, latency_cycles.data(),
        [&] {
            global_store_scheduler_record_kernel<V><<<1, 32>>>(
                output.data(), latency_cycles.data(), latency_sinks.data(),
                options.iters, records_per_block, pattern);
        },
        [&] {
            global_store_scheduler_record_kernel<V><<<blocks, 32>>>(
                output.data(), throughput_cycles.data(), throughput_sinks.data(),
                options.iters, records_per_block, pattern);
        });
    require_nonzero_sinks(throughput_sinks, "scheduler record store");

    output.zero();
    global_store_scheduler_record_kernel<V><<<1, 32>>>(
        output.data(), latency_cycles.data(), latency_sinks.data(), 1,
        records_per_block, AccessPattern::kLocal);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    SchedulerRecord validation{};
    CUDA_CHECK(cudaMemcpy(&validation, output.data(), sizeof(validation),
                          cudaMemcpyDeviceToHost));
    if (validation.words[0] == 0 || validation.words[7] == 0) {
        throw std::runtime_error("scheduler record store validation failed");
    }

    constexpr double source_ops_per_round = 2.0;
    const double total_source_ops = static_cast<double>(blocks) * options.iters *
                                    source_ops_per_round;
    const double requested_bytes = static_cast<double>(blocks) * options.iters *
                                   sizeof(SchedulerRecord);
    JsonObject params;
    add_common_params(params, properties, options, blocks);
    params.add("working_set_records", records_per_block)
        .add("working_set_bytes",
             static_cast<uint64_t>(output_records * sizeof(SchedulerRecord)))
        .add("pattern", pattern_name)
        .add("record_bytes", sizeof(SchedulerRecord))
        .add("issuers", 1)
        .add("ptx", "2x st.global.v4.u32")
        .add("correct", true);
    print_memory_result(
        "dense_decode.global_store_scheduler_record_32b", params,
        timing.cycles_per_round,
        "thread0 writes one aligned 32-byte scheduler record as two stores",
        source_ops_per_round, total_source_ops, requested_bytes,
        timing.elapsed_ms, "st.global.v4.u32", "global", options.peak, true,
        timing.latency_samples, timing.event_ms_samples);
    return 0;
}

template <Variant V>
inline int run_global_store_u32_num_splits(int argc, char** argv) {
    static_assert(V == Variant::kGlobalStoreU32NumSplits);
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
    DeviceBuffer<uint64_t> latency_cycles(1);
    DeviceBuffer<uint64_t> throughput_cycles(blocks);
    DeviceBuffer<uint32_t> latency_sinks(1);
    DeviceBuffer<uint32_t> throughput_sinks(blocks);
    const TimingResult timing = measure_atomic(
        options, latency_cycles.data(),
        [&] {
            global_store_u32_num_splits_kernel<V><<<1, 32>>>(
                output.data(), latency_cycles.data(), latency_sinks.data(),
                options.iters, records_per_block, producers, pattern);
        },
        [&] {
            global_store_u32_num_splits_kernel<V><<<blocks, 32>>>(
                output.data(), throughput_cycles.data(), throughput_sinks.data(),
                options.iters, records_per_block, producers, pattern);
        });
    require_nonzero_sinks(throughput_sinks, "num_splits u32 store");

    output.zero();
    global_store_u32_num_splits_kernel<V><<<1, 32>>>(
        output.data(), latency_cycles.data(), latency_sinks.data(), 1,
        records_per_block, producers, AccessPattern::kLocal);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t validation = 0;
    CUDA_CHECK(cudaMemcpy(&validation, output.data(), sizeof(validation),
                          cudaMemcpyDeviceToHost));
    if (validation == 0) {
        throw std::runtime_error("num_splits u32 store validation failed");
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
        .add("correct", true);
    print_memory_result(
        "dense_decode.global_store_u32_num_splits", params,
        timing.cycles_per_round,
        "coalesced scheduler num_splits scalar-store issue loop",
        source_ops_per_round, total_source_ops, requested_bytes,
        timing.elapsed_ms, "st.global.u32", "global", options.peak, true,
        timing.latency_samples, timing.event_ms_samples);
    return 0;
}

template <Variant V>
inline int run_global_store_u64_output(int argc, char** argv) {
    static_assert(V == Variant::kGlobalStoreU64Output);
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
    DeviceBuffer<uint64_t> latency_cycles(1);
    DeviceBuffer<uint64_t> throughput_cycles(blocks);
    DeviceBuffer<uint32_t> latency_sinks(1);
    DeviceBuffer<uint32_t> throughput_sinks(blocks);
    const TimingResult timing = measure_atomic(
        options, latency_cycles.data(),
        [&] {
            global_store_u64_output_kernel<V><<<1, 256>>>(
                output.data(), latency_cycles.data(), latency_sinks.data(),
                options.iters, records_per_block, active_warps,
                vectors_per_thread, pattern, value_pattern);
        },
        [&] {
            global_store_u64_output_kernel<V><<<blocks, 256>>>(
                output.data(), throughput_cycles.data(), throughput_sinks.data(),
                options.iters, records_per_block, active_warps,
                vectors_per_thread, pattern, value_pattern);
        });
    require_nonzero_sinks(throughput_sinks, "BF16/FP16 output u64 store");

    output.zero();
    global_store_u64_output_kernel<V><<<1, 256>>>(
        output.data(), latency_cycles.data(), latency_sinks.data(), 1,
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
        .add("correct", true);
    print_memory_result(
        "dense_decode.global_store_u64_output", params,
        timing.cycles_per_round,
        "combine writes four packed b16x4 values per lane at full shape",
        source_ops_per_round, total_source_ops, requested_bytes,
        timing.elapsed_ms, "st.global.u64", "global", options.peak, true,
        timing.latency_samples, timing.event_ms_samples);
    return 0;
}

inline int parse_tensormap_mode(const std::string& value) {
    if (value == "q") return 0;
    if (value == "k") return 1;
    if (value == "o") return 2;
    if (value == "qko") return 3;
    throw std::invalid_argument("--mode must be q, k, o, or qko");
}

template <Variant V>
inline int run_tensormap_prefetch(int argc, char** argv) {
    static_assert(V == Variant::kTensorMapPrefetchQkoRank4);
    const Args args(argc, argv);
    args.require_only({"iters", "warmup", "samples", "blocks", "device",
                       "peak", "mode", "working-pages", "working-tiles"});
    const CommonOptions options = parse_common_options(args, 4096);
    const cudaDeviceProp properties = require_sm90(options.device);
    const std::string mode_name = args.get_string("mode", "qko");
    const int mode = parse_tensormap_mode(mode_name);
    const int working_pages = args.get_int("working-pages", 64, 1, 8192);
    const int working_tiles = args.get_int("working-tiles", 64, 1, 8192);
    const int blocks = resolve_blocks(options.blocks, properties, 4);
    constexpr std::size_t kQkElements = 64 * 576;
    constexpr std::size_t kOElements = 64 * 512;

    DeviceBuffer<uint16_t> q(
        static_cast<std::size_t>(working_pages) * kQkElements);
    DeviceBuffer<uint16_t> k(
        static_cast<std::size_t>(working_pages) * kQkElements);
    DeviceBuffer<uint16_t> o(
        static_cast<std::size_t>(working_tiles) * kOElements);
    const CUtensorMap q_map = make_tma_load_64x576_bf16_rank4_map(
        q.data(), working_pages);
    const CUtensorMap k_map = make_tma_load_64x576_bf16_rank4_map(
        k.data(), working_pages);
    const CUtensorMap o_map = make_tma_store_64x512_bf16_rank4_map(
        o.data(), working_tiles);

    DeviceBuffer<uint64_t> latency_cycles(1);
    DeviceBuffer<uint64_t> throughput_cycles(blocks);
    DeviceBuffer<uint32_t> latency_sinks(1);
    DeviceBuffer<uint32_t> throughput_sinks(blocks);
    const TimingResult timing = measure_atomic(
        options, latency_cycles.data(),
        [&] {
            tensormap_prefetch_kernel<V><<<1, 32>>>(
                q_map, k_map, o_map, latency_cycles.data(),
                latency_sinks.data(), options.iters, mode);
        },
        [&] {
            tensormap_prefetch_kernel<V><<<blocks, 32>>>(
                q_map, k_map, o_map, throughput_cycles.data(),
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
        .add("correct", true);
    const JsonObject latency = latency_metric(
        timing.cycles_per_round,
        "one issuer prefetches selected kernel-parameter descriptors",
        descriptors_per_round, timing.latency_samples);
    const JsonObject throughput = throughput_metric(
        total_source_ops, timing.elapsed_ms, "prefetch.tensormap",
        timing.event_ms_samples);
    const JsonObject bandwidth = null_metric(
        "GB/s", "tensor-map prefetch has no architected payload-byte count");
    const JsonObject hardware = null_metric(
        "ratio", "no published tensor-map-prefetch peak");
    print_result("dense_decode.tensormap_prefetch_qko_rank4", params,
                 latency, throughput, bandwidth, hardware);
    return 0;
}

template <Variant V>
int run(int argc, char** argv) {
    try {
        if constexpr (V == Variant::kSharedLoadU32Patterns) {
            return run_shared_load_u32<V>(argc, argv);
        } else if constexpr (V == Variant::kSharedStoreU32Scalar) {
            return run_shared_store_u32<V>(argc, argv);
        } else if constexpr (V == Variant::kSharedStoreU64Dense) {
            return run_shared_store_u64<V>(argc, argv);
        } else if constexpr (V == Variant::kGlobalLoadI32Cached) {
            return run_global_load_i32_cached<V>(argc, argv);
        } else if constexpr (V == Variant::kGlobalLoadI32Ordinary) {
            return run_global_load_i32_cached<V>(argc, argv);
        } else if constexpr (V == Variant::kGlobalLoadSchedulerRecord32B) {
            return run_global_load_scheduler_record<V>(argc, argv);
        } else if constexpr (V == Variant::kGlobalLoadFloat4OAccum) {
            return run_global_load_float4_oaccum<V>(argc, argv);
        } else if constexpr (V == Variant::kGlobalLoadF32LseStrided) {
            return run_global_load_f32_lse_strided<V>(argc, argv);
        } else if constexpr (V == Variant::kGlobalStoreF32Lse) {
            return run_global_store_f32_lse<V>(argc, argv);
        } else if constexpr (V == Variant::kGlobalStoreSchedulerRecord32B) {
            return run_global_store_scheduler_record<V>(argc, argv);
        } else if constexpr (V == Variant::kGlobalStoreU32NumSplits) {
            return run_global_store_u32_num_splits<V>(argc, argv);
        } else if constexpr (V == Variant::kGlobalStoreU64Output) {
            return run_global_store_u64_output<V>(argc, argv);
        } else if constexpr (V == Variant::kTensorMapPrefetchQkoRank4) {
            return run_tensormap_prefetch<V>(argc, argv);
        } else {
            static_assert(V != V, "unsupported memory atomic variant");
        }
    } catch (const std::exception& error) {
        std::cerr << "memory atomic benchmark error: " << error.what() << '\n';
        return 1;
    }
}

}  // namespace microbench::memory_atomic
