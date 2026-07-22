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
    kSharedStoreU32Scalar,
    kSharedStoreV2U32Stride520,
    kSharedStoreU64Sw128,
};
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
template <bool IssueTarget>
__device__ __forceinline__ uint32_t store_shared_u32_or_baseline(
        uint32_t* pointer, uint32_t value) {
    const uint32_t address = shared_address(pointer);
    if constexpr (IssueTarget) {
        asm volatile("st.shared.u32 [%0], %1;"
                     :
                     : "r"(address), "r"(value)
                     : "memory");
    } else {
        // The returned token keeps baseline address/value generation live.
        asm volatile("" : : "r"(address), "r"(value) : "memory");
    }
    return address ^ value;
}

template <bool IssueTarget>
__device__ __forceinline__ uint32_t store_shared_u64_or_baseline(
        uint64_t* pointer, uint64_t value) {
    const uint32_t address = shared_address(pointer);
    if constexpr (IssueTarget) {
        asm volatile("st.shared.u64 [%0], %1;"
                     :
                     : "r"(address), "l"(value)
                     : "memory");
    } else {
        asm volatile("" : : "r"(address), "l"(value) : "memory");
    }
    return address ^ static_cast<uint32_t>(value) ^
           static_cast<uint32_t>(value >> 32);
}

template <bool IssueTarget>
__device__ __forceinline__ uint32_t store_shared_v2_u32_or_baseline(
        uint32_t* pointer, uint32_t value0, uint32_t value1) {
    const uint32_t address = shared_address(pointer);
    if constexpr (IssueTarget) {
        asm volatile("st.shared.v2.u32 [%0], {%1, %2};"
                     :
                     : "r"(address), "r"(value0), "r"(value1)
                     : "memory");
    } else {
        asm volatile(""
                     :
                     : "r"(address), "r"(value0), "r"(value1)
                     : "memory");
    }
    return address ^ value0 ^ value1;
}

template <Variant V, bool IssueTarget, bool CollectChecksum>
__device__ __forceinline__ void shared_store_u32_body(
        uint64_t* cycles,
        uint32_t* sinks,
        int iterations,
        int working_words,
        int producers,
        int topology) {
    static_assert(V == Variant::kSharedStoreU32Scalar);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    extern __shared__ __align__(128) uint32_t storage[];
    uint64_t start = 0;
    if constexpr (CollectChecksum) {
        if (threadIdx.x == 0) start = read_clock64();
    }
    __syncthreads();

    uint32_t checksum = static_cast<uint32_t>(threadIdx.x + 1);
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
            const uint32_t token = store_shared_u32_or_baseline<IssueTarget>(
                storage + index,
                static_cast<uint32_t>(iteration + producer_index + 1));
            if constexpr (CollectChecksum) {
                checksum = (checksum << 5) ^ (checksum >> 2) ^ token;
            }
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        if constexpr (CollectChecksum) {
            cycles[blockIdx.x] = read_clock64() - start;
            sinks[blockIdx.x] = checksum == 0 ? 1 : checksum;
        } else {
            sinks[blockIdx.x] = static_cast<uint32_t>(iterations + producers + 1);
        }
    }
#else
    if (threadIdx.x == 0) {
        cycles[blockIdx.x] = 0;
        sinks[blockIdx.x] = 0;
    }
#endif
}

template <Variant V>
__global__ void shared_store_u32_target_kernel(uint64_t* cycles,
                                               uint32_t* sinks,
                                               int iterations,
                                               int working_words,
                                               int producers,
                                               int topology) {
    shared_store_u32_body<V, true, true>(
        cycles, sinks, iterations, working_words, producers, topology);
}

template <Variant V>
__global__ void shared_store_u32_baseline_kernel(uint64_t* cycles,
                                                 uint32_t* sinks,
                                                 int iterations,
                                                 int working_words,
                                                 int producers,
                                                 int topology) {
    shared_store_u32_body<V, false, true>(
        cycles, sinks, iterations, working_words, producers, topology);
}

template <Variant V>
__global__ void shared_store_u32_throughput_kernel(uint32_t* sinks,
                                                   int iterations,
                                                   int working_words,
                                                   int producers,
                                                   int topology) {
    shared_store_u32_body<V, true, false>(
        nullptr, sinks, iterations, working_words, producers, topology);
}

__host__ __device__ constexpr uint32_t sw128_u64_byte_address(
        int head_group,
        int token) {
    // CUTE layout, expressed without a CUTE dependency:
    // Sw<3,4,3> tile ((_16,_4),(_8,_8)):((_1,_1024),(_16,_128)).
    const uint32_t logical_element = static_cast<uint32_t>(
        (head_group & 15) + (head_group >> 4) * 1024 +
        (token & 7) * 16 + (token >> 3) * 128);
    const uint32_t physical_element =
        logical_element ^ ((logical_element & 0x380u) >> 3);
    return physical_element * static_cast<uint32_t>(sizeof(uint64_t));
}
static_assert(sw128_u64_byte_address(0, 0) == 0);
static_assert(sw128_u64_byte_address(0, 8) == 1152);
static_assert(sw128_u64_byte_address(16, 0) == 8192);
static_assert(sw128_u64_byte_address(63, 63) == 31864);

template <Variant V, bool IssueTarget, bool CollectChecksum>
__device__ __forceinline__ void shared_store_8b_body(
        uint64_t* cycles,
        uint32_t* sinks,
        int iterations,
        int warpgroups,
        int stores_per_thread,
        int invalid_tokens) {
    static_assert(V == Variant::kSharedStoreV2U32Stride520 ||
                  V == Variant::kSharedStoreU64Sw128);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    extern __shared__ __align__(128) unsigned char storage[];
    uint64_t start = 0;
    if constexpr (CollectChecksum) {
        if (threadIdx.x == 0) start = read_clock64();
    }
    __syncthreads();

    uint32_t checksum = static_cast<uint32_t>(threadIdx.x + 1);
#pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if constexpr (V == Variant::kSharedStoreV2U32Stride520) {
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
                    const uint32_t token =
                        store_shared_v2_u32_or_baseline<IssueTarget>(
                        destination, value, value ^ 0x5a5a5a5au);
                    if constexpr (CollectChecksum) {
                        checksum = (checksum << 5) ^ (checksum >> 2) ^ token;
                    }
                }
            }
        } else if (threadIdx.x < 128) {
            const int lane = static_cast<int>(threadIdx.x);
            const int first_invalid = 64 - invalid_tokens;
            const int head_group = lane & 63;
#pragma unroll 1
            for (int token = first_invalid + lane / 64; token < 64; token += 2) {
                const uint32_t physical_address =
                    sw128_u64_byte_address(head_group, token);
                auto* destination = reinterpret_cast<uint64_t*>(
                    storage + physical_address);
                const uint32_t value =
                    store_shared_u64_or_baseline<IssueTarget>(destination, 0ull);
                if constexpr (CollectChecksum) {
                    checksum = (checksum << 5) ^ (checksum >> 2) ^ value;
                }
            }
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        if constexpr (CollectChecksum) {
            cycles[blockIdx.x] = read_clock64() - start;
            sinks[blockIdx.x] = checksum == 0 ? 1 : checksum;
        } else {
            sinks[blockIdx.x] = static_cast<uint32_t>(
                iterations +
                (V == Variant::kSharedStoreV2U32Stride520 ? 1 : 2));
        }
    }
#else
    if (threadIdx.x == 0) {
        cycles[blockIdx.x] = 0;
        sinks[blockIdx.x] = 0;
    }
#endif
}

template <Variant V>
__global__ void shared_store_8b_target_kernel(uint64_t* cycles,
                                              uint32_t* sinks,
                                              int iterations,
                                              int warpgroups,
                                              int stores_per_thread,
                                              int invalid_tokens) {
    shared_store_8b_body<V, true, true>(
        cycles, sinks, iterations, warpgroups, stores_per_thread,
        invalid_tokens);
}

template <Variant V>
__global__ void shared_store_8b_baseline_kernel(uint64_t* cycles,
                                                uint32_t* sinks,
                                                int iterations,
                                                int warpgroups,
                                                int stores_per_thread,
                                                int invalid_tokens) {
    shared_store_8b_body<V, false, true>(
        cycles, sinks, iterations, warpgroups, stores_per_thread,
        invalid_tokens);
}

template <Variant V>
__global__ void shared_store_8b_throughput_kernel(uint32_t* sinks,
                                                  int iterations,
                                                  int warpgroups,
                                                  int stores_per_thread,
                                                  int invalid_tokens) {
    shared_store_8b_body<V, true, false>(
        nullptr, sinks, iterations, warpgroups, stores_per_thread,
        invalid_tokens);
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
        shared_store_u32_target_kernel<V>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
    CUDA_CHECK(cudaFuncSetAttribute(
        shared_store_u32_baseline_kernel<V>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
    CUDA_CHECK(cudaFuncSetAttribute(
        shared_store_u32_throughput_kernel<V>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
    DeviceBuffer<uint64_t> latency_target_cycles(1);
    DeviceBuffer<uint64_t> latency_baseline_cycles(1);
    DeviceBuffer<uint32_t> latency_target_sinks(1);
    DeviceBuffer<uint32_t> latency_baseline_sinks(1);
    DeviceBuffer<uint32_t> throughput_sinks(blocks);
    const TimingResult timing = measure_atomic(
        options, latency_target_cycles.data(), latency_baseline_cycles.data(),
        [&] {
            shared_store_u32_target_kernel<V><<<1, threads, shared_bytes>>>(
                latency_target_cycles.data(), latency_target_sinks.data(),
                options.iters, working_words, producers, topology);
            shared_store_u32_baseline_kernel<V><<<1, threads, shared_bytes>>>(
                latency_baseline_cycles.data(), latency_baseline_sinks.data(),
                options.iters, working_words, producers, topology);
        },
        [&] {
            shared_store_u32_throughput_kernel<V>
                <<<blocks, threads, shared_bytes>>>(
                throughput_sinks.data(), options.iters,
                working_words, producers, topology);
        });
    require_nonzero_sinks(latency_target_sinks, "shared st.u32 target");
    require_nonzero_sinks(latency_baseline_sinks, "shared st.u32 baseline");
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
        .add("access_shape", "scalar producers into a shared-memory working set")
        .add("ptx", "st.shared.u32")
        .add("clock_baseline",
             "matched address/value/predicate/control kernel without STS")
        .add("throughput_protocol", "CUDA-event target kernel only")
        .add("correct", true);
    print_memory_result(
        "st_shared_u32", params, timing.cycles_per_round,
        "target issue loop minus matched address, value, predicate, "
        "loop-control, and CTA-convergence baseline",
        source_ops_per_round, timing.target_cycles_per_round,
        timing.baseline_cycles_per_round, total_source_ops, requested_bytes,
        timing.elapsed_ms, "st.shared.u32", "shared", options.peak, false,
        timing.latency_samples, timing.target_samples, timing.baseline_samples,
        timing.event_ms_samples);
    return 0;
}

template <Variant V>
inline int run_shared_store_8b(int argc, char** argv) {
    static_assert(V == Variant::kSharedStoreV2U32Stride520 ||
                  V == Variant::kSharedStoreU64Sw128);
    constexpr int mode =
        V == Variant::kSharedStoreV2U32Stride520 ? 0 : 1;
    const Args args(argc, argv);
    args.require_only({"iters", "warmup", "samples", "blocks", "device",
                       "peak", "warpgroups", "stores-per-thread",
                       "invalid-tokens"});
    const CommonOptions options = parse_common_options(args, 1024);
    const cudaDeviceProp properties = require_sm90(options.device);
    const int warpgroups = args.get_int("warpgroups", 2, 1, 2);
    const int stores_per_thread = args.get_int(
        "stores-per-thread", 64, 1, 64);
    const int invalid_tokens = args.get_int("invalid-tokens", 8, 1, 64);
    const int threads = mode == 0 ? warpgroups * 128 : 128;
    const int shared_bytes = mode == 0 ? 64 * 520 * 4 : 64 * 64 * 8;
    if (shared_bytes > static_cast<int>(properties.sharedMemPerBlockOptin)) {
        throw std::runtime_error(
            "shared-store tile exceeds sharedMemPerBlockOptin");
    }
    const int blocks = resolve_blocks(options.blocks, properties, 1);

    CUDA_CHECK(cudaFuncSetAttribute(
        shared_store_8b_target_kernel<V>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
    CUDA_CHECK(cudaFuncSetAttribute(
        shared_store_8b_baseline_kernel<V>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
    CUDA_CHECK(cudaFuncSetAttribute(
        shared_store_8b_throughput_kernel<V>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
    DeviceBuffer<uint64_t> latency_target_cycles(1);
    DeviceBuffer<uint64_t> latency_baseline_cycles(1);
    DeviceBuffer<uint32_t> latency_target_sinks(1);
    DeviceBuffer<uint32_t> latency_baseline_sinks(1);
    DeviceBuffer<uint32_t> throughput_sinks(blocks);
    const TimingResult timing = measure_atomic(
        options, latency_target_cycles.data(), latency_baseline_cycles.data(),
        [&] {
            shared_store_8b_target_kernel<V><<<1, threads, shared_bytes>>>(
                latency_target_cycles.data(), latency_target_sinks.data(),
                options.iters, warpgroups, stores_per_thread,
                invalid_tokens);
            shared_store_8b_baseline_kernel<V><<<1, threads, shared_bytes>>>(
                latency_baseline_cycles.data(), latency_baseline_sinks.data(),
                options.iters, warpgroups, stores_per_thread,
                invalid_tokens);
        },
        [&] {
            shared_store_8b_throughput_kernel<V>
                <<<blocks, threads, shared_bytes>>>(
                throughput_sinks.data(), options.iters,
                warpgroups, stores_per_thread, invalid_tokens);
        });
    require_nonzero_sinks(latency_target_sinks, "shared st.u64/v2.u32 target");
    require_nonzero_sinks(latency_baseline_sinks,
                          "shared st.u64/v2.u32 baseline");
    require_nonzero_sinks(throughput_sinks, "shared st.u64/v2.u32");

    const double source_ops_per_round = mode == 0
        ? static_cast<double>(threads) * stores_per_thread
        : static_cast<double>(invalid_tokens) * 64.0;
    const double total_source_ops = static_cast<double>(blocks) * options.iters *
                                    source_ops_per_round;
    const double requested_bytes = total_source_ops * sizeof(uint64_t);
    JsonObject params;
    add_common_params(params, properties, options, blocks);
    params.add("mode", mode == 0 ? "stride520_v2_u32" : "sw128_u64")
        .add("warpgroups", warpgroups)
        .add("stores_per_thread", stores_per_thread)
        .add("invalid_tokens", invalid_tokens)
        .add("threads", threads)
        .add("shared_bytes", shared_bytes)
        .add("ptx", mode == 0 ? "st.shared.v2.u32" : "st.shared.u64")
        .add("clock_baseline",
             "matched layout/address/value/control kernel without STS")
        .add("throughput_protocol", "CUDA-event target kernel only")
        .add("correct", true);
    print_memory_result(
        mode == 0 ? "st_shared_v2_u32_stride520" : "st_shared_u64_sw128",
        params,
        timing.cycles_per_round,
        mode == 0
            ? "64x520 v2.u32 target issue loop minus matched layout/address/"
              "value/control baseline"
            : "64x64 SW128 u64 target issue loop minus matched layout/address/"
              "value/control baseline",
        source_ops_per_round, timing.target_cycles_per_round,
        timing.baseline_cycles_per_round, total_source_ops, requested_bytes,
        timing.elapsed_ms,
        mode == 0 ? "st.shared.v2.u32" : "st.shared.u64",
        "shared", options.peak, false, timing.latency_samples,
        timing.target_samples, timing.baseline_samples, timing.event_ms_samples);
    return 0;
}

}  // namespace microbench::memory_atomic
