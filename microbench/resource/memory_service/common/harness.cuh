#pragma once

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/bench.hpp"

namespace microbench::memory_service {

enum class Operation { kLoadU64, kStoreU64 };
enum class Pattern : int { kLocal, kSequential, kRandom, kReuse };

template <Operation Op>
struct Traits;

template <>
struct Traits<Operation::kLoadU64> {
    static constexpr const char* kName = "ld_global_u64_saturation";
    static constexpr const char* kPtx = "ld.global.u64";
};

template <>
struct Traits<Operation::kStoreU64> {
    static constexpr const char* kName = "st_global_u64_saturation";
    static constexpr const char* kPtx = "st.global.u64";
};

inline Pattern parse_pattern(const std::string& value) {
    if (value == "local") return Pattern::kLocal;
    if (value == "sequential") return Pattern::kSequential;
    if (value == "random") return Pattern::kRandom;
    if (value == "reuse") return Pattern::kReuse;
    throw std::invalid_argument(
        "--pattern must be local, sequential, random, or reuse");
}

__device__ __forceinline__ uint64_t load_u64(const uint64_t* pointer) {
    uint64_t value;
    asm volatile("ld.global.u64 %0, [%1];"
                 : "=l"(value) : "l"(pointer) : "memory");
    return value;
}

__device__ __forceinline__ void store_u64(uint64_t* pointer, uint64_t value) {
    asm volatile("st.global.u64 [%0], %1;"
                 : : "l"(pointer), "l"(value) : "memory");
}

__device__ __forceinline__ uint64_t mix(uint64_t value) {
    value ^= value >> 33;
    value *= 0xff51afd7ed558ccdULL;
    value ^= value >> 33;
    return value;
}

template <Operation Op>
__global__ void kernel(uint64_t* data,
                       uint64_t elements,
                       uint64_t* cycles,
                       uint64_t* sinks,
                       uint32_t* smids,
                       int iterations,
                       int outstanding_depth,
                       Pattern pattern) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    const uint64_t global_thread =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t grid_threads =
        static_cast<uint64_t>(gridDim.x) * blockDim.x;
    uint64_t checksum = global_thread + 1;
    const uint64_t start = read_clock64();
#pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
#pragma unroll 1
        for (int depth = 0; depth < outstanding_depth; ++depth) {
            uint64_t logical = global_thread +
                static_cast<uint64_t>(depth) * grid_threads;
            if (pattern == Pattern::kSequential) {
                logical += static_cast<uint64_t>(iteration) * grid_threads *
                           outstanding_depth;
            } else if (pattern == Pattern::kRandom) {
                logical = mix(logical ^ static_cast<uint64_t>(iteration));
            } else if (pattern == Pattern::kReuse) {
                logical = global_thread & 1023ULL;
            }
            const uint64_t index = logical % elements;
            if constexpr (Op == Operation::kLoadU64) {
                checksum ^= load_u64(data + index);
            } else {
                const uint64_t value = checksum + index + iteration;
                store_u64(data + index, value);
                checksum ^= value;
            }
        }
    }
    const uint64_t stop = read_clock64();
    if (threadIdx.x == 0) cycles[blockIdx.x] = stop - start;
    if (threadIdx.x == 0) smids[blockIdx.x] = read_smid();
    sinks[global_thread] = checksum | 1ULL;
#else
    if (threadIdx.x == 0) cycles[blockIdx.x] = 0;
    sinks[static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x] = 1;
#endif
}

__global__ void evict_l2_kernel(uint64_t* data, uint64_t elements) {
    uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (; index < elements; index += stride) {
        const uint64_t value = load_u64(data + index);
        store_u64(data + index, value + index + 1);
    }
}

template <Operation Op>
int run(int argc, char** argv) {
    try {
        const Args args(argc, argv);
        args.require_only({"iters", "warmup", "samples", "blocks", "device",
                           "peak", "working-set-bytes", "pattern",
                           "cache-mode", "outstanding-depth", "threads"});
        const CommonOptions options = parse_common_options(args, 1024);
        const cudaDeviceProp properties = require_sm90(options.device);
        const int clock_khz = device_clock_khz(options.device);
        const int threads = args.get_int("threads", 256, 32, 1024);
        if (threads % 32 != 0) {
            throw std::invalid_argument("--threads must be a warp multiple");
        }
        const int blocks = resolve_blocks(options.blocks, properties, 4);
        const int outstanding_depth =
            args.get_int("outstanding-depth", 4, 1, 32);
        const uint64_t working_set_bytes = static_cast<uint64_t>(
            args.get_double("working-set-bytes", 1ULL << 30, 4096.0,
                            static_cast<double>(1ULL << 40)));
        const uint64_t elements = std::max<uint64_t>(
            working_set_bytes / sizeof(uint64_t), 1);
        const std::string pattern_name =
            args.get_string("pattern", "sequential");
        const Pattern pattern = parse_pattern(pattern_name);
        const std::string cache_mode =
            args.get_string("cache-mode", "hbm_stream");
        if (cache_mode != "l2_hot" && cache_mode != "hbm_stream") {
            throw std::invalid_argument(
                "--cache-mode must be l2_hot or hbm_stream");
        }

        DeviceBuffer<uint64_t> data(elements);
        DeviceBuffer<uint64_t> cycles(blocks);
        DeviceBuffer<uint64_t> sinks(
            static_cast<std::size_t>(blocks) * threads);
        DeviceBuffer<uint32_t> smids(blocks);
        constexpr uint64_t kEvictionBytes = 128ULL << 20;
        DeviceBuffer<uint64_t> eviction(kEvictionBytes / sizeof(uint64_t));
        data.zero();
        auto launch = [&] {
            kernel<Op><<<blocks, threads>>>(
                data.data(), elements, cycles.data(), sinks.data(), smids.data(),
                options.iters, outstanding_depth, pattern);
        };
        auto prepare_cache = [&] {
            if (cache_mode == "l2_hot") {
                launch();
            } else {
                evict_l2_kernel<<<512, 256>>>(
                    eviction.data(), eviction.size());
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        };
        for (int warmup = 0; warmup < options.warmup; ++warmup) {
            prepare_cache();
            launch();
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        std::vector<double> latency_samples;
        std::vector<double> event_samples;
        CudaEvent start;
        CudaEvent stop;
        for (int sample = 0; sample < options.samples; ++sample) {
            prepare_cache();
            CUDA_CHECK(cudaEventRecord(start));
            launch();
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float elapsed_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
            event_samples.push_back(elapsed_ms);
            const auto host_cycles = cycles.copy_to_host();
            latency_samples.push_back(
                static_cast<double>(*std::max_element(
                    host_cycles.begin(), host_cycles.end())) / options.iters);
        }
        const double operations = static_cast<double>(blocks) * threads *
            options.iters * outstanding_depth;
        const double bytes = operations * sizeof(uint64_t);
        std::vector<double> throughput_samples = event_samples;
        std::vector<double> bandwidth_samples = event_samples;
        for (double& value : throughput_samples) {
            value = operations / value / 1.0e6;
        }
        for (double& value : bandwidth_samples) {
            value = bytes / value / 1.0e6;
        }
        std::vector<double> service_interval_samples = event_samples;
        for (double& value : service_interval_samples) {
            value = value * clock_khz /
                    (static_cast<double>(blocks) * threads * options.iters *
                     outstanding_depth);
        }
        std::map<uint32_t, int> smid_counts;
        for (uint32_t smid : smids.copy_to_host()) ++smid_counts[smid];
        JsonObject smid_histogram;
        for (const auto& [smid, count] : smid_counts) {
            smid_histogram.add(std::to_string(smid), count);
        }

        JsonObject params;
        params.add("gpu", properties.name)
            .add("resource", cache_mode == "l2_hot" ? "l2" : "hbm")
            .add("iters", options.iters)
            .add("warmup", options.warmup)
            .add("samples", options.samples)
            .add("blocks", options.blocks)
            .add("resolved_blocks", blocks)
            .add("threads", threads)
            .add("outstanding_depth", outstanding_depth)
            .add("working_set_bytes", working_set_bytes)
            .add("pattern", pattern_name)
            .add("cache_mode", cache_mode)
            .add("cache_preparation", cache_mode == "l2_hot"
                    ? "untimed_target_prewarm_each_sample"
                    : "untimed_128MiB_l2_eviction_each_sample")
            .add("initiation_interval_cycles", median(service_interval_samples))
            .add("initiation_interval_scope", "aggregate_source_operation")
            .add("unique_active_sms", static_cast<int>(smid_counts.size()))
            .add("smid_histogram", smid_histogram)
            .add("target_ptx", Traits<Op>::kPtx)
            .add("device", options.device)
            .add("peak", options.peak);
        JsonObject latency;
        latency.add("value", median(latency_samples))
            .add("unit", "cycles/round")
            .add("timer", "clock64")
            .add("scope", "max_across_ctas")
            .add_raw("samples", json_number_array(latency_samples));
        JsonObject throughput = metric(median(throughput_samples), "Gop/s");
        throughput.add("timer", "cuda_event")
            .add("scope", "grid")
            .add_raw("samples", json_number_array(throughput_samples));
        JsonObject bandwidth = metric(median(bandwidth_samples), "GB/s");
        bandwidth.add("kind", "requested")
            .add_raw("samples", json_number_array(bandwidth_samples));
        print_result(Traits<Op>::kName, params, latency, throughput, bandwidth,
                     utilization(median(bandwidth_samples), options.peak,
                                 "GB/s"));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << Traits<Op>::kName << ": " << error.what() << '\n';
        return 1;
    }
}

}  // namespace microbench::memory_service
