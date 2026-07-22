#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/bench.hpp"

namespace microbench::griddepcontrol_producer_consumer_bench {

constexpr int kThreads = 256;

struct ProducerStamp {
    uint64_t start;
    uint64_t trigger;
    uint64_t end;
    uint32_t smid;
};

struct ConsumerStamp {
    uint64_t start;
    uint64_t wait_end;
    uint64_t end;
    uint32_t smid;
};

__device__ __forceinline__ uint64_t globaltimer() {
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

__device__ __forceinline__ float burn(float value, int iterations) {
#pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        asm volatile("fma.rn.ftz.f32 %0, %0, %1, %2;"
                     : "+f"(value) : "f"(0.9999847412109375f),
                       "f"(0.0000152587890625f));
    }
    return value;
}

__global__ void producer_kernel(ProducerStamp* stamps,
                                float* sinks,
                                int prefix_iterations,
                                int suffix_iterations) {
    float value = 0.25f + static_cast<float>(threadIdx.x & 31) * 0.001f;
    if (threadIdx.x == 0) {
        stamps[blockIdx.x].start = globaltimer();
        stamps[blockIdx.x].smid = read_smid();
    }
    value = burn(value, prefix_iterations);
    __syncthreads();
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    __syncthreads();
    if (threadIdx.x == 0) stamps[blockIdx.x].trigger = globaltimer();
    value = burn(value, suffix_iterations);
    sinks[static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x] =
        value;
    if (threadIdx.x == 0) stamps[blockIdx.x].end = globaltimer();
}

__global__ void consumer_kernel(ConsumerStamp* stamps,
                                float* sinks,
                                int work_iterations) {
    float value = 0.5f + static_cast<float>(threadIdx.x & 31) * 0.001f;
    if (threadIdx.x == 0) {
        stamps[blockIdx.x].start = globaltimer();
        stamps[blockIdx.x].smid = read_smid();
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    __syncthreads();
    if (threadIdx.x == 0) stamps[blockIdx.x].wait_end = globaltimer();
    value = burn(value, work_iterations);
    sinks[static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x] =
        value;
    if (threadIdx.x == 0) stamps[blockIdx.x].end = globaltimer();
}

template <typename Kernel, typename... Args>
inline void launch_pdl(Kernel kernel,
                       int blocks,
                       cudaStream_t stream,
                       Args... args) {
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{
        dim3(blocks), dim3(kThreads), 0, stream, &attribute, 1};
    CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, args...));
}

inline double median_copy(const std::vector<double>& values) {
    return median(values);
}

inline int run(int argc, char** argv) {
    try {
        const Args args(argc, argv);
        args.require_only({"iters", "warmup", "samples", "blocks", "device",
                           "peak", "producer-blocks", "consumer-blocks",
                           "prefix-iters", "suffix-iters", "consumer-iters"});
        const auto options = parse_common_options(args, 1);
        if (options.iters != 1) {
            throw std::invalid_argument("PDL pair uses --iters=1");
        }
        const auto properties = require_sm90(options.device);
        const int clock_khz = device_clock_khz(options.device);
        const int producer_blocks = args.get_int(
            "producer-blocks", properties.multiProcessorCount, 1, 1 << 20);
        const int consumer_blocks = args.get_int(
            "consumer-blocks", properties.multiProcessorCount, 1, 1 << 20);
        const int prefix_iterations =
            args.get_int("prefix-iters", 4096, 1, 1 << 26);
        const int suffix_iterations =
            args.get_int("suffix-iters", 4096, 1, 1 << 26);
        const int consumer_iterations =
            args.get_int("consumer-iters", 4096, 1, 1 << 26);
        if (options.blocks != 0 && options.blocks != producer_blocks) {
            throw std::invalid_argument(
                "--blocks must be 0 or match --producer-blocks");
        }

        DeviceBuffer<ProducerStamp> producer_stamps(producer_blocks);
        DeviceBuffer<ConsumerStamp> consumer_stamps(consumer_blocks);
        DeviceBuffer<float> producer_sinks(
            static_cast<std::size_t>(producer_blocks) * kThreads);
        DeviceBuffer<float> consumer_sinks(
            static_cast<std::size_t>(consumer_blocks) * kThreads);
        cudaStream_t stream = nullptr;
        auto launch_pair = [&] {
            launch_pdl(producer_kernel, producer_blocks, stream,
                       producer_stamps.data(), producer_sinks.data(),
                       prefix_iterations, suffix_iterations);
            launch_pdl(consumer_kernel, consumer_blocks, stream,
                       consumer_stamps.data(), consumer_sinks.data(),
                       consumer_iterations);
        };
        for (int index = 0; index < options.warmup; ++index) launch_pair();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CudaEvent start_event;
        CudaEvent stop_event;
        std::vector<double> event_samples_ms;
        std::vector<double> overlap_samples_us;
        std::vector<double> suffix_samples_us;
        std::vector<double> wait_samples_us;
        std::vector<double> overlap_ratio_samples;
        event_samples_ms.reserve(options.samples);
        overlap_samples_us.reserve(options.samples);
        suffix_samples_us.reserve(options.samples);
        wait_samples_us.reserve(options.samples);
        overlap_ratio_samples.reserve(options.samples);
        for (int sample = 0; sample < options.samples; ++sample) {
            CUDA_CHECK(cudaEventRecord(start_event, stream));
            launch_pair();
            CUDA_CHECK(cudaEventRecord(stop_event, stream));
            CUDA_CHECK(cudaEventSynchronize(stop_event));
            float elapsed = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed, start_event, stop_event));
            event_samples_ms.push_back(elapsed);
            const auto producer = producer_stamps.copy_to_host();
            const auto consumer = consumer_stamps.copy_to_host();
            uint64_t trigger = 0;
            uint64_t producer_end = 0;
            for (const auto& stamp : producer) {
                trigger = std::max(trigger, stamp.trigger);
                producer_end = std::max(producer_end, stamp.end);
            }
            uint64_t consumer_start = std::numeric_limits<uint64_t>::max();
            uint64_t consumer_wait_end = 0;
            for (const auto& stamp : consumer) {
                consumer_start = std::min(consumer_start, stamp.start);
                consumer_wait_end = std::max(consumer_wait_end, stamp.wait_end);
            }
            const uint64_t overlap_start = std::max(trigger, consumer_start);
            const uint64_t overlap_ns = producer_end > overlap_start
                ? producer_end - overlap_start : 0;
            const uint64_t suffix_ns = producer_end > trigger
                ? producer_end - trigger : 0;
            const uint64_t wait_ns = consumer_wait_end > consumer_start
                ? consumer_wait_end - consumer_start : 0;
            overlap_samples_us.push_back(overlap_ns / 1000.0);
            suffix_samples_us.push_back(suffix_ns / 1000.0);
            wait_samples_us.push_back(wait_ns / 1000.0);
            overlap_ratio_samples.push_back(
                suffix_ns == 0 ? 0.0 : static_cast<double>(overlap_ns) / suffix_ns);
        }

        auto throughput_samples = event_samples_ms;
        for (double& value : throughput_samples) value = 1.0 / value / 1.0e6;
        const double elapsed_ms = median_copy(event_samples_ms);
        const double overlap_us = median_copy(overlap_samples_us);
        const double suffix_us = median_copy(suffix_samples_us);
        const double wait_us = median_copy(wait_samples_us);
        const double throughput_gpair = median_copy(throughput_samples);
        const double overlap_ratio = median_copy(overlap_ratio_samples);
        for (const float value : producer_sinks.copy_to_host()) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("PDL producer sink is not finite");
            }
        }
        for (const float value : consumer_sinks.copy_to_host()) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("PDL consumer sink is not finite");
            }
        }
        std::vector<uint32_t> producer_smids;
        for (const auto& stamp : producer_stamps.copy_to_host()) {
            producer_smids.push_back(stamp.smid);
        }
        std::vector<uint32_t> consumer_smids;
        for (const auto& stamp : consumer_stamps.copy_to_host()) {
            consumer_smids.push_back(stamp.smid);
        }
        std::set<uint32_t> active_smids(
            producer_smids.begin(), producer_smids.end());
        active_smids.insert(consumer_smids.begin(), consumer_smids.end());
        std::vector<double> overlap_samples_cycles = overlap_samples_us;
        for (double& value : overlap_samples_cycles) {
            value = value * clock_khz / 1000.0;
        }
        const double overlap_cycles = median_copy(overlap_samples_cycles);

        JsonObject params;
        params.add("gpu", properties.name)
            .add("resource", "grid")
            .add("producer_blocks", producer_blocks)
            .add("consumer_blocks", consumer_blocks)
            .add("threads", kThreads)
            .add("prefix_iters", prefix_iterations)
            .add("suffix_iters", suffix_iterations)
            .add("consumer_iters", consumer_iterations)
            .add("launch_attribute", "programmatic_stream_serialization")
            .add("producer_signal", "griddepcontrol.launch_dependents")
            .add("consumer_wait", "griddepcontrol.wait")
            .add("iters", options.iters).add("warmup", options.warmup)
            .add("samples", options.samples).add("blocks", options.blocks)
            .add("resolved_blocks", producer_blocks)
            .add("unique_active_sms", static_cast<int>(active_smids.size()))
            .add("initiation_interval_cycles", overlap_cycles)
            .add("device", options.device).add("peak", options.peak)
            .add("correct", true);
        JsonObject latency;
        latency.add("value", overlap_cycles).add("unit", "cycles/pair")
            .add("timer", "globaltimer")
            .add("scope", "producer_suffix_consumer_wait_overlap")
            .add("producer_suffix_us", suffix_us)
            .add("consumer_wait_us", wait_us)
            .add_raw("samples", json_number_array(overlap_samples_cycles))
            .add_raw("overlap_samples_us", json_number_array(overlap_samples_us))
            .add_raw("producer_suffix_samples_us", json_number_array(suffix_samples_us))
            .add_raw("consumer_wait_samples_us", json_number_array(wait_samples_us))
            .add_raw("producer_observed_smids",
                     json_number_array(producer_smids))
            .add_raw("consumer_observed_smids",
                     json_number_array(consumer_smids));
        JsonObject throughput;
        throughput.add("value", throughput_gpair).add("unit", "Gpair/s")
            .add("timer", "cuda_event").add("scope", "producer_consumer_pair")
            .add("event_ms", elapsed_ms)
            .add_raw("samples", json_number_array(throughput_samples))
            .add_raw("event_samples_ms", json_number_array(event_samples_ms));
        JsonObject bandwidth;
        bandwidth.add_null("value").add("unit", "GB/s")
            .add("reason", "PDL probe uses register-resident synthetic work");
        JsonObject hardware;
        hardware.add("value", overlap_ratio).add("unit", "ratio")
            .add("basis", "consumer scheduled wait overlap / producer suffix")
            .add_raw("samples", json_number_array(overlap_ratio_samples));
        print_result("griddepcontrol_producer_consumer", params, latency,
                     throughput, bandwidth, hardware);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "griddepcontrol_producer_consumer: " << error.what() << '\n';
        return 1;
    }
}

}  // namespace microbench::griddepcontrol_producer_consumer_bench
