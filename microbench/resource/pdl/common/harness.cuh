#pragma once

#include <cstdint>
#include <exception>
#include <iostream>
#include <string>

#include "common/bench.hpp"

namespace microbench::pdl_atomic {

enum class Operation { kLaunchDependents, kWait };

template <Operation Op>
struct Traits;

template <>
struct Traits<Operation::kLaunchDependents> {
    static constexpr const char* kName = "griddepcontrol_launch_dependents";
    static constexpr const char* kPtx = "griddepcontrol.launch_dependents";
};

template <>
struct Traits<Operation::kWait> {
    static constexpr const char* kName = "griddepcontrol_wait";
    static constexpr const char* kPtx = "griddepcontrol.wait";
};

template <Operation Op, bool Target = true>
__global__ void kernel(uint64_t* cycles, uint64_t* sinks, int iterations) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    if (threadIdx.x == 0) {
        const uint64_t start = read_clock64();
#pragma unroll 1
        for (int iteration = 0; iteration < iterations; ++iteration) {
            if constexpr (Target) {
                if constexpr (Op == Operation::kLaunchDependents) {
                    asm volatile(
                        "griddepcontrol.launch_dependents;" ::: "memory");
                } else {
                    asm volatile("griddepcontrol.wait;" ::: "memory");
                }
            } else {
                asm volatile("" : : "r"(iteration) : "memory");
            }
        }
        const uint64_t stop = read_clock64();
        cycles[blockIdx.x] = stop - start;
        sinks[blockIdx.x] = stop ^ static_cast<uint64_t>(blockIdx.x + 1);
    }
#else
    if (threadIdx.x == 0) {
        cycles[blockIdx.x] = 0;
        sinks[blockIdx.x] = 0;
    }
#endif
}

template <Operation Op, bool Target = true>
inline void launch(int blocks,
                   cudaStream_t stream,
                   uint64_t* cycles,
                   uint64_t* sinks,
                   int iterations) {
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{
        dim3(blocks), dim3(32), 0, stream, &attribute, 1};
    CUDA_CHECK(cudaLaunchKernelEx(
        &config, kernel<Op, Target>, cycles, sinks, iterations));
}

template <Operation Op>
int run(int argc, char** argv) {
    try {
        const Args args(argc, argv);
        args.require_only(
            {"iters", "warmup", "samples", "blocks", "device", "peak"});
        const CommonOptions options = parse_common_options(args, 1024);
        const cudaDeviceProp properties = require_sm90(options.device);
        const int blocks = resolve_blocks(options.blocks, properties, 1);
        DeviceBuffer<uint64_t> latency_cycles(1);
        DeviceBuffer<uint64_t> latency_baseline_cycles(1);
        DeviceBuffer<uint64_t> latency_sink(1);
        DeviceBuffer<uint64_t> latency_baseline_sink(1);
        DeviceBuffer<uint64_t> throughput_cycles(blocks);
        DeviceBuffer<uint64_t> throughput_sinks(blocks);
        cudaStream_t stream = nullptr;

        const auto latency_clock_samples = measure_paired_clock_cycles(
            options.warmup, options.samples, latency_cycles.data(),
            latency_baseline_cycles.data(), 1, [&] {
                launch<Op, true>(1, stream, latency_cycles.data(),
                                 latency_sink.data(), options.iters);
                launch<Op, false>(1, stream, latency_baseline_cycles.data(),
                                  latency_baseline_sink.data(), options.iters);
            });
        std::vector<double> latency_samples;
        latency_samples.reserve(options.samples);
        for (int index = 0; index < options.samples; ++index) {
            latency_samples.push_back(
                (latency_clock_samples.target[index] -
                 latency_clock_samples.baseline[index]) /
                options.iters);
        }
        const auto event_samples = measure_event_ms(
            options.warmup, options.samples, [&] {
                launch<Op, true>(blocks, stream, throughput_cycles.data(),
                                 throughput_sinks.data(), options.iters);
            });
        const double elapsed_ms = median(event_samples);
        const double instructions =
            static_cast<double>(blocks) * options.iters;
        std::vector<double> throughput_samples = event_samples;
        for (double& value : throughput_samples) {
            value = instructions / value / 1.0e6;
        }

        JsonObject params;
        params.add("gpu", properties.name)
            .add("resource", "grid")
            .add("iters", options.iters)
            .add("warmup", options.warmup)
            .add("samples", options.samples)
            .add("blocks", options.blocks)
            .add("resolved_blocks", blocks)
            .add("threads", 32)
            .add("target_ptx", Traits<Op>::kPtx)
            .add("initiation_interval_cycles", median(latency_samples))
            .add("clock_baseline",
                 "same programmatic launch attribute and loop; no timed "
                 "griddepcontrol instruction")
            .add("device", options.device)
            .add("peak", options.peak);
        JsonObject latency;
        latency.add("value", median(latency_samples))
            .add("unit", "cycles/instruction")
            .add("timer", "clock64")
            .add("scope", "single_cta")
            .add("boundary", "ready-path instruction minus matched loop")
            .add_raw("samples", json_number_array(latency_samples))
            .add_raw("target_samples_cycles",
                     json_number_array(latency_clock_samples.target))
            .add_raw("baseline_samples_cycles",
                     json_number_array(latency_clock_samples.baseline));
        JsonObject throughput = metric(
            median(throughput_samples), "Ginst/s");
        throughput.add("timer", "cuda_event")
            .add("scope", "grid")
            .add("event_ms", elapsed_ms)
            .add_raw("samples", json_number_array(throughput_samples));
        JsonObject bandwidth;
        bandwidth.add_null("value").add("unit", "GB/s")
            .add("reason", "instruction has no payload bytes");
        print_result(Traits<Op>::kName, params, latency, throughput, bandwidth,
                     utilization(median(throughput_samples), options.peak,
                                 "Ginst/s"));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << Traits<Op>::kName << ": " << error.what() << '\n';
        return 1;
    }
}

}  // namespace microbench::pdl_atomic
