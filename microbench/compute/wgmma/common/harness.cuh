#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/bench.hpp"
#include "ptx.cuh"

namespace microbench::wgmma_bench {

enum class Operation { kM64N64Ss, kM64N64Rs, kM64N256Ss, kM64N256Rs };

template <Operation Op>
struct Traits;

template <>
struct Traits<Operation::kM64N64Ss> {
    static constexpr int kN = 64;
    static constexpr bool kRegisterA = false;
    static constexpr const char* kOp = "m64n64k16_ss";
};

template <>
struct Traits<Operation::kM64N64Rs> {
    static constexpr int kN = 64;
    static constexpr bool kRegisterA = true;
    static constexpr const char* kOp = "m64n64k16_rs";
};

template <>
struct Traits<Operation::kM64N256Ss> {
    static constexpr int kN = 256;
    static constexpr bool kRegisterA = false;
    static constexpr const char* kOp = "m64n256k16_ss";
};

template <>
struct Traits<Operation::kM64N256Rs> {
    static constexpr int kN = 256;
    static constexpr bool kRegisterA = true;
    static constexpr const char* kOp = "m64n256k16_rs";
};

constexpr int kWarpgroupThreads = 128;
constexpr int kM = 64;
constexpr int kK = 16;
constexpr int kSharedM64N64Bytes = 64 * 64 * 2;
constexpr int kSharedM64N256Bytes = 256 * 64 * 2;
constexpr double kContribution = 1.0 / 65536.0;
constexpr double kMaxExactFp32Increments = 16777216.0;

#if defined(MB1_WGMMA_USE_F16)
constexpr uint32_t kPackedInput = 0x14001400u;
constexpr const char* kDtype = "fp16";
#else
constexpr uint32_t kPackedInput = 0x3a803a80u;
constexpr const char* kDtype = "bf16";
#endif

inline double expected_sink(double instructions) {
    return std::min(instructions, kMaxExactFp32Increments) * kContribution;
}

inline void validate_sinks(const std::vector<float>& values,
                           double expected,
                           const char* phase) {
    const double tolerance = std::max(1.0e-6, std::abs(expected) * 1.0e-5);
    for (const float value : values) {
        if (!std::isfinite(value) ||
            std::abs(static_cast<double>(value) - expected) > tolerance) {
            throw std::runtime_error(
                std::string("WGMMA ") + phase + " sink validation failed");
        }
    }
}

template <Operation Op>
__host__ __device__ constexpr int shared_a_bytes() {
    return Traits<Op>::kRegisterA ? 0 : kSharedM64N64Bytes;
}

template <Operation Op>
__host__ __device__ constexpr int shared_b_bytes() {
    return Traits<Op>::kN == 64 ? kSharedM64N64Bytes : kSharedM64N256Bytes;
}

template <Operation Op>
__host__ __device__ constexpr int shared_bytes() {
    return shared_a_bytes<Op>() + shared_b_bytes<Op>();
}

template <Operation Op, int GroupSize, int Depth>
__global__ void kernel(uint64_t* cycles,
                       uint64_t* baseline_cycles,
                       float* sinks,
                       uint32_t iterations,
                       int warpgroups) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    extern __shared__ __align__(1024) unsigned char storage[];
    const int warpgroup = threadIdx.x / kWarpgroupThreads;
    const int lane = threadIdx.x % kWarpgroupThreads;
    unsigned char* warpgroup_storage =
        storage + warpgroup * shared_bytes<Op>();
    auto* operand_a = reinterpret_cast<uint32_t*>(warpgroup_storage);
    unsigned char* operand_b = warpgroup_storage + shared_a_bytes<Op>();
    auto* words = reinterpret_cast<uint32_t*>(warpgroup_storage);
    for (int word = lane; word < shared_bytes<Op>() / 4;
         word += kWarpgroupThreads) {
        words[word] = kPackedInput;
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    const uint64_t desc_a = ptx::make_sw128_kmajor_descriptor(operand_a);
    const uint64_t desc_b = Traits<Op>::kN == 64
        ? ptx::make_sw128_kmajor_descriptor(operand_b)
        : ptx::make_sw128_transposed_descriptor(operand_b);
    uint64_t start = 0;
    uint64_t stop = 0;
    float sink = 0.0f;
    if constexpr (Op == Operation::kM64N64Ss) {
        ptx::run_m64n64k16_ss<GroupSize, Depth>(
            desc_a, desc_b, iterations, start, stop, sink);
    } else if constexpr (Op == Operation::kM64N64Rs) {
        ptx::run_m64n64k16_rs<GroupSize, Depth>(
            desc_b, kPackedInput, kPackedInput, kPackedInput, kPackedInput,
            iterations, start, stop, sink);
    } else if constexpr (Op == Operation::kM64N256Ss) {
        ptx::run_m64n256k16_ss<GroupSize, Depth>(
            desc_a, desc_b, iterations, start, stop, sink);
    } else {
        ptx::run_m64n256k16_rs<GroupSize, Depth>(
            desc_b, kPackedInput, kPackedInput, kPackedInput, kPackedInput,
            iterations, start, stop, sink);
    }
    if (baseline_cycles != nullptr) {
        uint64_t baseline_start = 0;
        uint64_t baseline_stop = 0;
        uint32_t baseline_sink = 0;
        ptx::run_loop_control_baseline(
            iterations, baseline_start, baseline_stop, baseline_sink);
        if (lane == 0) {
            const int index =
                static_cast<int>(blockIdx.x) * warpgroups + warpgroup;
            baseline_cycles[index] = baseline_stop - baseline_start;
        }
        sink += static_cast<float>(baseline_sink);
    }
    if (lane == 0) {
        const int index = static_cast<int>(blockIdx.x) * warpgroups + warpgroup;
        cycles[index] = stop - start;
        sinks[index] = sink;
    }
#else
    if (threadIdx.x == 0) {
        cycles[blockIdx.x] = 0;
        sinks[blockIdx.x] = 0.0f;
    }
#endif
}

template <Operation Op, int GroupSize, int Depth>
inline void configure(int bytes) {
    CUDA_CHECK(cudaFuncSetAttribute(
        kernel<Op, GroupSize, Depth>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}

template <Operation Op>
inline void configure_for(int bytes, int group_size, int depth) {
#define MB1_CONFIG(G, D) configure<Op, G, D>(bytes)
    if constexpr (Op == Operation::kM64N64Ss) {
        if (group_size == 36 && depth == 1) {
            MB1_CONFIG(36, 1);
            return;
        }
    }
    if (group_size == 1 && depth == 1) MB1_CONFIG(1, 1);
    else if (group_size == 1 && depth == 2) MB1_CONFIG(1, 2);
    else if (group_size == 1 && depth == 4) MB1_CONFIG(1, 4);
    else if (group_size == 4 && depth == 1) MB1_CONFIG(4, 1);
    else if (group_size == 4 && depth == 2) MB1_CONFIG(4, 2);
    else if (group_size == 4 && depth == 4) MB1_CONFIG(4, 4);
    else throw std::invalid_argument(
        "WGMMA supports --group-size=1|4 and --depth=1|2|4");
#undef MB1_CONFIG
}

template <Operation Op, int GroupSize, int Depth>
inline void launch_fixed(int blocks,
                         int threads,
                         int bytes,
                         uint64_t* cycles,
                         uint64_t* baseline_cycles,
                         float* sinks,
                         uint32_t iterations,
                         int warpgroups) {
    kernel<Op, GroupSize, Depth><<<blocks, threads, bytes>>>(
        cycles, baseline_cycles, sinks, iterations, warpgroups);
}

template <Operation Op>
inline void launch(int blocks,
                   int threads,
                   int bytes,
                   uint64_t* cycles,
                   uint64_t* baseline_cycles,
                   float* sinks,
                   uint32_t iterations,
                   int group_size,
                   int depth,
                   int warpgroups) {
#define MB1_LAUNCH(G, D) \
    launch_fixed<Op, G, D>(blocks, threads, bytes, cycles, baseline_cycles, \
                           sinks, \
                           iterations, warpgroups)
    if constexpr (Op == Operation::kM64N64Ss) {
        if (group_size == 36 && depth == 1) {
            MB1_LAUNCH(36, 1);
            return;
        }
    }
    if (group_size == 1 && depth == 1) MB1_LAUNCH(1, 1);
    else if (group_size == 1 && depth == 2) MB1_LAUNCH(1, 2);
    else if (group_size == 1 && depth == 4) MB1_LAUNCH(1, 4);
    else if (group_size == 4 && depth == 1) MB1_LAUNCH(4, 1);
    else if (group_size == 4 && depth == 2) MB1_LAUNCH(4, 2);
    else if (group_size == 4 && depth == 4) MB1_LAUNCH(4, 4);
    else throw std::invalid_argument("unsupported WGMMA issue configuration");
#undef MB1_LAUNCH
}

template <Operation Op>
int run(int argc, char** argv) {
    try {
        const Args args(argc, argv);
        args.require_only({
            "iters", "warmup", "samples", "blocks", "device", "peak",
            "warpgroups", "group-size", "depth"});
        const auto options = parse_common_options(args, 4096);
        const auto properties = require_sm90(options.device);
        const int warpgroups = args.get_int("warpgroups", 1, 1, 2);
        const int group_size = args.get_int("group-size", 4, 1, 36);
        const int depth = args.get_int("depth", 1, 1, 4);
        constexpr bool kSupportsGroup36 = Op == Operation::kM64N64Ss;
        const bool valid_group = group_size == 1 || group_size == 4 ||
            (kSupportsGroup36 && group_size == 36);
        const bool valid_depth = depth == 1 || depth == 2 || depth == 4;
        if (!valid_group || !valid_depth || (group_size == 36 && depth != 1)) {
            throw std::invalid_argument(
                "WGMMA supports group_size=1|4 at depth=1|2|4; "
                "m64n64 shared/shared also supports group_size=36 at depth=1");
        }
        const int blocks = resolve_blocks(options.blocks, properties, 1);
        const int threads = warpgroups * kWarpgroupThreads;
        const int bytes = warpgroups * shared_bytes<Op>();
        if (bytes > static_cast<int>(properties.sharedMemPerBlockOptin)) {
            throw std::invalid_argument(
                "requested WGMMA storage exceeds sharedMemPerBlockOptin");
        }
        configure_for<Op>(bytes, group_size, 1);
        configure_for<Op>(bytes, group_size, depth);

        DeviceBuffer<uint64_t> latency_cycles(warpgroups);
        DeviceBuffer<uint64_t> latency_baseline_cycles(warpgroups);
        DeviceBuffer<float> latency_sinks(warpgroups);
        const auto latency_clock_samples = measure_paired_clock_cycles(
            options.warmup, options.samples, latency_cycles.data(),
            latency_baseline_cycles.data(), warpgroups, [&] {
                launch<Op>(1, threads, bytes, latency_cycles.data(),
                           latency_baseline_cycles.data(), latency_sinks.data(),
                           options.iters, group_size, 1, warpgroups);
            });
        std::vector<double> latency_metric_samples;
        latency_metric_samples.reserve(options.samples);
        for (int index = 0; index < options.samples; ++index) {
            latency_metric_samples.push_back(
                (latency_clock_samples.target[index] -
                 latency_clock_samples.baseline[index]) / options.iters);
        }
        const double protocol_cycles = median(latency_metric_samples);
        validate_sinks(latency_sinks.copy_to_host(),
                       expected_sink(
                           1.0 + static_cast<double>(options.iters) * group_size),
                       "latency");

        const std::size_t result_count =
            static_cast<std::size_t>(blocks) * warpgroups;
        DeviceBuffer<uint64_t> throughput_cycles(result_count);
        DeviceBuffer<uint64_t> throughput_baseline_cycles(result_count);
        DeviceBuffer<float> throughput_sinks(result_count);
        const auto initiation_clock_samples = measure_paired_clock_cycles(
            options.warmup, options.samples, throughput_cycles.data(),
            throughput_baseline_cycles.data(), result_count, [&] {
                launch<Op>(blocks, threads, bytes, throughput_cycles.data(),
                           throughput_baseline_cycles.data(),
                           throughput_sinks.data(), options.iters, group_size,
                           depth, warpgroups);
            });
        std::vector<double> initiation_interval_samples;
        initiation_interval_samples.reserve(options.samples);
        for (int index = 0; index < options.samples; ++index) {
            initiation_interval_samples.push_back(
                (initiation_clock_samples.target[index] -
                 initiation_clock_samples.baseline[index]) / options.iters);
        }
        const double initiation_interval_cycles =
            median(initiation_interval_samples);
        const auto event_samples = measure_event_ms(
            options.warmup, options.samples, [&] {
                launch<Op>(blocks, threads, bytes, throughput_cycles.data(),
                           nullptr, throughput_sinks.data(), options.iters,
                           group_size, depth, warpgroups);
            });
        const double elapsed_ms = median(event_samples);
        const double instructions_per_warpgroup =
            1.0 + static_cast<double>(options.iters) * group_size;
        validate_sinks(throughput_sinks.copy_to_host(),
                       expected_sink(instructions_per_warpgroup), "throughput");

        const double target_instructions =
            static_cast<double>(blocks) * warpgroups * options.iters * group_size;
        const double all_instructions = target_instructions +
            static_cast<double>(blocks) * warpgroups;
        const double flop_per_instruction = 2.0 * kM * Traits<Op>::kN * kK;
        auto throughput_metric_samples = event_samples;
        for (double& value : throughput_metric_samples) {
            value = all_instructions * flop_per_instruction / value / 1.0e9;
        }
        const double achieved_tflops = median(throughput_metric_samples);

        JsonObject params;
        params.add("gpu", properties.name)
            .add("op", Traits<Op>::kOp)
            .add("m", kM).add("n", Traits<Op>::kN).add("k", kK)
            .add("input_dtype", kDtype)
            .add("accumulator_dtype", "f32")
            .add("operand_a", Traits<Op>::kRegisterA ? "register" : "shared")
            .add("operand_b", "shared")
            .add("source_mode", Traits<Op>::kRegisterA ? "rs" : "ss")
            .add("a_major", Traits<Op>::kRegisterA ? "register" : "k_major")
            .add("b_major", Traits<Op>::kN == 64 ? "k_major" : "transposed_k")
            .add("swizzle", "128B")
            .add("transpose", Traits<Op>::kN == 256 ? "b" : "none")
            .add("transpose_a", false)
            .add("transpose_b", Traits<Op>::kN == 256)
            .add("scale_modifier", "scale_a=1,scale_b=1,scale_d=predicate")
            .add("warpgroups", warpgroups)
            .add("group_size", group_size)
            .add("depth", depth)
            .add("latency_group_size", group_size)
            .add("latency_issue_depth", 1)
            .add("initiation_interval_group_size", group_size)
            .add("initiation_interval_issue_depth", depth)
            .add("clock_baseline", "matched inline-PTX add+setp+branch loop")
            .add("initiation_interval_cycles", initiation_interval_cycles)
            .add("iters", options.iters)
            .add("warmup", options.warmup)
            .add("samples", options.samples)
            .add("blocks", options.blocks)
            .add("resolved_blocks", blocks)
            .add("device", options.device)
            .add("peak", options.peak);

        JsonObject latency;
        latency.add("value", protocol_cycles)
            .add("unit", "cycle/committed_group")
            .add("timer", "clock64")
            .add("scope", "max_across_resident_warpgroups")
            .add("boundary", "selected_group_size+commit+wait0; dependency_depth=1")
            .add_raw("samples", json_number_array(latency_metric_samples))
            .add_raw("raw_target_samples_cycles",
                     json_number_array(latency_clock_samples.target))
            .add_raw("loop_baseline_samples_cycles",
                     json_number_array(latency_clock_samples.baseline));
        JsonObject throughput;
        throughput.add("value", achieved_tflops)
            .add("unit", "TFLOP/s")
            .add("timer", "cuda_event")
            .add("scope", "grid")
            .add("event_ms", elapsed_ms)
            .add("instructions", all_instructions)
            .add_raw("initiation_interval_samples_cycles",
                     json_number_array(initiation_interval_samples))
            .add_raw("samples", json_number_array(throughput_metric_samples))
            .add_raw("event_samples_ms", json_number_array(event_samples));
        JsonObject bandwidth;
        bandwidth.add_null("value").add("unit", "GB/s")
            .add("reason", "WGMMA benchmark excludes global-memory traffic");

        const std::string name = std::string() +
            Traits<Op>::kOp + "_" + kDtype;
        print_result(name, params, latency, throughput, bandwidth,
                     utilization(achieved_tflops, options.peak, "TFLOP/s"));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << Traits<Op>::kOp << '_' << kDtype << ": "
                  << error.what() << '\n';
        return 1;
    }
}

}  // namespace microbench::wgmma_bench
