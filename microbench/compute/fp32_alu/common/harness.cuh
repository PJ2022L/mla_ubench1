#pragma once

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "common/bench.hpp"

namespace microbench::scalar_atomic {

constexpr int kLatencyThreads = 32;
constexpr int kThroughputThreads = 256;
constexpr int kLatencyUnroll = 16;
constexpr int kThroughputChains = 8;

struct Ffma {
    using Value = float;
    static constexpr const char* kName = "ffma_rn_ftz_f32";
    static constexpr const char* kOpcode = "fma.rn.ftz.f32";
    static constexpr const char* kProtocol = "dependent FFMA recurrence";
    static constexpr bool kHasBaseline = false;
    __device__ static __forceinline__ Value target(Value x, uint32_t lane) {
        Value y;
        const float a = 0.9999847412109375f;
        const float b = 0.0000152587890625f * static_cast<float>((lane & 3U) + 1U);
        asm volatile("fma.rn.ftz.f32 %0, %1, %2, %3;"
                     : "=f"(y) : "f"(x), "f"(a), "f"(b));
        return y;
    }
    __device__ static __forceinline__ Value stabilize(Value x) { return x; }
    __device__ static __forceinline__ Value seed(uint32_t lane) {
        return 0.25f + static_cast<float>(lane & 15U) * 0.015625f;
    }
};

struct Fadd {
    using Value = float;
    static constexpr const char* kName = "fadd_rn_ftz_f32";
    static constexpr const char* kOpcode = "add.rn.ftz.f32";
    static constexpr const char* kProtocol = "dependent FADD recurrence";
    static constexpr bool kHasBaseline = false;
    __device__ static __forceinline__ Value target(Value x, uint32_t lane) {
        Value y;
        const float increment = 0.00000095367431640625f * static_cast<float>((lane & 3U) + 1U);
        asm volatile("add.rn.ftz.f32 %0, %1, %2;"
                     : "=f"(y) : "f"(x), "f"(increment));
        return y;
    }
    __device__ static __forceinline__ Value stabilize(Value x) { return x; }
    __device__ static __forceinline__ Value seed(uint32_t lane) {
        return 0.5f + static_cast<float>(lane & 15U) * 0.015625f;
    }
};

struct Fmul {
    using Value = float;
    static constexpr const char* kName = "fmul_rn_ftz_f32";
    static constexpr const char* kOpcode = "mul.rn.ftz.f32";
    static constexpr const char* kProtocol = "dependent FMUL recurrence";
    static constexpr bool kHasBaseline = false;
    __device__ static __forceinline__ Value target(Value x, uint32_t) {
        Value y;
        asm volatile("mul.rn.ftz.f32 %0, %1, %2;"
                     : "=f"(y) : "f"(x), "f"(0.99999904632568359375f));
        return y;
    }
    __device__ static __forceinline__ Value stabilize(Value x) { return x; }
    __device__ static __forceinline__ Value seed(uint32_t lane) {
        return 0.75f + static_cast<float>(lane & 15U) * 0.0078125f;
    }
};

struct Fmax {
    using Value = float;
    static constexpr const char* kName = "max_ftz_f32";
    static constexpr const char* kOpcode = "max.ftz.f32";
    static constexpr const char* kProtocol = "dependent FMAX recurrence";
    static constexpr bool kHasBaseline = false;
    __device__ static __forceinline__ Value target(Value x, uint32_t lane) {
        Value y;
        const float candidate =
            0.5f + static_cast<float>(lane & 3U) * 0.0625f;
        asm volatile("max.ftz.f32 %0, %1, %2;"
                     : "=f"(y) : "f"(x), "f"(candidate));
        return y;
    }
    __device__ static __forceinline__ Value stabilize(Value x) { return x; }
    __device__ static __forceinline__ Value seed(uint32_t lane) {
        return 0.375f + static_cast<float>(lane & 7U) * 0.03125f;
    }
};

struct Fsetp {
    using Value = float;
    static constexpr const char* kName = "fsetp_lt_ftz_f32";
    static constexpr const char* kOpcode = "setp.lt.ftz.f32";
    static constexpr const char* kProtocol = "FSETP predicate materialized by SELP to preserve a dependency chain";
    static constexpr bool kHasBaseline = false;
    __device__ static __forceinline__ Value target(Value x, uint32_t lane) {
        Value y;
        const float threshold = 0.625f + static_cast<float>(lane & 1U) * 0.125f;
        asm volatile("{ .reg .pred p; setp.lt.ftz.f32 p, %1, %2; selp.f32 %0, %3, %4, p; }"
                     : "=f"(y)
                     : "f"(x), "f"(threshold), "f"(0.75f), "f"(0.5f));
        return y;
    }
    __device__ static __forceinline__ Value stabilize(Value x) { return x; }
    __device__ static __forceinline__ Value seed(uint32_t lane) {
        return 0.5f + static_cast<float>(lane & 3U) * 0.125f;
    }
};

struct Fsel {
    using Value = float;
    static constexpr const char* kName = "fsel_f32";
    static constexpr const char* kOpcode = "selp.f32";
    static constexpr const char* kProtocol = "FSEL target with an integer predicate setup in the same inline PTX block";
    static constexpr bool kHasBaseline = false;
    __device__ static __forceinline__ Value target(Value x, uint32_t lane) {
        Value y;
        const uint32_t selector = lane ^ __float_as_uint(x);
        asm volatile(
            "{ .reg .pred p; setp.ne.u32 p, %3, 0; selp.f32 %0, %1, %2, p; }"
            : "=f"(y) : "f"(0.75f), "f"(0.5f), "r"(selector));
        return y;
    }
    __device__ static __forceinline__ Value stabilize(Value x) { return x; }
    __device__ static __forceinline__ Value seed(uint32_t lane) {
        return 0.25f + static_cast<float>(lane & 15U) * 0.03125f;
    }
};

template <class Atom, class = void>
struct AtomDelta {
    static constexpr bool kPresent = false;
};

template <class Atom>
struct AtomDelta<Atom, std::void_t<decltype(Atom::kDelta)>> {
    static constexpr bool kPresent = true;
    static constexpr int kValue = Atom::kDelta;
};

template <class Atom, bool WithTarget>
__device__ __forceinline__ typename Atom::Value step(typename Atom::Value x,
                                                     uint32_t lane) {
    if constexpr (WithTarget) {
        x = Atom::target(x, lane);
    }
    return Atom::stabilize(x);
}

#define MB_SCALAR_REPEAT_16(expression) \
    expression; expression; expression; expression; \
    expression; expression; expression; expression; \
    expression; expression; expression; expression; \
    expression; expression; expression; expression

template <class Atom, bool WithTarget>
__global__ void latency_kernel(uint64_t* cycles,
                               typename Atom::Value* sink,
                               int iters) {
    using Value = typename Atom::Value;
    const uint32_t lane = threadIdx.x & 31U;
    Value value = Atom::seed(lane);
    asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
    const uint64_t start = microbench::read_clock64();
#pragma unroll 1
    for (int iteration = 0; iteration < iters; ++iteration) {
        MB_SCALAR_REPEAT_16((value = step<Atom, WithTarget>(value, lane)));
    }
    const uint64_t stop = microbench::read_clock64();
    sink[threadIdx.x] = value;
    if (threadIdx.x == 0) {
        cycles[0] = stop - start;
    }
}

template <class Atom>
__global__ __launch_bounds__(kThroughputThreads)
void throughput_kernel(typename Atom::Value* sink, int iters) {
    using Value = typename Atom::Value;
    const std::size_t global_thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint32_t lane = threadIdx.x & 31U;
    const Value seed = Atom::seed(static_cast<uint32_t>(global_thread));
    Value x0 = seed;
    Value x1 = seed + static_cast<Value>(1);
    Value x2 = seed + static_cast<Value>(2);
    Value x3 = seed + static_cast<Value>(3);
    Value x4 = seed + static_cast<Value>(4);
    Value x5 = seed + static_cast<Value>(5);
    Value x6 = seed + static_cast<Value>(6);
    Value x7 = seed + static_cast<Value>(7);
#pragma unroll 1
    for (int iteration = 0; iteration < iters; ++iteration) {
        x0 = step<Atom, true>(x0, lane + 0U);
        x1 = step<Atom, true>(x1, lane + 1U);
        x2 = step<Atom, true>(x2, lane + 2U);
        x3 = step<Atom, true>(x3, lane + 3U);
        x4 = step<Atom, true>(x4, lane + 4U);
        x5 = step<Atom, true>(x5, lane + 5U);
        x6 = step<Atom, true>(x6, lane + 6U);
        x7 = step<Atom, true>(x7, lane + 7U);
    }
    sink[global_thread] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}

#undef MB_SCALAR_REPEAT_16

template <class Value>
void validate_sink(const std::vector<Value>& values, const char* label) {
    for (const Value value : values) {
        if constexpr (std::is_floating_point_v<Value>) {
            if (!std::isfinite(value)) {
                throw std::runtime_error(std::string(label) + " sink is not finite");
            }
        }
        volatile Value consumed = value;
        (void)consumed;
    }
}

template <class Atom>
int run(int argc, char** argv) {
    try {
        const Args args(argc, argv);
        if constexpr (AtomDelta<Atom>::kPresent) {
            args.require_only({"iters", "warmup", "samples", "blocks", "peak", "device", "delta"});
            if (args.get_int("delta", AtomDelta<Atom>::kValue, 1, 16) !=
                AtomDelta<Atom>::kValue) {
                throw std::invalid_argument("compiled SHFL delta does not match --delta");
            }
        } else {
            args.require_only({"iters", "warmup", "samples", "blocks", "peak", "device"});
        }
        const CommonOptions options = parse_common_options(args);
        const cudaDeviceProp device = require_sm90(options.device);
        const int blocks = resolve_blocks(options.blocks, device);
        using Value = typename Atom::Value;

        DeviceBuffer<uint64_t> cycles(1);
        DeviceBuffer<Value> latency_sink(kLatencyThreads);
        DeviceBuffer<Value> baseline_sink(kLatencyThreads);
        DeviceBuffer<Value> throughput_sink(
            static_cast<std::size_t>(blocks) * kThroughputThreads);

        const auto target_samples = measure_clock_cycles(
            options.warmup, options.samples, cycles.data(), [&] {
                latency_kernel<Atom, true><<<1, kLatencyThreads>>>(
                    cycles.data(), latency_sink.data(), options.iters);
            });
        validate_sink(latency_sink.copy_to_host(), "latency");
        const double raw_cycles = median(target_samples);
        double baseline_cycles = 0.0;
        std::vector<double> baseline_samples;
        if constexpr (Atom::kHasBaseline) {
            baseline_samples = measure_clock_cycles(
                options.warmup, options.samples, cycles.data(), [&] {
                    latency_kernel<Atom, false><<<1, kLatencyThreads>>>(
                        cycles.data(), baseline_sink.data(), options.iters);
                });
            validate_sink(baseline_sink.copy_to_host(), "baseline");
            baseline_cycles = median(baseline_samples);
            if (!(raw_cycles > baseline_cycles)) {
                throw std::runtime_error("target latency did not exceed its matched baseline");
            }
        }
        const double operations = static_cast<double>(options.iters) * kLatencyUnroll;
        const double cycles_per_op = (raw_cycles - baseline_cycles) / operations;
        std::vector<double> net_cycles_per_op_samples;
        net_cycles_per_op_samples.reserve(target_samples.size());
        for (std::size_t index = 0; index < target_samples.size(); ++index) {
            const double baseline = baseline_samples.empty() ? 0.0 : baseline_samples[index];
            net_cycles_per_op_samples.push_back(
                (target_samples[index] - baseline) / operations);
        }

        const auto event_samples = measure_event_ms(
            options.warmup, options.samples, [&] {
                throughput_kernel<Atom><<<blocks, kThroughputThreads>>>(
                    throughput_sink.data(), options.iters);
            });
        const double elapsed_ms = median(event_samples);
        validate_sink(throughput_sink.copy_to_host(), "throughput");
        const double lane_operations = static_cast<double>(blocks) *
            kThroughputThreads * options.iters * kThroughputChains;
        const double throughput_gops = lane_operations / (elapsed_ms * 1.0e6);
        std::vector<double> throughput_samples;
        throughput_samples.reserve(event_samples.size());
        for (const double sample_ms : event_samples) {
            throughput_samples.push_back(lane_operations / (sample_ms * 1.0e6));
        }

        JsonObject params;
        params.add("iters", options.iters)
            .add("warmup", options.warmup)
            .add("samples", options.samples)
            .add("blocks", options.blocks)
            .add("resolved_blocks", blocks)
            .add("threads", kThroughputThreads)
            .add("latency_unroll", kLatencyUnroll)
            .add("throughput_chains", kThroughputChains)
            .add("target_ptx", Atom::kOpcode)
            .add("protocol", Atom::kProtocol)
            .add("peak", options.peak)
            .add("peak_unit", "Glane-op/s")
            .add("device", options.device)
            .add("gpu", device.name);
        if constexpr (AtomDelta<Atom>::kPresent) {
            params.add("delta", AtomDelta<Atom>::kValue);
        }

        JsonObject latency;
        latency.add("value", cycles_per_op)
            .add("unit", "cycles/target-op")
            .add("timer", "clock64")
            .add("scope", "single_warp_dependency_chain")
            .add("raw_median_cycles", raw_cycles)
            .add("baseline_median_cycles", baseline_cycles)
            .add("operations", operations)
            .add("protocol", Atom::kProtocol)
            .add_raw("samples", json_number_array(net_cycles_per_op_samples))
            .add_raw("target_samples_cycles", json_number_array(target_samples))
            .add_raw("baseline_samples_cycles", json_number_array(baseline_samples));
        JsonObject throughput = metric(throughput_gops, "Glane-op/s");
        throughput.add("timer", "cuda_event")
            .add("scope", "grid_independent_chains")
            .add("median_ms", elapsed_ms)
            .add("lane_operations", lane_operations)
            .add_raw("samples", json_number_array(throughput_samples))
            .add_raw("event_samples_ms", json_number_array(event_samples));
        JsonObject bandwidth;
        bandwidth.add_null("value")
            .add("unit", "GB/s")
            .add("reason", "register-only target; final sink traffic excluded");

        print_result(Atom::kName, params, latency, throughput, bandwidth,
                     utilization(throughput_gops, options.peak, "Glane-op/s"));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "scalar atomic benchmark error: " << error.what() << '\n';
        return 1;
    }
}

}  // namespace microbench::scalar_atomic
