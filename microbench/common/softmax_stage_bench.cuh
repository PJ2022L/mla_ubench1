#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "bench.hpp"

namespace microbench::softmax_stage_bench {

constexpr int kThreads = 256;
constexpr int kWarpgroupThreads = 128;
constexpr int kScoresPerLane = 32;

#define MB1_SOFTMAX_SCORE_LIST(M) \
    M(0)  M(1)  M(2)  M(3)  M(4)  M(5)  M(6)  M(7)  \
    M(8)  M(9)  M(10) M(11) M(12) M(13) M(14) M(15) \
    M(16) M(17) M(18) M(19) M(20) M(21) M(22) M(23) \
    M(24) M(25) M(26) M(27) M(28) M(29) M(30) M(31)

#define MB1_SOFTMAX_SCORE_TAIL_LIST(M) \
    M(1)  M(2)  M(3)  M(4)  M(5)  M(6)  M(7)  M(8)  \
    M(9)  M(10) M(11) M(12) M(13) M(14) M(15) M(16) \
    M(17) M(18) M(19) M(20) M(21) M(22) M(23) M(24) \
    M(25) M(26) M(27) M(28) M(29) M(30) M(31)

#if defined(MB1_SOFTMAX_USE_F16)
constexpr const char* kDtype = "fp16";
#else
constexpr const char* kDtype = "bf16";
#endif

__device__ __forceinline__ float select_max(float lhs, float rhs) {
    float result;
    asm volatile(
        "{ .reg .pred p; setp.gt.ftz.f32 p, %1, %2; "
        "selp.f32 %0, %1, %2, p; }"
        : "=f"(result) : "f"(lhs), "f"(rhs));
    return result;
}

__device__ __forceinline__ float shfl_xor(float value, int delta) {
    float result;
    asm volatile(
        "shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
        : "=f"(result) : "f"(value), "r"(delta));
    return result;
}

__device__ __forceinline__ float exp2_value(float value) {
    float result;
    asm volatile("ex2.approx.ftz.f32 %0, %1;"
                 : "=f"(result) : "f"(value));
    return result;
}

__device__ __forceinline__ float reciprocal(float value) {
    float result;
    asm volatile("rcp.approx.ftz.f32 %0, %1;"
                 : "=f"(result) : "f"(value));
    return result;
}

__device__ __forceinline__ float ffma(float a, float b, float c) {
    float result;
    asm volatile("fma.rn.ftz.f32 %0, %1, %2, %3;"
                 : "=f"(result) : "f"(a), "f"(b), "f"(c));
    return result;
}

__device__ __forceinline__ uint16_t convert(float value) {
    uint16_t result;
#if defined(MB1_SOFTMAX_USE_F16)
    asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(value));
#else
    asm volatile("cvt.rn.bf16.f32 %0, %1;" : "=h"(result) : "f"(value));
#endif
    return result;
}

__global__ __launch_bounds__(kThreads, 1)
void softmax_stage_kernel(uint64_t* cycles,
                          float* sinks,
                          uint32_t* smids,
                          int iterations) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    const int warpgroup = static_cast<int>(threadIdx.x) / kWarpgroupThreads;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    float seed = -0.25f - static_cast<float>(lane) * 0.00390625f;
#define MB1_DECLARE_SCORE(index) \
    float score##index = seed - static_cast<float>(index) * 0.015625f;
    MB1_SOFTMAX_SCORE_LIST(MB1_DECLARE_SCORE)
#undef MB1_DECLARE_SCORE
    uint32_t checksum = static_cast<uint32_t>(threadIdx.x + 1);
    asm volatile("bar.sync 0;" ::: "memory");
    const uint64_t start = read_clock64();
#pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        float row_max = score0;
#define MB1_ACCUMULATE_MAX(index) \
        row_max = select_max(row_max, score##index);
        MB1_SOFTMAX_SCORE_TAIL_LIST(MB1_ACCUMULATE_MAX)
#undef MB1_ACCUMULATE_MAX
#pragma unroll
        for (int delta = 16; delta >= 1; delta /= 2) {
            row_max = select_max(row_max, shfl_xor(row_max, delta));
        }
        float row_sum = 0.0f;
#define MB1_EXP_AND_SUM(index) \
        score##index = exp2_value( \
            ffma(score##index - row_max, 1.4426950408889634f, 0.0f)); \
        row_sum += score##index;
        MB1_SOFTMAX_SCORE_LIST(MB1_EXP_AND_SUM)
#undef MB1_EXP_AND_SUM
#pragma unroll
        for (int delta = 16; delta >= 1; delta /= 2) {
            row_sum += shfl_xor(row_sum, delta);
        }
        const float inverse = reciprocal(row_sum);
#define MB1_CONVERT_AND_UPDATE(index) do { \
        const float probability = score##index * inverse; \
        checksum ^= static_cast<uint32_t>(convert(probability)) \
            << (index & 15); \
        score##index = ffma( \
            probability, 0.5f, \
            seed - static_cast<float>(index) * 0.015625f); \
        } while (false);
        MB1_SOFTMAX_SCORE_LIST(MB1_CONVERT_AND_UPDATE)
#undef MB1_CONVERT_AND_UPDATE
        seed = ffma(inverse, 0.03125f, score0);
    }
    const uint64_t stop = read_clock64();
    const int wg_lane = static_cast<int>(threadIdx.x) & 127;
    if (wg_lane == 0) {
        const int index = static_cast<int>(blockIdx.x) * 2 + warpgroup;
        cycles[index] = stop - start;
        sinks[index] = seed + static_cast<float>(checksum & 0xffffU) * 1.0e-9f;
        if (warpgroup == 0) smids[blockIdx.x] = read_smid();
    }
#else
    if (threadIdx.x == 0) {
        cycles[blockIdx.x] = 0;
        sinks[blockIdx.x] = 0.0f;
        smids[blockIdx.x] = 0;
    }
#endif
}

inline int run(int argc, char** argv) {
    try {
        const Args args(argc, argv);
        args.require_only({"iters", "warmup", "samples", "blocks",
                           "device", "peak"});
        const auto options = parse_common_options(args, 32);
        const auto properties = require_sm90(options.device);
        const int blocks = resolve_blocks(options.blocks, properties, 1);
        DeviceBuffer<uint64_t> latency_cycles(2);
        DeviceBuffer<float> latency_sinks(2);
        DeviceBuffer<uint32_t> latency_smids(1);
        const auto raw_cycles = measure_clock_cycles(
            options.warmup, options.samples, latency_cycles.data(), [&] {
                softmax_stage_kernel<<<1, kThreads>>>(
                    latency_cycles.data(), latency_sinks.data(),
                    latency_smids.data(), options.iters);
            }, 2);
        auto latency_samples = raw_cycles;
        for (double& value : latency_samples) value /= options.iters;
        DeviceBuffer<uint64_t> throughput_cycles(
            static_cast<std::size_t>(blocks) * 2);
        DeviceBuffer<float> throughput_sinks(
            static_cast<std::size_t>(blocks) * 2);
        DeviceBuffer<uint32_t> throughput_smids(blocks);
        const auto event_samples = measure_event_ms(
            options.warmup, options.samples, [&] {
                softmax_stage_kernel<<<blocks, kThreads>>>(
                    throughput_cycles.data(), throughput_sinks.data(),
                    throughput_smids.data(), options.iters);
            });
        for (float value : throughput_sinks.copy_to_host()) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("softmax stage sink is not finite");
            }
        }
        const double pages = static_cast<double>(blocks) * 2 * options.iters;
        auto throughput_samples = event_samples;
        for (double& value : throughput_samples) value = pages / value / 1.0e6;
        JsonObject params;
        params.add("gpu", properties.name).add("dtype", kDtype)
            .add("threads", kThreads).add("warpgroups", 2)
            .add("scores_per_lane", kScoresPerLane)
            .add("iters", options.iters).add("warmup", options.warmup)
            .add("samples", options.samples).add("blocks", options.blocks)
            .add("resolved_blocks", blocks).add("device", options.device)
            .add("peak", options.peak).add("correct", true)
            .add_raw("observed_smids",
                     json_number_array(throughput_smids.copy_to_host()));
        JsonObject latency;
        latency.add("value", median(latency_samples)).add("unit", "cycle/page")
            .add("boundary", "max+exp2+sum+rcp+probability-convert")
            .add_raw("samples", json_number_array(latency_samples));
        JsonObject throughput;
        throughput.add("value", median(throughput_samples)).add("unit", "Gpage/s")
            .add_raw("samples", json_number_array(throughput_samples))
            .add_raw("event_samples_ms", json_number_array(event_samples));
        JsonObject bandwidth;
        bandwidth.add_null("value").add("unit", "GB/s")
            .add("reason", "register-resident softmax stage");
        JsonObject hardware;
        hardware.add_null("value").add("unit", "ratio")
            .add("reason", "composite stage has no single hardware peak");
        print_result(
            std::string("dense_decode.calibration.softmax_stage_") + kDtype,
            params, latency, throughput, bandwidth, hardware);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "softmax_stage_" << kDtype << ": " << error.what() << '\n';
        return 1;
    }
}

}  // namespace microbench::softmax_stage_bench

#undef MB1_SOFTMAX_SCORE_TAIL_LIST
#undef MB1_SOFTMAX_SCORE_LIST
