#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../../../../../microbench/common/bench.hpp"
#include "../../../../../../microbench/compute/wgmma/common/ptx.cuh"

namespace microbench::page_pair_transition_bench {

constexpr int kThreads = 256;
constexpr int kWarpgroupThreads = 128;
constexpr int kPageBytes = 64 * 576 * 2;
constexpr int kPTileBytes = 64 * 64 * 2;
constexpr int kDynamicSharedBytes = 2 * kPageBytes + 2 * kPTileBytes;
constexpr int kQkGroupsPerWarpgroup = 9;
constexpr int kQkInstructionsPerGroup = 4;
constexpr int kPvInstructionsPerGroup = 4;
constexpr int kNamedHandoffs = 4;
constexpr int kLogicalSharedBytes = 4 * kPTileBytes;

#if defined(MB1_WGMMA_USE_F16)
#define MB1_PP_INPUT_TYPE "f16"
constexpr const char* kDtype = "fp16";
constexpr uint16_t kInputBits = 0x1400u;
constexpr uint32_t kPackedInput = 0x14001400u;
#else
#define MB1_PP_INPUT_TYPE "bf16"
constexpr const char* kDtype = "bf16";
constexpr uint16_t kInputBits = 0x3a80u;
constexpr uint32_t kPackedInput = 0x3a803a80u;
#endif

#define MB1_PP_ACCUM_32 \
    "{d0, d1, d2, d3, d4, d5, d6, d7, " \
    "d8, d9, d10, d11, d12, d13, d14, d15, " \
    "d16, d17, d18, d19, d20, d21, d22, d23, " \
    "d24, d25, d26, d27, d28, d29, d30, d31}"

#define MB1_PP_ACCUM_128 \
    "{o0, o1, o2, o3, o4, o5, o6, o7, " \
    "o8, o9, o10, o11, o12, o13, o14, o15, " \
    "o16, o17, o18, o19, o20, o21, o22, o23, " \
    "o24, o25, o26, o27, o28, o29, o30, o31, " \
    "o32, o33, o34, o35, o36, o37, o38, o39, " \
    "o40, o41, o42, o43, o44, o45, o46, o47, " \
    "o48, o49, o50, o51, o52, o53, o54, o55, " \
    "o56, o57, o58, o59, o60, o61, o62, o63, " \
    "o64, o65, o66, o67, o68, o69, o70, o71, " \
    "o72, o73, o74, o75, o76, o77, o78, o79, " \
    "o80, o81, o82, o83, o84, o85, o86, o87, " \
    "o88, o89, o90, o91, o92, o93, o94, o95, " \
    "o96, o97, o98, o99, o100, o101, o102, o103, " \
    "o104, o105, o106, o107, o108, o109, o110, o111, " \
    "o112, o113, o114, o115, o116, o117, o118, o119, " \
    "o120, o121, o122, o123, o124, o125, o126, o127}"

#define MB1_PP_LOCAL_PV_RS \
    "wgmma.mma_async.sync.aligned.m64n256k16.f32." \
    MB1_PP_INPUT_TYPE "." MB1_PP_INPUT_TYPE " " MB1_PP_ACCUM_128 \
    ", {%2, %3, %4, %5}, %1, pvpred, 1, 1, 1;\n\t"

#define MB1_PP_PV_RS \
    "wgmma.mma_async.sync.aligned.m64n256k16.f32." \
    MB1_PP_INPUT_TYPE "." MB1_PP_INPUT_TYPE " " MB1_PP_ACCUM_128 \
    ", {%5, %6, %7, %8}, %2, pvpred, 1, 1, 1;\n\t"

#define MB1_PP_PV_SS \
    "wgmma.mma_async.sync.aligned.m64n256k16.f32." \
    MB1_PP_INPUT_TYPE "." MB1_PP_INPUT_TYPE " " MB1_PP_ACCUM_128 \
    ", %1, %2, pvpred, 1, 1, 0, 1;\n\t"

#define MB1_PP_QK_SS \
    "wgmma.mma_async.sync.aligned.m64n64k16.f32." \
    MB1_PP_INPUT_TYPE "." MB1_PP_INPUT_TYPE " " MB1_PP_ACCUM_32 \
    ", qa, kb, qpred, 1, 1, 0, 0;\n\t"

#define MB1_PP_QK_RS \
    "wgmma.mma_async.sync.aligned.m64n64k16.f32." \
    MB1_PP_INPUT_TYPE "." MB1_PP_INPUT_TYPE " " MB1_PP_ACCUM_32 \
    ", {%5, %6, %7, %8}, kb, qpred, 1, 1, 0;\n\t"

#define MB1_PP_QK_SS_GROUP \
    "wgmma.fence.sync.aligned;\n\t" \
    MB1_PP_QK_SS \
    "setp.ne.u32 qpred, 1, 0; add.u64 qa, qa, 2; add.u64 kb, kb, 2;\n\t" \
    MB1_PP_QK_SS \
    "add.u64 qa, qa, 2; add.u64 kb, kb, 2;\n\t" \
    MB1_PP_QK_SS \
    "add.u64 qa, qa, 2; add.u64 kb, kb, 2;\n\t" \
    MB1_PP_QK_SS \
    "wgmma.commit_group.sync.aligned;\n\t" \
    "add.u64 qa, qa, 506; add.u64 kb, kb, 506;\n\t"

#define MB1_PP_QK_RS_GROUP \
    "wgmma.fence.sync.aligned;\n\t" \
    MB1_PP_QK_RS \
    "setp.ne.u32 qpred, 1, 0; add.u64 kb, kb, 2;\n\t" \
    MB1_PP_QK_RS \
    "add.u64 kb, kb, 2;\n\t" \
    MB1_PP_QK_RS \
    "add.u64 kb, kb, 2;\n\t" \
    MB1_PP_QK_RS \
    "wgmma.commit_group.sync.aligned;\n\t"

__device__ __forceinline__ uint16_t convert_input(float value) {
    uint16_t result;
#if defined(MB1_WGMMA_USE_F16)
    asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(value));
#else
    asm volatile("cvt.rn.bf16.f32 %0, %1;" : "=h"(result) : "f"(value));
#endif
    return result;
}

__device__ __forceinline__ uint32_t make_softmax_pair(float seed,
                                                       int lane,
                                                       int iteration) {
    float value = seed + 0.0009765625f *
        static_cast<float>((lane + iteration) & 31);
    float exponential;
    asm volatile("ex2.approx.ftz.f32 %0, %1;"
                 : "=f"(exponential) : "f"(-value));
    const uint16_t lo = convert_input(exponential);
    asm volatile("ex2.approx.ftz.f32 %0, %1;"
                 : "=f"(exponential) : "f"(-value - 0.03125f));
    const uint16_t hi = convert_input(exponential);
    return static_cast<uint32_t>(lo) |
        (static_cast<uint32_t>(hi) << 16);
}

__device__ __forceinline__ uint32_t p_matrix_offset(int instruction) {
    const uint32_t lane = static_cast<uint32_t>(threadIdx.x & 127);
    const uint32_t lane_offset =
        (((lane << 5) & ~1023U) |
         ((lane << 6) & 960U) |
         ((lane >> 1) & 8U)) * 2U;
    const uint32_t logical = lane_offset +
        static_cast<uint32_t>(instruction) * 32U;
    return logical ^ ((logical & 896U) >> 3);
}

__device__ __forceinline__ void stmatrix_p(uint8_t* destination,
                                           uint32_t p0,
                                           uint32_t p1,
                                           uint32_t p2,
                                           uint32_t p3) {
    const uint32_t base = shared_address(destination);
#pragma unroll
    for (int instruction = 0; instruction < 4; ++instruction) {
        const uint32_t address = base + p_matrix_offset(instruction);
        asm volatile(
            "stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], "
            "{%1, %2, %3, %4};"
            :: "r"(address), "r"(p0), "r"(p1), "r"(p2), "r"(p3)
            : "memory");
    }
}

__device__ __forceinline__ uint32_t ldmatrix_p(const uint8_t* source) {
    const uint32_t base = shared_address(source);
    uint32_t checksum = 0;
#pragma unroll
    for (int instruction = 0; instruction < 4; ++instruction) {
        const uint32_t address = base + p_matrix_offset(instruction);
        uint32_t x0, x1, x2, x3;
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
            "{%0, %1, %2, %3}, [%4];"
            : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3)
            : "r"(address) : "memory");
        checksum ^= x0 ^ x1 ^ x2 ^ x3;
    }
    return checksum;
}

template <int BarrierId>
__device__ __forceinline__ void named_arrive() {
    static_assert(BarrierId >= 1 && BarrierId <= 4);
    asm volatile("bar.arrive %0, 256;" :: "n"(BarrierId) : "memory");
}

template <int BarrierId>
__device__ __forceinline__ void named_sync() {
    static_assert(BarrierId >= 1 && BarrierId <= 4);
    asm volatile("bar.sync %0, 256;" :: "n"(BarrierId) : "memory");
}

__device__ __forceinline__ float wg0_local_pv(uint64_t v_desc,
                                              uint32_t p0,
                                              uint32_t p1,
                                              uint32_t p2,
                                              uint32_t p3) {
    float sink;
    asm volatile(
        "{\n\t"
        ".reg .pred pvpred;\n\t"
        ".reg .f32 o<128>;\n\t"
        "wgmma.fence.sync.aligned;\n\t"
        "setp.ne.u32 pvpred, 0, 0;\n\t"
        MB1_PP_LOCAL_PV_RS
        "setp.ne.u32 pvpred, 1, 0;\n\t"
        MB1_PP_LOCAL_PV_RS MB1_PP_LOCAL_PV_RS MB1_PP_LOCAL_PV_RS
        "wgmma.commit_group.sync.aligned;\n\t"
        "wgmma.wait_group.sync.aligned 0;\n\t"
        "mov.f32 %0, o127;\n\t"
        "}\n"
        : "=f"(sink)
        : "l"(v_desc), "r"(p0), "r"(p1), "r"(p2), "r"(p3)
        : "memory");
    return sink;
}

__device__ __forceinline__ float wg0_remote_pv_and_qk(
        uint64_t remote_p_desc,
        uint64_t v_desc,
        uint64_t q_desc,
        uint64_t k_desc,
        uint32_t p0,
        uint32_t p1,
        uint32_t p2,
        uint32_t p3) {
    float sink;
    asm volatile(
        "{\n\t"
        ".reg .pred pvpred, qpred;\n\t"
        ".reg .f32 o<128>, d<32>;\n\t"
        ".reg .u64 qa, kb;\n\t"
        "mov.u64 qa, %3; mov.u64 kb, %4;\n\t"
        "wgmma.fence.sync.aligned;\n\t"
        "setp.ne.u32 pvpred, 0, 0;\n\t"
        MB1_PP_PV_SS
        "setp.ne.u32 pvpred, 1, 0;\n\t"
        MB1_PP_PV_SS MB1_PP_PV_SS MB1_PP_PV_SS
        "wgmma.commit_group.sync.aligned;\n\t"
        "setp.ne.u32 qpred, 0, 0;\n\t"
        MB1_PP_QK_SS_GROUP MB1_PP_QK_SS_GROUP
        MB1_PP_QK_SS_GROUP MB1_PP_QK_SS_GROUP
        "wgmma.wait_group.sync.aligned 4;\n\t"
        MB1_PP_QK_SS_GROUP MB1_PP_QK_SS_GROUP
        MB1_PP_QK_SS_GROUP MB1_PP_QK_SS_GROUP
        MB1_PP_QK_RS_GROUP
        "wgmma.wait_group.sync.aligned 0;\n\t"
        "add.f32 %0, o127, d31;\n\t"
        "}\n"
        : "=f"(sink)
        : "l"(remote_p_desc), "l"(v_desc), "l"(q_desc), "l"(k_desc),
          "r"(p0), "r"(p1), "r"(p2), "r"(p3)
        : "memory");
    return sink;
}

__device__ __forceinline__ float wg1_pv_and_qk(
        uint64_t remote_p_desc,
        uint64_t v_desc,
        uint64_t q_desc,
        uint64_t k_desc,
        uint32_t p0,
        uint32_t p1,
        uint32_t p2,
        uint32_t p3) {
    float sink;
    asm volatile(
        "{\n\t"
        ".reg .pred pvpred, qpred;\n\t"
        ".reg .f32 o<128>, d<32>;\n\t"
        ".reg .u64 qa, kb;\n\t"
        "mov.u64 qa, %3; mov.u64 kb, %4;\n\t"
        "wgmma.fence.sync.aligned;\n\t"
        "setp.ne.u32 pvpred, 0, 0;\n\t"
        MB1_PP_PV_RS
        "setp.ne.u32 pvpred, 1, 0;\n\t"
        MB1_PP_PV_RS MB1_PP_PV_RS MB1_PP_PV_RS
        "wgmma.commit_group.sync.aligned;\n\t"
        MB1_PP_PV_SS MB1_PP_PV_SS MB1_PP_PV_SS MB1_PP_PV_SS
        "wgmma.commit_group.sync.aligned;\n\t"
        "wgmma.wait_group.sync.aligned 1;\n\t"
        "wgmma.wait_group.sync.aligned 0;\n\t"
        "setp.ne.u32 qpred, 0, 0;\n\t"
        MB1_PP_QK_SS_GROUP MB1_PP_QK_SS_GROUP
        MB1_PP_QK_SS_GROUP MB1_PP_QK_SS_GROUP
        MB1_PP_QK_SS_GROUP MB1_PP_QK_SS_GROUP
        MB1_PP_QK_SS_GROUP MB1_PP_QK_SS_GROUP
        MB1_PP_QK_RS_GROUP
        "wgmma.wait_group.sync.aligned 0;\n\t"
        "add.f32 %0, o127, d31;\n\t"
        "}\n"
        : "=f"(sink)
        : "l"(remote_p_desc), "l"(v_desc), "l"(q_desc), "l"(k_desc),
          "r"(p0), "r"(p1), "r"(p2), "r"(p3)
        : "memory");
    return sink;
}

__global__ __launch_bounds__(kThreads, 1)
void transition_kernel(uint64_t* cycles,
                       float* sinks,
                       uint32_t* smids,
                       int iterations) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    extern __shared__ __align__(1024) uint8_t storage[];
    uint8_t* q_storage = storage;
    uint8_t* k_storage = q_storage + kPageBytes;
    uint8_t* p0_storage = k_storage + kPageBytes;
    uint8_t* p1_storage = p0_storage + kPTileBytes;
    auto* words = reinterpret_cast<uint32_t*>(storage);
    for (int word = threadIdx.x;
         word < (2 * kPageBytes) / static_cast<int>(sizeof(uint32_t));
         word += blockDim.x) {
        words[word] = kPackedInput;
    }
    asm volatile("bar.sync 0;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("bar.sync 0;" ::: "memory");

    const int warpgroup = threadIdx.x / kWarpgroupThreads;
    const int lane = threadIdx.x & 127;
    const uint64_t q_desc = ptx::make_sw128_kmajor_descriptor(q_storage);
    const uint64_t k_desc = ptx::make_sw128_kmajor_descriptor(k_storage);
    const uint64_t v_desc = ptx::make_sw128_transposed_descriptor(k_storage);
    const uint64_t p0_desc = ptx::make_sw128_kmajor_descriptor(p0_storage);
    const uint64_t p1_desc = ptx::make_sw128_kmajor_descriptor(p1_storage);
    float seed = 0.5f + static_cast<float>(lane & 31) * 0.00390625f;
    uint32_t exchange_checksum = 0;
    asm volatile("bar.sync 0;" ::: "memory");
    const uint64_t start = read_clock64();
#pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const uint32_t p0 = make_softmax_pair(seed, lane, iteration);
        const uint32_t p1 = make_softmax_pair(seed + 0.03125f, lane, iteration);
        const uint32_t p2 = make_softmax_pair(seed + 0.0625f, lane, iteration);
        const uint32_t p3 = make_softmax_pair(seed + 0.09375f, lane, iteration);
        if (warpgroup == 0) {
            named_arrive<1>();
            seed = wg0_local_pv(v_desc, p0, p1, p2, p3);
            named_sync<2>();
            stmatrix_p(p0_storage, p0, p1, p2, p3);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            named_arrive<3>();
            named_sync<4>();
            exchange_checksum ^= ldmatrix_p(p1_storage);
            seed += wg0_remote_pv_and_qk(
                p1_desc, v_desc, q_desc, k_desc, p0, p1, p2, p3);
        } else {
            named_sync<1>();
            named_arrive<2>();
            stmatrix_p(p1_storage, p0, p1, p2, p3);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            named_sync<3>();
            exchange_checksum ^= ldmatrix_p(p0_storage);
            named_arrive<4>();
            seed = wg1_pv_and_qk(
                p0_desc, v_desc, q_desc, k_desc, p0, p1, p2, p3);
        }
    }
    const uint64_t stop = read_clock64();
    const int leader = lane == 0 ? warpgroup : -1;
    if (leader >= 0) {
        const int index = static_cast<int>(blockIdx.x) * 2 + leader;
        cycles[index] = stop - start;
        if (leader == 0) smids[blockIdx.x] = read_smid();
        sinks[index] = seed +
            static_cast<float>(exchange_checksum & 0xffffU) * 1.0e-9f;
    }
#else
    if (threadIdx.x < 2) {
        const int index = static_cast<int>(blockIdx.x) * 2 + threadIdx.x;
        cycles[index] = 0;
        sinks[index] = 0.0f;
    }
#endif
}

inline std::vector<double> divide_samples(const std::vector<double>& samples,
                                          double divisor) {
    std::vector<double> output;
    output.reserve(samples.size());
    for (const double sample : samples) output.push_back(sample / divisor);
    return output;
}

inline std::vector<double> rate_samples(const std::vector<double>& event_ms,
                                        double work,
                                        double scale) {
    std::vector<double> output;
    output.reserve(event_ms.size());
    for (const double sample : event_ms) output.push_back(work / sample / scale);
    return output;
}

inline void validate_sinks(const std::vector<float>& values) {
    for (const float value : values) {
        if (!std::isfinite(value) || value == 0.0f) {
            throw std::runtime_error("page-pair transition sink is invalid");
        }
    }
}

inline int run(int argc, char** argv) {
    try {
        const Args args(argc, argv);
        args.require_only({"iters", "warmup", "samples", "blocks",
                           "device", "peak"});
        const auto options = parse_common_options(args, 32);
        const auto properties = require_sm90(options.device);
        const int blocks = resolve_blocks(options.blocks, properties, 1);
        if (kDynamicSharedBytes >
            static_cast<int>(properties.sharedMemPerBlockOptin)) {
            throw std::runtime_error(
                "page-pair transition shared storage exceeds device opt-in limit");
        }
        CUDA_CHECK(cudaFuncSetAttribute(
            transition_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            kDynamicSharedBytes));

        DeviceBuffer<uint64_t> latency_cycles(2);
        DeviceBuffer<float> latency_sinks(2);
        DeviceBuffer<uint32_t> latency_smids(1);
        const auto raw_cycle_samples = measure_clock_cycles(
            options.warmup, options.samples, latency_cycles.data(), [&] {
                transition_kernel<<<1, kThreads, kDynamicSharedBytes>>>(
                    latency_cycles.data(), latency_sinks.data(),
                    latency_smids.data(), options.iters);
            }, 2);
        validate_sinks(latency_sinks.copy_to_host());
        const auto latency_samples =
            divide_samples(raw_cycle_samples, options.iters);
        const double cycles_per_pair = median(latency_samples);

        DeviceBuffer<uint64_t> throughput_cycles(
            static_cast<std::size_t>(blocks) * 2);
        DeviceBuffer<float> throughput_sinks(
            static_cast<std::size_t>(blocks) * 2);
        DeviceBuffer<uint32_t> throughput_smids(blocks);
        const auto event_samples_ms = measure_event_ms(
            options.warmup, options.samples, [&] {
                transition_kernel<<<blocks, kThreads, kDynamicSharedBytes>>>(
                    throughput_cycles.data(), throughput_sinks.data(),
                    throughput_smids.data(), options.iters);
            });
        validate_sinks(throughput_sinks.copy_to_host());
        const double pairs = static_cast<double>(blocks) * options.iters;
        const auto throughput_samples =
            rate_samples(event_samples_ms, pairs, 1.0e6);
        const double throughput_gpairs = median(throughput_samples);
        const double elapsed_ms = median(event_samples_ms);
        const double logical_bytes = pairs * kLogicalSharedBytes;
        const auto bandwidth_samples =
            rate_samples(event_samples_ms, logical_bytes, 1.0e6);
        const double bandwidth_gbs = median(bandwidth_samples);

        JsonObject params;
        params.add("gpu", properties.name).add("dtype", kDtype)
            .add("protocol", "dense_steady_page_pair_transition")
            .add("boundary", "both_rP_ready_to_both_next_rP_ready")
            .add("warpgroups", 2).add("threads", kThreads)
            .add("named_barrier_handoffs_per_pair", kNamedHandoffs)
            .add("stmatrix_instructions_per_pair", 32)
            .add("ldmatrix_instructions_per_pair", 32)
            .add("ldmatrix_role", "validation_only_not_dense_dag_work")
            .add("pv_wgmma_instructions_per_pair", 16)
            .add("qk_wgmma_instructions_per_pair", 72)
            .add("qk_committed_groups_per_warpgroup", 9)
            .add("wait_group_4_per_pair", 1)
            .add("wait_group_1_per_pair", 1)
            .add("wait_group_0_per_pair", 4)
            .add("shared_bytes", kDynamicSharedBytes)
            .add("qk_buffer_aliasing", "two_wg_read_only_shared_qk")
            .add("iters", options.iters).add("warmup", options.warmup)
            .add("samples", options.samples).add("blocks", options.blocks)
            .add("resolved_blocks", blocks).add("device", options.device)
            .add("peak", options.peak);
        const auto observed_smids = throughput_smids.copy_to_host();
        JsonObject latency;
        latency.add("value", cycles_per_pair).add("unit", "cycle/page-pair")
            .add("timer", "clock64").add("scope", "max_of_two_wg_leaders")
            .add("raw_median_cycles", median(raw_cycle_samples))
            .add_raw("samples", json_number_array(latency_samples))
            .add_raw("target_samples_cycles",
                     json_number_array(raw_cycle_samples))
            .add_raw("observed_smids", json_number_array(observed_smids));
        JsonObject throughput = metric(throughput_gpairs, "Gpage-pair/s");
        throughput.add("timer", "cuda_event").add("scope", "grid")
            .add("median_ms", elapsed_ms).add("page_pairs", pairs)
            .add_raw("samples", json_number_array(throughput_samples))
            .add_raw("event_samples_ms", json_number_array(event_samples_ms));
        JsonObject bandwidth = metric(bandwidth_gbs, "GB/s");
        bandwidth.add("kind", "logical_P_exchange")
            .add("bytes", logical_bytes)
            .add("bytes_per_page_pair", kLogicalSharedBytes)
            .add_raw("samples", json_number_array(bandwidth_samples));
        const std::string name =
            std::string("page_pair_transition_") + kDtype;
        print_result(name, params, latency, throughput, bandwidth,
                     utilization(throughput_gpairs, options.peak,
                                 "Gpage-pair/s"));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "page-pair transition benchmark error: "
                  << error.what() << '\n';
        return 1;
    }
}

}  // namespace microbench::page_pair_transition_bench

#undef MB1_PP_QK_RS_GROUP
#undef MB1_PP_QK_SS_GROUP
#undef MB1_PP_QK_RS
#undef MB1_PP_QK_SS
#undef MB1_PP_PV_SS
#undef MB1_PP_PV_RS
#undef MB1_PP_LOCAL_PV_RS
#undef MB1_PP_ACCUM_128
#undef MB1_PP_ACCUM_32
#undef MB1_PP_INPUT_TYPE
