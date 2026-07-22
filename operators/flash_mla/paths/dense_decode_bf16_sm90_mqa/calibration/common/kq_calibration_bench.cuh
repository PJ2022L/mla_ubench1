#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../../../../../../microbench/common/bench.hpp"
#include "../../../../../../microbench/memory/tma_load/common/ptx.cuh"
#include "../../../../../../microbench/compute/wgmma/common/ptx.cuh"
#include "../../../../../../microbench/memory/tma_load/common/tensor_map.hpp"

namespace microbench::kq_calibration_bench {

enum class Protocol { kFirstPage, kSteadyPage };

constexpr int kThreads = 128;
constexpr int kRows = 64;
constexpr int kHeadDimension = 576;
constexpr int kTileColumns = 64;
constexpr int kTiles = 9;
constexpr int kKBlocksPerTile = 4;
constexpr int kTileBytes = kRows * kTileColumns * 2;
constexpr int kPageBytes = kRows * kHeadDimension * 2;
constexpr int kPageElements = kRows * kHeadDimension;
constexpr int kQkInstructions = kTiles * kKBlocksPerTile;
constexpr int kDynamicSharedBytes = 2 * kPageBytes;
constexpr double kFlopsPerInstruction = 2.0 * 64.0 * 64.0 * 16.0;
constexpr double kContribution = 1.0 / 65536.0;

#if defined(MB1_WGMMA_USE_F16)
#define MB1_KQ_INPUT_TYPE "f16"
constexpr uint16_t kInputBits = 0x1400u;
constexpr uint32_t kPackedInput = 0x14001400u;
constexpr const char* kDtype = "fp16";
constexpr CUtensorMapDataType kTensorMapType = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
#else
#define MB1_KQ_INPUT_TYPE "bf16"
constexpr uint16_t kInputBits = 0x3a80u;
constexpr uint32_t kPackedInput = 0x3a803a80u;
constexpr const char* kDtype = "bf16";
constexpr CUtensorMapDataType kTensorMapType = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
#endif

#define MB1_KQ_ACCUM \
    "{%0, %1, %2, %3, %4, %5, %6, %7, " \
    "%8, %9, %10, %11, %12, %13, %14, %15, " \
    "%16, %17, %18, %19, %20, %21, %22, %23, " \
    "%24, %25, %26, %27, %28, %29, %30, %31}"

#define MB1_KQ_OUTPUTS(D) \
    "+f"((D)[0]), "+f"((D)[1]), "+f"((D)[2]), "+f"((D)[3]), \
    "+f"((D)[4]), "+f"((D)[5]), "+f"((D)[6]), "+f"((D)[7]), \
    "+f"((D)[8]), "+f"((D)[9]), "+f"((D)[10]), "+f"((D)[11]), \
    "+f"((D)[12]), "+f"((D)[13]), "+f"((D)[14]), "+f"((D)[15]), \
    "+f"((D)[16]), "+f"((D)[17]), "+f"((D)[18]), "+f"((D)[19]), \
    "+f"((D)[20]), "+f"((D)[21]), "+f"((D)[22]), "+f"((D)[23]), \
    "+f"((D)[24]), "+f"((D)[25]), "+f"((D)[26]), "+f"((D)[27]), \
    "+f"((D)[28]), "+f"((D)[29]), "+f"((D)[30]), "+f"((D)[31])

__device__ __forceinline__ void qk_ss(float (&d)[32],
                                      uint64_t desc_a,
                                      uint64_t desc_b,
                                      uint32_t accumulate) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.u32 p, %34, 0;\n\t"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32."
        MB1_KQ_INPUT_TYPE "." MB1_KQ_INPUT_TYPE " "
        MB1_KQ_ACCUM ", %32, %33, p, 1, 1, 0, 0;\n\t"
        "}\n"
        : MB1_KQ_OUTPUTS(d)
        : "l"(desc_a), "l"(desc_b), "r"(accumulate)
        : "memory");
}

__device__ __forceinline__ void qk_rs(float (&d)[32],
                                      uint32_t a0,
                                      uint32_t a1,
                                      uint32_t a2,
                                      uint32_t a3,
                                      uint64_t desc_b,
                                      uint32_t accumulate) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.u32 p, %37, 0;\n\t"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32."
        MB1_KQ_INPUT_TYPE "." MB1_KQ_INPUT_TYPE " "
        MB1_KQ_ACCUM ", {%32, %33, %34, %35}, %36, p, 1, 1, 0;\n\t"
        "}\n"
        : MB1_KQ_OUTPUTS(d)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b),
          "r"(accumulate)
        : "memory");
}

__device__ __forceinline__ void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void wgmma_wait0() {
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
}

__global__ void initialize_input(uint16_t* input, std::size_t elements) {
    std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (; index < elements; index += stride) input[index] = kInputBits;
}

__device__ __forceinline__ int select_page(int iteration,
                                           int block,
                                           int grid_blocks,
                                           int working_pages) {
    const uint64_t linear = static_cast<uint64_t>(iteration) * grid_blocks + block;
    return static_cast<int>(linear % static_cast<uint64_t>(working_pages));
}

__device__ __forceinline__ uint64_t tile_descriptor(uint64_t base,
                                                    int tile,
                                                    int k_block) {
    constexpr int kDescriptorUnitBytes = 16;
    constexpr int kKBlockPhysicalOffsetBytes = 32;
    return base + static_cast<uint64_t>(
        (tile * kTileBytes + k_block * kKBlockPhysicalOffsetBytes) /
        kDescriptorUnitBytes);
}

template <Protocol P>
__device__ __forceinline__ void issue_qk_page(float (&accum)[32],
                                               uint64_t q_desc,
                                               uint64_t k_desc) {
    if constexpr (P == Protocol::kFirstPage) {
        wgmma_fence();
#pragma unroll
        for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
            for (int k_block = 0; k_block < kKBlocksPerTile; ++k_block) {
                const uint32_t scale_d = tile == 0 && k_block == 0 ? 0u : 1u;
                qk_ss(accum,
                      tile_descriptor(q_desc, tile, k_block),
                      tile_descriptor(k_desc, tile, k_block), scale_d);
            }
        }
        wgmma_commit();
    }
}

template <Protocol P>
__global__ void calibration_kernel(
        __grid_constant__ const CUtensorMap tensor_map,
        uint64_t* cycles,
        float* sinks,
        uint32_t* smids,
        int iterations,
        int working_pages) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    extern __shared__ __align__(1024) unsigned char storage[];
    unsigned char* q_storage = storage;
    unsigned char* k_storage = storage + kPageBytes;
    __shared__ alignas(8) uint64_t barriers[kTiles];
    auto* q_words = reinterpret_cast<uint32_t*>(q_storage);
    for (int word = threadIdx.x; word < kPageBytes / 4; word += blockDim.x) {
        q_words[word] = kPackedInput;
    }
    if (threadIdx.x == 0) {
#pragma unroll
        for (int tile = 0; tile < kTiles; ++tile) {
            ptx::mbarrier_init(&barriers[tile], 1);
        }
        ptx::mbarrier_init_fence();
    }
    __syncthreads();
    ptx::async_shared_fence();
    __syncthreads();

    const uint64_t q_desc = ptx::make_sw128_kmajor_descriptor(q_storage);
    const uint64_t k_desc = ptx::make_sw128_kmajor_descriptor(k_storage);
    float accum[32] = {};
    uint32_t phase = 0;
    const uint64_t start = read_clock64();
#pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const int page = select_page(
            iteration, static_cast<int>(blockIdx.x),
            static_cast<int>(gridDim.x), working_pages);
        if (threadIdx.x == 0) {
#pragma unroll
            for (int tile = 0; tile < kTiles; ++tile) {
                ptx::tma_load_4d(
                    &tensor_map, &barriers[tile],
                    k_storage + tile * kTileBytes,
                    tile * kTileColumns, 0, 0, page, ptx::kTmaEvictFirst);
            }
        }

        if constexpr (P == Protocol::kFirstPage) {
#pragma unroll
            for (int tile = 0; tile < kTiles; ++tile) {
                if (threadIdx.x == 0) {
                    ptx::mbarrier_arrive_expect_tx(
                        &barriers[tile], kTileBytes);
                }
                ptx::mbarrier_wait_parity(&barriers[tile], phase);
            }
            issue_qk_page<P>(accum, q_desc, k_desc);
        } else {
#pragma unroll
            for (int tile = 0; tile < kTiles; ++tile) {
                if (threadIdx.x == 0) {
                    ptx::mbarrier_arrive_expect_tx(
                        &barriers[tile], kTileBytes);
                }
                ptx::mbarrier_wait_parity(&barriers[tile], phase);
                wgmma_fence();
#pragma unroll
                for (int k_block = 0; k_block < kKBlocksPerTile; ++k_block) {
                    const uint32_t scale_d =
                        tile == 0 && k_block == 0 ? 0u : 1u;
                    const uint64_t b_desc =
                        tile_descriptor(k_desc, tile, k_block);
                    if (tile == 8) {
                        qk_rs(accum, kPackedInput, kPackedInput,
                              kPackedInput, kPackedInput, b_desc, scale_d);
                    } else {
                        qk_ss(accum,
                              tile_descriptor(q_desc, tile, k_block),
                              b_desc, scale_d);
                    }
                }
                wgmma_commit();
            }
        }
        wgmma_wait0();
        phase ^= 1u;
    }
    const uint64_t stop = read_clock64();
    if (threadIdx.x == 0) {
        cycles[blockIdx.x] = stop - start;
        sinks[blockIdx.x] = accum[0];
        smids[blockIdx.x] = read_smid();
    }
#else
    if (threadIdx.x == 0) {
        cycles[blockIdx.x] = 0;
        sinks[blockIdx.x] = 0.0f;
    }
#endif
}

inline CUtensorMap make_tensor_map(void* input, int working_pages) {
    CUtensorMap map{};
    constexpr uint32_t rank = 4;
    const std::array<uint64_t, rank> dimensions = {
        kHeadDimension, kRows, 1, static_cast<uint64_t>(working_pages)};
    const std::array<uint64_t, rank - 1> strides = {
        kHeadDimension * 2ull, kPageBytes, kPageBytes};
    const std::array<uint32_t, rank> box = {64, 64, 1, 1};
    const std::array<uint32_t, rank> element_strides = {1, 1, 1, 1};
    check_driver(
        cuTensorMapEncodeTiled(
            &map, kTensorMapType, rank, input, dimensions.data(),
            strides.data(), box.data(), element_strides.data(),
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
        "cuTensorMapEncodeTiled(KQ calibration rank4)");
    return map;
}

inline std::vector<double> divide_samples(const std::vector<double>& samples,
                                          double divisor) {
    std::vector<double> result;
    result.reserve(samples.size());
    for (const double value : samples) result.push_back(value / divisor);
    return result;
}

inline std::vector<double> rate_samples(const std::vector<double>& event_ms,
                                        double work,
                                        double scale) {
    std::vector<double> result;
    result.reserve(event_ms.size());
    for (const double ms : event_ms) result.push_back(work / ms / scale);
    return result;
}

template <Protocol P>
int run(int argc, char** argv) {
    try {
        const Args args(argc, argv);
        args.require_only({"iters", "warmup", "samples", "blocks", "device",
                           "peak", "working-set-pages"});
        const auto options = parse_common_options(args, 64);
        const auto properties = require_sm90(options.device);
        const int blocks = resolve_blocks(options.blocks, properties, 1);
        const int working_pages =
            args.get_int("working-set-pages", 1024, 1, 1 << 20);
        const std::size_t input_elements =
            static_cast<std::size_t>(working_pages) * kPageElements;
        DeviceBuffer<uint16_t> input(input_elements);
        initialize_input<<<256, 256>>>(input.data(), input_elements);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        const CUtensorMap tensor_map = make_tensor_map(input.data(), working_pages);
        CUDA_CHECK(cudaFuncSetAttribute(
            calibration_kernel<P>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            kDynamicSharedBytes));

        DeviceBuffer<uint64_t> latency_cycles(1);
        DeviceBuffer<float> latency_sink(1);
        DeviceBuffer<uint32_t> latency_smids(1);
        const auto raw_cycle_samples = measure_clock_cycles(
            options.warmup, options.samples, latency_cycles.data(), [&] {
                calibration_kernel<P><<<1, kThreads, kDynamicSharedBytes>>>(
                    tensor_map, latency_cycles.data(), latency_sink.data(),
                    latency_smids.data(), options.iters, working_pages);
            });
        const auto latency_samples =
            divide_samples(raw_cycle_samples, options.iters);
        const double cycles_per_page = median(latency_samples);

        DeviceBuffer<uint64_t> throughput_cycles(blocks);
        DeviceBuffer<float> throughput_sinks(blocks);
        DeviceBuffer<uint32_t> throughput_smids(blocks);
        const auto event_samples_ms = measure_event_ms(
            options.warmup, options.samples, [&] {
                calibration_kernel<P><<<blocks, kThreads, kDynamicSharedBytes>>>(
                    tensor_map, throughput_cycles.data(),
                    throughput_sinks.data(), throughput_smids.data(),
                    options.iters, working_pages);
            });
        const double pages = static_cast<double>(blocks) * options.iters;
        const auto throughput_samples = rate_samples(event_samples_ms, pages, 1.0e6);
        const double pages_per_second = median(throughput_samples);
        const double elapsed_ms = median(event_samples_ms);
        const double instructions = pages * kQkInstructions;
        const double achieved_tflops =
            instructions * kFlopsPerInstruction / elapsed_ms / 1.0e9;
        const double requested_bytes = pages * kPageBytes;
        const auto bandwidth_samples =
            rate_samples(event_samples_ms, requested_bytes, 1.0e6);
        const double bandwidth_gbs = median(bandwidth_samples);

        const double expected = kQkInstructions * kContribution;
        const double tolerance = std::max(1.0e-6, expected * 1.0e-5);
        for (const float value : throughput_sinks.copy_to_host()) {
            if (!std::isfinite(value) || std::abs(value - expected) > tolerance) {
                throw std::runtime_error("KQ calibration sink mismatch");
            }
        }

        JsonObject params;
        params.add("gpu", properties.name).add("dtype", kDtype)
            .add("protocol", P == Protocol::kFirstPage
                ? "first_page_all_k_ready" : "steady_page_tile_pipelined")
            .add("k_tma_transactions", kTiles)
            .add("k_tma_transaction_bytes", kTileBytes)
            .add("qk_wgmma_instructions", kQkInstructions)
            .add("qk_ss_instructions",
                 P == Protocol::kFirstPage ? 36 : 32)
            .add("qk_rs_instructions",
                 P == Protocol::kFirstPage ? 0 : 4)
            .add("committed_groups",
                 P == Protocol::kFirstPage ? 1 : 9)
            .add("final_wait_group", 0)
            .add("working_set_pages", working_pages)
            .add("working_set_bytes",
                 static_cast<uint64_t>(working_pages) * kPageBytes)
            .add("iters", options.iters).add("warmup", options.warmup)
            .add("samples", options.samples).add("blocks", options.blocks)
            .add("resolved_blocks", blocks).add("device", options.device)
            .add("peak", options.peak).add("correct", true);
        const auto observed_smids = throughput_smids.copy_to_host();

        JsonObject latency;
        latency.add("value", cycles_per_page).add("unit", "cycle/page")
            .add("timer", "clock64").add("scope", "single_warpgroup_cta")
            .add("boundary", P == Protocol::kFirstPage
                ? "9 TMA issue+all barrier wait+36 SS+commit+wait0"
                : "9 TMA issue+(barrier wait+4 QK+commit)x9+wait0")
            .add_raw("samples", json_number_array(latency_samples))
            .add_raw("observed_smids", json_number_array(observed_smids));
        JsonObject throughput;
        throughput.add("value", pages_per_second).add("unit", "Gpage/s")
            .add("timer", "cuda_event").add("scope", "grid")
            .add("event_ms", elapsed_ms).add("pages", pages)
            .add("achieved_tflops", achieved_tflops)
            .add_raw("samples", json_number_array(throughput_samples))
            .add_raw("event_samples_ms", json_number_array(event_samples_ms));
        JsonObject bandwidth;
        bandwidth.add("value", bandwidth_gbs).add("unit", "GB/s")
            .add("kind", "requested_K_TMA").add("bytes", requested_bytes)
            .add_raw("samples", json_number_array(bandwidth_samples));
        const std::string name =
            std::string(P == Protocol::kFirstPage
                ? "first_score_" : "steady_score_") + kDtype;
        print_result(name, params, latency, throughput, bandwidth,
                     utilization(pages_per_second, options.peak, "Gpage/s"));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << (P == Protocol::kFirstPage
                          ? "first_score_" : "steady_score_")
                  << kDtype << ": " << error.what() << '\n';
        return 1;
    }
}

#undef MB1_KQ_OUTPUTS
#undef MB1_KQ_ACCUM
#undef MB1_KQ_INPUT_TYPE

}  // namespace microbench::kq_calibration_bench
