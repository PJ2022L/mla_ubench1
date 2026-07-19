#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "bench.hpp"

namespace microbench::combine_stage_bench {

constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kHeadDimension = 512;
constexpr int kVectorsPerThread = 4;

#if defined(MB1_COMBINE_USE_F16)
constexpr uint16_t kOneBits = 0x3c00u;
constexpr const char* kDtype = "fp16";
#define MB1_COMBINE_CVT "cvt.rn.f16.f32"
#else
constexpr uint16_t kOneBits = 0x3f80u;
constexpr const char* kDtype = "bf16";
#define MB1_COMBINE_CVT "cvt.rn.bf16.f32"
#endif

struct Float4 { float x, y, z, w; };

__device__ __forceinline__ float positive_infinity() {
    return __int_as_float(0x7f800000);
}

__device__ __forceinline__ float negative_infinity() {
    return __int_as_float(0xff800000);
}

__device__ __forceinline__ Float4 load_float4(const float* pointer) {
    Float4 value;
    asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(value.x), "=f"(value.y), "=f"(value.z), "=f"(value.w)
                 : "l"(pointer) : "memory");
    return value;
}

__device__ __forceinline__ float load_f32(const float* pointer) {
    float value;
    asm volatile("ld.global.f32 %0, [%1];"
                 : "=f"(value) : "l"(pointer) : "memory");
    return value;
}

__device__ __forceinline__ int load_nc_i32(const int* pointer) {
    int value;
    asm volatile("ld.global.nc.u32 %0, [%1];"
                 : "=r"(value) : "l"(pointer) : "memory");
    return value;
}

__device__ __forceinline__ float shfl_xor_f32(float value, int delta) {
    float result;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
                 : "=f"(result) : "f"(value), "r"(delta));
    return result;
}

__device__ __forceinline__ float exp2_approx(float value) {
    float result;
    asm volatile("ex2.approx.ftz.f32 %0, %1;"
                 : "=f"(result) : "f"(value));
    return result;
}

__device__ __forceinline__ float log2_approx(float value) {
    float result;
    asm volatile("lg2.approx.ftz.f32 %0, %1;"
                 : "=f"(result) : "f"(value));
    return result;
}

__device__ __forceinline__ float ffma(float a, float b, float c) {
    float result;
    asm volatile("fma.rn.ftz.f32 %0, %1, %2, %3;"
                 : "=f"(result) : "f"(a), "f"(b), "f"(c));
    return result;
}

__device__ __forceinline__ uint16_t convert_output(float value) {
    uint16_t result;
    asm volatile(MB1_COMBINE_CVT " %0, %1;"
                 : "=h"(result) : "f"(value));
    return result;
}

__device__ __forceinline__ void store_output4(uint16_t* pointer,
                                              const Float4& value) {
    const uint16_t h0 = convert_output(value.x);
    const uint16_t h1 = convert_output(value.y);
    const uint16_t h2 = convert_output(value.z);
    const uint16_t h3 = convert_output(value.w);
    const uint64_t packed = static_cast<uint64_t>(h0) |
        (static_cast<uint64_t>(h1) << 16) |
        (static_cast<uint64_t>(h2) << 32) |
        (static_cast<uint64_t>(h3) << 48);
    asm volatile("st.global.u64 [%0], %1;"
                 : : "l"(pointer), "l"(packed) : "memory");
}

__global__ void initialize_inputs(float* oaccum,
                                  float* lseaccum,
                                  int total_splits,
                                  int num_splits) {
    const std::size_t linear =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    const std::size_t o_elements =
        static_cast<std::size_t>(total_splits) * kWarps * kHeadDimension;
    for (std::size_t index = linear; index < o_elements; index += stride) {
        oaccum[index] = 1.0f;
    }
    const std::size_t lse_elements =
        static_cast<std::size_t>(total_splits) * kWarps;
    for (std::size_t index = linear; index < lse_elements; index += stride) {
        const int split = static_cast<int>(index / kWarps) % num_splits;
        lseaccum[index] = split == 0 ? 0.0f : negative_infinity();
    }
}

template <int MaxSplits>
__global__ __launch_bounds__(kThreads)
void combine_kernel(const float* oaccum,
                    const float* lseaccum,
                    const int* split_offsets,
                    uint16_t* output,
                    float* lse_output,
                    uint64_t* cycles,
                    float* sinks,
                    uint32_t* smids) {
    extern __shared__ float scales[];
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int start_split = load_nc_i32(split_offsets + blockIdx.x);
    const int end_split = load_nc_i32(split_offsets + blockIdx.x + 1);
    const int num_splits = end_split - start_split;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    const uint64_t start = read_clock64();

    const float* o_base = oaccum +
        (static_cast<std::size_t>(start_split) * kWarps + warp) *
            kHeadDimension;
    Float4 data[kVectorsPerThread];
#pragma unroll
    for (int vector = 0; vector < kVectorsPerThread; ++vector) {
        data[vector] = load_float4(
            o_base + lane * 4 + vector * 128);
    }

    constexpr int kLsePerThread = MaxSplits / 32;
    float local_lse[kLsePerThread];
#pragma unroll
    for (int index = 0; index < kLsePerThread; ++index) {
        const int split = index * 32 + lane;
        local_lse[index] = split < num_splits
            ? load_f32(lseaccum +
                (static_cast<std::size_t>(start_split + split) * kWarps + warp))
            : negative_infinity();
    }
    float max_lse = negative_infinity();
#pragma unroll
    for (int index = 0; index < kLsePerThread; ++index) {
        max_lse = fmaxf(max_lse, local_lse[index]);
    }
#pragma unroll
    for (int delta = 16; delta >= 1; delta /= 2) {
        max_lse = fmaxf(max_lse, shfl_xor_f32(max_lse, delta));
    }
    if (max_lse == negative_infinity()) max_lse = 0.0f;
    float sum_lse = 0.0f;
#pragma unroll
    for (int index = 0; index < kLsePerThread; ++index) {
        sum_lse += exp2_approx(local_lse[index] - max_lse);
    }
#pragma unroll
    for (int delta = 16; delta >= 1; delta /= 2) {
        sum_lse += shfl_xor_f32(sum_lse, delta);
    }
    const float global_lse = sum_lse == 0.0f
        ? positive_infinity()
        : log2_approx(sum_lse) + max_lse;
    if (lane == 0) {
        lse_output[static_cast<std::size_t>(blockIdx.x) * kWarps + warp] =
            global_lse / 1.4426950408889634f;
    }
#pragma unroll
    for (int index = 0; index < kLsePerThread; ++index) {
        const int split = index * 32 + lane;
        scales[warp * MaxSplits + split] =
            exp2_approx(local_lse[index] - global_lse);
    }
    __syncwarp();

    Float4 result[kVectorsPerThread];
#pragma unroll
    for (int vector = 0; vector < kVectorsPerThread; ++vector) {
        result[vector] = Float4{0.0f, 0.0f, 0.0f, 0.0f};
    }
#pragma unroll 1
    for (int split = 0; split < num_splits; ++split) {
        const float scale = scales[warp * MaxSplits + split];
#pragma unroll
        for (int vector = 0; vector < kVectorsPerThread; ++vector) {
            result[vector].x = ffma(scale, data[vector].x, result[vector].x);
            result[vector].y = ffma(scale, data[vector].y, result[vector].y);
            result[vector].z = ffma(scale, data[vector].z, result[vector].z);
            result[vector].w = ffma(scale, data[vector].w, result[vector].w);
            if (split + 1 < num_splits) {
                data[vector] = load_float4(
                    o_base + static_cast<std::size_t>(split + 1) *
                        kWarps * kHeadDimension +
                    lane * 4 + vector * 128);
            }
        }
    }
    uint16_t* out_base = output +
        (static_cast<std::size_t>(blockIdx.x) * kWarps + warp) *
            kHeadDimension;
#pragma unroll
    for (int vector = 0; vector < kVectorsPerThread; ++vector) {
        store_output4(out_base + lane * 4 + vector * 128, result[vector]);
    }
    const uint64_t stop = read_clock64();
    if (lane == 0) {
        cycles[static_cast<std::size_t>(blockIdx.x) * kWarps + warp] =
            stop - start;
        sinks[static_cast<std::size_t>(blockIdx.x) * kWarps + warp] =
            result[0].x;
        if (warp == 0) smids[blockIdx.x] = read_smid();
    }
}

template <int MaxSplits>
inline void launch(int blocks,
                   int shared_bytes,
                   cudaStream_t stream,
                   const float* oaccum,
                   const float* lseaccum,
                   const int* split_offsets,
                   uint16_t* output,
                   float* lse_output,
                   uint64_t* cycles,
                   float* sinks,
                   uint32_t* smids) {
    auto function = &combine_kernel<MaxSplits>;
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{
        dim3(blocks), dim3(kThreads), static_cast<std::size_t>(shared_bytes),
        stream, &attribute, 1};
    CUDA_CHECK(cudaLaunchKernelEx(
        &config, function, oaccum, lseaccum, split_offsets, output,
        lse_output, cycles, sinks, smids));
}

inline int bucket_for(int splits) {
    if (splits <= 32) return 32;
    if (splits <= 64) return 64;
    if (splits <= 96) return 96;
    if (splits <= 128) return 128;
    return 160;
}

template <typename Launch>
inline void dispatch(int bucket, Launch&& launch) {
    if (bucket == 32) launch(std::integral_constant<int, 32>{});
    else if (bucket == 64) launch(std::integral_constant<int, 64>{});
    else if (bucket == 96) launch(std::integral_constant<int, 96>{});
    else if (bucket == 128) launch(std::integral_constant<int, 128>{});
    else launch(std::integral_constant<int, 160>{});
}

inline int run(int argc, char** argv) {
    try {
        const Args args(argc, argv);
        args.require_only({"iters", "warmup", "samples", "blocks", "device",
                           "peak", "num-splits"});
        const auto options = parse_common_options(args, 1);
        if (options.iters != 1) {
            throw std::invalid_argument(
                "combine stage uses one source-real stage per kernel; --iters=1");
        }
        const auto properties = require_sm90(options.device);
        const int num_splits = args.get_int("num-splits", 8, 2, 160);
        const int bucket = bucket_for(num_splits);
        const int blocks = resolve_blocks(options.blocks, properties, 1);
        const int shared_bytes = kWarps * bucket * sizeof(float);
        dispatch(bucket, [&](auto max_splits) {
            constexpr int kMax = decltype(max_splits)::value;
            CUDA_CHECK(cudaFuncSetAttribute(
                combine_kernel<kMax>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                shared_bytes));
        });

        const int total_splits = blocks * num_splits;
        const std::size_t o_elements =
            static_cast<std::size_t>(total_splits) * kWarps * kHeadDimension;
        const std::size_t lse_elements =
            static_cast<std::size_t>(total_splits) * kWarps;
        DeviceBuffer<float> oaccum(o_elements);
        DeviceBuffer<float> lseaccum(lse_elements);
        DeviceBuffer<int> split_offsets(blocks + 1);
        DeviceBuffer<uint16_t> output(
            static_cast<std::size_t>(blocks) * kWarps * kHeadDimension);
        DeviceBuffer<float> lse_output(static_cast<std::size_t>(blocks) * kWarps);
        DeviceBuffer<uint64_t> cycles(static_cast<std::size_t>(blocks) * kWarps);
        DeviceBuffer<float> sinks(static_cast<std::size_t>(blocks) * kWarps);
        DeviceBuffer<uint32_t> smids(blocks);
        initialize_inputs<<<256, 256>>>(
            oaccum.data(), lseaccum.data(), total_splits, num_splits);
        std::vector<int> host_offsets(blocks + 1);
        for (int block = 0; block <= blocks; ++block) {
            host_offsets[block] = block * num_splits;
        }
        CUDA_CHECK(cudaMemcpy(split_offsets.data(), host_offsets.data(),
                              host_offsets.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        auto launch_grid = [&](int launch_blocks) {
            dispatch(bucket, [&](auto max_splits) {
                constexpr int kMax = decltype(max_splits)::value;
                launch<kMax>(launch_blocks, shared_bytes, nullptr,
                             oaccum.data(), lseaccum.data(), split_offsets.data(),
                             output.data(), lse_output.data(), cycles.data(),
                             sinks.data(), smids.data());
            });
        };
        const auto raw_cycle_samples = measure_clock_cycles(
            options.warmup, options.samples, cycles.data(), [&] {
                launch_grid(1);
            }, kWarps);
        const auto latency_samples = raw_cycle_samples;
        const double cycles_per_cta = median(latency_samples);
        const auto event_samples_ms = measure_event_ms(
            options.warmup, options.samples, [&] { launch_grid(blocks); });
        auto throughput_samples = event_samples_ms;
        for (double& value : throughput_samples) {
            value = blocks / value / 1.0e6;
        }
        const double throughput_gcta = median(throughput_samples);
        const double elapsed_ms = median(event_samples_ms);
        const double read_bytes_per_cta =
            static_cast<double>(num_splits) * kWarps *
                (kHeadDimension * sizeof(float) + sizeof(float)) +
            2 * sizeof(int);
        const double write_bytes_per_cta =
            kWarps * (kHeadDimension * sizeof(uint16_t) + sizeof(float));
        const double bytes = blocks * (read_bytes_per_cta + write_bytes_per_cta);
        auto bandwidth_samples = event_samples_ms;
        for (double& value : bandwidth_samples) value = bytes / value / 1.0e6;
        const double bandwidth_gbs = median(bandwidth_samples);

        for (const float value : sinks.copy_to_host()) {
            if (!std::isfinite(value) || std::abs(value - 1.0f) > 1.0e-5f) {
                throw std::runtime_error("combine FP32 sink mismatch");
            }
        }
        std::vector<uint16_t> output_prefix(64);
        CUDA_CHECK(cudaMemcpy(output_prefix.data(), output.data(),
                              output_prefix.size() * sizeof(uint16_t),
                              cudaMemcpyDeviceToHost));
        for (const uint16_t value : output_prefix) {
            if (value != kOneBits) {
                throw std::runtime_error("combine converted output mismatch");
            }
        }
        const auto host_lse = lse_output.copy_to_host();
        for (const float value : host_lse) {
            if (std::abs(value) > 1.0e-6f) {
                throw std::runtime_error("combine LSE output mismatch");
            }
        }

        JsonObject params;
        params.add("gpu", properties.name).add("dtype", kDtype)
            .add("warps", kWarps).add("threads", kThreads)
            .add("head_dimension_v", kHeadDimension)
            .add("num_splits", num_splits).add("max_splits_bucket", bucket)
            .add("shared_bytes", shared_bytes)
            .add("iters", options.iters).add("warmup", options.warmup)
            .add("samples", options.samples).add("blocks", options.blocks)
            .add("resolved_blocks", blocks).add("device", options.device)
            .add("peak", options.peak).add("correct", true);
        params.add_raw("observed_smids",
                       json_number_array(smids.copy_to_host()));
        JsonObject latency;
        latency.add("value", cycles_per_cta).add("unit", "cycle/cta")
            .add("timer", "clock64").add("scope", "max_across_8_warps")
            .add("boundary", "griddep wait through LSE/reduction/convert/store")
            .add_raw("samples", json_number_array(latency_samples));
        JsonObject throughput;
        throughput.add("value", throughput_gcta).add("unit", "Gcta/s")
            .add("timer", "cuda_event").add("scope", "grid")
            .add("event_ms", elapsed_ms)
            .add_raw("samples", json_number_array(throughput_samples))
            .add_raw("event_samples_ms", json_number_array(event_samples_ms));
        JsonObject bandwidth;
        bandwidth.add("value", bandwidth_gbs).add("unit", "GB/s")
            .add("kind", "requested_global").add("bytes", bytes)
            .add_raw("samples", json_number_array(bandwidth_samples));
        const std::string name =
            std::string("dense_decode.calibration.combine_stage_") + kDtype;
        print_result(name, params, latency, throughput, bandwidth,
                     utilization(bandwidth_gbs, options.peak, "GB/s"));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "combine_stage_" << kDtype << ": "
                  << error.what() << '\n';
        return 1;
    }
}

#undef MB1_COMBINE_CVT

}  // namespace microbench::combine_stage_bench
