// FlashMLA D_V=512 split-KV combine reduction benchmark.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include "benchmark_utils.hpp"
#include "clock.cuh"

namespace {

using microbench::CliArgs;
using microbench::DeviceBuffer;

constexpr int kThreads = 256;
constexpr int kWarps = 8;
constexpr int kHeads = 8;
constexpr int kHeadDimension = 512;
constexpr int kFloat4PerThread = kHeadDimension / (32 * 4);
static_assert(kFloat4PerThread == 4);

enum class Pattern : int {
    Sequential = 0,
    Random = 1,
};

__host__ __device__ __forceinline__ int select_rowset(
    Pattern pattern, int iteration, int rowsets, int block, int grid_blocks) {
    if (pattern == Pattern::Sequential) {
        return (block + iteration * grid_blocks) % rowsets;
    }

    uint32_t value = static_cast<uint32_t>(block) * 0x9e3779b9u +
                     static_cast<uint32_t>(iteration + 1) * 0x85ebca6bu;
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return static_cast<int>(value % static_cast<uint32_t>(rowsets));
}

template <int MaxSplits>
__global__ __launch_bounds__(kThreads)
void splitkv_reduce_kernel(const float* __restrict__ o_accum,
                           const float* __restrict__ lse_accum,
                           uint16_t* __restrict__ output,
                           float* __restrict__ lse_output,
                           uint64_t* __restrict__ starts,
                           uint64_t* __restrict__ stops,
                           int num_splits,
                           int rowsets,
                           int repeat,
                           Pattern pattern) {
    static_assert(MaxSplits % 32 == 0);
    constexpr int kLsePerThread = MaxSplits / 32;
    extern __shared__ float shared_scales[];

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int o_split_stride = kHeads * kHeadDimension;
    const int o_rowset_stride = num_splits * o_split_stride;
    const int lse_split_stride = kHeads;
    const int lse_rowset_stride = num_splits * lse_split_stride;
    const int output_block_stride = kHeads * kHeadDimension;

    uint64_t start = 0;
    uint64_t stop = 0;
    CLK_START(start);
#pragma unroll 1
    for (int iteration = 0; iteration < repeat; ++iteration) {
        const int rowset = select_rowset(
            pattern, iteration, rowsets, static_cast<int>(blockIdx.x),
            static_cast<int>(gridDim.x));
        const float* const rowset_o =
            o_accum + static_cast<int64_t>(rowset) * o_rowset_stride;
        const float* const rowset_lse =
            lse_accum + static_cast<int64_t>(rowset) * lse_rowset_stride;
        const float* const head_o = rowset_o + warp * kHeadDimension;

        // Match combine.cu: prefetch split zero before reducing the LSEs.
        float4 data[kFloat4PerThread];
#pragma unroll
        for (int i = 0; i < kFloat4PerThread; ++i) {
            data[i] = *reinterpret_cast<const float4*>(
                head_o + lane * 4 + i * 128);
        }

        float local_lse[kLsePerThread];
#pragma unroll
        for (int i = 0; i < kLsePerThread; ++i) {
            const int split = i * 32 + lane;
            local_lse[i] = split < num_splits
                ? rowset_lse[split * lse_split_stride + warp]
                : -CUDART_INF_F;
        }

        float max_lse = -CUDART_INF_F;
#pragma unroll
        for (int i = 0; i < kLsePerThread; ++i) {
            max_lse = fmaxf(max_lse, local_lse[i]);
        }
#pragma unroll
        for (int offset = 16; offset >= 1; offset /= 2) {
            max_lse = fmaxf(
                max_lse,
                __shfl_xor_sync(0xffffffffu, max_lse, offset));
        }
        max_lse = max_lse == -CUDART_INF_F ? 0.0f : max_lse;

        float sum_lse = 0.0f;
#pragma unroll
        for (int i = 0; i < kLsePerThread; ++i) {
            sum_lse += exp2f(local_lse[i] - max_lse);
        }
#pragma unroll
        for (int offset = 16; offset >= 1; offset /= 2) {
            sum_lse += __shfl_xor_sync(0xffffffffu, sum_lse, offset);
        }
        const float global_lse =
            (sum_lse == 0.0f || sum_lse == -CUDART_INF_F)
                ? CUDART_INF_F
                : log2f(sum_lse) + max_lse;
        if (lane == 0) {
            lse_output[static_cast<int64_t>(blockIdx.x) * kHeads + warp] =
                global_lse / CUDART_L2E_F;
        }

#pragma unroll
        for (int i = 0; i < kLsePerThread; ++i) {
            const int split = i * 32 + lane;
            shared_scales[warp * MaxSplits + split] =
                exp2f(local_lse[i] - global_lse);
        }
        __syncwarp();

        float4 result[kFloat4PerThread];
#pragma unroll
        for (int i = 0; i < kFloat4PerThread; ++i) {
            result[i] = {0.0f, 0.0f, 0.0f, 0.0f};
        }

#pragma unroll 1
        for (int split = 0; split < num_splits; ++split) {
            const float scale = shared_scales[warp * MaxSplits + split];
#pragma unroll
            for (int i = 0; i < kFloat4PerThread; ++i) {
                result[i].x = fmaf(scale, data[i].x, result[i].x);
                result[i].y = fmaf(scale, data[i].y, result[i].y);
                result[i].z = fmaf(scale, data[i].z, result[i].z);
                result[i].w = fmaf(scale, data[i].w, result[i].w);
                if (split != num_splits - 1) {
                    data[i] = *reinterpret_cast<const float4*>(
                        head_o + (split + 1) * o_split_stride +
                        lane * 4 + i * 128);
                }
            }
        }

        uint16_t* const head_output =
            output + static_cast<int64_t>(blockIdx.x) * output_block_stride +
            warp * kHeadDimension;
#pragma unroll
        for (int i = 0; i < kFloat4PerThread; ++i) {
            const __nv_bfloat16 converted[4] = {
                __float2bfloat16(result[i].x),
                __float2bfloat16(result[i].y),
                __float2bfloat16(result[i].z),
                __float2bfloat16(result[i].w),
            };
            *reinterpret_cast<uint64_t*>(
                head_output + lane * 4 + i * 128) =
                *reinterpret_cast<const uint64_t*>(converted);
        }
    }
    CLK_STOP(stop);

    const int uid = static_cast<int>(blockIdx.x) * kThreads + tid;
    starts[uid] = start;
    stops[uid] = stop;
}

template <int MaxSplits>
void configure_kernel_shared_memory() {
    const int shared_bytes = kWarps * MaxSplits * sizeof(float);
    microbench::throw_if_cuda_error(
        cudaFuncSetAttribute(splitkv_reduce_kernel<MaxSplits>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             shared_bytes),
        "splitkv_reduce_kernel dynamic shared-memory attribute");
}

template <int MaxSplits>
void launch_reduce(const float* o_accum,
                   const float* lse_accum,
                   uint16_t* output,
                   float* lse_output,
                   uint64_t* starts,
                   uint64_t* stops,
                   int num_splits,
                   int rowsets,
                   int repeat,
                   Pattern pattern,
                   int blocks) {
    const int shared_bytes = kWarps * MaxSplits * sizeof(float);
    splitkv_reduce_kernel<MaxSplits><<<blocks, kThreads, shared_bytes>>>(
        o_accum, lse_accum, output, lse_output, starts, stops, num_splits,
        rowsets, repeat, pattern);
}

int max_splits_bucket(int num_splits) {
    if (num_splits <= 32) return 32;
    if (num_splits <= 64) return 64;
    if (num_splits <= 96) return 96;
    if (num_splits <= 128) return 128;
    return 160;
}

void configure_selected_kernel(int bucket) {
    switch (bucket) {
        case 32: configure_kernel_shared_memory<32>(); break;
        case 64: configure_kernel_shared_memory<64>(); break;
        case 96: configure_kernel_shared_memory<96>(); break;
        case 128: configure_kernel_shared_memory<128>(); break;
        case 160: configure_kernel_shared_memory<160>(); break;
        default: throw std::logic_error("invalid split bucket");
    }
}

void launch_selected_kernel(int bucket,
                            const float* o_accum,
                            const float* lse_accum,
                            uint16_t* output,
                            float* lse_output,
                            uint64_t* starts,
                            uint64_t* stops,
                            int num_splits,
                            int rowsets,
                            int repeat,
                            Pattern pattern,
                            int blocks) {
    switch (bucket) {
        case 32:
            launch_reduce<32>(o_accum, lse_accum, output, lse_output, starts,
                              stops, num_splits, rowsets, repeat, pattern,
                              blocks);
            break;
        case 64:
            launch_reduce<64>(o_accum, lse_accum, output, lse_output, starts,
                              stops, num_splits, rowsets, repeat, pattern,
                              blocks);
            break;
        case 96:
            launch_reduce<96>(o_accum, lse_accum, output, lse_output, starts,
                              stops, num_splits, rowsets, repeat, pattern,
                              blocks);
            break;
        case 128:
            launch_reduce<128>(o_accum, lse_accum, output, lse_output, starts,
                               stops, num_splits, rowsets, repeat, pattern,
                               blocks);
            break;
        case 160:
            launch_reduce<160>(o_accum, lse_accum, output, lse_output, starts,
                               stops, num_splits, rowsets, repeat, pattern,
                               blocks);
            break;
        default: throw std::logic_error("invalid split bucket");
    }
}

Pattern parse_pattern(const std::string& value) {
    if (value == "sequential") return Pattern::Sequential;
    if (value == "random") return Pattern::Random;
    throw std::invalid_argument("--pattern must be sequential or random");
}

double median_cta_cycles(const std::vector<uint64_t>& starts,
                         const std::vector<uint64_t>& stops,
                         int blocks) {
    std::vector<double> cycles(static_cast<std::size_t>(blocks));
    for (int block = 0; block < blocks; ++block) {
        const std::size_t offset = static_cast<std::size_t>(block) * kThreads;
        cycles[block] = static_cast<double>(microbench::reduce_cycles(
            starts.data() + offset, stops.data() + offset, kThreads));
    }
    std::sort(cycles.begin(), cycles.end());
    const std::size_t middle = cycles.size() / 2;
    return cycles.size() % 2 == 0
        ? 0.5 * (cycles[middle - 1] + cycles[middle])
        : cycles[middle];
}

float bfloat16_to_float(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

void validate_result(const std::vector<float>& o_accum,
                     const std::vector<float>& lse_accum,
                     const std::vector<uint16_t>& output,
                     const std::vector<float>& lse_output,
                     int num_splits,
                     int rowsets,
                     int repeat,
                     Pattern pattern,
                     int blocks) {
    constexpr int kColumnsToCheck[] = {0, 1, 127, 128, 255, 511};
    constexpr int kHeadsToCheck[] = {0, 7};
    const int block = 0;
    const int rowset = select_rowset(
        pattern, repeat - 1, rowsets, block, blocks);

    for (int head : kHeadsToCheck) {
        float max_lse = -std::numeric_limits<float>::infinity();
        for (int split = 0; split < num_splits; ++split) {
            const std::size_t index =
                (static_cast<std::size_t>(rowset) * num_splits + split) *
                    kHeads +
                head;
            max_lse = std::max(max_lse, lse_accum[index]);
        }
        float sum_lse = 0.0f;
        for (int split = 0; split < num_splits; ++split) {
            const std::size_t index =
                (static_cast<std::size_t>(rowset) * num_splits + split) *
                    kHeads +
                head;
            sum_lse += std::exp2(lse_accum[index] - max_lse);
        }
        const float global_lse = std::log2(sum_lse) + max_lse;
        const float expected_lse = global_lse / CUDART_L2E_F;
        const float observed_lse = lse_output[block * kHeads + head];
        if (std::abs(expected_lse - observed_lse) >
            2.0e-3f * std::max(1.0f, std::abs(expected_lse))) {
            throw std::runtime_error("split-KV LSE CPU validation failed");
        }

        for (int column : kColumnsToCheck) {
            float expected = 0.0f;
            for (int split = 0; split < num_splits; ++split) {
                const std::size_t lse_index =
                    (static_cast<std::size_t>(rowset) * num_splits + split) *
                        kHeads +
                    head;
                const float scale =
                    std::exp2(lse_accum[lse_index] - global_lse);
                const std::size_t o_index =
                    ((static_cast<std::size_t>(rowset) * num_splits + split) *
                         kHeads +
                     head) *
                        kHeadDimension +
                    column;
                expected = std::fma(scale, o_accum[o_index], expected);
            }
            const std::size_t output_index =
                (static_cast<std::size_t>(block) * kHeads + head) *
                    kHeadDimension +
                column;
            const float observed = bfloat16_to_float(output[output_index]);
            const float tolerance =
                std::max(5.0e-3f, 2.0e-2f * std::abs(expected));
            if (std::abs(expected - observed) > tolerance) {
                throw std::runtime_error(
                    "split-KV output CPU validation failed");
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const CliArgs args(argc, argv);
        const int num_splits = args.get_int("num-splits", 128, 2, 160);
        const std::string working_set =
            args.get_string("working-set", "hbm");
        if (working_set != "l2" && working_set != "hbm") {
            throw std::invalid_argument("--working-set must be l2 or hbm");
        }
        const int default_working_set_mib = working_set == "l2" ? 16 : 256;
        const int working_set_mib = args.get_int(
            "working-set-mib", default_working_set_mib, 1, 16384);
        const std::string pattern_name =
            args.get_string("pattern", "sequential");
        const Pattern pattern = parse_pattern(pattern_name);
        const int repeat = args.get_int("repeat", 32, 1, 1 << 16);
        const int warmup = args.get_int("warmup", 5, 0, 1000);
        const int samples = args.get_int("samples", 20, 1, 10000);
        const auto device = microbench::require_sm90(args.get_int("device", 0));
        const int blocks = args.get_int(
            "blocks", device.properties.multiProcessorCount, 1, 4096);

        constexpr std::size_t kMib = 1024u * 1024u;
        const std::size_t bytes_per_rowset =
            static_cast<std::size_t>(num_splits) * kHeads *
            (kHeadDimension * sizeof(float) + sizeof(float));
        const std::size_t requested_working_set_bytes =
            static_cast<std::size_t>(working_set_mib) * kMib;
        const std::size_t rowsets = std::max<std::size_t>(
            1, (requested_working_set_bytes + bytes_per_rowset - 1) /
                   bytes_per_rowset);
        if (rowsets > static_cast<std::size_t>(
                          std::numeric_limits<int>::max())) {
            throw std::overflow_error("working set requires too many rowsets");
        }

        const std::size_t o_elements = rowsets * num_splits * kHeads *
                                       kHeadDimension;
        const std::size_t lse_elements = rowsets * num_splits * kHeads;
        std::vector<float> host_o(o_elements);
        std::vector<float> host_lse(lse_elements);
        for (std::size_t i = 0; i < host_o.size(); ++i) {
            host_o[i] = 0.001f * static_cast<float>(1 + (i % 251));
        }
        for (std::size_t rowset = 0; rowset < rowsets; ++rowset) {
            for (int split = 0; split < num_splits; ++split) {
                for (int head = 0; head < kHeads; ++head) {
                    const std::size_t index =
                        (rowset * num_splits + split) * kHeads + head;
                    host_lse[index] =
                        -0.03125f * static_cast<float>(split) +
                        0.00390625f * static_cast<float>(head) +
                        0.00001f * static_cast<float>(rowset % 97);
                }
            }
        }

        DeviceBuffer<float> o_accum(host_o.size());
        DeviceBuffer<float> lse_accum(host_lse.size());
        DeviceBuffer<uint16_t> output(
            static_cast<std::size_t>(blocks) * kHeads * kHeadDimension);
        DeviceBuffer<float> lse_output(
            static_cast<std::size_t>(blocks) * kHeads);
        DeviceBuffer<uint64_t> starts(
            static_cast<std::size_t>(blocks) * kThreads);
        DeviceBuffer<uint64_t> stops(
            static_cast<std::size_t>(blocks) * kThreads);
        o_accum.copy_from_host(host_o);
        lse_accum.copy_from_host(host_lse);

        const int bucket = max_splits_bucket(num_splits);
        configure_selected_kernel(bucket);

        auto measure_once = [&]() -> double {
            launch_selected_kernel(
                bucket, o_accum.data(), lse_accum.data(), output.data(),
                lse_output.data(), starts.data(), stops.data(), num_splits,
                static_cast<int>(rowsets), repeat, pattern, blocks);
            microbench::throw_if_cuda_error(
                cudaGetLastError(), "splitkv_reduce_kernel launch");
            microbench::throw_if_cuda_error(
                cudaDeviceSynchronize(),
                "splitkv_reduce_kernel synchronize");
            const auto host_starts = starts.copy_to_host();
            const auto host_stops = stops.copy_to_host();
            return median_cta_cycles(host_starts, host_stops, blocks);
        };

        const auto series = microbench::run_samples(warmup, samples, measure_once);
        const auto summary = series.summary();
        const auto host_output = output.copy_to_host();
        const auto host_lse_output = lse_output.copy_to_host();
        const uint64_t checksum = std::accumulate(
            host_output.begin(), host_output.end(), uint64_t{0});
        if (checksum == 0) {
            throw std::runtime_error(
                "split-KV reduction sink is zero; target work may have been removed");
        }
        validate_result(host_o, host_lse, host_output, host_lse_output,
                        num_splits, static_cast<int>(rowsets), repeat,
                        pattern, blocks);

        const double partial_read_bytes_per_cta =
            static_cast<double>(repeat) * num_splits * kHeads *
            (kHeadDimension * sizeof(float) + sizeof(float));
        const double output_bytes_per_cta =
            static_cast<double>(repeat) * kHeads * kHeadDimension *
                sizeof(uint16_t) +
            static_cast<double>(repeat) * kHeads * sizeof(float);
        const double fmas_per_cta = static_cast<double>(repeat) *
                                    num_splits * kHeads * kHeadDimension;

        microbench::JsonLine json;
        json.add("benchmark", "splitkv_reduce/dv512_f32_sm90")
            .add("gpu", device.properties.name)
            .add("num_splits", num_splits)
            .add("max_splits_bucket", bucket)
            .add("working_set", working_set)
            .add("working_set_mib_requested", working_set_mib)
            .add("working_set_bytes_actual",
                 (o_elements + lse_elements) * sizeof(float))
            .add("rowsets", rowsets)
            .add("pattern", pattern_name)
            .add("blocks", blocks)
            .add("threads", kThreads)
            .add("repeat", repeat)
            .add("cpu_validation", true)
            .add("checksum", checksum);
        microbench::add_measurement_summary(json, summary);
        json.add("cycle_per_cta_iteration", summary.median / repeat)
            .add("cycle_per_row", summary.median /
                    (static_cast<double>(repeat) * kHeads))
            .add("partial_read_byte_per_clk_sm",
                 partial_read_bytes_per_cta / summary.median)
            .add("requested_byte_per_clk_sm",
                 (partial_read_bytes_per_cta + output_bytes_per_cta) /
                     summary.median)
            .add("fma_per_clk_sm", fmas_per_cta / summary.median)
            .print();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "split-KV reduction benchmark error: "
                  << error.what() << '\n';
        return 1;
    }
}
