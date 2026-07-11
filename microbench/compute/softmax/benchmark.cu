// Online-softmax vector chain for SM90 WGMMA m64n64 fragments.

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cutlass/arch/barrier.h>

#include "benchmark_utils.hpp"
#include "clock.cuh"

namespace {

constexpr int kWarpgroupThreads = 128;
constexpr int kRowsPerThread = 2;
constexpr int kScoresPerRowThread = 16;
constexpr int kOutputPerRowThread = 64;
constexpr float kScaleLog2 = 0.125f;

enum class SoftmaxMode : int {
    kLocal = 0,
    kDensePair = 1,
};

__device__ __forceinline__ int fragment_row(int local_row, int lane_in_wg) {
    return (lane_in_wg / 32) * 16 + local_row * 8 +
           ((lane_in_wg % 32) / 4);
}

__device__ __forceinline__ float row_max(float values[kScoresPerRowThread]) {
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < kScoresPerRowThread; ++i) {
        maximum = fmaxf(maximum, values[i]);
    }
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, 1));
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, 2));
    return maximum;
}

__device__ __forceinline__ uint32_t bf16x2_bits(__nv_bfloat162 value) {
    return *reinterpret_cast<uint32_t*>(&value);
}

__device__ __forceinline__ float online_step(
        float scores[kRowsPerThread][kScoresPerRowThread],
        float output[kRowsPerThread][kOutputPerRowThread],
        float maxima[kRowsPerThread],
        float sums[kRowsPerThread],
        float output_scales[kRowsPerThread],
        uint32_t conversion_checks[4],
        const float comparison_maxima[kRowsPerThread] = nullptr,
        float comparison_scales[kRowsPerThread] = nullptr) {
    float checksum = 0.0f;
#pragma unroll
    for (int row = 0; row < kRowsPerThread; ++row) {
        const float current_max = row_max(scores[row]) * kScaleLog2;
        const float old_max = maxima[row];
        const float comparison_max = comparison_maxima == nullptr
            ? old_max
            : comparison_maxima[row];
        const float new_max = fmaxf(comparison_max, current_max);
        const float old_scale = exp2f(old_max - new_max);
        maxima[row] = new_max;
        output_scales[row] = old_scale;
        if (comparison_scales != nullptr) {
            comparison_scales[row] = exp2f(comparison_max - new_max);
        }

#pragma unroll
        for (int i = 0; i < kOutputPerRowThread; ++i) {
            output[row][i] *= old_scale;
            // Model the full register-resident O fragment. Without the
            // compiler dependency, only the few values used by the checksum
            // survive optimization and the rescale cost is under-counted.
            asm volatile("" : "+f"(output[row][i]));
        }

        float current_sum = 0.0f;
#pragma unroll
        for (int i = 0; i < kScoresPerRowThread; i += 2) {
            scores[row][i] = exp2f(scores[row][i] * kScaleLog2 - new_max);
            scores[row][i + 1] =
                exp2f(scores[row][i + 1] * kScaleLog2 - new_max);
            conversion_checks[(i / 2) & 3] += bf16x2_bits(
                __floats2bfloat162_rn(scores[row][i], scores[row][i + 1]));
            current_sum += scores[row][i] + scores[row][i + 1];
        }
        sums[row] = sums[row] * old_scale + current_sum;
        checksum += new_max + sums[row] + output[row][row];
    }
    return checksum;
}

template <SoftmaxMode Mode>
__global__ void softmax_kernel(uint64_t* __restrict__ starts,
                               uint64_t* __restrict__ stops,
                               float* __restrict__ sink,
                               float* __restrict__ fragment_sink,
                               uint32_t* __restrict__ conversion_sink,
                               int repeat) {
    constexpr int kWarpgroups = Mode == SoftmaxMode::kDensePair ? 2 : 1;
    __shared__ float shared_max[64];
    __shared__ float shared_merge_scale[64];

    const int tid = threadIdx.x;
    const int warpgroup = tid / kWarpgroupThreads;
    const int lane_in_wg = tid % kWarpgroupThreads;
    if (tid < 64) {
        shared_max[tid] = -1.0f;
        shared_merge_scale[tid] = 1.0f;
    }
    __syncthreads();

    float scores[kRowsPerThread][kScoresPerRowThread];
    float output[kRowsPerThread][kOutputPerRowThread];
    float maxima[kRowsPerThread] = {-1.0f, -1.0f};
    float sums[kRowsPerThread] = {0.0f, 0.0f};
    uint32_t conversion_checks[4] = {
        static_cast<uint32_t>(tid + 1),
        static_cast<uint32_t>(tid + 3),
        static_cast<uint32_t>(tid + 5),
        static_cast<uint32_t>(tid + 7),
    };
#pragma unroll
    for (int row = 0; row < kRowsPerThread; ++row) {
#pragma unroll
        for (int i = 0; i < kScoresPerRowThread; ++i) {
            scores[row][i] = -0.5f + 0.00390625f *
                static_cast<float>((lane_in_wg * 17 + row * 31 + i * 7) & 127) +
                static_cast<float>(warpgroup) * 0.125f;
        }
#pragma unroll
        for (int i = 0; i < kOutputPerRowThread; ++i) {
            output[row][i] = 0.001f *
                static_cast<float>(1 + ((lane_in_wg + row + i) & 31));
        }
    }

    float checksum = static_cast<float>(tid + 1);
    uint64_t start = 0;
    uint64_t stop = 0;
    CLK_START(start);
#pragma unroll 1
    for (int iteration = 0; iteration < repeat; ++iteration) {
        if constexpr (Mode == SoftmaxMode::kLocal) {
            float scales[kRowsPerThread];
            checksum += online_step(scores, output, maxima, sums, scales,
                                    conversion_checks);
        } else {
            if (warpgroup == 0) {
                float old_max[kRowsPerThread];
#pragma unroll
                for (int row = 0; row < kRowsPerThread; ++row) {
                    old_max[row] = shared_max[fragment_row(row, lane_in_wg)];
                }
                float scales[kRowsPerThread];
                checksum += online_step(scores, output, old_max, sums, scales,
                                        conversion_checks);
#pragma unroll
                for (int row = 0; row < kRowsPerThread; ++row) {
                    const int global_row = fragment_row(row, lane_in_wg);
                    if ((lane_in_wg & 3) == 0) {
                        shared_max[global_row] = old_max[row];
                    }
                    maxima[row] = old_max[row];
                }
            }
            cutlass::arch::NamedBarrier::arrive_and_wait(
                kWarpgroups * kWarpgroupThreads, 0);

            if (warpgroup == 1) {
                float old_max[kRowsPerThread];
#pragma unroll
                for (int row = 0; row < kRowsPerThread; ++row) {
                    old_max[row] = shared_max[fragment_row(row, lane_in_wg)];
                }
                float scales[kRowsPerThread];
                float merge_scales[kRowsPerThread];
                checksum += online_step(scores, output, maxima, sums, scales,
                                        conversion_checks, old_max,
                                        merge_scales);
#pragma unroll
                for (int row = 0; row < kRowsPerThread; ++row) {
                    const int global_row = fragment_row(row, lane_in_wg);
                    if ((lane_in_wg & 3) == 0) {
                        shared_max[global_row] = maxima[row];
                        shared_merge_scale[global_row] = merge_scales[row];
                    }
                }
            }
            cutlass::arch::NamedBarrier::arrive_and_wait(
                kWarpgroups * kWarpgroupThreads, 1);

            if (warpgroup == 0) {
#pragma unroll
                for (int row = 0; row < kRowsPerThread; ++row) {
                    const float correction =
                        shared_merge_scale[fragment_row(row, lane_in_wg)];
#pragma unroll
                    for (int i = 0; i < kScoresPerRowThread; i += 2) {
                        conversion_checks[(i / 2) & 3] += bf16x2_bits(
                            __floats2bfloat162_rn(
                                scores[row][i] * correction,
                                scores[row][i + 1] * correction));
                    }
#pragma unroll
                    for (int i = 0; i < kOutputPerRowThread; ++i) {
                        output[row][i] *= correction;
                        asm volatile("" : "+f"(output[row][i]));
                    }
                    sums[row] *= correction;
                    maxima[row] =
                        shared_max[fragment_row(row, lane_in_wg)];
                }
            }
            cutlass::arch::NamedBarrier::arrive_and_wait(
                kWarpgroups * kWarpgroupThreads, 2);
        }
    }
    CLK_STOP(stop);

    starts[tid] = start;
    stops[tid] = stop;
    sink[tid] = checksum + output[0][0] + scores[1][0] + sums[0];
    conversion_sink[tid] = conversion_checks[0] + conversion_checks[1] +
                           conversion_checks[2] + conversion_checks[3];
#pragma unroll
    for (int row = 0; row < kRowsPerThread; ++row) {
#pragma unroll
        for (int i = 0; i < kOutputPerRowThread; ++i) {
            fragment_sink[(tid * kRowsPerThread + row) *
                              kOutputPerRowThread +
                          i] = output[row][i];
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const microbench::CliArgs args(argc, argv);
        const std::string mode = args.get_string("mode", "local");
        if (mode != "local" && mode != "dense-pair") {
            throw std::invalid_argument("--mode must be local or dense-pair");
        }
        const int repeat = args.get_int("repeat", 256, 1, 1 << 20);
        const int warmup = args.get_int("warmup", 5, 0, 1000);
        const int samples = args.get_int("samples", 20, 1, 10000);
        const auto device = microbench::require_sm90(args.get_int("device", 0));
        const int threads = mode == "dense-pair" ? 256 : 128;

        microbench::DeviceBuffer<uint64_t> starts(threads);
        microbench::DeviceBuffer<uint64_t> stops(threads);
        microbench::DeviceBuffer<float> sink(threads);
        microbench::DeviceBuffer<float> fragment_sink(
            static_cast<std::size_t>(threads) * kRowsPerThread *
            kOutputPerRowThread);
        microbench::DeviceBuffer<uint32_t> conversion_sink(threads);

        auto measure_once = [&]() -> double {
            if (mode == "dense-pair") {
                softmax_kernel<SoftmaxMode::kDensePair><<<1, threads>>>(
                    starts.data(), stops.data(), sink.data(),
                    fragment_sink.data(), conversion_sink.data(), repeat);
            } else {
                softmax_kernel<SoftmaxMode::kLocal><<<1, threads>>>(
                    starts.data(), stops.data(), sink.data(),
                    fragment_sink.data(), conversion_sink.data(), repeat);
            }
            microbench::throw_if_cuda_error(cudaGetLastError(), "softmax kernel launch");
            microbench::throw_if_cuda_error(cudaDeviceSynchronize(),
                                             "softmax kernel synchronize");
            const auto host_starts = starts.copy_to_host();
            const auto host_stops = stops.copy_to_host();
            return static_cast<double>(microbench::reduce_cycles(
                host_starts.data(), host_stops.data(), host_starts.size()));
        };

        const auto series = microbench::run_samples(warmup, samples, measure_once);
        const auto summary = series.summary();
        const auto host_sink = sink.copy_to_host();
        const auto host_fragments = fragment_sink.copy_to_host();
        const auto host_conversion_sink = conversion_sink.copy_to_host();
        const double checksum =
            std::accumulate(host_sink.begin(), host_sink.end(), 0.0) +
            std::accumulate(host_fragments.begin(), host_fragments.end(), 0.0) +
            static_cast<double>(std::accumulate(
                host_conversion_sink.begin(), host_conversion_sink.end(),
                uint64_t{0}));
        if (!std::isfinite(checksum) || checksum == 0.0) {
            throw std::runtime_error("softmax sink is not finite and observable");
        }

        const int pages_per_iteration = mode == "dense-pair" ? 2 : 1;
        const double exp_elements = static_cast<double>(threads) *
            kRowsPerThread * kScoresPerRowThread * repeat;
        microbench::JsonLine json;
        json.add("benchmark", "softmax/online_m64n64_exp2_shfl_sm90")
            .add("gpu", device.properties.name)
            .add("mode", mode)
            .add("repeat", repeat)
            .add("threads", threads)
            .add("pages_per_iteration", pages_per_iteration)
            .add("checksum", checksum);
        microbench::add_measurement_summary(json, summary);
        json.add("cycle_per_iteration", summary.median / repeat)
            .add("cycle_per_page", summary.median /
                    (static_cast<double>(repeat) * pages_per_iteration))
            .add("exp2_element_per_clk_sm", exp_elements / summary.median)
            .print();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "softmax benchmark error: " << error.what() << '\n';
        return 1;
    }
}
