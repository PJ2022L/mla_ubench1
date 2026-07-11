// Exact SM90a BF16 WGMMA latency and committed-group throughput benchmark.

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

#include "benchmark_utils.hpp"
#include "clock.cuh"

#ifndef BENCH_M
#define BENCH_M 64
#endif
#ifndef BENCH_N
#define BENCH_N 64
#endif
#ifndef BENCH_K
#define BENCH_K 16
#endif

static_assert(BENCH_M == 64, "this family implements one 64-row warpgroup tile");
static_assert(BENCH_N == 64 || BENCH_N == 256,
              "supported WGMMA N sizes are 64 and 256");
static_assert(BENCH_K == 16, "this family implements BF16 k16 WGMMA");

namespace wgmma_bench {

using Element = cutlass::bfloat16_t;
using microbench::CliArgs;
using microbench::DeviceBuffer;
using namespace cute;

constexpr int kThreadsPerWarpgroup = 128;
constexpr int kMaxResidentWarpgroups = 2;

template <int N, GMMA::Major BMajor>
struct WgmmaConfigBase {
    static constexpr int kM = 64;
    static constexpr int kN = N;
    static constexpr int kK = 16;
    static constexpr int kSmemK = 64;
    static constexpr GMMA::Major kMajorA = GMMA::Major::K;
    static constexpr GMMA::Major kMajorB = BMajor;
    static constexpr int kAccumulatorElements = kM * kN / kThreadsPerWarpgroup;

    using MnkShape = Shape<Int<kM>, Int<kN>, Int<kK>>;
    using SmemLayoutA = decltype(tile_to_shape(
        GMMA::Layout_K_SW128_Atom<Element>{},
        Shape<Int<kM>, Int<kSmemK>>{}));
    using SmemLayoutBK = decltype(tile_to_shape(
        GMMA::Layout_K_SW128_Atom<Element>{},
        Shape<Int<kN>, Int<kSmemK>>{}));
    using SmemLayoutBMN = decltype(tile_to_shape(
        GMMA::Layout_MN_SW128_Atom<Element>{},
        Shape<Int<kN>, Int<kSmemK>>{}));
    using SmemLayoutB = std::conditional_t<BMajor == GMMA::Major::K,
                                           SmemLayoutBK, SmemLayoutBMN>;

    using TiledMmaSS = decltype(make_tiled_mma(
        GMMA::ss_op_selector<Element, Element, float, MnkShape,
                             GMMA::Major::K, BMajor>(),
        Layout<Shape<_1, _1, _1>>{}));
    using TiledMmaRS = decltype(make_tiled_mma(
        GMMA::rs_op_selector<Element, Element, float, MnkShape,
                             GMMA::Major::K, BMajor>(),
        Layout<Shape<_1, _1, _1>>{}));

    static constexpr int kSmemAElements = cosize_v<SmemLayoutA>;
    static constexpr int kSmemBElements = cosize_v<SmemLayoutB>;
};

template <int N>
struct WgmmaConfig;

template <>
struct WgmmaConfig<64> : WgmmaConfigBase<64, GMMA::Major::K> {
    static constexpr const char* kBenchmark =
        "wgmma/m64n64k16_bf16_rs_ss_sm90";
    static constexpr const char* kMajorBName = "K";
};

template <>
struct WgmmaConfig<256> : WgmmaConfigBase<256, GMMA::Major::MN> {
    static constexpr const char* kBenchmark =
        "wgmma/m64n256k16_bf16_rs_ss_sm90";
    static constexpr const char* kMajorBName = "MN";
};

using Config = WgmmaConfig<BENCH_N>;

template <class Cfg>
struct SharedStorage {
    cute::array_aligned<Element,
                        kMaxResidentWarpgroups * Cfg::kSmemAElements,
                        128> a;
    cute::array_aligned<Element,
                        kMaxResidentWarpgroups * Cfg::kSmemBElements,
                        128> b;
};

template <bool UseRs, class TiledMma, class TensorA, class TensorB, class TensorC>
__device__ __forceinline__ void issue_committed_group(
    TiledMma& tiled_mma,
    TensorA& operand_a,
    TensorB& operand_b,
    TensorC& accum,
    int instructions_per_group) {
    if constexpr (UseRs) {
        cute::warpgroup_fence_operand(operand_a);
    }
    cute::warpgroup_fence_operand(accum);
    cute::warpgroup_arrive();
#pragma unroll 1
    for (int instruction = 0; instruction < instructions_per_group; ++instruction) {
        cute::gemm(tiled_mma, operand_a, operand_b, accum);
    }
    cute::warpgroup_commit_batch();
}

template <bool UseRs, bool Throughput, int IssueDepth>
__global__ __launch_bounds__(256, 1) void wgmma_kernel(
    uint64_t* __restrict__ starts,
    uint64_t* __restrict__ stops,
    float* __restrict__ sink,
    int repeat,
    int instructions_per_group) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)
    static_assert(IssueDepth >= 1 && IssueDepth <= 8);
    if constexpr (!Throughput) {
        static_assert(IssueDepth == 1);
    }

    __shared__ SharedStorage<Config> storage;
    const int warpgroup = threadIdx.x / kThreadsPerWarpgroup;
    const int lane = threadIdx.x % kThreadsPerWarpgroup;

    Tensor sA = make_tensor(
        make_smem_ptr(storage.a.data() + warpgroup * Config::kSmemAElements),
        typename Config::SmemLayoutA{});
    Tensor sB = make_tensor(
        make_smem_ptr(storage.b.data() + warpgroup * Config::kSmemBElements),
        typename Config::SmemLayoutB{});

    for (int linear = lane; linear < Config::kM * Config::kSmemK;
         linear += kThreadsPerWarpgroup) {
        const int row = linear / Config::kSmemK;
        const int col = linear - row * Config::kSmemK;
        const float value = (1.0f + static_cast<float>((row + col + warpgroup) & 7)) /
                            512.0f;
        sA(row, col) = Element(value);
    }
    for (int linear = lane; linear < Config::kN * Config::kSmemK;
         linear += kThreadsPerWarpgroup) {
        const int row = linear / Config::kSmemK;
        const int col = linear - row * Config::kSmemK;
        const float value = (1.0f + static_cast<float>((row * 3 + col) & 7)) /
                            512.0f;
        sB(row, col) = Element(value);
    }
    __syncthreads();

    using TiledMma = std::conditional_t<UseRs, typename Config::TiledMmaRS,
                                       typename Config::TiledMmaSS>;
    TiledMma tiled_mma{};
    auto thr_mma = tiled_mma.get_slice(lane);
    auto operand_a_all = thr_mma.partition_fragment_A(sA);
    auto operand_b_all = thr_mma.partition_fragment_B(sB);
    auto operand_a = operand_a_all(_, _, _0{});
    auto operand_b = operand_b_all(_, _, _0{});
    if constexpr (UseRs) {
        auto partition_a_all = thr_mma.partition_A(sA);
        cute::copy(partition_a_all(_, _, _0{}), operand_a);
    }
    auto accum = partition_fragment_C(
        tiled_mma, Shape<Int<Config::kM>, Int<Config::kN>>{});
    cute::clear(accum);
    tiled_mma.accumulate_ = GMMA::ScaleOut::One;

    uint64_t start = 0;
    uint64_t stop = 0;
    CLK_START(start);
#pragma unroll 1
    for (int iteration = 0; iteration < repeat; ++iteration) {
        if constexpr (Throughput) {
#pragma unroll
            for (int group = 0; group < IssueDepth; ++group) {
                issue_committed_group<UseRs>(tiled_mma, operand_a, operand_b, accum,
                                             instructions_per_group);
            }
            cute::warpgroup_wait<0>();
            cute::warpgroup_fence_operand(accum);
            if constexpr (UseRs) {
                cute::warpgroup_fence_operand(operand_a);
            }
        } else {
            issue_committed_group<UseRs>(tiled_mma, operand_a, operand_b, accum, 1);
            cute::warpgroup_wait<0>();
            cute::warpgroup_fence_operand(accum);
            if constexpr (UseRs) {
                cute::warpgroup_fence_operand(operand_a);
            }
        }
    }
    CLK_STOP(stop);

    starts[threadIdx.x] = start;
    stops[threadIdx.x] = stop;
#pragma unroll
    for (int i = 0; i < Config::kAccumulatorElements; ++i) {
        sink[threadIdx.x * Config::kAccumulatorElements + i] = accum(i);
    }
#else
    if (threadIdx.x == 0) {
        starts[0] = stops[0] = 0;
        sink[0] = 0.0f;
    }
#endif
}

struct RunOptions {
    int repeat;
    int warmup;
    int samples;
    int resident_warpgroups;
    int instructions_per_group;
};

template <bool UseRs, bool Throughput, int IssueDepth>
void run_case(const RunOptions& options,
              const microbench::Sm90Device& device,
              DeviceBuffer<uint64_t>& starts,
              DeviceBuffer<uint64_t>& stops,
              DeviceBuffer<float>& sink) {
    const int threads = options.resident_warpgroups * kThreadsPerWarpgroup;
    auto measure_once = [&]() -> double {
        wgmma_kernel<UseRs, Throughput, IssueDepth><<<1, threads>>>(
            starts.data(), stops.data(), sink.data(), options.repeat,
            Throughput ? options.instructions_per_group : 1);
        microbench::throw_if_cuda_error(cudaGetLastError(), "wgmma_kernel launch");
        microbench::throw_if_cuda_error(cudaDeviceSynchronize(),
                                         "wgmma_kernel synchronize");
        const auto host_starts = starts.copy_to_host();
        const auto host_stops = stops.copy_to_host();
        return static_cast<double>(microbench::reduce_cycles(
            host_starts.data(), host_stops.data(), host_starts.size()));
    };

    const auto series = microbench::run_samples(options.warmup, options.samples,
                                                 measure_once);
    const auto summary = series.summary();
    const auto host_sink = sink.copy_to_host();
    long double checksum = 0.0;
    for (float value : host_sink) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("WGMMA sink contains a non-finite value");
        }
        checksum += value;
    }
    if (checksum == 0.0) {
        throw std::runtime_error("WGMMA sink is zero; target work may have been removed");
    }

    constexpr int depth = Throughput ? IssueDepth : 1;
    const int instructions_per_group =
        Throughput ? options.instructions_per_group : 1;
    const double committed_groups_per_warpgroup =
        static_cast<double>(options.repeat) * depth;
    const double instructions_per_warpgroup =
        committed_groups_per_warpgroup * instructions_per_group;
    const double instructions_per_sm =
        instructions_per_warpgroup * options.resident_warpgroups;
    const double flop_per_instruction =
        2.0 * Config::kM * Config::kN * Config::kK;

    microbench::JsonLine json;
    json.add("benchmark", Config::kBenchmark)
        .add("gpu", device.properties.name)
        .add("operand_mode", UseRs ? "rs" : "ss")
        .add("measurement", Throughput ? "throughput_groups" : "latency_full")
        .add("m", Config::kM)
        .add("n", Config::kN)
        .add("k", Config::kK)
        .add("dtype", "bf16")
        .add("accumulator_dtype", "f32")
        .add("major_a", "K")
        .add("major_b", Config::kMajorBName)
        .add("shared_layout", "SW128")
        .add("threads", threads)
        .add("resident_wg", options.resident_warpgroups)
        .add("issue_depth", depth)
        .add("instructions_per_group", instructions_per_group)
        .add("repeat", options.repeat)
        .add("committed_groups_per_wg", committed_groups_per_warpgroup)
        .add("instructions_per_wg", instructions_per_warpgroup)
        .add("flop_per_instruction", flop_per_instruction)
        .add("checksum", static_cast<double>(checksum));
    microbench::add_measurement_summary(json, summary);
    json.add("cycle_per_committed_group_wg",
             summary.median / committed_groups_per_warpgroup)
        .add("cycle_per_instruction_wg",
             summary.median / instructions_per_warpgroup)
        .add("committed_group_per_clk_sm",
             committed_groups_per_warpgroup * options.resident_warpgroups /
                 summary.median)
        .add("instruction_per_clk_sm", instructions_per_sm / summary.median)
        .add("flop_per_clk_sm",
             instructions_per_sm * flop_per_instruction / summary.median)
        .print();
}

template <bool UseRs, bool Throughput>
void dispatch_depth(int issue_depth,
                    const RunOptions& options,
                    const microbench::Sm90Device& device,
                    DeviceBuffer<uint64_t>& starts,
                    DeviceBuffer<uint64_t>& stops,
                    DeviceBuffer<float>& sink) {
    if constexpr (!Throughput) {
        run_case<UseRs, false, 1>(options, device, starts, stops, sink);
    } else {
        switch (issue_depth) {
            case 1: run_case<UseRs, true, 1>(options, device, starts, stops, sink); break;
            case 2: run_case<UseRs, true, 2>(options, device, starts, stops, sink); break;
            case 3: run_case<UseRs, true, 3>(options, device, starts, stops, sink); break;
            case 4: run_case<UseRs, true, 4>(options, device, starts, stops, sink); break;
            case 5: run_case<UseRs, true, 5>(options, device, starts, stops, sink); break;
            case 6: run_case<UseRs, true, 6>(options, device, starts, stops, sink); break;
            case 7: run_case<UseRs, true, 7>(options, device, starts, stops, sink); break;
            case 8: run_case<UseRs, true, 8>(options, device, starts, stops, sink); break;
            default: throw std::invalid_argument("issue_depth must be in [1, 8]");
        }
    }
}

bool selected(const std::string& requested, const char* value) {
    return requested == "both" || requested == value;
}

}  // namespace wgmma_bench

using namespace wgmma_bench;

int main(int argc, char** argv) {
    try {
        const CliArgs args(argc, argv);
        const std::string operand = args.get_string("operand", "both");
        const std::string measurement = args.get_string("measurement", "both");
        if (operand != "rs" && operand != "ss" && operand != "both") {
            throw std::invalid_argument("--operand must be rs, ss, or both");
        }
        if (measurement != "latency" && measurement != "throughput" &&
            measurement != "both") {
            throw std::invalid_argument(
                "--measurement must be latency, throughput, or both");
        }

        RunOptions options{
            args.get_int("repeat", 256, 1, 1 << 20),
            args.get_int("warmup", 5, 0, 1000),
            args.get_int("samples", 20, 1, 10000),
            args.get_int("resident-wg", 1, 1, 2),
            args.get_int("instructions-per-group", 1, 1, 36),
        };
        const int issue_depth = args.get_int("issue-depth", 4, 1, 8);
        const auto device = microbench::require_sm90(args.get_int("device", 0));
        const int threads = options.resident_warpgroups * kThreadsPerWarpgroup;
        DeviceBuffer<uint64_t> starts(threads);
        DeviceBuffer<uint64_t> stops(threads);
        DeviceBuffer<float> sink(
            static_cast<std::size_t>(threads) * Config::kAccumulatorElements);

        if (selected(operand, "ss")) {
            if (selected(measurement, "latency")) {
                dispatch_depth<false, false>(1, options, device, starts, stops, sink);
            }
            if (selected(measurement, "throughput")) {
                dispatch_depth<false, true>(issue_depth, options, device, starts, stops,
                                             sink);
            }
        }
        if (selected(operand, "rs")) {
            if (selected(measurement, "latency")) {
                dispatch_depth<true, false>(1, options, device, starts, stops, sink);
            }
            if (selected(measurement, "throughput")) {
                dispatch_depth<true, true>(issue_depth, options, device, starts, stops,
                                            sink);
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "WGMMA benchmark error: " << error.what() << '\n';
        return 1;
    }
}
