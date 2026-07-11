// Exact SM90a SM90_U32x4_STSM_N register-to-shared tile store benchmark.

#include <cstdint>
#include <exception>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>

#include "benchmark_utils.hpp"
#include "clock.cuh"

#ifndef BENCH_M
#define BENCH_M 64
#endif
#ifndef BENCH_N
#define BENCH_N 64
#endif
#ifndef BENCH_ELEMENT_BITS
#define BENCH_ELEMENT_BITS 16
#endif
#ifndef BENCH_X
#define BENCH_X 4
#endif

static_assert(BENCH_M == 64, "STMatrix benchmark requires M=64");
static_assert(BENCH_N == 64 || BENCH_N == 256,
              "supported STMatrix N sizes are 64 and 256");
static_assert(BENCH_ELEMENT_BITS == 16, "STMatrix benchmark stores B16 values");
static_assert(BENCH_X == 4, "this family uses SM90_U32x4_STSM_N");

namespace stmatrix_bench {

using Element = cutlass::bfloat16_t;
using microbench::CliArgs;
using microbench::DeviceBuffer;
using namespace cute;

constexpr int kThreads = 128;
constexpr int kWarps = 4;
constexpr uint16_t kExpectedBf16One = 0x3f80;

template <int N, GMMA::Major BMajor>
struct StmatrixConfigBase {
    static constexpr int kM = 64;
    static constexpr int kN = N;
    static constexpr int kK = 16;
    static constexpr int kInstructionsPerWarp = kN / 16;
    static constexpr int kWarpInstructionsPerTile =
        kWarps * kInstructionsPerWarp;
    static constexpr int kUsefulBytes = kM * kN * sizeof(Element);

    using MnkShape = Shape<Int<kM>, Int<kN>, Int<kK>>;
    using TiledMma = decltype(make_tiled_mma(
        GMMA::rs_op_selector<Element, Element, float, MnkShape,
                             GMMA::Major::K, BMajor>(),
        Layout<Shape<_1, _1, _1>>{}));
    using SmemLayout = decltype(tile_to_shape(
        GMMA::Layout_K_SW128_Atom<Element>{}, Shape<Int<kM>, Int<kN>>{}));
    static constexpr int kSmemElements = cosize_v<SmemLayout>;
};

template <int N>
struct StmatrixConfig;

template <>
struct StmatrixConfig<64> : StmatrixConfigBase<64, GMMA::Major::K> {
    static constexpr const char* kBenchmark = "stmatrix/m64n64_b16_x4_sm90";
};

template <>
struct StmatrixConfig<256> : StmatrixConfigBase<256, GMMA::Major::MN> {
    static constexpr const char* kBenchmark = "stmatrix/m64n256_b16_x4_sm90";
};

using Config = StmatrixConfig<BENCH_N>;

struct SharedStorage {
    cute::array_aligned<Element, Config::kSmemElements, 128> output;
};

template <bool TimedFence>
__global__ __launch_bounds__(kThreads, 1) void stmatrix_kernel(
    uint64_t* __restrict__ starts,
    uint64_t* __restrict__ stops,
    uint16_t* __restrict__ sink,
    int repeat) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)
    __shared__ SharedStorage storage;
    Tensor sOutput = make_tensor(make_smem_ptr(storage.output.data()),
                                 typename Config::SmemLayout{});

    typename Config::TiledMma tiled_mma{};
    auto rAccumulator = partition_fragment_C(
        tiled_mma, Shape<Int<Config::kM>, Int<Config::kN>>{});
    auto rOutput = make_tensor_like<Element>(rAccumulator);
#pragma unroll
    for (int i = 0; i < size(rOutput); ++i) {
        rOutput(i) = Element(1.0f);
    }

    auto r2s_copy = make_tiled_copy_C(
        Copy_Atom<SM90_U32x4_STSM_N, Element>{}, tiled_mma);
    auto thread_copy = r2s_copy.get_slice(threadIdx.x);
    auto source = thread_copy.retile_S(rOutput);
    auto destination = thread_copy.partition_D(sOutput);
    __syncthreads();

    uint64_t start = 0;
    uint64_t stop = 0;
    CLK_START(start);
#pragma unroll 1
    for (int iteration = 0; iteration < repeat; ++iteration) {
        cute::copy(r2s_copy, source, destination);
        if constexpr (TimedFence) {
            cutlass::arch::fence_view_async_shared();
        }
    }
    CLK_STOP(stop);

    // Validation always establishes proxy visibility outside the timed region.
    if constexpr (!TimedFence) {
        cutlass::arch::fence_view_async_shared();
    }
    __syncthreads();
    Element* typed_sink = reinterpret_cast<Element*>(sink);
    for (int linear = threadIdx.x; linear < Config::kM * Config::kN;
         linear += kThreads) {
        const int row = linear / Config::kN;
        const int col = linear - row * Config::kN;
        typed_sink[linear] = sOutput(row, col);
    }
    starts[threadIdx.x] = start;
    stops[threadIdx.x] = stop;
#else
    if (threadIdx.x == 0) {
        starts[0] = stops[0] = 0;
        sink[0] = 0;
    }
#endif
}

template <bool TimedFence>
void run_case(int repeat,
              int warmup,
              int samples,
              const microbench::Sm90Device& device,
              DeviceBuffer<uint64_t>& starts,
              DeviceBuffer<uint64_t>& stops,
              DeviceBuffer<uint16_t>& sink) {
    auto measure_once = [&]() -> double {
        stmatrix_kernel<TimedFence><<<1, kThreads>>>(
            starts.data(), stops.data(), sink.data(), repeat);
        microbench::throw_if_cuda_error(cudaGetLastError(),
                                         "stmatrix_kernel launch");
        microbench::throw_if_cuda_error(cudaDeviceSynchronize(),
                                         "stmatrix_kernel synchronize");
        const auto host_starts = starts.copy_to_host();
        const auto host_stops = stops.copy_to_host();
        return static_cast<double>(microbench::reduce_cycles(
            host_starts.data(), host_stops.data(), host_starts.size()));
    };

    const auto series = microbench::run_samples(warmup, samples, measure_once);
    const auto summary = series.summary();
    const auto host_sink = sink.copy_to_host();
    std::size_t mismatches = 0;
    uint64_t checksum = 0;
    for (uint16_t bits : host_sink) {
        checksum += bits;
        mismatches += bits != kExpectedBf16One;
    }
    if (mismatches != 0) {
        throw std::runtime_error("shared sink validation failed for " +
                                 std::to_string(mismatches) + " elements");
    }

    const double tiles = static_cast<double>(repeat);
    const double warp_instructions =
        tiles * Config::kWarpInstructionsPerTile;
    microbench::JsonLine json;
    json.add("benchmark", Config::kBenchmark)
        .add("gpu", device.properties.name)
        .add("m", Config::kM)
        .add("n", Config::kN)
        .add("element_bits", BENCH_ELEMENT_BITS)
        .add("x", BENCH_X)
        .add("shared_layout", "K_SW128")
        .add("threads", kThreads)
        .add("warps", kWarps)
        .add("repeat", repeat)
        .add("fence_timed", TimedFence)
        .add("validation_fence", true)
        .add("stmatrix_per_warp_per_tile", Config::kInstructionsPerWarp)
        .add("warp_instructions_per_tile", Config::kWarpInstructionsPerTile)
        .add("useful_bytes_per_tile", Config::kUsefulBytes)
        .add("checksum", checksum);
    microbench::add_measurement_summary(json, summary);
    json.add("cycle_per_tile", summary.median / tiles)
        .add("cycle_per_warp_instruction",
             summary.median /
                 (tiles * static_cast<double>(Config::kInstructionsPerWarp)))
        .add("warp_instruction_per_clk_sm", warp_instructions / summary.median)
        .add("byte_per_clk_sm",
             tiles * static_cast<double>(Config::kUsefulBytes) / summary.median)
        .print();
}

}  // namespace stmatrix_bench

using namespace stmatrix_bench;

int main(int argc, char** argv) {
    try {
        const CliArgs args(argc, argv);
        const int repeat = args.get_int("repeat", 512, 1, 1 << 24);
        const int warmup = args.get_int("warmup", 5, 0, 1000);
        const int samples = args.get_int("samples", 20, 1, 10000);
        const bool fence = args.get_bool("fence", true);
        const auto device = microbench::require_sm90(args.get_int("device", 0));

        DeviceBuffer<uint64_t> starts(kThreads);
        DeviceBuffer<uint64_t> stops(kThreads);
        DeviceBuffer<uint16_t> sink(Config::kM * Config::kN);
        if (fence) {
            run_case<true>(repeat, warmup, samples, device, starts, stops, sink);
        } else {
            run_case<false>(repeat, warmup, samples, device, starts, stops, sink);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "STMatrix benchmark error: " << error.what() << '\n';
        return 1;
    }
}
