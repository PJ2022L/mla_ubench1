// SM90 rank-2/3/4 TMA load benchmark for logical BF16 attention tiles.

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cutlass/arch/barrier.h>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include "benchmark_utils.hpp"
#include "clock.cuh"
#include "tma_util.cuh"

#ifndef BENCH_M
#define BENCH_M 64
#endif
#ifndef BENCH_N
#define BENCH_N 64
#endif

namespace {

constexpr int kThreads = 32;
constexpr int kInitThreads = 256;
constexpr int kInitBlocks = 256;
constexpr int kTransactionCols = 64;
constexpr int kTransactionsPerLogicalTile = BENCH_N / kTransactionCols;
constexpr int kTransactionBytes = BENCH_M * kTransactionCols * 2;
constexpr int kTransactionElements = BENCH_M * kTransactionCols;
constexpr int kLogicalTileBytes = BENCH_M * BENCH_N * 2;
constexpr int kLogicalTileElements = BENCH_M * BENCH_N;
constexpr int kMaxDepth = 8;

static_assert(BENCH_M == 64);
static_assert(BENCH_N % kTransactionCols == 0);

__host__ __device__ inline uint16_t input_pattern(uint64_t index) {
    uint64_t value = index + 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    value ^= value >> 31;
    return static_cast<uint16_t>(value);
}

__global__ void initialize_input(uint16_t* input, std::size_t elements) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (; index < elements; index += stride) {
        input[index] = input_pattern(static_cast<uint64_t>(index));
    }
}

template <int Rank>
__device__ __forceinline__ void issue_tma_load(
        const CUtensorMap* tensor_map,
        uint64_t* barrier,
        unsigned char* destination,
        int transaction,
        int logical_tile) {
    const uint64_t cache_hint =
        static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_FIRST);
    if constexpr (Rank == 2) {
        cute::SM90_TMA_LOAD_2D::copy(
            tensor_map, barrier, cache_hint, destination,
            transaction * kTransactionCols, logical_tile * BENCH_M);
    } else if constexpr (Rank == 3) {
        cute::SM90_TMA_LOAD_3D::copy(
            tensor_map, barrier, cache_hint, destination,
            transaction * kTransactionCols, 0, logical_tile);
    } else {
        static_assert(Rank == 4);
        cute::SM90_TMA_LOAD_4D::copy(
            tensor_map, barrier, cache_hint, destination,
            transaction * kTransactionCols, 0, logical_tile, 0);
    }
}

template <int Rank>
__global__ void tma_load_kernel(
        __grid_constant__ const CUtensorMap tensor_map,
        uint64_t* __restrict__ starts,
        uint64_t* __restrict__ stops,
        int repeat,
        int depth,
        int working_tiles) {
    extern __shared__ __align__(128) unsigned char shared_bytes[];
    __shared__ alignas(8) uint64_t barriers[kMaxDepth];

    if (threadIdx.x == 0) {
        for (int stage = 0; stage < depth; ++stage) {
            cutlass::arch::ClusterTransactionBarrier::init(&barriers[stage], 1);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    uint32_t phase_bits = 0;
    uint64_t start = 0;
    uint64_t stop = 0;
    CLK_START(start);

    if (threadIdx.x == 0) {
        int logical_tile = 0;
        for (int base = 0; base < repeat; base += depth) {
            const int active = min(depth, repeat - base);
            for (int stage = 0; stage < active; ++stage) {
                unsigned char* stage_ptr =
                    shared_bytes + static_cast<std::size_t>(stage) * kLogicalTileBytes;
                for (int transaction = 0;
                     transaction < kTransactionsPerLogicalTile;
                     ++transaction) {
                    issue_tma_load<Rank>(
                        &tensor_map,
                        &barriers[stage],
                        stage_ptr + transaction * kTransactionBytes,
                        transaction,
                        logical_tile);
                }
                if (++logical_tile == working_tiles) {
                    logical_tile = 0;
                }
                cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                    &barriers[stage], kLogicalTileBytes);
            }
            for (int stage = 0; stage < active; ++stage) {
                cutlass::arch::ClusterTransactionBarrier::wait(
                    &barriers[stage], (phase_bits >> stage) & 1u);
                phase_bits ^= 1u << stage;
            }
        }
    }
    CLK_STOP(stop);

    starts[threadIdx.x] = start;
    stops[threadIdx.x] = stop;
}

template <int Rank>
__global__ void validate_tma_load_kernel(
        __grid_constant__ const CUtensorMap tensor_map,
        uint64_t* __restrict__ checksum,
        int working_tiles) {
    extern __shared__ __align__(128) unsigned char shared_bytes[];
    __shared__ alignas(8) uint64_t barrier;

    if (threadIdx.x == 0) {
        cutlass::arch::ClusterTransactionBarrier::init(&barrier, 1);
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        uint64_t result = 0;
        uint32_t phase = 0;
        const int validation_tiles[2] = {0, working_tiles - 1};
        for (int sample = 0; sample < 2; ++sample) {
            for (int transaction = 0;
                 transaction < kTransactionsPerLogicalTile;
                 ++transaction) {
                issue_tma_load<Rank>(
                    &tensor_map,
                    &barrier,
                    shared_bytes + transaction * kTransactionBytes,
                    transaction,
                    validation_tiles[sample]);
            }
            cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                &barrier, kLogicalTileBytes);
            cutlass::arch::ClusterTransactionBarrier::wait(&barrier, phase);
            phase ^= 1u;

            const uint16_t* values =
                reinterpret_cast<const uint16_t*>(shared_bytes);
            for (int transaction = 0;
                 transaction < kTransactionsPerLogicalTile;
                 ++transaction) {
                const uint64_t weight = static_cast<uint64_t>(
                    1 + sample * kTransactionsPerLogicalTile + transaction);
                for (int element = 0; element < kTransactionElements; ++element) {
                    result += weight * values[
                        transaction * kTransactionElements + element];
                }
            }
        }
        checksum[0] = result;
    }
}

CUtensorMap make_tensor_map(void* input, int working_tiles, int rank) {
    if (rank == 2) {
        microbench::TensorMapSpec<2> spec;
        spec.data_type = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
        spec.global_address = input;
        spec.global_dims = {BENCH_N,
                            static_cast<uint64_t>(working_tiles) * BENCH_M};
        spec.global_strides = {static_cast<uint64_t>(BENCH_N) * 2};
        spec.box_dims = {kTransactionCols, BENCH_M};
        spec.swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
        return microbench::encode_tensor_map_2d(spec);
    }
    if (rank == 3) {
        microbench::TensorMapSpec<3> spec;
        spec.data_type = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
        spec.global_address = input;
        spec.global_dims = {BENCH_N, BENCH_M,
                            static_cast<uint64_t>(working_tiles)};
        spec.global_strides = {
            static_cast<uint64_t>(BENCH_N) * 2,
            static_cast<uint64_t>(BENCH_N) * BENCH_M * 2};
        spec.box_dims = {kTransactionCols, BENCH_M, 1};
        spec.swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
        return microbench::encode_tensor_map_3d(spec);
    }

    microbench::TensorMapSpec<4> spec;
    spec.data_type = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    spec.global_address = input;
    spec.global_dims = {BENCH_N, BENCH_M,
                        static_cast<uint64_t>(working_tiles), 1};
    spec.global_strides = {
        static_cast<uint64_t>(BENCH_N) * 2,
        static_cast<uint64_t>(BENCH_N) * BENCH_M * 2,
        static_cast<uint64_t>(working_tiles) * BENCH_N * BENCH_M * 2};
    spec.box_dims = {kTransactionCols, BENCH_M, 1, 1};
    spec.swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
    return microbench::encode_tensor_map_4d(spec);
}

void set_dynamic_shared_memory(int rank, int shared_bytes) {
    cudaError_t benchmark_status = cudaSuccess;
    cudaError_t validation_status = cudaSuccess;
    if (rank == 2) {
        benchmark_status = cudaFuncSetAttribute(
            tma_load_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_bytes);
        validation_status = cudaFuncSetAttribute(
            validate_tma_load_kernel<2>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kLogicalTileBytes);
    } else if (rank == 3) {
        benchmark_status = cudaFuncSetAttribute(
            tma_load_kernel<3>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_bytes);
        validation_status = cudaFuncSetAttribute(
            validate_tma_load_kernel<3>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kLogicalTileBytes);
    } else {
        benchmark_status = cudaFuncSetAttribute(
            tma_load_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_bytes);
        validation_status = cudaFuncSetAttribute(
            validate_tma_load_kernel<4>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kLogicalTileBytes);
    }
    microbench::throw_if_cuda_error(benchmark_status,
                                    "set tma_load dynamic shared memory");
    microbench::throw_if_cuda_error(validation_status,
                                    "set tma_load validation shared memory");
}

void launch_tma_load(int rank,
                     const CUtensorMap& tensor_map,
                     uint64_t* starts,
                     uint64_t* stops,
                     int repeat,
                     int depth,
                     int working_tiles,
                     std::size_t shared_bytes) {
    if (rank == 2) {
        tma_load_kernel<2><<<1, kThreads, shared_bytes>>>(
            tensor_map, starts, stops, repeat, depth, working_tiles);
    } else if (rank == 3) {
        tma_load_kernel<3><<<1, kThreads, shared_bytes>>>(
            tensor_map, starts, stops, repeat, depth, working_tiles);
    } else {
        tma_load_kernel<4><<<1, kThreads, shared_bytes>>>(
            tensor_map, starts, stops, repeat, depth, working_tiles);
    }
}

void launch_tma_validation(int rank,
                           const CUtensorMap& tensor_map,
                           uint64_t* checksum,
                           int working_tiles) {
    if (rank == 2) {
        validate_tma_load_kernel<2><<<1, kThreads, kLogicalTileBytes>>>(
            tensor_map, checksum, working_tiles);
    } else if (rank == 3) {
        validate_tma_load_kernel<3><<<1, kThreads, kLogicalTileBytes>>>(
            tensor_map, checksum, working_tiles);
    } else {
        validate_tma_load_kernel<4><<<1, kThreads, kLogicalTileBytes>>>(
            tensor_map, checksum, working_tiles);
    }
}

uint64_t expected_validation_checksum(int working_tiles) {
    uint64_t result = 0;
    const int validation_tiles[2] = {0, working_tiles - 1};
    for (int sample = 0; sample < 2; ++sample) {
        const std::size_t tile_base =
            static_cast<std::size_t>(validation_tiles[sample]) *
            kLogicalTileElements;
        for (int transaction = 0;
             transaction < kTransactionsPerLogicalTile;
             ++transaction) {
            const uint64_t weight = static_cast<uint64_t>(
                1 + sample * kTransactionsPerLogicalTile + transaction);
            for (int row = 0; row < BENCH_M; ++row) {
                for (int column = 0; column < kTransactionCols; ++column) {
                    const std::size_t index =
                        tile_base + static_cast<std::size_t>(row) * BENCH_N +
                        transaction * kTransactionCols + column;
                    result += weight * input_pattern(
                        static_cast<uint64_t>(index));
                }
            }
        }
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const microbench::CliArgs args(argc, argv);
        const int repeat = args.get_int("repeat", BENCH_N == 64 ? 256 : 64,
                                        1, 1 << 24);
        const int depth = args.get_int("depth", 1, 1, kMaxDepth);
        const int rank = args.get_int("rank", 4, 2, 4);
        const int working_tiles = args.get_int("working-set-tiles", 64,
                                                1, 1 << 20);
        const int warmup = args.get_int("warmup", 5, 0, 1000);
        const int samples = args.get_int("samples", 20, 1, 10000);
        const auto device = microbench::require_sm90(args.get_int("device", 0));

        const std::size_t shared_bytes =
            static_cast<std::size_t>(depth) * kLogicalTileBytes;
        if (shared_bytes > device.properties.sharedMemPerBlockOptin) {
            throw std::invalid_argument("requested TMA depth needs " +
                std::to_string(shared_bytes) +
                " shared bytes, exceeding sharedMemPerBlockOptin=" +
                std::to_string(device.properties.sharedMemPerBlockOptin));
        }

        const std::size_t input_elements = static_cast<std::size_t>(working_tiles) *
                                           BENCH_M * BENCH_N;
        microbench::DeviceBuffer<uint16_t> input(input_elements);
        microbench::DeviceBuffer<uint64_t> starts(kThreads);
        microbench::DeviceBuffer<uint64_t> stops(kThreads);
        microbench::DeviceBuffer<uint64_t> sink(1);
        initialize_input<<<kInitBlocks, kInitThreads>>>(input.data(),
                                                        input_elements);
        microbench::throw_if_cuda_error(cudaGetLastError(),
                                         "initialize_input launch");
        microbench::throw_if_cuda_error(cudaDeviceSynchronize(),
                                         "initialize_input synchronize");

        const CUtensorMap tensor_map =
            make_tensor_map(input.data(), working_tiles, rank);
        set_dynamic_shared_memory(rank, static_cast<int>(shared_bytes));

        auto measure_once = [&]() -> double {
            launch_tma_load(rank, tensor_map, starts.data(), stops.data(),
                            repeat, depth, working_tiles, shared_bytes);
            microbench::throw_if_cuda_error(cudaGetLastError(),
                                             "tma_load_kernel launch");
            microbench::throw_if_cuda_error(cudaDeviceSynchronize(),
                                             "tma_load_kernel synchronize");
            const auto host_starts = starts.copy_to_host();
            const auto host_stops = stops.copy_to_host();
            return static_cast<double>(microbench::reduce_cycles(
                host_starts.data(), host_stops.data(), host_starts.size()));
        };

        const auto series = microbench::run_samples(warmup, samples, measure_once);
        const auto summary = series.summary();
        launch_tma_validation(rank, tensor_map, sink.data(), working_tiles);
        microbench::throw_if_cuda_error(cudaGetLastError(),
                                         "validate_tma_load_kernel launch");
        microbench::throw_if_cuda_error(cudaDeviceSynchronize(),
                                         "validate_tma_load_kernel synchronize");
        const uint64_t checksum = sink.copy_to_host().front();
        const uint64_t expected_checksum =
            expected_validation_checksum(working_tiles);
        if (checksum != expected_checksum) {
            throw std::runtime_error(
                "TMA load checksum mismatch: expected " +
                std::to_string(expected_checksum) + ", got " +
                std::to_string(checksum));
        }
        const double requested_bytes =
            static_cast<double>(repeat) * kLogicalTileBytes;
        const double transactions =
            static_cast<double>(repeat) * kTransactionsPerLogicalTile;

        microbench::JsonLine json;
        json.add("benchmark", "tma_load/tile64x" + std::to_string(BENCH_N) +
                                  "_bf16_sm90")
            .add("gpu", device.properties.name)
            .add("m", BENCH_M)
            .add("n", BENCH_N)
            .add("rank", rank)
            .add("repeat", repeat)
            .add("depth", depth)
            .add("working_set_tiles", working_tiles)
            .add("transactions_per_tile", kTransactionsPerLogicalTile)
            .add("logical_tile_bytes", kLogicalTileBytes)
            .add("checksum", checksum)
            .add("expected_checksum", expected_checksum);
        microbench::add_measurement_summary(json, summary);
        json.add("cycle_per_tile", summary.median / repeat)
            .add("transaction_per_clk_sm", transactions / summary.median)
            .add("requested_byte_per_clk_sm", requested_bytes / summary.median)
            .print();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tma_load benchmark error: " << error.what() << '\n';
        return 1;
    }
}
