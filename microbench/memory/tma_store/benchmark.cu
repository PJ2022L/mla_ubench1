// SM90 TMA-only BF16 shared-to-global store benchmark.

#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cutlass/arch/barrier.h>
#include <cute/arch/copy_sm90_tma.hpp>

#include "benchmark_utils.hpp"
#include "clock.cuh"
#include "tma_util.cuh"

#ifndef BENCH_M
#define BENCH_M 64
#endif
#ifndef BENCH_N
#define BENCH_N 512
#endif
#ifndef BENCH_TENSOR_RANK
#define BENCH_TENSOR_RANK 2
#endif

namespace {

constexpr int kThreads = 32;
constexpr int kFullOutputColumns = 512;
constexpr int kTransactionColumns = 64;
constexpr int kTransactionBytes = BENCH_M * kTransactionColumns * 2;
constexpr int kTransactionElements = BENCH_M * kTransactionColumns;
constexpr int kTransactionsPerTile =
    (BENCH_TENSOR_RANK == 2 || BENCH_TENSOR_RANK == 4)
        ? BENCH_N / kTransactionColumns
        : 1;
constexpr int kLogicalTileBytes = BENCH_M * BENCH_N * 2;
constexpr int kLogicalTileElements = BENCH_M * BENCH_N;
constexpr int kMaxDepth = 8;

static_assert(BENCH_M == 64);
static_assert(BENCH_TENSOR_RANK >= 2 && BENCH_TENSOR_RANK <= 5);
static_assert(BENCH_N == 64 || BENCH_N == 512);

__host__ __device__ inline uint16_t store_pattern(uint64_t index) {
    uint64_t value = index + 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    value ^= value >> 31;
    return static_cast<uint16_t>(value);
}

__device__ __forceinline__ void issue_store(
        const CUtensorMap* tensor_map,
        const unsigned char* source,
        int logical_tile,
        int iteration) {
    if constexpr (BENCH_TENSOR_RANK == 2) {
#pragma unroll
        for (int transaction = 0; transaction < kTransactionsPerTile;
             ++transaction) {
            cute::SM90_TMA_STORE_2D::copy(
                tensor_map,
                source + transaction * kTransactionBytes,
                transaction * kTransactionColumns,
                logical_tile * BENCH_M);
        }
    } else if constexpr (BENCH_TENSOR_RANK == 3) {
        const int output_tile = iteration & 7;
        cute::SM90_TMA_STORE_3D::copy(
            tensor_map, source, output_tile * kTransactionColumns, 0,
            logical_tile);
    } else if constexpr (BENCH_TENSOR_RANK == 4) {
#pragma unroll
        for (int transaction = 0; transaction < kTransactionsPerTile;
             ++transaction) {
            cute::SM90_TMA_STORE_4D::copy(
                tensor_map,
                source + transaction * kTransactionBytes,
                transaction * kTransactionColumns,
                0,
                logical_tile,
                0);
        }
    } else {
        cute::SM90_TMA_STORE_5D::copy(
            tensor_map, source, 0, 0, 0, logical_tile, 0);
    }
}

__global__ void tma_store_kernel(
        __grid_constant__ const CUtensorMap tensor_map,
        uint64_t* __restrict__ starts,
        uint64_t* __restrict__ stops,
        int repeat,
        int depth,
        int working_tiles) {
    extern __shared__ __align__(128) unsigned char shared_bytes[];
    uint16_t* shared_values = reinterpret_cast<uint16_t*>(shared_bytes);
    for (std::size_t element = threadIdx.x;
         element < static_cast<std::size_t>(depth) * kLogicalTileElements;
         element += blockDim.x) {
        shared_values[element] = store_pattern(static_cast<uint64_t>(element));
    }
    __syncthreads();
    cutlass::arch::fence_view_async_shared();
    __syncthreads();

    uint64_t start = 0;
    uint64_t stop = 0;
    CLK_START(start);
    if (threadIdx.x == 0) {
        int logical_tile = 0;
        for (int base = 0; base < repeat; base += depth) {
            const int active = min(depth, repeat - base);
            for (int stage = 0; stage < active; ++stage) {
                const int iteration = base + stage;
                issue_store(
                    &tensor_map,
                    shared_bytes + static_cast<std::size_t>(stage) *
                                       kLogicalTileBytes,
                    logical_tile,
                    iteration);
                if (++logical_tile == working_tiles) {
                    logical_tile = 0;
                }
                cute::tma_store_arrive();
            }
            cute::tma_store_wait<0>();
        }
    }
    CLK_STOP(stop);

    starts[threadIdx.x] = start;
    stops[threadIdx.x] = stop;
}

__global__ void validate_tma_store_kernel(
        __grid_constant__ const CUtensorMap tensor_map,
        int working_tiles) {
    extern __shared__ __align__(128) unsigned char shared_bytes[];
    uint16_t* shared_values = reinterpret_cast<uint16_t*>(shared_bytes);
    for (int element = threadIdx.x;
         element < kLogicalTileElements;
         element += blockDim.x) {
        shared_values[element] = store_pattern(static_cast<uint64_t>(element));
    }
    __syncthreads();
    cutlass::arch::fence_view_async_shared();
    __syncthreads();

    if (threadIdx.x == 0) {
        constexpr bool kAlwaysUseTwoSamples = BENCH_TENSOR_RANK == 3;
        const int samples = kAlwaysUseTwoSamples || working_tiles > 1 ? 2 : 1;
        for (int sample = 0; sample < samples; ++sample) {
            const int logical_tile = sample == 0 ? 0 : working_tiles - 1;
            const int iteration =
                BENCH_TENSOR_RANK == 3 && sample == 1 ? 7 : 0;
            issue_store(&tensor_map, shared_bytes, logical_tile, iteration);
            cute::tma_store_arrive();
            cute::tma_store_wait<0>();
        }
    }
}

struct ValidationResult {
    uint64_t checksum = 0;
    uint64_t expected_checksum = 0;
    bool untouched_zero = true;
};

uint64_t source_pattern_sum(int first_element, int element_count) {
    uint64_t result = 0;
    for (int element = 0; element < element_count; ++element) {
        result += store_pattern(
            static_cast<uint64_t>(first_element + element));
    }
    return result;
}

ValidationResult validate_output(const std::vector<uint16_t>& output,
                                 int working_tiles) {
    ValidationResult result;
    constexpr bool kAlwaysUseTwoSamples = BENCH_TENSOR_RANK == 3;
    const int samples = kAlwaysUseTwoSamples || working_tiles > 1 ? 2 : 1;

    if constexpr (BENCH_TENSOR_RANK == 2 || BENCH_TENSOR_RANK == 4) {
        for (int sample = 0; sample < samples; ++sample) {
            const int logical_tile = sample == 0 ? 0 : working_tiles - 1;
            const std::size_t tile_base =
                static_cast<std::size_t>(logical_tile) * BENCH_M *
                kFullOutputColumns;
            for (int transaction = 0;
                 transaction < kTransactionsPerTile;
                 ++transaction) {
                const uint64_t weight = static_cast<uint64_t>(
                    1 + sample * kTransactionsPerTile + transaction);
                result.expected_checksum += weight * source_pattern_sum(
                    transaction * kTransactionElements,
                    kTransactionElements);
                for (int row = 0; row < BENCH_M; ++row) {
                    for (int column = 0;
                         column < kTransactionColumns;
                         ++column) {
                        const std::size_t index =
                            tile_base + row * kFullOutputColumns +
                            transaction * kTransactionColumns + column;
                        result.checksum += weight * output[index];
                    }
                }
            }
        }
    } else if constexpr (BENCH_TENSOR_RANK == 3) {
        const uint64_t source_sum =
            source_pattern_sum(0, kLogicalTileElements);
        for (int sample = 0; sample < samples; ++sample) {
            const int logical_tile = sample == 0 ? 0 : working_tiles - 1;
            const int output_tile = sample == 0 ? 0 : 7;
            const uint64_t weight = static_cast<uint64_t>(sample + 1);
            const std::size_t tile_base =
                static_cast<std::size_t>(logical_tile) * BENCH_M *
                kFullOutputColumns;
            result.expected_checksum += weight * source_sum;
            for (int row = 0; row < BENCH_M; ++row) {
                for (int column = 0; column < kTransactionColumns; ++column) {
                    const std::size_t index =
                        tile_base + row * kFullOutputColumns +
                        output_tile * kTransactionColumns + column;
                    result.checksum += weight * output[index];
                }
            }
        }
    } else {
        const uint64_t source_sum =
            source_pattern_sum(0, kLogicalTileElements);
        for (int sample = 0; sample < samples; ++sample) {
            const int logical_tile = sample == 0 ? 0 : working_tiles - 1;
            const uint64_t weight = static_cast<uint64_t>(sample + 1);
            const std::size_t tile_base =
                static_cast<std::size_t>(logical_tile) * BENCH_M *
                kFullOutputColumns;
            result.expected_checksum += weight * source_sum;
            for (int element = 0; element < kLogicalTileElements; ++element) {
                result.checksum += weight * output[tile_base + element];
            }
        }
    }

    for (int logical_tile = 0; logical_tile < working_tiles; ++logical_tile) {
        for (int row = 0; row < BENCH_M; ++row) {
            for (int column = 0; column < kFullOutputColumns; ++column) {
                bool touched = false;
                if constexpr (BENCH_TENSOR_RANK == 3) {
                    touched = (logical_tile == 0 && column < kTransactionColumns) ||
                              (logical_tile == working_tiles - 1 &&
                               column >= kFullOutputColumns - kTransactionColumns);
                } else {
                    touched = logical_tile == 0 ||
                              (samples == 2 && logical_tile == working_tiles - 1);
                }
                const std::size_t index =
                    (static_cast<std::size_t>(logical_tile) * BENCH_M + row) *
                        kFullOutputColumns +
                    column;
                if (!touched && output[index] != 0) {
                    result.untouched_zero = false;
                }
            }
        }
    }
    return result;
}

CUtensorMap make_tensor_map(void* output, int working_tiles) {
    if constexpr (BENCH_TENSOR_RANK == 2) {
        microbench::TensorMapSpec<2> spec;
        spec.data_type = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
        spec.global_address = output;
        spec.global_dims = {kFullOutputColumns,
                            static_cast<uint64_t>(working_tiles) * BENCH_M};
        spec.global_strides = {kFullOutputColumns * 2ull};
        spec.box_dims = {kTransactionColumns, BENCH_M};
        spec.swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
        return microbench::encode_tensor_map_2d(spec);
    } else if constexpr (BENCH_TENSOR_RANK == 3) {
        microbench::TensorMapSpec<3> spec;
        spec.data_type = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
        spec.global_address = output;
        spec.global_dims = {kFullOutputColumns, BENCH_M,
                            static_cast<uint64_t>(working_tiles)};
        spec.global_strides = {kFullOutputColumns * 2ull,
                               kFullOutputColumns * BENCH_M * 2ull};
        spec.box_dims = {kTransactionColumns, BENCH_M, 1};
        spec.swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
        return microbench::encode_tensor_map_3d(spec);
    } else if constexpr (BENCH_TENSOR_RANK == 4) {
        microbench::TensorMapSpec<4> spec;
        spec.data_type = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
        spec.global_address = output;
        spec.global_dims = {kFullOutputColumns, BENCH_M,
                            static_cast<uint64_t>(working_tiles), 1};
        spec.global_strides = {kFullOutputColumns * 2ull,
                               kFullOutputColumns * BENCH_M * 2ull,
                               static_cast<uint64_t>(working_tiles) *
                                   kFullOutputColumns * BENCH_M * 2ull};
        spec.box_dims = {kTransactionColumns, BENCH_M, 1, 1};
        spec.swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
        return microbench::encode_tensor_map_4d(spec);
    } else {
        microbench::TensorMapSpec<5> spec;
        spec.data_type = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
        spec.global_address = output;
        spec.global_dims = {kTransactionColumns, BENCH_M,
                            kFullOutputColumns / kTransactionColumns,
                            static_cast<uint64_t>(working_tiles), 1};
        spec.global_strides = {kFullOutputColumns * 2ull,
                               kTransactionColumns * 2ull,
                               kFullOutputColumns * BENCH_M * 2ull,
                               static_cast<uint64_t>(working_tiles) *
                                   kFullOutputColumns * BENCH_M * 2ull};
        spec.box_dims = {kTransactionColumns, BENCH_M,
                         kFullOutputColumns / kTransactionColumns, 1, 1};
        spec.swizzle = CU_TENSOR_MAP_SWIZZLE_128B;
        return microbench::encode_tensor_map_5d(spec);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const microbench::CliArgs args(argc, argv);
        const int repeat = args.get_int("repeat", BENCH_N == 64 ? 256 : 64,
                                        1, 1 << 24);
        const int depth = args.get_int("depth", 1, 1, kMaxDepth);
        const int working_tiles = args.get_int("working-set-tiles", 64,
                                                1, 1 << 20);
        const int warmup = args.get_int("warmup", 5, 0, 1000);
        const int samples = args.get_int("samples", 20, 1, 10000);
        if constexpr (BENCH_TENSOR_RANK != 3) {
            const int active_depth = depth < repeat ? depth : repeat;
            if (working_tiles < active_depth) {
                throw std::invalid_argument(
                    "rank-2/4/5 TMA store requires --working-set-tiles >= "
                    "min(--depth, --repeat) to avoid outstanding WAW aliases");
            }
        }
        const auto device = microbench::require_sm90(args.get_int("device", 0));

        const std::size_t shared_bytes =
            static_cast<std::size_t>(depth) * kLogicalTileBytes;
        if (shared_bytes > device.properties.sharedMemPerBlockOptin) {
            throw std::invalid_argument("requested TMA depth needs " +
                std::to_string(shared_bytes) +
                " shared bytes, exceeding sharedMemPerBlockOptin=" +
                std::to_string(device.properties.sharedMemPerBlockOptin));
        }

        const std::size_t output_elements =
            static_cast<std::size_t>(working_tiles) * BENCH_M *
            kFullOutputColumns;
        microbench::DeviceBuffer<uint16_t> output(output_elements);
        microbench::DeviceBuffer<uint64_t> starts(kThreads);
        microbench::DeviceBuffer<uint64_t> stops(kThreads);
        output.zero();
        const CUtensorMap tensor_map = make_tensor_map(output.data(), working_tiles);

        const auto kernel = &tma_store_kernel;
        microbench::throw_if_cuda_error(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(shared_bytes)),
            "set tma_store dynamic shared memory");
        const auto validation_kernel = &validate_tma_store_kernel;
        microbench::throw_if_cuda_error(
            cudaFuncSetAttribute(validation_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 kLogicalTileBytes),
            "set tma_store validation shared memory");

        auto measure_once = [&]() -> double {
            tma_store_kernel<<<1, kThreads, shared_bytes>>>(
                tensor_map, starts.data(), stops.data(), repeat, depth,
                working_tiles);
            microbench::throw_if_cuda_error(cudaGetLastError(),
                                             "tma_store_kernel launch");
            microbench::throw_if_cuda_error(cudaDeviceSynchronize(),
                                             "tma_store_kernel synchronize");
            const auto host_starts = starts.copy_to_host();
            const auto host_stops = stops.copy_to_host();
            return static_cast<double>(microbench::reduce_cycles(
                host_starts.data(), host_stops.data(), host_starts.size()));
        };

        const auto series = microbench::run_samples(warmup, samples, measure_once);
        const auto summary = series.summary();
        output.zero();
        validate_tma_store_kernel<<<1, kThreads, kLogicalTileBytes>>>(
            tensor_map, working_tiles);
        microbench::throw_if_cuda_error(cudaGetLastError(),
                                         "validate_tma_store_kernel launch");
        microbench::throw_if_cuda_error(cudaDeviceSynchronize(),
                                         "validate_tma_store_kernel synchronize");
        const std::vector<uint16_t> host_output = output.copy_to_host();
        const ValidationResult validation =
            validate_output(host_output, working_tiles);
        if (validation.checksum != validation.expected_checksum ||
            !validation.untouched_zero) {
            throw std::runtime_error(
                "TMA store validation failed: expected checksum " +
                std::to_string(validation.expected_checksum) + ", got " +
                std::to_string(validation.checksum) +
                (validation.untouched_zero
                     ? std::string()
                     : "; non-zero data found outside validation tiles"));
        }
        const double requested_bytes =
            static_cast<double>(repeat) * kLogicalTileBytes;
        const double transactions =
            static_cast<double>(repeat) * kTransactionsPerTile;

        microbench::JsonLine json;
        json.add("benchmark", "tma_store/tile64x" + std::to_string(BENCH_N) +
                                  "_bf16_" + std::to_string(BENCH_TENSOR_RANK) +
                                  "d_sm90")
            .add("gpu", device.properties.name)
            .add("rank", BENCH_TENSOR_RANK)
            .add("m", BENCH_M)
            .add("n", BENCH_N)
            .add("repeat", repeat)
            .add("depth", depth)
            .add("working_set_tiles", working_tiles)
            .add("transactions_per_tile", kTransactionsPerTile)
            .add("logical_tile_bytes", kLogicalTileBytes)
            .add("checksum", validation.checksum)
            .add("expected_checksum", validation.expected_checksum)
            .add("untouched_zero", validation.untouched_zero);
        microbench::add_measurement_summary(json, summary);
        json.add("cycle_per_tile", summary.median / repeat)
            .add("transaction_per_clk_sm", transactions / summary.median)
            .add("requested_byte_per_clk_sm", requested_bytes / summary.median)
            .print();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tma_store benchmark error: " << error.what() << '\n';
        return 1;
    }
}
