// FlashMLA split-epilogue FP32 shared-to-global bulk stores.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>

#include "benchmark_utils.hpp"
#include "clock.cuh"

namespace {

using microbench::CliArgs;
using microbench::DeviceBuffer;

constexpr int kThreads = 256;
constexpr int kWarps = 8;
constexpr int kRows = 64;
constexpr int kColumns = 512;
constexpr int kSharedStride = 520;
constexpr int kRowsPerWarp = kRows / kWarps;
constexpr int kRowBytes = kColumns * sizeof(float);
constexpr int kTileBytes = kRows * kRowBytes;
constexpr int kSharedBytes = kRows * kSharedStride * sizeof(float);
constexpr int kBulkCopiesPerTile = kRows;

static_assert(kRowsPerWarp == 8);
static_assert(kRowBytes == 2048);
static_assert(kTileBytes == 131072);

enum class Pattern : int {
    Sequential = 0,
    Local = 1,
    Random = 2,
};

__host__ __device__ __forceinline__ int select_tile(
    Pattern pattern, int iteration, int working_tiles, int block,
    int grid_blocks) {
    if (pattern == Pattern::Sequential) {
        return (block + iteration * grid_blocks) % working_tiles;
    }
    if (pattern == Pattern::Local) {
        return block % working_tiles;
    }

    uint32_t value = static_cast<uint32_t>(block) * 0x9e3779b9u +
                     static_cast<uint32_t>(iteration + 1) * 0x85ebca6bu;
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return static_cast<int>(value % static_cast<uint32_t>(working_tiles));
}

__global__ __launch_bounds__(kThreads)
void bulk_store_kernel(float* __restrict__ output,
                       uint64_t* __restrict__ starts,
                       uint64_t* __restrict__ stops,
                       uint32_t* __restrict__ sink,
                       int working_tiles,
                       int repeat,
                       Pattern pattern) {
    extern __shared__ __align__(128) float shared_tile[];
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;

    for (int index = tid; index < kRows * kSharedStride;
         index += kThreads) {
        const int row = index / kSharedStride;
        const int column = index - row * kSharedStride;
        shared_tile[index] =
            0.0009765625f * static_cast<float>(
                1 + ((row * 37 + column * 13 + tid) & 1023));
    }
    __syncthreads();
    cutlass::arch::fence_view_async_shared();
    __syncthreads();

    int final_tile = 0;
    uint64_t start = 0;
    uint64_t stop = 0;
    CLK_START(start);
#pragma unroll 1
    for (int iteration = 0; iteration < repeat; ++iteration) {
        const int tile = select_tile(
            pattern, iteration, working_tiles,
            static_cast<int>(blockIdx.x), static_cast<int>(gridDim.x));
        final_tile = tile;
        if (lane == 0) {
#pragma unroll
            for (int local_row = 0; local_row < kRowsPerWarp; ++local_row) {
                const int row = local_row * kWarps + warp;
                cute::SM90_BULK_COPY_S2G::copy(
                    shared_tile + row * kSharedStride,
                    output +
                        (static_cast<int64_t>(tile) * kRows + row) * kColumns,
                    kRowBytes);
            }
            cute::tma_store_arrive();
            cute::tma_store_wait<0>();
        }
    }
    CLK_STOP(stop);

    const int uid = static_cast<int>(blockIdx.x) * kThreads + tid;
    starts[uid] = start;
    stops[uid] = stop;
    if (tid == 0) {
        const float observed = output[
            static_cast<int64_t>(final_tile) * kRows * kColumns];
        sink[blockIdx.x] = __float_as_uint(observed) ^
                           static_cast<uint32_t>(final_tile + 1);
    }
}

Pattern parse_pattern(const std::string& value) {
    if (value == "sequential") return Pattern::Sequential;
    if (value == "local") return Pattern::Local;
    if (value == "random") return Pattern::Random;
    throw std::invalid_argument(
        "--pattern must be sequential, local, or random");
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

float expected_source_value(int row, int column) {
    const int index = row * kSharedStride + column;
    const int tid = index % kThreads;
    return 0.0009765625f * static_cast<float>(
        1 + ((row * 37 + column * 13 + tid) & 1023));
}

void validate_final_tile(const DeviceBuffer<float>& output,
                         Pattern pattern,
                         int working_tiles,
                         int repeat,
                         int blocks) {
    struct Coordinate {
        int row;
        int column;
    };
    constexpr Coordinate kCoordinates[] = {
        {0, 0}, {0, 511}, {17, 3}, {31, 255}, {63, 0}, {63, 511},
    };
    const int tile = select_tile(
        pattern, repeat - 1, working_tiles, 0, blocks);
    std::vector<float> host_tile(
        static_cast<std::size_t>(kRows) * kColumns);
    output.copy_to_host(
        host_tile.data(), host_tile.size(),
        static_cast<std::size_t>(tile) * kRows * kColumns);

    for (const Coordinate coordinate : kCoordinates) {
        const float observed = host_tile[
            static_cast<std::size_t>(coordinate.row) * kColumns +
            coordinate.column];
        const float expected =
            expected_source_value(coordinate.row, coordinate.column);
        if (observed != expected) {
            throw std::runtime_error(
                "bulk-store global output validation failed");
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const CliArgs args(argc, argv);
        const std::string pattern_name =
            args.get_string("pattern", "sequential");
        const Pattern pattern = parse_pattern(pattern_name);
        const std::string working_set =
            args.get_string("working-set", "hbm");
        if (working_set != "l2" && working_set != "hbm") {
            throw std::invalid_argument("--working-set must be l2 or hbm");
        }
        const int default_working_set_mib = working_set == "l2" ? 16 : 256;
        const int working_set_mib = args.get_int(
            "working-set-mib", default_working_set_mib, 1, 16384);
        const int repeat = args.get_int("repeat", 64, 1, 1 << 20);
        const int warmup = args.get_int("warmup", 5, 0, 1000);
        const int samples = args.get_int("samples", 20, 1, 10000);
        const auto device = microbench::require_sm90(args.get_int("device", 0));
        const int blocks = args.get_int(
            "blocks", device.properties.multiProcessorCount, 1, 4096);

        if (kSharedBytes > device.properties.sharedMemPerBlockOptin) {
            throw std::runtime_error(
                "64x520 FP32 source tile exceeds sharedMemPerBlockOptin");
        }

        constexpr std::size_t kMib = 1024u * 1024u;
        const std::size_t requested_working_set_bytes =
            static_cast<std::size_t>(working_set_mib) * kMib;
        const std::size_t working_tiles = std::max<std::size_t>(
            1, (requested_working_set_bytes + kTileBytes - 1) / kTileBytes);
        if (working_tiles > static_cast<std::size_t>(
                                std::numeric_limits<int>::max())) {
            throw std::overflow_error("working set requires too many tiles");
        }
        const std::size_t output_elements =
            working_tiles * kRows * kColumns;

        DeviceBuffer<float> output(output_elements);
        DeviceBuffer<uint64_t> starts(
            static_cast<std::size_t>(blocks) * kThreads);
        DeviceBuffer<uint64_t> stops(
            static_cast<std::size_t>(blocks) * kThreads);
        DeviceBuffer<uint32_t> sink(blocks);
        output.zero();

        microbench::throw_if_cuda_error(
            cudaFuncSetAttribute(bulk_store_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 kSharedBytes),
            "bulk_store_kernel dynamic shared-memory attribute");

        auto measure_once = [&]() -> double {
            bulk_store_kernel<<<blocks, kThreads, kSharedBytes>>>(
                output.data(), starts.data(), stops.data(), sink.data(),
                static_cast<int>(working_tiles), repeat, pattern);
            microbench::throw_if_cuda_error(
                cudaGetLastError(), "bulk_store_kernel launch");
            microbench::throw_if_cuda_error(
                cudaDeviceSynchronize(), "bulk_store_kernel synchronize");
            const auto host_starts = starts.copy_to_host();
            const auto host_stops = stops.copy_to_host();
            return median_cta_cycles(host_starts, host_stops, blocks);
        };

        const auto series = microbench::run_samples(warmup, samples, measure_once);
        const auto summary = series.summary();
        const auto host_sink = sink.copy_to_host();
        const uint64_t checksum = std::accumulate(
            host_sink.begin(), host_sink.end(), uint64_t{0});
        if (checksum == 0) {
            throw std::runtime_error(
                "bulk-store sink is zero; global output was not observed");
        }
        validate_final_tile(output, pattern, static_cast<int>(working_tiles),
                            repeat, blocks);

        const double copies = static_cast<double>(repeat) *
                              kBulkCopiesPerTile;
        const double requested_bytes = static_cast<double>(repeat) * kTileBytes;
        microbench::JsonLine json;
        json.add("benchmark", "bulk_store/tile64x512_f32_sm90")
            .add("gpu", device.properties.name)
            .add("pattern", pattern_name)
            .add("working_set", working_set)
            .add("working_set_mib_requested", working_set_mib)
            .add("working_set_tiles", working_tiles)
            .add("working_set_bytes_actual", output.bytes())
            .add("blocks", blocks)
            .add("threads", kThreads)
            .add("warp_leaders", kWarps)
            .add("completion_depth", 1)
            .add("rows_per_warp_leader", kRowsPerWarp)
            .add("bulk_copies_per_tile", kBulkCopiesPerTile)
            .add("bytes_per_bulk_copy", kRowBytes)
            .add("logical_tile_bytes", kTileBytes)
            .add("repeat", repeat)
            .add("cpu_validation", true)
            .add("checksum", checksum);
        microbench::add_measurement_summary(json, summary);
        json.add("cycle_per_tile", summary.median / repeat)
            .add("cycle_per_row_store", summary.median / copies)
            .add("bulk_copy_per_clk_sm", copies / summary.median)
            .add("requested_byte_per_clk_sm",
                 requested_bytes / summary.median)
            .print();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "bulk-store benchmark error: " << error.what() << '\n';
        return 1;
    }
}
