// FlashMLA sparse-decode 128-bit cluster shared-memory producer stores.

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>

#include "benchmark_utils.hpp"
#include "clock.cuh"

namespace {

using microbench::CliArgs;
using microbench::DeviceBuffer;
using TransactionBarrier = cutlass::arch::ClusterTransactionBarrier;

constexpr int kThreads = 128;
constexpr int kClusterSize = 2;
constexpr int kStoresPerThread = 18;
constexpr int kRowsPerCta = 32;
constexpr int kDimension = 576;
constexpr int kTileBytes = kRowsPerCta * kDimension * sizeof(uint16_t);
constexpr int kBarrierOffset = (kTileBytes + 15) & ~15;
constexpr int kSharedBytes = kBarrierOffset + sizeof(uint64_t);

enum class Mode : int {
    Peer = 0,
    Local = 1,
};

// Exact instruction used by FlashMLA's sparse FP8 decode st_async_128b helper.
__device__ __forceinline__ void store_async_cluster_128(
    uint32_t destination, const uint4& value, uint32_t transaction_barrier) {
    const long2 data = *reinterpret_cast<const long2*>(&value);
    asm volatile(
        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.s64 "
        "[%0], {%1, %2}, [%3];\n"
        :
        : "r"(destination), "l"(data.x), "l"(data.y),
          "r"(transaction_barrier)
        : "memory");
}

__device__ __forceinline__ int kmajor_byte_offset(int tid, int store_idx) {
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int token = warp * 8 + lane % 8;
    const int dimension_lane = lane / 8;

    int dimension_base = 0;
    if (store_idx < 16) {
        const int nope_tile = store_idx / 2;
        const int half = store_idx % 2;
        dimension_base = nope_tile * 64 + dimension_lane * 16 + half * 8;
    } else {
        const int rope_tile = store_idx - 16;
        dimension_base = 512 + rope_tile * 32 + dimension_lane * 8;
    }
    const int element_offset = dimension_base * kRowsPerCta + token * 8;
    return element_offset * static_cast<int>(sizeof(uint16_t));
}

__global__ void dsm_store_kernel(uint64_t* __restrict__ starts,
                                 uint64_t* __restrict__ stops,
                                 uint32_t* __restrict__ sink,
                                 int repeat,
                                 Mode mode) {
    extern __shared__ __align__(16) unsigned char shared_raw[];
    unsigned char* const tile = shared_raw;
    auto* const barrier = reinterpret_cast<TransactionBarrier*>(
        shared_raw + kBarrierOffset);

    const int tid = threadIdx.x;
    const int rank = static_cast<int>(cute::block_rank_in_cluster());
    const int destination_rank = mode == Mode::Peer ? rank ^ 1 : rank;
    const uint32_t local_tile_address = cute::cast_smem_ptr_to_uint(tile);
    const uint32_t local_barrier_address =
        cute::cast_smem_ptr_to_uint(barrier);
    const uint32_t destination_tile_address =
        cute::set_block_rank(local_tile_address, destination_rank);
    const uint32_t destination_barrier_address =
        cute::set_block_rank(local_barrier_address, destination_rank);

    if (tid == 0) {
        barrier->init(1);
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();
    cute::cluster_sync();

    uint4 value = {
        static_cast<uint32_t>(tid + 1 + rank * kThreads),
        static_cast<uint32_t>(tid * 3 + 5 + rank * 17),
        static_cast<uint32_t>(tid * 7 + 11 + rank * 29),
        static_cast<uint32_t>(tid * 13 + 17 + rank * 43),
    };
    uint32_t phase = 0;
    uint64_t start = 0;
    uint64_t stop = 0;

    CLK_START(start);
#pragma unroll 1
    for (int iteration = 0; iteration < repeat; ++iteration) {
        if (tid == 0) {
            barrier->arrive_and_expect_tx(kTileBytes);
        }
        __syncthreads();
        cute::cluster_sync();

#pragma unroll
        for (int store_idx = 0; store_idx < kStoresPerThread; ++store_idx) {
            const uint32_t destination = destination_tile_address +
                static_cast<uint32_t>(kmajor_byte_offset(tid, store_idx));
            store_async_cluster_128(
                destination, value, destination_barrier_address);
        }

        if (tid == 0) {
            barrier->wait(phase);
        }
        __syncthreads();
        phase ^= 1u;
        value.x += 0x9e3779b9u;
        value.y ^= value.x;
    }
    CLK_STOP(stop);

    const int sink_slot = kmajor_byte_offset(
        tid, tid % kStoresPerThread) / static_cast<int>(sizeof(uint4));
    const uint4 observed = reinterpret_cast<const uint4*>(tile)[sink_slot];
    const int uid = static_cast<int>(blockIdx.x) * kThreads + tid;
    starts[uid] = start;
    stops[uid] = stop;
    sink[uid] = observed.x ^ observed.y ^ observed.z ^ observed.w ^
                static_cast<uint32_t>(uid + 1);
}

Mode parse_mode(const std::string& value) {
    if (value == "peer") {
        return Mode::Peer;
    }
    if (value == "local") {
        return Mode::Local;
    }
    throw std::invalid_argument("--mode must be peer or local");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const CliArgs args(argc, argv);
        const std::string mode_name = args.get_string("mode", "peer");
        const Mode mode = parse_mode(mode_name);
        const int repeat = args.get_int("repeat", 512, 1, 1 << 20);
        const int warmup = args.get_int("warmup", 5, 0, 1000);
        const int samples = args.get_int("samples", 20, 1, 10000);
        const auto device = microbench::require_sm90(args.get_int("device", 0));

        constexpr int kOutputThreads = kClusterSize * kThreads;
        DeviceBuffer<uint64_t> starts(kOutputThreads);
        DeviceBuffer<uint64_t> stops(kOutputThreads);
        DeviceBuffer<uint32_t> sink(kOutputThreads);

        microbench::throw_if_cuda_error(
            cudaFuncSetAttribute(dsm_store_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 kSharedBytes),
            "dsm_store_kernel dynamic shared-memory attribute");

        auto measure_once = [&]() -> double {
            cudaLaunchAttribute attribute{};
            attribute.id = cudaLaunchAttributeClusterDimension;
            attribute.val.clusterDim.x = kClusterSize;
            attribute.val.clusterDim.y = 1;
            attribute.val.clusterDim.z = 1;

            cudaLaunchConfig_t config{};
            config.gridDim = dim3(kClusterSize, 1, 1);
            config.blockDim = dim3(kThreads, 1, 1);
            config.dynamicSmemBytes = kSharedBytes;
            config.stream = nullptr;
            config.attrs = &attribute;
            config.numAttrs = 1;

            microbench::throw_if_cuda_error(
                cudaLaunchKernelEx(&config, dsm_store_kernel,
                                   starts.data(), stops.data(), sink.data(),
                                   repeat, mode),
                "dsm_store_kernel launch");
            microbench::throw_if_cuda_error(cudaDeviceSynchronize(),
                                             "dsm_store_kernel synchronize");

            const auto host_starts = starts.copy_to_host();
            const auto host_stops = stops.copy_to_host();
            double slowest_cta_cycles = 0.0;
            for (int cta = 0; cta < kClusterSize; ++cta) {
                const std::size_t offset =
                    static_cast<std::size_t>(cta) * kThreads;
                slowest_cta_cycles = std::max(
                    slowest_cta_cycles,
                    static_cast<double>(microbench::reduce_cycles(
                        host_starts.data() + offset,
                        host_stops.data() + offset, kThreads)));
            }
            return slowest_cta_cycles;
        };

        const auto series = microbench::run_samples(warmup, samples, measure_once);
        const auto summary = series.summary();
        const auto host_sink = sink.copy_to_host();
        const uint64_t checksum = std::accumulate(
            host_sink.begin(), host_sink.end(), uint64_t{0});
        if (checksum == 0) {
            throw std::runtime_error(
                "DSM-store sink is zero; target work may have been removed");
        }

        const double cluster_stores = static_cast<double>(kClusterSize) *
                                      kThreads * kStoresPerThread * repeat;
        const double cluster_bytes = cluster_stores * sizeof(uint4);
        microbench::JsonLine json;
        json.add("benchmark", "dsm_store/128b_cluster2_sm90")
            .add("gpu", device.properties.name)
            .add("mode", mode_name)
            .add("cluster_size", kClusterSize)
            .add("threads_per_cta", kThreads)
            .add("rows_per_cta", kRowsPerCta)
            .add("stores_per_thread_per_block", kStoresPerThread)
            .add("bytes_per_cta_block", kTileBytes)
            .add("repeat", repeat)
            .add("checksum", checksum);
        microbench::add_measurement_summary(json, summary);
        json.add("cycle_per_64_token_block", summary.median / repeat)
            .add("cycle_per_source_store", summary.median /
                    (static_cast<double>(repeat) * kStoresPerThread))
            .add("source_store_per_clk_cluster",
                 cluster_stores / summary.median)
            .add("byte_per_clk_cluster", cluster_bytes / summary.median)
            .print();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "DSM-store benchmark error: " << error.what() << '\n';
        return 1;
    }
}
