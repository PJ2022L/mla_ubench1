#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "common/bench.hpp"
#include "compute/wgmma/common/ptx.cuh"
#include "memory/tma_load/common/ptx.cuh"
#include "memory/tma_load/common/tensor_map.hpp"

namespace microbench::interference {

enum class Probe { kMixedWgmma, kWgmmaTma, kWgmmaSfuShared };

template <Probe P> struct Traits;
template <> struct Traits<Probe::kMixedWgmma> {
    static constexpr const char* kName = "wgmma_mixed_shape_mode";
    static constexpr const char* kPeerResource = "tensor";
};
template <> struct Traits<Probe::kWgmmaTma> {
    static constexpr const char* kName = "wgmma_tma_interference";
    static constexpr const char* kPeerResource = "tma";
};
template <> struct Traits<Probe::kWgmmaSfuShared> {
    static constexpr const char* kName = "wgmma_sfu_shared_interference";
    static constexpr const char* kPeerResource = "sfu+shared";
};

constexpr int kWarpgroupThreads = 128;
constexpr int kPrimaryBytes = 2 * 64 * 64 * 2;
constexpr int kPeerWgmmaBytes = 256 * 64 * 2;
constexpr int kTmaBytes = 64 * 64 * 2;
constexpr uint32_t kPackedBf16 = 0x3a803a80u;

__device__ __forceinline__ void initialize_words(
        unsigned char* storage, int bytes) {
    auto* words = reinterpret_cast<uint32_t*>(storage);
    for (int index = threadIdx.x; index < bytes / 4; index += blockDim.x) {
        words[index] = kPackedBf16;
    }
}

template <int Actors>
__global__ void mixed_wgmma_kernel(uint64_t* cycles,
                                   float* sinks,
                                   uint32_t* smids,
                                   int iterations,
                                   int active_actors) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    extern __shared__ __align__(1024) unsigned char storage[];
    constexpr int kBytes = kPrimaryBytes +
        (Actors == 2 ? kPeerWgmmaBytes : 0);
    initialize_words(storage, kBytes);
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    const int actor = threadIdx.x / kWarpgroupThreads;
    const int lane = threadIdx.x % kWarpgroupThreads;
    uint64_t start = 0, stop = 0;
    float sink = 0.0f;
    if (actor == 0) {
        const uint64_t a = ptx::make_sw128_kmajor_descriptor(storage);
        const uint64_t b = ptx::make_sw128_kmajor_descriptor(
            storage + 64 * 64 * 2);
        ptx::run_m64n64k16_ss<4, 1>(
            a, b, iterations, start, stop, sink);
    } else if constexpr (Actors == 2) {
        if (active_actors == 2) {
            const uint64_t b = ptx::make_sw128_transposed_descriptor(
                storage + kPrimaryBytes);
            ptx::run_m64n256k16_rs<4, 1>(
                b, kPackedBf16, kPackedBf16, kPackedBf16, kPackedBf16,
                iterations, start, stop, sink);
        }
    }
    if (lane == 0 && actor < active_actors) {
        const int index = static_cast<int>(blockIdx.x) * 2 + actor;
        cycles[index] = stop - start;
        sinks[index] = sink;
    }
    if (threadIdx.x == 0) smids[blockIdx.x] = read_smid();
#endif
}

__global__ void wgmma_sfu_shared_kernel(uint64_t* cycles,
                                        float* sinks,
                                        uint32_t* smids,
                                        int iterations,
                                        int active_actors) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    extern __shared__ __align__(1024) unsigned char storage[];
    initialize_words(storage, kPrimaryBytes + 4096);
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    const int actor = threadIdx.x / kWarpgroupThreads;
    const int lane = threadIdx.x % kWarpgroupThreads;
    uint64_t start = 0, stop = 0;
    float sink = -0.75f;
    if (actor == 0) {
        const uint64_t a = ptx::make_sw128_kmajor_descriptor(storage);
        const uint64_t b = ptx::make_sw128_kmajor_descriptor(
            storage + 64 * 64 * 2);
        ptx::run_m64n64k16_ss<4, 1>(
            a, b, iterations, start, stop, sink);
    } else if (actor == 1 && active_actors == 2) {
        uint32_t checksum = 0;
        start = read_clock64();
#pragma unroll 1
        for (int iteration = 0; iteration < iterations; ++iteration) {
            float exponential;
            asm volatile("ex2.approx.ftz.f32 %0, %1;"
                         : "=f"(exponential) : "f"(sink));
            const uint32_t address = shared_address(
                storage + kPrimaryBytes +
                ((lane + iteration * 4) & 4092));
            uint32_t value;
            asm volatile("ld.shared.u32 %0, [%1];"
                         : "=r"(value) : "r"(address) : "memory");
            checksum ^= value;
            sink = exponential * 0.125f - 0.75f;
        }
        stop = read_clock64();
        sink += static_cast<float>(checksum & 1U);
    }
    if (lane == 0 && actor < active_actors) {
        const int index = static_cast<int>(blockIdx.x) * 2 + actor;
        cycles[index] = stop - start;
        sinks[index] = sink;
    }
    if (threadIdx.x == 0) smids[blockIdx.x] = read_smid();
#endif
}

__global__ void wgmma_tma_kernel(
        __grid_constant__ const CUtensorMap map,
        uint64_t* cycles,
        float* sinks,
        uint32_t* smids,
        int iterations,
        int working_pages,
        int active_actors) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    extern __shared__ __align__(1024) unsigned char storage[];
    __shared__ alignas(8) uint64_t barrier;
    initialize_words(storage, kPrimaryBytes);
    if (threadIdx.x == 128 && active_actors == 2) {
        ptx::mbarrier_init(&barrier, 1);
        ptx::mbarrier_init_fence();
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    const int actor = threadIdx.x / kWarpgroupThreads;
    const int lane = threadIdx.x % kWarpgroupThreads;
    uint64_t start = 0, stop = 0;
    float sink = 0.0f;
    if (actor == 0) {
        const uint64_t a = ptx::make_sw128_kmajor_descriptor(storage);
        const uint64_t b = ptx::make_sw128_kmajor_descriptor(
            storage + 64 * 64 * 2);
        ptx::run_m64n64k16_ss<4, 1>(
            a, b, iterations, start, stop, sink);
    } else if (threadIdx.x == 128 && active_actors == 2) {
        unsigned int phase = 0;
        start = read_clock64();
#pragma unroll 1
        for (int iteration = 0; iteration < iterations; ++iteration) {
            ptx::tma_load_4d(
                &map, &barrier, storage + kPrimaryBytes,
                0, 0, 0, iteration % working_pages, ptx::kTmaEvictFirst);
            ptx::mbarrier_arrive_expect_tx(&barrier, kTmaBytes);
            ptx::mbarrier_wait_parity(&barrier, phase);
            phase ^= 1U;
        }
        stop = read_clock64();
        sink = static_cast<float>(iterations);
    }
    if ((actor == 0 && lane == 0) ||
        (threadIdx.x == 128 && active_actors == 2)) {
        const int index = static_cast<int>(blockIdx.x) * 2 + actor;
        cycles[index] = stop - start;
        sinks[index] = sink;
    }
    if (threadIdx.x == 0) smids[blockIdx.x] = read_smid();
#endif
}

inline int unique_sm_count(const DeviceBuffer<uint32_t>& smids) {
    const auto host = smids.copy_to_host();
    return static_cast<int>(std::set<uint32_t>(host.begin(), host.end()).size());
}

template <Probe P>
int run(int argc, char** argv) {
    try {
        const Args args(argc, argv);
        if constexpr (P == Probe::kWgmmaTma) {
            args.require_only({
                "iters", "warmup", "samples", "blocks", "device", "peak",
                "working-set-pages", "actors"});
        } else {
            args.require_only({
                "iters", "warmup", "samples", "blocks", "device", "peak",
                "actors"});
        }
        const CommonOptions options = parse_common_options(args, 1024);
        const cudaDeviceProp properties = require_sm90(options.device);
        const int clock_khz = device_clock_khz(options.device);
        const int blocks = resolve_blocks(options.blocks, properties, 1);
        const int working_pages =
            args.get_int("working-set-pages", 64, 1, 1 << 20);
        const int actors = args.get_int("actors", 2, 1, 2);
        DeviceBuffer<uint64_t> cycles(static_cast<std::size_t>(blocks) * 2);
        DeviceBuffer<float> sinks(static_cast<std::size_t>(blocks) * 2);
        DeviceBuffer<uint32_t> smids(blocks);
        DeviceBuffer<uint16_t> tma_input;
        CUtensorMap map{};
        int shared_bytes = 0;
        if constexpr (P == Probe::kWgmmaTma) {
            tma_input.resize(
                static_cast<std::size_t>(working_pages) * 64 * 576);
            tma_input.zero();
            map = make_tma_load_64x576_b16_rank4_map(
                tma_input.data(), working_pages);
            shared_bytes = kPrimaryBytes + kTmaBytes;
            CUDA_CHECK(cudaFuncSetAttribute(
                wgmma_tma_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
        } else if constexpr (P == Probe::kWgmmaSfuShared) {
            shared_bytes = kPrimaryBytes + 4096;
            CUDA_CHECK(cudaFuncSetAttribute(
                wgmma_sfu_shared_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
        } else {
            shared_bytes = kPrimaryBytes + kPeerWgmmaBytes;
            CUDA_CHECK(cudaFuncSetAttribute(
                mixed_wgmma_kernel<2>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                kPrimaryBytes + kPeerWgmmaBytes));
        }
        auto launch = [&] {
            if constexpr (P == Probe::kMixedWgmma) {
                mixed_wgmma_kernel<2><<<blocks, 256, shared_bytes>>>(
                    cycles.data(), sinks.data(), smids.data(), options.iters,
                    actors);
            } else if constexpr (P == Probe::kWgmmaTma) {
                wgmma_tma_kernel<<<blocks, 2 * kWarpgroupThreads, shared_bytes>>>(
                    map, cycles.data(), sinks.data(), smids.data(),
                    options.iters, working_pages, actors);
            } else {
                wgmma_sfu_shared_kernel<<<
                    blocks, 2 * kWarpgroupThreads, shared_bytes>>>(
                    cycles.data(), sinks.data(), smids.data(), options.iters,
                    actors);
            }
        };
        const auto event_ms_samples = measure_event_ms(
            options.warmup, options.samples, launch);
        std::vector<double> grid_round_cycle_samples = event_ms_samples;
        for (double& value : grid_round_cycle_samples) {
            value = value * clock_khz / options.iters;
        }
        std::vector<double> service_interval_samples = grid_round_cycle_samples;
        for (double& value : service_interval_samples) value /= blocks;
        const double work_items = static_cast<double>(blocks) * options.iters;
        std::vector<double> rate_samples = event_ms_samples;
        for (double& value : rate_samples) value = work_items / value / 1.0e6;
        const auto host_cycles = cycles.copy_to_host();
        const auto host_sinks = sinks.copy_to_host();
        for (int block = 0; block < blocks; ++block) {
            for (int actor = 0; actor < actors; ++actor) {
                const int index = block * 2 + actor;
                if (host_cycles[index] == 0 || !std::isfinite(host_sinks[index])) {
                    throw std::runtime_error(
                        "interference actor produced an invalid timing or sink");
                }
            }
        }
        std::map<uint32_t, int> smid_counts;
        for (uint32_t smid : smids.copy_to_host()) ++smid_counts[smid];
        JsonObject smid_histogram;
        for (const auto& [smid, count] : smid_counts) {
            smid_histogram.add(std::to_string(smid), count);
        }

        JsonObject params;
        params.add("gpu", properties.name)
            .add("resource", "tensor")
            .add("interaction", Traits<P>::kName)
            .add("topology", "single_cta")
            .add("actors", actors)
            .add("unique_active_sms", static_cast<int>(smid_counts.size()))
            .add("smid_histogram", smid_histogram)
            .add("shared_bytes", shared_bytes)
            .add("matched_launch_threads", 2 * kWarpgroupThreads)
            .add("matched_shared_bytes", shared_bytes)
            .add("isolated_row_protocol",
                 "peer actor disabled in the same two-actor launch footprint")
            .add("iters", options.iters)
            .add("warmup", options.warmup)
            .add("samples", options.samples)
            .add("blocks", options.blocks)
            .add("resolved_blocks", blocks)
            .add("initiation_interval_cycles", median(service_interval_samples))
            .add("initiation_interval_scope", "aggregate_pair_round")
            .add("device", options.device)
            .add("peak", options.peak);
        if constexpr (P == Probe::kWgmmaTma) {
            params.add("working_set_pages", working_pages);
        }
        if (actors == 2) {
            params.add("peer_resource", Traits<P>::kPeerResource);
        }
        JsonObject latency;
        latency.add("value", median(grid_round_cycle_samples))
            .add("unit", "cycles/grid-round")
            .add("timer", "cuda_event+device_clock")
            .add("scope", "whole_grid")
            .add_raw("samples", json_number_array(grid_round_cycle_samples));
        JsonObject throughput = metric(median(rate_samples), "Gprobe-round/s");
        throughput.add("timer", "cuda_event")
            .add("scope", "grid")
            .add_raw("samples", json_number_array(rate_samples))
            .add_raw("event_samples_ms", json_number_array(event_ms_samples));
        JsonObject bandwidth;
        bandwidth.add_null("value").add("unit", "GB/s")
            .add("reason", "mixed-resource curve");
        JsonObject hardware;
        hardware.add_null("value").add("unit", "ratio")
            .add("reason", "no single-resource peak");
        print_result(Traits<P>::kName, params, latency, throughput, bandwidth,
                     hardware);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << Traits<P>::kName << ": " << error.what() << '\n';
        return 1;
    }
}

}  // namespace microbench::interference
