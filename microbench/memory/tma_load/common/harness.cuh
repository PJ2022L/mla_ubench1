#pragma once

#include <cstdint>
#include <exception>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include "common/bench.hpp"
#include "ptx.cuh"
#include "tensor_map.hpp"

namespace microbench::tma_load_bench {

enum class Mode {
    kTile64x64,
    kTile64x576,
};

constexpr int kRows = 64;
constexpr int kHeadDimension = 576;
constexpr int kTileColumns = 64;
constexpr int kHeadTilesPerPage = kHeadDimension / kTileColumns;
constexpr int kTileElements = kRows * kTileColumns;
constexpr int kTileBytes = kTileElements * 2;
constexpr int kPageElements = kRows * kHeadDimension;
constexpr int kPageBytes = kPageElements * 2;
constexpr int kThreads = 32;
constexpr int kMaxBarrierStages = kHeadTilesPerPage;

static_assert(kHeadDimension % kTileColumns == 0);

enum class Pattern : int {
    kLocal,
    kSequential,
    kRandom,
    kReuse,
};

inline Pattern parse_pattern(const std::string& value) {
    if (value == "local") return Pattern::kLocal;
    if (value == "sequential") return Pattern::kSequential;
    if (value == "random") return Pattern::kRandom;
    if (value == "reuse") return Pattern::kReuse;
    throw std::invalid_argument(
        "--pattern must be local, sequential, random, or reuse");
}

__host__ __device__ inline uint16_t input_pattern(uint64_t index) {
    uint32_t value = static_cast<uint32_t>(index) * 747796405u + 2891336453u;
    value = ((value >> ((value >> 28) + 4)) ^ value) * 277803737u;
    value = (value >> 22) ^ value;
    return static_cast<uint16_t>((value & 0x7fffu) + 1u);
}

__global__ void initialize_input(uint16_t* input, std::size_t elements) {
    std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (; index < elements; index += stride) {
        input[index] = input_pattern(index);
    }
}

#if defined(MB_TMA_RESULT_NAME)
__global__ void evict_l2(uint64_t* data, std::size_t elements) {
    std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (; index < elements; index += stride) {
        uint64_t value;
        asm volatile("ld.global.u64 %0, [%1];"
                     : "=l"(value) : "l"(data + index) : "memory");
        value += index + 1;
        asm volatile("st.global.u64 [%0], %1;"
                     : : "l"(data + index), "l"(value) : "memory");
    }
}
#endif

__host__ __device__ __forceinline__ int select_page(
        Pattern pattern,
        int page_iteration,
        int working_pages,
        int block,
        int grid_blocks) {
    if (pattern == Pattern::kLocal) {
        return block % working_pages;
    }
    if (pattern == Pattern::kSequential) {
        const uint64_t linear =
            static_cast<uint64_t>(block) +
            static_cast<uint64_t>(page_iteration) *
                static_cast<uint64_t>(grid_blocks);
        return static_cast<int>(
            linear % static_cast<uint64_t>(working_pages));
    }
    if (pattern == Pattern::kReuse) {
        const int reuse_pages = working_pages < 4 ? working_pages : 4;
        return page_iteration % reuse_pages;
    }
    uint32_t value = static_cast<uint32_t>(block + 1) * 0x9e3779b9u;
    value ^= static_cast<uint32_t>(page_iteration + 1) * 0x85ebca6bu;
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    return static_cast<int>(value % static_cast<uint32_t>(working_pages));
}

template <Mode mode>
__host__ __device__ constexpr int logical_bytes() {
    if constexpr (mode == Mode::kTile64x64) {
        return kTileBytes;
    } else {
        return kPageBytes;
    }
}

template <Mode mode>
__host__ __device__ constexpr int max_depth() {
    if constexpr (mode == Mode::kTile64x64) {
        return kHeadTilesPerPage;
    } else {
        return 3;
    }
}

template <Mode mode>
__host__ __device__ constexpr int transactions_per_logical_operation() {
    if constexpr (mode == Mode::kTile64x64) {
        return 1;
    } else {
        return kHeadTilesPerPage;
    }
}

template <Mode mode>
__device__ __forceinline__ void issue_logical_load(
        const CUtensorMap* tensor_map,
        uint64_t* barrier,
        unsigned char* destination,
        int iteration,
        int working_pages,
        Pattern pattern) {
    if constexpr (mode == Mode::kTile64x64) {
        const int head_tile = iteration % kHeadTilesPerPage;
        const int page_iteration = iteration / kHeadTilesPerPage;
        const int page = select_page(
            pattern, page_iteration, working_pages,
            static_cast<int>(blockIdx.x), static_cast<int>(gridDim.x));
        ptx::tma_load_4d(
            tensor_map, barrier, destination,
            head_tile * kTileColumns, 0, 0, page, ptx::kTmaEvictFirst);
        ptx::mbarrier_arrive_expect_tx(barrier, kTileBytes);
    } else {
        const int page = select_page(
            pattern, iteration, working_pages,
            static_cast<int>(blockIdx.x), static_cast<int>(gridDim.x));
#pragma unroll
        for (int head_tile = 0;
             head_tile < kHeadTilesPerPage;
             ++head_tile) {
            ptx::tma_load_4d(
                tensor_map, barrier,
                destination + head_tile * kTileBytes,
                head_tile * kTileColumns, 0, 0, page,
                ptx::kTmaEvictFirst);
        }
        ptx::mbarrier_arrive_expect_tx(barrier, kPageBytes);
    }
}

template <Mode mode>
__device__ __forceinline__ uint64_t consume_logical_load(
        const unsigned char* source,
        int iteration) {
    const auto* values = reinterpret_cast<const uint16_t*>(source);
    if constexpr (mode == Mode::kTile64x64) {
        return static_cast<uint64_t>(values[0]) *
               static_cast<uint64_t>(iteration + 1);
    } else {
        uint64_t checksum = 0;
#pragma unroll
        for (int head_tile = 0;
             head_tile < kHeadTilesPerPage;
             ++head_tile) {
            const uint64_t weight =
                static_cast<uint64_t>(iteration) * kHeadTilesPerPage +
                head_tile + 1;
            checksum += weight * values[head_tile * kTileElements];
        }
        return checksum;
    }
}

template <Mode mode, bool Target = true>
__global__ void tma_load_kernel(
        __grid_constant__ const CUtensorMap tensor_map,
        uint64_t* cycles,
        uint64_t* sinks,
        uint32_t* smids,
        int iterations,
        int depth,
        int working_pages,
        Pattern pattern) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    extern __shared__ __align__(128) unsigned char shared_storage[];
    __shared__ alignas(8) uint64_t barriers[kMaxBarrierStages];
    if constexpr (Target) {
        if (threadIdx.x == 0) {
            for (int stage = 0; stage < depth; ++stage) {
                ptx::mbarrier_init(&barriers[stage], 1);
            }
            ptx::mbarrier_init_fence();
        }
    } else {
        auto* values = reinterpret_cast<uint16_t*>(shared_storage);
        constexpr int kSamplesPerStage =
            mode == Mode::kTile64x64 ? 1 : kHeadTilesPerPage;
        for (int index = threadIdx.x; index < depth * kSamplesPerStage;
             index += blockDim.x) {
            const int stage = index / kSamplesPerStage;
            const int head_tile = index % kSamplesPerStage;
            const int stage_elements = logical_bytes<mode>() / 2;
            values[stage * stage_elements + head_tile * kTileElements] =
                static_cast<uint16_t>(index + 1);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        uint32_t phase_bits = 0;
        uint64_t checksum = 0;
        const uint64_t start = read_clock64();
        for (int base = 0; base < iterations; base += depth) {
            const int active = min(depth, iterations - base);
            for (int stage = 0; stage < active; ++stage) {
                if constexpr (Target) {
                    issue_logical_load<mode>(
                        &tensor_map, &barriers[stage],
                        shared_storage + stage * logical_bytes<mode>(),
                        base + stage, working_pages, pattern);
                } else {
                    const int iteration = base + stage;
                    const int page_iteration =
                        mode == Mode::kTile64x64
                            ? iteration / kHeadTilesPerPage
                            : iteration;
                    const int page = select_page(
                        pattern, page_iteration, working_pages,
                        static_cast<int>(blockIdx.x),
                        static_cast<int>(gridDim.x));
                    asm volatile("" : : "r"(page));
                }
            }
            for (int stage = 0; stage < active; ++stage) {
                if constexpr (Target) {
                    ptx::mbarrier_wait_parity(
                        &barriers[stage], (phase_bits >> stage) & 1u);
                    phase_bits ^= 1u << stage;
                }
                checksum += consume_logical_load<mode>(
                    shared_storage + stage * logical_bytes<mode>(),
                    base + stage);
            }
        }
        const uint64_t stop = read_clock64();
        cycles[blockIdx.x] = stop - start;
        sinks[blockIdx.x] = checksum;
        if (smids != nullptr) smids[blockIdx.x] = read_smid();
    }
#else
    if (threadIdx.x == 0) {
        cycles[blockIdx.x] = 0;
        sinks[blockIdx.x] = 0;
    }
#endif
}

template <Mode mode>
__global__ void validate_tma_load_kernel(
        __grid_constant__ const CUtensorMap tensor_map,
        uint64_t* checksum,
        int working_pages) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    extern __shared__ __align__(128) unsigned char shared_storage[];
    __shared__ alignas(8) uint64_t barrier;
    if (threadIdx.x == 0) {
        ptx::mbarrier_init(&barrier, 1);
        ptx::mbarrier_init_fence();
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        uint32_t phase = 0;
        uint64_t result = 0;
        const int validation_pages[2] = {0, working_pages - 1};
        for (int sample = 0; sample < 2; ++sample) {
            if constexpr (mode == Mode::kTile64x576) {
#pragma unroll
                for (int head_tile = 0;
                     head_tile < kHeadTilesPerPage;
                     ++head_tile) {
                    ptx::tma_load_4d(
                        &tensor_map, &barrier,
                        shared_storage + head_tile * kTileBytes,
                        head_tile * kTileColumns, 0, 0,
                        validation_pages[sample], ptx::kTmaEvictFirst);
                }
                ptx::mbarrier_arrive_expect_tx(&barrier, kPageBytes);
                ptx::mbarrier_wait_parity(&barrier, phase);
                phase ^= 1u;
            }

            for (int head_tile = 0;
                 head_tile < kHeadTilesPerPage;
                 ++head_tile) {
                if constexpr (mode == Mode::kTile64x64) {
                    ptx::tma_load_4d(
                        &tensor_map, &barrier, shared_storage,
                        head_tile * kTileColumns, 0, 0,
                        validation_pages[sample], ptx::kTmaEvictFirst);
                    ptx::mbarrier_arrive_expect_tx(&barrier, kTileBytes);
                    ptx::mbarrier_wait_parity(&barrier, phase);
                    phase ^= 1u;
                }
                const auto* values = reinterpret_cast<const uint16_t*>(
                    shared_storage +
                    (mode == Mode::kTile64x576 ? head_tile * kTileBytes : 0));
                const uint64_t weight =
                    1 + sample * kHeadTilesPerPage + head_tile;
                for (int element = 0; element < kTileElements; ++element) {
                    result += weight * values[element];
                }
            }
        }
        checksum[0] = result;
    }
#else
    if (threadIdx.x == 0) {
        checksum[0] = 0;
    }
#endif
}

template <Mode mode>
inline uint64_t expected_checksum(Pattern pattern,
                                  int iterations,
                                  int working_pages,
                                  int block,
                                  int grid_blocks) {
    uint64_t checksum = 0;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if constexpr (mode == Mode::kTile64x64) {
            const int head_tile = iteration % kHeadTilesPerPage;
            const int page_iteration = iteration / kHeadTilesPerPage;
            const int page = select_page(
                pattern, page_iteration, working_pages, block, grid_blocks);
            const uint16_t value = input_pattern(
                static_cast<uint64_t>(page) * kPageElements +
                head_tile * kTileColumns);
            checksum += static_cast<uint64_t>(value) *
                        static_cast<uint64_t>(iteration + 1);
        } else {
            const int page = select_page(
                pattern, iteration, working_pages, block, grid_blocks);
            for (int head_tile = 0;
                 head_tile < kHeadTilesPerPage;
                 ++head_tile) {
                const uint16_t value = input_pattern(
                    static_cast<uint64_t>(page) * kPageElements +
                    head_tile * kTileColumns);
                const uint64_t weight =
                    static_cast<uint64_t>(iteration) * kHeadTilesPerPage +
                    head_tile + 1;
                checksum += weight * value;
            }
        }
    }
    return checksum;
}

inline uint64_t expected_validation_checksum(int working_pages) {
    uint64_t checksum = 0;
    const int validation_pages[2] = {0, working_pages - 1};
    for (int sample = 0; sample < 2; ++sample) {
        for (int head_tile = 0;
             head_tile < kHeadTilesPerPage;
             ++head_tile) {
            const uint64_t weight =
                1 + sample * kHeadTilesPerPage + head_tile;
            for (int row = 0; row < kRows; ++row) {
                for (int column = 0; column < kTileColumns; ++column) {
                    const uint64_t index =
                        static_cast<uint64_t>(validation_pages[sample]) *
                            kPageElements +
                        static_cast<uint64_t>(row) * kHeadDimension +
                        head_tile * kTileColumns + column;
                    checksum += weight * input_pattern(index);
                }
            }
        }
    }
    return checksum;
}

template <Mode mode>
int run(int argc, char** argv) {
    try {
        const Args args(argc, argv);
#if defined(MB_TMA_RESULT_NAME)
        args.require_only({
            "iters", "warmup", "samples", "blocks", "device", "peak",
            "depth", "working-set-pages", "pattern", "cache-mode"});
#else
        args.require_only({
            "iters", "warmup", "samples", "blocks", "device", "peak",
            "depth", "working-set-pages", "pattern"});
#endif
        const auto options = parse_common_options(
            args, mode == Mode::kTile64x64 ? 1024 : 256);
        const auto properties = require_sm90(options.device);
        const int depth = args.get_int("depth", 1, 1, max_depth<mode>());
        const int working_pages =
            args.get_int("working-set-pages", 1024, 1, 1 << 20);
        const std::string pattern_name =
            args.get_string("pattern", "sequential");
        const Pattern pattern = parse_pattern(pattern_name);
#if defined(MB_TMA_RESULT_NAME)
        const std::string cache_mode = args.get_string(
            "cache-mode", "hbm_stream");
        if (cache_mode != "l2_hot" && cache_mode != "hbm_stream") {
            throw std::invalid_argument(
                "--cache-mode must be l2_hot or hbm_stream");
        }
#else
        const std::string cache_mode = "uncontrolled";
#endif
        const int blocks = resolve_blocks(options.blocks, properties, 1);
        const int shared_bytes = depth * logical_bytes<mode>();
        if (shared_bytes >
            static_cast<int>(properties.sharedMemPerBlockOptin)) {
            throw std::invalid_argument(
                "requested TMA depth exceeds sharedMemPerBlockOptin");
        }

        const std::size_t input_elements =
            static_cast<std::size_t>(working_pages) * kPageElements;
        DeviceBuffer<uint16_t> input(input_elements);
        initialize_input<<<256, 256>>>(input.data(), input_elements);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        const CUtensorMap tensor_map =
            make_tma_load_64x576_b16_rank4_map(
                input.data(), working_pages);
        CUDA_CHECK(cudaFuncSetAttribute(
            tma_load_kernel<mode, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_bytes));
        CUDA_CHECK(cudaFuncSetAttribute(
            tma_load_kernel<mode, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_bytes));
        CUDA_CHECK(cudaFuncSetAttribute(
            validate_tma_load_kernel<mode>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kPageBytes));

        DeviceBuffer<uint64_t> latency_cycles(1);
        DeviceBuffer<uint64_t> latency_baseline_cycles(1);
        DeviceBuffer<uint64_t> latency_sinks(1);
        DeviceBuffer<uint64_t> latency_baseline_sinks(1);
        auto launch_latency_pair = [&] {
            tma_load_kernel<mode, true>
                <<<1, kThreads, logical_bytes<mode>()>>>(
                    tensor_map, latency_cycles.data(), latency_sinks.data(),
                    nullptr, options.iters, 1, working_pages, pattern);
            tma_load_kernel<mode, false>
                <<<1, kThreads, logical_bytes<mode>()>>>(
                    tensor_map, latency_baseline_cycles.data(),
                    latency_baseline_sinks.data(), nullptr, options.iters, 1,
                    working_pages, pattern);
        };
        PairedClockSamples latency_clock_samples;
#if defined(MB_TMA_RESULT_NAME)
        DeviceBuffer<uint64_t> eviction((128ULL << 20) / sizeof(uint64_t));
        auto prepare_latency = [&] {
            if (cache_mode == "l2_hot") {
                tma_load_kernel<mode, true>
                    <<<1, kThreads, logical_bytes<mode>()>>>(
                        tensor_map, latency_cycles.data(),
                        latency_sinks.data(), nullptr, options.iters, 1,
                        working_pages, pattern);
            } else {
                evict_l2<<<512, 256>>>(eviction.data(), eviction.size());
            }
        };
        latency_clock_samples = measure_paired_clock_cycles_prepared(
            options.warmup, options.samples, latency_cycles.data(),
            latency_baseline_cycles.data(), 1, prepare_latency,
            launch_latency_pair);
#else
        latency_clock_samples = measure_paired_clock_cycles(
            options.warmup, options.samples, latency_cycles.data(),
            latency_baseline_cycles.data(), 1, launch_latency_pair);
#endif
        std::vector<double> latency_metric_samples;
        latency_metric_samples.reserve(options.samples);
        for (int index = 0; index < options.samples; ++index) {
            latency_metric_samples.push_back(
                (latency_clock_samples.target[index] -
                 latency_clock_samples.baseline[index]) /
                options.iters);
        }
        const double completion_cycles = median(latency_metric_samples);

        DeviceBuffer<uint64_t> throughput_cycles(blocks);
        DeviceBuffer<uint64_t> throughput_baseline_cycles(blocks);
        DeviceBuffer<uint64_t> throughput_sinks(blocks);
        DeviceBuffer<uint64_t> throughput_baseline_sinks(blocks);
        DeviceBuffer<uint32_t> throughput_smids(blocks);
        auto launch_initiation_pair = [&] {
            tma_load_kernel<mode, true><<<blocks, kThreads, shared_bytes>>>(
                tensor_map, throughput_cycles.data(), throughput_sinks.data(),
                throughput_smids.data(), options.iters, depth,
                working_pages, pattern);
            tma_load_kernel<mode, false><<<blocks, kThreads, shared_bytes>>>(
                tensor_map, throughput_baseline_cycles.data(),
                throughput_baseline_sinks.data(), nullptr, options.iters,
                depth, working_pages, pattern);
        };
        PairedClockSamples initiation_clock_samples;
#if defined(MB_TMA_RESULT_NAME)
        auto launch_throughput = [&] {
            tma_load_kernel<mode, true><<<blocks, kThreads, shared_bytes>>>(
                tensor_map, throughput_cycles.data(), throughput_sinks.data(),
                throughput_smids.data(), options.iters, depth,
                working_pages, pattern);
        };
        auto prepare_throughput = [&] {
            if (cache_mode == "l2_hot") {
                launch_throughput();
            } else {
                evict_l2<<<512, 256>>>(eviction.data(), eviction.size());
            }
        };
        initiation_clock_samples = measure_paired_clock_cycles_prepared(
            options.warmup, options.samples, throughput_cycles.data(),
            throughput_baseline_cycles.data(), blocks, prepare_throughput,
            launch_initiation_pair);
        const auto event_samples = measure_event_ms_prepared(
            options.warmup, options.samples, prepare_throughput,
            launch_throughput);
#else
        initiation_clock_samples = measure_paired_clock_cycles(
            options.warmup, options.samples, throughput_cycles.data(),
            throughput_baseline_cycles.data(), blocks,
            launch_initiation_pair);
        const auto event_samples = measure_event_ms(
            options.warmup,
            options.samples,
            [&] {
                tma_load_kernel<mode, true><<<blocks, kThreads, shared_bytes>>>(
                    tensor_map, throughput_cycles.data(),
                    throughput_sinks.data(), throughput_smids.data(),
                    options.iters, depth,
                    working_pages, pattern);
            });
#endif
        const double elapsed_ms = median(event_samples);
        std::vector<double> initiation_interval_samples;
        initiation_interval_samples.reserve(options.samples);
        for (int index = 0; index < options.samples; ++index) {
            initiation_interval_samples.push_back(
                (initiation_clock_samples.target[index] -
                 initiation_clock_samples.baseline[index]) /
                options.iters);
        }
        const double initiation_interval_cycles =
            median(initiation_interval_samples);
        const auto host_sinks = throughput_sinks.copy_to_host();
        for (int block = 0; block < blocks; ++block) {
            const uint64_t expected = expected_checksum<mode>(
                pattern, options.iters, working_pages, block, blocks);
            if (host_sinks[block] != expected) {
                throw std::runtime_error(
                    "TMA load checksum mismatch in block " +
                    std::to_string(block));
            }
        }

        DeviceBuffer<uint64_t> validation_checksum(1);
        validate_tma_load_kernel<mode><<<1, kThreads, kPageBytes>>>(
            tensor_map, validation_checksum.data(), working_pages);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        const uint64_t actual_validation =
            validation_checksum.copy_to_host().front();
        const uint64_t expected_validation =
            expected_validation_checksum(working_pages);
        if (actual_validation != expected_validation) {
            throw std::runtime_error(
                "TMA load full-page validation checksum mismatch");
        }

        const double logical_operations =
            static_cast<double>(blocks) * options.iters;
        const double transactions =
            logical_operations * transactions_per_logical_operation<mode>();
        const double requested_bytes =
            logical_operations * logical_bytes<mode>();
        auto throughput_metric_samples = event_samples;
        auto bandwidth_metric_samples = event_samples;
        for (double& value : throughput_metric_samples) {
            value = logical_operations / value / 1.0e6;
        }
        for (double& value : bandwidth_metric_samples) {
            value = requested_bytes / value / 1.0e6;
        }
        const double giga_operations_per_second =
            median(throughput_metric_samples);
        const double bandwidth_gbps = median(bandwidth_metric_samples);

        JsonObject params;
        params.add("gpu", properties.name)
            .add("operation",
                 mode == Mode::kTile64x64 ? "tile_64x64" : "tile_64x576")
            .add("tile_rows", kRows)
            .add("tile_columns",
                 mode == Mode::kTile64x64
                     ? kTileColumns
                     : kHeadDimension)
            .add("transaction_shape", "64x64")
            .add("transactions_per_logical_operation",
                 transactions_per_logical_operation<mode>())
            .add("source_head_dimension", kHeadDimension)
            .add("source_row_stride_bytes", kHeadDimension * 2)
            .add("source_page_bytes", kPageBytes)
            .add("dtype", kB16DtypeName)
            .add("tensor_rank", 4)
            .add("cache_hint", "evict_first")
            .add("pattern", pattern_name)
            .add("cache_mode", cache_mode)
            .add("working_set_pages", working_pages)
            .add("working_set_bytes",
                 static_cast<uint64_t>(working_pages) * kPageBytes)
            .add("allocation_bytes",
                 static_cast<uint64_t>(working_pages) * kPageBytes)
            .add("blocks", options.blocks)
            .add("resolved_blocks", blocks)
            .add("iters", options.iters)
            .add("warmup", options.warmup)
            .add("samples", options.samples)
            .add("device", options.device)
            .add("peak", options.peak)
            .add("depth", depth)
            .add("initiation_interval_cycles", initiation_interval_cycles)
            .add("initiation_interval_boundary",
                 "target protocol minus matched address/loop/consume baseline")
            .add("throughput_depth", depth)
            .add("latency_depth", 1)
            .add("clock_baseline",
                 "same page selection, loop nesting, shared consume, and sink; "
                 "no TMA issue, expect_tx, or wait")
            .add("descriptor_tiles", kHeadTilesPerPage)
            .add("requested_bytes_per_logical_operation",
                 logical_bytes<mode>())
            .add("correct", true)
            .add("correctness",
                 "timed first-element sink plus full first/last-page validation");
#if defined(MB_TMA_RESULT_NAME)
        std::map<uint32_t, int> smid_counts;
        for (uint32_t smid : throughput_smids.copy_to_host()) {
            ++smid_counts[smid];
        }
        JsonObject smid_histogram;
        for (const auto& [smid, count] : smid_counts) {
            smid_histogram.add(std::to_string(smid), count);
        }
        params.add("resource", "tma")
            .add("unique_active_sms", static_cast<int>(smid_counts.size()))
            .add("smid_histogram", smid_histogram)
            .add("cache_preparation", cache_mode == "l2_hot"
                    ? "untimed_target_prewarm_each_sample"
                    : "untimed_128MiB_l2_eviction_each_sample");
#endif
        if constexpr (mode == Mode::kTile64x64) {
            params.add("transaction_sequence", "head_tile=iteration%9")
                .add("barriers_per_full_page", kHeadTilesPerPage);
        } else {
            params.add("barrier_expected_bytes", kPageBytes)
                .add("barriers_per_tile", 1);
        }

        JsonObject latency;
        latency.add("value", completion_cycles)
            .add("unit",
                 mode == Mode::kTile64x64
                     ? "cycle/transaction"
                     : "cycle/tile")
            .add("timer", "clock64")
            .add("scope", "cta")
            .add("boundary",
                 mode == Mode::kTile64x64
                     ? "baseline-subtracted 1x issue+expect_tx(8192)+wait"
                     : "baseline-subtracted 9x issue+expect_tx(73728)+wait")
            .add_raw("samples", json_number_array(latency_metric_samples))
            .add_raw("target_samples_cycles",
                     json_number_array(latency_clock_samples.target))
            .add_raw("baseline_samples_cycles",
                     json_number_array(latency_clock_samples.baseline));

        JsonObject throughput;
        throughput.add("value", giga_operations_per_second)
            .add("unit",
                 mode == Mode::kTile64x64
                     ? "Gtransaction/s"
                     : "Gtile/s")
            .add("timer", "cuda_event")
            .add("scope", "grid")
            .add("event_ms", elapsed_ms)
            .add("logical_operations", logical_operations)
            .add("transactions", transactions)
            .add_raw("samples", json_number_array(throughput_metric_samples))
            .add_raw("event_samples_ms", json_number_array(event_samples));

        JsonObject bandwidth;
        bandwidth.add("value", bandwidth_gbps)
            .add("unit", "GB/s")
            .add("kind", "requested")
            .add("bytes", requested_bytes)
            .add_raw("samples", json_number_array(bandwidth_metric_samples));

        const std::string result_name =
#if defined(MB_TMA_RESULT_NAME)
            MB_TMA_RESULT_NAME;
#else
            std::string(mode == Mode::kTile64x64
                            ? "tensor_4d_64x64_"
                            : "tensor_4d_64x576_") +
            kB16DtypeName;
#endif
        print_result(
            result_name,
            params,
            latency,
            throughput,
            bandwidth,
            utilization(bandwidth_gbps, options.peak, "GB/s"));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << (mode == Mode::kTile64x64
                          ? "tma_load"
                          : "tma_load_tile576")
                  << ": " << error.what() << '\n';
        return 1;
    }
}

}  // namespace microbench::tma_load_bench
