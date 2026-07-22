#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "../../../../../../microbench/common/bench.hpp"

namespace microbench::metadata_stage_bench {

struct alignas(32) SchedulerRecord {
    int begin_req_idx;
    int end_req_idx;
    int begin_block_idx;
    int end_block_idx;
    int begin_split_idx;
    int is_first_req_splitted;
    int is_last_req_splitted;
    int pad;
};
static_assert(sizeof(SchedulerRecord) == 32);

enum class Distribution : int { kUniform, kRamp, kSkewed, kRandom };

inline Distribution parse_distribution(const std::string& value) {
    if (value == "uniform") return Distribution::kUniform;
    if (value == "ramp") return Distribution::kRamp;
    if (value == "skewed") return Distribution::kSkewed;
    if (value == "random") return Distribution::kRandom;
    throw std::invalid_argument(
        "--seqlen-distribution must be uniform, ramp, skewed, or random");
}

inline std::vector<int> make_seqlens(int batch,
                                     int minimum,
                                     int maximum,
                                     Distribution distribution,
                                     uint32_t seed) {
    std::vector<int> values(batch);
    uint32_t state = seed;
    for (int index = 0; index < batch; ++index) {
        if (distribution == Distribution::kUniform) {
            values[index] = maximum;
        } else if (distribution == Distribution::kRamp) {
            values[index] = batch == 1
                ? maximum
                : minimum + static_cast<int>(
                    static_cast<int64_t>(maximum - minimum) * index /
                    (batch - 1));
        } else if (distribution == Distribution::kSkewed) {
            values[index] = index % 8 == 0 ? maximum : minimum;
        } else {
            state = state * 1664525u + 1013904223u;
            const uint32_t range =
                static_cast<uint32_t>(maximum - minimum) + 1u;
            values[index] = minimum + static_cast<int>(state % range);
        }
    }
    return values;
}

__device__ __forceinline__ int load_nc_i32(const int* pointer) {
    int value;
    asm volatile("ld.global.nc.u32 %0, [%1];"
                 : "=r"(value) : "l"(pointer) : "memory");
    return value;
}

__device__ __forceinline__ int shfl_xor_i32(int value, int delta) {
    int result;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
                 : "=r"(result) : "r"(value), "r"(delta));
    return result;
}

__global__ __launch_bounds__(32, 1)
void metadata_kernel(const int* seqlens,
                     SchedulerRecord* records,
                     int* num_splits,
                     uint64_t* cycles,
                     uint32_t* sink,
                     uint32_t* smids,
                     int batch,
                     int num_sm_parts,
                     int iterations) {
    extern __shared__ int shared[];
    int* num_blocks_shared = shared;
    int* num_splits_shared = shared + batch;
    int* seqlens_shared = shared + batch * 2 + 1;
    int* first_block_shared = shared + batch * 3 + 1;
    int* last_block_shared = shared + batch * 4 + 1;
    constexpr int kBlockSize = 64;
    constexpr int kFixedOverhead = 5;
    uint32_t checksum = 0;
    const uint64_t start = read_clock64();
#pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        int total_num_blocks = 0;
        for (int index = threadIdx.x; index < batch; index += 32) {
            const int seqlen = load_nc_i32(seqlens + index);
            seqlens_shared[index] = seqlen;
            const int last_token = seqlen > 0 ? seqlen - 1 : 0;
            const int first_block = 0;
            const int last_block = last_token / kBlockSize;
            const int num_blocks = last_block - first_block + 1;
            total_num_blocks += num_blocks + kFixedOverhead;
            num_blocks_shared[index] = num_blocks;
            first_block_shared[index] = first_block;
            last_block_shared[index] = last_block;
        }
#pragma unroll
        for (int offset = 16; offset >= 1; offset /= 2) {
            total_num_blocks += shfl_xor_i32(total_num_blocks, offset);
        }
        __syncwarp();

        if (threadIdx.x == 0) {
            const int payload =
                (total_num_blocks + num_sm_parts - 1) / num_sm_parts +
                kFixedOverhead;
            int request = 0;
            int block = 0;
            int split = 0;
            int cumulative_splits = 0;
            num_splits_shared[0] = 0;
            for (int part = 0; part < num_sm_parts; ++part) {
                SchedulerRecord record{};
                record.begin_req_idx = request;
                record.begin_block_idx = block + first_block_shared[request];
                record.begin_split_idx = split;
                record.is_first_req_splitted = block != 0;
                int remaining_payload = payload;
                while (request < batch) {
                    const int num_blocks = num_blocks_shared[request];
                    const int remaining_blocks = num_blocks - block;
                    if (remaining_payload >=
                        remaining_blocks + kFixedOverhead) {
                        cumulative_splits += split + 1;
                        num_splits_shared[request + 1] = cumulative_splits;
                        remaining_payload -=
                            remaining_blocks + kFixedOverhead;
                        ++request;
                        block = 0;
                        split = 0;
                    } else {
                        if (remaining_payload - kFixedOverhead > 0) {
                            block += remaining_payload - kFixedOverhead;
                            ++split;
                        }
                        break;
                    }
                }
                record.end_req_idx = block > 0 ? request : request - 1;
                record.end_block_idx = block > 0
                    ? block + first_block_shared[request]
                    : (seqlens_shared[request - 1] == 0
                        ? 0 : last_block_shared[request - 1] + 1);
                record.is_last_req_splitted =
                    record.end_block_idx !=
                        last_block_shared[record.end_req_idx] + 1 &&
                    seqlens_shared[record.end_req_idx] != 0;
                if (record.begin_req_idx == record.end_req_idx) {
                    const int any = record.is_first_req_splitted ||
                                    record.is_last_req_splitted;
                    record.is_first_req_splitted = any;
                    record.is_last_req_splitted = any;
                }
                records[part] = record;
                checksum ^= static_cast<uint32_t>(
                    record.begin_req_idx * 131 + record.end_block_idx * 17 +
                    record.begin_split_idx);
            }
        }
        __syncwarp();
        for (int index = threadIdx.x; index <= batch; index += 32) {
            num_splits[index] = num_splits_shared[index];
            checksum ^= static_cast<uint32_t>(num_splits_shared[index] + index);
        }
        __syncwarp();
    }
    const uint64_t stop = read_clock64();
    if (threadIdx.x == 0) {
        cycles[0] = stop - start;
        sink[0] = checksum == 0 ? 1u : checksum;
        smids[0] = read_smid();
    }
}

inline void reference_scheduler(const std::vector<int>& seqlens,
                                int num_sm_parts,
                                std::vector<SchedulerRecord>& records,
                                std::vector<int>& num_splits) {
    constexpr int kBlockSize = 64;
    constexpr int kFixedOverhead = 5;
    const int batch = static_cast<int>(seqlens.size());
    std::vector<int> num_blocks(batch), first(batch, 0), last(batch);
    int total = 0;
    for (int index = 0; index < batch; ++index) {
        last[index] = std::max(seqlens[index] - 1, 0) / kBlockSize;
        num_blocks[index] = last[index] + 1;
        total += num_blocks[index] + kFixedOverhead;
    }
    const int payload =
        (total + num_sm_parts - 1) / num_sm_parts + kFixedOverhead;
    int request = 0, block = 0, split = 0, cumulative = 0;
    num_splits[0] = 0;
    for (int part = 0; part < num_sm_parts; ++part) {
        if (request >= batch) {
            throw std::invalid_argument(
                "num-sm-parts is too large for this seqlen distribution");
        }
        SchedulerRecord record{};
        record.begin_req_idx = request;
        record.begin_block_idx = block + first[request];
        record.begin_split_idx = split;
        record.is_first_req_splitted = block != 0;
        int remaining = payload;
        while (request < batch) {
            const int available = num_blocks[request] - block;
            if (remaining >= available + kFixedOverhead) {
                cumulative += split + 1;
                num_splits[request + 1] = cumulative;
                remaining -= available + kFixedOverhead;
                ++request;
                block = 0;
                split = 0;
            } else {
                if (remaining - kFixedOverhead > 0) {
                    block += remaining - kFixedOverhead;
                    ++split;
                }
                break;
            }
        }
        record.end_req_idx = block > 0 ? request : request - 1;
        record.end_block_idx = block > 0
            ? block + first[request]
            : (seqlens[request - 1] == 0 ? 0 : last[request - 1] + 1);
        record.is_last_req_splitted =
            record.end_block_idx != last[record.end_req_idx] + 1 &&
            seqlens[record.end_req_idx] != 0;
        if (record.begin_req_idx == record.end_req_idx) {
            const int any = record.is_first_req_splitted ||
                            record.is_last_req_splitted;
            record.is_first_req_splitted = any;
            record.is_last_req_splitted = any;
        }
        records[part] = record;
    }
    if (request != batch || block != 0 || split != 0) {
        throw std::invalid_argument(
            "num-sm-parts does not cover the generated scheduler workload");
    }
}

inline int run(int argc, char** argv) {
    try {
        const Args args(argc, argv);
        args.require_only({"iters", "warmup", "samples", "blocks", "device",
                           "peak", "batch", "num-sm-parts",
                           "seqlen-distribution", "seqlen-min", "seqlen-max",
                           "seed"});
        const auto options = parse_common_options(args, 256);
        if (options.blocks > 1) {
            throw std::invalid_argument("metadata stage always uses one CTA");
        }
        const auto properties = require_sm90(options.device);
        const int batch = args.get_int("batch", 128, 1, 8192);
        const int num_sm_parts =
            args.get_int("num-sm-parts", properties.multiProcessorCount,
                         1, 4096);
        const int seqlen_min = args.get_int("seqlen-min", 64, 0, 1 << 26);
        const int seqlen_max =
            args.get_int("seqlen-max", 8192, seqlen_min, 1 << 26);
        const uint32_t seed = static_cast<uint32_t>(
            args.get_int("seed", 20260719, 0, std::numeric_limits<int>::max()));
        const std::string distribution_name =
            args.get_string("seqlen-distribution", "uniform");
        const auto distribution = parse_distribution(distribution_name);
        const auto host_seqlens = make_seqlens(
            batch, seqlen_min, seqlen_max, distribution, seed);
        std::vector<SchedulerRecord> expected_records(num_sm_parts);
        std::vector<int> expected_splits(batch + 1);
        reference_scheduler(host_seqlens, num_sm_parts,
                            expected_records, expected_splits);

        const int shared_bytes = (batch * 5 + 1) * sizeof(int);
        if (shared_bytes > static_cast<int>(properties.sharedMemPerBlockOptin)) {
            throw std::invalid_argument(
                "metadata shared memory exceeds sharedMemPerBlockOptin");
        }
        CUDA_CHECK(cudaFuncSetAttribute(
            metadata_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_bytes));
        DeviceBuffer<int> seqlens(batch);
        DeviceBuffer<SchedulerRecord> records(num_sm_parts);
        DeviceBuffer<int> num_splits(batch + 1);
        DeviceBuffer<uint64_t> cycles(1);
        DeviceBuffer<uint32_t> sink(1);
        DeviceBuffer<uint32_t> smids(1);
        CUDA_CHECK(cudaMemcpy(seqlens.data(), host_seqlens.data(),
                              batch * sizeof(int), cudaMemcpyHostToDevice));

        const auto raw_cycle_samples = measure_clock_cycles(
            options.warmup, options.samples, cycles.data(), [&] {
                metadata_kernel<<<1, 32, shared_bytes>>>(
                    seqlens.data(), records.data(), num_splits.data(),
                    cycles.data(), sink.data(), smids.data(), batch,
                    num_sm_parts, options.iters);
            });
        auto latency_samples = raw_cycle_samples;
        for (double& value : latency_samples) value /= options.iters;
        const double cycles_per_stage = median(latency_samples);
        const auto event_samples_ms = measure_event_ms(
            options.warmup, options.samples, [&] {
                metadata_kernel<<<1, 32, shared_bytes>>>(
                    seqlens.data(), records.data(), num_splits.data(),
                    cycles.data(), sink.data(), smids.data(), batch,
                    num_sm_parts, options.iters);
            });
        auto throughput_samples = event_samples_ms;
        for (double& value : throughput_samples) {
            value = options.iters / value / 1.0e6;
        }
        const double throughput_gstage = median(throughput_samples);
        const double elapsed_ms = median(event_samples_ms);
        const double bytes_per_stage =
            batch * sizeof(int) + num_sm_parts * sizeof(SchedulerRecord) +
            (batch + 1) * sizeof(int);
        auto bandwidth_samples = event_samples_ms;
        for (double& value : bandwidth_samples) {
            value = options.iters * bytes_per_stage / value / 1.0e6;
        }
        const double bandwidth_gbs = median(bandwidth_samples);

        const auto actual_records = records.copy_to_host();
        const auto actual_splits = num_splits.copy_to_host();
        if (std::memcmp(actual_records.data(), expected_records.data(),
                        expected_records.size() * sizeof(SchedulerRecord)) != 0 ||
            actual_splits != expected_splits) {
            throw std::runtime_error("metadata stage CPU reference mismatch");
        }

        JsonObject params;
        params.add("gpu", properties.name).add("batch", batch)
            .add("num_sm_parts", num_sm_parts).add("block_size_n", 64)
            .add("fixed_overhead_num_blocks", 5)
            .add("seqlen_distribution", distribution_name)
            .add("seqlen_min", seqlen_min).add("seqlen_max", seqlen_max)
            .add("seed", seed).add("shared_bytes", shared_bytes)
            .add("iters", options.iters).add("warmup", options.warmup)
            .add("samples", options.samples).add("blocks", options.blocks)
            .add("resolved_blocks", 1).add("device", options.device)
            .add("peak", options.peak).add("correct", true);
        const auto observed_smids = smids.copy_to_host();
        JsonObject latency;
        latency.add("value", cycles_per_stage).add("unit", "cycle/stage")
            .add("timer", "clock64").add("scope", "single_warp_cta")
            .add("boundary", "complete dense scheduler metadata stage")
            .add_raw("samples", json_number_array(latency_samples))
            .add_raw("observed_smids", json_number_array(observed_smids));
        JsonObject throughput;
        throughput.add("value", throughput_gstage).add("unit", "Gstage/s")
            .add("timer", "cuda_event").add("scope", "single_cta")
            .add("event_ms", elapsed_ms)
            .add_raw("samples", json_number_array(throughput_samples))
            .add_raw("event_samples_ms", json_number_array(event_samples_ms));
        JsonObject bandwidth;
        bandwidth.add("value", bandwidth_gbs).add("unit", "GB/s")
            .add("kind", "requested_global").add("bytes_per_stage", bytes_per_stage)
            .add_raw("samples", json_number_array(bandwidth_samples));
        JsonObject hardware;
        hardware.add_null("value").add("unit", "ratio")
            .add("reason", "composite single-warp scheduler stage has no peak");
        print_result("metadata_stage", params, latency,
                     throughput, bandwidth, hardware);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metadata_stage: " << error.what() << '\n';
        return 1;
    }
}

}  // namespace microbench::metadata_stage_bench
