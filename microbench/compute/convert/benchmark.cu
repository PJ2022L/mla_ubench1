// FlashMLA FP8 e4m3 x8 + scale -> BF16 x8 conversion throughput.

#include <cstdint>
#include <exception>
#include <iostream>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>

#include "benchmark_utils.hpp"
#include "clock.cuh"
#include "sm90/decode/sparse_fp8/components/dequant.h"

namespace {

using microbench::CliArgs;
using microbench::DeviceBuffer;
using sm90::decode::sparse_fp8::cvt_fp8x8_bf16x8;
using sm90::decode::sparse_fp8::fp8x8;

constexpr int kThreads = 128;
constexpr int kCallsPerThread = 16;

union Fp8Bits {
    uint64_t bits;
    fp8x8 value;
};

union Bf16Bits {
    uint4 words;
    bf16x8 value;
};

static_assert(sizeof(Fp8Bits) == sizeof(uint64_t));
static_assert(sizeof(Bf16Bits) == sizeof(uint4));

__global__ void convert_kernel(const uint64_t* __restrict__ inputs,
                               uint64_t* __restrict__ starts,
                               uint64_t* __restrict__ stops,
                               uint4* __restrict__ sink,
                               int repeat) {
    const int tid = threadIdx.x;
    uint64_t states[kCallsPerThread];
#pragma unroll
    for (int i = 0; i < kCallsPerThread; ++i) {
        states[i] = inputs[tid * kCallsPerThread + i];
    }
    const __nv_bfloat162 scale = __floats2bfloat162_rn(0.75f, 0.75f);
    Bf16Bits outputs[kCallsPerThread];

    uint64_t start = 0;
    uint64_t stop = 0;
    CLK_START(start);
#pragma unroll 1
    for (int iteration = 0; iteration < repeat; ++iteration) {
#pragma unroll
        for (int i = 0; i < kCallsPerThread; ++i) {
            // Empty read/write asm is a compiler dependency only. It keeps the
            // conversion in the loop without adding a GPU instruction.
            asm volatile("" : "+l"(states[i]));
            Fp8Bits input;
            input.bits = states[i];
            outputs[i].value = cvt_fp8x8_bf16x8(input.value, scale);
            asm volatile(""
                         : "+r"(outputs[i].words.x),
                           "+r"(outputs[i].words.y),
                           "+r"(outputs[i].words.z),
                           "+r"(outputs[i].words.w));
        }
    }
    CLK_STOP(stop);

    starts[tid] = start;
    stops[tid] = stop;
#pragma unroll
    for (int i = 0; i < kCallsPerThread; ++i) {
        sink[tid * kCallsPerThread + i] = outputs[i].words;
    }
}

uint64_t make_fp8_bits(int tid, int lane) {
    uint64_t result = 0;
    for (int byte = 0; byte < 8; ++byte) {
        // Restrict values to finite, normal e4m3 encodings.
        const uint8_t encoded = static_cast<uint8_t>(0x20 +
            ((tid * 13 + lane * 7 + byte * 3) % 0x38));
        result |= static_cast<uint64_t>(encoded) << (byte * 8);
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const CliArgs args(argc, argv);
        const int repeat = args.get_int("repeat", 512, 1, 1 << 24);
        const int warmup = args.get_int("warmup", 5, 0, 1000);
        const int samples = args.get_int("samples", 20, 1, 10000);
        const auto device = microbench::require_sm90(args.get_int("device", 0));

        std::vector<uint64_t> host_inputs(kThreads * kCallsPerThread);
        for (int tid = 0; tid < kThreads; ++tid) {
            for (int i = 0; i < kCallsPerThread; ++i) {
                host_inputs[tid * kCallsPerThread + i] = make_fp8_bits(tid, i);
            }
        }

        DeviceBuffer<uint64_t> inputs(host_inputs.size());
        DeviceBuffer<uint64_t> starts(kThreads);
        DeviceBuffer<uint64_t> stops(kThreads);
        DeviceBuffer<uint4> sink(kThreads * kCallsPerThread);
        inputs.copy_from_host(host_inputs);

        auto measure_once = [&]() -> double {
            convert_kernel<<<1, kThreads>>>(inputs.data(), starts.data(),
                                            stops.data(), sink.data(), repeat);
            microbench::throw_if_cuda_error(cudaGetLastError(), "convert_kernel launch");
            microbench::throw_if_cuda_error(cudaDeviceSynchronize(),
                                             "convert_kernel synchronize");
            const auto host_starts = starts.copy_to_host();
            const auto host_stops = stops.copy_to_host();
            return static_cast<double>(microbench::reduce_cycles(
                host_starts.data(), host_stops.data(), host_starts.size()));
        };

        const auto series = microbench::run_samples(warmup, samples, measure_once);
        const auto summary = series.summary();
        const auto host_sink = sink.copy_to_host();
        uint64_t checksum = 0;
        for (const uint4& value : host_sink) {
            checksum += value.x;
            checksum += value.y;
            checksum += value.z;
            checksum += value.w;
        }
        if (checksum == 0) {
            throw std::runtime_error("conversion sink is zero; target work may have been removed");
        }

        const double calls_per_warpgroup =
            static_cast<double>(kThreads) * kCallsPerThread * repeat;
        microbench::JsonLine json;
        json.add("benchmark", "convert/fp8x8_to_bf16x8_sm90")
            .add("gpu", device.properties.name)
            .add("repeat", repeat)
            .add("threads", kThreads)
            .add("calls_per_thread", kCallsPerThread)
            .add("checksum", checksum);
        microbench::add_measurement_summary(json, summary);
        json.add("cycle_per_cvt_thread", summary.median /
                    (static_cast<double>(repeat) * kCallsPerThread))
            .add("cvt_per_clk_sm", calls_per_warpgroup / summary.median)
            .add("element_per_clk_sm", 8.0 * calls_per_warpgroup / summary.median)
            .print();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "convert benchmark error: " << error.what() << '\n';
        return 1;
    }
}
