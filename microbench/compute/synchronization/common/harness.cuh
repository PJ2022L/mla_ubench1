#pragma once

#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/bench.hpp"

namespace microbench::sync_atomic {

constexpr int kUnroll = 8;

__device__ __forceinline__ uint32_t smem_address(const void* pointer) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

template <int Participants>
struct BarSync {
    static_assert(Participants == 128 || Participants == 256);
    static constexpr const char* kName =
        Participants == 128 ? "bar_sync_128"
                            : "bar_sync_256";
    static constexpr const char* kOpcode = "bar.sync";
    static constexpr const char* kProtocol =
        Participants == 128
            ? "128-thread symmetric named-barrier generation"
            : "256-thread symmetric named-barrier generation";
    static constexpr int kThreads = Participants;
    static constexpr int kCycleLeaders = 1;
    static constexpr double kBlockOperations = 1.0;
    __device__ static __forceinline__ void setup(uint64_t*, uint32_t&) {}
    __device__ static __forceinline__ void once(uint64_t*, uint32_t&) {
        if constexpr (Participants == 128) {
            asm volatile("bar.sync 1, 128;" ::: "memory");
        } else {
            asm volatile("bar.sync 1, 256;" ::: "memory");
        }
    }
};

struct BarArrive {
    static constexpr const char* kName = "bar_arrive_2wg";
    static constexpr const char* kOpcode = "bar.arrive + bar.sync";
    static constexpr const char* kProtocol = "WG0 BAR.ARV and WG1 BAR.SYNC on a 256-participant named barrier; BAR.SYNC 0 prevents generation overrun";
    static constexpr int kThreads = 256;
    static constexpr int kCycleLeaders = 2;
    static constexpr double kBlockOperations = 1.0;
    __device__ static __forceinline__ void setup(uint64_t*, uint32_t&) {}
    __device__ static __forceinline__ void once(uint64_t*, uint32_t&) {
        if (threadIdx.x < 128) {
            asm volatile("bar.arrive 1, 256;" ::: "memory");
        } else {
            asm volatile("bar.sync 1, 256;" ::: "memory");
        }
        asm volatile("bar.sync 0, 256;" ::: "memory");
    }
};

struct WarpSync {
    static constexpr const char* kName = "warp_sync";
    static constexpr const char* kOpcode = "bar.warp.sync";
    static constexpr const char* kProtocol = "converged full-mask BAR.WARP.SYNC PTX; SM90a ptxas elides it, so measured time is the explicit zero-hardware-cost control";
    static constexpr int kThreads = 256;
    static constexpr int kCycleLeaders = 1;
    static constexpr double kBlockOperations = 8.0;
    static constexpr bool kSassElided = true;
    __device__ static __forceinline__ void setup(uint64_t*, uint32_t&) {}
    __device__ static __forceinline__ void once(uint64_t*, uint32_t& membermask) {
        asm volatile("bar.warp.sync %0;" :: "r"(membermask) : "memory");
    }
};

template <class Atom, class = void>
struct SassElision {
    static constexpr bool kElided = false;
};

template <class Atom>
struct SassElision<Atom, std::void_t<decltype(Atom::kSassElided)>> {
    static constexpr bool kElided = Atom::kSassElided;
};

struct MbarrierInit {
    static constexpr const char* kName = "mbarrier_init";
    static constexpr const char* kOpcode = "mbarrier.init.shared::cta.b64";
    static constexpr const char* kProtocol = "single elected thread; MBarrier invalidate precedes each reinitialization";
    static constexpr int kThreads = 32;
    static constexpr int kCycleLeaders = 1;
    static constexpr double kBlockOperations = 1.0;
    __device__ static __forceinline__ void setup(uint64_t* barrier, uint32_t&) {
        if (threadIdx.x == 0) {
            const uint32_t address = smem_address(barrier);
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(address) : "memory");
        }
    }
    __device__ static __forceinline__ void once(uint64_t* barrier, uint32_t&) {
        if (threadIdx.x == 0) {
            const uint32_t address = smem_address(barrier);
            asm volatile("mbarrier.inval.shared::cta.b64 [%0]; mbarrier.init.shared::cta.b64 [%0], 1;"
                         :: "r"(address) : "memory");
        }
    }
};

struct MbarrierExpectTx {
    static constexpr const char* kName = "mbarrier_expect_tx";
    static constexpr const char* kOpcode = "mbarrier.expect_tx.shared::cta.b64";
    static constexpr const char* kProtocol = "single elected thread; each EXPECT_TX is balanced by COMPLETE_TX without advancing the phase";
    static constexpr int kThreads = 32;
    static constexpr int kCycleLeaders = 1;
    static constexpr double kBlockOperations = 1.0;
    __device__ static __forceinline__ void setup(uint64_t* barrier, uint32_t&) {
        if (threadIdx.x == 0) {
            const uint32_t address = smem_address(barrier);
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(address) : "memory");
        }
    }
    __device__ static __forceinline__ void once(uint64_t* barrier, uint32_t&) {
        if (threadIdx.x == 0) {
            const uint32_t address = smem_address(barrier);
            asm volatile(
                "mbarrier.expect_tx.shared::cta.b64 [%0], 16;"
                "mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [%0], 16;"
                :: "r"(address) : "memory");
        }
    }
};

template <int Waiters>
struct MbarrierWait {
    static_assert(Waiters == 128 || Waiters == 256);
    static constexpr const char* kName =
        Waiters == 128 ? "mbarrier_wait_128"
                       : "mbarrier_wait_256";
    static constexpr const char* kOpcode = "mbarrier.try_wait.parity.shared::cta.b64";
    static constexpr const char* kProtocol =
        Waiters == 128
            ? "elected-thread ARRIVE, 128 ready TRY_WAIT participants, then CTA synchronization"
            : "elected-thread ARRIVE, 256 ready TRY_WAIT participants, then CTA synchronization";
    static constexpr int kThreads = Waiters;
    static constexpr int kCycleLeaders = Waiters;
    static constexpr double kBlockOperations = static_cast<double>(Waiters);
    __device__ static __forceinline__ void setup(uint64_t* barrier, uint32_t& phase) {
        phase = 0;
        if (threadIdx.x == 0) {
            const uint32_t address = smem_address(barrier);
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(address) : "memory");
        }
    }
    __device__ static __forceinline__ void once(uint64_t* barrier, uint32_t& phase) {
        const uint32_t address = smem_address(barrier);
        if (threadIdx.x == 0) {
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(address) : "memory");
        }
        uint32_t complete;
        do {
            asm volatile(
                "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.b32 %0, 1, 0, p; }"
                : "=r"(complete) : "r"(address), "r"(phase) : "memory");
        } while (complete == 0);
        phase ^= 1U;
        asm volatile("bar.sync 0;" ::: "memory");
    }
};

#define MB_SYNC_REPEAT_8(expression) \
    expression; expression; expression; expression; \
    expression; expression; expression; expression

__device__ __forceinline__ void loop_control_baseline(uint32_t& state) {
    asm volatile("" : "+r"(state) : : "memory");
}

template <class Atom, bool RecordCycles, bool Target = true>
__global__ __launch_bounds__(Atom::kThreads)
void kernel(uint64_t* cycles, uint64_t* sink, int iters, uint32_t runtime_state) {
    __shared__ alignas(8) uint64_t barrier;
    uint32_t state = runtime_state;
    Atom::setup(&barrier, state);
    asm volatile("bar.sync 0;" ::: "memory");
    const uint64_t start = microbench::read_clock64();
#pragma unroll 1
    for (int iteration = 0; iteration < iters; ++iteration) {
        if constexpr (Target) {
            MB_SYNC_REPEAT_8(Atom::once(&barrier, state));
        } else {
            MB_SYNC_REPEAT_8(loop_control_baseline(state));
        }
    }
    const uint64_t stop = microbench::read_clock64();
    asm volatile("bar.sync 0;" ::: "memory");
    if constexpr (RecordCycles) {
        if (threadIdx.x == 0) {
            cycles[0] = stop - start;
        }
        if constexpr (Atom::kCycleLeaders == 2) {
            if (threadIdx.x == 128) {
                cycles[1] = stop - start;
            }
        } else if constexpr (Atom::kCycleLeaders == Atom::kThreads) {
            cycles[threadIdx.x] = stop - start;
        }
    }
    if (threadIdx.x == 0) {
        sink[blockIdx.x] = stop ^ static_cast<uint64_t>(state + 1U);
    }
}

#undef MB_SYNC_REPEAT_8

inline void validate_sink(const std::vector<uint64_t>& values) {
    for (const uint64_t value : values) {
        if (value == 0) {
            throw std::runtime_error("synchronization sink is zero");
        }
    }
}

template <class Atom>
int run(int argc, char** argv) {
    try {
        const Args args(argc, argv);
        args.require_only({"iters", "warmup", "samples", "blocks", "peak", "device"});
        const CommonOptions options = parse_common_options(args, 2048);
        const cudaDeviceProp device = require_sm90(options.device);
        const int blocks = resolve_blocks(options.blocks, device, 2);

        DeviceBuffer<uint64_t> cycles(Atom::kCycleLeaders);
        DeviceBuffer<uint64_t> baseline_cycles(Atom::kCycleLeaders);
        DeviceBuffer<uint64_t> latency_sink(1);
        DeviceBuffer<uint64_t> baseline_latency_sink(1);
        DeviceBuffer<uint64_t> throughput_sink(blocks);
        std::vector<double> target_cycle_samples;
        std::vector<double> baseline_cycle_samples;
        if constexpr (SassElision<Atom>::kElided) {
            target_cycle_samples = measure_clock_cycles(
                options.warmup, options.samples, cycles.data(), [&] {
                    kernel<Atom, true, true><<<1, Atom::kThreads>>>(
                        cycles.data(), latency_sink.data(), options.iters,
                        0xffffffffU);
                }, Atom::kCycleLeaders);
        } else {
            auto paired = measure_paired_clock_cycles(
                options.warmup, options.samples, cycles.data(),
                baseline_cycles.data(), Atom::kCycleLeaders, [&] {
                    kernel<Atom, true, true><<<1, Atom::kThreads>>>(
                        cycles.data(), latency_sink.data(), options.iters,
                        0xffffffffU);
                    kernel<Atom, true, false><<<1, Atom::kThreads>>>(
                        baseline_cycles.data(), baseline_latency_sink.data(),
                        options.iters, 0xffffffffU);
                });
            target_cycle_samples = std::move(paired.target);
            baseline_cycle_samples = std::move(paired.baseline);
        }
        validate_sink(latency_sink.copy_to_host());
        if constexpr (!SassElision<Atom>::kElided) {
            validate_sink(baseline_latency_sink.copy_to_host());
        }
        const double raw_cycles = median(target_cycle_samples);
        const double baseline_raw_cycles = baseline_cycle_samples.empty()
            ? 0.0
            : median(baseline_cycle_samples);
        const double latency_operations = static_cast<double>(options.iters) * kUnroll;
        std::vector<double> latency_per_op_samples;
        latency_per_op_samples.reserve(target_cycle_samples.size());
        for (std::size_t index = 0; index < target_cycle_samples.size(); ++index) {
            latency_per_op_samples.push_back(
                SassElision<Atom>::kElided ? 0.0
                    : (target_cycle_samples[index] -
                       baseline_cycle_samples[index]) /
                        latency_operations);
        }
        const double cycles_per_operation =
            median(latency_per_op_samples);

        const auto event_samples = measure_event_ms(
            options.warmup, options.samples, [&] {
                kernel<Atom, false, true><<<blocks, Atom::kThreads>>>(
                    nullptr, throughput_sink.data(), options.iters,
                    0xffffffffU);
            });
        const double elapsed_ms = median(event_samples);
        validate_sink(throughput_sink.copy_to_host());
        const double target_operations = static_cast<double>(blocks) *
            options.iters * kUnroll * Atom::kBlockOperations;
        const double throughput_gops = target_operations / (elapsed_ms * 1.0e6);
        std::vector<double> throughput_samples;
        throughput_samples.reserve(event_samples.size());
        for (const double sample_ms : event_samples) {
            throughput_samples.push_back(target_operations / (sample_ms * 1.0e6));
        }

        JsonObject params;
        params.add("iters", options.iters)
            .add("warmup", options.warmup)
            .add("samples", options.samples)
            .add("blocks", options.blocks)
            .add("resolved_blocks", blocks)
            .add("threads", Atom::kThreads)
            .add("unroll", kUnroll)
            .add("target_ptx", Atom::kOpcode)
            .add("protocol", Atom::kProtocol)
            .add("sass_elided", SassElision<Atom>::kElided)
            .add("initiation_interval_cycles", cycles_per_operation)
            .add("clock_baseline",
                 SassElision<Atom>::kElided
                     ? "target is elided in SASS; formal cost is zero"
                     : "same unroll and loop with volatile register dependency; "
                       "no timed synchronization instruction")
            .add("peak", options.peak)
            .add("peak_unit", "Gtarget-op/s")
            .add("device", options.device)
            .add("gpu", device.name);
        JsonObject latency;
        latency.add("value", cycles_per_operation)
            .add("unit", "cycles/target-op")
            .add("timer", "clock64")
            .add("scope", Atom::kCycleLeaders == Atom::kThreads
                              ? "max_of_all_waiters"
                              : (Atom::kCycleLeaders == 2
                                     ? "max_of_two_wg_leaders"
                                     : "cta_leader"))
            .add("raw_median_cycles", raw_cycles)
            .add("baseline_median_cycles", baseline_raw_cycles)
            .add("operations", latency_operations)
            .add("protocol", Atom::kProtocol)
            .add_raw("samples", json_number_array(latency_per_op_samples))
            .add_raw("target_samples_cycles",
                     json_number_array(target_cycle_samples))
            .add_raw("baseline_samples_cycles",
                     json_number_array(baseline_cycle_samples));
        JsonObject throughput;
        if constexpr (SassElision<Atom>::kElided) {
            throughput.add_null("value")
                .add("unit", "Gtarget-op/s")
                .add("timer", "cuda_event")
                .add("scope", "grid_framework_only")
                .add("median_ms", elapsed_ms)
                .add("operations", 0)
                .add_raw("samples", "[]")
                .add_raw("event_samples_ms", json_number_array(event_samples))
                .add("reason", "SM90a ptxas elides converged BAR.WARP.SYNC");
        } else {
            throughput.add("value", throughput_gops)
                .add("unit", "Gtarget-op/s")
                .add("timer", "cuda_event")
                .add("scope", "grid")
                .add("median_ms", elapsed_ms)
                .add("operations", target_operations)
                .add_raw("samples", json_number_array(throughput_samples))
                .add_raw("event_samples_ms", json_number_array(event_samples));
        }
        JsonObject bandwidth;
        bandwidth.add_null("value")
            .add("unit", "GB/s")
            .add("reason", "synchronization/ordering target has no payload bytes");
        JsonObject hardware_utilization;
        if constexpr (SassElision<Atom>::kElided) {
            hardware_utilization.add_null("value")
                .add_null("percent")
                .add_null("peak")
                .add("unit", "ratio")
                .add("peak_unit", "Gtarget-op/s")
                .add("reason", "no executable target SASS");
        } else {
            hardware_utilization =
                utilization(throughput_gops, options.peak, "Gtarget-op/s");
        }
        print_result(Atom::kName, params, latency, throughput, bandwidth,
                     hardware_utilization);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "synchronization atomic benchmark error: " << error.what() << '\n';
        return 1;
    }
}

}  // namespace microbench::sync_atomic
