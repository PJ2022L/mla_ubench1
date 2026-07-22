#pragma once

#include <cstdint>

namespace microbench::ptx {

constexpr uint64_t kTmaEvictNormal = 0x1000000000000000ull;
constexpr uint64_t kTmaEvictFirst = 0x12f0000000000000ull;
constexpr uint64_t kTmaEvictLast = 0x14f0000000000000ull;

__device__ __forceinline__ uint32_t shared_address(const void* pointer) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void mbarrier_init(uint64_t* barrier,
                                               uint32_t arrival_count = 1) {
    const uint32_t address = shared_address(barrier);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 : : "r"(address), "r"(arrival_count) : "memory");
}

__device__ __forceinline__ void mbarrier_init_fence() {
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(
        uint64_t* barrier,
        uint32_t transaction_bytes) {
    const uint32_t address = shared_address(barrier);
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        : : "r"(address), "r"(transaction_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_parity(uint64_t* barrier,
                                                     uint32_t phase) {
    const uint32_t address = shared_address(barrier);
    uint32_t complete = 0;
    do {
        asm volatile(
            "{\n\t"
            ".reg .pred done;\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 done, [%1], %2;\n\t"
            "selp.u32 %0, 1, 0, done;\n\t"
            "}\n"
            : "=r"(complete)
            : "r"(address), "r"(phase)
            : "memory");
    } while (complete == 0);
}

__device__ __forceinline__ void tma_load_4d(
        const void* tensor_map,
        uint64_t* barrier,
        void* destination,
        int32_t coordinate0,
        int32_t coordinate1,
        int32_t coordinate2,
        int32_t coordinate3,
        uint64_t cache_hint = kTmaEvictFirst) {
    const uint64_t descriptor = reinterpret_cast<uint64_t>(tensor_map);
    const uint32_t barrier_address = shared_address(barrier);
    const uint32_t destination_address = shared_address(destination);
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.tile."
        "mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0], [%1, {%3, %4, %5, %6}], [%2], %7;"
        :
        : "r"(destination_address), "l"(descriptor),
          "r"(barrier_address), "r"(coordinate0), "r"(coordinate1),
          "r"(coordinate2), "r"(coordinate3), "l"(cache_hint)
        : "memory");
}

__device__ __forceinline__ void async_shared_fence() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void tma_store_4d(
        const void* tensor_map,
        const void* source,
        int32_t coordinate0,
        int32_t coordinate1,
        int32_t coordinate2,
        int32_t coordinate3) {
    const uint64_t descriptor = reinterpret_cast<uint64_t>(tensor_map);
    const uint32_t source_address = shared_address(source);
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.bulk_group "
        "[%0, {%2, %3, %4, %5}], [%1];"
        :
        : "l"(descriptor), "r"(source_address),
          "r"(coordinate0), "r"(coordinate1), "r"(coordinate2),
          "r"(coordinate3)
        : "memory");
}

__device__ __forceinline__ void bulk_store_shared_to_global(
        void* destination,
        const void* source,
        uint32_t bytes) {
    const uint32_t source_address = shared_address(source);
    asm volatile(
        "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
        :
        : "l"(destination), "r"(source_address), "r"(bytes)
        : "memory");
}

__device__ __forceinline__ void bulk_commit_group() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

template <int PendingGroups>
__device__ __forceinline__ void bulk_wait_group() {
    static_assert(PendingGroups >= 0 && PendingGroups <= 7);
    asm volatile("cp.async.bulk.wait_group.read %0;"
                 : : "n"(PendingGroups) : "memory");
}

}  // namespace microbench::ptx
