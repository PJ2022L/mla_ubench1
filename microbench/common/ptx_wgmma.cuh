#pragma once

#include <cstdint>

namespace microbench::ptx {

#if defined(MB1_WGMMA_USE_F16)
#define MB1_WGMMA_INPUT_TYPE "f16"
#else
#define MB1_WGMMA_INPUT_TYPE "bf16"
#endif

enum class GmmaLayout : uint8_t {
    kInterleave = 0,
    kSwizzle128B = 1,
    kSwizzle64B = 2,
    kSwizzle32B = 3,
};

union GmmaDescriptor {
    uint64_t value;
    struct {
        uint16_t start_address : 14;
        uint16_t : 2;
        uint16_t leading_byte_offset : 14;
        uint16_t : 2;
        uint16_t stride_byte_offset : 14;
        uint16_t : 2;
        uint8_t : 1;
        uint8_t base_offset : 3;
        uint8_t : 4;
        uint8_t : 6;
        uint8_t layout_type : 2;
    } bits;
};

static_assert(sizeof(GmmaDescriptor) == sizeof(uint64_t));

__device__ __forceinline__ uint64_t make_swizzle128_bf16_descriptor(
        const void* shared_ptr,
        uint16_t leading_byte_offset,
        uint16_t stride_byte_offset) {
    GmmaDescriptor descriptor{};
    const auto address = static_cast<uint32_t>(__cvta_generic_to_shared(shared_ptr));
    descriptor.bits.start_address = address >> 4;
    descriptor.bits.leading_byte_offset = leading_byte_offset;
    descriptor.bits.stride_byte_offset = stride_byte_offset;
    descriptor.bits.base_offset = 0;
    descriptor.bits.layout_type = static_cast<uint8_t>(GmmaLayout::kSwizzle128B);
    return descriptor.value;
}

// Exact descriptors emitted by make_gmma_desc for FlashMLA dense's BF16
// Layout_K_SW128_Atom and its transposed SmemLayoutV view. Offsets are in 16B.
__device__ __forceinline__ uint64_t make_dense_k_bf16_descriptor(
        const void* shared_ptr) {
    return make_swizzle128_bf16_descriptor(shared_ptr, 1, 64);
}

__device__ __forceinline__ uint64_t make_dense_v_bf16_descriptor(
        const void* shared_ptr) {
    return make_swizzle128_bf16_descriptor(shared_ptr, 512, 64);
}

#define MB1_ACCUM_32 \
    "{d0, d1, d2, d3, d4, d5, d6, d7, " \
    "d8, d9, d10, d11, d12, d13, d14, d15, " \
    "d16, d17, d18, d19, d20, d21, d22, d23, " \
    "d24, d25, d26, d27, d28, d29, d30, d31}"

#define MB1_ACCUM_128 \
    "{d0, d1, d2, d3, d4, d5, d6, d7, " \
    "d8, d9, d10, d11, d12, d13, d14, d15, " \
    "d16, d17, d18, d19, d20, d21, d22, d23, " \
    "d24, d25, d26, d27, d28, d29, d30, d31, " \
    "d32, d33, d34, d35, d36, d37, d38, d39, " \
    "d40, d41, d42, d43, d44, d45, d46, d47, " \
    "d48, d49, d50, d51, d52, d53, d54, d55, " \
    "d56, d57, d58, d59, d60, d61, d62, d63, " \
    "d64, d65, d66, d67, d68, d69, d70, d71, " \
    "d72, d73, d74, d75, d76, d77, d78, d79, " \
    "d80, d81, d82, d83, d84, d85, d86, d87, " \
    "d88, d89, d90, d91, d92, d93, d94, d95, " \
    "d96, d97, d98, d99, d100, d101, d102, d103, " \
    "d104, d105, d106, d107, d108, d109, d110, d111, " \
    "d112, d113, d114, d115, d116, d117, d118, d119, " \
    "d120, d121, d122, d123, d124, d125, d126, d127}"

#define MB1_WGMMA_QK_SS \
    "wgmma.mma_async.sync.aligned.m64n64k16.f32." \
    MB1_WGMMA_INPUT_TYPE "." MB1_WGMMA_INPUT_TYPE " " \
    MB1_ACCUM_32 ", %3, %4, p, 1, 1, 0, 0;\n\t"

#define MB1_WGMMA_QK_RS \
    "wgmma.mma_async.sync.aligned.m64n64k16.f32." \
    MB1_WGMMA_INPUT_TYPE "." MB1_WGMMA_INPUT_TYPE " " \
    MB1_ACCUM_32 ", {%4, %5, %6, %7}, %3, " \
    "p, 1, 1, 0;\n\t"

#define MB1_WGMMA_PV_SS \
    "wgmma.mma_async.sync.aligned.m64n256k16.f32." \
    MB1_WGMMA_INPUT_TYPE "." MB1_WGMMA_INPUT_TYPE " " \
    MB1_ACCUM_128 ", %3, %4, p, 1, 1, 0, 1;\n\t"

#define MB1_WGMMA_PV_RS \
    "wgmma.mma_async.sync.aligned.m64n256k16.f32." \
    MB1_WGMMA_INPUT_TYPE "." MB1_WGMMA_INPUT_TYPE " " \
    MB1_ACCUM_128 ", {%4, %5, %6, %7}, %3, " \
    "p, 1, 1, 1;\n\t"

#define MB1_COMMITTED_GROUP_1(INSTRUCTION) \
    INSTRUCTION \
    "wgmma.commit_group.sync.aligned;\n\t"

#define MB1_COMMITTED_GROUP_4(INSTRUCTION) \
    INSTRUCTION INSTRUCTION INSTRUCTION INSTRUCTION \
    "wgmma.commit_group.sync.aligned;\n\t"

#define MB1_ISSUE_DEPTH_1(GROUP) \
    GROUP \
    "wgmma.wait_group.sync.aligned 0;\n\t"

#define MB1_ISSUE_DEPTH_2(GROUP) \
    GROUP \
    "wgmma.wait_group.sync.aligned 1;\n\t"

#define MB1_ISSUE_DEPTH_4(GROUP) \
    GROUP \
    "wgmma.wait_group.sync.aligned 3;\n\t"

#define MB1_DRAIN_DEPTH_1 ""
#define MB1_DRAIN_DEPTH_2 "wgmma.wait_group.sync.aligned 0;\n\t"
#define MB1_DRAIN_DEPTH_4 "wgmma.wait_group.sync.aligned 0;\n\t"

#define MB1_WGMMA_STATIC_LOOP( \
        INSTRUCTION, GROUP_MACRO, ISSUE_MACRO, DRAIN_MACRO, \
        LABEL_PREFIX, ITERATIONS) \
    "wgmma.fence.sync.aligned;\n\t" \
    "setp.ne.u32 p, 0, 0;\n\t" \
    MB1_COMMITTED_GROUP_1(INSTRUCTION) \
    "wgmma.wait_group.sync.aligned 0;\n\t" \
    "setp.ne.u32 p, 1, 0;\n\t" \
    "mov.u64 %0, %%clock64;\n\t" \
    "mov.u32 outer, " ITERATIONS ";\n\t" \
    LABEL_PREFIX "_outer:\n\t" \
    ISSUE_MACRO(GROUP_MACRO(INSTRUCTION)) \
    "add.u32 outer, outer, -1;\n\t" \
    "setp.ne.u32 loop_pred, outer, 0;\n\t" \
    "@loop_pred bra.uni " LABEL_PREFIX "_outer;\n\t" \
    DRAIN_MACRO \
    "mov.u64 %1, %%clock64;\n\t" \
    "mov.f32 %2, d0;\n\t"

#define MB1_RUN_SS_CASE( \
        REGISTER_DECL, INSTRUCTION, GROUP_MACRO, ISSUE_MACRO, \
        DRAIN_MACRO, LABEL_PREFIX) \
    asm volatile( \
        "{\n\t" \
        ".reg .pred p, loop_pred;\n\t" \
        ".reg .u32 outer;\n\t" \
        REGISTER_DECL \
        MB1_WGMMA_STATIC_LOOP( \
            INSTRUCTION, GROUP_MACRO, ISSUE_MACRO, DRAIN_MACRO, \
            LABEL_PREFIX, "%5") \
        "}\n" \
        : "=l"(start), "=l"(stop), "=f"(sink) \
        : "l"(desc_a), "l"(desc_b), "r"(iterations) \
        : "memory")

#define MB1_RUN_RS_CASE( \
        REGISTER_DECL, INSTRUCTION, GROUP_MACRO, ISSUE_MACRO, \
        DRAIN_MACRO, LABEL_PREFIX) \
    asm volatile( \
        "{\n\t" \
        ".reg .pred p, loop_pred;\n\t" \
        ".reg .u32 outer;\n\t" \
        REGISTER_DECL \
        MB1_WGMMA_STATIC_LOOP( \
            INSTRUCTION, GROUP_MACRO, ISSUE_MACRO, DRAIN_MACRO, \
            LABEL_PREFIX, "%8") \
        "}\n" \
        : "=l"(start), "=l"(stop), "=f"(sink) \
        : "l"(desc_b), "r"(a0), "r"(a1), "r"(a2), "r"(a3), \
          "r"(iterations) \
        : "memory")

template <int GroupSize, int Depth>
__device__ __forceinline__ void run_qk_ss(
        uint64_t desc_a,
        uint64_t desc_b,
        uint32_t iterations,
        uint64_t& start,
        uint64_t& stop,
        float& sink) {
    static_assert(GroupSize == 1 || GroupSize == 4);
    static_assert(Depth == 1 || Depth == 2 || Depth == 4);
    if constexpr (GroupSize == 1 && Depth == 1) {
        MB1_RUN_SS_CASE(".reg .f32 d<32>;\n\t", MB1_WGMMA_QK_SS,
                        MB1_COMMITTED_GROUP_1, MB1_ISSUE_DEPTH_1,
                        MB1_DRAIN_DEPTH_1, "mb1_qk_ss_1_1");
    } else if constexpr (GroupSize == 1 && Depth == 2) {
        MB1_RUN_SS_CASE(".reg .f32 d<32>;\n\t", MB1_WGMMA_QK_SS,
                        MB1_COMMITTED_GROUP_1, MB1_ISSUE_DEPTH_2,
                        MB1_DRAIN_DEPTH_2, "mb1_qk_ss_1_2");
    } else if constexpr (GroupSize == 1 && Depth == 4) {
        MB1_RUN_SS_CASE(".reg .f32 d<32>;\n\t", MB1_WGMMA_QK_SS,
                        MB1_COMMITTED_GROUP_1, MB1_ISSUE_DEPTH_4,
                        MB1_DRAIN_DEPTH_4, "mb1_qk_ss_1_4");
    } else if constexpr (GroupSize == 4 && Depth == 1) {
        MB1_RUN_SS_CASE(".reg .f32 d<32>;\n\t", MB1_WGMMA_QK_SS,
                        MB1_COMMITTED_GROUP_4, MB1_ISSUE_DEPTH_1,
                        MB1_DRAIN_DEPTH_1, "mb1_qk_ss_4_1");
    } else if constexpr (GroupSize == 4 && Depth == 2) {
        MB1_RUN_SS_CASE(".reg .f32 d<32>;\n\t", MB1_WGMMA_QK_SS,
                        MB1_COMMITTED_GROUP_4, MB1_ISSUE_DEPTH_2,
                        MB1_DRAIN_DEPTH_2, "mb1_qk_ss_4_2");
    } else {
        MB1_RUN_SS_CASE(".reg .f32 d<32>;\n\t", MB1_WGMMA_QK_SS,
                        MB1_COMMITTED_GROUP_4, MB1_ISSUE_DEPTH_4,
                        MB1_DRAIN_DEPTH_4, "mb1_qk_ss_4_4");
    }
}

template <int GroupSize, int Depth>
__device__ __forceinline__ void run_qk_rs(
        uint64_t desc_b,
        uint32_t a0,
        uint32_t a1,
        uint32_t a2,
        uint32_t a3,
        uint32_t iterations,
        uint64_t& start,
        uint64_t& stop,
        float& sink) {
    static_assert(GroupSize == 1 || GroupSize == 4);
    static_assert(Depth == 1 || Depth == 2 || Depth == 4);
    if constexpr (GroupSize == 1 && Depth == 1) {
        MB1_RUN_RS_CASE(".reg .f32 d<32>;\n\t", MB1_WGMMA_QK_RS,
                        MB1_COMMITTED_GROUP_1, MB1_ISSUE_DEPTH_1,
                        MB1_DRAIN_DEPTH_1, "mb1_qk_rs_1_1");
    } else if constexpr (GroupSize == 1 && Depth == 2) {
        MB1_RUN_RS_CASE(".reg .f32 d<32>;\n\t", MB1_WGMMA_QK_RS,
                        MB1_COMMITTED_GROUP_1, MB1_ISSUE_DEPTH_2,
                        MB1_DRAIN_DEPTH_2, "mb1_qk_rs_1_2");
    } else if constexpr (GroupSize == 1 && Depth == 4) {
        MB1_RUN_RS_CASE(".reg .f32 d<32>;\n\t", MB1_WGMMA_QK_RS,
                        MB1_COMMITTED_GROUP_1, MB1_ISSUE_DEPTH_4,
                        MB1_DRAIN_DEPTH_4, "mb1_qk_rs_1_4");
    } else if constexpr (GroupSize == 4 && Depth == 1) {
        MB1_RUN_RS_CASE(".reg .f32 d<32>;\n\t", MB1_WGMMA_QK_RS,
                        MB1_COMMITTED_GROUP_4, MB1_ISSUE_DEPTH_1,
                        MB1_DRAIN_DEPTH_1, "mb1_qk_rs_4_1");
    } else if constexpr (GroupSize == 4 && Depth == 2) {
        MB1_RUN_RS_CASE(".reg .f32 d<32>;\n\t", MB1_WGMMA_QK_RS,
                        MB1_COMMITTED_GROUP_4, MB1_ISSUE_DEPTH_2,
                        MB1_DRAIN_DEPTH_2, "mb1_qk_rs_4_2");
    } else {
        MB1_RUN_RS_CASE(".reg .f32 d<32>;\n\t", MB1_WGMMA_QK_RS,
                        MB1_COMMITTED_GROUP_4, MB1_ISSUE_DEPTH_4,
                        MB1_DRAIN_DEPTH_4, "mb1_qk_rs_4_4");
    }
}

template <int GroupSize, int Depth>
__device__ __forceinline__ void run_pv_ss(
        uint64_t desc_a,
        uint64_t desc_b,
        uint32_t iterations,
        uint64_t& start,
        uint64_t& stop,
        float& sink) {
    static_assert(GroupSize == 1 || GroupSize == 4);
    static_assert(Depth == 1 || Depth == 2 || Depth == 4);
    if constexpr (GroupSize == 1 && Depth == 1) {
        MB1_RUN_SS_CASE(".reg .f32 d<128>;\n\t", MB1_WGMMA_PV_SS,
                        MB1_COMMITTED_GROUP_1, MB1_ISSUE_DEPTH_1,
                        MB1_DRAIN_DEPTH_1, "mb1_pv_ss_1_1");
    } else if constexpr (GroupSize == 1 && Depth == 2) {
        MB1_RUN_SS_CASE(".reg .f32 d<128>;\n\t", MB1_WGMMA_PV_SS,
                        MB1_COMMITTED_GROUP_1, MB1_ISSUE_DEPTH_2,
                        MB1_DRAIN_DEPTH_2, "mb1_pv_ss_1_2");
    } else if constexpr (GroupSize == 1 && Depth == 4) {
        MB1_RUN_SS_CASE(".reg .f32 d<128>;\n\t", MB1_WGMMA_PV_SS,
                        MB1_COMMITTED_GROUP_1, MB1_ISSUE_DEPTH_4,
                        MB1_DRAIN_DEPTH_4, "mb1_pv_ss_1_4");
    } else if constexpr (GroupSize == 4 && Depth == 1) {
        MB1_RUN_SS_CASE(".reg .f32 d<128>;\n\t", MB1_WGMMA_PV_SS,
                        MB1_COMMITTED_GROUP_4, MB1_ISSUE_DEPTH_1,
                        MB1_DRAIN_DEPTH_1, "mb1_pv_ss_4_1");
    } else if constexpr (GroupSize == 4 && Depth == 2) {
        MB1_RUN_SS_CASE(".reg .f32 d<128>;\n\t", MB1_WGMMA_PV_SS,
                        MB1_COMMITTED_GROUP_4, MB1_ISSUE_DEPTH_2,
                        MB1_DRAIN_DEPTH_2, "mb1_pv_ss_4_2");
    } else {
        MB1_RUN_SS_CASE(".reg .f32 d<128>;\n\t", MB1_WGMMA_PV_SS,
                        MB1_COMMITTED_GROUP_4, MB1_ISSUE_DEPTH_4,
                        MB1_DRAIN_DEPTH_4, "mb1_pv_ss_4_4");
    }
}

template <int GroupSize, int Depth>
__device__ __forceinline__ void run_pv_rs(
        uint64_t desc_b,
        uint32_t a0,
        uint32_t a1,
        uint32_t a2,
        uint32_t a3,
        uint32_t iterations,
        uint64_t& start,
        uint64_t& stop,
        float& sink) {
    static_assert(GroupSize == 1 || GroupSize == 4);
    static_assert(Depth == 1 || Depth == 2 || Depth == 4);
    if constexpr (GroupSize == 1 && Depth == 1) {
        MB1_RUN_RS_CASE(".reg .f32 d<128>;\n\t", MB1_WGMMA_PV_RS,
                        MB1_COMMITTED_GROUP_1, MB1_ISSUE_DEPTH_1,
                        MB1_DRAIN_DEPTH_1, "mb1_pv_rs_1_1");
    } else if constexpr (GroupSize == 1 && Depth == 2) {
        MB1_RUN_RS_CASE(".reg .f32 d<128>;\n\t", MB1_WGMMA_PV_RS,
                        MB1_COMMITTED_GROUP_1, MB1_ISSUE_DEPTH_2,
                        MB1_DRAIN_DEPTH_2, "mb1_pv_rs_1_2");
    } else if constexpr (GroupSize == 1 && Depth == 4) {
        MB1_RUN_RS_CASE(".reg .f32 d<128>;\n\t", MB1_WGMMA_PV_RS,
                        MB1_COMMITTED_GROUP_1, MB1_ISSUE_DEPTH_4,
                        MB1_DRAIN_DEPTH_4, "mb1_pv_rs_1_4");
    } else if constexpr (GroupSize == 4 && Depth == 1) {
        MB1_RUN_RS_CASE(".reg .f32 d<128>;\n\t", MB1_WGMMA_PV_RS,
                        MB1_COMMITTED_GROUP_4, MB1_ISSUE_DEPTH_1,
                        MB1_DRAIN_DEPTH_1, "mb1_pv_rs_4_1");
    } else if constexpr (GroupSize == 4 && Depth == 2) {
        MB1_RUN_RS_CASE(".reg .f32 d<128>;\n\t", MB1_WGMMA_PV_RS,
                        MB1_COMMITTED_GROUP_4, MB1_ISSUE_DEPTH_2,
                        MB1_DRAIN_DEPTH_2, "mb1_pv_rs_4_2");
    } else {
        MB1_RUN_RS_CASE(".reg .f32 d<128>;\n\t", MB1_WGMMA_PV_RS,
                        MB1_COMMITTED_GROUP_4, MB1_ISSUE_DEPTH_4,
                        MB1_DRAIN_DEPTH_4, "mb1_pv_rs_4_4");
    }
}

#undef MB1_RUN_RS_CASE
#undef MB1_RUN_SS_CASE
#undef MB1_WGMMA_STATIC_LOOP
#undef MB1_DRAIN_DEPTH_4
#undef MB1_DRAIN_DEPTH_2
#undef MB1_DRAIN_DEPTH_1
#undef MB1_ISSUE_DEPTH_4
#undef MB1_ISSUE_DEPTH_2
#undef MB1_ISSUE_DEPTH_1
#undef MB1_COMMITTED_GROUP_4
#undef MB1_COMMITTED_GROUP_1
#undef MB1_WGMMA_PV_RS
#undef MB1_WGMMA_PV_SS
#undef MB1_WGMMA_QK_RS
#undef MB1_WGMMA_QK_SS
#undef MB1_ACCUM_128
#undef MB1_ACCUM_32
#undef MB1_WGMMA_INPUT_TYPE

}  // namespace microbench::ptx
