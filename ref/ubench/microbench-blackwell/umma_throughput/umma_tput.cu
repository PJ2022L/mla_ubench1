// Unified tcgen05 MMA throughput microbenchmark
// Specializes into all configuration combinations via compile-time macros:
//   - AB_LAYOUT: SS_MODE (A from SMEM) or TS_MODE (A from TMEM)
//   - MMA_FORMAT: Dense (0-3) or MX block-scaled (4-5)
//   - CTA_GROUP: 1SM or 2SM operation

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// ============================================================
// Required Configuration Macros
// ============================================================

#ifndef MMA_FORMAT
#error "MMA_FORMAT must be defined (0=BF16, 1=E4M3, 2=S8, 3=F4, 4=MXF8, 5=MXF4)"
#endif
#ifndef MMA_M
#error "MMA_M must be defined (e.g., -DMMA_M=128)"
#endif
#ifndef MMA_N
#error "MMA_N must be defined (e.g., -DMMA_N=64)"
#endif
#ifndef MMA_K
#error "MMA_K must be defined (e.g., -DMMA_K=16)"
#endif
#ifndef MMA_DEPTH
#error "MMA_DEPTH must be defined (e.g., -DMMA_DEPTH=256)"
#endif
#ifndef CTA_GROUP
#error "CTA_GROUP must be defined (1=1SM, 2=2SM)"
#endif
#ifndef AB_LAYOUT
#error "AB_LAYOUT must be defined (0=SS_MODE, 1=TS_MODE)"
#endif

// ============================================================
// Macro Constants
// ============================================================

#define SS_MODE 0    // A from SMEM, B from SMEM
#define TS_MODE 1    // A from TMEM, B from SMEM

#define MX_SCALED (MMA_FORMAT > 3)

#if MMA_FORMAT == 0
    #define MMA_KIND "f16"           // BF16
#elif MMA_FORMAT == 1 || MMA_FORMAT == 3
    #define MMA_KIND "f8f6f4"        // E4M3, F4
#elif MMA_FORMAT == 2
    #define MMA_KIND "i8"            // S8
#endif

// ============================================================
// Compile-time Validation
// ============================================================

#if MX_SCALED && MMA_M < 128
#error "MX_SCALED requires MMA_M >= 128 (M/128 descriptor encoding)"
#endif

// ============================================================
// MMA Type Traits (CUTLASS-style nested structs)
// ============================================================

enum class MMAFormat : uint8_t {
    BF16   = 0,   // BF16 × BF16 → FP32, K=16
    E4M3   = 1,   // FP8 E4M3 × E4M3 → FP32, K=32
    S8     = 2,   // INT8 × INT8 → INT32, K=32
    F4     = 3,   // FP4 E2M1 × E2M1 → FP32, K=64
    MXF8   = 4,   // MX-scaled E4M3 × E4M3 → FP32, K=32
    MXF4   = 5,   // MX-scaled E2M1 × E2M1 → FP32, K=64
};

template <MMAFormat Fmt> struct MMATraits;

// --- Dense types (0-3) ---

template <> struct MMATraits<MMAFormat::BF16> {
    struct A { using Elem = nv_bfloat16; static constexpr int Bits = 16; };
    struct B { using Elem = nv_bfloat16; static constexpr int Bits = 16; };
    struct D { using Elem = float; };
};

template <> struct MMATraits<MMAFormat::E4M3> {
    struct A { using Elem = __nv_fp8_e4m3; static constexpr int Bits = 8; };
    struct B { using Elem = __nv_fp8_e4m3; static constexpr int Bits = 8; };
    struct D { using Elem = float; }; };

template <> struct MMATraits<MMAFormat::S8> {
    struct A { using Elem = int8_t; static constexpr int Bits = 8; };
    struct B { using Elem = int8_t; static constexpr int Bits = 8; };
    struct D { using Elem = int32_t; };
};

template <> struct MMATraits<MMAFormat::F4> {
    struct A { using Elem = uint8_t; static constexpr int Bits = 4; };  // packed: 2x FP4 per byte
    struct B { using Elem = uint8_t; static constexpr int Bits = 4; };
    struct D { using Elem = float; };
};

// --- MX types (4-5) - adds SF (Scale Factor) ---

template <> struct MMATraits<MMAFormat::MXF8> {
    struct A { using Elem = __nv_fp8_e4m3; static constexpr int Bits = 8; };
    struct B { using Elem = __nv_fp8_e4m3; static constexpr int Bits = 8; };
    struct D { using Elem = float; };
    struct SF { using Elem = uint8_t; };
};

template <> struct MMATraits<MMAFormat::MXF4> {
    struct A { using Elem = uint8_t; static constexpr int Bits = 4; };  // packed: 2x FP4 per byte
    struct B { using Elem = uint8_t; static constexpr int Bits = 4; };
    struct D { using Elem = float; };
    struct SF { using Elem = uint8_t; };
};

// Alias for current configuration
using MT = MMATraits<static_cast<MMAFormat>(MMA_FORMAT)>;

// ============================================================
// Helper Functions
// ============================================================

__host__ __device__ constexpr int next_power_of_2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

template <typename T, typename U>
__host__ __device__ constexpr auto cdiv(T a, U b) {
    return (a + b - 1) / b;
}

template <typename T, typename U>
__host__ __device__ constexpr auto align_up(T x, U boundary) {
    return (x + boundary - 1) & ~(boundary - 1);
}

__device__ __forceinline__
uint64_t make_smem_desc(const void* ptr, int height) {
    int addr = static_cast<int>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0;
    desc |= (addr >> 4) & 0x3FFF;           // bits 0-13: encoded addr
    desc |= ((height * 16) >> 4) << 16;     // bits 16-29: encoded LBO
    desc |= (8ULL << 32);                   // bits 32-45: encoded SBO
    desc |= (1ULL << 46);                   // bit 46: enable flag
    return desc;
}

__device__ inline uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}"
        : "+r"(pred)
        : "r"(0xFFFFFFFF)
    );
    return pred;
}

template <int CtaGroup>
__device__ __forceinline__ void barrier_sync();

template <>
__device__ __forceinline__ void barrier_sync<1>() {
    __syncthreads();
}

template <>
__device__ __forceinline__ void barrier_sync<2>() {
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
}

// ============================================================
// Instruction Descriptor Encoding
// ============================================================

template <MMAFormat Fmt>
__device__ constexpr uint32_t make_i_desc();

template <> __device__ constexpr uint32_t make_i_desc<MMAFormat::BF16>() {
    uint32_t desc = 0;
    desc |= (1U << 4);                      // bits 4-6: dtype = FP32
    desc |= (1U << 7);                      // bits 7-9: atype = BF16
    desc |= (1U << 10);                     // bits 10-12: btype = BF16
    desc |= ((MMA_N >> 3) << 17);           // bits 17-23: N / 8
    desc |= ((MMA_M >> 4) << 24);           // bits 24-30: M / 16
    return desc;
}

template <> __device__ constexpr uint32_t make_i_desc<MMAFormat::E4M3>() {
    uint32_t desc = 0;
    desc |= (1U << 4);                      // bits 4-6: dtype = FP32
    desc |= (2U << 7);                      // bits 7-9: atype = E4M3
    desc |= (2U << 10);                     // bits 10-12: btype = E4M3
    desc |= ((MMA_N >> 3) << 17);           // bits 17-23: N / 8
    desc |= ((MMA_M >> 4) << 24);           // bits 24-30: M / 16
    return desc;
}

template <> __device__ constexpr uint32_t make_i_desc<MMAFormat::S8>() {
    uint32_t desc = 0;
    desc |= (2U << 4);                      // bits 4-6: dtype = S32
    desc |= (1U << 7);                      // bits 7-9: atype = INT8
    desc |= (1U << 10);                     // bits 10-12: btype = INT8
    desc |= ((MMA_N >> 3) << 17);           // bits 17-23: N / 8
    desc |= ((MMA_M >> 4) << 24);           // bits 24-30: M / 16
    return desc;
}

template <> __device__ constexpr uint32_t make_i_desc<MMAFormat::F4>() {
    uint32_t desc = 0;
    desc |= (1U << 4);                      // bits 4-6: dtype = FP32
    desc |= (5U << 7);                      // bits 7-9: atype = E2M1
    desc |= (5U << 10);                     // bits 10-12: btype = E2M1
    desc |= ((MMA_N >> 3) << 17);           // bits 17-23: N / 8
    desc |= ((MMA_M >> 4) << 24);           // bits 24-30: M / 16
    return desc;
}

template <> __device__ constexpr uint32_t make_i_desc<MMAFormat::MXF8>() {
    uint32_t desc = 0;
    desc |= (0U << 4);                      // bits 4-5: B scale factor ID = 0
    desc |= (0U << 7);                      // bits 7-9: atype = E4M3
    desc |= (0U << 10);                     // bits 10-12: btype = E4M3
    desc |= ((MMA_N >> 3) << 17);           // bits 17-22: N / 8
    desc |= (1U << 23);                     // bit 23: scale type = UE8M0
    desc |= ((MMA_M >> 7) << 27);           // bits 27-28: M / 128
    desc |= (0U << 29);                     // bits 29-30: A scale factor ID = 0
    return desc;
}

template <> __device__ constexpr uint32_t make_i_desc<MMAFormat::MXF4>() {
    uint32_t desc = 0;
    desc |= (0U << 4);                      // bits 4-5: B scale factor ID = 0
    desc |= (5U << 7);                      // bits 7-9: atype = E2M1
    desc |= (5U << 10);                     // bits 10-12: btype = E2M1
    desc |= ((MMA_N >> 3) << 17);           // bits 17-22: N / 8
    desc |= (1U << 23);                     // bit 23: scale type = UE8M0
    desc |= ((MMA_M >> 7) << 27);           // bits 27-28: M / 128
    desc |= (0U << 29);                     // bits 29-30: A scale factor ID = 0
    return desc;
}

// Non-zero fill: each byte = ((i + byte_idx) % 127) + 1, giving values 1-127
template <typename T>
__device__ __forceinline__ T fill_value(int i) {
    T val;
    uint8_t* p = reinterpret_cast<uint8_t*>(&val);
    #pragma unroll
    for (int b = 0; b < sizeof(T); b++)
        p[b] = static_cast<uint8_t>(((i + b) % 127) + 1);
    return val;
}

// ============================================================
// Size Calculation
// ============================================================

constexpr int MMA_M_PER_CTA = MMA_M / CTA_GROUP;
constexpr int A_SIZE = (MMA_M_PER_CTA * MMA_K) * MT::A::Bits / 8;
constexpr int B_SIZE = (MMA_N * MMA_K) * MT::B::Bits / 8;

#if MX_SCALED
constexpr int TMEM_NUM_LANES = 128;  // TMEM has 128 lanes (rows)
constexpr int SF_A_SIZE = align_up(MMA_M_PER_CTA, TMEM_NUM_LANES);
constexpr int SF_B_SIZE = align_up(MMA_N, TMEM_NUM_LANES);
#endif

// ============================================================
// Kernel
// ============================================================

__global__ __cluster_dims__(CTA_GROUP, 1, 1) __launch_bounds__(128)
void umma_tput_kernel() {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;

    int cta_rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Dynamic shared memory
    extern __shared__ __align__(128) char smem[];
    auto* A = reinterpret_cast<MT::A::Elem*>(smem);
    auto* B = reinterpret_cast<MT::B::Elem*>(smem + A_SIZE);

    // Initialize A and B
    constexpr int A_NUMEL = A_SIZE / sizeof(MT::A::Elem);
    constexpr int B_NUMEL = B_SIZE / sizeof(MT::B::Elem);
    for (int i = tid; i < A_NUMEL; i += blockDim.x)
        A[i] = fill_value<MT::A::Elem>(i + cta_rank * A_NUMEL);
    for (int i = tid; i < B_NUMEL; i += blockDim.x)
        B[i] = fill_value<MT::B::Elem>(i);

#if MX_SCALED
    // Scale factors in SMEM
    char* sf_base = smem + A_SIZE + B_SIZE;
    sf_base = reinterpret_cast<char*>(align_up(reinterpret_cast<uintptr_t>(sf_base), 128));
    auto* SF_A = reinterpret_cast<MT::SF::Elem*>(sf_base);
    auto* SF_B = reinterpret_cast<MT::SF::Elem*>(sf_base + SF_A_SIZE);

    for (int i = tid; i < SF_A_SIZE; i += blockDim.x)
        SF_A[i] = i % TMEM_NUM_LANES;
    for (int i = tid; i < SF_B_SIZE; i += blockDim.x)
        SF_B[i] = i % TMEM_NUM_LANES;
#endif

    barrier_sync<CTA_GROUP>();


    // Static shared memory for mbar and tmem_addr
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ uint64_t mbar;
    __shared__ int tmem_addr;

    // Initialize mbarrier
    const int mbar_addr = static_cast<int>(__cvta_generic_to_shared(&mbar));
    if (warp_id == 0 && elect_sync()) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                    :: "r"(mbar_addr), "r"(1));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    // ----------------------------------------------------------------
    // TMEM Layout: [D][A if TS][SF if MX]
    // Cursor accumulates through regions per CLAUDE.md
    // ----------------------------------------------------------------
    int tmem_cols = MMA_N;

#if AB_LAYOUT == TS_MODE
    // A region (TS mode only)
    const int tmem_a_offset = tmem_cols;
    tmem_cols += 8;
#endif

#if MX_SCALED
    // SF region (MX mode only)
    const int tmem_sf_a_offset = tmem_cols;
    const int tmem_sf_b_offset = tmem_cols + 4;
    tmem_cols += 8;
#endif

    // Alloc requires power-of-2, minimum 32
    const int tmem_alloc_cols = next_power_of_2(tmem_cols < 32 ? 32 : tmem_cols);

    // Allocate TMEM (all threads in warp 0)
    if (warp_id == 0) {
        const int tmem_addr_smem = static_cast<int>(__cvta_generic_to_shared(&tmem_addr));
        asm volatile("tcgen05.alloc.cta_group::%2.sync.aligned.shared::cta.b32 [%0], %1;"
                    :: "r"(tmem_addr_smem), "r"(tmem_alloc_cols), "n"(CTA_GROUP));
    }
    barrier_sync<CTA_GROUP>();

    // Compute TMEM region addresses from base
    const uint32_t tmem_d = tmem_addr;

#if AB_LAYOUT == TS_MODE
    const uint32_t tmem_a = tmem_addr + tmem_a_offset;

    // Copy A from SMEM to TMEM (each row = 256 bits = 32 bytes)
    constexpr int A_ROW_BYTES = MMA_K * MT::A::Bits / 8;
    static_assert(A_ROW_BYTES == 32, "Each A row should be 256 bits = 32 bytes");

    uint32_t a_regs[8];
    if (tid >= MMA_M) {
        for (int i = 0; i < 8; i++) a_regs[i] = 0;
    } else {
        const uint32_t* row = reinterpret_cast<const uint32_t*>(
            reinterpret_cast<const char*>(A) + tid * A_ROW_BYTES);
        for (int i = 0; i < 8; i++) a_regs[i] = row[i];
    }
    asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "r"(tmem_a),
                   "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                   "r"(a_regs[4]), "r"(a_regs[5]), "r"(a_regs[6]), "r"(a_regs[7]));
#endif

#if MX_SCALED
    const uint32_t tmem_sf_a = tmem_addr + tmem_sf_a_offset;
    const uint32_t tmem_sf_b = tmem_addr + tmem_sf_b_offset;

    // Copy scale factors from SMEM to TMEM
    uint32_t sf_a_val = (uint32_t)SF_A[tid];
    uint32_t sf_b_val = (uint32_t)SF_B[tid];
    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};"
                :: "r"(tmem_sf_a), "r"(sf_a_val));
    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};"
                :: "r"(tmem_sf_b), "r"(sf_b_val));
#endif

    barrier_sync<CTA_GROUP>();

    // Build descriptors
    constexpr uint32_t i_desc = make_i_desc<static_cast<MMAFormat>(MMA_FORMAT)>();
    uint64_t b_desc = make_smem_desc(B, MMA_N);

#if AB_LAYOUT == SS_MODE
    uint64_t a_desc = make_smem_desc(A, MMA_M_PER_CTA);
#endif

    // Define MMA lambda based on mode
#if AB_LAYOUT == SS_MODE && !MX_SCALED
    // SS Dense
    auto mma = [&](int pred) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %5, 0;\n\t"
            "tcgen05.mma.cta_group::%4.kind::" MMA_KIND " [%0], %1, %2, %3, p;\n\t"
            "}"
            :: "r"(tmem_d), "l"(a_desc), "l"(b_desc), "r"(i_desc), "n"(CTA_GROUP), "r"(pred)
        );
    };

#elif AB_LAYOUT == SS_MODE && MX_SCALED
    // SS MX
    auto mma = [&](int pred) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %7, 0;\n\t"
            "tcgen05.mma.cta_group::%6.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%4], [%5], p;\n\t"
            "}"
            :: "r"(tmem_d), "l"(a_desc), "l"(b_desc), "r"(i_desc),
               "r"(tmem_sf_a), "r"(tmem_sf_b), "n"(CTA_GROUP), "r"(pred)
        );
    };

#elif AB_LAYOUT == TS_MODE && !MX_SCALED && CTA_GROUP == 1
    // TS Dense 1SM: 4-element negation vector
    auto mma = [&](int pred) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %9, 0;\n\t"
            "tcgen05.mma.cta_group::%8.kind::" MMA_KIND " [%0], [%1], %2, %3, {%4, %5, %6, %7}, p;\n\t"
            "}"
            :: "r"(tmem_d), "r"(tmem_a), "l"(b_desc), "r"(i_desc),
               "r"(0), "r"(0), "r"(0), "r"(0), "n"(CTA_GROUP), "r"(pred)
        );
    };

#elif AB_LAYOUT == TS_MODE && !MX_SCALED && CTA_GROUP == 2
    // TS Dense 2SM: 8-element negation vector
    auto mma = [&](int pred) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %13, 0;\n\t"
            "tcgen05.mma.cta_group::%12.kind::" MMA_KIND " [%0], [%1], %2, %3, {%4, %5, %6, %7, %8, %9, %10, %11}, p;\n\t"
            "}"
            :: "r"(tmem_d), "r"(tmem_a), "l"(b_desc), "r"(i_desc),
               "r"(0), "r"(0), "r"(0), "r"(0), "r"(0), "r"(0), "r"(0), "r"(0),
               "n"(CTA_GROUP), "r"(pred)
        );
    };

#elif AB_LAYOUT == TS_MODE && MX_SCALED
    // TS MX
    auto mma = [&](int pred) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %7, 0;\n\t"
            "tcgen05.mma.cta_group::%6.kind::mxf8f6f4.block_scale [%0], [%1], %2, %3, [%4], [%5], p;\n\t"
            "}"
            :: "r"(tmem_d), "r"(tmem_a), "l"(b_desc), "r"(i_desc),
               "r"(tmem_sf_a), "r"(tmem_sf_b), "n"(CTA_GROUP), "r"(pred)
        );
    };

#endif

    // Timing loop
    constexpr int NUM_ITERS = 1000;
    uint64_t start_clock, end_clock;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start_clock));

    for (int iter = 0, phase = 0; iter < NUM_ITERS; iter++) {
        // Warp 0 issues MMA (CTA 0 only in 2SM mode)
        if (cta_rank == 0 && warp_id == 0 && elect_sync()) {
            // Prime MMA (pred=false)
            mma(0);

            // Pipelined MMAs (pred=true)
            #pragma unroll
            for (int m = 1; m < MMA_DEPTH; m++) {
                mma(1);
            }

            // Commit with multicast
            const uint16_t cta_mask = (1 << CTA_GROUP) - 1;
            asm volatile("tcgen05.commit.cta_group::%2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                        :: "r"(mbar_addr), "h"(cta_mask), "n"(CTA_GROUP) : "memory");
        }

        // Wait for MMA completion
        asm volatile(
            "{\n\t"
            ".reg .pred P1;\n\t"
            "LAB_WAIT:\n\t"
            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1;\n\t"
            "@P1 bra.uni DONE;\n\t"
            "bra.uni LAB_WAIT;\n\t"
            "DONE:\n\t"
            "}"
            :: "r"(mbar_addr), "r"(phase)
        );
        phase ^= 1;
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end_clock));

    // Fence before dealloc
    asm volatile("tcgen05.fence::after_thread_sync;");

    // Deallocate TMEM
    barrier_sync<CTA_GROUP>();
    if (warp_id == 0) {
        asm volatile("tcgen05.dealloc.cta_group::%2.sync.aligned.b32 %0, %1;"
                    :: "r"(tmem_addr), "r"(tmem_alloc_cols), "n"(CTA_GROUP));
    }

    // Print result
    if (cta_rank == 0 && warp_id == 0 && elect_sync()) {
        uint64_t cycles = end_clock - start_clock;
        uint64_t total_mmas = (uint64_t)MMA_DEPTH * NUM_ITERS;
        printf("RESULT,%d,%d,%d,%d,%llu,%llu,%.4f\n",
               MMA_M, MMA_N, MMA_K, MMA_DEPTH,
               (unsigned long long)cycles, (unsigned long long)total_mmas,
               (double)cycles / total_mmas);
    }
}


int main() {
    int SMEM_SIZE = A_SIZE + B_SIZE;
#if MX_SCALED
    SMEM_SIZE += 128 + SF_A_SIZE + SF_B_SIZE;  // 128 = byte alignment padding
#endif

    umma_tput_kernel<<<CTA_GROUP, 128, SMEM_SIZE>>>();

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Done!\n");
    return 0;
}
