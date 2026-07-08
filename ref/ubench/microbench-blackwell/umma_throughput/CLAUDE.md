# umma_throughput - Unified MMA Throughput Microbenchmark

A unified `umma_tput.cu` that specializes into all configuration combinations via compile-time macros. Driven by `benchmark.py` for sweeping formats, modes, and dimensions.

---

## File Overview

| File | Purpose |
|------|---------|
| `umma_tput.cu` | Unified kernel, parameterized by compile-time macros |
| `benchmark.py` | Python driver: compiles, runs, collects CSV results |
| `Makefile` | Build system with configurable macro defaults |

---

## High-Level Kernel Workflow

```
1. Init A, B in SMEM; MX: also init SF_A, SF_B in SMEM
   (non-zero fill via fill_value<T>)

2. barrier_sync (1SM: __syncthreads, 2SM: cluster barrier)

3. Init mbarrier (warp0 elected thread)

4. Alloc TMEM (warp0, all threads participate)
   - Cursor accumulates: D cols + A cols (TS) + SF cols (MX)
   - Alloc next_power_of_2(max(total_cols, 32))

5. Pre-MMA data setup (if needed)
   - TS mode: copy A to TMEM via tcgen05.st (32x32b.x8)
   - MX mode: copy SF_A, SF_B to TMEM via tcgen05.st (32x32b.x1)

6. barrier_sync

7. Build descriptors
   - i_desc (compile-time via make_i_desc template)
   - b_desc (SMEM descriptor for B)
   - SS mode only: a_desc (SMEM descriptor for A)

8. Timing loop (NUM_ITERS=1000 iterations)
   - Warp0 elected thread (2SM: CTA0 only):
     - Prime: mma(0) with pred=false
     - Pipeline: for m in 1..MMA_DEPTH: mma(1)
     - Commit: multicast mbarrier arrive
   - All threads: mbarrier try_wait with phase flip

9. Fence (tcgen05.fence::after_thread_sync)

10. Dealloc TMEM (warp0)

11. Print: RESULT,M,N,K,depth,cycles,total_mmas,cycles_per_mma
```

---

## Compile-Time Configuration

### Required Macros

All macros must be defined via `-D` flags at compile time.

```cpp
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
```

### Derived Macros and Constants

```cpp
#define SS_MODE 0    // A from SMEM, B from SMEM
#define TS_MODE 1    // A from TMEM, B from SMEM
#define MX_SCALED (MMA_FORMAT > 3)
```

### PTX Instruction Kind

```cpp
#if MMA_FORMAT == 0
    #define MMA_KIND "f16"           // BF16
#elif MMA_FORMAT == 1 || MMA_FORMAT == 3
    #define MMA_KIND "f8f6f4"        // E4M3, F4
#elif MMA_FORMAT == 2
    #define MMA_KIND "i8"            // S8
#endif
// MX formats (4-5) don't use MMA_KIND; their MMA dispatch
// hardcodes "mxf8f6f4.block_scale" directly in the asm string.
```

### Compile-Time Validation

```cpp
#if MX_SCALED && MMA_M < 128
#error "MX_SCALED requires MMA_M >= 128 (M/128 descriptor encoding)"
#endif
```

### Macro Summary

| Macro | Values | Description |
|-------|--------|-------------|
| `MMA_FORMAT` | 0-5 | Data type (0=BF16, 1=E4M3, 2=S8, 3=F4, 4=MXF8, 5=MXF4) |
| `MMA_M` | 64, 128, 256 | M dimension (total across CTAs) |
| `MMA_N` | 32-256 | N dimension (multiple of 8 for 1SM, 16 for 2SM) |
| `MMA_K` | 16, 32, 64 | K dimension (fixed per data type) |
| `MMA_DEPTH` | varies | Pipeline depth for pipelined MMAs |
| `CTA_GROUP` | 1, 2 | 1SM or 2SM operation |
| `AB_LAYOUT` | 0, 1 | 0=SS_MODE (A from SMEM), 1=TS_MODE (A from TMEM) |

---

## Data Type Definitions

Uses nested structs (A, B, D, SF) within MMATraits. Alias `MT` refers to current config.

### MMAFormat Enum

```cpp
enum class MMAFormat : uint8_t {
    BF16   = 0,   // BF16 x BF16 -> FP32, K=16
    E4M3   = 1,   // FP8 E4M3 x E4M3 -> FP32, K=32
    S8     = 2,   // INT8 x INT8 -> INT32, K=32
    F4     = 3,   // FP4 E2M1 x E2M1 -> FP32, K=64
    MXF8   = 4,   // MX-scaled E4M3 x E4M3 -> FP32, K=32
    MXF4   = 5,   // MX-scaled E2M1 x E2M1 -> FP32, K=64
};
```

### MMATraits Template

```cpp
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
    struct D { using Elem = float; };
};

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
    struct A { using Elem = uint8_t; static constexpr int Bits = 4; };
    struct B { using Elem = uint8_t; static constexpr int Bits = 4; };
    struct D { using Elem = float; };
    struct SF { using Elem = uint8_t; };
};

// Alias for current configuration
using MT = MMATraits<static_cast<MMAFormat>(MMA_FORMAT)>;
```

---

## Size Calculation (File Scope)

```cpp
constexpr int MMA_M_PER_CTA = MMA_M / CTA_GROUP;
constexpr int A_SIZE = (MMA_M_PER_CTA * MMA_K) * MT::A::Bits / 8;
constexpr int B_SIZE = (MMA_N * MMA_K) * MT::B::Bits / 8;

#if MX_SCALED
constexpr int TMEM_NUM_LANES = 128;  // TMEM has 128 lanes (rows)
constexpr int SF_A_SIZE = align_up(MMA_M_PER_CTA, TMEM_NUM_LANES);
constexpr int SF_B_SIZE = align_up(MMA_N, TMEM_NUM_LANES);
#endif
```

SMEM layout (dynamic shared memory):

```
[A: A_SIZE bytes][B: B_SIZE bytes][128-byte align padding][SF_A: SF_A_SIZE][SF_B: SF_B_SIZE]
                                   ^--- MX only -------------------------------------------^
```

Host-side SMEM allocation:
```cpp
int SMEM_SIZE = A_SIZE + B_SIZE;
#if MX_SCALED
SMEM_SIZE += 128 + SF_A_SIZE + SF_B_SIZE;  // 128 = alignment padding
#endif
```

---

## TMEM Layout Configuration

Layout: `[D][A if TS][SF if MX]`

Uses a cursor (`tmem_cols`) that accumulates through each region.

```cpp
// --- D region (always) ---
int tmem_cols = MMA_N;

// --- A region (TS mode only) ---
#if AB_LAYOUT == TS_MODE
const int tmem_a_offset = tmem_cols;
tmem_cols += 8;
#endif

// --- SF region (MX scaled only) ---
#if MX_SCALED
const int tmem_sf_a_offset = tmem_cols;
const int tmem_sf_b_offset = tmem_cols + 4;
tmem_cols += 8;
#endif

// --- Finalize ---
const int tmem_alloc_cols = next_power_of_2(tmem_cols < 32 ? 32 : tmem_cols);
```

**Key constraint**: `tcgen05.alloc` requires power-of-2 column count (min 32), but MMA descriptor uses actual N.

---

## Instruction Descriptor Encoding

### Summary Table

| Format | ID | dtype | atype | btype | M enc | N enc | K |
|--------|-----|-------|-------|-------|-------|-------|---|
| BF16 | 0 | 1 (FP32) | 1 (BF16) | 1 (BF16) | M/16 @24 | N/8 @17 | 16 |
| E4M3 | 1 | 1 (FP32) | 2 (E4M3) | 2 (E4M3) | M/16 @24 | N/8 @17 | 32 |
| S8 | 2 | 2 (S32) | 1 (INT8) | 1 (INT8) | M/16 @24 | N/8 @17 | 32 |
| F4 | 3 | 1 (FP32) | 5 (E2M1) | 5 (E2M1) | M/16 @24 | N/8 @17 | 64 |
| MXF8 | 4 | 0 (SF_B) | 0 (E4M3) | 0 (E4M3) | M/128 @27 | N/8 @17 | 32 |
| MXF4 | 5 | 0 (SF_B) | 5 (E2M1) | 5 (E2M1) | M/128 @27 | N/8 @17 | 64 |

### make_i_desc Implementations

```cpp
template <MMAFormat Fmt>
__device__ constexpr uint32_t make_i_desc();

// MMA_FORMAT 0: BF16
template <> __device__ constexpr uint32_t make_i_desc<MMAFormat::BF16>() {
    uint32_t desc = 0;
    desc |= (1U << 4);                      // bits 4-6: dtype = FP32
    desc |= (1U << 7);                      // bits 7-9: atype = BF16
    desc |= (1U << 10);                     // bits 10-12: btype = BF16
    desc |= ((MMA_N >> 3) << 17);           // bits 17-23: N / 8
    desc |= ((MMA_M >> 4) << 24);           // bits 24-30: M / 16
    return desc;
}

// MMA_FORMAT 4: MXF8 (representative MX example)
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

// Usage (compile-time constant)
constexpr uint32_t i_desc = make_i_desc<static_cast<MMAFormat>(MMA_FORMAT)>();
```

---

## MMA Dispatch

Lambda-based dispatch inside kernel. Each mode defines an `mma` lambda that captures descriptors from kernel scope and takes only `pred` as argument.

```cpp
#if AB_LAYOUT == SS_MODE && !MX_SCALED
    // SS Dense: a_desc (SMEM), b_desc (SMEM)
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
    // SS MX: a_desc (SMEM), b_desc (SMEM), sf_a/sf_b (TMEM)
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
    // TS Dense 1SM: 4-element negation vector (M/16 sub-tiles / 2 = 4)
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
    // TS Dense 2SM: 8-element negation vector (2x M sub-tiles across CTAs)
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
    // TS MX: tmem_a (TMEM), b_desc (SMEM), sf_a/sf_b (TMEM)
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
```

### Lambda Captures by Mode

| Mode | Captured Variables |
|------|-------------------|
| SS Dense | `tmem_d, a_desc, b_desc, i_desc` |
| SS MX | `tmem_d, a_desc, b_desc, i_desc, tmem_sf_a, tmem_sf_b` |
| TS Dense 1SM | `tmem_d, tmem_a, b_desc, i_desc` + explicit `"r"(0)` x4 |
| TS Dense 2SM | `tmem_d, tmem_a, b_desc, i_desc` + explicit `"r"(0)` x8 |
| TS MX | `tmem_d, tmem_a, b_desc, i_desc, tmem_sf_a, tmem_sf_b` |

**Key details:**
- `CTA_GROUP` uses `"n"` constraint (compile-time immediate)
- `pred` uses `"r"` constraint (runtime argument)
- TS Dense negation vector: 4 registers for `cta_group::1` (M/16/2=4 sub-tile pairs), 8 for `cta_group::2` (2x M sub-tiles across CTAs)
- Lambda captures by reference (`[&]`) for zero-cost access to kernel-scope variables

---

## Timing Loop

```cpp
constexpr int NUM_ITERS = 1000;

for (int iter = 0, phase = 0; iter < NUM_ITERS; iter++) {
    if (cta_rank == 0 && warp_id == 0 && elect_sync()) {
        mma(0);   // prime (pred=false)
        for (int m = 1; m < MMA_DEPTH; m++)
            mma(1);  // pipelined MMAs (pred=true)

        // Commit with multicast
        const uint16_t cta_mask = (1 << CTA_GROUP) - 1;
        asm volatile("tcgen05.commit.cta_group::%2.mbarrier::arrive::one"
                     ".shared::cluster.multicast::cluster.b64 [%0], %1;"
                     :: "r"(mbar_addr), "h"(cta_mask), "n"(CTA_GROUP) : "memory");
    }

    // All threads wait
    asm volatile("mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 ..."
                 :: "r"(mbar_addr), "r"(phase));
    phase ^= 1;
}
```

---

## Kernel Launch Configuration

```cpp
__global__ __cluster_dims__(CTA_GROUP, 1, 1) __launch_bounds__(128)
void umma_tput_kernel();

// Launch: CTA_GROUP blocks, 128 threads each, dynamic SMEM
umma_tput_kernel<<<CTA_GROUP, 128, SMEM_SIZE>>>();
```

---

## Kernel Output Format

```
RESULT,<M>,<N>,<K>,<depth>,<cycles>,<total_mmas>,<cycles_per_mma>
Done!
```

Where `total_mmas = MMA_DEPTH * NUM_ITERS`, `cycles_per_mma = cycles / total_mmas`.

---

## Helper Functions

```cpp
// Power-of-2 rounding (for TMEM alloc)
__host__ __device__ constexpr int next_power_of_2(int n);

// Ceiling division
template <typename T, typename U>
__host__ __device__ constexpr auto cdiv(T a, U b);

// Align up to boundary
template <typename T, typename U>
__host__ __device__ constexpr auto align_up(T x, U boundary);

// SMEM descriptor builder
__device__ __forceinline__
uint64_t make_smem_desc(const void* ptr, int height);

// Warp-level election (keeps warp converged)
__device__ inline uint32_t elect_sync();

// Cluster-aware synchronization (specializations for CTA_GROUP 1 and 2)
template <int CtaGroup>
__device__ __forceinline__ void barrier_sync();

// Non-zero fill: each byte = ((i + byte_idx) % 127) + 1, giving values 1-127
template <typename T>
__device__ __forceinline__ T fill_value(int i);
```

---

## Configuration Combinations

| AB_LAYOUT | MX_SCALED | CTA_GROUP | Description |
|-----------|-----------|-----------|-------------|
| 0 (SS) | 0 | 1 | SS dense, 1SM |
| 0 (SS) | 0 | 2 | SS dense, 2SM |
| 0 (SS) | 1 | 1 | SS MX-scaled, 1SM |
| 0 (SS) | 1 | 2 | SS MX-scaled, 2SM |
| 1 (TS) | 0 | 1 | TS dense, 1SM (4-elem negation vector) |
| 1 (TS) | 0 | 2 | TS dense, 2SM (8-elem negation vector) |
| 1 (TS) | 1 | 1 | TS MX-scaled, 1SM |
| 1 (TS) | 1 | 2 | TS MX-scaled, 2SM |

---

## Makefile

### Targets

| Target | Command | Description |
|--------|---------|-------------|
| `all` / `umma_tput.out` | `make` | Build with current macro values |
| `ptx` | `make ptx` | Generate PTX assembly |
| `run` | `make run` | Build and run |
| `prof` | `make prof` | Profile with `ncu` (utcmma metrics) |
| `clean` | `make clean` | Remove `*.out *.ptx` |

### Macro Defaults

```makefile
MMA_FORMAT ?= 0    # BF16
MMA_M ?= 128
MMA_N ?= 64
MMA_K ?= 16
MMA_DEPTH ?= 256
CTA_GROUP ?= 1
AB_LAYOUT ?= 0     # SS_MODE
```

### Manual Build

```bash
make umma_tput.out MMA_FORMAT=0 MMA_M=128 MMA_N=64 MMA_K=16 MMA_DEPTH=256 CTA_GROUP=1 AB_LAYOUT=0
```

Or directly:
```bash
nvcc -std=c++17 -gencode arch=compute_100a,code=sm_100a \
     -DMMA_FORMAT=0 -DMMA_M=128 -DMMA_N=64 -DMMA_K=16 \
     -DMMA_DEPTH=256 -DCTA_GROUP=1 -DAB_LAYOUT=0 \
     umma_tput.cu -o umma_tput.out
```

---

## benchmark.py

### Usage

```bash
python3 benchmark.py FORMAT [FORMAT ...] [options]
```

### Format IDs

| ID | Name | K | Depths | MX |
|----|------|---|--------|----|
| 0 | BF16 | 16 | 16, 32, 64, 128, 256 | no |
| 1 | E4M3 | 32 | 32, 64, 128, 256, 512 | no |
| 2 | S8 | 32 | 32, 64, 128, 256, 512 | no |
| 3 | F4 | 64 | 64, 128, 256, 512, 1024 | no |
| 4 | MXF8 | 32 | 32, 64, 128, 256, 512 | yes |
| 5 | MXF4 | 64 | 64, 128, 256, 512, 1024 | yes |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-o FILE` | `benchmark_results.csv` | Output CSV file |
| `-v` / `--verbose` | off | Print build/run commands |
| `--overwrite` | off | Overwrite CSV instead of appending |
| `--cta-group {1,2}` | both | Restrict to 1SM or 2SM |
| `--mode {ss,ts,all}` | `ss` | AB layout mode: ss, ts, or all |
| `--n-sweep START:STOP:STEP` | none | Sweep N values instead of fixed configs |

### Default (M, N) Configs

**Dense 1SM**: (64,64), (64,80), (64,96), (64,112), (64,128), (64,256), (128,64), (128,80), (128,96), (128,112), (128,128), (128,256)

**Dense 2SM**: (128,64..256), (256,64..256) (same N steps)

**MX 1SM**: (128,64..256) — M >= 128 required

**MX 2SM**: (128,64..256), (256,64..256)

### CSV Output Columns

`Format, ABLayout, CTAGroup, M, N, K, PipelineDepth, Cycles, CyclesPerMMA, FLOPsPerCycle`

- ABLayout: `SS` or `TS` (orthogonal to CTAGroup)
- CTAGroup: `1` or `2` (both supported for SS and TS)

### Examples

```bash
# All formats, all modes (SS 1SM + SS 2SM + TS 1SM)
python3 benchmark.py 0 1 2 3 4 5 --mode all -o full.csv --overwrite

# BF16 only, N sweep
python3 benchmark.py 0 --mode all --n-sweep 32:256:8 -o bf16_sweep.csv --overwrite

# MX formats, 1SM only
python3 benchmark.py 4 5 --cta-group 1 --mode all -o mx_1sm.csv --overwrite
```

---

## Benchmark Results

### Theoretical Peak

Each MMA takes `CyclesPerMMA ≈ N/2` cycles at steady state (sufficient depth, M_perCTA >= 128).
FLOPs per MMA = `2 * M * N * K`. Therefore:

```
Peak FLOPs/cycle = 2*M*N*K / (N/2) = 4*M*K

1SM (M=128): 512*K FLOPs/cycle
2SM (M=256): 1024*K FLOPs/cycle
```

| Format | K | 1SM Peak | 2SM Peak |
|--------|---|----------|----------|
| BF16 | 16 | 8,192 | 16,384 |
| E4M3 | 32 | 16,384 | 32,768 |
| S8 | 32 | 16,384 | 32,768 |
| F4 | 64 | 32,768 | 65,536 |
| MXF8 | 32 | 16,384 | 32,768 |
| MXF4 | 64 | 32,768 | 65,536 |

All formats achieve **99.5-99.9%** of peak at large N and depth. The hardware processes
all formats at the same MMA instruction rate; FLOPs/cycle differs only because K differs.

### SS vs TS: A-Descriptor Overhead

SS mode pays a fixed **~16 cycle overhead** per MMA for the SMEM A descriptor.
TS mode (A from TMEM) eliminates this overhead.

At 1SM M=128 (all formats show the same pattern):

| N | SS cyc/MMA | TS cyc/MMA | SS-TS delta | TS/SS speedup | SS % of peak |
|---|-----------|-----------|-------------|---------------|--------------|
| 64 | 48.28 | 32.25 | 16.03 | 1.50x | 66% |
| 80 | 52.28 | 40.25 | 12.04 | 1.30x | 77% |
| 96 | 56.29 | 48.25 | 8.05 | 1.17x | 85% |
| 112 | 60.30 | 56.25 | 4.05 | 1.07x | 93% |
| 128 | 64.31 | 64.25 | 0.06 | 1.00x | 99.5% |
| 256 | 128.31 | 128.25 | 0.06 | 1.00x | 99.8% |

TS reaches **99%+ efficiency at all N values**. SS is capped at ~66% at N=64 but
catches up at N>=128 where the 16-cycle overhead is amortized into N/2 total cycles.

### 1SM vs 2SM Scaling

2SM provides near-perfect 2x throughput scaling at N>=112:

| M | N | 1SM FLOPs/cyc | 2SM FLOPs/cyc | 2SM/1SM |
|---|---|--------------|--------------|---------|
| 128 | 64 | 10,860 | 20,698 | 1.91x |
| 128 | 96 | 13,970 | 26,795 | 1.92x |
| 128 | 128 | 16,305 | 32,398 | 1.99x |
| 128 | 256 | 16,344 | 32,584 | 1.99x |

(E4M3 representative; all formats identical)

### MX Scale Factor Overhead

MX formats (MXF8, MXF4) have **zero throughput overhead** compared to their dense
counterparts (E4M3, F4). Scale factor loading adds no measurable cost.

### M=64 Limitation

M=64 at 1SM hits a ~4,080 FLOPs/cycle ceiling (half of M=128 peak) regardless of N or
SS/TS mode. Both SS and TS show the same ~16-cycle overhead at M=64, suggesting a
structural pipeline bottleneck when M_perCTA < 128 (insufficient M sub-tiles to saturate
the MMA pipeline).

### Summary

1. **Use TS for N<128** — up to 1.5x speedup at N=64, converges to SS at N>=128
2. **2SM gives ~2x** with minimal overhead at N>=112
3. **MX has zero cost** — scale factors add no throughput penalty
4. **All formats equal** per-K — throughput is geometry-limited, not format-limited
5. **M=128 is the sweet spot** for 1SM; M=64 halves peak throughput
6. **Peak: 65,427 FLOPs/cycle** (F4/MXF4 2SM M=256 N=256)

---

## Reference: utcmma_throughput Implementation

See `../utcmma_throughput/` for the original split implementation:

| File | Mode | Scaling | CTA Groups |
|------|------|---------|------------|
| `utcmma_tput.cu` | SS | Dense | 1, 2 |
| `utcmma_ts_tput.cu` | TS | Dense | 1 |
| `utcmma_mx_tput.cu` | SS | MX | 1, 2 |
| `utcmma_mx_ts_tput.cu` | TS | MX | 1 |
