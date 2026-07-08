# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Single-MMA **latency** microbenchmark for NVIDIA Blackwell's `tcgen05.mma` instruction. Measures end-to-end latency (issue -> commit -> mbarrier wait) of one MMA operation, reporting the median of 100 samples in GPU clock cycles. Sibling to `../umma_throughput/` which measures pipelined throughput.

## Build and Run

Requires `nvcc` with sm_100a support (Blackwell GPU). Must run on a Blackwell GPU.

```bash
# Build with defaults (BF16, M=128, N=64, K=16, 1SM, SS mode)
make

# Build with specific configuration
make umma_lat.out MMA_FORMAT=1 MMA_M=128 MMA_N=96 MMA_K=32 CTA_GROUP=2 AB_LAYOUT=1

# Generate PTX for inspection
make ptx

# Build and run
make run

# Clean
make clean
```

### Sweep via benchmark.py

```bash
# Sweep specific formats (0=BF16, 1=E4M3, 2=S8, 3=F4, 4=MXF8, 5=MXF4)
python3 benchmark.py 0 1 --mode all -o results.csv --overwrite

# Options: --cta-group {1,2}, --mode {ss,ts,all}, --n-sweep start:stop:step, -v
```

The benchmark driver does `make clean && make` per configuration, so each run recompiles.

## Compile-Time Macros

All configuration is via `-D` flags. The Makefile exposes them as overridable variables.

| Macro | Values | Notes |
|-------|--------|-------|
| `MMA_FORMAT` | 0-5 | 0=BF16(K=16), 1=E4M3(K=32), 2=S8(K=32), 3=F4(K=64), 4=MXF8(K=32), 5=MXF4(K=64) |
| `MMA_M` | 64, 128, 256 | Total M across CTAs. MX formats require M >= 128. |
| `MMA_N` | 32-256 | Multiple of 8 (1SM) or 16 (2SM) |
| `MMA_K` | 16, 32, 64 | Fixed per format, set automatically by benchmark.py |
| `CTA_GROUP` | 1, 2 | 1SM or 2SM cluster operation |
| `AB_LAYOUT` | 0, 1 | 0=SS (A+B from SMEM), 1=TS (A from TMEM, B from SMEM) |

## Architecture (umma_lat.cu)

The kernel is a single file parameterized entirely by compile-time macros. Key structure:

1. **MMATraits template** — maps `MMA_FORMAT` to element types (A, B, D, and SF for MX formats) via template specializations. Alias `MT` = current config.

2. **Instruction descriptor** — `make_i_desc<MMAFormat>()` returns a compile-time `uint32_t` encoding dtype/atype/btype/M/N for the PTX instruction. Dense and MX formats use different bit layouts.

3. **SMEM layout** — `[A: A_SIZE][B: B_SIZE]`, plus `[128-byte pad][SF_A][SF_B]` for MX formats. Sizes derived from `MMA_M_PER_CTA = MMA_M / CTA_GROUP`.

4. **TMEM layout** — `[D: MMA_N cols][A: 8 cols if TS][SF: 8 cols if MX]`. Allocated as `next_power_of_2(max(total, 32))` columns.

5. **MMA dispatch** — Six `#if` branches define an `mma(pred)` lambda for each combination of SS/TS x Dense/MX x CTA_GROUP. TS dense has different negation vector sizes: 4 registers for 1SM, 8 for 2SM.

6. **Timing loop** — 100 iterations. Warp 0's elected thread issues one MMA, commits with multicast mbarrier arrive, all threads wait. Reports median latency.

### Difference from umma_throughput

- **No pipeline depth** (`MMA_DEPTH`): issues exactly 1 MMA per iteration instead of a pipelined burst
- **No prime pass**: no `mma(0)` with pred=false before the timed MMA
- **Median of 100** instead of total cycles over 1000 iterations
- Output: `RESULT,M,N,K,median_cycles` (no depth/FLOPs columns)

## Combining Latency and Throughput

### MMA Time Breakdown

A single MMA operation consists of:

```
|---- issue ----|---- compute (N/2 cycles) ----|---- commit + mbarrier wait ----|
```

The **latency benchmark** measures this entire critical path: issue through commit through mbarrier wait completion. The **throughput benchmark** measures the initiation interval (II) -- how often the pipeline can accept a new MMA -- by issuing pipelined bursts of D MMAs.

### Throughput Model

The throughput benchmark issues D MMAs per batch with a prime MMA (`mma(0)`, pred=false) before each burst. The total batch time is:

```
batch_time = (D - 1) * II + latency + overhead
```

Where overhead is the prime MMA cost. The first MMA takes `latency` cycles end-to-end; each subsequent MMA adds `II` cycles due to pipelining. Dividing by D:

```
CyclesPerMMA = II + (latency + overhead - II) / D = a + b/D
```

This is a smooth 1/D curve with no knee -- throughput improves at every D due to amortization of the fixed per-batch cost `b`, not pipeline filling. The pipeline fills quickly (by D ~5), but the batch overhead continues to amortize indefinitely.

### Saturated Pipeline Depth

The saturated pipeline depth (minimum in-flight MMAs to keep the pipeline full) is:

```
D_sat = ceil(latency / II)
```

The intuition: we need to issue at least 1 MMA every II cycles to keep the pipeline full, or the pipeline idles. Since each MMA occupies the pipeline for `latency` cycles, we need `ceil(latency / II)` of them overlapping so that one exits every II cycles with no gaps.

The simple approach uses the measured CyclesPerMMA at max depth as II. This works because at large D, the `b/D` residual is small enough that `ceil()` rounds to the same value as using the true asymptotic II.

Comparing simple (measured II from `tput_results_max.csv`) vs fitted (asymptotic II from `fit_results.csv`): 258 out of 264 configs give identical D_sat. The 6 that differ are all off by exactly 1, all in 2SM configs where the II is small and `ceil()` rounding is sensitive. The simple approach is sufficient for pipeline depth estimation.

### Fitting for Overhead Estimation

Fitting `CyclesPerMMA = a + b/D` across the depth sweep (via `fit_throughput.py`) yields:
- **a** = asymptotic initiation interval (true II)
- **b** = latency + overhead - II

Since `b = latency + overhead - II`, the per-batch overhead is:

```
overhead = b + a - latency
```

Where `a` comes from the fit and `latency` comes from the latency benchmark. The fit is needed to isolate this overhead; the simple approach cannot extract it.

### Results

**Initiation interval (a):** Empirically converges to N/2 within 0.02 cycles across all properly-utilized configs (1SM M>=64, 2SM M>=256). This is an empirical finding from the throughput fit; NVIDIA has not publicly documented the II = N/2 relationship. Throughput differs between formats only because K differs.

### Why N/2: Systolic Array Interpretation

The II = N/2 result connects to classic systolic array theory. In a weight-stationary systolic array, compute time depends on which matrix dimension is **spatial** (processed in parallel by the physical array) vs **temporal** (streamed through over multiple cycles).

The empirical evidence constrains the architecture:
- II is **independent of K** (BF16 K=16, E4M3 K=32, F4 K=64 all give the same II for same N) -- K is not the streaming dimension
- II is **independent of M** -- M is not the streaming dimension
- II **scales linearly with N** -- N is the streaming/temporal dimension

This implies K is mapped **spatially** (the array has enough physical MACs to process all K multiply-accumulates in parallel), while N is mapped **temporally** (output columns stream through at 2 per cycle):

```
II = N / (columns_per_cycle) = N / 2
```

In classic systolic array terms, total time for one MMA = fill + N/2 (streaming) + drain. The fill + drain overhead is a fixed pipeline latency (~11 cycles per existing microbenchmark literature), consistent with the measured latency pattern `latency = N/2 + fixed_cost`.

The "2 columns per cycle" is a property of some N-dimensional resource in the pipeline -- the compute array width, the B input port, or the D output write port. The data alone cannot distinguish which, since B (K x N) and D (M x N) both have N columns. NVIDIA has not disclosed the internal tensor core architecture.

**Overhead:** ~12 cycles constant across all formats, SS/TS modes, 1SM/2SM, and all M/N values. This is the cost of the prime MMA (pred=false) instruction issue.

**Latency:** Follows the pattern `latency = N/2 + fixed_cost`, where the fixed cost depends on mode:

| Mode | Fixed cost (cycles) | What it includes |
|------|-------------------|------------------|
| TS 1SM | 104 | issue + commit + mbarrier wait |
| SS 1SM M=64 | 120 | TS cost + 16 (SMEM A descriptor, M=64) |
| SS 1SM M=128 | 136 | TS cost + 32 (SMEM A descriptor, M=128) |
| TS 2SM M=128 | 130 | TS 1SM cost + ~26 (cluster coordination) |
| TS 2SM M=256 | 138 | TS 1SM cost + ~34 (cluster coordination) |

### Analysis Scripts

| Script | Input | Output | What it does |
|--------|-------|--------|-------------|
| `fit_throughput.py` | `lat_results_full.csv`, `tput_results_full.csv` | `fit_results.csv` | Fits `a + b/D`, computes overhead |
| `compute_pipeline_depth.py` | `lat_results_full.csv`, `tput_results_max.csv` | `pipeline_depth.csv` | Computes `ceil(latency / II)` |
| `plot_pipeline_depth.py` | `pipeline_depth.csv` | `pipeline_depth.png` | D_sat vs N for 1SM/2SM, all formats |
| `plot_1sm_vs_2sm.py` | `pipeline_depth.csv` | `1sm_vs_2sm.png` | 2SM requires ~2x pipeline depth vs 1SM |

Both scripts require numpy; run with `/path/to/microbench-blackwell/.venv/bin/python`.
