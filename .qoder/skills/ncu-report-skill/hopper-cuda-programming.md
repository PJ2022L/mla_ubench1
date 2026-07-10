# Hopper CUDA Kernel Programming Guide

Use this companion when a Hopper NCU diagnosis calls for a kernel redesign. It targets NVIDIA H100 and H200 (compute capability 9.0). It does not cover `tcgen05` or Tensor Memory (TMEM) programming.

## Target and compile mode

| Item | Hopper guidance |
|---|---|
| GPU | H100 or H200; inspect the profiled device for SKU-specific SM count, clock, and HBM bandwidth |
| Compute capability | 9.0 |
| General CUDA | Compile for `sm_90` |
| WGMMA, TMA, clusters | Compile for `sm_90a` |
| On-chip resources | 64 resident warps and up to 228 KB shared memory per SM; occupancy remains limited by registers, shared memory, blocks, and warps |

```bash
# Use this mode when the kernel uses Hopper-accelerated features.
nvcc -O3 -std=c++17 -lineinfo -arch=sm_90a kernel.cu -o kernel
```

Do not hard-code an H100 SXM SM count into scheduling logic or performance calculations. H100 PCIe, H100 SXM, and H200 expose different SKU characteristics; use `device__attribute_multiprocessor_count` from the report or CUDA device properties.

## Hopper mechanisms

### WGMMA

Hopper's warpgroup MMA (`wgmma`) has four participating warps. It is a good fit for regular GEMM, attention, and convolution tiles with enough K-loop work to keep multiple MMA operations in flight.

- Keep the four warpgroup warps converged through the WGMMA region.
- Stage operands in shared memory using layouts supported by the selected WGMMA shape.
- Keep accumulators in registers and account for the resulting register pressure.
- Prefer CUTLASS 3.x collective builders or cuBLASLt before writing inline PTX.
- Do not prescribe WGMMA for small, irregular, reduction, or bandwidth-bound kernels simply because tensor-core utilization is low.

Profile evidence: check tensor-pipe utilization, `wait` stalls, source/SASS containing `wgmma`, and the achieved end-to-end duration against a tuned baseline.

### TMA and `mbarrier`

Tensor Memory Accelerator (TMA) asynchronously transfers regular multidimensional tiles between global and shared memory. A typical Hopper pipeline assigns a producer warp to issue TMA operations and a consumer warpgroup to execute WGMMA after the corresponding `mbarrier` arrives.

- Use TMA only for regular, sufficiently large tiles with reuse; descriptor setup and synchronization must be amortized.
- Allocate enough shared-memory stages to overlap transfer latency with compute, then tune stage count against occupancy.
- Use `mbarrier` correctly; early consumer reads or reused stages cause correctness failures, while excess barriers show up as barrier/wait stalls.
- For small or irregular transfers, use ordinary global loads or `cp.async` instead.

Profile evidence: source/SASS containing `cp.async.bulk.tensor`, low enough `long_scoreboard` stalls, no dominant barrier stalls, and a timeline without alternating load/compute bubbles.

### Thread-block clusters and distributed shared memory

Hopper can schedule a cluster of thread blocks together. A block can access another block's shared memory through distributed shared memory (DSMEM) after cluster synchronization.

- Use a cluster only when inter-CTA tile reuse removes enough global traffic to exceed cluster scheduling and synchronization cost.
- Keep cross-CTA shared-memory accesses coalesced and aligned.
- Measure both the clustered and non-clustered versions at the same input shape; clusters can reduce launch flexibility and occupancy.
- Use CUDA cluster launch APIs and verify the requested cluster size is supported by the target device.

### FP8 Tensor Cores

Hopper supports FP8 E4M3 and E5M2 tensor-core paths. Treat FP8 as an algorithm and numerical-validation decision first, then a kernel optimization.

- Start with the Transformer Engine, cuBLASLt, or CUTLASS FP8 path when applicable.
- Validate output/error criteria at representative shapes before comparing performance.
- Include scale/conversion work in the benchmark; peak tensor throughput alone is not an end-to-end speedup.

## Programming principles

### 1. Supply enough parallel work

Aim for multiple waves of blocks across the actual device. For tiny grids, fuse adjacent work, use split-K/split-N only when reduction overhead is justified, or specialize the small problem. For decode-like workloads, report the fundamental parallelism limit instead of promising an impossible occupancy target.

### 2. Coalesce global traffic and exploit reuse

Map consecutive lanes to consecutive addresses where possible. Use shared memory as a transpose/reuse buffer when it reduces global bytes. Diagnose with sectors per request, cache hit rates, DRAM throughput, and the source lines with `long_scoreboard` stalls.

### 3. Make shared memory and synchronization deliberate

Pad or swizzle shared-memory tiles that create bank conflicts. Replace CTA-wide barriers with warp-level synchronization only when the data dependency is genuinely warp-scoped. Do not remove a barrier merely because it appears in the top stall table; first establish the producer/consumer dependency.

### 4. Budget registers before forcing occupancy

WGMMA accumulators and deep pipelines increase register pressure. Check local-load/store spill counters before applying `__launch_bounds__`, changing tile shapes, or reducing stages. Extra occupancy cannot compensate for spills to local memory.

### 5. Match compute precision and MMA path to the algorithm

Use tensor cores only for matrix-shaped work with supported layouts, precision, and enough arithmetic intensity. Use FP8 only after numerical validation. Preserve FP32/FP64 paths when their accuracy requirements make lower precision invalid.

### 6. Hide memory latency with independent work

Unroll independent loads, use multiple resident warps, and pipeline tile N+1 while computing tile N. If DRAM bandwidth is low and `long_scoreboard` dominates, the bottleneck is latency or insufficient ILP, not peak HBM bandwidth.

### 7. Use TMA and warp specialization only for a measured pipeline win

Assign producer/consumer roles only if the workload has a regular tiled transfer and enough compute to overlap it. Measure barrier stalls, tensor-pipe utilization, and end-to-end runtime. A simpler `cp.async` pipeline often wins for smaller tiles.

### 8. Reduce atomics and use clusters only for real sharing

Aggregate atomics within a warp or CTA before issuing global updates. Use DSMEM clusters when it replaces repeated global loads or a material reduction stage. Otherwise, favor the simpler independent-CTA mapping.

### 9. Balance variable-size work

Sort or pack variable-length inputs, split long sequences, or use work stealing when the PM timeline shows a long tail. For tiny kernels, measure absolute tail cost before adding scheduling machinery.

## Design checklist

Before implementing a Hopper-specific redesign, answer each question with profile evidence:

1. Is the kernel compute-bound, bandwidth-bound, latency-bound, or limited by grid/tail effects?
2. Does the shape have regular tiles and enough reuse to amortize TMA and WGMMA setup?
3. Can four warps form a converged warpgroup without increasing register pressure into spills?
4. Does the shared-memory stage count preserve enough occupancy?
5. Does a cluster remove real global-memory traffic or only add synchronization?
6. Does the optimized precision meet the workload's numerical requirement?
7. Does the revised version win end-to-end on representative shapes, not only in a microbenchmark?

If any answer is unknown, collect the required profile or retain the simpler design.
