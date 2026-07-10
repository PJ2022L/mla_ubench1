---
name: kernel-analysis-skill
description: Analyze a concrete CUDA GPU kernel from source, launch code, PTX, SASS, or profiler evidence and write its implementation document. Use for kernel path analysis, tensor/argument shapes, grid/CTA partitioning, warp or warp-group responsibilities, TMA/WGMMA pipelines, FlashAttention-v3-style logical timelines, synchronization, or important CUDA/Hopper instructions; supports Chinese requests such as “kernel 分析”, “CTA 切分”, “warp group 流水”, “流水线图”, and “TMA/WGMMA 指令”.
---

# CUDA Kernel Structure Analysis

Produce an evidence-based explanation of the kernel's algorithm and execution schedule. Treat Hopper-specific mechanisms as confirmed only when the source, generated PTX/SASS, or profiler evidence shows them.

When working in a repository that follows `operators/<operator>/paths/<kernel-path>/`, write or update `01-kernel-implementation.md` and place generated figures in that path's `assets/` directory. Keep atom decomposition and performance modeling in their own documents; hand off only the facts they need, such as loop counts, instruction shapes, and overlap boundaries.

## Gather evidence first

Read the kernel and its launch site before explaining it. Record the target GPU/ISA, compile target, grid/block dimensions, runtime parameters, and representative input shapes. Inspect generated PTX/SASS if the question concerns actual instructions or scheduling. Ask for missing source, launch configuration, or shapes when they would change the mapping.

Start the report with source anchors: launcher, kernel body, template/config, explicit instantiation, and relevant instruction helpers. Pin the upstream commit when available.

Keep three kinds of statements separate:

- **Confirmed** — explicitly visible in source, launch code, PTX/SASS, or a trace.
- **Derived** — follows mechanically from dimensions or index arithmetic; show the derivation.
- **Inference** — plausible implementation detail not directly observed; label it and state what would confirm it.

Do not claim WGMMA, TMA, asynchronous overlap, a specific warp-group count, or an instruction latency merely because the GPU is Hopper.

## Write the analysis in this order

Use the following four numbered sections. Preserve symbols used by the kernel; define any replacement symbols once.

### 0. Problem definition and tensors

State the mathematical operation first, including reductions, masking, layout transforms, and output. Then give one concise argument table:

| Argument | Type/layout | Logical shape | Indexing/stride | Role |
|---|---|---|---|---|

Distinguish logical dimensions from physical storage dimensions, and show scalar arguments separately. For dynamic-shape kernels, identify the concrete shape used for the mapping and tail predicates. Include the CTA's output tile and its reduction/input tile using equations such as `CTA(bx, by) -> C[m0:m0+BM, n0:n0+BN]`.

### 1. CTA and warp-group partition

Map `gridDim`/`blockDim` to output work, then account for every thread, warp, and warp group. Use a table rather than prose alone:

| Execution unit | Threads/warps | Tile or stage responsibility | Registers/shared memory touched | Synchronization partner |
|---|---:|---|---|---|

For Hopper warp groups, name the four-warps collectively and show their common WGMMA tile. For producer/consumer designs, identify which threads issue TMA, which wait on the barrier, and which perform tensor-core or vector work. Explain idle/helper warps explicitly. For non-Hopper kernels, use warp-level roles and do not force a warp-group interpretation.

### 2. Pipeline and overlap diagrams

Analyze the steady-state loop, not only the prologue. First describe the inter-warp-group pipeline: which unit produces, consumes, computes, reduces, or writes each stage; which buffer is live; and each dependence. Then, only when it actually exists, describe intra-warp-group overlap such as `WGMMA(i+1)` overlapping softmax/vector work for `i`.

Generate two SVG timelines when the evidence supports both levels:

1. `warpgroup-pipeline.svg`: one lane per warp group (or producer/consumer role), showing inter-group overlap.
2. `intra-warpgroup-pipeline.svg`: one lane per operation stream inside a warp group, showing staggered issue/compute/reduction work.

Use [`scripts/render_pipeline_svg.py`](scripts/render_pipeline_svg.py) with the JSON schema and visual rules in [`references/timeline-notation.md`](references/timeline-notation.md). Embed the resulting SVGs in the report. The horizontal axis is logical time: show ordering and overlap, never imply measured latency unless the timeline is trace-derived. Use blue for data movement, pink for tensor/core compute, green for vector/reduction/epilogue work, and vertical dashed lines for synchronization. Label each box with an operation and iteration/buffer index.

Place synchronization at the exact dependency boundary. Label it with the mechanism when known: `mbarrier`, `wgmma.wait_group`, `bar.sync`, `__syncthreads()`, or a data dependency. If inter-group overlap exists but intra-group overlap does not, generate only `warpgroup-pipeline.svg`; show the intra-group serial chain in text or a dependency table and cite the wait/data-dependency boundary. Do not manufacture a second SVG to imitate a reference figure.

### 3. Important instructions and their purpose

List only instructions or source intrinsics that matter to this kernel's dataflow or scheduling. For each, report its evidence level, operands/address space, scope, and pipeline consequence.

| Instruction / intrinsic | Evidence | Data path or scope | What it enables here | Dependency / caveat |
|---|---|---|---|---|

Use [`references/instruction-catalog.md`](references/instruction-catalog.md) to classify TMA, WGMMA, `mma.sync`, `cp.async`, `ldmatrix`, loads/stores, barriers, and reductions. Do not confuse a CUDA/CUTLASS API name with an emitted ISA instruction. Quote the relevant source/PTX/SASS line or give its file and line range when available.

## Finish with a compact correctness check

Verify that the explained CTA tiles cover the output exactly once (or explain atomics/reductions), every producer has a matching consumer/wait, buffer reuse occurs only after its last consumer, and the timeline labels agree with the instruction table. End with open questions or the smallest next artifact needed to resolve any inference.

Before finishing, expose the facts needed by downstream documents without designing their model:

- loop trip counts and WGMMA/TMA/load/store shapes;
- the smallest meaningful per-CTA or per-block measurement unit;
- confirmed serial chains and confirmed overlap pairs;
- prologue, steady-state, and drain boundaries;
- optional paths such as split-KV or tail handling.
