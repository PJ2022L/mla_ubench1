---
name: ncu-report-skill
description: Profile and optimize CUDA kernels specifically for NVIDIA Hopper GPUs (H100/H200, compute capability 9.0, sm_90/sm_90a) with Nsight Compute. Use when the user asks to profile a kernel, analyze an NCU report, diagnose a bottleneck, or write an optimization plan for Hopper — including Chinese requests such as "profile 一下", "为什么慢", and "ncu 报告". For official/library implementations, provide measurements-only performance analysis unless the prompt explicitly requests “性能分析+诊断” or an exact equivalent.
---

# CUDA Kernel Profiling for Hopper

**Target hardware:** NVIDIA Hopper, including H100 and H200 (compute capability 9.0). H100/H200 differ in SKU, SM count, clocks, and HBM capacity/bandwidth, so take the device attributes and measured peak values from the report; never hard-code an SM count.

Compile general Hopper code for `sm_90`. Compile kernels that use Hopper-only instructions or features such as WGMMA, TMA, or thread-block clusters for `sm_90a` (for example, `nvcc -arch=sm_90a -lineinfo ...`).

---

## Select the output mode before profiling

Treat an implementation as **official** when the user identifies it as official or it is an NVIDIA/library/framework implementation (for example cuBLASLt, cuDNN, CUTLASS, TensorRT, or a framework-supplied kernel).

| Target and prompt | Required output |
|---|---|
| Official implementation; prompt does **not** explicitly say `性能分析+诊断` or `performance analysis + diagnosis` | **Performance analysis only.** Report measured duration, launch geometry, throughput, occupancy, tensor-core utilization, memory/cache metrics, and comparisons. Do not collect source-level stalls, infer a root cause, label a bottleneck, use the diagnosis playbook, or propose code changes. |
| Official implementation; prompt explicitly requests `性能分析+诊断` (or the exact English equivalent) | Run the complete profiling, diagnosis, and optimization workflow. |
| Non-official/custom implementation | Follow the user request; use the complete workflow when a diagnosis or optimization plan is requested. |

Treat “为什么慢”, “优化”, or “看一下 NCU 报告” alone as insufficient to enable diagnosis for an official implementation. Ask the user to request `性能分析+诊断` if diagnostic output is needed.

---

## Golden rule

**Profile → Diagnose → Plan, in that order — but enter Diagnose only in diagnosis mode. Never guess.**

In diagnosis mode, do not invent hypotheses before reading the report. Do not start coding a fix before matching an observed pattern to a known diagnosis. Rank suggestions by evidence and expected impact. In official performance-only mode, report measurements without hypotheses or suggestions.

---

## Quickstart (what to do when someone says "profile this kernel")

0. **Create a new run directory first** under `profile/<run_name>/` at the repo root — **one directory per run**, never reuse an existing one. Each run contains its own `harness/`, `reports/`, `analysis/`, and `REPORT.md`. This rule is mandatory in this repo. See [`reference/00-directory-layout.md`](reference/00-directory-layout.md).

1. **Decide what you're profiling.** What inputs? Which dispatch path? What question do you want answered? If the kernel takes variable-sized inputs (variable seq lengths, variable batch sizes), you must pick specific representative shapes from the user's workload — don't profile with arbitrary inputs.

2. **Build a standalone harness** unless the user is profiling through their existing binary. Harnesses compile in seconds, run the kernel in isolation, and let you use `-lineinfo` cleanly so ncu can map SASS back to source. Compile into `profile/<run_name>/harness/`. See [`reference/02-harness-guide.md`](reference/02-harness-guide.md) and the template in [`helpers/harness_template.cu`](helpers/harness_template.cu).

3. **Run the overview profile** with `--set full` and `PmSampling` sections. Run `--set source --section SourceCounters` only in diagnosis mode for per-line stall attribution. Write outputs to `profile/<run_name>/reports/`. See [`reference/03-collection.md`](reference/03-collection.md).

4. **Parse with `ncu_report`** Python module — not by eye-balling the CLI. Write analysis outputs to `profile/<run_name>/analysis/`. Use the helpers in [`helpers/`](helpers/). See [`reference/04-python-api.md`](reference/04-python-api.md).

5. **Stop at factual performance analysis for official performance-only work.** For diagnosis mode, work through the six analysis dimensions in [`reference/05-analysis-dimensions.md`](reference/05-analysis-dimensions.md).

6. **In diagnosis mode, match patterns to the diagnosis playbook.** See [`reference/06-diagnosis-playbook.md`](reference/06-diagnosis-playbook.md). It maps NCU signal → likely cause → concrete fix, with example counts for "how big is this".

7. **Write the report** at `profile/<run_name>/REPORT.md`. Keep official performance-only reports factual; include ranked recommendations only in diagnosis mode. See [`reference/07-report-template.md`](reference/07-report-template.md).

---

## File index

### Reference docs (read these when you need details)

| File | Purpose |
|---|---|
| [`reference/00-directory-layout.md`](reference/00-directory-layout.md) | **Read first.** Directory / naming conventions — one run = one subdirectory, no cross-contamination |
| [`reference/01-workflow.md`](reference/01-workflow.md) | End-to-end checklist from "user request" to "final report" |
| [`reference/02-harness-guide.md`](reference/02-harness-guide.md) | When and how to build a standalone harness (mandatory for TVM-FFI, PyTorch kernels, JIT-compiled code) |
| [`reference/03-collection.md`](reference/03-collection.md) | ncu command recipes: full, source-level, PM sampling, custom sections |
| [`reference/04-python-api.md`](reference/04-python-api.md) | `ncu_report` Python API patterns with copy-pasteable code |
| [`reference/05-analysis-dimensions.md`](reference/05-analysis-dimensions.md) | Six analysis dimensions for diagnosis mode: occupancy, balance, stalls, tensor core, timeline, memory |
| [`reference/06-diagnosis-playbook.md`](reference/06-diagnosis-playbook.md) | Diagnosis-mode only: pattern → diagnosis → fix, including Hopper-specific WGMMA/TMA guidance |
| [`reference/07-report-template.md`](reference/07-report-template.md) | How to structure the final report |
| [`reference/08-hopper-metric-names.md`](reference/08-hopper-metric-names.md) | Hopper/sm_90 metric set and metric-discovery procedure |
| [`reference/09-common-issues.md`](reference/09-common-issues.md) | Permissions, PM sampling gaps, TVM-FFI / PyTorch gotchas |

### Helpers (reusable code)

| File | Purpose |
|---|---|
| [`helpers/harness_template.cu`](helpers/harness_template.cu) | Standalone harness template — paste your kernel, fill in input allocation, done |
| [`helpers/safetensors_loader.h`](helpers/safetensors_loader.h) | Header-only safetensors reader (no external deps) for loading real workload tensors |
| [`helpers/analyze_reports.py`](helpers/analyze_reports.py) | Extract key metrics, produce side-by-side comparisons |
| [`helpers/extract_stall_hotspots.py`](helpers/extract_stall_hotspots.py) | Per-line stall aggregation via `action.source_info(pc)` |
| [`helpers/plot_timeline.py`](helpers/plot_timeline.py) | ASCII PM-sampling timeline plotter — makes tail effect visible |
| [`helpers/list_flashinfer_workloads.py`](helpers/list_flashinfer_workloads.py) | Browse a flashinfer-trace dataset — shape histograms, filter by axis, resolve safetensors paths for specific UUIDs |
| [`helpers/ncu_utils.py`](helpers/ncu_utils.py) | Shared Python helpers: safe metric access, per-instance extraction, report loading |

---

## Critical lessons (don't skip)

1. **Validate metric names on the actual Hopper report.** Nsight Compute metrics vary by release and collection section. Use the curated names in [`reference/08-hopper-metric-names.md`](reference/08-hopper-metric-names.md), but fall back to `action.metric_names()` or `ncu --query-metrics --chip gh100` before drawing a conclusion from a missing value.

2. **Always compile with `-lineinfo`.** Without it, ncu's source view is blank and you cannot do per-line stall analysis. If you can't add `-lineinfo` to the build system (TVM-FFI, PyTorch inline, JIT), **build a standalone harness** — that's the whole point.

3. **PM sampling is the only way to see tail effects.** Static metrics average over the whole kernel; only the time-series (either `pmsampling:` metrics or the ASCII plotter in `helpers/`) shows the shape of utilization over time.

4. **Use Hopper features only when the profile supports them.** WGMMA is warpgroup-collective; TMA needs regular bulk transfers and a correctly synchronized pipeline; DSMEM needs genuine inter-CTA reuse. Do not prescribe any of them as generic fixes.

5. **In diagnosis mode, NCU's rule engine (`--page details`) already does half the work.** Each rule comes with `Est. Speedup: X%`. Do not use rule suggestions in official performance-only reports.

6. **Cite specific metric values.** In official performance-only work, state only measured facts and comparisons. In diagnosis mode, support each conclusion with two or three relevant metrics (for example, DRAM utilization plus a stall metric).

---

## Related skills

- [`hopper-cuda-programming.md`](hopper-cuda-programming.md) — Diagnosis-mode only: Hopper-specific programming principles for turning evidence into a WGMMA, TMA, or thread-block-cluster design.
