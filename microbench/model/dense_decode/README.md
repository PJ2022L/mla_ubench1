# FlashMLA dense-decode H800 model

This package predicts the public dense-decode API, not just one CTA. Its input
profile is built exclusively from accepted `microbench` and interaction
calibration records. E2E measurements are validation data and are never used to
fit the base model.

## Model boundary

The modeled path is:

```text
optional metadata kernel -> persistent SM90 main grid -> PDL combine grid
```

The scheduler is a CPU replica of `get_mla_metadata_kernel`. It produces the
same partition ranges and split prefix used to construct the main CTA jobs and
combine CTAs. Each persistent main CTA keeps its assigned request slices in
order and expands them into source-shaped phases: Q/K prologue, first-page QK,
steady even/odd page pairs, tail, L reduction, and the no-split or split
epilogue.

Within a phase, atom tasks advance concurrently. Per-SM resources (Tensor Core,
TMA issue, SFU, FP/INT issue, shuffle, shared memory, barriers) and card-wide
resources (L2/HBM) use event-driven bottleneck sharing. Base service times are
interpolated from nearby microbenchmark parameter points. Main and combine CTA
residency is derived from real-kernel registers, shared memory, warps, CTA
slots, and launch bounds. Full-grid memory pressure uses block-count curves,
logical/unique physical pages, and measured HBM/L2 capacities.

Source-real KQ, page-pair, softmax, epilogue, metadata, combine, and PDL
calibration records take precedence over atom fallback. A measured softmax or
epilogue protocol replaces, rather than supplements, its atom expansion. Missing
PDL calibration never earns speculative overlap: the prediction is marked
incomplete and conservatively serializes the main/combine contribution.

## Commands

```bash
python -m microbench.model.dense_decode build-profile \
  --microbench-results microbench/results/<run-id>/profile \
  --static-artifacts microbench/results/<run-id>/static \
  --output microbench/results/<run-id>/h800-profile.json

python -m microbench.model.dense_decode predict \
  --profile microbench/results/<run-id>/h800-profile.json \
  --workload microbench/model/dense_decode/workload.example.json \
  --bootstrap 1000 \
  --output prediction.json

python -m microbench.model.dense_decode validate \
  --profile microbench/results/<run-id>/h800-profile.json \
  --cases heldout.jsonl \
  --e2e-results e2e-results.jsonl \
  --output validation.json
```

`build-profile` rejects result names not registered in `manifest.json` and
records every source record plus static artifact hash. Bootstrap uses metric
samples preserved by the benchmark. If raw samples are absent, the model does
not invent a noise distribution and reports uncertainty as unavailable.

The workload accepts exact `seqlens_k` and exact `block_table`, or deterministic
sequence/block-table distributions. Dense SM90 fixes `head_dim_qk=576`,
`head_dim_v=512`, and `page_size=64`; dtype is BF16 or FP16. Cache mode,
physical-page reuse, and block-table locality alter L2/HBM contention.

## Prediction output

The JSON includes:

- p10/p50/p90 E2E microseconds and metadata/main/combine/launch breakdown;
- exact scheduler metadata, split prefix, request slices, CTA waves and tail;
- requested Q/K/output/combine bytes and the profile's L2/HBM references;
- resource-sharing policy, CTA residency, calibration warnings and immutable
  microbenchmark provenance.

The target metadata source has an undefined boundary when it consumes all
requests before emitting all `num_sm_parts` records. The model reports
`scheduler.source_defined=false` and marks the prediction incomplete instead
of assigning valid semantics to the out-of-range source access.

The acceptance target is held-out MAPE at most 10% and P90 absolute percentage
error at most 15%. A passing aggregate is not sufficient if scheduler splits,
CTA waves, memory traffic, or the profiler-identified bottleneck disagree.
