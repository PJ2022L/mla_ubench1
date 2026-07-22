# Dense Decode Atom-DAG Model

This package predicts the GPU interval `metadata + persistent main + combine`.
It excludes host launch, allocator, Python, and input preparation time.

Prediction reads only generic operation and resource-curve rows declared by
`microbench/manifest.json`. Calibration is a separate residual check and never
changes an atom cost, overlap, multiplier, or offset.

```bash
python3 -m operators.flash_mla.paths.dense_decode_bf16_sm90_mqa.model build-dag \
  --workload operators/flash_mla/paths/dense_decode_bf16_sm90_mqa/model/workload.example.json \
  --kernel-resources operators/flash_mla/paths/dense_decode_bf16_sm90_mqa/model/dense-resources.example.json \
  --output dag.json

python3 -m operators.flash_mla.paths.dense_decode_bf16_sm90_mqa.model predict \
  --microbench-root microbench --kernel-resources dense-resources.json \
  --workload workload.json --output prediction.json
```

`phase_timing.wall_span_cycles` is a non-additive timeline span. The
`critical_path_contribution_cycles` values are additive and sum exactly to the
reported E2E cycles.

Operation throughput in the generic microbench CSV is a whole-grid CUDA-event
rate. Prediction normalizes it by the measured active-SM count before applying
it to a per-SM execution queue; a throughput unit explicitly marked `/SM` is
already normalized. L2 and HBM remain single whole-GPU service queues.

`resource_utilization` is paired with `resource_capacity`. Tensor/TMA/SFU/FP,
shared/barrier/issue, PDL `grid`, and global-memory LSU issue are per-SM queues.
L2 and HBM are whole-GPU byte queues. A load miss traverses HBM then L2; a
store traverses L2 then HBM. Cache residency becomes visible only after the
modeled L2 fill completes, and scheduling replays hit/miss routes to a fixed
point. Partial output/LSE stores seed request-local producer/consumer regions;
an oversized region is not treated as unconditionally L2-hot.

Operation and resource-curve queries use the CTA's actual wave-local
`blocks`, `active_sm`, and `resident_cta`, including the final partial wave.
The first combine wave also waits for the corresponding SM/resident slot's
last main wave to release occupancy. Critical-path entries expose canonical
`issue`, `service`, and `complete` events. A queued service entry names its
`service_resource` and `service_unit`; an isolated instruction-latency tail is
marked with `service_kind: isolated_latency`.

The resource JSON is only a planning example. The remote H800 run must replace
its main/combine register and shared-memory fields with values extracted from
the actual cubins used for the prediction.
