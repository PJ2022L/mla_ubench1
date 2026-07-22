# Dense decode calibration

This directory contains operator-specific, high-level probes for FlashMLA dense
decode on H800/SM90a. They are intentionally outside `microbench/`: a probe may
cover several generic atoms and interactions, so it is not a prediction input.

## Hard policy

- Official latency prediction reads only generic inline-PTX results under
  `microbench/**/result.csv` and generic resource curves.
- Calibration results are used only to compare atom-DAG predictions with a
  measured high-level boundary. They never provide latency constants,
  correction factors, offsets, multipliers, HBM fractions, or overlap credits.
- A residual may lead to fixing the DAG, correcting an operation count, or
  adding a new generic microbenchmark. It must not be fitted away here.
- These CUDA binaries are executed only on the remote H800. Local work may do
  schema checks, dry runs, compilation, and PTX/SASS/resource inspection, but
  must not launch a calibration kernel.

## Probe boundaries

| Probe | Timed boundary | Important exclusion or caveat |
|---|---|---|
| `first_score_{bf16,fp16}` | 9 complete K TMA protocols, then 36 SS WGMMA in one committed group and wait-group 0 | Each generic TMA atom already includes issue, `expect_tx`, and completion wait; no separate barrier-wait atom is charged |
| `steady_score_{bf16,fp16}` | Issue all 9 K TMAs first, then consume 9 completion-dependent four-WGMMA groups in order; tile 8 is RS; final wait-group 0 | Independent TMA DAG nodes permit the source's issue-ahead overlap; TMA and WGMMA protocol overhead is owned exactly once |
| `page_pair_transition_{bf16,fp16}` | Both WGs have current `rP` until both have next `rP` | Its LDSM is a validation-only visibility check, is timed by the probe, and is not dense-kernel work |
| `softmax_page_update_{bf16,fp16}` | Per-page max, EX2, sum, rescale arithmetic, and P conversion | No RCP or final normalization; those belong to the final reduction/epilogue |
| `nosplit_store_protocol_b16` | O STSM, async-proxy fence, barrier, rank-4 TMA store and wait | Excludes normalization and LSE generation; not a full epilogue |
| `split_store_protocol_f32` | Stride-520 FP32 staging, async-proxy fence, barrier, bulk S2G store and wait | Excludes normalization and LSE generation; not a full epilogue |
| `combine_stage_{bf16,fp16}` | Clock64 starts after the two split-offset loads and `griddepcontrol.wait`, then covers partial/LSE loads, reduction, accumulation, conversion and stores | Offset loads and grid wait are CUDA-event work but are outside the residual Clock64 DAG |
| `metadata_stage` | Complete one-warp metadata scheduling stage | Compared with its own source-shaped probe DAG |
| `pdl_overlap` | Full producer-prefix/signal/tail through consumer-wait/work pair from cross-SM `%globaltimer` stamps | Pair time is the residual boundary; tail/wait overlap is retained only as diagnostic telemetry and never becomes a credit |

Each entry in `manifest.json` points to a `probe_dags/*.json` description of the
code actually timed. These probe DAGs may overlap each other and are never added
together as dense phases.

## Remote workflow

```bash
CAL=operators/flash_mla/paths/dense_decode_bf16_sm90_mqa/calibration
python3 "$CAL/scripts/static_check.py"
python3 "$CAL/scripts/build.py" --dry-run
python3 "$CAL/scripts/build.py"
python3 "$CAL/scripts/run.py" --preset quick
python3 "$CAL/scripts/run.py" --preset full
```

Only a complete successful `full` sweep atomically replaces `result.csv`.
Commands, full arguments, start/end timestamps, duration, failures, stdout and
raw JSON stay under `build/logs/` and `build/raw/`; wall time is not copied into
the result table. `quick` runs never update the formal result.

The result table contains measurements, provenance and hashes only. The dense
model's calibration validator owns residual computation and emits
`probe, atom_predicted_cycles, measured_cycles, residual_cycles,
relative_error, status, suspected_resources` with pass/warn/fail thresholds of
10% and 20%. Clock64 probes already report cycles. The PDL raw JSON remains in
`us/pair`; when publishing the formal CSV, the runner converts its samples to
cycles using the `nvidia-smi` SM clock sampled immediately before that binary
invocation and records the source unit, clock and conversion in `params_json`.
