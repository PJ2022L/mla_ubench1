# Dense decode held-out E2E

`benchmark.py` measures the public `flash_mla_with_kvcache` path on the remote
H800. It captures one API invocation in a CUDA graph before timing, so replay
contains GPU work without Python, allocator, or per-launch host gaps.

- `--metadata-mode generate` captures metadata + persistent main + combine.
- `--metadata-mode reuse` initializes metadata before capture and measures main
  + combine only.
- `--cache-mode l2_hot` performs an untimed graph replay immediately before the
  sample.
- `--cache-mode hbm_stream` performs an untimed 128 MiB eviction write before a
  single timed replay.
- BF16/FP16, exact per-request sequence lengths, causal, contiguous/random/reuse
  block tables, zero/tail pages, and different Q/KV head counts are supported.

Local argument checks do not import Torch or execute CUDA:

```bash
python3 operators/flash_mla/paths/dense_decode_bf16_sm90_mqa/e2e/benchmark.py \
  --validate-only --batch 3 --seqlens-k 0,65,4097 --dtype fp16 \
  --s-q 2 --h-q 128 --h-kv 2 --causal --metadata-mode generate \
  --block-pattern random --cache-mode hbm_stream
```

Run formal cases only after all microbenchmark full sweeps and predictions are
frozen. Use one result directory per case; command, full args, wall time and
errors stay in `run.log`, while stdout is the single JSON record. A nonzero run
may still write `result.jsonl`; that file is diagnostic evidence, not an
accepted result.

```bash
e2e=operators/flash_mla/paths/dense_decode_bf16_sm90_mqa/e2e
run_id="$(date +%Y%m%d-%H%M%S)-bf16-generate-tail"
run_dir="$e2e/result/runs/$run_id"
mkdir -p "$run_dir"
started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'started_utc=%s\n' "$started_utc" >"$run_dir/run.log"
/usr/bin/time -a -o "$run_dir/run.log" \
  -f 'command=%C\nwall_seconds=%e\nexit_status=%x' \
  python3 "$e2e/benchmark.py" \
    --case-id bf16-generate-tail --dtype bf16 --batch 128 --s-q 1 \
    --s-k 4097 --metadata-mode generate --block-pattern contiguous \
    --cache-mode hbm_stream --samples 20 --check-correctness \
    >"$run_dir/result.jsonl" 2>>"$run_dir/run.log"
status=$?
printf 'ended_utc=%s\nreturncode=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >>"$run_dir/run.log"
test "$status" -eq 0
```

An accepted matrix must cover both dtypes, metadata generate/reuse,
`N_page=0/1/even/odd`, non-64 tails, short/long and skewed batches, causal
queries, split/no-split, all block-table patterns, both cache modes, and several
head configurations. Formal runs must pass `--check-correctness`; an unchecked
run is rejected. `correctness.passed` and `acceptance_gate.passed` must both be
true. The record includes
actual `num_splits`, p10/p50/p90 CUDA-event latency, GPU UUID/clocks/power and
the exact case. It compares GPU `tile_scheduler_metadata` row by row and the
complete `num_splits` prefix against `model/scheduler.py`. A metadata/split
mismatch or an upstream `source_defined=false` result makes
`scheduler_validation.passed=false`. Any of those scheduler failures, an
unchecked correctness result, or a correctness mismatch makes
`acceptance_gate.passed=false`, returns status 2, and cannot be used as an
accepted held-out case. `acceptance_gate.rejection_reasons` states the exact
gate(s); keep such output only as diagnostic evidence.

Held-out results validate the atom-DAG model only. They must never be used to
fit atom latency, resource slowdown, offsets, multipliers, or overlap credits.

## Accepted H800 Results

| Run | Case | Correct | P50 us | Metadata mode | Split distribution |
|---|---|---:|---:|---|---|
| No accepted H800 run yet | - | - | - | - | - |
