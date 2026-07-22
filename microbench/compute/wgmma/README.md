# compute / wgmma

Generic SM90a inline-PTX microbenchmarks for this instruction family. Sources, binaries, JSON `name`, and manifest IDs use the same stem.

Run `python3 scripts/build.py --dry-run` locally to inspect commands. Compile and execute full sweeps only on the remote H800. A successful `--mode full` sweep atomically replaces `result.csv`; quick and failed runs remain only under `build/logs` and `build/raw`.

No accepted H800 full-sweep result is present yet. Add only stable conclusions from the current `result.csv`; keep raw timing detail in `build/`.

`latency_value` is dependency completion for one committed group containing the
selected `group_size`, measured with issue depth 1 and `wait_group 0`.
`initiation_interval_cycles` is cycles per committed group at the selected
`group_size` and `depth`; it is not cycles per individual WGMMA instruction.

The `m64n64k16_ss_{bf16,fp16}` binaries additionally support
`group_size=36, depth=1`. This protocol issues 36 WGMMA instructions into one
group, executes one `commit_group`, and then `wait_group 0`; it measures a long
dependency chain without replacing it by nine four-instruction commits. Other
shapes and source modes accept only `group_size=1|4`. The full sweep uses
explicit cases so no invalid `group_size=36, depth=2|4` rows are produced.
Both clock-derived metrics subtract a same-CTA, same-WG inline-PTX
`add + setp + branch` loop baseline. CUDA-event TFLOP/s is measured in a
separate launch path that skips the baseline loop.

The formal `result.csv` intentionally uses a compact WGMMA-only schema. Its
`args` JSON contains only sweep/replay fields (`iters`, `warmup`, `samples`,
requested/resolved blocks, warpgroups, group size, and depth). Shape, source
mode, dtype, accumulator/layout, swizzle, transpose, and scale contracts are
encoded by the generic benchmark name and manifest, so they are not repeated
in every row. Global-memory bandwidth and generic utilization columns are
omitted because they are not WGMMA measurements. GPU identity/clocks,
latency, initiation interval, throughput, latency percentiles, sample count,
and source/SASS hashes remain for interpretation and provenance.
