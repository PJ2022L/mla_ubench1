# memory / global_load

Generic SM90a inline-PTX microbenchmarks for this instruction family. Sources, binaries, JSON `name`, and manifest IDs use the same stem.

Run `python3 scripts/build.py --dry-run` locally to inspect commands. Compile and execute full sweeps only on the remote H800. A successful `--mode full` sweep atomically replaces `result.csv`; quick and failed runs remain only under `build/logs` and `build/raw`.

Latency is the paired clock result from a target specialization minus a separately compiled matched address/pattern/checksum/control baseline. Throughput is measured with CUDA events around target-specialization launches only; the baseline is never inside the event interval. Raw JSON records both target and baseline clock samples together with the `matched_target_minus_baseline` and `target_only` protocol markers.

No accepted H800 full-sweep result is present yet. Add only stable conclusions from the current `result.csv`; keep raw timing detail in `build/`.
