# memory / shared_load

Generic SM90a inline-PTX microbenchmarks for this instruction family. Sources, binaries, JSON `name`, and manifest IDs use the same stem.

Latency is reported as a paired `target - baseline` result. The baseline is a separately compiled kernel with the same shared-address calculation, access-pattern selection, checksum dependency, loop control, and CTA barriers, but no `ld.shared`/`LDS`. Raw target and baseline clock samples remain in the JSON record. The CUDA-event throughput interval launches only the target kernel; baseline work is never counted as source operations or bandwidth.

Static compilation rejects the family if the baseline function regains `LDS`, loses its checksum/control dataflow, or the target function loses `ld.shared.u32`.

Run `python3 scripts/build.py --dry-run` locally to inspect commands. Compile and execute full sweeps only on the remote H800. A successful `--mode full` sweep atomically replaces `result.csv`; quick and failed runs remain only under `build/logs` and `build/raw`.

No accepted H800 full-sweep result is present yet. Add only stable conclusions from the current `result.csv`; keep raw timing detail in `build/`.
