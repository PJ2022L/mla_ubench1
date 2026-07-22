# memory / shared_store

Generic SM90a inline-PTX microbenchmarks for this instruction family. Sources, binaries, JSON `name`, and manifest IDs use the same stem.

Latency is reported as a paired `target - baseline` result. Target and baseline use the same layout, address, value, predicate, checksum, loop-control, and CTA-barrier path; the separately compiled baseline emits no `st.shared`/`STS`. Throughput uses a distinct target-only kernel with no baseline/checksum work in the CUDA-event grid.

The `u32`, `u64 SW128`, and `v2.u32 stride-520` leaves compile independently. Static gates reject a baseline containing `STS`, a target missing its requested store, or an 8-byte target containing the other leaf's PTX opcode.

Run `python3 scripts/build.py --dry-run` locally to inspect commands. Compile and execute full sweeps only on the remote H800. A successful `--mode full` sweep atomically replaces `result.csv`; quick and failed runs remain only under `build/logs` and `build/raw`.

No accepted H800 full-sweep result is present yet. Add only stable conclusions from the current `result.csv`; keep raw timing detail in `build/`.
