# memory / tma_load

Generic SM90a inline-PTX microbenchmarks for this instruction family. Sources, binaries, JSON `name`, and manifest IDs use the same stem.

Run `python3 scripts/build.py --dry-run` locally to inspect commands. Compile and execute full sweeps only on the remote H800. A successful `--mode full` sweep atomically replaces `result.csv`; quick and failed runs remain only under `build/logs` and `build/raw`.

No accepted H800 full-sweep result is present yet. Add only stable conclusions from the current `result.csv`; keep raw timing detail in `build/`.

The clock metric is the complete TMA load protocol: one logical tile issues one or nine rank-4 transactions, performs `mbarrier.arrive.expect_tx`, waits for completion, and then exposes the data to shared memory. A separately compiled specialization with the same page selection, loop nesting, shared consume, and sink but no TMA/mbarrier transaction is subtracted sample by sample. CUDA-event throughput launches only the target specialization.
