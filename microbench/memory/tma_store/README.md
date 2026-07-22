# memory / tma_store

Generic SM90a inline-PTX microbenchmarks for this instruction family. Sources, binaries, JSON `name`, and manifest IDs use the same stem.

Run `python3 scripts/build.py --dry-run` locally to inspect commands. Compile and execute full sweeps only on the remote H800. A successful `--mode full` sweep atomically replaces `result.csv`; quick and failed runs remain only under `build/logs` and `build/raw`.

No accepted H800 full-sweep result is present yet. Add only stable conclusions from the current `result.csv`; keep raw timing detail in `build/`.

The clock metric is the complete eight-transaction rank-4 store protocol, including `commit_group` and `wait_group 0`. A matched specialization retains the depth loop, tile selection, and sink while removing TMA store/commit/wait, and is subtracted sample by sample. CUDA-event throughput launches only the target specialization.
