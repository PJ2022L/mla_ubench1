# memory / tensormap_prefetch

Generic SM90a inline-PTX microbenchmarks for this instruction family. Sources, binaries, JSON `name`, and manifest IDs use the same stem.

Run `python3 scripts/build.py --dry-run` locally to inspect commands. Compile and execute full sweeps only on the remote H800. A successful `--mode full` sweep atomically replaces `result.csv`; quick and failed runs remain only under `build/logs` and `build/raw`.

No accepted H800 full-sweep result is present yet. Add only stable conclusions from the current `result.csv`; keep raw timing detail in `build/`.

Latency is reported per selected descriptor round after subtracting a separately compiled specialization with identical mode branches and loop control but no `prefetch.tensormap`. `mode=all` remains one round containing three descriptor prefetches. CUDA-event throughput launches only the target specialization.
