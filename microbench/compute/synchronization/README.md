# compute / synchronization

Generic SM90a inline-PTX microbenchmarks for this instruction family. Sources, binaries, JSON `name`, and manifest IDs use the same stem.

Run `python3 scripts/build.py --dry-run` locally to inspect commands. Compile and execute full sweeps only on the remote H800. A successful `--mode full` sweep atomically replaces `result.csv`; quick and failed runs remain only under `build/logs` and `build/raw`.

No accepted H800 full-sweep result is present yet. Add only stable conclusions from the current `result.csv`; keep raw timing detail in `build/`.

Latency keeps every instruction required for a valid barrier generation, including arrive/complete operations and protocol-internal CTA barriers, while subtracting only a matched volatile register-loop specialization. The converged full-mask warp-sync atom is explicitly zero-cost because SM90a ptxas elides its executable SASS. CUDA-event throughput launches only the target specialization.
