# SM90a Microbenchmarks

This directory contains kernel-agnostic CUDA + inline-PTX microbenchmarks. It contains no dense-decode model or operator calibration code. The remote NVIDIA H800 is the only supported execution target; local work is limited to schema checks, dry-runs, compilation, PTX/SASS inspection, and CPU-side tests.

Each family owns its concrete opcode helpers and harness under `common/`, one source per generic benchmark ID, family-local build/sweep scripts, generated artifacts under `build/`, a formal `result.csv`, and a concise README. Root `common/bench.hpp` contains only shared CLI, CUDA error handling, timing, statistics, and six-key JSON utilities.

Identity is strict: manifest `id`, source stem, binary, and JSON `name` are identical. `manifest.json` registers generic operations and resource curves only. Operator roles and source anchors belong in the operator model's mapping file.

The current manifest contains 67 entries across 21 families: 60 operations and 7 resource curves. This count is descriptive, not a compatibility target. The M64N64 shared/shared WGMMA family includes a generic `group_size=36, depth=1` dependency protocol in addition to the ordinary group-size 1/4 curves.

Formal results use the latest complete accepted full sweep. A quick or failed run never replaces `result.csv`; commands, complete arguments, durations, failures, and raw samples stay in family-local `build/logs` and `build/raw`.
All checked-in `result.csv` files are currently header-only, so no local timing is accepted. Generated `build/`, `__pycache__`, and `.pyc` content is ignored and may be removed after local verification. On the remote H800, retain the accepted full log/raw pair and matching SASS/resource evidence.

Local checks:

```bash
make -C microbench validate
make -C microbench static
make -C microbench dry-build
make -C microbench dry-run
```

Remote H800 flow:

```bash
make -C microbench build
make -C microbench quick
make -C microbench run
```

`resource/memory_service` covers active-grid, working-set, cache-mode, access-pattern, and outstanding-depth curves for global memory. `resource/tma_service` provides TMA service curves. `resource/interference` measures mixed WGMMA shape/source mode, WGMMA+TMA, and WGMMA+SFU/shared competition. `resource/pdl` contains standalone PDL instruction atoms plus the producer-consumer grid dependency curve.
