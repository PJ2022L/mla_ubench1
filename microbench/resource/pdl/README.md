# resource / pdl

Generic SM90a inline-PTX microbenchmarks for this instruction family. Sources, binaries, JSON `name`, and manifest IDs use the same stem.

Run `python3 scripts/build.py --dry-run` locally to inspect commands. Compile and execute full sweeps only on the remote H800. A successful `--mode full` sweep atomically replaces `result.csv`; quick and failed runs remain only under `build/logs` and `build/raw`.

No accepted H800 full-sweep result is present yet. Add only stable conclusions from the current `result.csv`; keep raw timing detail in `build/`.

`griddepcontrol_launch_dependents` and `griddepcontrol_wait` publish ready-path instruction cost after subtracting a matched programmatic-launch loop. `griddepcontrol_producer_consumer` is a grid-level resource curve for scheduling and overlap; it is never substituted for either operation atom and never acts as calibration data.
