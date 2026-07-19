# Dense Metadata Stage Calibration

Replays the complete one-warp dense scheduler metadata kernel with page size
64 and fixed overhead 5. It scans batch size, SM partitions, and deterministic
uniform/ramp/skewed/random sequence-length distributions, and validates every
32-byte scheduler record and split prefix against a CPU reference.

This is a composite interaction stage and is not an atomic benchmark.
