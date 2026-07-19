# First-page KQ FP16 Calibration

Replays the dense first-page KQ boundary with FP16 operands: nine independent
rank-4 K TMA barriers become ready before one 36-operation shared/shared
WGMMA committed group, followed by wait 0.

This is an interaction calibration probe, not an atomic microbenchmark.
