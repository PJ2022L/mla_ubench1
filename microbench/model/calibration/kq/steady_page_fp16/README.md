# Steady-page KQ FP16 Calibration

Replays the tile-pipelined dense KQ path with FP16 operands. Nine barrier
waits each feed four WGMMA operations; tile 8 uses register/shared WGMMA and
the page ends with wait 0.

This is an interaction calibration probe, not an atomic microbenchmark.
