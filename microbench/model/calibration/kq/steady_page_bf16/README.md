# Steady-page KQ BF16 Calibration

Replays the tile-pipelined dense KQ path. Each of nine K barriers is waited in
order and immediately feeds a four-operation committed group; tiles 0-7 use
shared/shared Q, tile 8 uses the retained register Q fragment, then wait 0.

This is an interaction calibration probe, not an atomic microbenchmark.
