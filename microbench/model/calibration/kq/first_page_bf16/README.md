# First-page KQ BF16 Calibration

Replays the dense first-page KQ boundary: issue nine rank-4 K TMA loads,
wait until all nine independent barriers are ready, issue 36 shared/shared
`m64n64k16.f32.bf16.bf16` operations in one committed group, then wait 0.

This is an interaction calibration probe, not an atomic microbenchmark.
