# Dense Combine Stage BF16 Calibration

Replays one or more source-shaped combine CTAs with eight warps, `d_v=512`,
the actual 32/64/96/128/160 MAX_SPLITS buckets, vector OAccum loads, LSE
reduction, shared scales, FP32 FMA accumulation, BF16 conversion, and u64
output stores. The standalone launch retains the source PDL wait instruction.

This is a composite interaction stage and is not an atomic benchmark.
