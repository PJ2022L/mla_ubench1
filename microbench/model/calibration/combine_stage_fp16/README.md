# Dense Combine Stage FP16 Calibration

Replays the complete eight-warp `d_v=512` combine stage with the source
MAX_SPLITS buckets. It includes vector loads, LSE reduction, FP32 FMA,
FP16 conversion, u64 stores, and the source PDL wait boundary.

This is a composite interaction stage and is not an atomic benchmark.
