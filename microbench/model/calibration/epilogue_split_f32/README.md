# Dense Split Epilogue Calibration

One 256-thread CTA performs the 64x520 FP32 float2 shared staging, async-proxy
fence, and 64 row-wise bulk shared-to-global stores as one ordered protocol.
