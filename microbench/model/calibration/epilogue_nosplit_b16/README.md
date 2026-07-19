# Dense No-Split Epilogue Calibration

One 256-thread CTA performs two-WG O STSM staging, async-proxy fence, and the
eight-transaction rank-4 64x512 b16 TMA store as one ordered protocol.
