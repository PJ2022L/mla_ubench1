# Split-KV reduction benchmarks

This family isolates FlashMLA's post-kernel combine path: FP32 partial output
and LSE reads, warp-level LSE normalization, shared scale staging, and
vectorized FP32 accumulation. Inputs use the physical logical order
`[rowset, split, 8 heads, 512 values]`; extra rowsets control cache residency.
