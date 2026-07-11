# Shared-to-global bulk-store benchmarks

This family isolates `cp.async.bulk.global.shared::cta.bulk_group` stores. The
source tile is prepared in shared memory before timing; register-to-shared
staging belongs to the `stmatrix` or `shared_store` families.

The benchmark keeps FlashMLA's split-epilogue ownership: eight warp leaders
each store eight rows from a `[64,520]` FP32 shared layout. Output working-set
size and traversal pattern can expose cache-resident and HBM-backed behavior.
Completion is intentionally source-faithful depth 1; this family does not scan
an artificial multi-group issue depth.
