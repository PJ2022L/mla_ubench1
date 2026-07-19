# Dense-decode interaction calibration

This directory is deliberately outside `memory/` and `compute/`. Its programs
measure interactions between already-registered atomic operations and therefore
must not be reported as new hardware atoms.

Required remote H800 calibration records use the same six-key JSON contract and
the `dense_decode.calibration.*` namespace:

| Name | Boundary |
|---|---|
| `kq_first_page_{bf16,fp16}` | Nine K TMA transactions become ready, then the source-real first-page QK group completes. |
| `kq_steady_page_{bf16,fp16}` | Nine per-tile TMA waits and nine four-HGMMA groups, with the final Q tile using RS. |
| `page_pair_transition_{bf16,fp16}` | Two-WG `rP ready -> next rP ready`, preserving asymmetric barriers and wait-group immediates. |
| `metadata_stage` | Complete one-warp metadata kernel for a `(batch_size,num_sm_parts,seqlen distribution)` case. |
| `softmax_stage_{bf16,fp16}` | Two-WG register-heavy online softmax, including reduction, EX2/RCP, FP32 rescale, and probability conversion. |
| `epilogue_nosplit_b16` | Ordered two-WG O STSM, proxy fence, and rank-4 b16 TMA store protocol. |
| `epilogue_split_f32` | Ordered stride-520 FP32 staging, proxy fence, and row-wise bulk shared-to-global store protocol. |
| `combine_stage_{bf16,fp16}` | One combine CTA for an actual split count and MAX_SPLITS bucket. |
| `pdl_overlap` | Producer main-grid tail and dependent combine-grid overlap using `griddepcontrol`. |

KQ covers WGMMA+TMA. Page-pair covers WGMMA+SFU and
STMatrix+remote-WGMMA. Softmax isolates the register-heavy SFU/FP/shuffle
interaction. Each epilogue probe measures staging, proxy fence, and store as one
ordered protocol while its block-count and working-set scans expose cross-CTA
memory contention. Remote records retain resident CTA/WG count, working set,
cache mode, clock/power state, and SASS hashes through the result plus provenance
sidecar.

These binaries are built and run only on the remote H800 after all atomic
benchmarks pass static and correctness checks. The base model remains usable
without them, but marks the prediction incomplete and applies no speculative
PDL overlap credit.
