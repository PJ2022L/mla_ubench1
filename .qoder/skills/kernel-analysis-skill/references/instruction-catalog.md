# Instruction catalog for analysis

Classify an instruction from generated PTX/SASS when possible. Source APIs and compiler intrinsics are supporting evidence, not proof of a particular emitted mnemonic.

| Family | Typical evidence | Analyze in the report |
|---|---|---|
| TMA | PTX `cp.async.bulk.tensor...`, tensor-map descriptor use, `mbarrier` transaction count | Direction (GMEM↔SMEM), tensor/map layout, multicast if present, issuing thread(s), destination buffer, and barrier that establishes readiness. |
| Hopper WGMMA | PTX `wgmma.mma_async...`, plus `wgmma.fence`, `commit_group`, or `wait_group` when present | Operand sources (register/descriptor), M×N×K shape, accumulator lifetime, group commit/wait, and warp-group ownership. |
| Warp MMA | `mma.sync...`, often preceded by `ldmatrix` | Fragment shape/layout, warp ownership, shared-memory feed, and reduction/epilogue dependency. |
| Async copy | `cp.async...` with `commit_group`/`wait_group` | Copy stage, group depth, shared-memory buffer, and the consumer wait. Keep this distinct from TMA. |
| Matrix fragment load | `ldmatrix...` | Shared-memory source layout, fragment distribution, and the MMA it feeds. |
| Ordinary loads/stores | `ld.global`, `ld.shared`, `st.shared`, `st.global`, vectorized variants | Address space, vector width, coalescing/layout implication, and whether the operation is on the critical path. |
| Synchronization | `mbarrier.*`, `bar.sync`, `barrier.cluster`, `__syncthreads`, warp-level synchronization | Participants/scope, the data made safe, and which wait/arrival appears on the timeline. |
| Warp communication/reduction | `shfl.sync`, `redux`, shared-memory reduction, atomics | Reduction axis, participating lanes, accumulator ownership, and any output race/atomic requirement. |

On Hopper, use `sm_90a` for code that actually emits WGMMA, TMA, or thread-block-cluster features. Do not infer these features from `sm_90` alone. SASS mnemonics vary across tool versions; quote the observed form instead of normalizing it to a guessed name.
