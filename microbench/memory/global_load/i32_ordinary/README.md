# Dense ordinary i32 global load

Targets `ld.global.u32` for the main kernel's PDL-sensitive
`num_splits_ptr[batch_idx]` read. The upstream source explicitly avoids
`__ldg` here because the dependent combine launch and instruction ordering
matter, so this leaf is intentionally separate from `i32_cached`.

Scan `issuers`, `pattern`, `working-set-entries`, and grid size. Run only on
the remote SM90a H800.
