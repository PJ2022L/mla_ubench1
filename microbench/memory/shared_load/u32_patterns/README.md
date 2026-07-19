# Dense shared u32 load

Targets only `ld.shared.u32`. `--pattern=unique` models scheduler/final-LSE reads,
`quad_broadcast` models the four-lane `sM/sScale` multicast, and
`warp_broadcast` models combine-scale broadcast.

Sweep `threads`, `pattern`, `working-set-words`, `blocks`, and the common timing
parameters. Requested bandwidth is shared-memory traffic; HBM peak utilization
is intentionally null. Run only on the remote SM90a H800.
