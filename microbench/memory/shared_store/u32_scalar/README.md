# Dense shared u32 store

Targets only `st.shared.u32` for scheduler scratch, `sM/sScale`, and the final
cross-warpgroup L reduction workspace. `--producers` scans the one-thread,
warp, 64-thread, 128-thread, and full-CTA producer populations. Use
`--topology=contiguous|quad_leaders|warp_leaders`; `quad_leaders` reproduces
the `lane % 4 == 0` softmax and reduction writers.

The reported latency is issue-loop time, not a cross-proxy visibility latency.
Run only on the remote SM90a H800.
