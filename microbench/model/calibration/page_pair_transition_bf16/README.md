# Dense Page-Pair Transition BF16

This composite probe times the steady dense-decode boundary from both warpgroups holding ready `rP` fragments to both warpgroups completing the next page pair's `rP`.

It retains two 128-thread warpgroups, four asymmetric named-barrier handoffs, BF16 P conversion, P `STSM` exchange, `LDSM` visibility checks, local RS PV, remote SS PV, nine committed QK groups per warpgroup, and the source `wait_group<4>`, `<1>`, and `<0>` ordering.

Residual differences are explicit: Q and K are initialized shared-memory operands rather than TMA-fed dual ping-pong buffers; both warpgroups read the same Q/K storage; softmax is a compact EX2/conversion proxy rather than the full row reduction; the requested LDSM visibility check is additional because dense remote PV consumes exchanged P directly from shared memory; WG1 issues its local PV after the P0 handoff so both PV groups can retain the source wait1/wait0 relationship without ending an inline-PTX accumulator lifetime while asynchronous work is outstanding.
