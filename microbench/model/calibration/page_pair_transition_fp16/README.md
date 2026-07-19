# Dense Page-Pair Transition FP16

This is the FP16 form of the dense steady page-pair transition probe. The measured boundary and synchronization/WGMMA protocol are identical to the BF16 variant, while all WGMMA input types and P conversions use FP16.

It retains two 128-thread warpgroups, four asymmetric named-barrier handoffs, P `STSM` exchange, `LDSM` visibility checks, local RS PV, remote SS PV, nine committed QK groups per warpgroup, and `wait_group<4>`, `<1>`, and `<0>` ordering.

Residual differences are the same as BF16: no K TMA or dual ping-pong K buffers, read-only Q/K storage is shared by both warpgroups, softmax is a compact EX2/conversion proxy, LDSM is an additional visibility check, and WG1's local PV is moved adjacent to remote PV so the source wait1/wait0 relationship remains representable without unsafe asynchronous accumulator lifetimes across C++ control flow.
