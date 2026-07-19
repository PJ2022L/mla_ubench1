# Dense combine packed output store

Targets only `st.global.u64` for the combine epilogue's packed four-element b16
output. `--dtype=bf16|fp16` changes the payload pattern, while the hardware store
instruction remains identical. The full dense shape is eight warps, four stores
per lane, or 8192 bytes per CTA round.

Run only on the remote SM90a H800.
