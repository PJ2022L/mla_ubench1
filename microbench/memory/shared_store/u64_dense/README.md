# Dense shared 64-bit store

Targets one 64-bit shared-store opcode per source operation.

- `--role=stride520` emits `st.shared.v2.u32` with the exact split-O
  register-to-shared 64x520 address mapping.
- `--role=tail_zero` emits `st.shared.u64` with the dense SW128 half-V tail-fill
  mapping; sweep `invalid-tokens`.

Proxy fences are deliberately outside this leaf. Run only on remote H800.
