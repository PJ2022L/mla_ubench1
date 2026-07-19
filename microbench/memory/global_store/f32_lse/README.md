# Dense FP32 LSE global store

Targets only `st.global.f32`.

- `--role=main`: 64 contiguous output rows from the SM90 main kernel.
- `--role=combine`: lane 0 of each of eight warps writes one LSE value.

CTA-private record ranges avoid inter-CTA WAW. Latency is store issue-loop time;
CUDA-event bandwidth includes kernel completion. Run only on remote H800.
