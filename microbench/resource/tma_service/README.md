# resource / tma_service

Generic rank-4 TMA load service curves across outstanding depth, resident blocks,
working-set size, cache preparation, and local/sequential/random/reuse access
patterns. The reuse case cycles a shared four-page subset across CTAs. This
resource curve reuses the canonical TMA operation protocol and has its own result
identity. Build and run only on the remote H800.
