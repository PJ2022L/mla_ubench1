# Dense cached i32 global load

Targets `ld.global.nc.u32`, matching `__ldg` uses for `seqlens_k`, block-table
indices, and read-only split metadata. Scan `issuers`, `pattern`, and
`working-set-entries` to separate broadcast/L1-hot, L2-resident, and HBM cases.

Random address-generation cost remains in the full issue loop and is identified
as such in the JSON latency boundary. Run only on remote H800.
