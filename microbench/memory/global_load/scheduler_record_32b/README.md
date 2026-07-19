# Dense scheduler-record global load

Loads one aligned 32-byte `DecodingSchedMeta` record as two explicit
`ld.global.v4.u32` instructions. `broadcast` reproduces all lanes selecting one
partition record; `issuers` reveals whether a single-issuer lowering would be
materially different.

The remote build must compare this forced form with the upstream kernel SASS;
the C++ aggregate load is not guaranteed to lower identically across toolchains.
