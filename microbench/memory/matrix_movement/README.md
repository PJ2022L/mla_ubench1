# memory / matrix_movement

Generic SM90a inline-PTX STMatrix and LDMatrix microbenchmarks. STSM and LDSM share only swizzle/addressing and measurement utilities in this family; each source still emits one target instruction kind. Build and run only on the remote H800.

Every result row uses one complete `m64n{64,256}` movement as its work unit.
`latency_value` and `initiation_interval_cycles` are therefore cycles per tile,
not cycles per individual warp instruction. LDSM completion combines its
dependency-chain latency with the remaining tile issue span. STSM reports the
tile issue span; async-proxy visibility and synchronization remain separate
atoms. Clock-derived tile spans subtract a matched inline-PTX loop baseline.

No accepted H800 full-sweep result is present yet. Stable conclusions belong
here only after the family `result.csv` is published by a complete full sweep.
