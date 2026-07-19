# Dense num_splits u32 global store

Targets only `st.global.u32`, normally with the scheduler's 32 coalesced producer
threads. `--producers` also supports smaller populations for tail batches.
Each CTA owns a disjoint output interval.

Run only on the remote SM90a H800.
