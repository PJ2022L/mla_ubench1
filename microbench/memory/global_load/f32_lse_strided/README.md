# Dense combine strided LSE load

Targets only `ld.global.f32`. Each active warp reads one scalar for every split;
lanes span split indices and therefore use the real large `split-stride` gather
pattern. Scan `num-splits`, `split-stride`, `rowsets`, `warps`, and cache-sized
working sets.

Run only on the remote SM90a H800.
