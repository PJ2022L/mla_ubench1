# Dense combine OAccum float4 load

Targets only `ld.global.v4.f32` using the combine kernel's exact layout:
eight warps, Dv=512, four float4 vectors per lane and a 4096-float split stride.
Scan `num-splits`, `rowsets`, `warps`, `pattern`, and `blocks`.

This is the dominant split-combine read atom. It intentionally excludes FMA,
softmax, shared-scale traffic, and output stores. Run only on remote H800.
