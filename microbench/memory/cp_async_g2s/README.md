# cp.async GMEM-to-shared benchmarks

This family measures explicit 16-byte `cp.async.cg.shared.global` gather paths.
It is separate from `global_load`: the destination is shared memory, completion
uses a `cp.async` group or mbarrier, and the instruction occupies a different
producer path than a register-returning `ld.global`.

