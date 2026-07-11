# Distributed shared-memory store benchmarks

This family isolates FlashMLA's 128-bit asynchronous cluster store. Each CTA
owns the same 32-by-576 BF16 K-tile slice as one producer CTA in the sparse
FP8 decode cluster and receives completion through a transaction barrier.

The `peer` mode exchanges slices between the two CTAs. The `local` mode maps
the destination back to the issuing CTA while retaining the same cluster
instruction and barrier protocol, providing a transport-distance control.
