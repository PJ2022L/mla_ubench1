# Dense Q/K/O tensor-map prefetch

Targets only `prefetch.tensormap` for the rank-4 Q, K, and O descriptors passed
as grid-constant kernel parameters. Use `--mode=q|k|o|qko`; QKO issues the three
prefetches used by the dense main kernel.

The instruction has no architected payload-byte or completion metric, so JSON
bandwidth and utilization are null. Only issue cycles and Gprefetch/s are
reported. Run only on remote H800.
