# resource / interference

Generic concurrent-grid SM90a service-degradation curves for mixed WGMMA shape/source mode, WGMMA+TMA, and WGMMA+SFU/shared traffic. These are resource curves, not DAG operations. Build and run only on the remote H800.

Every probe sweeps `actors=1,2`. Both rows use the same 256-thread launch and
the same dynamic-shared-memory footprint; the one-actor row disables the peer
inside that matched footprint and intentionally omits `peer_resource`. The
two-actor row enables the named peer. Slowdown is valid only as a matched
baseline/peer ratio from the same probe and sweep condition. `working_set_pages`
is meaningful and swept only for the WGMMA+TMA probe; the shared-only probes do
not publish a fictitious working-set axis.
