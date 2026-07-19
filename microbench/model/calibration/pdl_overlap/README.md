# Dense PDL Overlap Calibration

Launches a producer and dependent consumer with
`cudaLaunchAttributeProgrammaticStreamSerialization`. Every producer CTA
issues `griddepcontrol.launch_dependents`; every consumer CTA immediately
issues `griddepcontrol.wait`. `%globaltimer` timestamps quantify how much of
the producer tail overlaps resident consumer waiting before dependency release.

This is a composite interaction calibration and not an atomic benchmark.
