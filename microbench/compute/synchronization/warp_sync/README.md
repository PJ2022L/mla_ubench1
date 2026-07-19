# WARPSYNC

Contains full-mask `bar.warp.sync` inline PTX with a runtime member mask. SM90a ptxas elides the instruction on this converged path; the static artifact therefore records no executable hardware instruction, which is the behavior the dense model must use for the same converged call site.
