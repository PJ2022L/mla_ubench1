# Dense Softmax Stage BF16 Calibration

Two-warpgroup register-heavy online-softmax stage with 32 score fragments per
lane, max/sum butterfly reductions, EX2, reciprocal, rescale arithmetic, and
BF16 probability conversion. This is an interaction stage, not an atom.
