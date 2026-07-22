#!/usr/bin/env python3
"""Compatibility entry point for the dense-decode atom-DAG predictor.

The former additive/calibration-cost composer is intentionally disabled: a
dense prediction must come from generic microbenchmark CSVs and the source DAG.
"""

from __future__ import annotations

import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from operators.flash_mla.paths.dense_decode_bf16_sm90_mqa.model.cli import main as model_main


def compose(*_args, **_kwargs):
    raise RuntimeError(
        "additive cycle dictionaries are no longer a supported model; use "
        "build-dag/predict with --microbench-root and --workload"
    )


def main(argv: list[str] | None = None) -> int:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    if "--cycles-json" in arguments or "--profile" in arguments:
        raise SystemExit(
            "legacy calibration/additive inputs are disabled; invoke the atom-DAG "
            "predict interface instead"
        )
    if not arguments or arguments[0] not in {
        "build-dag", "predict", "validate-calibration"
    }:
        arguments.insert(0, "predict")
    return model_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
