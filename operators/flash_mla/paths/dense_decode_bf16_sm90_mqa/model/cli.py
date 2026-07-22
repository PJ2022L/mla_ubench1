"""CLI for DAG construction, atom-only prediction, and residual validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

from .cost_database import CostDatabase, CostDatabaseError
from .dag import AtomMap, build_dense_decode_dag
from .schema import load_kernel_resources, load_workload
from .simulator import simulate


MODEL_ROOT = Path(__file__).resolve().parent
DEFAULT_ATOM_MAP = MODEL_ROOT / "atom_map.json"


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load JSON {path}: {exc}") from exc


def _write_json(path: Path | None, value: Any) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path is None:
        sys.stdout.write(encoded)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _inputs(args: argparse.Namespace):
    workload_value = _load_json(args.workload)
    if not isinstance(workload_value, dict):
        raise ValueError("workload JSON must be an object")
    resources_value = _load_json(args.kernel_resources) if args.kernel_resources else None
    if resources_value is not None and not isinstance(resources_value, dict):
        raise ValueError("kernel resources JSON must be an object")
    workload = load_workload(workload_value)
    resources = load_kernel_resources(resources_value)
    atom_map = AtomMap.load(args.atom_map)
    return workload, resources, atom_map


def command_build_dag(args: argparse.Namespace) -> int:
    workload, resources, atom_map = _inputs(args)
    _write_json(args.output, build_dense_decode_dag(workload, resources, atom_map).to_json())
    return 0


def command_predict(args: argparse.Namespace) -> int:
    workload, resources, atom_map = _inputs(args)
    dag = build_dense_decode_dag(workload, resources, atom_map)
    result = simulate(dag, CostDatabase(args.microbench_root), resources).result
    _write_json(args.output, result)
    return 0


def command_validate_calibration(args: argparse.Namespace) -> int:
    # The calibration reader is intentionally absent from all prediction
    # modules and imported only for this diagnostic subcommand.
    from .calibration import validate_calibration

    resources_value = _load_json(args.kernel_resources) if args.kernel_resources else None
    resources = load_kernel_resources(resources_value)
    report = validate_calibration(
        args.microbench_root, args.calibration_root, resources
    )
    _write_json(args.output, report)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FlashMLA dense-decode atom-DAG model")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-dag")
    build.add_argument("--workload", required=True, type=Path)
    build.add_argument("--kernel-resources", type=Path, default=None)
    build.add_argument("--atom-map", type=Path, default=DEFAULT_ATOM_MAP)
    build.add_argument("--output", required=True, type=Path)
    build.set_defaults(handler=command_build_dag)

    predict = sub.add_parser("predict")
    predict.add_argument("--microbench-root", required=True, type=Path)
    predict.add_argument("--kernel-resources", required=True, type=Path)
    predict.add_argument("--workload", required=True, type=Path)
    predict.add_argument("--atom-map", type=Path, default=DEFAULT_ATOM_MAP)
    predict.add_argument("--output", required=True, type=Path)
    predict.set_defaults(handler=command_predict)

    validate = sub.add_parser("validate-calibration")
    validate.add_argument("--microbench-root", required=True, type=Path)
    validate.add_argument("--calibration-root", required=True, type=Path)
    validate.add_argument("--kernel-resources", type=Path, default=None)
    validate.add_argument("--output", required=True, type=Path)
    validate.set_defaults(handler=command_validate_calibration)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.handler(args))
    except (OSError, ValueError, KeyError, CostDatabaseError) as exc:
        parser.error(str(exc))
    return 2
