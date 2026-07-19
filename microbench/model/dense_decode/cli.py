"""Command-line interface for profile construction, prediction, and validation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

from .profile import ProfileError, build_profile, load_profile
from .schema import load_workload
from .simulator import predict


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PACKAGE_ROOT / "manifest.json"
DEFAULT_H800_CONFIG = PACKAGE_ROOT / "config" / "h800.json"


def _write_json(path: Path | None, value: Any) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path is None:
        sys.stdout.write(encoded)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(path)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load JSON {path}: {exc}") from exc


def _target_from_config(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    value = _load_json(path)
    target = dict(value.get("target", {})) if isinstance(value, dict) else {}
    peaks = value.get("peaks", {}) if isinstance(value, dict) else {}
    hardware = value.get("hardware", {}) if isinstance(value, dict) else {}
    target.update(hardware if isinstance(hardware, dict) else {})
    target["sm_count"] = args.sm_count or target.get("sm_count")
    target["sm_clock_mhz"] = args.sm_clock_mhz or target.get("sm_clock_mhz")
    target["l2_bytes"] = args.l2_bytes or target.get("l2_bytes")
    target["hbm_gbps"] = args.hbm_gbps or peaks.get("memory_gbps")
    return target


def command_build_profile(args: argparse.Namespace) -> int:
    target = _target_from_config(args.hardware_config, args)
    profile = build_profile(
        args.microbench_results,
        manifest_path=args.manifest,
        static_artifacts=args.static_artifacts,
        target=target,
    )
    _write_json(args.output, profile)
    return 0


def command_predict(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    workload_value = _load_json(args.workload)
    if not isinstance(workload_value, dict):
        raise ValueError("workload JSON must be an object")
    workload = load_workload(workload_value)
    result = predict(profile, workload, bootstrap=args.bootstrap).result
    _write_json(args.output, result)
    return 0


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    paths = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
    for candidate in paths:
        for line in candidate.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    records.append(value)
    return records


def command_validate(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    cases = _load_jsonl(args.cases)
    measured = _load_jsonl(args.e2e_results)
    measured_by_id = {
        str(item.get("case_id")): item for item in measured if item.get("case_id") is not None
    }
    comparisons = []
    errors = []
    for case in cases:
        workload = load_workload(case)
        prediction = predict(profile, workload, bootstrap=args.bootstrap).result
        actual = measured_by_id.get(workload.case_id)
        if actual is None:
            raise ValueError(f"missing e2e measurement for case_id={workload.case_id}")
        if "latency_us" in actual:
            measured_us = float(actual["latency_us"])
        elif "latency_ms" in actual:
            measured_us = float(actual["latency_ms"]) * 1000.0
        else:
            raise ValueError(f"e2e record {workload.case_id} lacks latency_us/latency_ms")
        predicted_us = float(prediction["predicted_e2e_us"]["p50"])
        if not math.isfinite(measured_us) or measured_us <= 0:
            raise ValueError("measured latency must be finite and positive")
        ape = abs(predicted_us - measured_us) / measured_us
        errors.append(ape)
        comparisons.append(
            {
                "case_id": workload.case_id,
                "predicted_us": predicted_us,
                "measured_us": measured_us,
                "signed_error_us": predicted_us - measured_us,
                "absolute_percentage_error": ape,
                "prediction": prediction,
            }
        )
    ordered = sorted(errors)
    p90 = ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)] if ordered else 0.0
    summary = {
        "schema_version": 1,
        "case_count": len(comparisons),
        "mape": sum(errors) / len(errors) if errors else 0.0,
        "p90_absolute_percentage_error": p90,
        "target": {"mape": 0.10, "p90_absolute_percentage_error": 0.15},
        "accepted": bool(errors) and sum(errors) / len(errors) <= 0.10 and p90 <= 0.15,
        "comparisons": comparisons,
    }
    _write_json(args.output, summary)
    return 0 if summary["accepted"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FlashMLA dense-decode H800 model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile = subparsers.add_parser("build-profile")
    profile.add_argument("--microbench-results", required=True, type=Path)
    profile.add_argument("--static-artifacts", type=Path, default=None)
    profile.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    profile.add_argument("--hardware-config", type=Path, default=DEFAULT_H800_CONFIG)
    profile.add_argument("--sm-count", type=int, default=None)
    profile.add_argument("--sm-clock-mhz", type=float, default=None)
    profile.add_argument("--l2-bytes", type=int, default=None)
    profile.add_argument("--hbm-gbps", type=float, default=None)
    profile.add_argument("--output", required=True, type=Path)
    profile.set_defaults(handler=command_build_profile)

    prediction = subparsers.add_parser("predict")
    prediction.add_argument("--profile", required=True, type=Path)
    prediction.add_argument("--workload", required=True, type=Path)
    prediction.add_argument("--bootstrap", type=int, default=0)
    prediction.add_argument("--output", type=Path, default=None)
    prediction.set_defaults(handler=command_predict)

    validation = subparsers.add_parser("validate")
    validation.add_argument("--profile", required=True, type=Path)
    validation.add_argument("--cases", required=True, type=Path)
    validation.add_argument("--e2e-results", required=True, type=Path)
    validation.add_argument("--bootstrap", type=int, default=0)
    validation.add_argument("--output", type=Path, default=None)
    validation.set_defaults(handler=command_validate)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if getattr(args, "bootstrap", 0) < 0:
        parser.error("--bootstrap must be non-negative")
    try:
        return int(args.handler(args))
    except (OSError, ValueError, ProfileError) as exc:
        parser.error(str(exc))
    return 2
