#!/usr/bin/env python3
"""Validate and query microbench/manifest.json for build scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "manifest.json"


class ManifestError(ValueError):
    pass


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ManifestError("manifest schema_version must be 1")
    benchmarks = value.get("benchmarks")
    calibrations = value.get("calibrations", [])
    if not isinstance(benchmarks, list) or not isinstance(calibrations, list):
        raise ManifestError("benchmarks and calibrations must be arrays")
    seen_binary: set[str] = set()
    seen_result: set[str] = set()
    for kind, entries in (("atom", benchmarks), ("calibration", calibrations)):
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ManifestError(f"{kind} entry {index} must be an object")
            required = {
                "binary", "source", "result_name", "category", "family",
                "variant", "scan_family", "ptx_patterns", "sass_patterns",
                "source_anchors",
            }
            missing = required - set(entry)
            if missing:
                raise ManifestError(f"{kind} entry {index} missing {sorted(missing)}")
            binary = entry["binary"]
            result_name = entry["result_name"]
            if not isinstance(binary, str) or not binary or "/" in binary:
                raise ManifestError(f"invalid flat binary name: {binary!r}")
            if binary in seen_binary:
                raise ManifestError(f"duplicate binary: {binary}")
            if result_name in seen_result:
                raise ManifestError(f"duplicate result_name: {result_name}")
            seen_binary.add(binary)
            seen_result.add(result_name)
            source = ROOT / entry["source"]
            if not source.is_file():
                raise ManifestError(f"source does not exist: {source}")
            relative = Path(entry["source"])
            if kind == "atom":
                if relative.parts[0] not in {"memory", "compute"}:
                    raise ManifestError(f"atomic source must be below memory/ or compute/: {relative}")
                if len(relative.parts) != 4 or relative.name != "benchmark.cu":
                    raise ManifestError(
                        "atomic source must use <category>/<family>/<variant>/benchmark.cu: "
                        f"{relative}"
                    )
                if entry["category"] != relative.parts[0]:
                    raise ManifestError(f"category/path mismatch for {binary}")
                if entry["family"] != relative.parts[1]:
                    raise ManifestError(f"family/path mismatch for {binary}")
            elif relative.parts[:2] != ("model", "calibration"):
                raise ManifestError(f"calibration source must be below model/calibration: {relative}")
            readme = source.with_name("README.md")
            if not readme.is_file():
                raise ManifestError(f"leaf README does not exist: {readme}")
            for field in ("ptx_patterns", "sass_patterns", "source_anchors"):
                if not isinstance(entry[field], list):
                    raise ManifestError(f"{binary}.{field} must be an array")

    declared_sources = {
        str(Path(entry["source"]))
        for entry in benchmarks + calibrations
    }
    actual_sources = {
        str(path.relative_to(ROOT))
        for base in (ROOT / "memory", ROOT / "compute", ROOT / "model" / "calibration")
        if base.exists()
        for path in base.rglob("benchmark.cu")
    }
    unregistered = sorted(actual_sources - declared_sources)
    stale = sorted(declared_sources - actual_sources)
    if unregistered:
        raise ManifestError(f"unregistered benchmark leaves: {unregistered}")
    if stale:
        raise ManifestError(f"manifest sources are not benchmark leaves: {stale}")
    scan_parameters = value.get("scan_parameters")
    if not isinstance(scan_parameters, dict):
        raise ManifestError("scan_parameters must be an object")
    common = scan_parameters.get("common")
    families = scan_parameters.get("families")
    overrides = scan_parameters.get("bench_overrides")
    if not isinstance(common, list) or not isinstance(families, dict) or not isinstance(overrides, dict):
        raise ManifestError("scan_parameters requires common, families, and bench_overrides")
    missing_families = sorted({
        entry.get("scan_family", entry["family"])
        for entry in benchmarks + calibrations
    } - set(families))
    if missing_families:
        raise ManifestError(f"scan parameter contracts missing families: {missing_families}")
    unknown_overrides = sorted(set(overrides) - seen_binary)
    if unknown_overrides:
        raise ManifestError(f"scan parameter overrides have unknown binaries: {unknown_overrides}")
    groups = value.get("groups", {})
    if not isinstance(groups, dict):
        raise ManifestError("groups must be an object")
    known = seen_binary
    for name, members in groups.items():
        if not isinstance(name, str) or not isinstance(members, list) or not members:
            raise ManifestError("group names must map to non-empty arrays")
        unknown = set(members) - known
        if unknown:
            raise ManifestError(f"group {name} has unknown binaries: {sorted(unknown)}")
    return value


def entries(value: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    if kind == "atom":
        return value["benchmarks"]
    if kind == "calibration":
        return value.get("calibrations", [])
    return value["benchmarks"] + value.get("calibrations", [])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--kind", choices=("atom", "calibration", "all"), default="atom")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    subparsers.add_parser("list")
    source = subparsers.add_parser("source")
    source.add_argument("binary")
    result = subparsers.add_parser("result-name")
    result.add_argument("binary")
    args = parser.parse_args(argv)
    try:
        value = load_manifest(args.manifest.resolve())
        selected = entries(value, args.kind)
        if args.command == "validate":
            return 0
        if args.command == "list":
            print(" ".join(entry["binary"] for entry in selected))
            return 0
        matching = [entry for entry in selected if entry["binary"] == args.binary]
        if len(matching) != 1:
            raise ManifestError(f"unknown binary for kind={args.kind}: {args.binary}")
        print(matching[0]["source" if args.command == "source" else "result_name"])
        return 0
    except ManifestError as exc:
        print(f"manifest error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
