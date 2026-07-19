#!/usr/bin/env python3
"""Map performance-relevant dense kernel SASS instructions to manifest owners."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

from manifest_tool import ManifestError, load_manifest


ROOT = Path(__file__).resolve().parents[1]
INTERESTING = re.compile(
    r"\b(?:HGMMA|MUFU|FFMA|FADD|FMUL|FSETP|FSEL|F2FP?|IADD3|IMAD|ISETP|"
    r"SHFL|BAR|SYNCS|FENCE\.VIEW|UTMALDG|UTMASTG|UBLKCP|UTMACCTL|STSM|LDSM|"
    r"LDG|STG|LDS|STS|PREEXIT|ACQBULK)\b",
    re.IGNORECASE,
)


def owner_available(owner: str, manifest: dict) -> bool:
    entries = manifest["benchmarks"] + manifest.get("calibrations", [])
    families = {entry["family"] for entry in entries}
    category_families = {
        "tma": {"tma_load", "tma_store", "bulk_store", "tensormap_prefetch"},
        "matrix_movement": {"stmatrix", "ldmatrix"},
        "ordinary_memory": {"shared_load", "shared_store", "global_load", "global_store"},
        "pdl_calibration": {"pdl"},
    }
    return owner in families or bool(category_families.get(owner, set()) & families)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifest.json")
    parser.add_argument("--sass", type=Path, action="append", required=True)
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest.resolve())
        rules = [
            (re.compile(rule["regex"], re.IGNORECASE), rule["owner"])
            for rule in manifest.get("coverage_rules", [])
        ]
        files: list[Path] = []
        for supplied in args.sass:
            files.extend(sorted(supplied.rglob("*.sass")) if supplied.is_dir() else [supplied])
        if not files:
            raise ValueError("no SASS files selected")
        failures: list[str] = []
        observed: dict[str, int] = {}
        for path in files:
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if INTERESTING.search(line) is None:
                    continue
                owners = [owner for regex, owner in rules if regex.search(line)]
                if not owners:
                    failures.append(f"{path}:{number}: unclassified instruction: {line.strip()}")
                    continue
                for owner in owners:
                    observed[owner] = observed.get(owner, 0) + 1
                    if not owner_available(owner, manifest):
                        failures.append(
                            f"{path}:{number}: owner {owner!r} has no registered atom/calibration"
                        )
        if failures:
            print("\n".join(failures), file=sys.stderr)
            return 1
        for owner, count in sorted(observed.items()):
            print(f"{owner}: {count}")
        return 0
    except (ManifestError, OSError, ValueError) as exc:
        print(f"coverage check error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
