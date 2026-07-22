#!/usr/bin/env python3
"""Build dense calibration probes for SM90a without executing them."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import re


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[4]
sys.path.insert(0, str(REPO / "microbench" / "scripts"))

from family_runner import sha256, source_closure_sha256


def load_probes() -> list[dict]:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    return manifest["probes"]


def run(command: list[str], log: Path, output: Path | None = None) -> None:
    started = dt.datetime.now(dt.timezone.utc)
    completed = subprocess.run(
        command, cwd=REPO, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    ended = dt.datetime.now(dt.timezone.utc)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        f"command={shlex.join(command)}\n"
        f"started_utc={started.isoformat()}\n"
        f"ended_utc={ended.isoformat()}\n"
        f"duration_seconds={(ended - started).total_seconds():.6f}\n"
        f"returncode={completed.returncode}\n\n{completed.stdout}",
        encoding="utf-8",
    )
    if completed.returncode:
        raise RuntimeError(f"command failed; see {log}")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(completed.stdout, encoding="utf-8")


def validate_static(probe: dict) -> None:
    name = probe["id"]
    ptx = (ROOT / "build/ptx" / f"{name}.ptx").read_text(encoding="utf-8")
    sass = (ROOT / "build/sass" / f"{name}.sass").read_text(encoding="utf-8")
    resources = (ROOT / "build/resources" / f"{name}.txt").read_text(encoding="utf-8")
    for pattern in probe.get("ptx_patterns", []):
        if re.search(pattern, ptx, re.IGNORECASE | re.MULTILINE) is None:
            raise RuntimeError(f"{name}: required PTX pattern absent: {pattern}")
    for pattern in probe.get("sass_patterns", []):
        if re.search(pattern, sass, re.IGNORECASE | re.MULTILINE) is None:
            raise RuntimeError(f"{name}: required SASS pattern absent: {pattern}")
    if re.search(r"(^|[^A-Z0-9_])(LDL|STL)([^A-Z0-9_]|$)", sass,
                 re.IGNORECASE):
        raise RuntimeError(f"{name}: local-memory LDL/STL found")
    if re.search(r"(^|[\s,])(STACK|LOCAL):\s*[1-9][0-9]*", resources):
        raise RuntimeError(f"{name}: non-zero STACK/LOCAL resource usage")
    if re.search(r"^\s*\.local(?:\s|$)", ptx, re.MULTILINE):
        raise RuntimeError(f"{name}: PTX local declaration found")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--nvcc", default="nvcc")
    args = parser.parse_args()

    selected = set(args.probe)
    probes = [p for p in load_probes() if not selected or p["id"] in selected]
    missing = selected - {p["id"] for p in probes}
    if missing:
        raise SystemExit(f"unknown probes: {sorted(missing)}")

    for kind in ("bin", "ptx", "cubin", "sass", "resources", "logs"):
        (ROOT / "build" / kind).mkdir(parents=True, exist_ok=True)
    for probe in probes:
        name = probe["id"]
        source = ROOT / probe["source"]
        common = [
            args.nvcc, "-std=c++17", "-O3",
            "-gencode=arch=compute_90a,code=sm_90a",
            f"-I{ROOT}", f"-I{ROOT / 'common'}",
            f"-I{REPO / 'microbench'}", f"-I{REPO / 'microbench' / 'common'}",
        ]
        commands = [
            ("binary", [*common, str(source), "-o", str(ROOT / "build/bin" / name), "-lcuda"]),
            ("ptx", [*common, "--ptx", str(source), "-o", str(ROOT / "build/ptx" / f"{name}.ptx")]),
            ("cubin", [*common, "--cubin", str(source), "-o", str(ROOT / "build/cubin" / f"{name}.cubin")]),
        ]
        for label, command in commands:
            if args.dry_run:
                print(shlex.join(command))
            else:
                run(command, ROOT / "build/logs" / f"build-{name}-{label}.log")
        if args.dry_run:
            continue
        cubin = ROOT / "build/cubin" / f"{name}.cubin"
        disassembler = shutil.which("nvdisasm") or shutil.which("cuobjdump")
        if not disassembler:
            raise RuntimeError("nvdisasm or cuobjdump is required")
        disasm = ([disassembler, "--print-line-info", str(cubin)]
                  if Path(disassembler).name == "nvdisasm"
                  else [disassembler, "--dump-sass", str(cubin)])
        run(disasm, ROOT / "build/logs" / f"build-{name}-sass.log",
            ROOT / "build/sass" / f"{name}.sass")
        run(["cuobjdump", "--dump-resource-usage", str(cubin)],
            ROOT / "build/logs" / f"build-{name}-resources.log",
            ROOT / "build/resources" / f"{name}.txt")
        validate_static(probe)
        metadata = {
            "id": name,
            "source_closure_sha256": source_closure_sha256(
                source, [ROOT, REPO / "microbench"]),
            "binary_sha256": sha256(ROOT / "build/bin" / name),
            "ptx_sha256": sha256(ROOT / "build/ptx" / f"{name}.ptx"),
            "cubin_sha256": sha256(cubin),
            "sass_sha256": sha256(ROOT / "build/sass" / f"{name}.sass"),
            "architecture": "sm_90a",
            "built_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        (ROOT / "build/resources" / f"{name}.build.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
