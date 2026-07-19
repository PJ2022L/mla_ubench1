#!/usr/bin/env python3
"""Compile and validate manifest-declared PTX/SASS without executing kernels."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from manifest_tool import ManifestError, entries, load_manifest


ROOT = Path(__file__).resolve().parents[1]


class StaticError(RuntimeError):
    pass


def run(command: list[str], *, stdout: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise StaticError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout}{completed.stderr}"
        )
    output = completed.stdout
    if stdout is not None:
        stdout.write_text(output, encoding="utf-8")
    return output


def require_patterns(text: str, patterns: list[str], where: str) -> None:
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE | re.MULTILINE) is None:
            raise StaticError(f"required pattern {pattern!r} absent from {where}")


def reject_cutlass_cute() -> None:
    include = re.compile(
        r'^\s*#\s*include\s*[<"][^">]*(?:cutlass|cute)(?:[/">])',
        re.IGNORECASE | re.MULTILINE,
    )
    roots = [ROOT / "common", ROOT / "compute", ROOT / "memory", ROOT / "model" / "calibration"]
    for base in roots:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.suffix not in {".h", ".hpp", ".cuh", ".cu"}:
                continue
            if include.search(path.read_text(encoding="utf-8")):
                raise StaticError(f"CUTLASS/CUTE include detected: {path}")


def minimum_count(binary: str, sass: str, ptx: str) -> None:
    requirements: dict[str, list[tuple[str, int, str]]] = {
        "tma_load_q_bf16_rank4": [(r"cp\.async\.bulk\.tensor\.4d", 9, ptx)],
        "kq_first_page_bf16": [(r"UTMALDG\.4D", 9, sass), (r"HGMMA\.", 36, sass)],
        "kq_first_page_fp16": [(r"UTMALDG\.4D", 9, sass), (r"HGMMA\.", 36, sass)],
        "kq_steady_page_bf16": [(r"UTMALDG\.4D", 9, sass), (r"HGMMA\.", 36, sass)],
        "kq_steady_page_fp16": [(r"UTMALDG\.4D", 9, sass), (r"HGMMA\.", 36, sass)],
    }
    for pattern, minimum, text in requirements.get(binary, []):
        count = len(re.findall(pattern, text, re.IGNORECASE))
        if count < minimum:
            raise StaticError(
                f"{binary}: pattern {pattern!r} count {count} is below {minimum}"
            )
    exact_requirements: dict[str, list[tuple[str, int, str]]] = {
        "page_pair_transition_bf16": [
            (r"HGMMA\.", 88, sass), (r"BAR\.ARV", 4, sass),
            (r"STSM", 8, sass), (r"LDSM", 8, sass),
        ],
        "page_pair_transition_fp16": [
            (r"HGMMA\.", 88, sass), (r"BAR\.ARV", 4, sass),
            (r"STSM", 8, sass), (r"LDSM", 8, sass),
        ],
        "metadata_stage": [(r"SHFL", 5, sass)],
    }
    for pattern, expected, text in exact_requirements.get(binary, []):
        count = len(re.findall(pattern, text, re.IGNORECASE))
        if count != expected:
            raise StaticError(
                f"{binary}: pattern {pattern!r} count {count} != {expected}"
            )


def validate_resources(resources: str, ptx: str, sass: str, binary: str) -> None:
    if re.search(r"(^|[^A-Z0-9_])(LDL|STL)([^A-Z0-9_]|$)", sass, re.IGNORECASE):
        raise StaticError(f"{binary}: local-memory LDL/STL found")
    if re.search(r"(^|[\s,])(STACK|LOCAL):\s*[1-9][0-9]*", resources):
        raise StaticError(f"{binary}: non-zero STACK/LOCAL usage")
    if re.search(r"^\s*\.local(?:\s|$)", ptx, re.MULTILINE):
        raise StaticError(f"{binary}: PTX local-memory declaration found")
    require_patterns(resources, [r"STACK:\s*[0-9]+", r"LOCAL:\s*[0-9]+"], binary)


def validate_hgmma(entry: dict, sass: str) -> None:
    if entry["family"] != "wgmma":
        return
    expected = re.compile(entry["sass_patterns"][0], re.IGNORECASE)
    unexpected = [line for line in sass.splitlines() if "HGMMA." in line.upper() and not expected.search(line)]
    if unexpected:
        raise StaticError(
            f"{entry['binary']}: unexpected HGMMA instructions: {unexpected[:4]}"
        )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_entry(entry: dict, out: Path, disassembler: str) -> None:
    binary = entry["binary"]
    source = ROOT / entry["source"]
    ptx_path = out / f"{binary}.ptx"
    cubin_path = out / f"{binary}.cubin"
    sass_path = out / f"{binary}.sass"
    resources_path = out / f"{binary}.resources.txt"
    hashes_path = out / f"{binary}.sha256"
    flags = ["-std=c++17", "-O3", "-arch=sm_90a", f"-I{ROOT}", f"-I{ROOT / 'common'}"]

    print(f"[static] {binary}", file=sys.stderr, flush=True)
    run(["nvcc", *flags, "--ptx", str(source), "-o", str(ptx_path)])
    run(["nvcc", *flags, "--cubin", str(source), "-o", str(cubin_path)])
    if disassembler == "nvdisasm":
        run([disassembler, "--print-line-info", str(cubin_path)], stdout=sass_path)
    else:
        run([disassembler, "--dump-sass", str(cubin_path)], stdout=sass_path)
    run(["cuobjdump", "--dump-resource-usage", str(cubin_path)], stdout=resources_path)

    ptx = ptx_path.read_text(encoding="utf-8")
    sass = sass_path.read_text(encoding="utf-8")
    resources = resources_path.read_text(encoding="utf-8")
    require_patterns(ptx, entry.get("ptx_patterns", []), f"{binary}.ptx")
    require_patterns(sass, entry.get("sass_patterns", []), f"{binary}.sass")
    validate_hgmma(entry, sass)
    validate_resources(resources, ptx, sass, binary)
    minimum_count(binary, sass, ptx)

    hashes = "\n".join(
        f"{sha256(path)}  {path}"
        for path in (ptx_path, cubin_path, sass_path, resources_path)
    ) + "\n"
    hashes_path.write_text(hashes, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifest.json")
    parser.add_argument("--kind", choices=("atom", "calibration", "all"), default="atom")
    parser.add_argument("bench", nargs="*")
    args = parser.parse_args(argv)
    try:
        for tool in ("nvcc", "cuobjdump"):
            if shutil.which(tool) is None:
                raise StaticError(f"required tool not found: {tool}")
        disassembler = "nvdisasm" if shutil.which("nvdisasm") else "cuobjdump"
        manifest = load_manifest(args.manifest.resolve())
        selected = entries(manifest, args.kind)
        if args.bench:
            requested = set(args.bench)
            selected = [entry for entry in selected if entry["binary"] in requested]
            unknown = requested - {entry["binary"] for entry in selected}
            if unknown:
                raise StaticError(f"unknown selected benchmarks: {sorted(unknown)}")
        reject_cutlass_cute()
        out = ROOT / "build" / "static"
        out.mkdir(parents=True, exist_ok=True)
        for entry in selected:
            check_entry(entry, out, disassembler)
        print(f"static artifacts: {out}", file=sys.stderr)
        return 0
    except (ManifestError, StaticError, OSError) as exc:
        print(f"static check error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
