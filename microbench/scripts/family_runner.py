#!/usr/bin/env python3
"""Shared implementation for family-local build.py and sweep.py wrappers."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable


COMMON_CSV_COLUMNS = [
    "name",
    "params_json",
    "gpu_uuid",
    "sm_clock_mhz",
    "memory_clock_mhz",
    "latency_value",
    "latency_unit",
    "initiation_interval_cycles",
    "throughput_value",
    "throughput_unit",
    "memory_bandwidth_value",
    "memory_bandwidth_unit",
    "hardware_utilization",
    "p10",
    "p50",
    "p90",
    "sample_count",
    "source_sha256",
    "sass_sha256",
    "blocks",
]

WGMMA_CSV_COLUMNS = [
    "name",
    "args",
    "gpu_name",
    "gpu_uuid",
    "sm_clock_mhz",
    "memory_clock_mhz",
    "latency_value",
    "latency_unit",
    "initiation_interval_cycles",
    "throughput_value",
    "throughput_unit",
    "p10",
    "p50",
    "p90",
    "sample_count",
    "source_sha256",
    "sass_sha256",
]

WGMMA_ARG_KEYS = (
    "iters",
    "warmup",
    "samples",
    "blocks",
    "resolved_blocks",
    "warpgroups",
    "group_size",
    "depth",
)


def csv_columns_for_family(family: str) -> list[str]:
    if family == "wgmma":
        return list(WGMMA_CSV_COLUMNS)
    return list(COMMON_CSV_COLUMNS)


def microbench_root(family_dir: Path) -> Path:
    family_dir = family_dir.resolve()
    for parent in (family_dir, *family_dir.parents):
        if (parent / "manifest.json").is_file() and (parent / "common").is_dir():
            return parent
    raise RuntimeError(f"cannot locate microbench root above {family_dir}")


def load_manifest(root: Path) -> dict[str, Any]:
    with (root / "manifest.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def family_entries(root: Path, family_dir: Path) -> list[dict[str, Any]]:
    prefix = family_dir.resolve().relative_to(root.resolve()).as_posix() + "/"
    entries = [
        item
        for item in load_manifest(root)["benchmarks"]
        if item["source"].startswith(prefix)
    ]
    return sorted(entries, key=lambda item: item["id"])


def ensure_build_dirs(family_dir: Path) -> dict[str, Path]:
    paths = {
        name: family_dir / "build" / name
        for name in ("bin", "ptx", "cubin", "sass", "resources", "logs", "raw")
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


QUOTED_INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def source_closure(source: Path, include_roots: list[Path]) -> list[Path]:
    roots = [root.resolve() for root in include_roots]
    pending = [source.resolve()]
    visited: set[Path] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        if not current.is_file() or not any(_under(current, root) for root in roots):
            raise RuntimeError(f"source closure escaped its roots: {current}")
        visited.add(current)
        text_value = current.read_text(encoding="utf-8")
        for include in QUOTED_INCLUDE.findall(text_value):
            candidates = [current.parent / include]
            candidates.extend(root / include for root in roots)
            resolved = next(
                (candidate.resolve() for candidate in candidates
                 if candidate.is_file() and
                 any(_under(candidate.resolve(), root) for root in roots)),
                None,
            )
            if resolved is not None and resolved not in visited:
                pending.append(resolved)
    return sorted(visited, key=lambda path: str(path))


def source_closure_sha256(source: Path, include_roots: list[Path]) -> str:
    roots = [root.resolve() for root in include_roots]
    digest = hashlib.sha256()
    for path in source_closure(source, roots):
        owner_index, relative = next(
            (index, path.relative_to(root))
            for index, root in enumerate(roots)
            if _under(path, root)
        )
        label = f"{owner_index}:{relative.as_posix()}".encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(label).to_bytes(8, "little"))
        digest.update(label)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def build_metadata_path(paths: dict[str, Path], atom_id: str) -> Path:
    return paths["resources"] / f"{atom_id}.build.json"


def validate_build_provenance(root: Path, entry: dict[str, Any],
                              paths: dict[str, Path]) -> None:
    atom_id = entry["id"]
    metadata_path = build_metadata_path(paths, atom_id)
    if not metadata_path.is_file():
        raise RuntimeError(
            f"{atom_id}: missing build provenance; rebuild the family")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "source_closure_sha256": source_closure_sha256(
            root / entry["source"], [root]),
        "binary_sha256": sha256(paths["bin"] / atom_id),
        "ptx_sha256": sha256(paths["ptx"] / f"{atom_id}.ptx"),
        "cubin_sha256": sha256(paths["cubin"] / f"{atom_id}.cubin"),
        "sass_sha256": sha256(paths["sass"] / f"{atom_id}.sass"),
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise RuntimeError(
                f"{atom_id}: stale build provenance for {field}; rebuild "
                "before running a sweep")


def run_logged(command: list[str], log_handle: Any, dry_run: bool,
               output_path: Path | None = None) -> None:
    started = time.time()
    start_record = {
        "event": "command",
        "argv": command,
        "shell": shlex.join(command),
        "started_unix_s": started,
    }
    if output_path is not None:
        start_record["output_path"] = str(output_path)
    log_handle.write(json.dumps(start_record, sort_keys=True) + "\n")
    log_handle.flush()
    if dry_run:
        return
    if output_path is None:
        completed = subprocess.run(
            command, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False)
    else:
        with output_path.open("w", encoding="utf-8") as output_handle:
            completed = subprocess.run(
                command, text=True, stdout=output_handle,
                stderr=subprocess.PIPE, check=False)
    completion_record = {
        "event": "command_complete",
        "argv": command,
        "duration_s": time.time() - started,
        "returncode": completed.returncode,
    }
    if output_path is not None:
        completion_record["output_path"] = str(output_path)
    if completed.stdout:
        completion_record["stdout"] = completed.stdout
    if completed.stderr:
        completion_record["stderr"] = completed.stderr
    log_handle.write(json.dumps(completion_record, sort_keys=True) + "\n")
    log_handle.flush()
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)


def build_family(family_dir: Path, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--cuobjdump", default="cuobjdump")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args(argv)
    del args.jobs  # Reserved for a future parallel implementation.

    root = microbench_root(family_dir)
    entries = family_entries(root, family_dir)
    paths = ensure_build_dirs(family_dir)
    if not entries:
        raise RuntimeError(f"manifest has no entries for {family_dir}")
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    log_path = paths["logs"] / f"build-{timestamp}.jsonl"
    common_flags = [
        args.nvcc,
        "-std=c++17",
        "-O3",
        "--use_fast_math",
        "-arch=compute_90a",
        "-code=sm_90a",
        "-lineinfo",
        f"-I{root}",
    ]
    with log_path.open("w", encoding="utf-8") as log_handle:
        for entry in entries:
            source = root / entry["source"]
            name = entry["id"]
            outputs = {
                "bin": paths["bin"] / name,
                "ptx": paths["ptx"] / f"{name}.ptx",
                "cubin": paths["cubin"] / f"{name}.cubin",
                "sass": paths["sass"] / f"{name}.sass",
                "resources": paths["resources"] / f"{name}.txt",
            }
            run_logged([*common_flags, str(source), "-o", str(outputs["bin"]),
                        "-lcuda"],
                       log_handle, args.dry_run)
            run_logged([*common_flags, "--ptx", str(source), "-o", str(outputs["ptx"])],
                       log_handle, args.dry_run)
            run_logged([*common_flags, "--cubin", str(source), "-o", str(outputs["cubin"])],
                       log_handle, args.dry_run)
            run_logged(
                [args.cuobjdump, "--dump-sass", str(outputs["cubin"])],
                log_handle, args.dry_run, outputs["sass"])
            run_logged(
                [args.cuobjdump, "--dump-resource-usage",
                 str(outputs["cubin"])],
                log_handle, args.dry_run, outputs["resources"])
            if not args.dry_run:
                metadata = {
                    "id": name,
                    "source_closure_sha256": source_closure_sha256(
                        source, [root]),
                    "binary_sha256": sha256(outputs["bin"]),
                    "ptx_sha256": sha256(outputs["ptx"]),
                    "cubin_sha256": sha256(outputs["cubin"]),
                    "sass_sha256": sha256(outputs["sass"]),
                    "architecture": "sm_90a",
                    "built_unix_s": time.time(),
                }
                build_metadata_path(paths, name).write_text(
                    json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    print(log_path)
    return 0


def product(params: dict[str, list[Any]]) -> Iterable[dict[str, Any]]:
    keys = list(params)
    for values in itertools.product(*(params[key] for key in keys)):
        yield dict(zip(keys, values))


def sweep_parameter_sets(
    defaults: dict[str, list[Any]],
    benchmark: dict[str, Any],
) -> Iterable[dict[str, Any]]:
    cases = benchmark.get("cases")
    if cases is None:
        grid = dict(defaults)
        grid.update(benchmark)
        yield from product(grid)
        return
    if set(benchmark) != {"cases"} or not isinstance(cases, list) or not cases:
        raise ValueError("sweep cases must be a non-empty exclusive list")
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("each sweep case must be an object")
        grid = dict(defaults)
        grid.update(case)
        yield from product(grid)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * fraction))
    return ordered[index]


def query_gpu_metadata(device: int) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        f"--id={device}",
        "--query-gpu=name,uuid,clocks.sm,clocks.mem",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, text=True, capture_output=True,
                               check=True)
    fields = [field.strip() for field in completed.stdout.strip().split(",")]
    if len(fields) != 4:
        raise RuntimeError("unexpected nvidia-smi metadata output")
    return {
        "gpu_name": fields[0],
        "gpu_uuid": fields[1],
        "sm_clock_mhz": float(fields[2]),
        "memory_clock_mhz": float(fields[3]),
    }


def nested_value(record: dict[str, Any], field: str) -> Any:
    value: Any = record
    for part in field.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def reject_raw_param_arrays(value: Any, path: str = "params") -> None:
    if isinstance(value, list):
        raise ValueError(
            f"{path}: raw arrays belong in build/raw, not result.csv params")
    if isinstance(value, dict):
        for key, child in value.items():
            reject_raw_param_arrays(child, f"{path}.{key}")


def flatten_result(record: dict[str, Any], entry: dict[str, Any],
                   source: Path, sass_path: Path) -> dict[str, Any]:
    expected = {"name", "params", "latency", "throughput",
                "memory_bandwidth", "hardware_utilization"}
    if set(record) != expected:
        raise ValueError(f"{entry['id']}: JSON keys must be exactly {sorted(expected)}")
    if record["name"] != entry["id"]:
        raise ValueError(
            f"{entry['id']}: JSON name is {record['name']!r}, expected exact ID")
    params = record["params"]
    reject_raw_param_arrays(params)
    missing_params = sorted(
        parameter for parameter in entry.get("parameters", [])
        if parameter not in params)
    if missing_params:
        raise ValueError(
            f"{entry['id']}: result is missing manifest parameters "
            f"{missing_params}")
    if "initiation_interval_cycles" not in params:
        fallback = nested_value(record, "latency.value")
        if fallback is None:
            raise ValueError(
                f"{entry['id']}: no measured initiation interval or latency fallback")
        params["initiation_interval_cycles"] = fallback
        params["initiation_interval_source"] = "latency_fallback"
    else:
        params["initiation_interval_source"] = "throughput_clock_measurement"
    latency_samples = nested_value(record, "latency.samples") or []
    family = str(entry.get("family", ""))
    if family == "wgmma":
        args = {
            key: params[key]
            for key in WGMMA_ARG_KEYS
            if key in params
        }
        return {
            "name": record["name"],
            "args": json.dumps(args, sort_keys=True, separators=(",", ":")),
            "gpu_name": params.get("gpu_name", params.get("gpu", "")),
            "gpu_uuid": params.get("gpu_uuid", ""),
            "sm_clock_mhz": params.get("sm_clock_mhz", ""),
            "memory_clock_mhz": params.get("memory_clock_mhz", ""),
            "latency_value": nested_value(record, "latency.value"),
            "latency_unit": nested_value(record, "latency.unit"),
            "initiation_interval_cycles": params["initiation_interval_cycles"],
            "throughput_value": nested_value(record, "throughput.value"),
            "throughput_unit": nested_value(record, "throughput.unit"),
            "p10": percentile(latency_samples, 0.10),
            "p50": percentile(latency_samples, 0.50),
            "p90": percentile(latency_samples, 0.90),
            "sample_count": len(latency_samples),
            "source_sha256": source_closure_sha256(
                source, [microbench_root(source.parent)]),
            "sass_sha256": sha256(sass_path) if sass_path.is_file() else "",
        }

    row = {column: "" for column in csv_columns_for_family(family)}
    row.update({
        "name": record["name"],
        "params_json": json.dumps(params, sort_keys=True, separators=(",", ":")),
        "gpu_uuid": params.get("gpu_uuid", ""),
        "sm_clock_mhz": params.get("sm_clock_mhz", ""),
        "memory_clock_mhz": params.get("memory_clock_mhz", ""),
        "latency_value": nested_value(record, "latency.value"),
        "latency_unit": nested_value(record, "latency.unit"),
        "initiation_interval_cycles": params["initiation_interval_cycles"],
        "throughput_value": nested_value(record, "throughput.value"),
        "throughput_unit": nested_value(record, "throughput.unit"),
        "memory_bandwidth_value": nested_value(record, "memory_bandwidth.value"),
        "memory_bandwidth_unit": nested_value(record, "memory_bandwidth.unit"),
        "hardware_utilization": nested_value(record, "hardware_utilization.value"),
        "p10": percentile(latency_samples, 0.10),
        "p50": percentile(latency_samples, 0.50),
        "p90": percentile(latency_samples, 0.90),
        "sample_count": len(latency_samples),
        "source_sha256": source_closure_sha256(
            source, [microbench_root(source.parent)]),
        "sass_sha256": sha256(sass_path) if sass_path.is_file() else "",
        "blocks": params.get("resolved_blocks", ""),
    })
    return row


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _normalized_sweep_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key.replace("-", "_"): value for key, value in params.items()}


def _case_signature(atom_id: str, params: dict[str, Any]) -> str:
    return atom_id + ":" + json.dumps(
        _normalized_sweep_params(params), sort_keys=True, separators=(",", ":"))


def validate_result_params(atom_id: str, requested: dict[str, Any],
                           actual: dict[str, Any]) -> None:
    for key, expected in _normalized_sweep_params(requested).items():
        if key not in actual:
            raise ValueError(
                f"{atom_id}: result params omit requested sweep key {key!r}")
        if actual[key] != expected:
            raise ValueError(
                f"{atom_id}: result param {key!r} is {actual[key]!r}, "
                f"expected requested value {expected!r}")


def sweep_family(family_dir: Path, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("quick", "full"), default="full")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args(argv)
    root = microbench_root(family_dir)
    entries = family_entries(root, family_dir)
    paths = ensure_build_dirs(family_dir)
    with (family_dir / "scripts" / "sweep.json").open(encoding="utf-8") as handle:
        sweep = json.load(handle)
    timestamp = _run_id()
    log_path = paths["logs"] / f"sweep-{args.mode}-{timestamp}.jsonl"
    raw_path = paths["raw"] / f"sweep-{args.mode}-{timestamp}.jsonl"
    rows: list[dict[str, Any]] = []
    expected_cases: set[str] = set()
    for entry in entries:
        mode = sweep.get(args.mode, {})
        defaults = dict(mode.get("default", {}))
        benchmark = dict(mode.get("benchmarks", {}).get(entry["id"], {}))
        parameter_sets = list(sweep_parameter_sets(defaults, benchmark))
        if not parameter_sets:
            raise RuntimeError(
                f"{entry['id']}: {args.mode} sweep has no parameter cases")
        for params in parameter_sets:
            signature = _case_signature(entry["id"], params)
            if signature in expected_cases:
                raise RuntimeError(
                    f"{entry['id']}: duplicate {args.mode} sweep case {params}")
            expected_cases.add(signature)
    completed_cases: set[str] = set()
    if not args.dry_run:
        for entry in entries:
            validate_build_provenance(root, entry, paths)
    gpu_metadata = {} if args.dry_run else query_gpu_metadata(args.device)
    if (not args.dry_run and args.mode == "full" and
            "H800" not in str(gpu_metadata["gpu_name"]).upper()):
        raise RuntimeError(
            "formal full sweeps may be published only from an NVIDIA H800; "
            f"detected {gpu_metadata['gpu_name']!r}")
    with log_path.open("w", encoding="utf-8") as log_handle, \
            raw_path.open("w", encoding="utf-8") as raw_handle:
        for entry in entries:
            binary = paths["bin"] / entry["id"]
            mode = sweep.get(args.mode, {})
            defaults = dict(mode.get("default", {}))
            benchmark = dict(mode.get("benchmarks", {}).get(entry["id"], {}))
            for params in sweep_parameter_sets(defaults, benchmark):
                command = [str(binary), "--device", str(args.device)]
                for key, value in params.items():
                    command.extend([f"--{key.replace('_', '-')}", str(value).lower()
                                    if isinstance(value, bool) else str(value)])
                started = time.time()
                if args.dry_run:
                    log_handle.write(json.dumps({"argv": command, "dry_run": True}) + "\n")
                    continue
                completed = subprocess.run(command, text=True, capture_output=True,
                                           check=False)
                log_handle.write(json.dumps({
                    "argv": command,
                    "duration_s": time.time() - started,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }, sort_keys=True) + "\n")
                if completed.returncode:
                    raise RuntimeError(f"full sweep failed: {shlex.join(command)}")
                record = json.loads(completed.stdout)
                if not isinstance(record.get("params"), dict):
                    raise ValueError(
                        f"{entry['id']}: result params must be an object")
                validate_result_params(entry["id"], params, record["params"])
                record["params"].update(gpu_metadata)
                raw_handle.write(json.dumps(record, sort_keys=True) + "\n")
                row = flatten_result(
                    record, entry, root / entry["source"],
                    paths["sass"] / f"{entry['id']}.sass")
                signature = _case_signature(entry["id"], params)
                if signature in completed_cases:
                    raise RuntimeError(
                        f"{entry['id']}: duplicate completed sweep case {params}")
                completed_cases.add(signature)
                rows.append(row)
    if args.dry_run:
        print(log_path)
        return 0
    if args.mode == "full":
        if completed_cases != expected_cases or len(rows) != len(expected_cases):
            missing = sorted(expected_cases - completed_cases)
            extra = sorted(completed_cases - expected_cases)
            raise RuntimeError(
                "refusing to publish an incomplete full sweep: "
                f"rows={len(rows)}, expected={len(expected_cases)}, "
                f"missing={missing[:3]}, extra={extra[:3]}")
        fd, temporary = tempfile.mkstemp(
            prefix="result.", suffix=".csv", dir=family_dir)
        os.close(fd)
        temporary_path = Path(temporary)
        try:
            with temporary_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=csv_columns_for_family(
                        str(entries[0].get("family", ""))),
                )
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            temporary_path.chmod(0o644)
            os.replace(temporary_path, family_dir / "result.csv")
            directory_fd = os.open(family_dir, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
    print(raw_path)
    return 0


if __name__ == "__main__":
    raise SystemExit("use a family-local scripts/build.py or scripts/sweep.py")
