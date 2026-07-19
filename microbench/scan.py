#!/usr/bin/env python3
"""Build optionally, sweep benchmark parameter grids, and persist strict JSON."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

try:
    from .scripts.manifest_tool import (
        ManifestError,
        entries as manifest_entries,
        load_manifest,
    )
except ImportError:
    from scripts.manifest_tool import (
        ManifestError,
        entries as manifest_entries,
        load_manifest,
    )


ROOT = Path(__file__).resolve().parent
SCHEMA_KEYS = {
    "name",
    "params",
    "latency",
    "throughput",
    "memory_bandwidth",
    "hardware_utilization",
}
METRIC_KEYS = (
    "latency",
    "throughput",
    "memory_bandwidth",
    "hardware_utilization",
)
class ResultError(ValueError):
    pass


def load_config(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultError(f"cannot load config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ResultError("config root must be a JSON object")
    return value


def as_grid(mapping: Mapping[str, Any], where: str) -> Dict[str, List[Any]]:
    grid: Dict[str, List[Any]] = {}
    for key, value in mapping.items():
        if not isinstance(key, str) or not key or key == "peak":
            raise ResultError(f"invalid parameter name in {where}: {key!r}")
        values = value if isinstance(value, list) else [value]
        if not values:
            raise ResultError(f"empty value list for {where}.{key}")
        for item in values:
            if isinstance(item, (dict, list)) or item is None:
                raise ResultError(
                    f"parameter values must be scalar in {where}.{key}: {item!r}"
                )
        grid[key] = list(values)
    return grid


def parameter_grid(
    config: Mapping[str, Any],
    preset_name: str,
    entry: Mapping[str, Any],
    device: Optional[int],
) -> Iterable[Dict[str, Any]]:
    presets = config.get("presets")
    if not isinstance(presets, dict) or preset_name not in presets:
        raise ResultError(f"config has no preset {preset_name!r}")
    preset = presets[preset_name]
    if not isinstance(preset, dict):
        raise ResultError(f"preset {preset_name!r} must be an object")
    bench = str(entry["binary"])
    family_defaults = preset.get("family_defaults", {})
    bench_grids = preset.get("benches", {})
    if not isinstance(family_defaults, dict) or not isinstance(bench_grids, dict):
        raise ResultError("preset family_defaults and benches must be objects")
    scan_family = str(entry.get("scan_family", entry["family"]))
    family_grid = family_defaults.get(scan_family, {})
    bench_grid = bench_grids.get(bench, {})
    if not isinstance(family_grid, dict) or not isinstance(bench_grid, dict):
        raise ResultError(f"grid for {bench!r} must be an object")
    if scan_family not in family_defaults and bench not in bench_grids:
        raise ResultError(
            f"preset {preset_name!r} has no family or bench grid for {bench!r}"
        )

    cases = bench_grid.get("cases", [None])
    if not isinstance(cases, list) or not cases:
        raise ResultError(f"presets.{preset_name}.benches.{bench}.cases must be non-empty")
    bench_base = {key: value for key, value in bench_grid.items() if key != "cases"}
    merged: Dict[str, Any] = {}
    for label, section in (
        ("defaults", config.get("defaults", {})),
        (f"presets.{preset_name}.common", preset.get("common", {})),
        (f"presets.{preset_name}.family_defaults.{scan_family}", family_grid),
        (f"presets.{preset_name}.benches.{bench}", bench_base),
    ):
        if not isinstance(section, dict):
            raise ResultError(f"{label} must be an object")
        merged.update(as_grid(section, label))

    if device is not None:
        merged["device"] = [device]

    for case_index, case in enumerate(cases):
        case_grid = {}
        if case is not None:
            if not isinstance(case, dict):
                raise ResultError(
                    f"presets.{preset_name}.benches.{bench}.cases[{case_index}] "
                    "must be an object"
                )
            case_grid = as_grid(
                case,
                f"presets.{preset_name}.benches.{bench}.cases[{case_index}]",
            )
        resolved = dict(merged)
        resolved.update(case_grid)
        keys = list(resolved)
        for values in itertools.product(*(resolved[key] for key in keys)):
            yield dict(zip(keys, values))


def validate_scan_parameters(
    manifest: Mapping[str, Any],
    entry: Mapping[str, Any],
    params: Mapping[str, Any],
) -> None:
    contracts = manifest.get("scan_parameters", {})
    if not isinstance(contracts, dict):
        raise ResultError("manifest.scan_parameters must be an object")
    common = contracts.get("common", [])
    families = contracts.get("families", {})
    overrides = contracts.get("bench_overrides", {})
    if not isinstance(common, list) or not isinstance(families, dict) or not isinstance(overrides, dict):
        raise ResultError("invalid manifest scan-parameter contract")
    family = str(entry.get("scan_family", entry["family"]))
    allowed = set(common)
    allowed.update(families.get(family, []))
    allowed.update(overrides.get(entry["binary"], []))
    unknown = set(params) - allowed
    if unknown:
        raise ResultError(
            f"scan grid for {entry['binary']} contains unsupported parameters: "
            f"{sorted(unknown)}"
        )


def cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def make_command(binary: Path, params: Mapping[str, Any], peak: Optional[float]) -> List[str]:
    command = [str(binary)]
    for key, value in params.items():
        command.append(f"--{key.replace('_', '-')}={cli_value(value)}")
    if peak is not None:
        command.append(f"--peak={peak:g}")
    return command


def validate_metric(name: str, metric: Any) -> None:
    if not isinstance(metric, dict):
        raise ResultError(f"{name} must be an object")
    missing = {"value", "unit"} - set(metric)
    if missing:
        raise ResultError(f"{name} is missing keys: {sorted(missing)}")
    value = metric["value"]
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ResultError(f"{name}.value must be a finite number or null")
        if not math.isfinite(float(value)):
            raise ResultError(f"{name}.value must be finite")
    if not isinstance(metric["unit"], str) or not metric["unit"]:
        raise ResultError(f"{name}.unit must be a non-empty string")


def validate_result(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ResultError("stdout JSON must be an object")
    actual = set(value)
    if actual != SCHEMA_KEYS:
        missing = sorted(SCHEMA_KEYS - actual)
        extra = sorted(actual - SCHEMA_KEYS)
        raise ResultError(f"top-level schema mismatch; missing={missing}, extra={extra}")
    if not isinstance(value["name"], str) or not value["name"]:
        raise ResultError("name must be a non-empty string")
    if not isinstance(value["params"], dict):
        raise ResultError("params must be an object")
    for metric_name in METRIC_KEYS:
        validate_metric(metric_name, value[metric_name])
    return value


def values_equal(expected: Any, actual: Any) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(expected) is type(actual) and expected == actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(
            float(expected), float(actual), rel_tol=1.0e-12, abs_tol=0.0
        )
    return expected == actual


def validate_param_echo(
    result: Mapping[str, Any],
    requested: Mapping[str, Any],
    peak: Optional[float],
) -> None:
    echoed = result["params"]
    expected = dict(requested)
    if peak is not None:
        expected["peak"] = peak
    for key, expected_value in expected.items():
        if key not in echoed:
            raise ResultError(f"params is missing swept parameter {key!r}")
        if not values_equal(expected_value, echoed[key]):
            raise ResultError(
                f"params.{key} does not echo the requested value: "
                f"requested={expected_value!r}, returned={echoed[key]!r}"
            )


def reject_json_constant(token: str) -> Any:
    raise ResultError(f"non-finite JSON token is forbidden: {token}")


def parse_stdout(stdout: str) -> Dict[str, Any]:
    if not stdout.strip():
        raise ResultError("benchmark produced empty stdout")
    try:
        value = json.loads(stdout, parse_constant=reject_json_constant)
    except json.JSONDecodeError as exc:
        raise ResultError(
            f"stdout must contain exactly one JSON object: {exc.msg} at char {exc.pos}"
        ) from exc
    return validate_result(value)


def write_jsonl(handle: Any, value: Mapping[str, Any]) -> None:
    json.dump(value, handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    handle.write("\n")
    handle.flush()


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        write_jsonl(handle, value)


def tail(text: Any, limit: int = 16384) -> str:
    if text is None:
        value = ""
    elif isinstance(text, bytes):
        value = text.decode("utf-8", errors="replace")
    else:
        value = str(text)
    return value if len(value) <= limit else value[-limit:]


def failure_record(
    bench: str,
    params: Optional[Mapping[str, Any]],
    command: Sequence[str],
    kind: str,
    message: str,
    duration_seconds: float,
    returncode: Optional[int] = None,
    stdout: Optional[str] = None,
    stderr: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": bench,
        "args": dict(params) if params is not None else None,
        "command": shlex.join(command),
        "error": {"kind": kind, "message": message},
        "duration_seconds": duration_seconds,
        "returncode": returncode,
        "stdout_tail": tail(stdout),
        "stderr_tail": tail(stderr),
    }


def resolve_peak(
    entry: Mapping[str, Any],
    config: Mapping[str, Any],
    peak_bf16_tflops: Optional[float],
    peak_memory_gbps: Optional[float],
) -> Optional[float]:
    bench = str(entry["binary"])
    kind = entry.get("peak_kind")
    if kind is None:
        return None
    peaks = config.get("peaks", {})
    if not isinstance(peaks, dict):
        raise ResultError("config.peaks must be an object")
    value: Any
    if kind == "bf16":
        value = peak_bf16_tflops
        if value is None:
            value = peaks.get("bf16_tflops")
    elif kind == "memory":
        value = peak_memory_gbps
        if value is None:
            value = peaks.get("memory_gbps")
    else:
        raise ResultError(f"unknown peak_kind for {bench}: {kind!r}")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResultError(f"peak for {bench} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ResultError(f"peak for {bench} must be finite and positive")
    return value


def default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "results" / stamp


def file_sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_gpu_environment(device: int) -> Dict[str, Any]:
    command = [
        "nvidia-smi",
        f"--id={device}",
        "--query-gpu=uuid,name,driver_version,pstate,clocks.current.sm,"
        "clocks.current.memory,power.draw,power.limit",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc), "command": shlex.join(command)}
    if completed.returncode != 0:
        return {
            "available": False,
            "error": tail(completed.stderr),
            "command": shlex.join(command),
        }
    fields = [item.strip() for item in completed.stdout.strip().split(",")]
    names = (
        "uuid", "name", "driver_version", "pstate", "sm_clock_mhz",
        "memory_clock_mhz", "power_draw_w", "power_limit_w",
    )
    return {
        "available": len(fields) == len(names),
        "values": dict(zip(names, fields)),
        "command": shlex.join(command),
    }


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep FlashMLA dense-decode SM90a hardware micro-benchmarks."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--preset", choices=("quick", "full"), default="quick")
    parser.add_argument("--timeout", type=positive_float, default=None)
    parser.add_argument("--peak-bf16-tflops", type=positive_float, default=None)
    parser.add_argument("--peak-memory-gbps", type=positive_float, default=None)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "h800.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifest.json")
    parser.add_argument("--bin-dir", type=Path, default=ROOT / "build" / "bin")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument(
        "--kind", choices=("atom", "calibration", "all"), default="atom",
        help="select atomic benchmarks, interaction calibrations, or both",
    )
    parser.add_argument(
        "--bench", action="append", default=None,
        help="canonical binary or manifest group alias; repeatable",
    )
    parser.add_argument(
        "--category", choices=("memory", "compute", "calibration"), default=None
    )
    parser.add_argument("--family", action="append", default=None)
    parser.add_argument("--build", action="store_true", help="run make before the sweep")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="validate selection and print commands as JSONL without building or running",
    )
    parser.add_argument("--jobs", type=positive_int, default=max(1, os.cpu_count() or 1))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = (args.output_dir or default_output_dir()).resolve()
    results_jsonl = output_dir / "results.jsonl"
    failures_jsonl = output_dir / "failures.jsonl"
    run_log = output_dir / "run.log"
    occupied = [path for path in (results_jsonl, failures_jsonl, run_log) if path.exists()]
    if occupied and not args.dry_run:
        parser.error("refusing to overwrite result files: " + ", ".join(map(str, occupied)))

    try:
        config = load_config(args.config.resolve())
        manifest = load_manifest(args.manifest.resolve())
        selected_entries = manifest_entries(manifest, args.kind)
        entries = selected_entries
        by_binary = {entry["binary"]: entry for entry in entries}
        groups = manifest.get("groups", {})
        preset = config.get("presets", {}).get(args.preset, {})
        configured_timeout = preset.get("timeout_seconds") if isinstance(preset, dict) else None
        timeout = args.timeout if args.timeout is not None else configured_timeout
        if (isinstance(timeout, bool) or
                not isinstance(timeout, (int, float)) or
                not math.isfinite(float(timeout)) or timeout <= 0):
            raise ResultError("timeout must be finite and positive")
        timeout = float(timeout)
        requested = args.bench or [entry["binary"] for entry in entries]
        expanded: list[str] = []
        for token in requested:
            if token in groups:
                members = [member for member in groups[token] if member in by_binary]
                if not members:
                    raise ResultError(
                        f"group alias {token!r} has no entries for kind={args.kind}"
                    )
                expanded.extend(members)
            elif token in by_binary:
                expanded.append(token)
            else:
                raise ResultError(f"unknown benchmark or group alias: {token!r}")
        selected = []
        for bench in dict.fromkeys(expanded):
            entry = by_binary[bench]
            if args.category is not None and entry["category"] != args.category:
                continue
            if args.family and entry["family"] not in set(args.family):
                continue
            selected.append(entry)
        if not selected:
            raise ResultError("benchmark filters selected no manifest entries")
        peaks = {
            entry["binary"]: resolve_peak(
                entry,
                config,
                args.peak_bf16_tflops,
                args.peak_memory_gbps,
            )
            for entry in selected
        }
    except (ResultError, ManifestError) as exc:
        parser.error(str(exc))

    if args.dry_run:
        try:
            for entry in selected:
                binary = (args.bin_dir / entry["binary"]).resolve()
                for params in parameter_grid(config, args.preset, entry, args.device):
                    validate_scan_parameters(manifest, entry, params)
                    print(json.dumps({
                        "benchmark": entry["binary"],
                        "kind": "calibration" if entry["category"] == "calibration" else "atom",
                        "expected_result_name": entry["result_name"],
                        "params": params,
                        "command": make_command(binary, params, peaks[entry["binary"]]),
                    }, allow_nan=False, separators=(",", ":")))
        except ResultError as exc:
            parser.error(str(exc))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failure_count = 0
    interrupted = False
    run_started_utc = datetime.now(timezone.utc).isoformat()
    run_started = time.monotonic()
    run_device = args.device
    if run_device is None:
        defaults = config.get("defaults", {})
        configured = defaults.get("device", [0]) if isinstance(defaults, dict) else [0]
        run_device = int(configured[0] if isinstance(configured, list) else configured)
    gpu_environment = collect_gpu_environment(run_device)
    with results_jsonl.open("x", encoding="utf-8", buffering=1) as result_handle, \
            run_log.open("x", encoding="utf-8", buffering=1) as log_handle:
        write_jsonl(
            log_handle,
            {
                "event": "run_start",
                "started_utc": run_started_utc,
                "args": {
                    "preset": args.preset,
                    "kind": args.kind,
                    "bench": args.bench,
                    "category": args.category,
                    "family": args.family,
                    "device": run_device,
                    "timeout_seconds": timeout,
                    "build": args.build,
                    "jobs": args.jobs,
                    "config": str(args.config.resolve()),
                    "config_sha256": file_sha256(args.config.resolve()),
                    "manifest": str(args.manifest.resolve()),
                    "manifest_sha256": file_sha256(args.manifest.resolve()),
                    "bin_dir": str(args.bin_dir.resolve()),
                },
                "gpu_environment": gpu_environment,
            },
        )

        def finish_run(status: str) -> None:
            write_jsonl(
                log_handle,
                {
                    "event": "run_end",
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                    "duration_seconds": time.monotonic() - run_started,
                    "status": status,
                    "results": success_count,
                    "failures": failure_count,
                },
            )

        if args.build:
            target = {"atom": "atoms", "calibration": "calibration", "all": "everything"}[args.kind]
            command = ["make", "-C", str(ROOT), f"-j{args.jobs}", target]
            started = time.monotonic()
            try:
                completed = subprocess.run(
                    command,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=max(timeout, 300.0),
                    check=False,
                )
                if completed.returncode != 0:
                    failure_count += 1
                    append_jsonl(
                        failures_jsonl,
                        failure_record(
                            "build",
                            None,
                            command,
                            "build_exit",
                            "make failed; no benchmark was executed",
                            time.monotonic() - started,
                            completed.returncode,
                            completed.stdout,
                            completed.stderr,
                        ),
                    )
                    finish_run("failed")
                    return 1
                write_jsonl(
                    log_handle,
                    {
                        "event": "build",
                        "args": command,
                        "duration_seconds": time.monotonic() - started,
                        "status": "ok",
                    },
                )
            except subprocess.TimeoutExpired as exc:
                failure_count += 1
                append_jsonl(
                    failures_jsonl,
                    failure_record(
                        "build",
                        None,
                        command,
                        "build_timeout",
                        f"make exceeded {max(timeout, 300.0):g} seconds",
                        time.monotonic() - started,
                        stdout=exc.stdout,
                        stderr=exc.stderr,
                    ),
                )
                finish_run("failed")
                return 1
            except OSError as exc:
                failure_count += 1
                append_jsonl(
                    failures_jsonl,
                    failure_record(
                        "build",
                        None,
                        command,
                        "build_launch_error",
                        str(exc),
                        time.monotonic() - started,
                    ),
                )
                finish_run("failed")
                return 1

        try:
            for entry in selected:
                bench = entry["binary"]
                binary = (args.bin_dir / bench).resolve()
                if not binary.is_file() or not os.access(binary, os.X_OK):
                    failure_count += 1
                    append_jsonl(
                        failures_jsonl,
                        failure_record(
                            bench,
                            None,
                            [str(binary)],
                            "missing_binary",
                            "binary is missing or not executable",
                            0.0,
                        ),
                    )
                    continue

                try:
                    cases = parameter_grid(config, args.preset, entry, args.device)
                    for params in cases:
                        validate_scan_parameters(manifest, entry, params)
                        command = make_command(binary, params, peaks[bench])
                        started = time.monotonic()
                        try:
                            completed = subprocess.run(
                                command,
                                text=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                timeout=timeout,
                                check=False,
                            )
                        except subprocess.TimeoutExpired as exc:
                            failure_count += 1
                            append_jsonl(
                                failures_jsonl,
                                failure_record(
                                    bench,
                                    params,
                                    command,
                                    "timeout",
                                    f"benchmark exceeded {timeout:g} seconds",
                                    time.monotonic() - started,
                                    stdout=exc.stdout,
                                    stderr=exc.stderr,
                                ),
                            )
                            continue
                        except OSError as exc:
                            failure_count += 1
                            append_jsonl(
                                failures_jsonl,
                                failure_record(
                                    bench,
                                    params,
                                    command,
                                    "launch_error",
                                    str(exc),
                                    time.monotonic() - started,
                                ),
                            )
                            continue

                        duration = time.monotonic() - started
                        if completed.returncode != 0:
                            failure_count += 1
                            append_jsonl(
                                failures_jsonl,
                                failure_record(
                                    bench,
                                    params,
                                    command,
                                    "nonzero_exit",
                                    "benchmark returned a non-zero exit status",
                                    duration,
                                    completed.returncode,
                                    completed.stdout,
                                    completed.stderr,
                                ),
                            )
                            continue
                        try:
                            result = parse_stdout(completed.stdout)
                            validate_param_echo(result, params, peaks[bench])
                            if result["name"] != entry["result_name"]:
                                raise ResultError(
                                    "result name mismatch: expected "
                                    f"{entry['result_name']!r}, got {result['name']!r}"
                                )
                        except ResultError as exc:
                            failure_count += 1
                            append_jsonl(
                                failures_jsonl,
                                failure_record(
                                    bench,
                                    params,
                                    command,
                                    "invalid_json",
                                    str(exc),
                                    duration,
                                    completed.returncode,
                                    completed.stdout,
                                    completed.stderr,
                                ),
                            )
                            continue

                        write_jsonl(result_handle, result)
                        source = ROOT / entry["source"]
                        static_hashes = ROOT / "build" / "static" / f"{bench}.sha256"
                        write_jsonl(
                            log_handle,
                            {
                                "event": "result",
                                "result_index": success_count,
                                "name": result["name"],
                                "binary": bench,
                                "args": params,
                                "command": command,
                                "duration_seconds": duration,
                                "source": entry["source"],
                                "source_sha256": file_sha256(source),
                                "static_hash_manifest": str(static_hashes.resolve()),
                                "static_hash_manifest_sha256": file_sha256(static_hashes),
                            },
                        )
                        success_count += 1
                except ResultError as exc:
                    failure_count += 1
                    append_jsonl(
                        failures_jsonl,
                        failure_record(
                            bench,
                            None,
                            [str(binary)],
                            "invalid_grid",
                            str(exc),
                            0.0,
                        ),
                    )
        except KeyboardInterrupt:
            interrupted = True
        finish_run("interrupted" if interrupted else (
            "failed" if failure_count else "ok"
        ))

    print(
        f"wrote {success_count} results and {failure_count} failures to {output_dir}",
        file=sys.stderr,
        flush=True,
    )
    if interrupted:
        return 130
    return 1 if failure_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
