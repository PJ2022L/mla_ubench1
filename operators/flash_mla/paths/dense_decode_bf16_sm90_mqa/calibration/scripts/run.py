#!/usr/bin/env python3
"""Run calibration sweeps on remote H800 and atomically publish full results."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[4]
sys.path.insert(0, str(REPO / "microbench" / "scripts"))

from family_runner import source_closure_sha256

FIELDS = [
    "probe", "params_json", "gpu_uuid", "sm_clock_mhz", "memory_clock_mhz",
    "measured_latency_value", "measured_latency_unit", "throughput_value",
    "throughput_unit", "memory_bandwidth_value", "memory_bandwidth_unit",
    "hardware_utilization", "p10", "p50", "p90", "sample_count",
    "source_sha256", "sass_sha256",
]
TOP_KEYS = {"name", "params", "latency", "throughput",
            "memory_bandwidth", "hardware_utilization"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_build_provenance(probe: dict) -> None:
    name = probe["id"]
    metadata_path = ROOT / "build/resources" / f"{name}.build.json"
    if not metadata_path.is_file():
        raise RuntimeError(
            f"{name}: missing build provenance; rebuild calibration first")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "source_closure_sha256": source_closure_sha256(
            ROOT / probe["source"], [ROOT, REPO / "microbench"]),
        "binary_sha256": sha256(ROOT / "build/bin" / name),
        "ptx_sha256": sha256(ROOT / "build/ptx" / f"{name}.ptx"),
        "cubin_sha256": sha256(ROOT / "build/cubin" / f"{name}.cubin"),
        "sass_sha256": sha256(ROOT / "build/sass" / f"{name}.sass"),
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise RuntimeError(
                f"{name}: stale calibration artifact for {field}; rebuild")


def percentile(samples: list[float], fraction: float) -> float | str:
    if not samples:
        return ""
    values = sorted(samples)
    index = fraction * (len(values) - 1)
    lo, hi = math.floor(index), math.ceil(index)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def normalize_latency_to_cycles(
    latency: dict, sm_clock_mhz: str
) -> tuple[float, list[float], str, dict[str, object]]:
    """Return the residual-comparison latency in cycles.

    Most probes measure clock64 cycles directly.  PDL uses the cross-SM
    ``%globaltimer`` and therefore reports a time unit; convert that time using
    the nvidia-smi SM clock sampled immediately before the binary invocation.
    The native JSON remains untouched in build/raw.
    """
    unit = str(latency["unit"])
    lowered = unit.lower().replace(" ", "")
    value = float(latency["value"])
    samples = [float(item) for item in latency.get("samples", [])]
    if "cycle" in lowered:
        return value, samples, unit, {}

    clock = float(sm_clock_mhz)
    if lowered.startswith("ns"):
        factor = clock / 1000.0
        suffix = unit[2:]
    elif lowered.startswith("us"):
        factor = clock
        suffix = unit[2:]
    elif lowered.startswith("ms"):
        factor = clock * 1000.0
        suffix = unit[2:]
    else:
        raise ValueError(f"unsupported residual latency unit: {unit}")
    conversion = {
        "residual_latency_source_unit": unit,
        "residual_latency_clock_mhz": clock,
        "residual_latency_conversion": "globaltimer_time_x_pre_run_nvidia_smi_sm_clock",
    }
    return value * factor, [item * factor for item in samples], f"cycle{suffix}", conversion


def expand(value: dict) -> list[dict]:
    keys = list(value)
    values = [v if isinstance(v, list) else [v] for v in value.values()]
    return [dict(zip(keys, row)) for row in itertools.product(*values)]


def command(binary: Path, params: dict) -> list[str]:
    return [str(binary), *(f"--{key}={str(value).lower() if isinstance(value, bool) else value}"
                           for key, value in params.items())]


def gpu_provenance(device: int) -> tuple[str, str, str, str]:
    completed = subprocess.run(
        ["nvidia-smi", f"--id={device}",
         "--query-gpu=name,uuid,clocks.sm,clocks.mem",
         "--format=csv,noheader,nounits"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"nvidia-smi provenance query failed: {completed.stderr}")
    values = [value.strip() for value in completed.stdout.strip().split(",")]
    if len(values) != 4:
        raise RuntimeError("unexpected nvidia-smi provenance response")
    return values[0], values[1], values[2], values[3]


def validate_result(probe: str, value: object) -> dict:
    if not isinstance(value, dict) or set(value) != TOP_KEYS:
        raise ValueError(f"{probe}: invalid six-key JSON result")
    if value["name"] != probe:
        raise ValueError(f"{probe}: result name is {value['name']!r}")
    for metric in ("latency", "throughput", "memory_bandwidth", "hardware_utilization"):
        item = value[metric]
        if not isinstance(item, dict) or "value" not in item or "unit" not in item:
            raise ValueError(f"{probe}: invalid {metric}")
    return value


def reject_raw_param_arrays(value: object, path: str = "params") -> None:
    if isinstance(value, list):
        raise ValueError(
            f"{path}: raw arrays belong in calibration build/raw, not result.csv")
    if isinstance(value, dict):
        for key, child in value.items():
            reject_raw_param_arrays(child, f"{path}.{key}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=("quick", "full"), required=True)
    parser.add_argument("--probe", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((ROOT / "scripts/sweep.json").read_text(encoding="utf-8"))[args.preset]
    selected = set(args.probe)
    probes = [p for p in manifest["probes"] if not selected or p["id"] in selected]
    if selected - {p["id"] for p in probes}:
        raise SystemExit("unknown probe selection")
    rows: list[dict] = []
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for probe in probes:
        cases = config.get("cases", {}).get(probe["id"], [{}])
        for case_index, case in enumerate(cases):
            params = dict(config["defaults"])
            params.update(case)
            cmd = command(ROOT / "build/bin" / probe["binary"], params)
            if args.dry_run:
                print(shlex.join(cmd))
                continue
            validate_build_provenance(probe)
            gpu_name, gpu_uuid, sm_clock_mhz, memory_clock_mhz = gpu_provenance(
                int(params["device"]))
            if args.preset == "full" and "H800" not in gpu_name.upper():
                raise RuntimeError(
                    "formal calibration full sweeps require NVIDIA H800; "
                    f"detected {gpu_name!r}")
            started = dt.datetime.now(dt.timezone.utc)
            completed = subprocess.run(cmd, cwd=ROOT, text=True,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, check=False)
            ended = dt.datetime.now(dt.timezone.utc)
            stem = f"{stamp}-{probe['id']}-{case_index:03d}"
            log = ROOT / "build/logs" / f"run-{stem}.log"
            raw = ROOT / "build/raw" / f"{stem}.json"
            log.parent.mkdir(parents=True, exist_ok=True)
            raw.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(
                f"command={shlex.join(cmd)}\nstarted_utc={started.isoformat()}\n"
                f"ended_utc={ended.isoformat()}\n"
                f"duration_seconds={(ended-started).total_seconds():.6f}\n"
                f"gpu_name={gpu_name}\ngpu_uuid={gpu_uuid}\n"
                f"sm_clock_mhz={sm_clock_mhz}\n"
                f"memory_clock_mhz={memory_clock_mhz}\n"
                f"returncode={completed.returncode}\n\nSTDOUT\n{completed.stdout}\nSTDERR\n{completed.stderr}",
                encoding="utf-8",
            )
            if completed.returncode:
                raise RuntimeError(f"{probe['id']} failed; see {log}")
            result = validate_result(probe["id"], json.loads(completed.stdout))
            raw.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
            latency = result["latency"]
            latency_value, samples, latency_unit, conversion = (
                normalize_latency_to_cycles(latency, sm_clock_mhz)
            )
            hw = result["hardware_utilization"].get("value")
            params_out = dict(result["params"])
            params_out.update(conversion)
            reject_raw_param_arrays(params_out)
            rows.append({
                "probe": probe["id"],
                "params_json": json.dumps(params_out, sort_keys=True, separators=(",", ":")),
                "gpu_uuid": gpu_uuid,
                "sm_clock_mhz": sm_clock_mhz,
                "memory_clock_mhz": memory_clock_mhz,
                "measured_latency_value": latency_value,
                "measured_latency_unit": latency_unit,
                "throughput_value": result["throughput"]["value"],
                "throughput_unit": result["throughput"]["unit"],
                "memory_bandwidth_value": result["memory_bandwidth"]["value"],
                "memory_bandwidth_unit": result["memory_bandwidth"]["unit"],
                "hardware_utilization": hw if hw is not None else "",
                "p10": percentile(samples, 0.10), "p50": percentile(samples, 0.50),
                "p90": percentile(samples, 0.90), "sample_count": len(samples),
                "source_sha256": source_closure_sha256(
                    ROOT / probe["source"], [ROOT, REPO / "microbench"]),
                "sass_sha256": sha256(ROOT / "build/sass" / f"{probe['id']}.sass"),
            })
    if args.dry_run:
        return 0
    if args.preset != "full" or selected:
        print("formal result.csv unchanged: only an unfiltered full sweep may publish")
        return 0
    destination = ROOT / "result.csv"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="",
                                     dir=ROOT, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, destination)
    print(f"published {len(rows)} rows to {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
