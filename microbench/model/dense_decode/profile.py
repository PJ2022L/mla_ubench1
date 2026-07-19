"""Build and query immutable hardware profiles from microbench JSON records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 1


class ProfileError(ValueError):
    pass


def _finite_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ProfileError(f"{where} must be finite")
    return result


def _read_records(path: Path) -> list[dict[str, Any]]:
    files: list[Path]
    if path.is_dir():
        files = []
        directories = sorted({candidate.parent for candidate in path.rglob("results.json*")})
        for directory in directories:
            jsonl = directory / "results.jsonl"
            json_array = directory / "results.json"
            if jsonl.is_file():
                files.append(jsonl)
            elif json_array.is_file():
                files.append(json_array)
        files.extend(sorted(path.rglob("result.jsonl")))
    else:
        files = [path]
    records: list[dict[str, Any]] = []
    for candidate in files:
        text = candidate.read_text(encoding="utf-8")
        try:
            if candidate.suffix == ".jsonl":
                loaded = [json.loads(line) for line in text.splitlines() if line.strip()]
            else:
                value = json.loads(text)
                loaded = value if isinstance(value, list) else [value]
        except json.JSONDecodeError as exc:
            raise ProfileError(f"cannot parse {candidate}: {exc}") from exc
        for index, record in enumerate(loaded):
            if not isinstance(record, dict):
                raise ProfileError(f"{candidate} record {index} is not an object")
            record = dict(record)
            record["_source_file"] = str(candidate.resolve())
            record["_source_index"] = index
            log_path = candidate.parent / "run.log"
            legacy_path = candidate.parent / "provenance.jsonl"
            provenance_path = log_path if log_path.is_file() else legacy_path
            if provenance_path.is_file():
                provenance_records = [
                    json.loads(line)
                    for line in provenance_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                matching = [
                    item for item in provenance_records
                    if isinstance(item, dict)
                    and item.get("result_index") == index
                    and item.get("event", "result") == "result"
                ]
                if matching:
                    record["_run_provenance"] = matching[-1]
            records.append(record)
    if not records:
        raise ProfileError(f"no benchmark result records found below {path}")
    return records


def _validate_record(record: Mapping[str, Any]) -> None:
    required = {
        "name",
        "params",
        "latency",
        "throughput",
        "memory_bandwidth",
        "hardware_utilization",
    }
    missing = required - set(record)
    if missing:
        raise ProfileError(f"result record is missing keys: {sorted(missing)}")
    if not isinstance(record["name"], str) or not record["name"]:
        raise ProfileError("record name must be a non-empty string")
    if not isinstance(record["params"], dict):
        raise ProfileError("record params must be an object")
    for key in ("latency", "throughput", "memory_bandwidth", "hardware_utilization"):
        metric = record[key]
        if not isinstance(metric, dict) or "value" not in metric or "unit" not in metric:
            raise ProfileError(f"record metric {key} must contain value and unit")


def _static_hashes(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    if not path.exists():
        raise ProfileError(f"static artifact path does not exist: {path}")
    result: dict[str, str] = {}
    candidates = [path] if path.is_file() else sorted(
        list(path.rglob("*.sha256")) +
        list(path.rglob("dense_decode_resources.json"))
    )
    for candidate in candidates:
        result[str(candidate.resolve())] = hashlib.sha256(candidate.read_bytes()).hexdigest()
    return result


def _load_resource_contract(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    roots = [path] if path.is_dir() else [path.parent]
    for root in roots:
        candidate = root / "dense_decode_resources.json"
        if candidate.is_file():
            value = json.loads(candidate.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ProfileError(f"{candidate} must contain an object")
            return value
    return {}


def _round_up(value: int, granularity: int) -> int:
    return ((value + granularity - 1) // granularity) * granularity


def _cta_residency(target: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    threads = int(target.get(f"{prefix}_threads", 0))
    registers = int(target.get(f"{prefix}_registers_per_thread", 0))
    shared = int(target.get(f"{prefix}_shared_memory_bytes", 0))
    launch_bound = int(target.get(f"{prefix}_launch_bound_ctas", 0))
    if threads <= 0 or registers <= 0 or shared < 0:
        fallback = int(target.get(f"{prefix}_cta_residency", 1 if prefix == "main" else 2))
        return {
            "residency": max(fallback, 1),
            "source": "planning_fallback",
            "threads": threads,
            "registers_per_thread": registers,
            "shared_memory_bytes": shared,
            "limits": {},
        }
    registers_per_sm = int(target.get("registers_per_sm", 65536))
    shared_per_sm = int(target.get("shared_memory_per_sm", 233472))
    max_warps = int(target.get("max_warps_per_sm", 64))
    max_ctas = int(target.get("max_ctas_per_sm", 32))
    warps = math.ceil(threads / 32)
    registers_per_warp = _round_up(registers * 32, 256)
    registers_per_cta = registers_per_warp * warps
    shared_per_cta = _round_up(shared, 256) if shared else 0
    limits = {
        "registers": registers_per_sm // max(registers_per_cta, 1),
        "shared_memory": shared_per_sm // shared_per_cta if shared_per_cta else max_ctas,
        "warps": max_warps // max(warps, 1),
        "cta_slots": max_ctas,
    }
    if launch_bound > 0:
        limits["launch_bound"] = launch_bound
    return {
        "residency": max(1, min(limits.values())),
        "source": "resource_contract",
        "threads": threads,
        "warps": warps,
        "registers_per_thread": registers,
        "registers_per_cta_allocated": registers_per_cta,
        "shared_memory_bytes": shared,
        "shared_memory_bytes_allocated": shared_per_cta,
        "limits": limits,
    }


def build_profile(
    results_path: Path,
    *,
    manifest_path: Path,
    static_artifacts: Path | None,
    target: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("benchmarks"), list):
        raise ProfileError("manifest must contain a benchmarks array")
    records = _read_records(results_path)
    for record in records:
        _validate_record(record)

    manifest_entries = manifest["benchmarks"] + manifest.get("calibrations", [])
    manifest_names = {
        entry.get("result_name")
        for entry in manifest_entries
        if isinstance(entry, dict) and isinstance(entry.get("result_name"), str)
    }
    unknown = sorted({record["name"] for record in records} - manifest_names)
    if unknown:
        raise ProfileError(f"results contain names absent from manifest: {unknown}")

    normalized: list[dict[str, Any]] = []
    for record in records:
        normalized.append(
            {
                "name": record["name"],
                "params": record["params"],
                "latency": record["latency"],
                "throughput": record["throughput"],
                "memory_bandwidth": record["memory_bandwidth"],
                "hardware_utilization": record["hardware_utilization"],
                "raw_samples": {
                    metric_name: record[metric_name].get("samples", [])
                    for metric_name in (
                        "latency", "throughput", "memory_bandwidth"
                    )
                    if isinstance(record[metric_name], dict)
                },
                "provenance": {
                    "result_name": record["name"],
                    "source_file": record["_source_file"],
                    "record_index": record["_source_index"],
                    "run": record.get("_run_provenance", {}),
                },
            }
        )

    resolved_target = dict(target)
    resource_contract = _load_resource_contract(static_artifacts)
    if resource_contract:
        resolved_target.update(resource_contract.get("hardware", resource_contract))
    occupancy = {
        "main": _cta_residency(resolved_target, "main"),
        "combine": _cta_residency(resolved_target, "combine"),
    }
    resolved_target["main_cta_residency"] = occupancy["main"]["residency"]
    resolved_target["combine_cta_residency"] = occupancy["combine"]["residency"]
    present_names = {record["name"] for record in normalized}

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target": resolved_target,
        "occupancy": occupancy,
        "coverage": {
            "present_result_names": sorted(present_names),
            "missing_result_names": sorted(manifest_names - present_names),
        },
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "static_artifact_hashes": _static_hashes(static_artifacts),
        "records": normalized,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["profile_id"] = hashlib.sha256(encoded).hexdigest()
    return payload


def load_profile(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ProfileError("unsupported or invalid profile schema")
    if not isinstance(value.get("target"), dict) or not isinstance(value.get("records"), list):
        raise ProfileError("profile requires target and records")
    return value


@dataclass(frozen=True)
class MetricPoint:
    name: str
    params: Mapping[str, Any]
    metric: Mapping[str, Any]
    provenance: Mapping[str, Any]
    raw_samples: Mapping[str, Any]


def _param_distance(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> float:
    distance = 0.0
    for key, wanted in expected.items():
        if key not in actual:
            distance += 4.0
            continue
        found = actual[key]
        if isinstance(wanted, bool) or isinstance(found, bool):
            distance += 0.0 if type(wanted) is type(found) and wanted == found else 20.0
        elif isinstance(wanted, (int, float)) and isinstance(found, (int, float)):
            scale = max(abs(float(wanted)), abs(float(found)), 1.0)
            distance += abs(float(wanted) - float(found)) / scale
        else:
            distance += 0.0 if wanted == found else 20.0
    return distance


class ProfileLookup:
    def __init__(self, profile: Mapping[str, Any]) -> None:
        self.profile = profile
        self.records = profile["records"]
        target = profile["target"]
        self.sm_count = int(target.get("sm_count", 0))
        self.sm_clock_mhz = _finite_number(target.get("sm_clock_mhz", 0), "sm_clock_mhz")
        self.l2_bytes = int(target.get("l2_bytes", 0))
        self.hbm_gbps = _finite_number(target.get("hbm_gbps", 0), "hbm_gbps")
        self.l2_gbps = _finite_number(
            target.get("l2_gbps", self.hbm_gbps * 2.0), "l2_gbps"
        )
        self.main_cta_residency = int(target.get("main_cta_residency", 1))
        self.combine_cta_residency = int(target.get("combine_cta_residency", 2))
        if self.sm_count <= 0 or self.sm_clock_mhz <= 0 or self.hbm_gbps <= 0:
            raise ProfileError("target requires positive sm_count, sm_clock_mhz, and hbm_gbps")
        if self.main_cta_residency <= 0 or self.combine_cta_residency <= 0:
            raise ProfileError("CTA residency values must be positive")

    def points(
        self,
        names: Iterable[str],
        metric_name: str,
        params: Mapping[str, Any],
    ) -> list[MetricPoint]:
        accepted = set(names)
        candidates: list[tuple[float, MetricPoint]] = []
        for record in self.records:
            if record.get("name") not in accepted:
                continue
            metric = record.get(metric_name)
            if not isinstance(metric, dict) or metric.get("value") is None:
                continue
            _finite_number(metric["value"], f"{record['name']}.{metric_name}.value")
            point = MetricPoint(
                name=record["name"],
                params=record.get("params", {}),
                metric=metric,
                provenance=record.get("provenance", {}),
                raw_samples=record.get("raw_samples", {}),
            )
            candidates.append((_param_distance(params, point.params), point))
        candidates.sort(key=lambda item: item[0])
        return [point for _, point in candidates[:4]]

    def interpolate(
        self,
        names: Iterable[str],
        metric_name: str,
        params: Mapping[str, Any],
    ) -> tuple[float, str, list[Mapping[str, Any]]]:
        points = self.points(names, metric_name, params)
        if not points:
            raise ProfileError(
                f"profile has no {metric_name} points for any of {sorted(set(names))}"
            )
        weighted = 0.0
        total_weight = 0.0
        provenances: list[Mapping[str, Any]] = []
        for point in points:
            distance = _param_distance(params, point.params)
            weight = 1.0 / max(distance, 1.0e-6)
            weighted += weight * _finite_number(point.metric["value"], "metric value")
            total_weight += weight
            provenances.append(point.provenance)
        return weighted / total_weight, str(points[0].metric["unit"]), provenances
