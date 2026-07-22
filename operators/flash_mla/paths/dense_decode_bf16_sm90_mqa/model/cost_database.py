"""Generic microbenchmark CSV loader used by official prediction.

This module deliberately has no calibration support. It accepts only records
declared by ``microbench/manifest.json`` as operations or resource curves.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
from itertools import product
from typing import Any, Iterable, Mapping


class CostDatabaseError(ValueError):
    pass


class CoverageError(CostDatabaseError):
    pass


_METRIC_COLUMNS = {
    "latency_value", "latency_unit", "initiation_interval_cycles",
    "throughput_value", "throughput_unit", "memory_bandwidth_value",
    "memory_bandwidth_unit", "hardware_utilization", "p10", "p50", "p90",
    "sample_count", "source_sha256", "sass_sha256", "gpu_uuid",
    "sm_clock_mhz", "memory_clock_mhz", "gpu_name", "name",
}


def _parse(value: str | None) -> Any:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if stripped.lower() in {"true", "false"}:
        return stripped.lower() == "true"
    try:
        if any(mark in stripped.lower() for mark in (".", "e")):
            return float(stripped)
        return int(stripped)
    except ValueError:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return stripped


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CostDatabaseError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or result < 0:
        raise CostDatabaseError(f"{name} must be finite and non-negative")
    return result


def _finite_or(value: Any, default: float, name: str) -> float:
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    return _finite(value, name)


@dataclass(frozen=True)
class CostRecord:
    atom_id: str
    kind: str
    family: str
    params: Mapping[str, Any]
    latency_cycles: float
    initiation_interval_cycles: float
    throughput_value: float
    throughput_unit: str
    memory_bandwidth_value: float
    memory_bandwidth_unit: str
    p10: float
    p50: float
    p90: float
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class OperationCost:
    latency_cycles: float
    initiation_interval_cycles: float
    p10_cycles: float
    p50_cycles: float
    p90_cycles: float
    throughput_value: float
    throughput_unit: str
    memory_bandwidth_value: float
    memory_bandwidth_unit: str
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class ResourceService:
    throughput_value: float
    throughput_unit: str
    memory_bandwidth_value: float
    memory_bandwidth_unit: str
    provenance: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class ResourceInteraction:
    slowdown: float
    provenance: tuple[Mapping[str, Any], ...]


class CostDatabase:
    def __init__(self, microbench_root: Path) -> None:
        self.root = microbench_root.resolve()
        if "calibration" in {part.lower() for part in self.root.parts}:
            raise CostDatabaseError("official prediction cannot load a calibration path")
        manifest_path = self.root / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CostDatabaseError(f"cannot load microbenchmark manifest: {exc}") from exc
        entries = manifest.get("operations") or manifest.get("benchmarks") or []
        resource_entries = manifest.get("resource_curves") or []
        if not isinstance(entries, list) or not isinstance(resource_entries, list):
            raise CostDatabaseError("manifest operation/resource collections must be arrays")
        all_entries = list(entries) + list(resource_entries)
        self.records: dict[str, list[CostRecord]] = {}
        self.kinds: dict[str, str] = {}
        self.required_params: dict[str, tuple[str, ...]] = {}
        self.contracts: dict[str, Mapping[str, Any]] = {}
        self._load_entries(all_entries)

    def _load_entries(self, entries: Iterable[Mapping[str, Any]]) -> None:
        loaded_csv: dict[Path, list[dict[str, str]]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            atom_id = entry.get("id") or entry.get("binary") or entry.get("result_name")
            if not isinstance(atom_id, str):
                raise CostDatabaseError("manifest entry lacks a generic id")
            if "calibration" in atom_id.lower() or any(
                "calibration" in str(entry.get(key, "")).lower()
                for key in ("source", "result_csv")
            ):
                raise CostDatabaseError("calibration records are forbidden in CostDatabase")
            kind = str(entry.get("kind", "operation"))
            if kind not in {"operation", "resource_curve"}:
                continue
            family = str(entry.get("family", "unknown"))
            self.required_params[atom_id] = tuple(str(item) for item in entry.get("parameters", []))
            contract = dict(entry.get("protocol", {})) | dict(entry.get("fixed_modifiers", {}))
            self.contracts[atom_id] = contract
            result_csv = entry.get("result_csv")
            if not result_csv:
                source = Path(str(entry.get("source", "")))
                # Generic layout is <category>/<family>/<source>.cu.
                result_csv = str(source.parent / "result.csv")
            csv_path = (self.root / str(result_csv)).resolve()
            if self.root not in csv_path.parents:
                raise CostDatabaseError(f"result_csv escapes microbench root: {result_csv}")
            if "calibration" in {part.lower() for part in csv_path.parts}:
                raise CostDatabaseError("calibration CSV is forbidden in CostDatabase")
            if not csv_path.exists():
                self.kinds[atom_id] = kind
                self.records.setdefault(atom_id, [])
                continue
            if csv_path not in loaded_csv:
                with csv_path.open(newline="", encoding="utf-8") as handle:
                    loaded_csv[csv_path] = list(csv.DictReader(handle))
            matching = [row for row in loaded_csv[csv_path] if row.get("name") == atom_id]
            self.kinds[atom_id] = kind
            target = self.records.setdefault(atom_id, [])
            for row_index, row in enumerate(matching, 2):
                latency_unit = str(row.get("latency_unit", "cycles"))
                if "cycle" not in latency_unit.lower():
                    raise CostDatabaseError(f"{atom_id} latency_unit must be cycles")
                latency = _finite_or(row.get("latency_value"), 0.0, "latency_value")
                p50 = _finite_or(row.get("p50"), latency, "p50")
                ii_fallback = row.get("initiation_interval_cycles") in (None, "")
                ii = _finite_or(row.get("initiation_interval_cycles"), latency, "initiation_interval_cycles")
                params = {
                    key: _parse(value) for key, value in row.items()
                    if key not in _METRIC_COLUMNS and value not in (None, "")
                }
                encoded_params = params.pop("params_json", None)
                if isinstance(encoded_params, dict):
                    params = dict(encoded_params) | params
                compact_args = params.pop("args", None)
                if isinstance(compact_args, dict):
                    params = dict(compact_args) | params
                # The sweep CLI may record ``blocks=0`` for an automatic
                # launch while the benchmark reports the actual launch width
                # separately.  Prediction and throughput normalization must
                # use the measured launch, not the CLI sentinel.
                resolved_blocks = params.get("resolved_blocks")
                if isinstance(resolved_blocks, (int, float)) and resolved_blocks > 0:
                    params["blocks"] = int(resolved_blocks)
                active_sms = params.get("unique_active_sms")
                if isinstance(active_sms, (int, float)) and active_sms > 0:
                    params["active_sm"] = int(active_sms)
                target.append(CostRecord(
                    atom_id, kind, family, params, latency, ii,
                    _finite_or(row.get("throughput_value"), 0.0, "throughput_value"),
                    str(row.get("throughput_unit", "")),
                    _finite_or(row.get("memory_bandwidth_value"), 0.0, "memory_bandwidth_value"),
                    str(row.get("memory_bandwidth_unit", "")),
                    _finite_or(row.get("p10"), p50, "p10"), p50,
                    _finite_or(row.get("p90"), p50, "p90"),
                    {
                        "result_csv": str(csv_path.relative_to(self.root)),
                        "row": row_index,
                        "gpu_uuid": row.get("gpu_uuid"),
                        "gpu_name": row.get("gpu_name"),
                        "sm_clock_mhz": row.get("sm_clock_mhz"),
                        "memory_clock_mhz": row.get("memory_clock_mhz"),
                        "source_sha256": row.get("source_sha256"),
                        "sass_sha256": row.get("sass_sha256"),
                        "initiation_interval_fallback": ii_fallback,
                    },
                ))

    def ensure_coverage(self, atom_ids: Iterable[str]) -> None:
        unknown = sorted(set(atom_ids) - set(self.kinds))
        empty = sorted(atom_id for atom_id in set(atom_ids) if not self.records.get(atom_id))
        failures = []
        if unknown:
            failures.append(f"absent from manifest: {unknown}")
        if empty:
            failures.append(f"missing accepted full-sweep rows: {empty}")
        if failures:
            raise CoverageError("DAG atom coverage failed; " + "; ".join(failures))
        incomplete = sorted(
            atom_id for atom_id in set(atom_ids)
            if any(
                any(parameter not in record.params for parameter in self.required_params.get(atom_id, ()))
                for record in self.records.get(atom_id, [])
            )
        )
        if incomplete:
            raise CoverageError(
                "accepted full-sweep rows omit mandatory manifest parameters: "
                f"{incomplete}"
            )

    @staticmethod
    def _distance(record: CostRecord, query: Mapping[str, Any]) -> tuple[float, int]:
        distance = 0.0
        exact = 0
        for key, wanted in query.items():
            if wanted is None or key not in record.params:
                continue
            actual = record.params[key]
            if actual == wanted or str(actual).lower() == str(wanted).lower():
                exact += 1
            elif isinstance(actual, (int, float)) and isinstance(wanted, (int, float)):
                distance += abs(float(actual) - float(wanted)) / max(abs(float(wanted)), 1.0)
            else:
                distance += 1000.0
        return distance, -exact

    @staticmethod
    def _selected_row(record: CostRecord) -> Mapping[str, Any]:
        """Return only the accepted CSV row actually used by a lookup."""
        return {"atom_id": record.atom_id} | dict(record.provenance)

    @staticmethod
    def _per_sm_throughput(
        record: CostRecord, query: Mapping[str, Any],
    ) -> float:
        """Normalize a grid-scoped operation rate to one active SM.

        Generic operation harnesses publish CUDA-event throughput for the
        complete launch.  Local simulator queues represent one SM, so copying
        that rate to every SM would multiply hardware throughput by the launch
        width.  Explicit per-SM units are left untouched; otherwise the
        measured active-SM count (or the block count capped by device SMs) is
        the divisor.
        """
        value = record.throughput_value
        if value <= 0:
            return value
        unit = record.throughput_unit.lower().replace(" ", "").replace("-", "_")
        if any(marker in unit for marker in ("/sm", "per_sm", "persm")):
            return value
        active = record.params.get("active_sm", record.params.get("unique_active_sms"))
        if not isinstance(active, (int, float)) or active <= 0:
            active = record.params.get("resolved_blocks", record.params.get("blocks"))
            device_sms = query.get("device_sm_count")
            if (isinstance(active, (int, float)) and active > 0
                    and isinstance(device_sms, (int, float)) and device_sms > 0):
                active = min(float(active), float(device_sms))
        divisor = float(active) if isinstance(active, (int, float)) and active > 0 else 1.0
        return value / divisor

    def lookup(self, atom_id: str, params: Mapping[str, Any]) -> OperationCost:
        records = self.records.get(atom_id, [])
        if not records:
            raise CoverageError(f"no full-sweep result for DAG atom {atom_id}")
        missing = sorted(
            parameter for parameter in self.required_params.get(atom_id, ())
            if parameter not in params
        )
        if missing:
            raise CoverageError(
                f"DAG query for {atom_id} omits mandatory sweep parameters: {missing}"
            )
        list_parameters = {
            key: tuple(value) for key, value in params.items()
            if key in self.required_params.get(atom_id, ())
            and isinstance(value, (list, tuple))
        }
        if list_parameters:
            keys = tuple(list_parameters)
            costs = [
                self.lookup(atom_id, dict(params) | dict(zip(keys, values)))
                for values in product(*(list_parameters[key] for key in keys))
            ]
            if not costs:
                raise CoverageError(f"DAG query for {atom_id} has an empty parameter list")
            def average(field: str) -> float:
                return sum(float(getattr(cost, field)) for cost in costs) / len(costs)
            first = costs[0]
            selected_rows = [
                row
                for cost in costs
                for row in cost.provenance.get("selected_rows", ())
            ]
            return OperationCost(
                average("latency_cycles"), average("initiation_interval_cycles"),
                average("p10_cycles"), average("p50_cycles"), average("p90_cycles"),
                average("throughput_value"), first.throughput_unit,
                average("memory_bandwidth_value"), first.memory_bandwidth_unit,
                {"selected_rows": selected_rows},
            )
        contract = self.contracts.get(atom_id, {})
        aliases = {"dtype": "input_dtype"}
        for query_key, expected in params.items():
            contract_key = aliases.get(query_key, query_key)
            if contract_key in contract and str(contract[contract_key]).lower() != str(expected).lower():
                raise CoverageError(
                    f"{atom_id} fixed {contract_key}={contract[contract_key]!r}, "
                    f"DAG requested {expected!r}"
                )
        selected, weights = self._interpolation(records, params)
        def blend(field: str) -> float:
            return sum(float(getattr(record, field)) * weight for record, weight in zip(selected, weights))
        record = selected[0]
        return OperationCost(
            blend("latency_cycles"), blend("initiation_interval_cycles"),
            blend("p10"), blend("p50"), blend("p90"),
            sum(
                self._per_sm_throughput(item, params) * weight
                for item, weight in zip(selected, weights)
            ), record.throughput_unit,
            blend("memory_bandwidth_value"), record.memory_bandwidth_unit,
            {"selected_rows": [self._selected_row(item) for item in selected]},
        )

    @staticmethod
    def _interpolation(
        records: list[CostRecord], query: Mapping[str, Any]
    ) -> tuple[list[CostRecord], list[float]]:
        categorical = {
            key: value for key, value in query.items()
            if value is not None and not isinstance(value, (int, float, bool))
        }
        relevant_categorical = {
            key: value for key, value in categorical.items()
            if any(key in record.params for record in records)
        }
        filtered = [
            record for record in records
            if all(
                key not in record.params
                or str(record.params[key]).lower() == str(value).lower()
                for key, value in relevant_categorical.items()
            )
        ]
        if relevant_categorical and not filtered:
            raise CoverageError(
                "no resource/operation row matches categorical query "
                f"{relevant_categorical}"
            )
        if not filtered:
            filtered = records

        def numeric_distance(record: CostRecord) -> float:
            total = 0.0
            dimensions = 0
            for key, wanted in query.items():
                actual = record.params.get(key)
                if not isinstance(actual, (int, float)) or not isinstance(wanted, (int, float)):
                    continue
                if actual > 0 and wanted > 0 and key in {
                    "working_set_bytes", "blocks", "active_sm", "depth", "outstanding_depth"
                }:
                    total += abs(math.log(float(actual) / float(wanted)))
                else:
                    total += abs(float(actual) - float(wanted)) / max(abs(float(wanted)), 1.0)
                dimensions += 1
            return total / max(dimensions, 1)

        ordered = sorted(filtered, key=numeric_distance)[:4]
        distances = [numeric_distance(record) for record in ordered]
        if not ordered or distances[0] == 0:
            return [ordered[0] if ordered else records[0]], [1.0]
        inverse = [1.0 / max(distance, 1e-12) for distance in distances]
        total = sum(inverse)
        return ordered, [value / total for value in inverse]

    @staticmethod
    def _direction(record: CostRecord) -> str | None:
        explicit = record.params.get("direction")
        if explicit is not None:
            return str(explicit).lower()
        atom_id = record.atom_id.lower()
        if atom_id.startswith(("ld_", "tma_load")):
            return "load"
        if atom_id.startswith("st_"):
            return "store"
        return None

    def _resource_candidates(
        self, resource: str, direction: str | None = None,
    ) -> list[CostRecord]:
        candidates = [
            record for atom_id, records in self.records.items()
            if self.kinds.get(atom_id) == "resource_curve"
            for record in records
            if str(record.params.get(
                "resource", record.params.get("primary_resource", record.family)
            )).lower() == resource.lower()
        ]
        if direction is None:
            return candidates
        directed = [
            record for record in candidates
            if self._direction(record) in {None, direction.lower()}
        ]
        return directed

    def resource_service(
        self, resource: str, params: Mapping[str, Any],
        *, direction: str | None = None,
    ) -> ResourceService | None:
        candidates = [
            record for record in self._resource_candidates(resource, direction)
            if "peer_resource" not in record.params
        ]
        if not candidates:
            return None
        selected, weights = self._interpolation(candidates, params)

        def blend(field: str) -> float:
            return sum(
                float(getattr(record, field)) * weight
                for record, weight in zip(selected, weights)
            )

        first = selected[0]
        return ResourceService(
            blend("throughput_value"), first.throughput_unit,
            blend("memory_bandwidth_value"), first.memory_bandwidth_unit,
            tuple(
                {"atom_id": record.atom_id} | dict(record.provenance)
                for record in selected
            ),
        )

    def resource_interaction(
        self, resource: str, params: Mapping[str, Any],
    ) -> ResourceInteraction:
        all_candidates = self._resource_candidates(resource)
        peer = params.get("peer_resource")
        if peer is None:
            # Operation rows are already selected at the actual block count.
            # Applying a second max/chosen saturation ratio here would charge
            # the same grid scaling twice. Base resource curves supply service
            # rates to the queues; only matched mixed-resource rows are factors.
            return ResourceInteraction(1.0, ())

        peer_candidates = [
            record for record in all_candidates
            if str(record.params.get("peer_resource", "")).lower()
            == str(peer).lower()
        ]
        if not peer_candidates:
            return ResourceInteraction(1.0, ())
        selected, weights = self._interpolation(peer_candidates, params)
        match_keys = {
            "blocks", "working_set_pages", "working_set_bytes", "pattern",
            "cache_mode", "topology", "interaction", "warpgroups",
            "group_size", "depth", "resident_cta", "active_sm",
        }
        slowdowns = []
        provenance: list[Mapping[str, Any]] = []
        for peer_record in selected:
            baseline_candidates = [
                record for record in all_candidates
                if record.atom_id == peer_record.atom_id
                and "peer_resource" not in record.params
                and int(record.params.get("actors", 1) or 1) == 1
            ]
            if not baseline_candidates:
                raise CoverageError(
                    f"resource curve {peer_record.atom_id} peer={peer!r} "
                    "has no actors=1 no-peer baseline"
                )
            baseline_query = {
                key: value for key, value in peer_record.params.items()
                if key in match_keys
            }
            baseline_query["actors"] = 1
            baseline_selected, baseline_weights = self._interpolation(
                baseline_candidates, baseline_query
            )
            peer_rate = (
                peer_record.throughput_value
                or peer_record.memory_bandwidth_value
            )
            baseline_rate = sum(
                (record.throughput_value or record.memory_bandwidth_value) * weight
                for record, weight in zip(baseline_selected, baseline_weights)
            )
            if peer_rate <= 0 or baseline_rate <= 0:
                raise CoverageError(
                    f"resource curve {peer_record.atom_id} has a non-positive "
                    "baseline or peer service rate"
                )
            slowdowns.append(max(1.0, baseline_rate / peer_rate))
            provenance.append(
                {"atom_id": peer_record.atom_id} | dict(peer_record.provenance)
            )
            provenance.extend(
                {"atom_id": record.atom_id} | dict(record.provenance)
                for record in baseline_selected
            )
        return ResourceInteraction(
            sum(value * weight for value, weight in zip(slowdowns, weights)),
            tuple(provenance),
        )

    def resource_slowdown(self, resource: str, params: Mapping[str, Any]) -> float:
        return self.resource_interaction(resource, params).slowdown

    def has_resource_curve(
        self, resource: str, direction: str | None = None,
    ) -> bool:
        return bool(self._resource_candidates(resource, direction))

    def resource_curve_ids(
        self, resource: str, peer_resource: str | None = None,
    ) -> set[str]:
        return {
            record.atom_id for record in self._resource_candidates(resource)
            if (
                (peer_resource is None and "peer_resource" not in record.params)
                or str(record.params.get("peer_resource", "")).lower()
                   == str(peer_resource).lower()
            )
        }

    def provenance(self, atom_ids: Iterable[str]) -> list[Mapping[str, Any]]:
        seen: set[tuple[Any, ...]] = set()
        result = []
        for atom_id in sorted(set(atom_ids)):
            for record in self.records.get(atom_id, []):
                item = {"atom_id": atom_id} | dict(record.provenance)
                key = tuple(sorted(item.items()))
                if key not in seen:
                    seen.add(key)
                    result.append(item)
        return result
