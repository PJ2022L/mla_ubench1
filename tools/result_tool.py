#!/usr/bin/env python3
"""Archive benchmark runs and rebuild machine-readable result summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import selectors
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
ENVIRONMENT_WHITELIST = (
    "CUDA_DEVICE_ORDER",
    "CUDA_HOME",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
)
RESULT_FILE_BY_KIND = {
    "micro": "result.jsonl",
    "e2e": "result.jsonl",
    "model": "predictions.jsonl",
}
COMPARISON_FIELDS = (
    "case_id",
    "model_kind",
    "n_page",
    "num_splits",
    "predicted_cycles",
    "measured_composite_cycles",
    "cycle_error_pct",
    "predicted_e2e_ms",
    "measured_e2e_ms",
    "e2e_error_pct",
    "microbench_run_ids",
    "e2e_run_id",
    "notes",
)
SUMMARY_BASE_FIELDS = (
    "run_id",
    "kind",
    "status",
    "parse_status",
    "started_at",
    "finished_at",
    "duration_seconds",
    "exit_code",
    "record_index",
    "result_file",
    "hostname",
    "git_commit",
    "git_dirty",
    "command",
)


class ResultToolError(RuntimeError):
    """An expected input, execution, or archive error."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def generate_run_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    hostname = re.sub(r"[^A-Za-z0-9._-]+", "-", socket.gethostname()).strip("-._")
    return f"{timestamp}_{hostname or 'host'}_{uuid.uuid4().hex[:8]}"


def _validate_run_id(run_id: str) -> None:
    if not RUN_ID_RE.fullmatch(run_id) or run_id in {".", ".."}:
        raise ResultToolError(
            "run ID must contain only letters, digits, '.', '_', and '-'"
        )


def _run_probe(command: Sequence[str], cwd: Path, timeout: float = 5.0) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc)}
    result: dict[str, Any] = {
        "available": completed.returncode == 0,
        "exit_code": completed.returncode,
    }
    stdout = completed.stdout.rstrip()
    stderr = completed.stderr.rstrip()
    if stdout:
        result["stdout"] = stdout
    if stderr:
        result["stderr"] = stderr
    return result


def _git_status_paths(porcelain: str) -> list[str]:
    entries = porcelain.split("\0")
    paths: list[str] = []
    index = 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if not entry:
            continue
        if len(entry) < 4 or entry[2] != " ":
            continue
        status = entry[:2]
        paths.append(entry[3:])
        if "R" in status or "C" in status:
            if index < len(entries) and entries[index]:
                paths.append(entries[index])
                index += 1
    return paths


def _is_result_archive_path(path: str) -> bool:
    return "result" in Path(path).parts


def _collect_git(cwd: Path) -> dict[str, Any]:
    root_probe = _run_probe(["git", "rev-parse", "--show-toplevel"], cwd)
    if not root_probe.get("available"):
        return root_probe
    root = Path(str(root_probe["stdout"]))
    commit = _run_probe(["git", "rev-parse", "HEAD"], root)
    dirty = _run_probe(
        [
            "git",
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignored=no",
        ],
        root,
    )
    paths = _git_status_paths(str(dirty.get("stdout", ""))) if dirty.get("available") else []
    source_paths = [path for path in paths if not _is_result_archive_path(path)]
    archive_paths = [path for path in paths if _is_result_archive_path(path)]
    return {
        "available": bool(commit.get("available") and dirty.get("available")),
        "root": str(root),
        "commit": commit.get("stdout"),
        "dirty": bool(source_paths) if dirty.get("available") else None,
        "archive_dirty": bool(archive_paths) if dirty.get("available") else None,
        "error": commit.get("error") or dirty.get("error"),
    }


def _collect_toolchain(cwd: Path) -> dict[str, Any]:
    cuda_home = os.environ.get("CUDA_HOME")
    nvcc = Path(cuda_home) / "bin" / "nvcc" if cuda_home else None
    nvcc_command = str(nvcc) if nvcc and nvcc.is_file() else shutil.which("nvcc")
    result: dict[str, Any] = {
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
        }
    }
    if nvcc_command:
        result["nvcc"] = _run_probe([nvcc_command, "--version"], cwd)
        result["nvcc"]["path"] = nvcc_command
    else:
        result["nvcc"] = {"available": False, "error": "nvcc not found"}
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        driver = _run_probe(
            [
                nvidia_smi,
                "--query-gpu=index,name,uuid,driver_version",
                "--format=csv,noheader,nounits",
            ],
            cwd,
        )
        driver["path"] = nvidia_smi
        result["nvidia_smi"] = driver
    else:
        result["nvidia_smi"] = {
            "available": False,
            "error": "nvidia-smi not found",
        }
    return result


def _collect_gpu_snapshot(cwd: Path) -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return {"available": False, "error": "nvidia-smi not found"}
    query = (
        "index,name,uuid,pstate,clocks.current.sm,clocks.current.memory,"
        "power.draw,temperature.gpu"
    )
    result = _run_probe(
        [nvidia_smi, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        cwd,
    )
    result["query"] = query
    return result


def _display_path(path: Path, git_root: Path | None) -> str:
    resolved = path.resolve()
    if git_root is not None:
        try:
            return str(resolved.relative_to(git_root.resolve()))
        except ValueError:
            pass
    return str(resolved)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_local_command(command: str, cwd: Path) -> Path | None:
    candidate = Path(command)
    if candidate.is_absolute():
        return candidate if candidate.is_file() else None
    local = cwd / candidate
    if local.is_file():
        return local
    return None


def _collect_artifacts(
    command: Sequence[str], extra_artifacts: Iterable[Path], cwd: Path, git_root: Path | None
) -> list[dict[str, Any]]:
    candidates: list[tuple[str, Path]] = []
    if command:
        binary = _resolve_local_command(command[0], cwd)
        if binary is not None:
            candidates.append(("command_binary", binary))
            sass = binary.with_suffix(".sass")
            if sass.is_file():
                candidates.append(("sass", sass))
    candidates.extend(("explicit", path if path.is_absolute() else cwd / path) for path in extra_artifacts)

    artifacts: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for role, path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.is_file():
            artifacts.append(
                {"role": role, "path": _display_path(resolved, git_root), "available": False}
            )
            continue
        try:
            artifacts.append(
                {
                    "role": role,
                    "path": _display_path(resolved, git_root),
                    "available": True,
                    "size_bytes": resolved.stat().st_size,
                    "sha256": _sha256(resolved),
                }
            )
        except OSError as exc:
            artifacts.append(
                {
                    "role": role,
                    "path": _display_path(resolved, git_root),
                    "available": False,
                    "error": str(exc),
                }
            )
    return artifacts


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _load_json_text(text: str) -> Any:
    return json.loads(text, parse_constant=_reject_json_constant)


def parse_json_records(stdout: bytes) -> list[dict[str, Any]]:
    text = stdout.decode("utf-8", errors="replace").strip()
    if not text:
        raise ResultToolError("command stdout is empty; expected JSON output")

    try:
        value = _load_json_text(text)
    except (json.JSONDecodeError, ValueError):
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = _load_json_text(line)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ResultToolError(
                    f"stdout is neither one JSON object nor JSONL; line {line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ResultToolError(f"JSONL record on line {line_number} is not an object")
            records.append(value)
        if not records:
            raise ResultToolError("command stdout contains no JSON objects")
        return records

    if not isinstance(value, dict):
        raise ResultToolError("command stdout must contain a JSON object")
    return [value]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(
                json.dumps(
                    record,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
            stream.write("\n")


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = _load_json_text(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ResultToolError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ResultToolError(f"{label} must contain a JSON object")
    return value


def _numeric(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResultToolError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ResultToolError(f"{label} must be finite")
    return result


def _resolve_repo_file(source: str, git_root: Path) -> tuple[Path, Path]:
    root = git_root.resolve()
    path = Path(source).expanduser()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ResultToolError(f"source file is outside the git repository: {resolved}") from exc
    return resolved, relative


def _metric_value(record: Mapping[str, Any], metric: str) -> Any:
    value: Any = record
    for component in metric.split("."):
        if not component or not isinstance(value, Mapping) or component not in value:
            raise ResultToolError(f"source record does not contain metric {metric!r}")
        value = value[component]
    return value


def _load_verified_run_records(
    result_path: Path, run_id: str, expected_kind: str
) -> list[dict[str, Any]]:
    if not result_path.is_file():
        raise ResultToolError(f"result file does not exist: {result_path}")
    try:
        records = parse_json_records(result_path.read_bytes())
    except (OSError, ResultToolError) as exc:
        raise ResultToolError(f"cannot read source records {result_path}: {exc}") from exc
    metadata = _load_json_object(result_path.parent / "metadata.json", "source metadata")
    source_parse = metadata.get("parse")
    if metadata.get("run_id") != run_id:
        raise ResultToolError(
            f"source metadata run_id does not match referenced run_id {run_id!r}"
        )
    if metadata.get("kind") != expected_kind:
        raise ResultToolError(
            f"source metadata for {run_id!r} must have kind={expected_kind}"
        )
    if metadata.get("status") != "ok":
        raise ResultToolError(f"source metadata for {run_id!r} must have status=ok")
    if not isinstance(source_parse, dict) or source_parse.get("status") != "ok":
        raise ResultToolError(
            f"source metadata for {run_id!r} must have parse.status=ok"
        )
    record_count = source_parse.get("record_count")
    if (
        isinstance(record_count, bool)
        or not isinstance(record_count, int)
        or record_count != len(records)
    ):
        raise ResultToolError(
            f"source metadata for {run_id!r} record_count does not match result.jsonl"
        )
    return records


def validate_model_inputs(
    cycles_path: Path, provenance_path: Path, cwd: Path, git_root: Path | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    del cwd
    if git_root is None:
        raise ResultToolError("model provenance requires running inside a git repository")
    cycles = _load_json_object(cycles_path, "cycles JSON")
    provenance = _load_json_object(provenance_path, "provenance JSON")
    cycle_keys = set(cycles)
    provenance_keys = set(provenance)
    if cycle_keys != provenance_keys:
        missing = sorted(cycle_keys - provenance_keys)
        extra = sorted(provenance_keys - cycle_keys)
        details = []
        if missing:
            details.append(f"missing provenance for {missing}")
        if extra:
            details.append(f"unknown provenance keys {extra}")
        raise ResultToolError("provenance keys must match cycles JSON: " + "; ".join(details))

    source_cache: dict[Path, list[dict[str, Any]]] = {}
    normalized_provenance: dict[str, Any] = {}
    for key in sorted(cycles):
        expected = _numeric(cycles[key], f"cycles[{key!r}]")
        entry = provenance[key]
        if not isinstance(entry, dict):
            raise ResultToolError(f"provenance[{key!r}] must be an object")
        required = ("source_file", "record_index", "metric", "run_id")
        missing_fields = [field for field in required if field not in entry]
        if missing_fields:
            raise ResultToolError(
                f"provenance[{key!r}] is missing fields {missing_fields}"
            )
        source_file = entry["source_file"]
        record_index = entry["record_index"]
        metric = entry["metric"]
        if not isinstance(source_file, str) or not source_file:
            raise ResultToolError(f"provenance[{key!r}].source_file must be a string")
        if isinstance(record_index, bool) or not isinstance(record_index, int) or record_index < 0:
            raise ResultToolError(
                f"provenance[{key!r}].record_index must be a non-negative integer"
            )
        if not isinstance(metric, str) or not metric:
            raise ResultToolError(f"provenance[{key!r}].metric must be a string")
        run_id = entry["run_id"]
        if not isinstance(run_id, str) or not run_id:
            raise ResultToolError(f"provenance[{key!r}].run_id must be a non-empty string")
        _validate_run_id(run_id)

        resolved, relative = _resolve_repo_file(source_file, git_root)
        if not resolved.is_file():
            raise ResultToolError(
                f"provenance[{key!r}] source file does not exist: {resolved}"
            )
        expected_suffix = ("result", "runs", run_id, "result.jsonl")
        if len(relative.parts) < len(expected_suffix) or tuple(relative.parts[-4:]) != expected_suffix:
            raise ResultToolError(
                f"provenance[{key!r}] source_file must end in "
                f"result/runs/{run_id}/result.jsonl"
            )
        if resolved not in source_cache:
            source_cache[resolved] = _load_verified_run_records(
                resolved, run_id, "micro"
            )
        records = source_cache[resolved]
        if record_index >= len(records):
            raise ResultToolError(
                f"provenance[{key!r}].record_index={record_index} exceeds "
                f"{len(records)} records in {resolved}"
            )
        actual = _numeric(
            _metric_value(records[record_index], metric),
            f"provenance[{key!r}] source metric",
        )
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-6):
            raise ResultToolError(
                f"provenance[{key!r}] metric value {actual} does not match cycles value {expected}"
            )
        normalized_entry = dict(entry)
        normalized_entry["source_file"] = relative.as_posix()
        normalized_provenance[key] = normalized_entry
    return cycles, normalized_provenance


def load_comparison(path: Path) -> list[dict[str, str]]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ResultToolError(f"cannot read comparison CSV {path}: {exc}") from exc
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = next(reader)
    except StopIteration as exc:
        raise ResultToolError("comparison CSV is empty") from exc
    if header != list(COMPARISON_FIELDS):
        raise ResultToolError(
            "comparison CSV header must exactly match: " + ",".join(COMPARISON_FIELDS)
        )
    rows: list[dict[str, str]] = []
    for line_number, values in enumerate(reader, start=2):
        if len(values) != len(COMPARISON_FIELDS):
            raise ResultToolError(
                f"comparison CSV row {line_number} has {len(values)} fields; "
                f"expected {len(COMPARISON_FIELDS)}"
            )
        rows.append(dict(zip(COMPARISON_FIELDS, values)))
    if not rows:
        raise ResultToolError("comparison CSV must contain at least one data row")
    return rows


def _csv_number(
    row: Mapping[str, str], field: str, case_id: str, *, required: bool = False
) -> float | None:
    text = row[field].strip()
    if not text:
        if required:
            raise ResultToolError(
                f"comparison case {case_id!r} requires a numeric {field}"
            )
        return None
    try:
        value = float(text)
    except ValueError as exc:
        raise ResultToolError(
            f"comparison case {case_id!r} has non-numeric {field}={text!r}"
        ) from exc
    if not math.isfinite(value):
        raise ResultToolError(
            f"comparison case {case_id!r} requires finite {field}"
        )
    return value


def _format_number(value: float) -> str:
    return format(value, ".17g")


def _normalize_error_pct(
    row: Mapping[str, str],
    case_id: str,
    field: str,
    predicted: float | None,
    measured: float | None,
) -> str:
    error = _csv_number(row, field, case_id)
    if predicted is None or measured is None:
        if error is not None:
            raise ResultToolError(
                f"comparison case {case_id!r} may set {field} only when prediction and measurement exist"
            )
        return ""
    expected = 100.0 * (predicted - measured) / measured
    if error is not None and not math.isclose(
        error, expected, rel_tol=1e-9, abs_tol=1e-6
    ):
        raise ResultToolError(
            f"comparison case {case_id!r} {field}={error} does not match "
            f"signed relative error {expected}"
        )
    return _format_number(expected)


def _parse_run_ids(text: str, case_id: str) -> set[str]:
    values = [value.strip() for value in text.split(";") if value.strip()]
    if not values:
        raise ResultToolError(
            f"comparison case {case_id!r} requires microbench_run_ids"
        )
    if len(values) != len(set(values)):
        raise ResultToolError(
            f"comparison case {case_id!r} contains duplicate microbench_run_ids"
        )
    for value in values:
        _validate_run_id(value)
    return set(values)


def _write_comparison(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def validate_comparison(
    records: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, str]],
    cycles: Mapping[str, Any],
    provenance: Mapping[str, Any],
    model_result_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if len(records) != len(rows):
        raise ResultToolError(
            f"comparison row count {len(rows)} does not match prediction record count {len(records)}"
        )
    rows_by_case: dict[str, Mapping[str, str]] = {}
    for row_index, row in enumerate(rows):
        case_id = row["case_id"].strip()
        if not case_id:
            raise ResultToolError(f"comparison row {row_index + 2} requires case_id")
        if case_id in rows_by_case:
            raise ResultToolError(f"comparison case_id must be unique: {case_id!r}")
        rows_by_case[case_id] = row

    if len(records) == 1:
        pairs = [(records[0], rows[0])]
    else:
        records_by_case: dict[str, Mapping[str, Any]] = {}
        for record_index, record in enumerate(records):
            case_id = record.get("case_id")
            if not isinstance(case_id, str) or not case_id:
                raise ResultToolError(
                    f"multi-case prediction record {record_index} requires case_id"
                )
            if case_id in records_by_case:
                raise ResultToolError(
                    f"multi-case prediction case_id must be unique: {case_id!r}"
                )
            records_by_case[case_id] = record
        if set(records_by_case) != set(rows_by_case):
            raise ResultToolError(
                "multi-case prediction case_ids must exactly match comparison case_ids"
            )
        pairs = [
            (record, rows_by_case[str(record["case_id"])]) for record in records
        ]

    provenance_run_ids = {
        str(entry.get("run_id"))
        for entry in provenance.values()
        if isinstance(entry, Mapping) and entry.get("run_id") is not None
    }
    normalized: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, str]] = []
    e2e_cache: dict[str, float] = {}
    missing = object()
    for index, (record, row) in enumerate(pairs):
        case_id = row["case_id"].strip()

        prediction = dict(record)
        record_case_id = prediction.get("case_id")
        if record_case_id is not None and record_case_id != case_id:
            raise ResultToolError(
                f"comparison case_id {case_id!r} does not match prediction case_id "
                f"{record_case_id!r} at record {index}"
            )
        prediction["case_id"] = case_id
        normalized_row = dict(row)
        normalized_row["case_id"] = case_id

        model_kind = prediction.get("model_kind")
        if not isinstance(model_kind, str) or not model_kind:
            raise ResultToolError(
                f"prediction record {index} requires a non-empty model_kind"
            )
        csv_model_kind = row["model_kind"].strip()
        if csv_model_kind and csv_model_kind != model_kind:
            raise ResultToolError(
                f"comparison case {case_id!r} model_kind={csv_model_kind!r} "
                f"does not match prediction value {model_kind!r}"
            )
        normalized_row["model_kind"] = model_kind

        predicted_source = prediction.get("predicted_cycles", prediction.get("T_model", missing))
        if predicted_source is missing:
            raise ResultToolError(
                f"prediction record {index} requires predicted_cycles or T_model"
            )
        predicted_cycles = _numeric(
            predicted_source, f"prediction record {index} predicted cycles"
        )
        prediction["predicted_cycles"] = predicted_cycles
        csv_predicted_cycles = _csv_number(row, "predicted_cycles", case_id)
        if csv_predicted_cycles is not None and not math.isclose(
            csv_predicted_cycles, predicted_cycles, rel_tol=1e-9, abs_tol=1e-6
        ):
            raise ResultToolError(
                f"comparison case {case_id!r} predicted_cycles={csv_predicted_cycles} "
                f"does not match prediction value {predicted_cycles}"
            )
        normalized_row["predicted_cycles"] = _format_number(predicted_cycles)

        record_n_page = prediction.get("n_page", prediction.get("N_page", missing))
        csv_n_page = _csv_number(row, "n_page", case_id)
        if record_n_page is missing:
            normalized_row["n_page"] = (
                _format_number(csv_n_page) if csv_n_page is not None else ""
            )
        else:
            predicted_n_page = _numeric(
                record_n_page, f"prediction record {index} n_page"
            )
            if csv_n_page is not None and not math.isclose(
                csv_n_page, predicted_n_page, rel_tol=0.0, abs_tol=1e-9
            ):
                raise ResultToolError(
                    f"comparison case {case_id!r} n_page must match prediction value "
                    f"{predicted_n_page}"
                )
            normalized_row["n_page"] = _format_number(predicted_n_page)

        record_num_splits = prediction.get("num_splits", missing)
        csv_num_splits = _csv_number(row, "num_splits", case_id)
        if record_num_splits is missing:
            normalized_row["num_splits"] = (
                _format_number(csv_num_splits) if csv_num_splits is not None else ""
            )
        else:
            predicted_num_splits = _numeric(
                record_num_splits, f"prediction record {index} num_splits"
            )
            if csv_num_splits is not None and not math.isclose(
                csv_num_splits, predicted_num_splits, rel_tol=0.0, abs_tol=1e-9
            ):
                raise ResultToolError(
                    f"comparison case {case_id!r} num_splits must match prediction value "
                    f"{predicted_num_splits}"
                )
            normalized_row["num_splits"] = _format_number(predicted_num_splits)

        measured_cycles = _csv_number(
            row, "measured_composite_cycles", case_id
        )
        measured_e2e_ms = _csv_number(row, "measured_e2e_ms", case_id)
        if measured_cycles is None and measured_e2e_ms is None:
            raise ResultToolError(
                f"comparison case {case_id!r} requires measured_composite_cycles "
                "or measured_e2e_ms"
            )
        for field, value in (
            ("measured_composite_cycles", measured_cycles),
            ("measured_e2e_ms", measured_e2e_ms),
        ):
            if value is not None and value <= 0:
                raise ResultToolError(
                    f"comparison case {case_id!r} requires {field} > 0"
                )
        if measured_cycles is not None:
            measured_key = (
                f"T_measured__{case_id}" if len(records) > 1 else "T_measured"
            )
            if measured_key not in cycles or measured_key not in provenance:
                raise ResultToolError(
                    f"comparison case {case_id!r} requires {measured_key} in cycles and provenance"
                )
            provenance_entry = provenance[measured_key]
            if not isinstance(provenance_entry, Mapping) or not provenance_entry.get(
                "run_id"
            ):
                raise ResultToolError(
                    f"comparison case {case_id!r} {measured_key} lacks strict micro provenance"
                )
            measured_source = _numeric(
                cycles[measured_key], f"cycles[{measured_key!r}]"
            )
            if not math.isclose(
                measured_cycles, measured_source, rel_tol=1e-9, abs_tol=1e-6
            ):
                raise ResultToolError(
                    f"comparison case {case_id!r} measured_composite_cycles={measured_cycles} "
                    f"does not match cycles[{measured_key!r}]={measured_source}"
                )
        normalized_row["measured_composite_cycles"] = (
            _format_number(measured_cycles) if measured_cycles is not None else ""
        )
        normalized_row["measured_e2e_ms"] = (
            _format_number(measured_e2e_ms) if measured_e2e_ms is not None else ""
        )

        record_e2e = prediction.get("predicted_e2e_ms", missing)
        csv_predicted_e2e = _csv_number(row, "predicted_e2e_ms", case_id)
        if record_e2e is missing:
            if csv_predicted_e2e is not None:
                raise ResultToolError(
                    f"comparison case {case_id!r} sets predicted_e2e_ms but prediction does not"
                )
            predicted_e2e = None
            normalized_row["predicted_e2e_ms"] = ""
        else:
            predicted_e2e = _numeric(
                record_e2e, f"prediction record {index} predicted_e2e_ms"
            )
            if csv_predicted_e2e is not None and not math.isclose(
                csv_predicted_e2e, predicted_e2e, rel_tol=1e-9, abs_tol=1e-6
            ):
                raise ResultToolError(
                    f"comparison case {case_id!r} predicted_e2e_ms must match "
                    f"prediction value {predicted_e2e}"
                )
            normalized_row["predicted_e2e_ms"] = _format_number(predicted_e2e)

        normalized_row["cycle_error_pct"] = _normalize_error_pct(
            row,
            case_id,
            "cycle_error_pct",
            predicted_cycles,
            measured_cycles,
        )
        normalized_row["e2e_error_pct"] = _normalize_error_pct(
            row,
            case_id,
            "e2e_error_pct",
            predicted_e2e,
            measured_e2e_ms,
        )
        supplied_run_ids = _parse_run_ids(row["microbench_run_ids"], case_id)
        if supplied_run_ids != provenance_run_ids:
            raise ResultToolError(
                f"comparison case {case_id!r} microbench_run_ids must equal "
                f"provenance run IDs {sorted(provenance_run_ids)}"
            )
        normalized_row["microbench_run_ids"] = ";".join(
            sorted(provenance_run_ids)
        )
        e2e_run_id = row["e2e_run_id"].strip()
        if measured_e2e_ms is not None:
            if not e2e_run_id:
                raise ResultToolError(
                    f"comparison case {case_id!r} requires e2e_run_id with measured_e2e_ms"
                )
            _validate_run_id(e2e_run_id)
            if e2e_run_id not in e2e_cache:
                e2e_result = (
                    model_result_dir.parent
                    / "e2e"
                    / "result"
                    / "runs"
                    / e2e_run_id
                    / "result.jsonl"
                )
                e2e_records = _load_verified_run_records(
                    e2e_result, e2e_run_id, "e2e"
                )
                if len(e2e_records) != 1:
                    raise ResultToolError(
                        f"e2e run {e2e_run_id!r} must contain exactly one result record"
                    )
                e2e_cache[e2e_run_id] = _numeric(
                    e2e_records[0].get("latency_ms"),
                    f"e2e run {e2e_run_id!r} latency_ms",
                )
            if not math.isclose(
                measured_e2e_ms,
                e2e_cache[e2e_run_id],
                rel_tol=1e-9,
                abs_tol=1e-6,
            ):
                raise ResultToolError(
                    f"comparison case {case_id!r} measured_e2e_ms={measured_e2e_ms} "
                    f"does not match e2e run latency_ms={e2e_cache[e2e_run_id]}"
                )
        normalized_row["e2e_run_id"] = e2e_run_id
        normalized.append(prediction)
        normalized_rows.append(normalized_row)
    return normalized, normalized_rows


def _copy_streams(command: Sequence[str], cwd: Path, log_path: Path) -> tuple[int, bytes]:
    stdout_chunks: list[bytes] = []
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        message = f"result_tool: failed to start command: {exc}\n".encode()
        log_path.write_bytes(message)
        try:
            sys.stderr.buffer.write(message)
            sys.stderr.buffer.flush()
        except (AttributeError, BrokenPipeError, OSError):
            pass
        return 127, b""

    assert process.stdout is not None and process.stderr is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, (True, sys.stdout))
    selector.register(process.stderr, selectors.EVENT_READ, (False, sys.stderr))
    with log_path.open("wb") as log:
        while selector.get_map():
            for key, _ in selector.select():
                chunk = os.read(key.fileobj.fileno(), 64 * 1024)
                if not chunk:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                is_stdout, destination = key.data
                log.write(chunk)
                log.flush()
                if is_stdout:
                    stdout_chunks.append(chunk)
                try:
                    destination.buffer.write(chunk)
                    destination.buffer.flush()
                except (AttributeError, BrokenPipeError, OSError):
                    pass
    return process.wait(), b"".join(stdout_chunks)


def _git_root(git: Mapping[str, Any]) -> Path | None:
    root = git.get("root")
    return Path(root) if isinstance(root, str) else None


def _safe_json_scalar(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _audit_model_auxiliary_files(
    run_dir: Path, records: Sequence[Mapping[str, Any]]
) -> tuple[str | None, dict[str, dict[str, str]]]:
    paths = {
        "cycles": run_dir / "cycles.json",
        "provenance": run_dir / "provenance.json",
        "comparison": run_dir / "comparison.csv",
    }
    if any(not path.is_file() for path in paths.values()):
        return "missing_result", {}
    try:
        cycles = _load_json_object(paths["cycles"], "cycles JSON")
        provenance = _load_json_object(paths["provenance"], "provenance JSON")
        if set(cycles) != set(provenance):
            raise ResultToolError("cycles and provenance keys do not match")
        for key, entry in provenance.items():
            if not isinstance(entry, dict):
                raise ResultToolError(f"provenance[{key!r}] must be an object")
            if any(
                field not in entry
                for field in ("source_file", "record_index", "metric", "run_id")
            ):
                raise ResultToolError(f"provenance[{key!r}] is incomplete")
        comparison_rows = load_comparison(paths["comparison"])
        if len(comparison_rows) != len(records):
            raise ResultToolError(
                "comparison row count does not match prediction record count"
            )
        comparison_by_case: dict[str, dict[str, str]] = {}
        for row in comparison_rows:
            case_id = row["case_id"].strip()
            if not case_id or case_id in comparison_by_case:
                raise ResultToolError("comparison case_id values must be non-empty and unique")
            comparison_by_case[case_id] = row
        prediction_case_ids: list[str] = []
        for record in records:
            case_id = record.get("case_id")
            if not isinstance(case_id, str) or not case_id:
                raise ResultToolError("prediction records require case_id")
            prediction_case_ids.append(case_id)
        if len(prediction_case_ids) != len(set(prediction_case_ids)):
            raise ResultToolError("prediction case_id values must be unique")
        if set(prediction_case_ids) != set(comparison_by_case):
            raise ResultToolError(
                "prediction case_ids do not match comparison case_ids"
            )
    except ResultToolError:
        return "invalid_result", {}
    return None, comparison_by_case


def _summary_rows(result_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    runs_dir = result_dir / "runs"
    rows: list[dict[str, Any]] = []
    result_fields: set[str] = set()
    if not runs_dir.is_dir():
        return rows, []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        metadata_path = run_dir / "metadata.json"
        try:
            metadata = _load_json_object(metadata_path, "metadata")
        except ResultToolError:
            metadata = {
                "run_id": run_dir.name,
                "status": "invalid_metadata",
                "parse": {"status": "unknown"},
            }
        kind = str(metadata.get("kind", ""))
        result_name = RESULT_FILE_BY_KIND.get(kind, str(metadata.get("result_file", "")))
        result_path = run_dir / result_name if result_name else None
        metadata_status = str(metadata.get("status", "unknown"))
        summary_status = metadata_status
        records: list[dict[str, Any]] = []
        comparison_by_case: dict[str, dict[str, str]] = {}
        if result_path is None or not result_path.is_file():
            if metadata_status == "ok":
                summary_status = "missing_result"
        else:
            try:
                records = parse_json_records(result_path.read_bytes())
            except (OSError, ResultToolError):
                if metadata_status == "ok":
                    summary_status = "invalid_result"
            if metadata_status == "ok" and summary_status == "ok":
                parse_metadata = metadata.get("parse")
                declared_count = (
                    parse_metadata.get("record_count")
                    if isinstance(parse_metadata, dict)
                    else None
                )
                if (
                    not isinstance(parse_metadata, dict)
                    or parse_metadata.get("status") != "ok"
                    or isinstance(declared_count, bool)
                    or not isinstance(declared_count, int)
                    or declared_count != len(records)
                ):
                    summary_status = "invalid_result"
        if kind == "model" and records and summary_status == "ok":
            auxiliary_status, comparison_by_case = _audit_model_auxiliary_files(
                run_dir, records
            )
            if auxiliary_status is not None:
                summary_status = auxiliary_status
        if not records:
            records = [None]  # type: ignore[list-item]
        git = metadata.get("git") if isinstance(metadata.get("git"), dict) else {}
        system = metadata.get("system") if isinstance(metadata.get("system"), dict) else {}
        parse = metadata.get("parse") if isinstance(metadata.get("parse"), dict) else {}
        command = metadata.get("command") if isinstance(metadata.get("command"), dict) else {}
        for index, record in enumerate(records):
            row: dict[str, Any] = {
                "run_id": metadata.get("run_id", run_dir.name),
                "kind": kind,
                "status": summary_status,
                "parse_status": parse.get("status", "unknown"),
                "started_at": metadata.get("started_at", ""),
                "finished_at": metadata.get("finished_at", ""),
                "duration_seconds": metadata.get("duration_seconds", ""),
                "exit_code": metadata.get("exit_code", ""),
                "record_index": index if record is not None else "",
                "result_file": result_name,
                "hostname": system.get("hostname", ""),
                "git_commit": git.get("commit", ""),
                "git_dirty": git.get("dirty", ""),
                "command": command.get("display", ""),
            }
            if record is not None:
                for key, value in record.items():
                    field = f"result.{key}"
                    result_fields.add(field)
                    row[field] = _safe_json_scalar(value)
                if kind == "model":
                    case_id = record.get("case_id")
                    comparison = comparison_by_case.get(str(case_id))
                    if comparison is not None:
                        for key, value in comparison.items():
                            field = f"comparison.{key}"
                            result_fields.add(field)
                            row[field] = value
            rows.append(row)
    return rows, sorted(result_fields)


def summarize(result_dir: Path) -> Path:
    result_dir.mkdir(parents=True, exist_ok=True)
    rows, result_fields = _summary_rows(result_dir)
    fields = [*SUMMARY_BASE_FIELDS, *result_fields]
    destination = result_dir / "summary.csv"
    fd, temporary_name = tempfile.mkstemp(prefix=".summary-", suffix=".csv", dir=result_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return destination


def _make_run_directory(result_dir: Path, requested_run_id: str | None) -> tuple[str, Path]:
    runs_dir = result_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    if requested_run_id is not None:
        _validate_run_id(requested_run_id)
        run_dir = runs_dir / requested_run_id
        try:
            run_dir.mkdir()
        except FileExistsError as exc:
            raise ResultToolError(f"run already exists and will not be overwritten: {run_dir}") from exc
        return requested_run_id, run_dir
    for _ in range(10):
        run_id = generate_run_id()
        run_dir = runs_dir / run_id
        try:
            run_dir.mkdir()
        except FileExistsError:
            continue
        return run_id, run_dir
    raise ResultToolError("could not allocate a unique run ID")


def run_archive(args: argparse.Namespace) -> int:
    cwd = Path.cwd().resolve()
    result_dir = args.result_dir.expanduser()
    if not result_dir.is_absolute():
        result_dir = (cwd / result_dir).resolve()
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise ResultToolError("run requires a command after '--'")

    git = _collect_git(cwd)
    git_root = _git_root(git)
    model_inputs: tuple[dict[str, Any], dict[str, Any]] | None = None
    comparison_rows: list[dict[str, str]] | None = None
    if args.kind == "model":
        if not args.cycles_json or not args.provenance_json or not args.comparison_csv:
            raise ResultToolError(
                "model runs require --cycles-json, --provenance-json, and --comparison-csv"
            )
        model_inputs = validate_model_inputs(
            args.cycles_json.expanduser().resolve(),
            args.provenance_json.expanduser().resolve(),
            cwd,
            git_root,
        )
        comparison_rows = load_comparison(args.comparison_csv.expanduser().resolve())
    elif args.cycles_json or args.provenance_json or args.comparison_csv:
        raise ResultToolError(
            "--cycles-json, --provenance-json, and --comparison-csv are only valid for model runs"
        )

    run_id, run_dir = _make_run_directory(result_dir, args.run_id)
    started_at = _utc_now()
    started_monotonic = time.monotonic()
    result_name = RESULT_FILE_BY_KIND[args.kind]
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "kind": args.kind,
        "status": "running",
        "started_at": started_at,
        "finished_at": None,
        "duration_seconds": None,
        "exit_code": None,
        "result_file": result_name,
        "log_file": "run.log",
        "command": {
            "argv": command,
            "display": shlex.join(command),
            "cwd": str(cwd),
        },
        "system": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
        },
        "environment": {
            key: os.environ[key] for key in ENVIRONMENT_WHITELIST if key in os.environ
        },
        "git": git,
        "toolchain": _collect_toolchain(cwd),
        "gpu": {"before": _collect_gpu_snapshot(cwd), "after": None},
        "artifacts": _collect_artifacts(command, args.artifact, cwd, git_root),
        "parse": {"status": "pending", "record_count": 0, "error": None},
    }
    _write_json(run_dir / "metadata.json", metadata)

    if model_inputs is not None:
        cycles, normalized_provenance = model_inputs
        _write_json(run_dir / "cycles.json", cycles)
        _write_json(run_dir / "provenance.json", normalized_provenance)

    exit_code, stdout = _copy_streams(command, cwd, run_dir / "run.log")
    parse_error: str | None = None
    records: list[dict[str, Any]] = []
    try:
        records = parse_json_records(stdout)
        if args.kind == "model":
            assert comparison_rows is not None and model_inputs is not None
            records, normalized_comparison = validate_comparison(
                records,
                comparison_rows,
                model_inputs[0],
                model_inputs[1],
                result_dir,
            )
        _write_jsonl(run_dir / result_name, records)
        if args.kind == "model":
            _write_comparison(run_dir / "comparison.csv", normalized_comparison)
        parse_status = "ok"
    except (ResultToolError, ValueError, TypeError) as exc:
        parse_status = "parse_error"
        parse_error = str(exc)
        records = []

    if exit_code != 0:
        status = "failed"
    elif parse_status != "ok":
        status = "parse_error"
    else:
        status = "ok"
    metadata.update(
        status=status,
        finished_at=_utc_now(),
        duration_seconds=round(time.monotonic() - started_monotonic, 6),
        exit_code=exit_code,
    )
    metadata["parse"] = {
        "status": parse_status,
        "record_count": len(records),
        "error": parse_error,
    }
    metadata["gpu"]["after"] = _collect_gpu_snapshot(cwd)
    metadata["artifacts"] = _collect_artifacts(command, args.artifact, cwd, git_root)
    _write_json(run_dir / "metadata.json", metadata)
    summarize(result_dir)

    message = f"result_tool: archived {status} run at {run_dir}\n"
    try:
        sys.stderr.write(message)
    except (BrokenPipeError, OSError):
        pass
    if exit_code != 0:
        return 128 + abs(exit_code) if exit_code < 0 else exit_code
    return 0 if parse_status == "ok" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    run_parser = subparsers.add_parser("run", help="run a command and archive its output")
    run_parser.add_argument("--result-dir", type=Path, required=True)
    run_parser.add_argument("--kind", choices=sorted(RESULT_FILE_BY_KIND), required=True)
    run_parser.add_argument("--run-id")
    run_parser.add_argument("--cycles-json", type=Path)
    run_parser.add_argument("--provenance-json", type=Path)
    run_parser.add_argument("--comparison-csv", type=Path)
    run_parser.add_argument(
        "--artifact",
        type=Path,
        action="append",
        default=[],
        help="additional binary or profiler artifact to hash in metadata",
    )
    run_parser.add_argument("command", nargs=argparse.REMAINDER)

    summarize_parser = subparsers.add_parser(
        "summarize", help="rebuild summary.csv from immutable run directories"
    )
    summarize_parser.add_argument("--result-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.subcommand == "run":
            return run_archive(args)
        summarize(args.result_dir.expanduser().resolve())
        return 0
    except ResultToolError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
