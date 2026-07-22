#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import sys

from family_runner import (
    COMMON_CSV_COLUMNS,
    WGMMA_CSV_COLUMNS,
    csv_columns_for_family,
    flatten_result,
    source_closure_sha256,
    sweep_parameter_sets,
)


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_TERMS = re.compile(r"(^|[_./])(qk|pv|oaccum|scheduler|tail)([_./]|$)", re.I)
SHA256 = re.compile(r"[0-9a-f]{64}", re.I)
FULL_BLOCK_CURVE = {1, 2, 4, 8, 16, 32, 66, 132, 264, 528}
MEMORY_PATTERNS = {"local", "sequential", "random", "reuse"}
CACHE_MODES = {"l2_hot", "hbm_stream"}
TMA_SERVICE_PAGES = {178, 356, 711, 1422, 2844, 5689}


def fail(message: str) -> None:
    raise SystemExit(message)


def normalized_params(params: dict[str, object]) -> dict[str, object]:
    return {key.replace("-", "_"): value for key, value in params.items()}


def case_signature(atom_id: str, params: dict[str, object]) -> str:
    return atom_id + ":" + json.dumps(
        params, sort_keys=True, separators=(",", ":"))


def csv_scalar(value: object) -> str:
    return "" if value is None else str(value)


def parameter_values(parameter_sets: list[dict[str, object]],
                     name: str) -> set[object]:
    return {
        normalized_params(params).get(name)
        for params in parameter_sets
        if name in normalized_params(params)
    }


def validate_accepted_rows(
    family: Path,
    family_name: str,
    rows: list[dict[str, str]],
    entries_by_id: dict[str, dict[str, object]],
    expected_cases: set[str],
    sweep_keys: dict[str, set[str]],
) -> None:
    if not rows:
        return
    completed_cases: set[str] = set()
    gpu_uuids: set[str] = set()
    for index, row in enumerate(rows, start=2):
        atom_id = row.get("name", "")
        if atom_id not in entries_by_id:
            fail(f"{family}/result.csv:{index}: foreign benchmark {atom_id!r}")
        param_column = "args" if family_name == "wgmma" else "params_json"
        try:
            params = json.loads(row.get(param_column, ""))
        except (TypeError, json.JSONDecodeError) as error:
            fail(
                f"{family}/result.csv:{index}: invalid {param_column}: "
                f"{error}")
        if not isinstance(params, dict):
            fail(
                f"{family}/result.csv:{index}: {param_column} must be an "
                "object")
        missing = sweep_keys[atom_id] - set(params)
        if missing:
            fail(
                f"{family}/result.csv:{index}: missing full-sweep params "
                f"{sorted(missing)}")
        requested = {key: params[key] for key in sweep_keys[atom_id]}
        signature = case_signature(atom_id, requested)
        if signature not in expected_cases:
            fail(
                f"{family}/result.csv:{index}: row is not in the declared "
                f"full grid: {requested}")
        if signature in completed_cases:
            fail(f"{family}/result.csv:{index}: duplicate full-sweep case")
        completed_cases.add(signature)

        gpu_name = row.get("gpu_name") if family_name == "wgmma" else \
            params.get("gpu_name")
        if not isinstance(gpu_name, str) or "H800" not in gpu_name.upper():
            fail(f"{family}/result.csv:{index}: accepted row is not from H800")
        for column in ("gpu_uuid", "sm_clock_mhz", "memory_clock_mhz"):
            value = row.get(column) if family_name == "wgmma" else \
                params.get(column)
            if value in (None, "") or row.get(column, "") != csv_scalar(value):
                fail(
                    f"{family}/result.csv:{index}: incomplete or mismatched "
                    f"{column} provenance")
        gpu_uuids.add(str(row["gpu_uuid"]))
        for column in ("sm_clock_mhz", "memory_clock_mhz"):
            try:
                positive = float(row[column]) > 0
            except (TypeError, ValueError):
                positive = False
            if not positive:
                fail(f"{family}/result.csv:{index}: invalid {column}")
        if not SHA256.fullmatch(row.get("sass_sha256", "")):
            fail(f"{family}/result.csv:{index}: missing SASS SHA-256")
        expected_source_hash = source_closure_sha256(
            ROOT / str(entries_by_id[atom_id]["source"]), [ROOT])
        if row.get("source_sha256") != expected_source_hash:
            fail(f"{family}/result.csv:{index}: stale source SHA-256")
        resolved_blocks = params.get("resolved_blocks")
        if resolved_blocks in (None, ""):
            fail(f"{family}/result.csv:{index}: blocks provenance mismatch")
        if family_name != "wgmma" and \
                row.get("blocks", "") != csv_scalar(resolved_blocks):
            fail(f"{family}/result.csv:{index}: blocks provenance mismatch")
    if completed_cases != expected_cases:
        missing = expected_cases - completed_cases
        extra = completed_cases - expected_cases
        fail(
            f"{family}/result.csv: incomplete accepted full sweep: "
            f"rows={len(rows)}, expected={len(expected_cases)}, "
            f"missing={len(missing)}, extra={len(extra)}")
    if len(gpu_uuids) != 1:
        fail(f"{family}/result.csv: accepted sweep mixes GPU UUIDs")


def main() -> int:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest.get("benchmarks")
    if not isinstance(entries, list) or not entries:
        fail("manifest benchmarks must be a non-empty list")
    seen: set[str] = set()
    families: set[Path] = set()
    for entry in entries:
        atom_id = entry.get("id")
        if not isinstance(atom_id, str) or not atom_id:
            fail("every manifest entry needs a non-empty id")
        if atom_id in seen:
            fail(f"duplicate manifest id: {atom_id}")
        seen.add(atom_id)
        source = ROOT / entry["source"]
        if not source.is_file():
            fail(f"missing source: {source}")
        if source.stem != atom_id or entry.get("binary") != atom_id:
            fail(f"identity invariant failed for {atom_id}")
        if entry.get("kind") not in {"operation", "resource_curve"}:
            fail(f"invalid kind for {atom_id}")
        if FORBIDDEN_TERMS.search(entry["source"]) or FORBIDDEN_TERMS.search(atom_id):
            fail(f"operator-specific name in generic microbenchmark: {atom_id}")
        text = source.read_text(encoding="utf-8")
        if "cutlass" in text.lower() or "cute/" in text.lower():
            fail(f"forbidden dependency in {source}")
        families.add(source.parent)
    if list(ROOT.rglob("benchmark.cu")):
        fail("benchmark.cu is forbidden; source names must equal IDs")
    for forbidden in (ROOT / "model", ROOT / "tests", ROOT / "scan.py"):
        if forbidden.exists():
            fail(f"forbidden microbench path exists: {forbidden}")
    root_common = sorted(path.name for path in (ROOT / "common").iterdir())
    if root_common != ["bench.hpp"]:
        fail(f"root common must contain only bench.hpp, got {root_common}")
    for family in sorted(families):
        family_entries = [
            item for item in entries
            if (ROOT / item["source"]).parent == family
        ]
        family_name = str(family_entries[0]["family"])
        for relative in ("README.md", "result.csv", "scripts/build.py",
                         "scripts/sweep.py", "scripts/sweep.json"):
            if not (family / relative).is_file():
                fail(f"missing {relative} in {family}")
        with (family / "result.csv").open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames or []
            accepted_rows = list(reader)
        expected_header = csv_columns_for_family(family_name)
        if header != expected_header:
            fail(f"result.csv header mismatch in {family}")
        family_ids = {
            entry["id"] for entry in entries
            if (ROOT / entry["source"]).parent == family
        }
        entries_by_id = {
            str(item["id"]): item for item in family_entries
        }
        sweep = json.loads(
            (family / "scripts" / "sweep.json").read_text(encoding="utf-8"))
        full = sweep.get("full", {})
        defaults = full.get("default", {})
        grids = full.get("benchmarks", {})
        expected_cases: set[str] = set()
        sweep_keys: dict[str, set[str]] = {}
        for entry in entries:
            if (ROOT / entry["source"]).parent != family:
                continue
            benchmark = dict(grids.get(entry["id"], {}))
            try:
                parameter_sets = list(sweep_parameter_sets(defaults, benchmark))
            except ValueError as error:
                fail(f"{entry['id']}: invalid full sweep: {error}")
            if not parameter_sets:
                fail(f"{entry['id']}: full sweep has no cases")
            normalized_sets = [normalized_params(params)
                               for params in parameter_sets]
            key_sets = {frozenset(params) for params in normalized_sets}
            if len(key_sets) != 1:
                fail(
                    f"{entry['id']}: full sweep cases use inconsistent "
                    "parameter axes")
            sweep_keys[entry["id"]] = set(next(iter(key_sets)))
            for params in normalized_sets:
                signature = case_signature(entry["id"], params)
                if signature in expected_cases:
                    fail(f"{entry['id']}: duplicate full-sweep case {params}")
                expected_cases.add(signature)
            available = {
                key.replace("-", "_")
                for params in parameter_sets
                for key in params
            }
            missing = set(entry.get("parameters", [])) - available
            if missing:
                fail(
                    f"{entry['id']}: manifest parameters absent from full "
                    f"sweep: {sorted(missing)}")
            if entry.get("family") == "wgmma":
                group36 = [
                    params for params in parameter_sets
                    if params.get("group-size") == 36
                ]
                supports_group36 = entry["id"].startswith("m64n64k16_ss_")
                if supports_group36 and not group36:
                    fail(f"{entry['id']}: full sweep omits group_size=36")
                if any(params.get("depth") != 1 for params in group36):
                    fail(f"{entry['id']}: group_size=36 requires depth=1")
                if not supports_group36 and group36:
                    fail(f"{entry['id']}: group_size=36 is only valid for m64n64 SS")
            if entry.get("family") == "matrix_movement":
                protocol = entry.get("protocol", {})
                if protocol.get("work_unit") != "m64_tile":
                    fail(f"{entry['id']}: matrix movement work unit must be m64_tile")
                if "cycles per complete m64 tile" not in str(
                        protocol.get("initiation_interval_boundary", "")):
                    fail(f"{entry['id']}: matrix tile initiation boundary is missing")
            if entry.get("family") in {"shared_load", "shared_store"}:
                protocol = entry.get("protocol", {})
                required = {
                    "latency_boundary", "throughput_boundary",
                    "latency_target_kernel", "latency_baseline_kernel",
                    "throughput_target_kernel",
                    "baseline_required_sass_patterns",
                }
                if not isinstance(protocol, dict) or not required <= set(protocol):
                    fail(f"{entry['id']}: incomplete matched-baseline protocol")
                if "minus a separately compiled matched" not in str(
                        protocol.get("latency_boundary", "")):
                    fail(f"{entry['id']}: latency boundary does not subtract baseline")
                if "target" not in str(protocol.get("throughput_boundary", "")):
                    fail(f"{entry['id']}: throughput boundary is not target-only")
            if entry.get("family") in {"global_load", "global_store"}:
                protocol = entry.get("protocol", {})
                required = {
                    "latency_boundary", "throughput_boundary",
                    "timed_kernel_marker", "baseline_required_sass_patterns",
                }
                if not isinstance(protocol, dict) or not required <= set(protocol):
                    fail(f"{entry['id']}: incomplete global matched-baseline protocol")
                if not protocol.get("timed_kernel_marker") or not protocol.get(
                        "baseline_required_sass_patterns"):
                    fail(f"{entry['id']}: empty global baseline/static marker")
                if "minus a separately compiled matched" not in str(
                        protocol.get("latency_boundary", "")):
                    fail(f"{entry['id']}: global latency does not subtract baseline")
                if "target" not in str(protocol.get("throughput_boundary", "")):
                    fail(f"{entry['id']}: global throughput is not target-only")
            if entry["id"] in {"ld_global_nc_u32", "ld_global_u32"}:
                if not any(params.get("threads") == 32 for params in parameter_sets):
                    fail(f"{entry['id']}: full sweep omits metadata's 32-thread path")
            if entry["id"] == "st_shared_u32":
                if not any(params.get("threads") == 32 for params in parameter_sets):
                    fail("st_shared_u32: full sweep omits metadata's 32-thread path")
            if entry.get("family") == "interference":
                actors = {params.get("actors") for params in parameter_sets}
                if actors != {1, 2}:
                    fail(f"{entry['id']}: interference sweep needs actors=1,2")
                has_working_pages = "working_set_pages" in available
                expects_working_pages = entry["id"] == "wgmma_tma_interference"
                if has_working_pages != expects_working_pages:
                    fail(
                        f"{entry['id']}: working_set_pages must belong only "
                        "to the WGMMA+TMA interference probe")
                manifest_has_pages = "working_set_pages" in entry.get(
                    "parameters", [])
                if manifest_has_pages != expects_working_pages:
                    fail(f"{entry['id']}: incorrect interference parameters")
                protocol = entry.get("protocol", {})
                if "identical 256-thread" not in str(
                        protocol.get("matched_actor_footprint", "")) or \
                        not protocol.get("timed_kernel_marker"):
                    fail(f"{entry['id']}: missing matched actor footprint protocol")
                if not FULL_BLOCK_CURVE <= parameter_values(
                        parameter_sets, "blocks"):
                    fail(f"{entry['id']}: incomplete interference block curve")
            if entry.get("family") == "memory_service":
                required_axes = {
                    "pattern": MEMORY_PATTERNS,
                    "cache_mode": CACHE_MODES,
                    "outstanding_depth": {1, 2, 4, 8, 16},
                    "threads": {128, 256},
                    "blocks": FULL_BLOCK_CURVE,
                }
                for axis, required_values in required_axes.items():
                    if not required_values <= parameter_values(
                            parameter_sets, axis):
                        fail(
                            f"{entry['id']}: incomplete memory-service "
                            f"{axis} curve")
                protocol = entry.get("protocol", {})
                if protocol.get("resource_scope") != \
                        "active_grid_memory_service" or not protocol.get(
                            "timed_kernel_marker"):
                    fail(f"{entry['id']}: incomplete memory-service protocol")
            if entry["id"] == "tma_load_4d_service":
                required_axes = {
                    "depth": {1, 2, 4, 8},
                    "working_set_pages": TMA_SERVICE_PAGES,
                    "pattern": MEMORY_PATTERNS,
                    "cache_mode": CACHE_MODES,
                    "blocks": FULL_BLOCK_CURVE,
                }
                for axis, required_values in required_axes.items():
                    values = parameter_values(parameter_sets, axis)
                    if axis == "working_set_pages":
                        complete = values == required_values
                    else:
                        complete = required_values <= values
                    if not complete:
                        fail(f"tma_load_4d_service: invalid {axis} curve")
                protocol = entry.get("protocol", {})
                if protocol.get("source_page_bytes") != 64 * 576 * 2:
                    fail(
                        "tma_load_4d_service: source_page_bytes must be "
                        "64*576*2=73728")
            if entry.get("family") == "pdl":
                protocol = entry.get("protocol", {})
                if entry["id"] == "griddepcontrol_producer_consumer":
                    required_parameters = {
                        "producer_blocks", "consumer_blocks", "prefix_iters",
                        "suffix_iters", "consumer_iters",
                    }
                    if entry.get("kind") != "resource_curve" or set(
                            entry.get("parameters", [])) != required_parameters:
                        fail("PDL producer/consumer manifest axes are incomplete")
                    required_axes = {
                        "producer_blocks": {1, 32, 66, 132, 264},
                        "consumer_blocks": {1, 32, 66, 132, 264},
                        "prefix_iters": {512, 4096},
                        "suffix_iters": {512, 4096},
                        "consumer_iters": {512, 4096},
                    }
                    for axis, required_values in required_axes.items():
                        if parameter_values(parameter_sets, axis) != required_values:
                            fail(f"PDL producer/consumer {axis} curve is incomplete")
                    if protocol.get("resource_scope") != \
                            "producer_consumer_grid_pair" or not all(
                                protocol.get(key) for key in (
                                    "producer_kernel_marker",
                                    "consumer_kernel_marker")):
                        fail("PDL producer/consumer kernel protocol is incomplete")
                else:
                    if entry["id"] not in {
                            "griddepcontrol_launch_dependents",
                            "griddepcontrol_wait"} or \
                            entry.get("kind") != "operation":
                        fail(f"{entry['id']}: invalid PDL operation entry")
                    if set(entry.get("parameters", [])) != {"blocks"} or \
                            not protocol.get("timed_kernel_marker") or \
                            "baseline-subtracted" not in str(
                                protocol.get(
                                    "initiation_interval_boundary", "")):
                        fail(f"{entry['id']}: incomplete PDL operation protocol")
        validate_accepted_rows(
            family, family_name, accepted_rows, entries_by_id,
            expected_cases, sweep_keys)
        for header_path in (family / "common").glob("**/*"):
            if header_path.suffix not in {".cu", ".cuh", ".hpp"}:
                continue
            text = header_path.read_text(encoding="utf-8")
            for literal in re.findall(
                    r'kName\s*=\s*"([a-z0-9_]+)"', text):
                if literal not in family_ids:
                    fail(f"{header_path}: owns foreign benchmark ID {literal}")
        if family.name == "matrix_movement":
            harness = (family / "common" / "harness.cuh").read_text(
                encoding="utf-8"
            )
            if 'latency.add_null("value")' in harness:
                fail("matrix movement must publish a non-null tile latency")
            if '.add("initiation_interval_cycles", tile_initiation_interval)' not in harness:
                fail("matrix movement must publish tile initiation_interval_cycles")
        if family.name in {
                "shared_load", "shared_store", "global_load", "global_store"}:
            harness = (family / "common" / "harness.cuh").read_text(
                encoding="utf-8"
            )
            for required_text in (
                    "measure_paired_clock_cycles",
                    '"matched_target_minus_baseline"',
                    '"target_only"'):
                if required_text not in harness:
                    fail(
                        f"{family.name}: missing matched-baseline harness marker "
                        f"{required_text}"
                    )
        if family.name == "interference":
            harness = (family / "common" / "harness.cuh").read_text(
                encoding="utf-8")
            for required_text in (
                    '<<<blocks, 256, shared_bytes>>>',
                    '2 * kWarpgroupThreads, shared_bytes',
                    '"matched_launch_threads"',
                    '"matched_shared_bytes"',
                    '"isolated_row_protocol"',
                    'active_actors == 2'):
                if required_text not in harness:
                    fail(
                        "interference harness lost its matched actors=1/2 "
                        f"footprint marker {required_text!r}")
        if family.name == "pdl":
            required_pdl_ids = {
                "griddepcontrol_launch_dependents",
                "griddepcontrol_wait",
                "griddepcontrol_producer_consumer",
            }
            if not required_pdl_ids <= family_ids:
                fail("PDL family must include both operations and the pair curve")
    for subtree in (ROOT / "common", ROOT / "compute", ROOT / "memory",
                    ROOT / "resource"):
        for path in subtree.glob("**/*"):
            if path.suffix not in {".cu", ".cuh", ".hpp"}:
                continue
            text = path.read_text(encoding="utf-8")
            lowered = text.lower()
            if "cutlass" in lowered or re.search(r'#include\s*[<"]cute(?:/|[."])', lowered):
                fail(f"forbidden CUTLASS/CUTE dependency in {path}")
            if re.search(r"\b(qk|pv|oaccum|scheduler|tail)\b", text, re.I):
                fail(f"operator-role term in generic microbenchmark code: {path}")
    source_ids = {
        source.stem
        for category in ("compute", "memory", "resource")
        for source in (ROOT / category).glob("*/*.cu")
    }
    if source_ids != seen:
        fail(f"manifest/source coverage mismatch: missing={source_ids-seen}, stale={seen-source_ids}")
    matrix_entry = next(
        entry for entry in entries if entry.get("family") == "matrix_movement"
    )
    matrix_record = {
        "name": matrix_entry["id"],
        "params": {
            "warpgroups": 1,
            "blocks": 1,
            "resolved_blocks": 1,
            "initiation_interval_cycles": 8.0,
        },
        "latency": {"value": 12.0, "unit": "cycles/m64_tile", "samples": [12.0]},
        "throughput": {"value": 1.0, "unit": "Gtile/s"},
        "memory_bandwidth": {"value": 1.0, "unit": "GB/s"},
        "hardware_utilization": {"value": 0.5, "unit": "ratio"},
    }
    matrix_row = flatten_result(
        matrix_record, matrix_entry, ROOT / matrix_entry["source"],
        ROOT / "missing-schema-only.sass",
    )
    if matrix_row["latency_value"] != 12.0 or \
            matrix_row["initiation_interval_cycles"] != 8.0:
        fail("matrix movement result schema lost tile latency or initiation interval")
    if list(matrix_row) != COMMON_CSV_COLUMNS:
        fail("non-WGMMA result schema leaked WGMMA-only columns")
    wgmma_entry = next(
        entry for entry in entries if entry.get("family") == "wgmma")
    wgmma_params = {
        "m": 64,
        "n": 64,
        "k": 16,
        "source_mode": "ss",
        "input_dtype": "bf16",
        "accumulator_dtype": "f32",
        "a_major": "k_major",
        "b_major": "k_major",
        "swizzle": "128B",
        "transpose": "none",
        "scale_modifier": "scale_a=1,scale_b=1,scale_d=predicate",
        "warpgroups": 1,
        "group_size": 1,
        "depth": 1,
        "blocks": 1,
        "resolved_blocks": 1,
        "initiation_interval_cycles": 8.0,
    }
    wgmma_record = {
        "name": wgmma_entry["id"],
        "params": wgmma_params,
        "latency": {"value": 8.0, "unit": "cycles", "samples": [8.0]},
        "throughput": {"value": 1.0, "unit": "TFLOP/s"},
        "memory_bandwidth": {"value": None, "unit": "GB/s"},
        "hardware_utilization": {"value": 0.5, "unit": "ratio"},
    }
    wgmma_row = flatten_result(
        wgmma_record, wgmma_entry, ROOT / wgmma_entry["source"],
        ROOT / "missing-schema-only.sass")
    if list(wgmma_row) != WGMMA_CSV_COLUMNS:
        fail("WGMMA result schema is not the compact family schema")
    compact_args = json.loads(wgmma_row["args"])
    expected_args = {
        key: value for key, value in wgmma_params.items()
        if key in {
            "iters", "warmup", "samples", "blocks", "resolved_blocks",
            "warpgroups", "group_size", "depth",
        }
    }
    if compact_args != expected_args:
        fail("WGMMA result schema lost or expanded its compact args")
    if any(column in wgmma_row for column in (
            "memory_bandwidth_value", "hardware_utilization", "m", "n", "k")):
        fail("WGMMA compact result schema retained redundant columns")
    tma_service_source = (
        ROOT / "resource" / "tma_service" / "tma_load_4d_service.cu"
    ).read_text(encoding="utf-8")
    define_at = tma_service_source.find(
        '#define MB_TMA_RESULT_NAME "tma_load_4d_service"'
    )
    include_at = tma_service_source.find('#include "common/harness.cuh"')
    if define_at < 0 or include_at < 0 or define_at > include_at:
        fail("tma_load_4d_service must define its result macro before the harness include")
    tma_harness = (
        ROOT / "memory" / "tma_load" / "common" / "harness.cuh"
    ).read_text(encoding="utf-8")
    required_patterns = {
        "cache-mode parsing": r'args\.get_string\(\s*"cache-mode",\s*"hbm_stream"\s*\)',
        "resource=tma output": r'params\.add\("resource",\s*"tma"\)',
        "result-name macro": r"MB_TMA_RESULT_NAME",
    }
    for label, pattern in required_patterns.items():
        if re.search(pattern, tma_harness) is None:
            fail(f"tma_service resource wrapper contract missing: {label}")
    source_page_patterns = {
        "64 rows": r"constexpr\s+int\s+kRows\s*=\s*64\s*;",
        "576 columns": r"constexpr\s+int\s+kHeadDimension\s*=\s*576\s*;",
        "two-byte source elements": (
            r"constexpr\s+int\s+kPageBytes\s*=\s*kPageElements\s*\*\s*2\s*;"),
        "published source-page bytes": (
            r'\.add\("source_page_bytes",\s*kPageBytes\)'),
    }
    for label, pattern in source_page_patterns.items():
        if re.search(pattern, tma_harness) is None:
            fail(f"tma_service 73728-byte source-page contract missing: {label}")
    runner_source = (ROOT / "scripts" / "family_runner.py").read_text(
        encoding="utf-8")
    runner_markers = {
        "pre-enumerated cases": "expected_cases: set[str] = set()",
        "completed cases": "completed_cases: set[str] = set()",
        "full-set equality": "completed_cases != expected_cases",
        "temporary result": "tempfile.mkstemp(",
        "file fsync": "os.fsync(handle.fileno())",
        "atomic replacement": "os.replace(temporary_path, family_dir / \"result.csv\")",
        "directory fsync": "os.fsync(directory_fd)",
    }
    for label, marker in runner_markers.items():
        if marker not in runner_source:
            fail(f"family runner lost {label} publication gate")
    print(f"manifest OK: {len(entries)} entries across {len(families)} families")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
