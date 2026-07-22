"""Calibration residual reporting; never imported by official prediction."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

from .cost_database import CostDatabase
from .dag import DenseDecodeDAG, Dependency, OperationNode
from .schema import KernelResources
from .simulator import simulate


def _resource(atom_id: str) -> str:
    if atom_id.startswith("m64n"):
        return "tensor"
    if "tensor_4d" in atom_id or "cp_async" in atom_id:
        return "tma"
    if atom_id.startswith(("ld_global", "prefetch")):
        return "l2"
    if atom_id.startswith("st_global"):
        return "hbm"
    if atom_id.startswith(("ld_shared", "st_shared", "stmatrix", "ldmatrix")):
        return "shared"
    if atom_id.startswith(("ex2", "lg2", "rcp")):
        return "sfu"
    if atom_id.startswith(("bar_", "mbarrier", "warp_sync", "fence_")):
        return "barrier"
    if atom_id.startswith(("iadd", "imad", "isetp")):
        return "int32"
    if atom_id.startswith("shfl"):
        return "shuffle"
    if atom_id.startswith("griddep"):
        return "grid"
    return "fp32"


def _amount(value: Any, params: Mapping[str, Any]) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).lower()
    splits = float(params.get("num_splits", 32))
    bucket = float(params.get("max_splits", params.get("bucket", 32)))
    d_v = float(params.get("d_v", 512))
    batch = float(params.get("batch", params.get("batch_size", 1)))
    sm_parts = float(params.get("num_sm_parts", 1))
    if text in {"batch", "batch dependent"}:
        return batch
    if text == "batch + 1":
        return batch + 1
    if text in {"num_sm_parts dependent", "actual cta records"}:
        return float(params.get("actual_cta_records", sm_parts))
    if text in {"prefix_iters", "tail_iters", "consumer_iters"}:
        return float(params.get(text, 1))
    if text == "num_splits":
        return splits
    if "2 *" in text and "split" in text:
        return 2 * splits
    if "2 *" in text and "bucket" in text:
        return 2 * bucket
    if "split" in text and "d_v" in text:
        return splits * d_v
    if "bucket" in text and "+ 5" in text:
        return bucket + 5
    if "split" in text:
        return splits
    if "bucket" in text:
        return bucket
    raise ValueError(f"unsupported calibration work_amount expression: {value}")


def _complete_probe_params(
    atom_id: str, actor: str, benchmark_params: dict[str, Any],
    run_params: Mapping[str, Any], work_amount: float,
) -> None:
    batch = int(run_params.get("batch", run_params.get("batch_size", 1)))
    splits = int(run_params.get("num_splits", 4))
    bucket = int(run_params.get("max_splits", run_params.get("bucket", 32)))
    if atom_id.startswith("tensor_4d_64x64"):
        benchmark_params.setdefault("depth", 8)
        benchmark_params.setdefault("working_set_pages", 64)
        benchmark_params.setdefault("pattern", "sequential")
    elif atom_id.startswith("tensor_4d_64x512"):
        benchmark_params.setdefault("depth", 1)
        benchmark_params.setdefault("working_set_tiles", 64)
    elif atom_id == "cp_async_bulk_s2g_64x512_f32":
        benchmark_params.setdefault("working_set_tiles", 64)
        benchmark_params.setdefault("pattern", "sequential")
    elif atom_id in {"ld_global_nc_u32", "ld_global_u32"}:
        benchmark_params.setdefault("threads", 256 if actor != "warp0" else 32)
        benchmark_params.setdefault("issuers", 32 if actor == "warp0" else 256)
        benchmark_params.setdefault("pattern", "sequential")
        benchmark_params.setdefault("working_set_entries", max(batch, 32))
    elif atom_id == "ld_global_v4_f32":
        benchmark_params.setdefault("segments", splits)
        benchmark_params.setdefault("rowsets", 256)
        benchmark_params.setdefault("warps", 8)
        benchmark_params.setdefault("vectors_per_thread", 4)
        benchmark_params.setdefault("pattern", "sequential")
    elif atom_id == "ld_global_f32_strided":
        benchmark_params.setdefault("segments", bucket)
        benchmark_params.setdefault("split_stride", 128)
        benchmark_params.setdefault("rowsets", 256)
        benchmark_params.setdefault("warps", 8)
        benchmark_params.setdefault("pattern", "sequential")
    elif atom_id == "ld_shared_u32_patterns":
        benchmark_params.setdefault("threads", 256)
        benchmark_params.setdefault("pattern", "warp_broadcast")
        benchmark_params.setdefault("working_set_words", bucket * 8)
    elif atom_id == "st_shared_u32":
        benchmark_params.setdefault("threads", 256)
        benchmark_params.setdefault("producers", 256)
        benchmark_params.setdefault("topology", "contiguous")
        benchmark_params.setdefault("working_set_words", bucket * 8)
    elif atom_id == "st_shared_v2_u32_stride520":
        benchmark_params.setdefault("warpgroups", 2)
        benchmark_params.setdefault("stores_per_thread", 64)
        benchmark_params.setdefault("invalid_tokens", 1)
    elif atom_id.startswith(("stmatrix_", "ldmatrix_")):
        benchmark_params.setdefault(
            "warpgroups", 2 if actor in {"both", "two_wg"} else 1
        )
    elif atom_id == "st_global_f32":
        benchmark_params.setdefault("lane_mode", "width8")
        benchmark_params.setdefault("working_set_records", 64)
        benchmark_params.setdefault("pattern", "sequential")
    elif atom_id == "st_global_u32":
        benchmark_params.setdefault("producers", min(batch + 1, 32))
        benchmark_params.setdefault("working_set_records", max(batch + 1, 64))
        benchmark_params.setdefault("pattern", "sequential")
    elif atom_id == "st_global_v4_u32_32b":
        benchmark_params.setdefault("working_set_records", 64)
        benchmark_params.setdefault("pattern", "sequential")
    elif atom_id == "st_global_u64":
        benchmark_params.setdefault("dtype", str(run_params.get("dtype", "bf16")))
        benchmark_params.setdefault("warps", 8)
        benchmark_params.setdefault("vectors_per_thread", 4)
        benchmark_params.setdefault("working_set_records", 8)
        benchmark_params.setdefault("pattern", "sequential")
    elif atom_id == "shfl_sync_bfly_b32":
        benchmark_params.setdefault(
            "delta", [16, 8, 4, 2, 1] if work_amount >= 5 else 1
        )


def _probe_dag(value: Mapping[str, Any], params: Mapping[str, Any]) -> DenseDecodeDAG:
    dag = DenseDecodeDAG()
    expanded_nodes = list(value.get("nodes", []))
    repeat = value.get("repeat")
    if isinstance(repeat, dict):
        count = int(repeat.get("count", 0))
        for index in range(count):
            for original in repeat.get("body", []):
                condition = str(original.get("active_when", "")).replace(" ", "")
                if condition == "i<8" and not index < 8:
                    continue
                if condition == "i==8" and index != 8:
                    continue
                item = dict(original)
                item["id"] = str(item["id"]).replace("[i]", f"[{index}]")
                expanded_nodes.append(item)
    for item in expanded_nodes:
        atom_id = str(item["atom_id"])
        benchmark_params = dict(item.get("benchmark_params", {})) | dict(params)
        work_amount = _amount(item.get("work_amount", 1), params)
        work_unit = str(item.get("work_unit", "parallel_work"))
        if atom_id.startswith("m64n"):
            instruction_group = int(
                benchmark_params.pop("committed_group_size", 4) or 4
            )
            # Composite probe descriptions record WGMMA instruction counts;
            # normalize by the source's actual committed-group size (1/4/36).
            if instruction_group not in {1, 4, 36}:
                raise ValueError(f"unsupported WGMMA committed group size: {instruction_group}")
            group_size = instruction_group
            work_amount = work_amount / group_size
            work_unit = "committed_group"
            benchmark_params.setdefault("group_size", group_size)
            benchmark_params.setdefault("depth", 1 if group_size == 36 else 4)
            benchmark_params.setdefault(
                "warpgroups", 2 if str(item.get("actor", "")) in {"both", "two_wg"} else 1
            )
        _complete_probe_params(
            atom_id, str(item.get("actor", "cta")), benchmark_params, params, work_amount
        )
        dag.add_node(OperationNode(
            id=str(item["id"]), phase="probe", actor=str(item.get("actor", "cta")),
            atom_id=atom_id,
            benchmark_params=benchmark_params,
            work_amount=work_amount,
            work_unit=work_unit,
            resource_class=str(item.get("resource_class", _resource(atom_id))),
            async_issue=bool(item.get("async_issue", atom_id.startswith(("m64n", "tensor_4d", "cp_async", "griddep")))),
            source_anchor=str(item.get("source_anchor", f"calibration:{value.get('probe', 'probe')}")),
        ))
    expanded_dependencies = []
    for item in value.get("dependencies", []):
        if isinstance(item, list) and any("[i]" in str(part) for part in item):
            count = int(repeat.get("count", 0)) if isinstance(repeat, dict) else 0
            expanded_dependencies.extend([
                [str(part).replace("[i]", f"[{index}]") for part in item]
                for index in range(count)
            ])
        else:
            expanded_dependencies.append(item)
    for item in expanded_dependencies:
        if isinstance(item, list) and len(item) >= 3:
            src_ref, dst_ref, kind = item[:3]
            src, src_event = str(src_ref).rsplit(".", 1)
            dst, dst_event = str(dst_ref).rsplit(".", 1)
            if src in dag.nodes and dst in dag.nodes:
                dag.dependencies.append(Dependency(src, dst, src_event, dst_event, str(kind)))
        elif isinstance(item, dict):
            dag.dependencies.append(Dependency(**item))
        else:
            raise ValueError("invalid calibration dependency")
    if str(value.get("probe", "")).startswith("steady_score_"):
        score_nodes = sorted(
            (node_id for node_id in dag.nodes if "score_" in node_id),
            key=lambda node_id: int(node_id.rsplit("[", 1)[1].rstrip("]")),
        )
        for previous, current in zip(score_nodes, score_nodes[1:]):
            dag.dependencies.append(Dependency(
                previous, current, "issue", "issue", "program",
                "source loop score issue order",
            ))
    dag.validate()
    return dag


def validate_calibration(
    microbench_root: Path,
    calibration_root: Path,
    resources: KernelResources,
) -> dict[str, Any]:
    database = CostDatabase(microbench_root)
    manifest = json.loads((calibration_root / "manifest.json").read_text(encoding="utf-8"))
    probes = manifest.get("probes", manifest.get("calibrations", []))
    if not isinstance(probes, list):
        raise ValueError("calibration manifest requires a probes array")
    result_path = calibration_root / str(manifest.get("result_csv", "result.csv"))
    with result_path.open(newline="", encoding="utf-8") as handle:
        measured_rows = list(csv.DictReader(handle))
    reports = []
    for entry in probes:
        probe = str(entry.get("id", entry.get("name")))
        dag_ref = entry.get("probe_dag")
        if isinstance(dag_ref, str):
            dag_value = json.loads((calibration_root / dag_ref).read_text(encoding="utf-8"))
        else:
            dag_value = entry.get("dag")
        if not isinstance(dag_value, dict):
            raise ValueError(f"calibration probe {probe} lacks its own probe_dag")
        rows = [row for row in measured_rows if row.get("probe", row.get("name")) == probe]
        if not rows:
            raise ValueError(f"calibration result.csv lacks probe {probe}")
        for row in rows:
            unit = str(row.get("measured_latency_unit", row.get("latency_unit", "cycles")))
            if "cycle" not in unit.lower():
                raise ValueError(f"calibration probe {probe} latency must use cycles")
            params = json.loads(row.get("params_json") or "{}")
            prediction = simulate(_probe_dag(dag_value, params), database, resources).result
            observed = float(row.get("p50", row.get("measured_latency_value", 0)))
            predicted = float(prediction["predicted_e2e_cycles"]["p50"])
            residual = observed - predicted
            relative = residual / observed if observed else 0.0
            magnitude = abs(relative)
            status = "pass" if magnitude <= 0.10 else "warn" if magnitude <= 0.20 else "fail"
            reports.append({
                "probe": probe, "params": params,
                "atom_predicted_cycles": predicted,
                "measured_cycles": observed,
                "residual_cycles": residual,
                "relative_error": relative,
                "status": status,
                "suspected_resources": entry.get("suspected_resources", []),
            })
    return {
        "schema_version": 1,
        "policy": "diagnostic_only_no_prediction_correction",
        "thresholds": {"pass": 0.10, "warn": 0.20},
        "probes": reports,
    }
