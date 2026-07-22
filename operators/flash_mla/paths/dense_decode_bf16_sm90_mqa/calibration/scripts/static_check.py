#!/usr/bin/env python3
"""CPU-only structure and timed-boundary checks for dense calibration."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import runpy
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[4]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def collect_atom_ids(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        if isinstance(value.get("atom_id"), str):
            found.add(value["atom_id"])
        require("atom_role" not in value,
                "probe DAG uses unresolved atom_role instead of manifest atom_id")
        for child in value.values():
            found.update(collect_atom_ids(child))
    elif isinstance(value, list):
        for child in value:
            found.update(collect_atom_ids(child))
    return found


def timed_nodes(value: dict) -> list[dict]:
    nodes = list(value.get("nodes", []))
    repeat = value.get("repeat")
    if isinstance(repeat, dict):
        nodes.extend(repeat.get("body", []))
    return nodes


def normalized_dtype_structure(value: dict) -> dict:
    """Strip probe-name aliases and normalize BF16/FP16 atom spellings."""
    result = dict(value)
    result.pop("probe", None)
    result.pop("same_structure_as", None)

    def normalize(item: object) -> object:
        if isinstance(item, dict):
            return {key: normalize(child) for key, child in item.items()}
        if isinstance(item, list):
            return [normalize(child) for child in item]
        if isinstance(item, str):
            return (item.replace("BF16", "DTYPE").replace("FP16", "DTYPE")
                    .replace("bf16", "dtype").replace("fp16", "dtype")
                    .replace("cvt_rn_f16_f32", "cvt_rn_dtype_f32"))
        return item

    return normalize(result)


def metadata_case_is_source_defined(case: dict) -> bool:
    batch = int(case["batch"])
    parts = int(case["num-sm-parts"])
    minimum = int(case["seqlen-min"])
    maximum = int(case["seqlen-max"])
    distribution = str(case["seqlen-distribution"])
    if distribution == "uniform":
        seqlens = [maximum] * batch
    elif distribution == "skewed":
        seqlens = [maximum if index % 8 == 0 else minimum
                   for index in range(batch)]
    else:
        return True  # Other generators are checked by the C++ CPU reference.
    fixed = 5
    blocks = [max(length - 1, 0) // 64 + 1 for length in seqlens]
    total = sum(value + fixed for value in blocks)
    payload = (total + parts - 1) // parts + fixed
    request = block = split = 0
    for _ in range(parts):
        if request >= batch:
            return False
        remaining = payload
        while request < batch:
            available = blocks[request] - block
            if remaining >= available + fixed:
                remaining -= available + fixed
                request += 1
                block = split = 0
            else:
                if remaining - fixed > 0:
                    block += remaining - fixed
                    split += 1
                break
    return request == batch and block == 0 and split == 0


def check_local_includes(paths: list[Path]) -> None:
    for path in paths:
        for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1):
            match = re.match(r'\s*#\s*include\s+"([^"]+)"', line)
            if match is None:
                continue
            include = match.group(1)
            candidates = (
                path.parent / include,
                ROOT / include,
                REPO / "microbench" / include,
            )
            require(
                any(candidate.is_file() for candidate in candidates),
                f"{path.relative_to(ROOT)}:{line_number}: missing include {include}",
            )


def main() -> int:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("purpose") == "residual_only", "manifest must be residual-only")
    require(manifest.get("prediction_input") is False,
            "calibration must not be prediction input")
    forbidden = set(manifest["policy"]["forbidden"])
    require({"correction_factor", "offset", "multiplier", "overlap_credit"} <= forbidden,
            "residual policy is incomplete")
    probes = manifest.get("probes", [])
    microbench_manifest = json.loads(
        (REPO / "microbench/manifest.json").read_text(encoding="utf-8"))
    operations = {
        entry["id"] for entry in microbench_manifest.get("benchmarks", [])
        if entry.get("kind") == "operation"
    }
    require(operations, "generic microbench manifest has no operation entries")
    ids = [entry["id"] for entry in probes]
    require(len(ids) == len(set(ids)) == 14, "expected 14 unique calibration probes")
    dags: dict[str, dict] = {}
    for entry in probes:
        probe = entry["id"]
        require(entry["binary"] == probe, f"{probe}: binary must match id")
        require(entry["source"] == f"{probe}.cu", f"{probe}: source must match id")
        source = ROOT / entry["source"]
        dag_path = ROOT / entry["probe_dag"]
        require(source.is_file(), f"missing source: {source}")
        require(dag_path.is_file(), f"missing probe DAG: {dag_path}")
        dag = json.loads(dag_path.read_text(encoding="utf-8"))
        dags[probe] = dag
        require(dag.get("probe") == probe, f"{probe}: probe DAG id mismatch")
        require(dag.get("timer_start") and dag.get("timer_stop"),
                f"{probe}: probe DAG must state timer boundary")
        missing_atoms = collect_atom_ids(dag) - operations
        require(not missing_atoms,
                f"{probe}: unresolved generic atom IDs: {sorted(missing_atoms)}")
        node_atoms = {node["atom_id"] for node in timed_nodes(dag)}
        if any(atom.startswith("tensor_4d_64x64_") for atom in node_atoms):
            require("mbarrier_wait_128" not in node_atoms,
                    f"{probe}: TMA atom already includes expect_tx/completion wait")
    for bf16, fp16 in (
        ("first_score_bf16", "first_score_fp16"),
        ("steady_score_bf16", "steady_score_fp16"),
        ("page_pair_transition_bf16", "page_pair_transition_fp16"),
        ("softmax_page_update_bf16", "softmax_page_update_fp16"),
        ("combine_stage_bf16", "combine_stage_fp16"),
    ):
        require(normalized_dtype_structure(dags[bf16]) ==
                normalized_dtype_structure(dags[fp16]),
                f"{bf16}/{fp16}: dtype variants must have identical probe structure")
    for probe in ("combine_stage_bf16", "combine_stage_fp16"):
        timed_ids = {node["id"] for node in timed_nodes(dags[probe])}
        outside = {node["id"]: node for node in
                   dags[probe].get("outside_clock64_boundary", [])}
        require("dispatch_load" not in timed_ids,
                f"{probe}: split-offset loads occur before clock64")
        require({"dispatch_load", "griddep_wait"} <= set(outside),
                f"{probe}: pre-clock64 operations must be documented")
        require(all(outside[item].get("timed_by_clock64") is False
                    for item in ("dispatch_load", "griddep_wait")),
                f"{probe}: outside operations cannot be charged to residual DAG")
    metadata_atoms = {node["atom_id"] for node in timed_nodes(dags["metadata_stage"])}
    require("ld_global_nc_u32" in metadata_atoms and "ld_global_u32" not in metadata_atoms,
            "metadata probe must map its explicit ld.global.nc.u32 source load")
    source_paths = [*ROOT.glob("*.cu"), *ROOT.joinpath("common").glob("*.cuh")]
    check_local_includes(source_paths)
    all_source = "\n".join(
        path.read_text(encoding="utf-8") for path in source_paths
    )
    require(not re.search(r"#\s*include\s*[<\"][^>\"]*(?:cutlass|cute)",
                          all_source, re.IGNORECASE),
            "CUTLASS/CUTE include found")
    softmax = (ROOT / "common/softmax_stage_bench.cuh").read_text(encoding="utf-8")
    require("rcp.approx" not in softmax,
            "softmax page-update probe must not include final reciprocal")
    page_pair = (ROOT / "common/page_pair_transition_bench.cuh").read_text(encoding="utf-8")
    require("validation_only_not_dense_dag_work" in page_pair,
            "page-pair LDSM must be identified as validation-only")
    epilogue = (ROOT / "common/epilogue_stage_bench.cuh").read_text(encoding="utf-8")
    require("excludes normalization and LSE" in epilogue,
            "store protocol exclusion is missing")
    combine = (ROOT / "common/combine_stage_bench.cuh").read_text(encoding="utf-8")
    dispatch_at = combine.index("const int start_split = load_nc_i32(")
    wait_at = combine.index('asm volatile("griddepcontrol.wait;')
    clock_at = combine.index("const uint64_t start = read_clock64();", wait_at)
    require(dispatch_at < wait_at < clock_at and "post-griddep-wait" in combine,
            "combine clock64 boundary must start after griddep wait")
    pdl = (ROOT / "common/pdl_overlap_bench.cuh").read_text(encoding="utf-8")
    require('latency.add("value", pair_us).add("unit", "us/pair")' in pdl and
            '.add("overlap_us", overlap_us)' in pdl,
            "PDL residual boundary must be the full pair; overlap is diagnostic")
    runner = runpy.run_path(str(ROOT / "scripts/run.py"))
    converted, samples, unit, conversion = runner["normalize_latency_to_cycles"](
        {"value": 2.0, "unit": "us/pair", "samples": [1.0, 3.0]}, "1500"
    )
    require(converted == 3000.0 and samples == [1500.0, 4500.0]
            and unit == "cycle/pair" and conversion["residual_latency_clock_mhz"] == 1500.0,
            "PDL globaltimer-to-cycle residual conversion is invalid")
    sweep = json.loads((ROOT / "scripts/sweep.json").read_text(encoding="utf-8"))
    for preset in ("quick", "full"):
        defaults = sweep[preset]["defaults"]
        cases = sweep[preset]["cases"]
        for probe in ("combine_stage_bf16", "combine_stage_fp16", "pdl_overlap"):
            require(cases.get(probe), f"{preset}: {probe} needs an explicit iters=1 case")
            require(all(int((defaults | case).get("iters", 0)) == 1
                        for case in cases[probe]),
                    f"{preset}: {probe} only accepts --iters=1")
        for case in cases.get("metadata_stage", []):
            require(metadata_case_is_source_defined(defaults | case),
                    f"{preset}: metadata calibration case is source-undefined: {case}")
    with (ROOT / "result.csv").open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    require("duration_seconds" not in header and "command" not in header,
            "result.csv must not contain run-log fields")
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/build.py"), "--dry-run"],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False,
    )
    require(completed.returncode == 0, completed.stderr)
    require(completed.stdout.count("-gencode=arch=compute_90a,code=sm_90a") == 42,
            "dry build must emit binary/PTX/cubin commands for all 14 probes")
    print("calibration static checks passed: 14 probes, residual-only policy, dry build")
    return 0


if __name__ == "__main__":
    sys.exit(main())
