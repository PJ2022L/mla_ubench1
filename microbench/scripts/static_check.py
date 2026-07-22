#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def executable_sections(text: str, kind: str) -> dict[str, str]:
    if kind == "ptx":
        header = re.compile(
            r"(?m)^(?:\.visible\s+)?\.entry\s+([^\s(]+)\(")
    elif kind == "sass":
        header = re.compile(r"(?m)^\s*Function :\s+(\S+)")
    else:
        raise ValueError(f"unknown executable text kind: {kind}")
    matches = list(header.finditer(text))
    return {
        match.group(1): text[match.start():
                             matches[index + 1].start()
                             if index + 1 < len(matches) else len(text)]
        for index, match in enumerate(matches)
    }


def unique_section(sections: dict[str, str], marker: str,
                   atom_id: str, kind: str) -> str:
    matches = [body for symbol, body in sections.items() if marker in symbol]
    if len(matches) != 1:
        raise SystemExit(
            f"{atom_id}: expected one {kind} function containing {marker!r}, "
            f"found {len(matches)}"
        )
    return matches[0]


def boolean_template_sections(sections: dict[str, str], marker: str,
                              atom_id: str, kind: str) -> tuple[str, str]:
    candidates = {
        symbol: body for symbol, body in sections.items() if marker in symbol
    }
    target = [body for symbol, body in candidates.items() if "Lb1E" in symbol]
    baseline = [body for symbol, body in candidates.items() if "Lb0E" in symbol]
    if len(target) != 1 or len(baseline) != 1:
        raise SystemExit(
            f"{atom_id}: expected one target and one baseline {kind} section "
            f"for {marker!r}, found target={len(target)}, "
            f"baseline={len(baseline)}, total={len(candidates)}"
        )
    return target[0], baseline[0]


def pattern_count(pattern: object, text: str) -> int:
    return len(re.findall(str(pattern), text, re.I | re.S))


def validate_matched_global_baseline(entry: dict[str, object],
                                     ptx: str, sass: str) -> None:
    atom_id = str(entry["id"])
    family = str(entry["family"])
    protocol = entry.get("protocol", {})
    if not isinstance(protocol, dict):
        raise SystemExit(f"{atom_id}: protocol must be an object")
    marker = str(protocol.get("timed_kernel_marker", ""))
    if not marker:
        raise SystemExit(f"{atom_id}: missing timed_kernel_marker")
    target_ptx, baseline_ptx = boolean_template_sections(
        executable_sections(ptx, "ptx"), marker, atom_id, "PTX")
    target_sass, baseline_sass = boolean_template_sections(
        executable_sections(sass, "sass"), marker, atom_id, "SASS")
    for pattern in entry["target_ptx_patterns"]:
        target_count = pattern_count(pattern, target_ptx)
        baseline_count = pattern_count(pattern, baseline_ptx)
        if target_count == 0:
            raise SystemExit(f"{atom_id}: timed target PTX misses {pattern}")
        if family == "global_load" and baseline_count != 0:
            raise SystemExit(
                f"{atom_id}: matched baseline PTX contains target load {pattern}")
        if family == "global_store" and target_count <= baseline_count:
            raise SystemExit(
                f"{atom_id}: target PTX does not add {pattern} beyond "
                "instrumentation stores in its matched baseline")
    for pattern in entry["target_sass_patterns"]:
        target_count = pattern_count(pattern, target_sass)
        baseline_count = pattern_count(pattern, baseline_sass)
        if target_count == 0:
            raise SystemExit(f"{atom_id}: timed target SASS misses {pattern}")
        if family == "global_load" and baseline_count != 0:
            raise SystemExit(
                f"{atom_id}: matched baseline SASS contains target load {pattern}")
        if family == "global_store" and target_count <= baseline_count:
            raise SystemExit(
                f"{atom_id}: target SASS does not add {pattern} beyond "
                "instrumentation stores in its matched baseline")
    for pattern in protocol.get("baseline_required_sass_patterns", []):
        if not re.search(str(pattern), baseline_sass, re.I | re.S):
            raise SystemExit(
                f"{atom_id}: global-memory baseline lost matched dataflow "
                f"{pattern}")


def validate_boolean_timed_kernel(entry: dict[str, object],
                                  ptx: str, sass: str) -> None:
    atom_id = str(entry["id"])
    protocol = entry.get("protocol", {})
    if not isinstance(protocol, dict):
        raise SystemExit(f"{atom_id}: protocol must be an object")
    marker = str(protocol.get("timed_kernel_marker", ""))
    target_ptx, baseline_ptx = boolean_template_sections(
        executable_sections(ptx, "ptx"), marker, atom_id, "PTX")
    target_sass, baseline_sass = boolean_template_sections(
        executable_sections(sass, "sass"), marker, atom_id, "SASS")
    for pattern in entry["target_ptx_patterns"]:
        if not re.search(str(pattern), target_ptx, re.I | re.S):
            raise SystemExit(f"{atom_id}: timed target PTX misses {pattern}")
        if re.search(str(pattern), baseline_ptx, re.I | re.S):
            raise SystemExit(f"{atom_id}: timed baseline PTX contains {pattern}")
    for pattern in entry["target_sass_patterns"]:
        if not re.search(str(pattern), target_sass, re.I | re.S):
            raise SystemExit(f"{atom_id}: timed target SASS misses {pattern}")
        if re.search(str(pattern), baseline_sass, re.I | re.S):
            raise SystemExit(f"{atom_id}: timed baseline SASS contains {pattern}")
    for pattern in entry.get("support_ptx_patterns", []):
        if not re.search(str(pattern), target_ptx, re.I | re.S):
            raise SystemExit(
                f"{atom_id}: timed target PTX misses protocol {pattern}")


def validate_single_timed_kernel(entry: dict[str, object],
                                 ptx: str, sass: str) -> None:
    atom_id = str(entry["id"])
    protocol = entry.get("protocol", {})
    if not isinstance(protocol, dict):
        raise SystemExit(f"{atom_id}: protocol must be an object")
    marker = str(protocol.get("timed_kernel_marker", ""))
    target_ptx = unique_section(
        executable_sections(ptx, "ptx"), marker, atom_id, "PTX")
    target_sass = unique_section(
        executable_sections(sass, "sass"), marker, atom_id, "SASS")
    for pattern in entry["target_ptx_patterns"]:
        if not re.search(str(pattern), target_ptx, re.I | re.S):
            raise SystemExit(f"{atom_id}: timed PTX section misses {pattern}")
    for pattern in entry["target_sass_patterns"]:
        if not re.search(str(pattern), target_sass, re.I | re.S):
            raise SystemExit(f"{atom_id}: timed SASS section misses {pattern}")


def validate_pdl_pair(entry: dict[str, object], ptx: str, sass: str) -> None:
    atom_id = str(entry["id"])
    protocol = entry.get("protocol", {})
    if not isinstance(protocol, dict):
        raise SystemExit(f"{atom_id}: protocol must be an object")
    ptx_sections = executable_sections(ptx, "ptx")
    sass_sections = executable_sections(sass, "sass")
    producer_ptx = unique_section(
        ptx_sections, str(protocol.get("producer_kernel_marker", "")),
        atom_id, "PTX")
    consumer_ptx = unique_section(
        ptx_sections, str(protocol.get("consumer_kernel_marker", "")),
        atom_id, "PTX")
    producer_sass = unique_section(
        sass_sections, str(protocol.get("producer_kernel_marker", "")),
        atom_id, "SASS")
    consumer_sass = unique_section(
        sass_sections, str(protocol.get("consumer_kernel_marker", "")),
        atom_id, "SASS")
    expected = (
        (entry["target_ptx_patterns"][0], producer_ptx, "producer PTX"),
        (entry["target_ptx_patterns"][1], consumer_ptx, "consumer PTX"),
        (entry["target_sass_patterns"][0], producer_sass, "producer SASS"),
        (entry["target_sass_patterns"][1], consumer_sass, "consumer SASS"),
    )
    for pattern, section, label in expected:
        if not re.search(str(pattern), section, re.I | re.S):
            raise SystemExit(f"{atom_id}: {label} misses {pattern}")


def validate_matched_shared_baseline(entry: dict[str, object],
                                     ptx: str, sass: str) -> None:
    atom_id = str(entry["id"])
    protocol = entry.get("protocol", {})
    if not isinstance(protocol, dict):
        raise SystemExit(f"{atom_id}: protocol must be an object")
    required = {
        "latency_target_kernel", "latency_baseline_kernel",
        "throughput_target_kernel", "baseline_required_sass_patterns",
    }
    if not required <= set(protocol):
        raise SystemExit(f"{atom_id}: incomplete matched-baseline protocol")
    ptx_sections = executable_sections(ptx, "ptx")
    sass_sections = executable_sections(sass, "sass")
    latency_target_ptx = unique_section(
        ptx_sections, str(protocol["latency_target_kernel"]), atom_id, "PTX")
    baseline_ptx = unique_section(
        ptx_sections, str(protocol["latency_baseline_kernel"]), atom_id, "PTX")
    throughput_ptx = unique_section(
        ptx_sections, str(protocol["throughput_target_kernel"]), atom_id, "PTX")
    latency_target_sass = unique_section(
        sass_sections, str(protocol["latency_target_kernel"]), atom_id, "SASS")
    baseline_sass = unique_section(
        sass_sections, str(protocol["latency_baseline_kernel"]), atom_id, "SASS")
    throughput_sass = unique_section(
        sass_sections, str(protocol["throughput_target_kernel"]), atom_id, "SASS")
    for pattern in entry["target_ptx_patterns"]:
        if re.search(pattern, baseline_ptx, re.I | re.S):
            raise SystemExit(f"{atom_id}: baseline PTX contains target {pattern}")
        for label, section in (("latency target", latency_target_ptx),
                               ("throughput target", throughput_ptx)):
            if not re.search(pattern, section, re.I | re.S):
                raise SystemExit(f"{atom_id}: {label} PTX misses {pattern}")
    for pattern in entry["target_sass_patterns"]:
        if re.search(pattern, baseline_sass, re.I | re.S):
            raise SystemExit(f"{atom_id}: baseline SASS contains target {pattern}")
        for label, section in (("latency target", latency_target_sass),
                               ("throughput target", throughput_sass)):
            if not re.search(pattern, section, re.I | re.S):
                raise SystemExit(f"{atom_id}: {label} SASS misses {pattern}")
    for pattern in protocol["baseline_required_sass_patterns"]:
        if not re.search(str(pattern), baseline_sass, re.I | re.S):
            raise SystemExit(
                f"{atom_id}: baseline SASS lost matched dataflow {pattern}")
    for pattern in protocol.get("forbidden_target_ptx_patterns", []):
        for label, section in (("latency target", latency_target_ptx),
                               ("throughput target", throughput_ptx)):
            if re.search(str(pattern), section, re.I | re.S):
                raise SystemExit(
                    f"{atom_id}: {label} PTX contains foreign target {pattern}")


def validate_wgmma_source_mode(atom_id: str, mode: str, ptx: str, sass: str) -> None:
    ptx_lines = [
        line for line in ptx.splitlines()
        if "wgmma.mma_async" in line
    ]
    sass_lines = [line for line in sass.splitlines() if "HGMMA." in line]
    if not ptx_lines or not sass_lines:
        raise SystemExit(f"{atom_id}: WGMMA source-mode check found no instructions")
    ptx_rs = re.compile(
        r"\},\s*\{%r[0-9]+(?:,\s*%r[0-9]+){3}\},\s*%rd[0-9]+,\s*p",
        re.I,
    )
    ptx_ss = re.compile(
        r"\},\s*%rd[0-9]+,\s*%rd[0-9]+,\s*p",
        re.I,
    )
    sass_rs = re.compile(
        r"HGMMA\.[A-Z0-9.]+\s+R[0-9]+,\s*R[0-9]+,\s*gdesc\[",
        re.I,
    )
    sass_ss = re.compile(
        r"HGMMA\.[A-Z0-9.]+\s+R[0-9]+,\s*gdesc\[",
        re.I,
    )
    expected_ptx, forbidden_ptx = (ptx_rs, ptx_ss) if mode == "rs" else (ptx_ss, ptx_rs)
    expected_sass, forbidden_sass = (sass_rs, sass_ss) if mode == "rs" else (sass_ss, sass_rs)
    if any(not expected_ptx.search(line) or forbidden_ptx.search(line) for line in ptx_lines):
        raise SystemExit(f"{atom_id}: PTX operand form does not match source_mode={mode}")
    if any(not expected_sass.search(line) or forbidden_sass.search(line) for line in sass_lines):
        raise SystemExit(f"{atom_id}: SASS operand form does not match source_mode={mode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_manifest.py")],
                   check=True)
    if not args.compile:
        return 0
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_all.py")],
                   check=True)
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for entry in manifest["benchmarks"]:
        family = (ROOT / entry["source"]).parent
        atom_id = entry["id"]
        ptx = (family / "build" / "ptx" / f"{atom_id}.ptx").read_text(errors="replace")
        sass = (family / "build" / "sass" / f"{atom_id}.sass").read_text(errors="replace")
        resources = (family / "build" / "resources" / f"{atom_id}.txt").read_text(errors="replace")
        for pattern in entry["target_ptx_patterns"]:
            if not re.search(pattern, ptx, re.I | re.S):
                raise SystemExit(f"{atom_id}: missing target PTX {pattern}")
        for pattern in entry.get("support_ptx_patterns", []):
            if not re.search(pattern, ptx, re.I | re.S):
                raise SystemExit(f"{atom_id}: missing protocol PTX {pattern}")
        for pattern in entry["target_sass_patterns"]:
            if not re.search(pattern, sass, re.I | re.S):
                raise SystemExit(f"{atom_id}: missing target SASS {pattern}")
        if entry.get("family") in {"shared_load", "shared_store"}:
            validate_matched_shared_baseline(entry, ptx, sass)
        if entry.get("family") in {"global_load", "global_store"}:
            validate_matched_global_baseline(entry, ptx, sass)
        if entry.get("family") in {"tma_load", "tma_service"}:
            validate_boolean_timed_kernel(entry, ptx, sass)
        if entry.get("family") in {"memory_service", "interference"}:
            validate_single_timed_kernel(entry, ptx, sass)
        if atom_id == "griddepcontrol_producer_consumer":
            validate_pdl_pair(entry, ptx, sass)
        elif entry.get("family") == "pdl":
            validate_boolean_timed_kernel(entry, ptx, sass)
        if entry.get("family") == "wgmma":
            source = (ROOT / entry["source"]).read_text(encoding="utf-8")
            expected = "Rs" if entry["source_mode"] == "rs" else "Ss"
            if f"Operation::kM64N" not in source or expected not in source:
                raise SystemExit(
                    f"{atom_id}: source-mode macro/trait validation failed")
            required = {"accumulator_dtype", "a_major", "b_major", "swizzle",
                        "transpose_a", "transpose_b", "scale_a", "scale_b",
                        "scale_d"}
            if set(entry.get("fixed_modifiers", {})) != required:
                raise SystemExit(f"{atom_id}: incomplete fixed_modifiers")
            supports_group36 = atom_id.startswith("m64n64k16_ss_")
            expected_groups = [1, 4, 36] if supports_group36 else [1, 4]
            if entry.get("protocol", {}).get("supported_group_sizes") != expected_groups:
                raise SystemExit(f"{atom_id}: incorrect supported_group_sizes")
            expected_constraints = {"36": [1]} if supports_group36 else {}
            if (entry.get("protocol", {}).get("group_size_depth_constraints")
                    != expected_constraints):
                raise SystemExit(f"{atom_id}: incorrect group_size_depth_constraints")
            group36_entry = re.search(
                r"(?ms)^\.visible \.entry [^\n]*ELi36ELi1[^\n]*\(.*?^\}",
                ptx,
            )
            if supports_group36:
                if group36_entry is None:
                    raise SystemExit(f"{atom_id}: missing group_size=36 PTX kernel")
                counts = tuple(group36_entry.group(0).count(opcode) for opcode in (
                    "wgmma.mma_async", "wgmma.commit_group", "wgmma.wait_group",
                ))
                if counts != (37, 2, 2):
                    raise SystemExit(
                        f"{atom_id}: group_size=36 PTX protocol counts are {counts}, "
                        "expected 36 target instructions plus one warmup and two "
                        "commit/wait boundaries"
                    )
                group36_sass = re.search(
                    r"(?ms)Function : [^\n]*ELi36ELi1[^\n]*\n.*?"
                    r"(?=\n\s*Function :|\Z)",
                    sass,
                )
                if group36_sass is None or group36_sass.group(0).count("HGMMA") != 37:
                    raise SystemExit(
                        f"{atom_id}: group_size=36 SASS must retain 37 HGMMA "
                        "instructions including warmup"
                    )
            elif group36_entry is not None:
                raise SystemExit(f"{atom_id}: unexpected group_size=36 PTX kernel")
            validate_wgmma_source_mode(
                atom_id, entry["source_mode"], ptx, sass
            )
        if re.search(r"(?m)^\s*\.local\b", ptx, re.I):
            raise SystemExit(f"{atom_id}: PTX local-memory declaration found")
        if re.search(r"\b(ld|st)\.local\b", ptx, re.I) or re.search(r"\b(LDL|STL)\b", sass):
            raise SystemExit(f"{atom_id}: local-memory instruction found")
        if re.search(r"\b(?:STACK|LOCAL):\s*[1-9][0-9]*", resources, re.I):
            raise SystemExit(f"{atom_id}: non-zero STACK/LOCAL usage found")
        if not re.search(r"STACK[^0-9]*0", resources, re.I) or not re.search(r"LOCAL[^0-9]*0", resources, re.I):
            raise SystemExit(f"{atom_id}: STACK/LOCAL=0 gate failed")
    print(f"static compile gates OK: {len(manifest['benchmarks'])} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
