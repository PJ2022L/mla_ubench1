#!/usr/bin/env python3
"""Build the dense-decode occupancy contract from cuobjdump resource reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


FUNCTION = re.compile(r"^ Function (.+):$", re.MULTILINE)
RESOURCE = re.compile(
    r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)",
    re.MULTILINE,
)


def parse_functions(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    matches = list(FUNCTION.finditer(text))
    records = []
    for index, match in enumerate(matches):
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        resource = RESOURCE.search(text, match.end(), stop)
        if resource is None:
            continue
        records.append(
            {
                "function": match.group(1),
                "registers_per_thread": int(resource.group(1)),
                "stack_bytes": int(resource.group(2)),
                "static_shared_memory_bytes": int(resource.group(3)),
                "local_memory_bytes": int(resource.group(4)),
            }
        )
    if not records:
        raise ValueError(f"no function resource records found in {path}")
    return records


def select(path: Path, pattern: str) -> dict[str, Any]:
    expression = re.compile(pattern)
    matching = [item for item in parse_functions(path) if expression.search(item["function"])]
    if not matching:
        raise ValueError(f"no function in {path} matches {pattern!r}")
    if any(item["stack_bytes"] or item["local_memory_bytes"] for item in matching):
        raise ValueError(f"selected function in {path} spills to stack/local memory")
    selected = max(
        matching,
        key=lambda item: (
            item["registers_per_thread"], item["static_shared_memory_bytes"]
        ),
    )
    selected["resource_report"] = str(path.resolve())
    selected["selector"] = pattern
    selected["matching_instantiations"] = len(matching)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-resources", required=True, type=Path)
    parser.add_argument("--combine-resources", required=True, type=Path)
    parser.add_argument("--main-function-regex", default=r"flash_fwd_splitkv_mla_kernel")
    parser.add_argument("--combine-function-regex", default=r"flash_fwd_mla_combine_kernel")
    parser.add_argument("--main-threads", type=int, default=256)
    parser.add_argument("--combine-threads", type=int, default=256)
    parser.add_argument("--main-dynamic-shared-bytes", required=True, type=int)
    parser.add_argument("--combine-dynamic-shared-bytes", required=True, type=int)
    parser.add_argument("--main-launch-bound-ctas", type=int, default=1)
    parser.add_argument("--combine-launch-bound-ctas", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    for name in ("main_threads", "combine_threads", "main_dynamic_shared_bytes"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.combine_dynamic_shared_bytes < 0:
        parser.error("--combine-dynamic-shared-bytes must be non-negative")
    main_record = select(args.main_resources, args.main_function_regex)
    combine_record = select(args.combine_resources, args.combine_function_regex)
    payload = {
        "schema_version": 1,
        "hardware": {
            "main_threads": args.main_threads,
            "main_registers_per_thread": main_record["registers_per_thread"],
            "main_shared_memory_bytes": (
                main_record["static_shared_memory_bytes"] +
                args.main_dynamic_shared_bytes
            ),
            "main_launch_bound_ctas": args.main_launch_bound_ctas,
            "combine_threads": args.combine_threads,
            "combine_registers_per_thread": combine_record["registers_per_thread"],
            "combine_shared_memory_bytes": (
                combine_record["static_shared_memory_bytes"] +
                args.combine_dynamic_shared_bytes
            ),
            "combine_launch_bound_ctas": args.combine_launch_bound_ctas,
        },
        "selected_functions": {"main": main_record, "combine": combine_record},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
