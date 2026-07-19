from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from microbench.model.dense_decode.profile import build_profile
from microbench.model.dense_decode.scheduler import schedule_requests
from microbench.scan import (
    load_config,
    main as scan_main,
    parameter_grid,
    validate_scan_parameters,
)
from microbench.scripts.manifest_tool import load_manifest


ROOT = Path(__file__).resolve().parents[1]


def metric(value, unit):
    return {"value": value, "unit": unit, "samples": []}


def result(name):
    return {
        "name": name,
        "params": {"blocks": 1},
        "latency": metric(10.0, "cycles/instruction"),
        "throughput": metric(1.0, "Ginst/s"),
        "memory_bandwidth": metric(None, "GB/s"),
        "hardware_utilization": metric(None, "ratio"),
    }


class ProfileTests(unittest.TestCase):
    def test_profile_prefers_jsonl_accepts_calibration_and_derives_occupancy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [
                result("dense_decode.exp2_approx_ftz_f32"),
                result("dense_decode.calibration.metadata_stage"),
            ]
            (root / "results.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in records),
                encoding="utf-8",
            )
            (root / "results.json").write_text(json.dumps(records), encoding="utf-8")
            (root / "run.log").write_text(
                json.dumps({
                    "event": "result", "result_index": 0, "binary": "exp2_f32"
                }) + "\n",
                encoding="utf-8",
            )
            (root / "dense_decode_resources.json").write_text(
                json.dumps(
                    {
                        "hardware": {
                            "main_threads": 256,
                            "main_registers_per_thread": 198,
                            "main_shared_memory_bytes": 230400,
                            "main_launch_bound_ctas": 1,
                            "combine_threads": 256,
                            "combine_registers_per_thread": 48,
                            "combine_shared_memory_bytes": 8192,
                        }
                    }
                ),
                encoding="utf-8",
            )
            profile = build_profile(
                root,
                manifest_path=ROOT / "manifest.json",
                static_artifacts=root,
                target={
                    "sm_count": 132,
                    "sm_clock_mhz": 1980.0,
                    "hbm_gbps": 3350.0,
                    "l2_bytes": 50 * 1024 * 1024,
                    "registers_per_sm": 65536,
                    "shared_memory_per_sm": 233472,
                    "max_warps_per_sm": 64,
                    "max_ctas_per_sm": 32,
                },
            )
            self.assertEqual(len(profile["records"]), 2)
            self.assertEqual(profile["occupancy"]["main"]["residency"], 1)
            self.assertGreaterEqual(profile["occupancy"]["combine"]["residency"], 2)
            self.assertEqual(
                profile["records"][0]["provenance"]["run"]["binary"],
                "exp2_f32",
            )


class ScanContractTests(unittest.TestCase):
    def test_runner_keeps_results_clean_and_runtime_in_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bin_dir = root / "bin"
            output_dir = root / "run"
            bin_dir.mkdir()
            binary = bin_dir / "exp2_f32"
            binary.write_text(
                """#!/usr/bin/env python3
import json
import sys

params = {}
for token in sys.argv[1:]:
    key, value = token[2:].split("=", 1)
    key = key.replace("-", "_")
    try:
        value = int(value)
    except ValueError:
        try:
            value = float(value)
        except ValueError:
            pass
    params[key] = value

metric = lambda value, unit: {"value": value, "unit": unit, "samples": [value]}
print(json.dumps({
    "name": "dense_decode.exp2_approx_ftz_f32",
    "params": params,
    "latency": metric(3.0, "cycles/instruction"),
    "throughput": metric(1.0, "Ginst/s"),
    "memory_bandwidth": {"value": None, "unit": "GB/s"},
    "hardware_utilization": {"value": None, "unit": "ratio"},
}))
""",
                encoding="utf-8",
            )
            os.chmod(binary, 0o755)

            status = scan_main([
                "--kind", "atom",
                "--preset", "quick",
                "--bench", "exp2_f32",
                "--bin-dir", str(bin_dir),
                "--output-dir", str(output_dir),
                "--timeout", "5",
            ])
            self.assertEqual(status, 0)
            self.assertTrue((output_dir / "results.jsonl").is_file())
            self.assertTrue((output_dir / "run.log").is_file())
            self.assertFalse((output_dir / "results.json").exists())
            self.assertFalse((output_dir / "provenance.jsonl").exists())
            self.assertFalse((output_dir / "failures.jsonl").exists())

            result = json.loads(
                (output_dir / "results.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(set(result), {
                "name", "params", "latency", "throughput",
                "memory_bandwidth", "hardware_utilization",
            })
            events = [
                json.loads(line)
                for line in (output_dir / "run.log").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(
                [item["event"] for item in events],
                ["run_start", "result", "run_end"],
            )
            self.assertIn("args", events[1])
            self.assertIn("duration_seconds", events[1])

    def test_all_grids_match_manifest_parameter_contracts(self):
        manifest = load_manifest(ROOT / "manifest.json")
        config = load_config(ROOT / "config" / "h800.json")
        entries = manifest["benchmarks"] + manifest["calibrations"]
        count = 0
        for preset in ("quick", "full"):
            for entry in entries:
                for params in parameter_grid(config, preset, entry, None):
                    validate_scan_parameters(manifest, entry, params)
                    count += 1
        self.assertGreater(count, len(entries))

    def test_metadata_scan_cases_stay_in_target_defined_domain(self):
        manifest = load_manifest(ROOT / "manifest.json")
        config = load_config(ROOT / "config" / "h800.json")
        entry = next(
            item for item in manifest["calibrations"]
            if item["binary"] == "metadata_stage"
        )
        for preset in ("quick", "full"):
            for params in parameter_grid(config, preset, entry, None):
                batch = params["batch"]
                minimum = params["seqlen_min"]
                maximum = params["seqlen_max"]
                distribution = params["seqlen_distribution"]
                state = params["seed"]
                lengths = []
                for index in range(batch):
                    if distribution == "uniform":
                        value = maximum
                    elif distribution == "ramp":
                        value = maximum if batch == 1 else (
                            minimum + (maximum - minimum) * index // (batch - 1)
                        )
                    elif distribution == "skewed":
                        value = maximum if index % 8 == 0 else minimum
                    else:
                        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
                        value = minimum + state % (maximum - minimum + 1)
                    lengths.append(value)
                scheduled = schedule_requests(
                    lengths,
                    sm_count=132,
                    seqlen_q=1,
                    num_heads_q=128,
                    num_heads_kv=1,
                    num_sm_parts=params["num_sm_parts"],
                )
                self.assertTrue(scheduled.source_defined, params)


if __name__ == "__main__":
    unittest.main()
