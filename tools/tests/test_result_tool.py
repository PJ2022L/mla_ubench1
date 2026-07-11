"""CPU-only tests for tools/result_tool.py."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL = REPO_ROOT / "tools" / "result_tool.py"
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


def run_git(directory: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=directory,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def initialize_git_repo(directory: Path) -> None:
    run_git(directory, "init", "-q")
    run_git(directory, "config", "user.email", "result-tool@example.invalid")
    run_git(directory, "config", "user.name", "Result Tool Test")
    (directory / ".gitignore").write_text("*.bin\n", encoding="utf-8")
    (directory / "tracked.txt").write_text("clean\n", encoding="utf-8")
    run_git(directory, "add", ".gitignore", "tracked.txt")
    run_git(directory, "commit", "-qm", "initial")


def create_micro_run(
    repository: Path,
    run_id: str,
    records: list[dict[str, object]],
    prefix: str = "micro",
) -> Path:
    run_dir = repository / prefix / "result" / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "result.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "kind": "micro",
                "status": "ok",
                "parse": {"status": "ok", "record_count": len(records)},
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def write_comparison(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def create_single_model_inputs(repository: Path) -> tuple[Path, Path]:
    source_run = create_micro_run(
        repository,
        "micro-a",
        [{"median_cycles": 12.5}, {"median_cycles": 40.0}],
    )
    cycles = repository / "cycles.json"
    cycles.write_text(
        json.dumps({"T_atom": 12.5, "T_measured": 40.0}), encoding="utf-8"
    )
    provenance = repository / "provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "T_atom": {
                    "source_file": str(source_run / "result.jsonl"),
                    "record_index": 0,
                    "metric": "median_cycles",
                    "run_id": "micro-a",
                },
                "T_measured": {
                    "source_file": str(source_run / "result.jsonl"),
                    "record_index": 1,
                    "metric": "median_cycles",
                    "run_id": "micro-a",
                }
            }
        ),
        encoding="utf-8",
    )
    return cycles, provenance


def create_e2e_run(
    model_path: Path, run_id: str, latency_ms: float, record_count: int = 1
) -> Path:
    run_dir = model_path / "e2e" / "result" / "runs" / run_id
    run_dir.mkdir(parents=True)
    records = [{"latency_ms": latency_ms} for _ in range(record_count)]
    (run_dir / "result.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "kind": "e2e",
                "status": "ok",
                "parse": {"status": "ok", "record_count": record_count},
            }
        ),
        encoding="utf-8",
    )
    return run_dir


class ResultToolTests(unittest.TestCase):
    def invoke(
        self,
        directory: Path,
        *arguments: str,
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        if extra_env:
            environment.update(extra_env)
        return subprocess.run(
            [sys.executable, str(TOOL), *arguments],
            cwd=directory,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    def run_command(
        self,
        directory: Path,
        result_dir: Path,
        run_id: str,
        program: str,
        kind: str = "micro",
        extra_options: list[str] | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return self.invoke(
            directory,
            "run",
            "--result-dir",
            str(result_dir),
            "--kind",
            kind,
            "--run-id",
            run_id,
            *(extra_options or []),
            "--",
            sys.executable,
            "-c",
            program,
            extra_env=extra_env,
        )

    def load_metadata(self, result_dir: Path, run_id: str) -> dict[str, object]:
        return json.loads(
            (result_dir / "runs" / run_id / "metadata.json").read_text(encoding="utf-8")
        )

    def test_pretty_json_is_normalized_and_environment_is_whitelisted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result_dir = directory / "result"
            completed = self.run_command(
                directory,
                result_dir,
                "pretty",
                "import json; print(json.dumps({'latency_ms': 1.25, 'correct': True}, indent=2))",
                kind="e2e",
                extra_env={
                    "RESULT_TOOL_TEST_SECRET": "must-not-be-recorded",
                    "CONDA_DEFAULT_ENV": "local-only",
                    "CONDA_PREFIX": "/local-only",
                },
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            run_dir = result_dir / "runs" / "pretty"
            lines = (run_dir / "result.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["latency_ms"], 1.25)
            self.assertIn('"latency_ms": 1.25', (run_dir / "run.log").read_text())
            metadata = self.load_metadata(result_dir, "pretty")
            self.assertEqual(metadata["status"], "ok")
            self.assertEqual(metadata["parse"]["record_count"], 1)
            self.assertNotIn("RESULT_TOOL_TEST_SECRET", metadata["environment"])
            self.assertNotIn("CONDA_DEFAULT_ENV", metadata["environment"])
            self.assertNotIn("CONDA_PREFIX", metadata["environment"])
            self.assertIn("before", metadata["gpu"])
            self.assertIn("after", metadata["gpu"])

    def test_jsonl_preserves_multiple_wgmma_cases_and_summary_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result_dir = directory / "result"
            program = (
                "import json; "
                "print(json.dumps({'benchmark':'wgmma','measurement':'latency_full','median_cycles':10})); "
                "print(json.dumps({'benchmark':'wgmma','measurement':'throughput_groups','median_cycles':4}))"
            )
            completed = self.run_command(directory, result_dir, "jsonl", program)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            records = [
                json.loads(line)
                for line in (result_dir / "runs/jsonl/result.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual([record["median_cycles"] for record in records], [10, 4])
            with (result_dir / "summary.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual([row["record_index"] for row in rows], ["0", "1"])
            self.assertEqual(rows[1]["result.measurement"], "throughput_groups")

    def test_failed_command_and_parse_error_are_archived(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result_dir = directory / "result"
            failed = self.run_command(
                directory,
                result_dir,
                "failed",
                "import sys; print('failure detail', file=sys.stderr); raise SystemExit(7)",
            )
            self.assertEqual(failed.returncode, 7)
            failed_metadata = self.load_metadata(result_dir, "failed")
            self.assertEqual(failed_metadata["status"], "failed")
            self.assertEqual(failed_metadata["exit_code"], 7)
            self.assertIn(
                "failure detail",
                (result_dir / "runs/failed/run.log").read_text(encoding="utf-8"),
            )

            parse_error = self.run_command(
                directory, result_dir, "parse-error", "print('not json')"
            )
            self.assertEqual(parse_error.returncode, 2)
            parse_metadata = self.load_metadata(result_dir, "parse-error")
            self.assertEqual(parse_metadata["status"], "parse_error")
            self.assertEqual(parse_metadata["parse"]["status"], "parse_error")

            with (result_dir / "summary.csv").open(newline="", encoding="utf-8") as stream:
                statuses = {row["run_id"]: row["status"] for row in csv.DictReader(stream)}
            self.assertEqual(statuses, {"failed": "failed", "parse-error": "parse_error"})

    def test_run_id_is_immutable_and_generated_id_has_expected_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result_dir = directory / "result"
            first = self.run_command(directory, result_dir, "fixed", "print('{}')")
            self.assertEqual(first.returncode, 0, first.stderr)
            original = (result_dir / "runs/fixed/run.log").read_bytes()
            second = self.run_command(directory, result_dir, "fixed", "print('{\"new\":1}')")
            self.assertEqual(second.returncode, 2)
            self.assertIn("will not be overwritten", second.stderr)
            self.assertEqual((result_dir / "runs/fixed/run.log").read_bytes(), original)

            generated = self.invoke(
                directory,
                "run",
                "--result-dir",
                str(result_dir),
                "--kind",
                "micro",
                "--",
                sys.executable,
                "-c",
                "print('{}')",
            )
            self.assertEqual(generated.returncode, 0, generated.stderr)
            run_ids = {path.name for path in (result_dir / "runs").iterdir()}
            auto_ids = run_ids - {"fixed"}
            self.assertEqual(len(auto_ids), 1)
            self.assertRegex(
                auto_ids.pop(),
                re.compile(r"^\d{8}-\d{6}_[A-Za-z0-9._-]+_[0-9a-f]{8}$"),
            )

    def test_summary_is_fully_rebuilt_from_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result_dir = directory / "result"
            completed = self.run_command(
                directory, result_dir, "rebuild", "print('{\"median_cycles\":12}')"
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            expected = (result_dir / "summary.csv").read_text(encoding="utf-8")
            (result_dir / "summary.csv").unlink()
            rebuilt = self.invoke(
                directory, "summarize", "--result-dir", str(result_dir)
            )
            self.assertEqual(rebuilt.returncode, 0, rebuilt.stderr)
            self.assertEqual((result_dir / "summary.csv").read_text(encoding="utf-8"), expected)

    def test_summary_marks_missing_invalid_and_count_mismatched_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result_dir = directory / "result"
            for run_id in ("missing", "corrupt", "count"):
                completed = self.run_command(
                    directory,
                    result_dir,
                    run_id,
                    f"print('{{\"run\":\"{run_id}\"}}')",
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

            (result_dir / "runs/missing/result.jsonl").unlink()
            (result_dir / "runs/corrupt/result.jsonl").write_text(
                "not json\n", encoding="utf-8"
            )
            count_metadata = self.load_metadata(result_dir, "count")
            count_metadata["parse"]["record_count"] = 2
            (result_dir / "runs/count/metadata.json").write_text(
                json.dumps(count_metadata), encoding="utf-8"
            )
            rebuilt = self.invoke(
                directory, "summarize", "--result-dir", str(result_dir)
            )
            self.assertEqual(rebuilt.returncode, 0, rebuilt.stderr)
            with (result_dir / "summary.csv").open(newline="", encoding="utf-8") as stream:
                statuses = {row["run_id"]: row["status"] for row in csv.DictReader(stream)}
            self.assertEqual(statuses["missing"], "missing_result")
            self.assertEqual(statuses["corrupt"], "invalid_result")
            self.assertEqual(statuses["count"], "invalid_result")

    def test_git_dirty_separates_source_and_result_archives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            initialize_git_repo(directory)
            archived = directory / "old/result/runs/prior"
            archived.mkdir(parents=True)
            (archived / "result.jsonl").write_text("{}\n", encoding="utf-8")
            (directory / "ignored.bin").write_bytes(b"build artifact")
            result_dir = directory / "bench/result"

            clean = self.run_command(
                directory, result_dir, "archive-only", "print('{}')"
            )
            self.assertEqual(clean.returncode, 0, clean.stderr)
            clean_git = self.load_metadata(result_dir, "archive-only")["git"]
            self.assertFalse(clean_git["dirty"])
            self.assertTrue(clean_git["archive_dirty"])

            (directory / "tracked.txt").write_text("modified\n", encoding="utf-8")
            dirty = self.run_command(
                directory, result_dir, "source-dirty", "print('{}')"
            )
            self.assertEqual(dirty.returncode, 0, dirty.stderr)
            dirty_git = self.load_metadata(result_dir, "source-dirty")["git"]
            self.assertTrue(dirty_git["dirty"])
            self.assertTrue(dirty_git["archive_dirty"])

    def test_model_summary_merges_comparison_and_audits_auxiliary_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            initialize_git_repo(directory)
            cycles, provenance = create_single_model_inputs(directory)
            comparison = directory / "comparison.csv"
            write_comparison(
                comparison,
                [
                    {
                        "case_id": "case-a",
                        "measured_composite_cycles": 40,
                        "microbench_run_ids": "micro-a",
                    }
                ],
            )
            result_dir = directory / "path/result"
            completed = self.run_command(
                directory,
                result_dir,
                "model-summary",
                "import json; print(json.dumps({'model_kind':'m','T_model':42}))",
                kind="model",
                extra_options=[
                    "--cycles-json",
                    str(cycles),
                    "--provenance-json",
                    str(provenance),
                    "--comparison-csv",
                    str(comparison),
                ],
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with (result_dir / "summary.csv").open(newline="", encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["comparison.measured_composite_cycles"], "40")
            self.assertAlmostEqual(float(row["comparison.cycle_error_pct"]), 5.0)

            run_dir = result_dir / "runs/model-summary"
            archived_aux = {
                name: (run_dir / name).read_bytes()
                for name in ("cycles.json", "provenance.json", "comparison.csv")
            }
            for name, contents in archived_aux.items():
                with self.subTest(missing_aux=name):
                    (run_dir / name).unlink()
                    rebuilt = self.invoke(
                        directory, "summarize", "--result-dir", str(result_dir)
                    )
                    self.assertEqual(rebuilt.returncode, 0, rebuilt.stderr)
                    with (result_dir / "summary.csv").open(
                        newline="", encoding="utf-8"
                    ) as stream:
                        row = next(csv.DictReader(stream))
                    self.assertEqual(row["status"], "missing_result")
                    (run_dir / name).write_bytes(contents)

            (run_dir / "provenance.json").write_text("[]\n", encoding="utf-8")
            self.invoke(directory, "summarize", "--result-dir", str(result_dir))
            with (result_dir / "summary.csv").open(newline="", encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["status"], "invalid_result")

    def test_model_inputs_are_copied_and_provenance_is_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            initialize_git_repo(directory)
            source_run = create_micro_run(
                directory,
                "micro-a",
                [
                    {"median_cycles": 12.5},
                    {"cycle_per_tile": 7},
                    {"median_cycles": 40},
                ],
            )
            cycles = directory / "cycles-input.json"
            cycles.write_text(
                json.dumps({"T_qk": 12.5, "T_tma": 7, "T_measured": 40}),
                encoding="utf-8",
            )
            provenance = directory / "provenance-input.json"
            provenance.write_text(
                json.dumps(
                    {
                        "T_qk": {
                            "source_file": str(source_run / "result.jsonl"),
                            "record_index": 0,
                            "metric": "median_cycles",
                            "run_id": "micro-a",
                        },
                        "T_tma": {
                            "source_file": "micro/result/runs/micro-a/result.jsonl",
                            "record_index": 1,
                            "metric": "cycle_per_tile",
                            "run_id": "micro-a",
                        },
                        "T_measured": {
                            "source_file": "micro/result/runs/micro-a/result.jsonl",
                            "record_index": 2,
                            "metric": "median_cycles",
                            "run_id": "micro-a",
                        },
                    }
                ),
                encoding="utf-8",
            )
            comparison = directory / "comparison-input.csv"
            write_comparison(
                comparison,
                [
                    {
                        "case_id": "p1",
                        "num_splits": 2,
                        "measured_composite_cycles": 40,
                        "microbench_run_ids": "micro-a",
                    }
                ],
            )
            result_dir = directory / "model-result"
            completed = self.run_command(
                directory,
                result_dir,
                "model",
                "import json; print(json.dumps({'N_page':1,'num_splits':2,'model_kind':'schedule','T_model':42}, indent=2))",
                kind="model",
                extra_options=[
                    "--cycles-json",
                    str(cycles),
                    "--provenance-json",
                    str(provenance),
                    "--comparison-csv",
                    str(comparison),
                ],
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            run_dir = result_dir / "runs/model"
            self.assertTrue((run_dir / "predictions.jsonl").is_file())
            prediction = json.loads((run_dir / "predictions.jsonl").read_text())
            self.assertEqual(prediction["case_id"], "p1")
            self.assertEqual(prediction["predicted_cycles"], 42)
            self.assertEqual(json.loads((run_dir / "cycles.json").read_text()), json.loads(cycles.read_text()))
            copied_provenance = json.loads((run_dir / "provenance.json").read_text())
            self.assertEqual(
                copied_provenance["T_qk"]["source_file"],
                "micro/result/runs/micro-a/result.jsonl",
            )
            with (run_dir / "comparison.csv").open(newline="", encoding="utf-8") as stream:
                copied_comparison = list(csv.DictReader(stream))
            self.assertEqual(len(copied_comparison), 1)
            self.assertEqual(copied_comparison[0]["model_kind"], "schedule")
            self.assertEqual(copied_comparison[0]["n_page"], "1")
            self.assertEqual(copied_comparison[0]["predicted_cycles"], "42")
            self.assertEqual(float(copied_comparison[0]["cycle_error_pct"]), 5.0)

            bad_provenance = directory / "bad-provenance.json"
            bad = json.loads(provenance.read_text())
            bad["T_qk"]["metric"] = "missing"
            bad_provenance.write_text(json.dumps(bad), encoding="utf-8")
            rejected = self.run_command(
                directory,
                result_dir,
                "bad-model",
                "print('{}')",
                kind="model",
                extra_options=[
                    "--cycles-json",
                    str(cycles),
                    "--provenance-json",
                    str(bad_provenance),
                    "--comparison-csv",
                    str(comparison),
                ],
            )
            self.assertEqual(rejected.returncode, 2)
            self.assertIn("does not contain metric", rejected.stderr)
            self.assertFalse((result_dir / "runs/bad-model").exists())

    def test_model_rejects_missing_cycles_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result_dir = directory / "model-result"
            completed = self.run_command(
                directory,
                result_dir,
                "missing-inputs",
                "print('{}')",
                kind="model",
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn(
                "model runs require --cycles-json, --provenance-json, and --comparison-csv",
                completed.stderr,
            )
            self.assertFalse((result_dir / "runs/missing-inputs").exists())

    def test_provenance_requires_real_successful_micro_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            initialize_git_repo(directory)
            cycles, provenance = create_single_model_inputs(directory)
            comparison = directory / "comparison.csv"
            write_comparison(
                comparison,
                [
                    {
                        "case_id": "p1",
                        "num_splits": 1,
                        "measured_composite_cycles": 10,
                        "microbench_run_ids": "micro-a",
                    }
                ],
            )
            result_dir = directory / "model/result"
            base_options = [
                "--cycles-json",
                str(cycles),
                "--provenance-json",
                str(provenance),
                "--comparison-csv",
                str(comparison),
            ]

            missing_run_id = json.loads(provenance.read_text())
            del missing_run_id["T_atom"]["run_id"]
            missing_path = directory / "missing-run-id.json"
            missing_path.write_text(json.dumps(missing_run_id), encoding="utf-8")
            options = list(base_options)
            options[3] = str(missing_path)
            rejected = self.run_command(
                directory,
                result_dir,
                "missing-run-id",
                "print('{}')",
                kind="model",
                extra_options=options,
            )
            self.assertEqual(rejected.returncode, 2)
            self.assertIn("missing fields ['run_id']", rejected.stderr)

            metadata_path = directory / "micro/result/runs/micro-a/metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["parse"]["record_count"] = 3
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            rejected = self.run_command(
                directory,
                result_dir,
                "bad-source-metadata",
                "print('{}')",
                kind="model",
                extra_options=base_options,
            )
            self.assertEqual(rejected.returncode, 2)
            self.assertIn("record_count does not match", rejected.stderr)

            metadata["parse"]["record_count"] = 2
            metadata["status"] = "failed"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            rejected = self.run_command(
                directory,
                result_dir,
                "failed-source-metadata",
                "print('{}')",
                kind="model",
                extra_options=base_options,
            )
            self.assertEqual(rejected.returncode, 2)
            self.assertIn("must have status=ok", rejected.stderr)

            metadata["status"] = "ok"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with tempfile.TemporaryDirectory() as outside_temporary:
                outside = Path(outside_temporary) / "result.jsonl"
                outside.write_text('{"median_cycles":12.5}\n', encoding="utf-8")
                outside_provenance = json.loads(provenance.read_text())
                outside_provenance["T_atom"]["source_file"] = str(outside)
                outside_path = directory / "outside.json"
                outside_path.write_text(json.dumps(outside_provenance), encoding="utf-8")
                options = list(base_options)
                options[3] = str(outside_path)
                rejected = self.run_command(
                    directory,
                    result_dir,
                    "outside-source",
                    "print('{}')",
                    kind="model",
                    extra_options=options,
                )
                self.assertEqual(rejected.returncode, 2)
                self.assertIn("outside the git repository", rejected.stderr)

    def test_comparison_is_completed_and_rejects_inconsistent_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            initialize_git_repo(directory)
            cycles, provenance = create_single_model_inputs(directory)
            result_dir = directory / "model/result"
            create_e2e_run(directory / "model", "e2e-a", 1.0)
            program = (
                "import json; print(json.dumps({'model_kind':'schedule','N_page':2,"
                "'num_splits':3,'T_model':42,'predicted_e2e_ms':1.2}))"
            )

            valid_comparison = directory / "valid.csv"
            write_comparison(
                valid_comparison,
                [
                    {
                        "case_id": "case-a",
                        "measured_composite_cycles": 40,
                        "measured_e2e_ms": 1.0,
                        "microbench_run_ids": "micro-a",
                        "e2e_run_id": "e2e-a",
                    }
                ],
            )
            options = [
                "--cycles-json",
                str(cycles),
                "--provenance-json",
                str(provenance),
                "--comparison-csv",
                str(valid_comparison),
            ]
            completed = self.run_command(
                directory,
                result_dir,
                "valid-comparison",
                program,
                kind="model",
                extra_options=options,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with (result_dir / "runs/valid-comparison/comparison.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["model_kind"], "schedule")
            self.assertEqual(row["n_page"], "2")
            self.assertEqual(row["num_splits"], "3")
            self.assertEqual(row["predicted_cycles"], "42")
            self.assertAlmostEqual(float(row["cycle_error_pct"]), 5.0)
            self.assertAlmostEqual(float(row["e2e_error_pct"]), 20.0)

            invalid_rows = {
                "bad-model-kind": {"model_kind": "wrong"},
                "bad-num-splits": {"num_splits": 4},
                "bad-predicted": {"predicted_cycles": 41},
                "bad-error": {"cycle_error_pct": 99},
                "bad-run-ids": {"microbench_run_ids": "other-run"},
                "missing-e2e-run": {"measured_e2e_ms": 1.0},
                "wrong-e2e-value": {
                    "measured_e2e_ms": 0.9,
                    "e2e_run_id": "e2e-a",
                },
                "missing-measurement": {"measured_composite_cycles": ""},
            }
            for run_id, override in invalid_rows.items():
                with self.subTest(run_id=run_id):
                    path = directory / f"{run_id}.csv"
                    row_input = {
                        "case_id": "case-a",
                        "num_splits": 3,
                        "measured_composite_cycles": 40,
                        "microbench_run_ids": "micro-a",
                    }
                    row_input.update(override)
                    write_comparison(path, [row_input])
                    invalid_options = list(options)
                    invalid_options[-1] = str(path)
                    rejected = self.run_command(
                        directory,
                        result_dir,
                        run_id,
                        program,
                        kind="model",
                        extra_options=invalid_options,
                    )
                    self.assertEqual(rejected.returncode, 2)
                    metadata = self.load_metadata(result_dir, run_id)
                    self.assertEqual(metadata["status"], "parse_error")

    def test_comparison_requires_exact_header_and_a_data_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            initialize_git_repo(directory)
            cycles, provenance = create_single_model_inputs(directory)
            result_dir = directory / "model/result"
            base_options = [
                "--cycles-json",
                str(cycles),
                "--provenance-json",
                str(provenance),
                "--comparison-csv",
                "",
            ]

            header_only = directory / "header-only.csv"
            write_comparison(header_only, [])
            options = list(base_options)
            options[-1] = str(header_only)
            rejected = self.run_command(
                directory,
                result_dir,
                "header-only",
                "print('{}')",
                kind="model",
                extra_options=options,
            )
            self.assertEqual(rejected.returncode, 2)
            self.assertIn("at least one data row", rejected.stderr)

            wrong_header = directory / "wrong-header.csv"
            wrong_header.write_text("case_id,predicted_cycles\np1,42\n", encoding="utf-8")
            options[-1] = str(wrong_header)
            rejected = self.run_command(
                directory,
                result_dir,
                "wrong-header",
                "print('{}')",
                kind="model",
                extra_options=options,
            )
            self.assertEqual(rejected.returncode, 2)
            self.assertIn("header must exactly match", rejected.stderr)

    def test_multi_case_predictions_join_by_case_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            initialize_git_repo(directory)
            source_run = create_micro_run(
                directory,
                "micro-a",
                [
                    {"median_cycles": 1},
                    {"median_cycles": 10},
                    {"median_cycles": 20},
                ],
            )
            cycles = directory / "cycles.json"
            cycles.write_text(
                json.dumps(
                    {
                        "T_atom": 1,
                        "T_measured__a": 10,
                        "T_measured__b": 20,
                    }
                ),
                encoding="utf-8",
            )
            provenance_entries = {}
            for key, index in (
                ("T_atom", 0),
                ("T_measured__a", 1),
                ("T_measured__b", 2),
            ):
                provenance_entries[key] = {
                    "source_file": str(source_run / "result.jsonl"),
                    "record_index": index,
                    "metric": "median_cycles",
                    "run_id": "micro-a",
                }
            provenance = directory / "provenance.json"
            provenance.write_text(json.dumps(provenance_entries), encoding="utf-8")
            comparison = directory / "comparison.csv"
            write_comparison(
                comparison,
                [
                    {
                        "case_id": "b",
                        "measured_composite_cycles": 20,
                        "microbench_run_ids": "micro-a",
                    },
                    {
                        "case_id": "a",
                        "measured_composite_cycles": 10,
                        "microbench_run_ids": "micro-a",
                    },
                ],
            )
            result_dir = directory / "path/result"
            options = [
                "--cycles-json",
                str(cycles),
                "--provenance-json",
                str(provenance),
                "--comparison-csv",
                str(comparison),
            ]
            program = (
                "import json; "
                "print(json.dumps({'case_id':'a','model_kind':'m','T_model':11})); "
                "print(json.dumps({'case_id':'b','model_kind':'m','T_model':21}))"
            )
            completed = self.run_command(
                directory,
                result_dir,
                "multi",
                program,
                kind="model",
                extra_options=options,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with (result_dir / "runs/multi/comparison.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["case_id"] for row in rows], ["a", "b"])
            self.assertEqual(
                [float(row["measured_composite_cycles"]) for row in rows],
                [10.0, 20.0],
            )

            missing_case_program = (
                "import json; "
                "print(json.dumps({'case_id':'a','model_kind':'m','T_model':11})); "
                "print(json.dumps({'model_kind':'m','T_model':21}))"
            )
            rejected = self.run_command(
                directory,
                result_dir,
                "missing-case",
                missing_case_program,
                kind="model",
                extra_options=options,
            )
            self.assertEqual(rejected.returncode, 2)
            metadata = self.load_metadata(result_dir, "missing-case")
            self.assertEqual(metadata["status"], "parse_error")
            self.assertIn("requires case_id", metadata["parse"]["error"])


if __name__ == "__main__":
    unittest.main()
