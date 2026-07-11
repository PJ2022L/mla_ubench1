"""CPU-only tests for path composition and e2e argument validation."""

from __future__ import annotations

import builtins
import contextlib
import importlib.util
import io
import json
import math
import tempfile
import unittest
from pathlib import Path


PATHS_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str, forbid_gpu_imports: bool = False):
    path = PATHS_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    original_import = builtins.__import__

    def guarded_import(import_name, *args, **kwargs):
        if forbid_gpu_imports and import_name.split(".", 1)[0] in {
            "flash_mla",
            "kernelkit",
            "lib",
            "ref",
            "torch",
        }:
            raise AssertionError(f"GPU dependency imported during module load: {import_name}")
        return original_import(import_name, *args, **kwargs)

    builtins.__import__ = guarded_import
    try:
        spec.loader.exec_module(module)
    finally:
        builtins.__import__ = original_import
    return module


DENSE_COMPOSE = load_module(
    "dense_decode_compose", "dense_decode_bf16_sm90_mqa/compose.py"
)
SPARSE_COMPOSE = load_module(
    "sparse_decode_compose",
    "sparse_decode_fp8_sm90_v32_mqa_h128_cluster2/compose.py",
)
DENSE_E2E = load_module(
    "dense_decode_e2e",
    "dense_decode_bf16_sm90_mqa/e2e/benchmark.py",
    forbid_gpu_imports=True,
)
SPARSE_DECODE_E2E = load_module(
    "sparse_decode_e2e",
    "sparse_decode_fp8_sm90_v32_mqa_h128_cluster2/e2e/benchmark.py",
    forbid_gpu_imports=True,
)
SPARSE_PREFILL_E2E = load_module(
    "sparse_prefill_e2e",
    "sparse_prefill_bf16_sm90_mqa/e2e/benchmark.py",
    forbid_gpu_imports=True,
)


def run_validate(module, argv: list[str]) -> dict[str, object]:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        module.main(["--validate-only", *argv])
    return json.loads(stdout.getvalue())


class DenseComposeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schedule = {
            "T_prologue_single": 10,
            "T_prologue_pair": 20,
            "T_pair_transition": 5,
            "T_pair_to_single": 7,
            "T_pair_drain": 11,
            "T_single_drain": 13,
            "T_output_store": 3,
            "T_reduce_l": 2,
        }

    def test_measured_schedule_single_even_and_odd(self) -> None:
        expected = {
            1: ("single", 23, 28),
            2: ("pair", 31, 36),
            3: ("pair_to_single", 40, 45),
            4: ("pair", 36, 41),
        }
        for n_page, (tail, body, model) in expected.items():
            with self.subTest(n_page=n_page):
                result = DENSE_COMPOSE.compose(self.schedule, n_page)
                self.assertEqual(result["tail_kind"], tail)
                self.assertEqual(result["T_body"], body)
                self.assertEqual(result["T_model"], model)

    def test_atom_fallback_and_input_validation(self) -> None:
        atoms = {
            "t_qk_ss": 1,
            "t_qk_rs": 2,
            "t_pv_rs": 3,
            "t_pv_ss": 4,
            "T_tma_k_tile": 5,
            "T_softmax": 6,
            "T_stmatrix_p": 7,
            "T_qload": 8,
            "T_output_store": 9,
        }
        result = DENSE_COMPOSE.compose(atoms, 2)
        self.assertEqual(result["model_kind"], "atom_fallback")
        with self.assertRaisesRegex(ValueError, "n_page"):
            DENSE_COMPOSE.compose(atoms, 0)
        atoms["t_qk_ss"] = math.nan
        with self.assertRaisesRegex(ValueError, "finite non-negative"):
            DENSE_COMPOSE.compose(atoms, 1)

    def test_split_epilogue_and_missing_schedule_cost(self) -> None:
        costs = dict(self.schedule)
        costs.update(T_output_store_split=4, T_combine=17)
        result = DENSE_COMPOSE.compose(costs, 1, split_kv=True)
        self.assertEqual(result["T_output_store"], 4)
        self.assertEqual(result["T_main_model"], 29)
        self.assertEqual(result["T_e2e_additive"], 46)
        self.assertTrue(result["combine_included"])

        del costs["T_single_drain"]
        with self.assertRaisesRegex(KeyError, "T_single_drain"):
            DENSE_COMPOSE.compose(costs, 1, split_kv=True)

    def test_cli_page_resolution_and_split_restriction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cycles.json"
            path.write_text(json.dumps(self.schedule), encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                DENSE_COMPOSE.main(
                    ["--cycles-json", str(path), "--seqlen-k", "65"]
                )
            self.assertEqual(json.loads(stdout.getvalue())["N_page"], 2)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                DENSE_COMPOSE.main(
                    ["--cycles-json", str(path), "--n-page", "3"]
                )
            self.assertEqual(json.loads(stdout.getvalue())["N_page"], 3)
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    DENSE_COMPOSE.main(
                        [
                            "--cycles-json",
                            str(path),
                            "--seqlen-k",
                            "65",
                            "--split-kv",
                        ]
                    )


class SparseComposeTests(unittest.TestCase):
    def base_cycles(self) -> dict[str, float]:
        return {
            "T_producer_direct": 100,
            "T_wgmma_qk_ss": 2,
            "T_wgmma_pv_rs": 3,
            "T_wgmma_pv_ss": 4,
            "T_softmax": 5,
            "T_output_store": 6,
        }

    def test_direct_and_decomposed_producer(self) -> None:
        direct = self.base_cycles()
        decomposed = dict(direct)
        del decomposed["T_producer_direct"]
        decomposed.update(
            T_ld_block=40,
            T_cvt_block=30,
            T_st_shared_block=20,
            T_st_dsm_block=10,
        )
        self.assertEqual(SPARSE_COMPOSE.producer_cycles(direct), 100)
        self.assertEqual(SPARSE_COMPOSE.producer_cycles(decomposed), 100)
        self.assertEqual(
            SPARSE_COMPOSE.compose(direct, 2)["T_model"],
            SPARSE_COMPOSE.compose(decomposed, 2)["T_model"],
        )

    def test_split_epilogue_and_measured_bounds(self) -> None:
        cycles = self.base_cycles()
        cycles.update(T_partial_store=7, T_measured=350)
        result = SPARSE_COMPOSE.compose(cycles, 2, split_kv=True)
        self.assertEqual(result["epilogue_kind"], "split_fp32_partial")
        self.assertTrue(result["within_bounds"])
        self.assertAlmostEqual(result["rho"], result["T_model"] / 350)

    def test_rejects_invalid_costs_and_zero_measurement(self) -> None:
        cycles = self.base_cycles()
        cycles["T_softmax"] = -1
        with self.assertRaisesRegex(ValueError, "finite non-negative"):
            SPARSE_COMPOSE.compose(cycles, 1)
        cycles = self.base_cycles()
        cycles["T_measured"] = 0
        with self.assertRaisesRegex(ValueError, "must be positive"):
            SPARSE_COMPOSE.compose(cycles, 1)

    def test_legacy_store_and_required_split_store(self) -> None:
        cycles = self.base_cycles()
        del cycles["T_output_store"]
        cycles["T_tma_store"] = 8
        result = SPARSE_COMPOSE.compose(cycles, 1)
        self.assertEqual(result["T_epilogue"], 8)
        with self.assertRaisesRegex(KeyError, "T_partial_store"):
            SPARSE_COMPOSE.compose(cycles, 1, split_kv=True)
        with self.assertRaisesRegex(ValueError, "n_block"):
            SPARSE_COMPOSE.compose(cycles, 0)

    def test_cli_rejects_non_object_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cycles.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must contain an object"):
                SPARSE_COMPOSE.main(["--cycles-json", str(path)])


class E2EValidationTests(unittest.TestCase):
    def test_dense_resolves_tail_pages_and_causal(self) -> None:
        result = run_validate(DENSE_E2E, ["--s-k", "4097", "--causal"])
        self.assertEqual(result["pages_per_request"], 65)
        self.assertFalse(result["causal_effective"])

    def test_dense_rejects_wrong_fixed_shape(self) -> None:
        args = DENSE_E2E.build_parser().parse_args(["--d-v", "256"])
        with self.assertRaisesRegex(ValueError, "fixes h_kv=1"):
            DENSE_E2E.validate_args(args)

    def test_sparse_decode_fixed_shape_and_device(self) -> None:
        result = run_validate(SPARSE_DECODE_E2E, ["--s-k", "16", "--topk", "32"])
        self.assertEqual((result["h_q"], result["d_qk"], result["d_v"]), (128, 576, 512))
        self.assertEqual((result["s_k"], result["topk"]), (16, 32))
        args = SPARSE_DECODE_E2E.build_parser().parse_args(["--device", "-1"])
        with self.assertRaisesRegex(ValueError, "device"):
            SPARSE_DECODE_E2E.validate_args(args)

    def test_sparse_prefill_allows_oob_topk_indices(self) -> None:
        result = run_validate(
            SPARSE_PREFILL_E2E, ["--s-q", "2", "--s-kv", "16", "--topk", "32"]
        )
        self.assertEqual((result["s_kv"], result["topk"]), (16, 32))


if __name__ == "__main__":
    unittest.main()
