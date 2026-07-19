from __future__ import annotations

import unittest

from microbench.model.dense_decode.scheduler import (
    resolve_num_sm_parts,
    schedule_requests,
)
from microbench.model.dense_decode.schema import load_workload
from microbench.model.dense_decode.profile import ProfileLookup
from microbench.model.dense_decode.scheduler import RequestSlice
from microbench.model.dense_decode.simulator import (
    AtomCosts,
    _calibrated_epilogue_phase,
    _softmax_phase,
    predict,
)


def metric(value, unit):
    return {"value": value, "unit": unit}


def record(name, *, latency=1.0, bandwidth=None, **params):
    return {
        "name": f"dense_decode.{name}",
        "params": params,
        "latency": metric(latency, "cycles/instruction"),
        "throughput": metric(1.0, "Ginst/s"),
        "memory_bandwidth": metric(bandwidth, "GB/s") if bandwidth else metric(None, "GB/s"),
        "hardware_utilization": metric(None, "ratio"),
        "provenance": {"source_file": "synthetic.jsonl", "record_index": len(params)},
    }


def synthetic_profile():
    latency_atoms = (
        "exp2_f32",
        "shfl_bfly_b32",
        "ffma_f32",
        "fsel_f32",
        "wgmma_qk_ss_bf16",
        "wgmma_qk_rs_bf16",
        "wgmma_pv_ss_bf16",
        "wgmma_pv_rs_bf16",
        "stmatrix_p_b16",
        "stmatrix_o_b16",
        "barrier_sync",
        "fadd_f32",
        "iadd3_u32",
        "shared_store_f32x2_stride520",
        "proxy_fence_async_shared",
    )
    records = [record(name, latency=2.0) for name in latency_atoms]
    records.extend(
        [
            record("tma_load_k_bf16_rank4", bandwidth=1500.0),
            record("tma_load_q_bf16_rank4", bandwidth=1500.0),
            record("tma_store_o_bf16_rank4", bandwidth=1200.0),
            record("bulk_store_oaccum_f32", bandwidth=1000.0),
        ]
    )
    return {
        "schema_version": 1,
        "profile_id": "synthetic",
        "target": {
            "sm_count": 2,
            "sm_clock_mhz": 1800.0,
            "l2_bytes": 50 * 1024 * 1024,
            "hbm_gbps": 3000.0,
        },
        "records": records,
    }


class SchedulerTests(unittest.TestCase):
    def test_num_sm_parts_matches_public_formula(self):
        self.assertEqual(resolve_num_sm_parts(132, 1, 128, 1), 66)
        self.assertEqual(resolve_num_sm_parts(132, 2, 128, 1), 33)

    def test_scheduler_preserves_pages_and_split_prefix(self):
        result = schedule_requests(
            [64, 129, 4096],
            sm_count=8,
            seqlen_q=1,
            num_heads_q=64,
            num_heads_kv=1,
            num_sm_parts=4,
        )
        pages = [0, 0, 0]
        for item in result.slices:
            pages[item.request_idx] += item.pages
        self.assertEqual(pages, [1, 3, 64])
        self.assertEqual(len(result.num_splits_prefix), 4)
        self.assertTrue(all(a <= b for a, b in zip(
            result.num_splits_prefix, result.num_splits_prefix[1:]
        )))

    def test_zero_length_request_has_no_main_pages(self):
        result = schedule_requests(
            [0, 64],
            sm_count=2,
            seqlen_q=1,
            num_heads_q=64,
            num_heads_kv=1,
            num_sm_parts=1,
        )
        by_request = {item.request_idx: item.pages for item in result.slices}
        self.assertEqual(by_request[0], 0)
        self.assertEqual(by_request[1], 1)

    def test_flags_target_metadata_undefined_empty_partitions(self):
        result = schedule_requests(
            [64],
            sm_count=132,
            seqlen_q=1,
            num_heads_q=128,
            num_heads_kv=1,
            num_sm_parts=66,
        )
        self.assertFalse(result.source_defined)
        self.assertIn("target get_mla_metadata_kernel", result.undefined_reason)


class ModelTests(unittest.TestCase):
    def test_stage_calibrations_take_precedence_over_atom_fallbacks(self):
        profile = synthetic_profile()
        profile["records"].extend(
            [
                record("calibration.softmax_stage_bf16", latency=101.0),
                record("calibration.epilogue_nosplit_b16", latency=202.0),
                record("calibration.epilogue_split_f32", latency=303.0),
            ]
        )
        workload = load_workload(
            {
                "batch_size": 1,
                "seqlens_k": [64],
                "metadata_mode": "reuse",
            }
        )
        costs = AtomCosts(ProfileLookup(profile), workload)

        softmax = _softmax_phase(costs, "unit")
        self.assertEqual(softmax.name, "softmax_unit_calibrated")
        self.assertEqual(softmax.tasks[0].isolated_cycles, 101.0)

        no_split = _calibrated_epilogue_phase(
            costs,
            RequestSlice(0, 0, 0, 1, 0, True),
        )
        split = _calibrated_epilogue_phase(
            costs,
            RequestSlice(0, 0, 0, 1, 0, False),
        )
        self.assertIsNotNone(no_split)
        self.assertIsNotNone(split)
        self.assertAlmostEqual(no_split.tasks[0].isolated_cycles, 202.0)
        self.assertAlmostEqual(split.tasks[0].isolated_cycles, 303.0)

    def test_stage_calibrations_remain_optional(self):
        profile = synthetic_profile()
        workload = load_workload(
            {
                "batch_size": 1,
                "seqlens_k": [64],
                "metadata_mode": "reuse",
            }
        )
        costs = AtomCosts(ProfileLookup(profile), workload)
        softmax = _softmax_phase(costs, "fallback")
        epilogue = _calibrated_epilogue_phase(
            costs,
            RequestSlice(0, 0, 0, 1, 0, True),
        )
        self.assertEqual(softmax.name, "softmax_fallback")
        self.assertIsNone(epilogue)

    def test_predicts_full_api_breakdown_without_gpu(self):
        workload = load_workload(
            {
                "case_id": "unit",
                "dtype": "bf16",
                "batch_size": 1,
                "seqlen_q": 1,
                "num_heads_q": 64,
                "num_heads_kv": 1,
                "seqlens_k": [128],
                "metadata_mode": "reuse",
            }
        )
        result = predict(synthetic_profile(), workload).result
        self.assertGreater(result["predicted_e2e_us"]["p50"], 0)
        self.assertGreater(result["breakdown_us"]["main"], 0)
        self.assertEqual(result["breakdown_us"]["metadata"], 0)
        self.assertEqual(result["scheduler"]["slices"][0]["pages"], 2)
        self.assertEqual(result["model_kind"], "discrete_event_resource")
        self.assertTrue(result["incomplete_calibration"])
        self.assertGreater(result["cta"]["combine"]["jobs"], 0)
        self.assertIn("parameter_sensitivity", result)

    def test_exact_block_table_tracks_physical_page_reuse(self):
        workload = load_workload(
            {
                "batch_size": 2,
                "seqlens_k": [128, 128],
                "block_table": [[7, 8], [7, 8]],
                "block_table_pattern": "reuse",
            }
        )
        self.assertEqual(workload.logical_k_pages, 4)
        self.assertEqual(workload.unique_k_pages, 2)
        self.assertEqual(workload.page_reuse_ratio, 0.5)
        self.assertEqual(workload.block_table_source, "explicit")

    def test_pdl_microseconds_are_used_as_overlap_credit(self):
        profile = synthetic_profile()
        profile["target"]["sm_count"] = 1
        profile["records"].append(
            {
                "name": "dense_decode.calibration.pdl_overlap",
                "params": {"producer_blocks": 2, "consumer_blocks": 2},
                "latency": metric(0.25, "us"),
                "throughput": metric(1.0, "Gpair/s"),
                "memory_bandwidth": metric(None, "GB/s"),
                "hardware_utilization": metric(0.5, "ratio"),
                "provenance": {"source_file": "pdl.jsonl", "record_index": 0},
            }
        )
        workload = load_workload(
            {
                "batch_size": 1,
                "seqlen_q": 1,
                "num_heads_q": 64,
                "num_heads_kv": 1,
                "seqlens_k": [128],
            }
        )
        result = predict(profile, workload).result
        self.assertFalse(result["incomplete_calibration"])
        self.assertGreater(result["breakdown_us"]["pdl_overlap_credit"], 0)

    def test_bootstrap_uses_raw_samples_without_recursive_simulation(self):
        profile = synthetic_profile()
        exp2 = next(
            item for item in profile["records"]
            if item["name"] == "dense_decode.exp2_f32"
        )
        exp2["raw_samples"] = {"latency": [1.5, 2.0, 2.5]}
        workload = load_workload(
            {
                "batch_size": 1,
                "seqlen_q": 1,
                "num_heads_q": 64,
                "num_heads_kv": 1,
                "seqlens_k": [128],
            }
        )
        result = predict(profile, workload, bootstrap=100).result
        self.assertLessEqual(
            result["predicted_e2e_us"]["p10"],
            result["predicted_e2e_us"]["p90"],
        )

    def test_rejects_wrong_dense_shape(self):
        with self.assertRaisesRegex(ValueError, "fixes head_dim"):
            load_workload(
                {
                    "batch_size": 1,
                    "seqlens_k": [64],
                    "head_dim_qk": 512,
                }
            )


if __name__ == "__main__":
    unittest.main()
