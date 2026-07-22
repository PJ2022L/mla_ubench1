from __future__ import annotations

import csv
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from operators.flash_mla.paths.dense_decode_bf16_sm90_mqa.model import (
    AtomMap, CostDatabase, CoverageError, DenseDecodeDAG, KernelResources, OperationNode,
    build_dense_decode_dag,
    load_kernel_resources, load_workload, simulate,
)
from operators.flash_mla.paths.dense_decode_bf16_sm90_mqa.model.cli import build_parser
from operators.flash_mla.paths.dense_decode_bf16_sm90_mqa.model.cost_database import CostDatabaseError
from operators.flash_mla.paths.dense_decode_bf16_sm90_mqa.model.calibration import (
    _probe_dag, validate_calibration,
)
from operators.flash_mla.paths.dense_decode_bf16_sm90_mqa.model.resources import ResourceModel
import operators.flash_mla.paths.dense_decode_bf16_sm90_mqa.model.simulator as simulator_module


MODEL_ROOT = Path(__file__).resolve().parents[1]
ATOM_MAP = AtomMap.load(MODEL_ROOT / "atom_map.json")


def workload(pages: int, **overrides):
    value = {
        "batch_size": 1, "dtype": "bf16", "seqlens_k": [pages * 64],
        "num_heads_q": 8, "num_heads_kv": 1,
        "block_table": [list(range(pages))],
    }
    value.update(overrides)
    return load_workload(value)


def database(root: Path, atom_ids: set[str]) -> CostDatabase:
    entries = [
        {"id": atom_id, "kind": "operation", "family": "synthetic",
         "source": f"compute/synthetic/{atom_id}.cu", "result_csv": "result.csv"}
        for atom_id in sorted(atom_ids)
    ]
    (root / "manifest.json").write_text(json.dumps({"benchmarks": entries}))
    fields = [
        "name", "latency_value", "latency_unit", "initiation_interval_cycles",
        "throughput_value", "throughput_unit", "memory_bandwidth_value",
        "memory_bandwidth_unit", "hardware_utilization", "p10", "p50", "p90",
        "sample_count", "gpu_uuid", "source_sha256", "sass_sha256",
    ]
    with (root / "result.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for atom_id in sorted(atom_ids):
            writer.writerow({
                "name": atom_id, "latency_value": 2, "latency_unit": "cycles",
                "initiation_interval_cycles": 1, "throughput_value": 128,
                "throughput_unit": "op/cycle", "memory_bandwidth_value": 1000,
                "memory_bandwidth_unit": "GB/s", "p10": 1.8, "p50": 2,
                "p90": 2.2, "sample_count": 10,
            })
    return CostDatabase(root)


class DagTests(unittest.TestCase):
    def test_all_dense_atoms_supply_manifest_query_parameters(self):
        manifest = json.loads(
            (MODEL_ROOT.parents[4] / "microbench" / "manifest.json").read_text()
        )
        required = {
            entry["id"]: set(entry.get("parameters", []))
            for entry in manifest["benchmarks"] if entry.get("kind") == "operation"
        }
        cases = [
            workload(0), workload(1, dtype="fp16"), workload(2), workload(3),
            workload(4), workload(5), workload(20),
        ]
        for value in cases:
            dag = build_dense_decode_dag(value, KernelResources(sm_count=2), ATOM_MAP)
            for node in dag.nodes.values():
                self.assertIn(
                    node.atom_id, required,
                    (node.atom_id, node.source_anchor),
                )
                supplied = set(node.benchmark_params) | {
                    "blocks", "active_sm", "resident_cta",
                }
                self.assertFalse(
                    required.get(node.atom_id, set()) - supplied,
                    (node.atom_id, node.source_anchor, node.benchmark_params),
                )

    def test_page_boundaries_and_dtypes(self):
        expected_update_phases = {
            0: set(),
            1: {"tail_update.single_drain"},
            2: {"tail_update.pair_drain"},
            3: {"tail_update.pair_to_single", "tail_update.single_drain"},
            4: {"pair_update[0]", "tail_update.pair_drain"},
            5: {"pair_update[0]", "tail_update.pair_to_single",
                "tail_update.single_drain"},
        }
        for dtype in ("bf16", "fp16"):
            for pages in range(6):
                dag = build_dense_decode_dag(
                    workload(pages, dtype=dtype), KernelResources(sm_count=1), ATOM_MAP
                )
                self.assertEqual(len(dag.topological_nodes()), len(dag.nodes))
                qk = [node for node in dag.nodes.values()
                      if node.atom_id.startswith("m64n64k16")]
                self.assertEqual(len(qk), 1 + 9 * max(pages - 1, 0))
                first = [node for node in qk if node.phase == "first_score"]
                self.assertEqual(len(first), 1)
                self.assertEqual(first[0].benchmark_params["group_size"], 36)
                self.assertEqual(first[0].benchmark_params["depth"], 1)
                if dtype == "fp16":
                    self.assertTrue(any(node.atom_id == "tensor_4d_64x64_fp16"
                                        for node in dag.nodes.values()))
                update_phases = {
                    node.phase for node in dag.nodes.values()
                    if node.phase.startswith(("pair_update", "tail_update"))
                }
                self.assertEqual(update_phases, expected_update_phases[pages])

    def test_protocol_edges(self):
        dag = build_dense_decode_dag(workload(4), KernelResources(sm_count=1), ATOM_MAP)
        first = next(node for node in dag.nodes.values()
                     if node.phase == "first_score" and node.atom_id == "m64n64k16_ss_bf16")
        first_ready = [edge for edge in dag.dependencies if edge.dst == first.id
                       and edge.kind == "tma_ready" and "all first-page" in (edge.label or "")]
        self.assertEqual(len(first_ready), 9)
        labels = {edge.label for edge in dag.dependencies if edge.kind == "barrier"}
        self.assertTrue({"sMInitialized", "sScale0Ready", "sScale1Ready", "sP0Ready",
                         "rO1sP0sV0RIssued"} <= labels)
        self.assertTrue(any(edge.kind == "buffer_reuse" for edge in dag.dependencies))
        async_program = [edge for edge in dag.dependencies if edge.kind == "program"
                         and dag.nodes[edge.src].async_issue]
        self.assertTrue(async_program)
        self.assertTrue(all(edge.src_event == "issue" for edge in async_program))

        k1_tiles = [
            node.benchmark_params["tile"]
            for node in dag.nodes.values()
            if node.phase == "steady_score[0]"
            and node.atom_id == "tensor_4d_64x64_bf16"
        ]
        self.assertEqual(k1_tiles, [4, 5, 6, 7, 8, 0, 1, 2, 3])

        sm_barrier = next(
            node for node in dag.nodes.values()
            if node.benchmark_params.get("named_barrier") == "sMInitialized"
        )
        score_issue = [
            edge for edge in dag.dependencies
            if edge.dst == sm_barrier.id and edge.src_event == "issue"
            and dag.nodes[edge.src].atom_id.startswith("m64n64k16")
        ]
        self.assertEqual(len(score_issue), 1)
        self.assertTrue(any(
            edge.src == sm_barrier.id and edge.label == "sMInitialized"
            for edge in dag.dependencies
        ))

        four_pages_waits = [
            edge for edge in dag.dependencies if edge.label == "wait_group<4> release"
        ]
        self.assertEqual(len(four_pages_waits), 5)
        self.assertTrue(all(edge.src_event == "complete" for edge in four_pages_waits))
        three_pages = build_dense_decode_dag(
            workload(3), KernelResources(sm_count=1), ATOM_MAP
        )
        three_page_waits = [
            edge for edge in three_pages.dependencies
            if edge.label == "wait_group<4> release"
        ]
        self.assertEqual(len(three_page_waits), 5)
        self.assertTrue(all(edge.src_event == "issue" for edge in three_page_waits))

    def test_two_warpgroup_named_barriers_are_single_split_event_protocols(self):
        dag = build_dense_decode_dag(workload(4), KernelResources(sm_count=1), ATOM_MAP)
        labels = {
            "sScale0Ready", "sScale1Ready", "sP0Ready",
            "rO1sP0sV0RIssued",
        }
        protocols = [
            node for node in dag.nodes.values()
            if node.benchmark_params.get("named_barrier") in labels
        ]
        self.assertEqual(len(protocols), 8)
        self.assertTrue(all(node.atom_id == "bar_arrive_2wg" for node in protocols))
        self.assertTrue(all(node.async_issue for node in protocols))
        self.assertFalse(any(
            node.atom_id == "bar_sync_256"
            and node.benchmark_params.get("named_barrier") in labels
            for node in dag.nodes.values()
        ))
        for protocol in protocols:
            incoming = [
                edge for edge in dag.dependencies if edge.dst == protocol.id
            ]
            self.assertEqual(len(incoming), 2, protocol.benchmark_params)
            self.assertEqual({edge.dst_event for edge in incoming}, {"issue", "complete"})
            self.assertEqual(
                {edge.label for edge in incoming},
                {protocol.benchmark_params["named_barrier"]},
            )

        for scale0 in (
            node for node in protocols
            if node.benchmark_params["named_barrier"] == "sScale0Ready"
        ):
            continuations = [
                edge for edge in dag.dependencies
                if edge.src == scale0.id and edge.src_event == "issue"
                and dag.nodes[edge.dst].source_anchor.endswith(
                    "warpgroup_cooperative_pv_gemm_localP"
                )
                and dag.nodes[edge.dst].actor.endswith(".wg0")
            ]
            wait_releases = [
                edge for edge in dag.dependencies
                if edge.src == scale0.id and edge.src_event == "complete"
                and dag.nodes[edge.dst].actor.endswith(".wg1")
            ]
            self.assertEqual(len(continuations), 1)
            self.assertTrue(wait_releases)

    def test_page3_k1_low_tma_waits_for_page2_phase0_issue_and_wait_group(self):
        dag = build_dense_decode_dag(workload(4), KernelResources(sm_count=1), ATOM_MAP)
        low_k1 = [
            node for node in dag.nodes.values()
            if node.phase == "steady_score[2]"
            and node.atom_id == "tensor_4d_64x64_bf16"
            and node.benchmark_params["tile"] < 4
        ]
        self.assertEqual(len(low_k1), 4)
        for load in low_k1:
            incoming = [edge for edge in dag.dependencies if edge.dst == load.id]
            phase0 = [
                edge for edge in incoming
                if edge.label == (
                    "page+2 phase-0 score issue before page+3 K1 low TMA"
                )
            ]
            wait_group = [
                edge for edge in incoming
                if edge.label == "wait_group<4> release before K1 low TMA"
            ]
            self.assertEqual(len(phase0), 1)
            self.assertEqual(len(wait_group), 1)
            self.assertEqual(phase0[0].src_event, "issue")
            self.assertEqual(wait_group[0].src_event, "complete")
            score = dag.nodes[phase0[0].src]
            self.assertEqual(score.phase, "steady_score[1]")
            self.assertEqual(score.benchmark_params["tile"], 3)

    def test_odd_stsm_local_pv_and_proxy_fence_source_order(self):
        normal = build_dense_decode_dag(
            workload(4), KernelResources(sm_count=1), ATOM_MAP
        )
        normal_nodes = [
            node for node in normal.nodes.values()
            if node.phase == "pair_update[0]" and node.actor.endswith(".wg1")
        ]
        normal_stsm = next(
            node for node in normal_nodes
            if node.source_anchor.endswith("save_rPb_to_sP WG1")
        )
        normal_local = next(
            node for node in normal_nodes
            if node.source_anchor.endswith("warpgroup_cooperative_pv_gemm_localP")
        )
        normal_fence = next(
            node for node in normal_nodes
            if node.source_anchor.endswith("sP1 post-local-P proxy fence")
        )
        self.assertTrue(any(
            edge.src == normal_stsm.id and edge.dst == normal_local.id
            for edge in normal.dependencies
        ))
        self.assertTrue(any(
            edge.src == normal_local.id and edge.dst == normal_fence.id
            and edge.src_event == "issue"
            for edge in normal.dependencies
        ))
        self.assertFalse(any(
            node.source_anchor.endswith("fill_oob_V async proxy fence")
            for node in normal_nodes
        ))

        tail_workload = load_workload({
            "batch_size": 1, "dtype": "bf16", "seqlens_k": [255],
            "num_heads_q": 8, "num_heads_kv": 1,
            "block_table": [[0, 1, 2, 3]],
        })
        tail = build_dense_decode_dag(
            tail_workload, KernelResources(sm_count=1), ATOM_MAP
        )
        tail_nodes = [
            node for node in tail.nodes.values()
            if node.phase == "tail_update.pair_drain"
            and node.actor.endswith(".wg1")
        ]
        tail_stsm = next(
            node for node in tail_nodes
            if node.source_anchor.endswith("save_rPb_to_sP WG1")
        )
        tail_fill = next(
            node for node in tail_nodes
            if node.source_anchor.endswith("fill_oob_V")
            and node.benchmark_params.get("half") == "right"
        )
        tail_fence = next(
            node for node in tail_nodes
            if node.source_anchor.endswith("fill_oob_V async proxy fence")
            and node.benchmark_params.get("half") == "right"
        )
        tail_local = next(
            node for node in tail_nodes
            if node.source_anchor.endswith("warpgroup_cooperative_pv_gemm_localP")
        )
        self.assertTrue(any(
            edge.src == tail_stsm.id and edge.dst == tail_fill.id
            for edge in tail.dependencies
        ))
        self.assertTrue(any(
            edge.src == tail_fill.id and edge.dst == tail_fence.id
            for edge in tail.dependencies
        ))
        self.assertTrue(any(
            edge.src == tail_fence.id and edge.dst == tail_local.id
            for edge in tail.dependencies
        ))
        self.assertFalse(any(
            node.source_anchor.endswith("sP1 post-local-P proxy fence")
            for node in tail_nodes
        ))

    def test_tail_barrier_precedes_mask_and_local_pv(self):
        value = load_workload({
            "batch_size": 1, "dtype": "bf16", "seqlens_k": [65],
            "num_heads_q": 8, "num_heads_kv": 1,
            "block_table": [[0, 1]],
        })
        dag = build_dense_decode_dag(value, KernelResources(sm_count=1), ATOM_MAP)
        odd_mask = next(
            node for node in dag.nodes.values()
            if node.actor.endswith(".wg1")
            and node.source_anchor.endswith("tail token predicate")
        )
        self.assertTrue(any(
            edge.dst == odd_mask.id and edge.label == "sScale0Ready"
            for edge in dag.dependencies
        ))
        local_pv = next(
            node for node in dag.nodes.values()
            if node.actor.endswith(".wg0")
            and node.source_anchor.endswith("warpgroup_cooperative_pv_gemm_localP")
        )
        self.assertTrue(any(
            edge.dst == local_pv.id and edge.label == "sScale0Ready WG0 arrive"
            for edge in dag.dependencies
        ))

        empty = build_dense_decode_dag(
            workload(0), KernelResources(sm_count=1), ATOM_MAP
        )
        self.assertTrue(any(
            node.benchmark_params.get("named_barrier") == "sMInitialized"
            for node in empty.nodes.values()
        ))

    def test_metadata_causal_split_and_noop(self):
        generated = build_dense_decode_dag(
            workload(1), KernelResources(sm_count=1), ATOM_MAP
        )
        self.assertNotIn("rem", generated.scheduler.operation_counts)
        self.assertFalse(any(
            node.atom_id == "rem_u32" for node in generated.nodes.values()
        ))
        reused = build_dense_decode_dag(
            workload(1, metadata_mode="reuse"), KernelResources(sm_count=1), ATOM_MAP
        )
        self.assertFalse(any(node.phase == "metadata" for node in reused.nodes.values()))
        causal = build_dense_decode_dag(
            workload(3, causal=True, seqlen_q=128, num_heads_q=1),
            KernelResources(sm_count=1), ATOM_MAP,
        )
        pages = {node.cta_id: node.benchmark_params["pages"] for node in causal.nodes.values()
                 if node.phase == "request_setup"
                 and node.source_anchor.endswith("seqlens_k request load")}
        self.assertEqual(pages["main.p0.h0.m0"], 2)
        self.assertEqual(pages["main.p0.h0.m1"], 3)
        noop = build_dense_decode_dag(
            workload(1, seqlen_q=2, num_heads_q=17), KernelResources(sm_count=1), ATOM_MAP
        )
        combine = {node.cta_id for node in noop.nodes.values()
                   if node.cta_id and node.cta_id.startswith("combine.")}
        self.assertEqual(len(combine), 6)
        self.assertFalse(any(node.atom_id == "griddepcontrol_wait" for node in noop.nodes.values()))
        split = build_dense_decode_dag(workload(20), KernelResources(sm_count=2), ATOM_MAP)
        self.assertEqual(split.scheduler.num_splits_prefix[-1], 2)
        first_loads = [node for node in split.nodes.values()
                       if node.source_anchor.endswith("first-split prefetch")]
        self.assertTrue(first_loads)
        for load in first_loads:
            incoming = [edge for edge in split.dependencies if edge.dst == load.id]
            self.assertTrue(any("partial stores" in (edge.label or "") for edge in incoming))

    def test_epilogue_pdl_and_combine_source_order(self):
        partial_rows = build_dense_decode_dag(
            workload(1, num_heads_q=10), KernelResources(sm_count=1), ATOM_MAP
        )
        output_stores = [
            node for node in partial_rows.nodes.values()
            if node.phase == "epilogue_nosplit"
            and node.atom_id == "tensor_4d_64x512_bf16"
        ]
        self.assertEqual([node.benchmark_params["bytes"] for node in output_stores],
                         [10 * 512 * 2])
        self.assertFalse(any(
            node.source_anchor.endswith("inter-request __syncthreads")
            for node in partial_rows.nodes.values()
        ))

        two_requests = load_workload({
            "batch_size": 2, "dtype": "bf16", "seqlens_k": [64, 64],
            "num_heads_q": 8, "num_heads_kv": 1,
            "block_table": [[0], [1]],
        })
        persistent = build_dense_decode_dag(
            two_requests, KernelResources(sm_count=1), ATOM_MAP
        )
        self.assertEqual(sum(
            node.source_anchor.endswith("inter-request __syncthreads")
            for node in persistent.nodes.values()
        ), 1)

        idle = build_dense_decode_dag(
            workload(1), KernelResources(sm_count=8), ATOM_MAP
        )
        active_main_ctas = {
            node.cta_id for node in idle.nodes.values()
            if node.phase == "request_setup" and node.cta_id
        }
        pdl_ctas = {
            node.cta_id for node in idle.nodes.values()
            if node.atom_id == "griddepcontrol_launch_dependents"
        }
        self.assertEqual(pdl_ctas, active_main_ctas)

        split = build_dense_decode_dag(
            workload(20), KernelResources(sm_count=2), ATOM_MAP
        )
        self.assertFalse(any(
            node.phase == "epilogue_split"
            and node.atom_id in {"cvt_rn_bf16_f32", "cvt_rn_f16_f32"}
            for node in split.nodes.values()
        ))
        combine_cta = next(
            node.cta_id for node in split.nodes.values()
            if node.cta_id and node.cta_id.startswith("combine.")
            and node.atom_id == "griddepcontrol_wait"
        )
        combine_nodes = [
            node for node in split.nodes.values() if node.cta_id == combine_cta
        ]
        ffma_chunks = [
            node for node in combine_nodes
            if node.source_anchor.endswith("weighted float4 accumulation chunk")
        ]
        next_load_chunks = [
            node for node in combine_nodes
            if node.source_anchor.endswith("interleaved next-split float4 load")
        ]
        self.assertEqual(len(ffma_chunks), 8)
        self.assertEqual(len(next_load_chunks), 4)
        lse_store = next(
            node for node in combine_nodes if node.source_anchor.endswith("LSE store")
        )
        scale_exp = next(
            node for node in combine_nodes
            if node.source_anchor.endswith("exp2f normalized scales")
        )
        self.assertTrue(any(
            edge.src == lse_store.id and edge.dst == scale_exp.id
            and edge.src_event == "issue"
            for edge in split.dependencies
        ))

    def test_partial_kv_tail_uses_mask_and_shared_fill_atoms(self):
        tail_workload = load_workload({
            "batch_size": 1, "dtype": "fp16", "seqlens_k": [65],
            "num_heads_q": 8, "num_heads_kv": 1,
            "block_table": [[0, 1]],
        })
        dag = build_dense_decode_dag(
            tail_workload, KernelResources(sm_count=1), ATOM_MAP
        )
        tail_stores = [
            node for node in dag.nodes.values()
            if node.atom_id == "st_shared_u64_sw128"
        ]
        self.assertEqual(len(tail_stores), 2)
        self.assertEqual(
            {node.benchmark_params["invalid_tokens"] for node in tail_stores},
            {63},
        )
        self.assertEqual(
            {node.benchmark_params["half"] for node in tail_stores},
            {"left", "right"},
        )
        self.assertTrue(any(
            node.source_anchor.endswith("masked score select")
            for node in dag.nodes.values()
        ))

    def test_scale_shared_traffic_and_probability_conversion(self):
        dag = build_dense_decode_dag(
            workload(2), KernelResources(sm_count=1), ATOM_MAP
        )
        scale1_loads = [
            node for node in dag.nodes.values()
            if node.benchmark_params.get("memory_object") == "sScale1"
            and node.atom_id == "ld_shared_u32_patterns"
        ]
        self.assertEqual(len(scale1_loads), 2)
        self.assertEqual({node.work_amount for node in scale1_loads}, {2 * 128})
        self.assertEqual(
            {node.benchmark_params["pattern"] for node in scale1_loads},
            {"quad_broadcast"},
        )
        rescale = next(
            node for node in dag.nodes.values()
            if node.source_anchor.endswith("wg0_scale_rP0")
        )
        convert = next(
            node for node in dag.nodes.values()
            if node.source_anchor.endswith("wg0_scale_rP0 conversion")
        )
        self.assertTrue(any(
            edge.src == rescale.id and edge.dst == convert.id
            for edge in dag.dependencies
        ))

    def test_partial_buffer_producer_consumer_identity(self):
        value = workload(20)
        dag = build_dense_decode_dag(
            value, KernelResources(sm_count=2), ATOM_MAP
        )
        split_count = 2
        output_working_set = (
            split_count * value.num_heads_kv * value.q_seq_per_hk * 512 * 4
        )
        lse_working_set = split_count * value.num_heads_kv * value.q_seq_per_hk * 4
        output_loads = [
            node for node in dag.nodes.values()
            if node.atom_id == "ld_global_v4_f32"
        ]
        self.assertEqual(
            {node.benchmark_params["chunk"] for node in output_loads},
            {0, 1, 2, 3},
        )
        self.assertTrue(all(
            node.benchmark_params["cache_mode"] == "producer_reuse"
            and node.benchmark_params["memory_object"] == "partial_output"
            and node.benchmark_params["request"] == 0
            and node.benchmark_params["working_set_bytes"] == output_working_set
            for node in output_loads
        ))
        lse_loads = [
            node for node in dag.nodes.values()
            if node.atom_id == "ld_global_f32_strided"
        ]
        self.assertTrue(all(
            node.benchmark_params["cache_mode"] == "producer_reuse"
            and node.benchmark_params["memory_object"] == "partial_lse"
            and node.benchmark_params["working_set_bytes"] == lse_working_set
            for node in lse_loads
        ))
        partial_stores = [
            node for node in dag.nodes.values()
            if node.benchmark_params.get("producer_consumer")
            and node.resource_class == "hbm"
            and node.benchmark_params.get("memory_object") in {
                "partial_output", "partial_lse",
            }
        ]
        self.assertTrue(partial_stores)
        self.assertTrue(all(
            "request" in node.benchmark_params
            and "split" in node.benchmark_params
            and "global_split_idx" in node.benchmark_params
            for node in partial_stores
        ))

    def test_source_derived_lane_operation_counts(self):
        dag = build_dense_decode_dag(
            workload(1), KernelResources(sm_count=1), ATOM_MAP
        )

        def anchored(suffix: str):
            return [
                node for node in dag.nodes.values()
                if node.source_anchor.endswith(suffix)
            ]

        real_page = [
            node for node in dag.nodes.values()
            if node.phase == "tail_update.single_drain"
            and node.actor.endswith(".wg0")
        ]
        maximum = [node for node in real_page if node.atom_id == "max_ftz_f32"]
        score_ffma = anchored("rP scale-minus-max FFMA")
        score_shuffle = anchored("two rows xor delta 1/2")
        exponent = [
            node for node in real_page if node.atom_id == "ex2_approx_ftz_f32"
        ]
        output_rescale = [
            node for node in real_page
            if node.source_anchor.endswith(("softmax scaling", "online rO rescale"))
        ]
        self.assertEqual(sum(node.work_amount for node in maximum), 4096 + 768)
        self.assertEqual([node.work_amount for node in score_ffma], [4096])
        self.assertEqual([node.work_amount for node in score_shuffle], [256, 256])
        self.assertEqual(sum(node.work_amount for node in exponent), 4096 + 256)
        self.assertEqual(sum(node.work_amount for node in output_rescale), 16384 + 256)
        self.assertTrue(all(
            node.work_unit == "lane_op"
            for node in maximum + score_ffma + score_shuffle + exponent + output_rescale
        ))

        partial_rows = build_dense_decode_dag(
            workload(1, num_heads_q=3), KernelResources(sm_count=1), ATOM_MAP
        )
        by_anchor = {
            node.source_anchor: node for node in partial_rows.nodes.values()
            if node.phase == "epilogue_nosplit"
        }
        self.assertEqual(
            by_anchor["splitkv_mla.cuh:store_o reciprocal normalization"].work_amount,
            64 * 8,
        )
        self.assertEqual(
            by_anchor["splitkv_mla.cuh:rO / rL"].work_amount, 64 * 512
        )
        self.assertEqual(
            by_anchor["splitkv_mla.cuh:output conversion"].work_amount,
            64 * 512,
        )

    def test_combine_lane_operation_counts(self):
        dag = build_dense_decode_dag(
            workload(20, num_heads_q=3), KernelResources(sm_count=2), ATOM_MAP
        )

        def anchored(suffix: str):
            return [
                node for node in dag.nodes.values()
                if node.source_anchor.endswith(suffix)
            ]

        self.assertEqual(
            [node.work_amount for node in anchored("LSE max bucket")], [576]
        )
        self.assertEqual(
            [node.work_amount for node in anchored("LSE sum and shuffle reduction")],
            [576],
        )
        shuffles = anchored("LSE warp shuffle")
        self.assertEqual(len(shuffles), 10)
        self.assertEqual({node.work_amount for node in shuffles}, {3 * 32})
        scale_loads = anchored("smem_buf scale load")
        self.assertEqual(len(scale_loads), 2)
        self.assertEqual({node.work_amount for node in scale_loads}, {3 * 32})


class SimulatorTests(unittest.TestCase):
    def test_cost_lookup_rejects_missing_mandatory_query_dimension(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({"benchmarks": [{
                "id": "parameterized", "kind": "operation", "family": "synthetic",
                "source": "compute/synthetic/parameterized.cu",
                "result_csv": "result.csv", "parameters": ["topology"],
            }]}))
            with (root / "result.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "name", "params_json", "latency_value", "latency_unit",
                    "initiation_interval_cycles", "p10", "p50", "p90",
                ])
                writer.writeheader()
                writer.writerow({
                    "name": "parameterized", "params_json": '{"topology":"unique"}',
                    "latency_value": 1, "latency_unit": "cycles",
                    "initiation_interval_cycles": 1, "p10": 1, "p50": 1, "p90": 1,
                })
            with self.assertRaisesRegex(CoverageError, "mandatory sweep parameters"):
                CostDatabase(root).lookup("parameterized", {})

    def test_operation_lookup_uses_actual_kernel_block_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({"benchmarks": [{
                "id": "block_sensitive", "kind": "operation", "family": "synthetic",
                "source": "compute/synthetic/block_sensitive.cu",
                "result_csv": "result.csv", "parameters": ["blocks"],
            }]}))
            fields = [
                "name", "params_json", "latency_value", "latency_unit",
                "initiation_interval_cycles", "throughput_value", "throughput_unit",
                "memory_bandwidth_value", "memory_bandwidth_unit", "hardware_utilization",
                "p10", "p50", "p90", "sample_count", "gpu_uuid",
                "source_sha256", "sass_sha256",
            ]
            with (root / "result.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for blocks, latency in ((1, 2), (8, 20)):
                    writer.writerow({
                        "name": "block_sensitive",
                        "params_json": json.dumps({"blocks": blocks}),
                        "latency_value": latency, "latency_unit": "cycles",
                        "initiation_interval_cycles": latency,
                        "p10": latency, "p50": latency, "p90": latency,
                    })
            dag = DenseDecodeDAG()
            for index in range(8):
                dag.add_node(OperationNode(
                    f"n{index}", "probe", f"actor{index}", "block_sensitive", {}, 1,
                    "instruction", "fp32", False, "synthetic",
                    cta_id=f"main.p0.h0.m{index}",
                ))
            result = simulate(
                dag, CostDatabase(root),
                KernelResources(sm_count=8),
            ).result
            self.assertEqual(result["predicted_e2e_cycles"]["p50"], 20)

    def test_wave_tail_uses_its_own_curve_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({"benchmarks": [{
                "id": "wave_sensitive", "kind": "operation", "family": "synthetic",
                "source": "compute/synthetic/wave_sensitive.cu",
                "result_csv": "result.csv",
                "parameters": ["blocks", "active_sm", "resident_cta"],
            }]}))
            fields = [
                "name", "params_json", "latency_value", "latency_unit",
                "initiation_interval_cycles", "throughput_value", "throughput_unit",
                "memory_bandwidth_value", "memory_bandwidth_unit", "hardware_utilization",
                "p10", "p50", "p90", "sample_count", "gpu_uuid",
                "source_sha256", "sass_sha256",
            ]
            with (root / "result.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for params, latency in (
                    ({"blocks": 2, "active_sm": 2, "resident_cta": 1}, 20),
                    ({"blocks": 1, "active_sm": 1, "resident_cta": 1}, 3),
                ):
                    writer.writerow({
                        "name": "wave_sensitive", "params_json": json.dumps(params),
                        "latency_value": latency, "latency_unit": "cycles",
                        "initiation_interval_cycles": latency,
                        "p10": latency, "p50": latency, "p90": latency,
                    })
            db = CostDatabase(root)
            resources = KernelResources(sm_count=2)
            model = ResourceModel(
                db, resources,
                [
                    "main.p0.h0.m0", "main.p0.h0.m1", "main.p0.h0.m2",
                ],
                {"fp32"},
            )
            full = model.benchmark_query({}, "main.p0.h0.m0", "wave_sensitive")
            tail = model.benchmark_query({}, "main.p0.h0.m2", "wave_sensitive")

            self.assertEqual(
                {key: full[key] for key in ("blocks", "active_sm", "resident_cta")},
                {"blocks": 2, "active_sm": 2, "resident_cta": 1},
            )
            self.assertEqual(
                {key: tail[key] for key in ("blocks", "active_sm", "resident_cta")},
                {"blocks": 1, "active_sm": 1, "resident_cta": 1},
            )
            self.assertEqual(db.lookup("wave_sensitive", full).p50_cycles, 20)
            self.assertEqual(db.lookup("wave_sensitive", tail).p50_cycles, 3)

    def test_pdl_grid_issue_is_per_sm_not_gpu_global(self):
        dag = DenseDecodeDAG()
        for index in range(2):
            dag.add_node(OperationNode(
                f"pdl{index}", "combine_dispatch", f"cta{index}",
                "griddepcontrol_wait", {}, 1, "instruction", "grid", False,
                "synthetic", cta_id=f"main.p0.h0.m{index}",
            ))
        with tempfile.TemporaryDirectory() as directory:
            prediction, scheduled, _ = simulator_module._schedule_once(
                dag, database(Path(directory), {"griddepcontrol_wait"}),
                KernelResources(sm_count=2), "p50",
            )

        self.assertEqual(scheduled["pdl0"].issue_start, 0)
        self.assertEqual(scheduled["pdl1"].issue_start, 0)
        self.assertEqual(prediction.result["predicted_e2e_cycles"], 2)
        self.assertEqual(
            prediction.result["resource_capacity"]["grid"]["parallel_units"], 2
        )

    def test_concurrent_requests_miss_until_first_hbm_fill_completes(self):
        dag = DenseDecodeDAG()
        for index in range(3):
            dag.add_node(OperationNode(
                f"load{index}", "probe", f"cta{index}", "ld_global_u32",
                {
                    "bytes": 128, "cache_mode": "hbm_stream",
                    "memory_object": "table", "cache_line": 7,
                    "working_set_bytes": 128,
                },
                1, "instruction", "l2", False, "synthetic",
                cta_id=f"main.p0.h0.m{index}",
            ))
        # load0/load1 request the same line concurrently on different SMs.
        # load2 is issued only after load0's modeled HBM->L2 fill completes.
        dag.add_dependency("load0", "load2")
        with tempfile.TemporaryDirectory() as directory:
            result = simulate(
                dag, database(Path(directory), {"ld_global_u32"}),
                KernelResources(sm_count=2),
            ).result

        self.assertEqual(result["cache_events"], {"l2_hit": 1, "hbm_miss": 2})
        self.assertEqual(result["memory_traffic_bytes"]["hbm"], 256)
        self.assertEqual(result["memory_traffic_bytes"]["l2"], 384)

    def test_partial_output_store_fill_is_reused_only_when_working_set_fits_l2(self):
        def run_case(l2_bytes: int):
            dag = DenseDecodeDAG()
            common = {
                "bytes": 128, "cache_mode": "hbm_stream",
                "memory_object": "partial_output", "request": 0,
                "working_set_bytes": 256,
            }
            dag.add_node(OperationNode(
                "store", "epilogue_split", "main", "st_global_f32",
                common | {"producer_consumer": True},
                1, "instruction", "hbm", False, "synthetic",
                cta_id="main.p0.h0.m0",
            ))
            dag.add_node(OperationNode(
                "load", "combine_accumulate", "combine", "ld_global_v4_f32",
                common,
                1, "instruction", "l2", False, "synthetic",
                cta_id="combine.r0.q0.h0",
            ))
            dag.add_dependency(
                "store", "load", kind="memory_visibility",
                label="partial output producer-consumer visibility",
            )
            with tempfile.TemporaryDirectory() as directory:
                return simulate(
                    dag,
                    database(Path(directory), {"st_global_f32", "ld_global_v4_f32"}),
                    KernelResources(sm_count=1, l2_bytes=l2_bytes),
                ).result

        retained = run_case(512)
        oversized = run_case(128)

        self.assertEqual(retained["cache_events"], {"l2_hit": 1, "hbm_miss": 0})
        self.assertEqual(retained["memory_traffic_bytes"], {"l2": 256, "hbm": 128})
        self.assertEqual(oversized["cache_events"], {"l2_hit": 0, "hbm_miss": 1})
        self.assertEqual(oversized["memory_traffic_bytes"], {"l2": 256, "hbm": 256})

    def test_tma_cold_route_uses_hbm_then_l2_cache_mode_curves(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = [
                {
                    "id": "tma_atom", "kind": "operation",
                    "family": "tma_load", "source": "memory/tma/tma_atom.cu",
                    "result_csv": "result.csv",
                },
                {
                    "id": "tma_curve", "kind": "resource_curve",
                    "family": "tma_service", "source": "resource/tma/tma_curve.cu",
                    "result_csv": "result.csv",
                },
            ]
            (root / "manifest.json").write_text(json.dumps({"benchmarks": entries}))
            fields = [
                "name", "params_json", "latency_value", "latency_unit",
                "initiation_interval_cycles", "throughput_value", "throughput_unit",
                "memory_bandwidth_value", "memory_bandwidth_unit",
                "p10", "p50", "p90",
            ]
            with (root / "result.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "name": "tma_atom", "params_json": "{}",
                    "latency_value": 10, "latency_unit": "cycles",
                    "initiation_interval_cycles": 1,
                    "memory_bandwidth_value": 50,
                    "memory_bandwidth_unit": "GB/s",
                    "p10": 10, "p50": 10, "p90": 10,
                })
                for cache_mode, bandwidth in (("hbm_stream", 100), ("l2_hot", 400)):
                    writer.writerow({
                        "name": "tma_curve",
                        "params_json": json.dumps({
                            "resource": "tma", "cache_mode": cache_mode,
                        }),
                        "latency_value": 1, "latency_unit": "cycles",
                        "initiation_interval_cycles": 1,
                        "memory_bandwidth_value": bandwidth,
                        "memory_bandwidth_unit": "GB/s",
                        "p10": 1, "p50": 1, "p90": 1,
                    })

            db = CostDatabase(root)
            resources = KernelResources(sm_count=1, sm_clock_mhz=1_000)
            model = ResourceModel(
                db, resources, ["main.p0.h0.m0"], {"tma"},
            )
            params = {
                "bytes": 1_000, "cache_mode": "hbm_stream",
                "memory_object": "k", "physical_page": 0, "kv_head": 0,
                "tile": 0, "working_set_bytes": resources.l2_bytes * 2,
            }
            plan = model.access_plan(
                "tma", "tma_atom", params, "main.p0.h0.m0",
            )
            cost = db.lookup("tma_atom", model.benchmark_query(
                params, "main.p0.h0.m0", "tma_atom",
            ))
            hbm_cycles = model.memory_service_cycles(
                plan, ("hbm", "gpu"), cost,
            )
            l2_cycles = model.memory_service_cycles(
                plan, ("l2", "gpu"), cost,
            )

        self.assertEqual(plan.memory_keys, (("hbm", "gpu"), ("l2", "gpu")))
        self.assertAlmostEqual(hbm_cycles, 10.0)
        self.assertAlmostEqual(l2_cycles, 2.5)

    def test_cold_load_services_hbm_before_l2(self):
        dag = DenseDecodeDAG()
        dag.add_node(OperationNode(
            "load", "probe", "cta", "ld_global_u32",
            {
                "bytes": 128_000, "cache_mode": "hbm_stream",
                "memory_object": "table", "cache_line": 0,
                "working_set_bytes": 256_000,
            },
            1, "instruction", "l2", False, "synthetic",
            cta_id="main.p0.h0.m0",
        ))
        resources = KernelResources(sm_count=1)
        with tempfile.TemporaryDirectory() as directory:
            result = simulate(
                dag, database(Path(directory), {"ld_global_u32"}), resources,
            ).result

        services = [
            item["service_resource"] for item in result["critical_path"]
            if item["node_id"] == "load" and item["event"] == "service"
            and item.get("service_kind") == "queue"
        ]
        self.assertEqual(services, ["hbm", "l2"])
        memory_cycles = 128_000 / (1_000 * 1_000 / resources.sm_clock_mhz)
        self.assertAlmostEqual(
            result["predicted_e2e_cycles"]["p50"], 1 + 2 * memory_cycles
        )

    def test_global_memory_issue_capacity_is_per_sm_lsu(self):
        def make_dag(cta_ids: tuple[str, str]) -> DenseDecodeDAG:
            value = DenseDecodeDAG()
            for index, cta_id in enumerate(cta_ids):
                value.add_node(OperationNode(
                    f"load{index}", "probe", f"cta{index}", "ld_global_u32",
                    {
                        "bytes": 128, "cache_mode": "hbm_stream",
                        "memory_object": "table", "cache_line": index,
                        "working_set_bytes": 256,
                    },
                    1, "instruction", "l2", False, "synthetic", cta_id=cta_id,
                ))
            return value

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = database(root, {"ld_global_u32"})
            _, same_sm, _ = simulator_module._schedule_once(
                make_dag(("main.p0.h0.m0", "main.p0.h0.m0")),
                db, KernelResources(sm_count=2), "p50",
            )
            _, different_sms, _ = simulator_module._schedule_once(
                make_dag(("main.p0.h0.m0", "main.p0.h0.m1")),
                db, KernelResources(sm_count=2), "p50",
            )

        self.assertEqual(same_sm["load0"].issue_start, 0)
        self.assertEqual(
            same_sm["load1"].issue_start, same_sm["load0"].issue_finish
        )
        self.assertEqual(different_sms["load0"].issue_start, 0)
        self.assertEqual(different_sms["load1"].issue_start, 0)

    def test_synthetic_critical_path_phase_contributions_are_conserved(self):
        dag = DenseDecodeDAG()
        for node_id, phase, cta_id in (
            ("critical_a", "phase_a", "main.p0.h0.m0"),
            ("critical_b", "phase_b", "main.p0.h0.m0"),
            ("parallel", "phase_parallel", "main.p0.h0.m1"),
        ):
            dag.add_node(OperationNode(
                node_id, phase, node_id, "compute_atom", {}, 1,
                "lane_op", "fp32", False, "synthetic", cta_id=cta_id,
            ))
        dag.add_dependency("critical_a", "critical_b")
        with tempfile.TemporaryDirectory() as directory:
            result = simulate(
                dag, database(Path(directory), {"compute_atom"}),
                KernelResources(sm_count=2),
            ).result

        contributions = {
            phase: timing["critical_path_contribution_cycles"]
            for phase, timing in result["phase_timing"].items()
        }
        makespan = result["predicted_e2e_cycles"]["p50"]
        self.assertEqual(contributions["phase_parallel"], 0)
        self.assertEqual(contributions["phase_a"], 2)
        self.assertEqual(contributions["phase_b"], 2)
        self.assertTrue(math.isclose(sum(contributions.values()), makespan))

    def test_grid_throughput_is_normalized_once_by_active_sms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = [
                {"id": "grid_rate", "kind": "operation", "family": "synthetic",
                 "source": "compute/synthetic/grid_rate.cu", "result_csv": "result.csv",
                 "parameters": ["blocks"]},
                {"id": "per_sm_rate", "kind": "operation", "family": "synthetic",
                 "source": "compute/synthetic/per_sm_rate.cu", "result_csv": "result.csv",
                 "parameters": ["blocks"]},
            ]
            (root / "manifest.json").write_text(json.dumps({"benchmarks": entries}))
            fields = [
                "name", "params_json", "latency_value", "latency_unit",
                "initiation_interval_cycles", "throughput_value", "throughput_unit",
                "p10", "p50", "p90",
            ]
            with (root / "result.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "name": "grid_rate",
                    "params_json": json.dumps({
                        "blocks": 0, "resolved_blocks": 8, "unique_active_sms": 4,
                    }),
                    "latency_value": 1, "latency_unit": "cycles",
                    "initiation_interval_cycles": 1,
                    "throughput_value": 400, "throughput_unit": "Glane-op/s",
                    "p10": 1, "p50": 1, "p90": 1,
                })
                writer.writerow({
                    "name": "per_sm_rate",
                    "params_json": json.dumps({"blocks": 8, "active_sm": 4}),
                    "latency_value": 1, "latency_unit": "cycles",
                    "initiation_interval_cycles": 1,
                    "throughput_value": 100,
                    "throughput_unit": "Glane-op/s/SM",
                    "p10": 1, "p50": 1, "p90": 1,
                })
            db = CostDatabase(root)
            self.assertEqual(db.records["grid_rate"][0].params["blocks"], 8)
            self.assertEqual(db.records["grid_rate"][0].params["active_sm"], 4)
            grid = db.lookup("grid_rate", {"blocks": 8, "device_sm_count": 132})
            per_sm = db.lookup("per_sm_rate", {"blocks": 8, "device_sm_count": 132})
            self.assertEqual(grid.throughput_value, 100)
            self.assertEqual(per_sm.throughput_value, 100)

    def test_schedule_once_scales_to_40001_node_chain(self):
        dag = DenseDecodeDAG()
        previous = None
        for index in range(40_001):
            node_id = f"n{index:05d}"
            dag.add_node(OperationNode(
                node_id, "probe", "warp", "scalar_atom", {}, 1,
                "lane_op", "fp32", False, "synthetic",
            ))
            if previous is not None:
                dag.add_dependency(previous, node_id)
            previous = node_id
        with tempfile.TemporaryDirectory() as directory:
            db = database(Path(directory), {"scalar_atom"})
            started = time.perf_counter()
            prediction, scheduled, _ = simulator_module._schedule_once(
                dag, db, KernelResources(sm_count=1), "p50"
            )
            elapsed = time.perf_counter() - started
        self.assertEqual(len(scheduled), 40_001)
        self.assertEqual(prediction.result["node_count"], 40_001)
        # The previous pending-list scan was quadratic and cannot finish this
        # chain on this order of time; retain generous headroom for CI hosts.
        self.assertLess(elapsed, 20.0)

    def test_async_pipeline_uses_latency_plus_initiation_intervals(self):
        dag = DenseDecodeDAG()
        previous = None
        for index in range(9):
            node_id = f"g{index}"
            dag.add_node(OperationNode(
                node_id, "probe", "wg0", "async_atom", {}, 1,
                "committed_group", "tensor", True, "synthetic",
            ))
            if previous is not None:
                dag.add_dependency(previous, node_id, src_event="issue",
                                   dst_event="issue", kind="program")
            previous = node_id
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = database(root, {"async_atom"})
            with (root / "result.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["latency_value"] = rows[0]["p50"] = "100"
            rows[0]["p10"] = rows[0]["p90"] = "100"
            rows[0]["initiation_interval_cycles"] = "5"
            with (root / "result.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            result = simulate(dag, CostDatabase(root), KernelResources(sm_count=1)).result
            self.assertAlmostEqual(result["predicted_e2e_cycles"]["p50"], 140.0)

    def test_q_and_generic_cacheline_lru_routing(self):
        with tempfile.TemporaryDirectory() as directory:
            db = database(Path(directory), {"ld_tma_q", "ld_global_u32"})
            resources = KernelResources(sm_count=2, l2_bytes=128)
            q_cache = ResourceModel(
                db, resources,
                ["main.p0.h0.m0", "main.p1.h0.m0"], {"tma"},
            )
            q0 = {
                "bytes": 128, "cache_mode": "hbm_stream",
                "memory_object": "q", "request": 0, "kv_head": 0,
                "m_block": 0, "working_set_bytes": 128,
            }
            first = q_cache.access_plan(
                "tma", "ld_tma_q", q0, "main.p0.h0.m0"
            )
            reused = q_cache.access_plan(
                "tma", "ld_tma_q", q0, "main.p1.h0.m0"
            )
            self.assertEqual(first.memory_keys, (("hbm", "gpu"), ("l2", "gpu")))
            self.assertEqual(reused.memory_keys, (("l2", "gpu"),))

            q1 = dict(q0) | {"request": 1}
            q_cache.access_plan("tma", "ld_tma_q", q1, "main.p1.h0.m0")
            evicted = q_cache.access_plan(
                "tma", "ld_tma_q", q0, "main.p0.h0.m0"
            )
            self.assertEqual(evicted.memory_keys, (("hbm", "gpu"), ("l2", "gpu")))
            self.assertEqual(q_cache.cache_events, {"l2_hit": 1, "hbm_miss": 3})

            line_cache = ResourceModel(db, resources, [], {"l2"})
            line = {
                "bytes": 4, "cache_mode": "hbm_stream",
                "memory_object": "block_table", "cache_line": 7,
                "working_set_bytes": 128,
            }
            cold = line_cache.access_plan("l2", "ld_global_u32", line, None)
            hot = line_cache.access_plan("l2", "ld_global_u32", line, None)
            self.assertEqual(cold.memory_keys, (("hbm", "gpu"), ("l2", "gpu")))
            self.assertEqual(hot.memory_keys, (("l2", "gpu"),))
            self.assertEqual(line_cache.cache_events, {"l2_hit": 1, "hbm_miss": 1})

    def test_cache_routing_is_invariant_to_node_construction_order(self):
        def make_dag(reverse: bool) -> DenseDecodeDAG:
            value = DenseDecodeDAG()
            specs = [
                ("access_a", "actor.a", 0),
                ("access_b", "actor.b", 1),
                ("reuse_a", "actor.c", 0),
            ]
            for node_id, actor, cache_line in reversed(specs) if reverse else specs:
                value.add_node(OperationNode(
                    node_id, "probe", actor, "ld_global_u32",
                    {"bytes": 128, "cache_mode": "hbm_stream",
                     "memory_object": "table", "cache_line": cache_line,
                     "working_set_bytes": 256},
                    1, "instruction", "l2", False, f"synthetic:{actor}",
                ))
            value.add_dependency("access_a", "reuse_a")
            return value

        with tempfile.TemporaryDirectory() as directory:
            db = database(Path(directory), {"ld_global_u32"})
            resources = KernelResources(sm_count=1, l2_bytes=128)
            forward = simulate(make_dag(False), db, resources).result
            reverse = simulate(make_dag(True), db, resources).result
        for field in (
            "predicted_e2e_cycles", "memory_traffic_bytes", "cache_events",
            "resource_utilization", "resource_capacity",
        ):
            self.assertEqual(forward[field], reverse[field], field)
        self.assertFalse(any(
            "cache-order fixed point" in warning for warning in forward["warnings"]
        ))

    def test_resource_utilization_uses_queue_capacity(self):
        dag = DenseDecodeDAG()
        for index in range(2):
            dag.add_node(OperationNode(
                f"compute{index}", "probe", f"warp{index}", "compute_atom", {}, 1,
                "lane_op", "fp32", False, "synthetic",
                cta_id=f"main.p0.h0.m{index}",
            ))
        with tempfile.TemporaryDirectory() as directory:
            result = simulate(
                dag, database(Path(directory), {"compute_atom"}),
                KernelResources(sm_count=2),
            ).result
        capacity = result["resource_capacity"]["fp32"]
        self.assertEqual(capacity["scope"], "per_sm")
        self.assertEqual(capacity["active_sm"], 2)
        self.assertEqual(capacity["parallel_units"], 2)
        self.assertEqual(capacity["available_cycles"], 4)
        self.assertEqual(capacity["busy_cycles"], 4)
        self.assertEqual(result["resource_utilization"]["fp32"], 1)
        self.assertTrue(all(
            item["event"] in {"issue", "service", "complete"}
            for item in result["critical_path"]
        ))
        service_events = [
            item for item in result["critical_path"] if item["event"] == "service"
        ]
        self.assertTrue(service_events)
        self.assertTrue(all("service_resource" in item for item in service_events))

        memory_dag = DenseDecodeDAG()
        for index in range(2):
            memory_dag.add_node(OperationNode(
                f"load{index}", "probe", f"warp{index}", "ld_global_u32",
                {"bytes": 128, "cache_mode": "hbm_stream",
                 "memory_object": "table", "cache_line": index,
                 "working_set_bytes": 256},
                1, "instruction", "l2", False, "synthetic",
                cta_id=f"main.p0.h0.m{index}",
            ))
        with tempfile.TemporaryDirectory() as directory:
            memory = simulate(
                memory_dag, database(Path(directory), {"ld_global_u32"}),
                KernelResources(sm_count=2),
            ).result
        for resource in ("l2", "hbm"):
            self.assertEqual(memory["resource_capacity"][resource]["scope"], "gpu")
            self.assertEqual(memory["resource_capacity"][resource]["parallel_units"], 1)
        self.assertTrue(all(
            0 <= value <= 1 for value in memory["resource_utilization"].values()
        ))

    def test_first_combine_wave_waits_for_last_main_wave_on_its_sm(self):
        dag = DenseDecodeDAG()
        placements = (
            ("main0", "main.p0.h0.m0"),
            ("main1", "main.p0.h0.m1"),
            ("main2", "main.p0.h0.m2"),
            ("combine", "combine.r0.q0.h0"),
        )
        for node_id, cta_id in placements:
            dag.add_node(OperationNode(
                node_id, "probe", node_id, "probe", {}, 1,
                "instruction", "fp32", False, "synthetic", cta_id=cta_id,
            ))
        with tempfile.TemporaryDirectory() as directory:
            db = database(Path(directory), {"probe"})
            model = ResourceModel(
                db, KernelResources(sm_count=2),
                [cta_id for _, cta_id in placements], {"fp32"},
            )
            predecessors = simulator_module._wave_predecessors(dag, model)
        self.assertEqual(predecessors["main2"], ["main0"])
        self.assertEqual(predecessors["combine"], ["main2"])

    def test_prediction_provenance_contains_only_selected_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = [
                {"id": "ld_probe", "kind": "operation", "family": "global_load",
                 "source": "memory/global_load/ld_probe.cu", "result_csv": "result.csv",
                 "parameters": ["blocks"]},
                {"id": "l2_curve", "kind": "resource_curve", "family": "l2",
                 "source": "resource/l2/l2_curve.cu", "result_csv": "result.csv"},
                {"id": "hbm_curve", "kind": "resource_curve", "family": "hbm",
                 "source": "resource/hbm/hbm_curve.cu", "result_csv": "result.csv"},
            ]
            (root / "manifest.json").write_text(json.dumps({"benchmarks": entries}))
            fields = [
                "name", "params_json", "latency_value", "latency_unit",
                "initiation_interval_cycles", "throughput_value", "throughput_unit",
                "memory_bandwidth_value", "memory_bandwidth_unit",
                "p10", "p50", "p90", "gpu_uuid", "source_sha256", "sass_sha256",
            ]
            rows = []
            for blocks in (1, 2, 8):
                rows.append({
                    "name": "ld_probe", "params_json": json.dumps({"blocks": blocks}),
                    "latency_value": blocks, "latency_unit": "cycles",
                    "initiation_interval_cycles": 1,
                    "memory_bandwidth_value": 1000, "memory_bandwidth_unit": "GB/s",
                    "p10": blocks, "p50": blocks, "p90": blocks,
                })
            for atom_id, resource, cache_mode in (
                ("l2_curve", "l2", "l2_hot"),
                ("hbm_curve", "hbm", "hbm_stream"),
            ):
                for working_set in (128, 4096):
                    rows.append({
                        "name": atom_id,
                        "params_json": json.dumps({
                            "resource": resource, "direction": "load",
                            "cache_mode": cache_mode,
                            "working_set_bytes": working_set,
                        }),
                        "latency_value": 1, "latency_unit": "cycles",
                        "initiation_interval_cycles": 1,
                        "memory_bandwidth_value": 2000,
                        "memory_bandwidth_unit": "GB/s",
                        "p10": 1, "p50": 1, "p90": 1,
                    })
            with (root / "result.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            dag = DenseDecodeDAG()
            dag.add_node(OperationNode(
                "load", "probe", "warp", "ld_probe",
                {"bytes": 128, "cache_mode": "hbm_stream",
                 "memory_object": "probe", "cache_line": 0,
                 "working_set_bytes": 128},
                1, "instruction", "l2", False, "synthetic",
                cta_id="main.p0.h0.m0",
            ))
            result = simulate(
                dag, CostDatabase(root), KernelResources(sm_count=1)
            ).result
        self.assertEqual(len(result["atom_provenance"]), 1)
        self.assertEqual(result["atom_provenance"][0]["atom_id"], "ld_probe")
        self.assertEqual(result["atom_provenance"][0]["row"], 2)
        resource_rows = {
            (item["atom_id"], item["row"])
            for item in result["resource_curve_provenance"]
        }
        self.assertEqual(resource_rows, {("l2_curve", 2), ("hbm_curve", 2)})

    def test_interference_curve_applies_only_to_timeline_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = [
                {"id": "tensor_atom", "kind": "operation", "family": "synthetic",
                 "source": "compute/synthetic/tensor_atom.cu", "result_csv": "result.csv"},
                {"id": "tma_atom", "kind": "operation", "family": "synthetic",
                 "source": "memory/synthetic/tma_atom.cu", "result_csv": "result.csv"},
                {"id": "tensor_tma_curve", "kind": "resource_curve",
                 "family": "synthetic", "source": "resource/synthetic/tensor_tma_curve.cu",
                 "result_csv": "result.csv"},
            ]
            (root / "manifest.json").write_text(json.dumps({"benchmarks": entries}))
            fields = [
                "name", "params_json", "latency_value", "latency_unit",
                "initiation_interval_cycles", "throughput_value", "throughput_unit",
                "memory_bandwidth_value", "memory_bandwidth_unit", "hardware_utilization",
                "p10", "p50", "p90", "sample_count", "gpu_uuid",
                "source_sha256", "sass_sha256",
            ]
            rows = [
                {"name": "tensor_atom", "params_json": "{}", "latency_value": 100,
                 "latency_unit": "cycles", "initiation_interval_cycles": 5,
                 "p10": 100, "p50": 100, "p90": 100},
                {"name": "tma_atom", "params_json": "{}", "latency_value": 20,
                 "latency_unit": "cycles", "initiation_interval_cycles": 5,
                 "p10": 20, "p50": 20, "p90": 20},
                {"name": "tensor_tma_curve",
                 "params_json": '{"resource":"tensor","actors":1}',
                 "latency_value": 1, "latency_unit": "cycles",
                 "initiation_interval_cycles": 1, "throughput_value": 100,
                 "throughput_unit": "Gop/s", "p10": 1, "p50": 1, "p90": 1},
                {"name": "tensor_tma_curve",
                 "params_json": '{"resource":"tensor","peer_resource":"tma","actors":2}',
                 "latency_value": 1, "latency_unit": "cycles",
                 "initiation_interval_cycles": 1, "throughput_value": 100 / 3,
                 "throughput_unit": "Gop/s", "p10": 1, "p50": 1, "p90": 1},
            ]
            with (root / "result.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)

            def make_dag(serial: bool, reverse_ids: bool = False) -> DenseDecodeDAG:
                value = DenseDecodeDAG()
                tma_id, tensor_id = (("z_tma", "a_tensor") if reverse_ids
                                     else ("a_tma", "b_tensor"))
                value.add_node(OperationNode(
                    tma_id, "probe", "tma", "tma_atom", {}, 1,
                    "instruction", "tma", True, "synthetic",
                ))
                value.add_node(OperationNode(
                    tensor_id, "probe", "wg", "tensor_atom", {}, 1,
                    "instruction", "tensor", True, "synthetic",
                ))
                if serial:
                    value.add_dependency(
                        tma_id, tensor_id, src_event="complete",
                        dst_event="issue", kind="tma_ready",
                    )
                return value

            db = CostDatabase(root)
            resources = KernelResources(sm_count=1)
            overlapped = simulate(make_dag(False), db, resources).result
            reverse = simulate(make_dag(False, True), db, resources).result
            serialized = simulate(make_dag(True), db, resources).result
            self.assertAlmostEqual(overlapped["predicted_e2e_cycles"]["p50"], 140.0)
            self.assertEqual(overlapped["predicted_e2e_cycles"],
                             reverse["predicted_e2e_cycles"])
            self.assertAlmostEqual(serialized["predicted_e2e_cycles"]["p50"], 120.0)
            self.assertTrue(overlapped["interaction_solver"]["converged"])

    def test_interference_curve_requires_no_peer_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({"benchmarks": [{
                "id": "tensor_tma_curve", "kind": "resource_curve",
                "family": "synthetic",
                "source": "resource/synthetic/tensor_tma_curve.cu",
                "result_csv": "result.csv",
            }]}))
            with (root / "result.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "name", "params_json", "latency_value", "latency_unit",
                    "initiation_interval_cycles", "throughput_value",
                    "throughput_unit", "p10", "p50", "p90",
                ])
                writer.writeheader()
                writer.writerow({
                    "name": "tensor_tma_curve",
                    "params_json": (
                        '{"resource":"tensor","peer_resource":"tma",'
                        '"actors":2}'
                    ),
                    "latency_value": 1, "latency_unit": "cycles",
                    "initiation_interval_cycles": 1,
                    "throughput_value": 50, "throughput_unit": "Gop/s",
                    "p10": 1, "p50": 1, "p90": 1,
                })
            with self.assertRaisesRegex(CoverageError, "actors=1"):
                CostDatabase(root).resource_slowdown(
                    "tensor", {"peer_resource": "tma", "actors": 2}
                )

    def test_intervals_conservation_and_calibration_isolation(self):
        resources = KernelResources(sm_count=1)
        dag = build_dense_decode_dag(workload(2), resources, ATOM_MAP)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = database(root, {node.atom_id for node in dag.nodes.values()})
            before = simulate(dag, db, resources).result
            contributions = sum(item["critical_path_contribution_cycles"]
                                for item in before["phase_timing"].values())
            self.assertTrue(math.isclose(
                contributions, before["predicted_e2e_cycles"]["p50"], rel_tol=1e-10
            ))
            self.assertLessEqual(before["predicted_e2e_cycles"]["p10"],
                                 before["predicted_e2e_cycles"]["p50"])
            self.assertLessEqual(before["predicted_e2e_cycles"]["p50"],
                                 before["predicted_e2e_cycles"]["p90"])
            (root.parent / "unrelated-calibration.csv").write_text("probe,p50\nx,999999\n")
            after = simulate(dag, CostDatabase(root), resources).result
            self.assertEqual(before, after)
        self.assertNotIn("from .calibration", inspect.getsource(simulator_module))
        action = next(item for item in build_parser()._subparsers._group_actions
                      if item.dest == "command")
        predict = action.choices["predict"]
        self.assertNotIn("calibration_root", {item.dest for item in predict._actions})

    def test_calibration_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            valid = parent / "microbench"
            valid.mkdir()
            database(valid, {"probe"})
            calibration = parent / "calibration"
            valid.rename(calibration)
            self.assertTrue((calibration / "manifest.json").is_file())
            self.assertTrue((calibration / "result.csv").is_file())
            with self.assertRaisesRegex(
                CostDatabaseError, "official prediction cannot load a calibration path"
            ):
                CostDatabase(calibration)

    def test_official_prediction_rejects_source_undefined_scheduler(self):
        dag = build_dense_decode_dag(workload(1), KernelResources(sm_count=8), ATOM_MAP)
        self.assertFalse(dag.scheduler.source_defined)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text('{"benchmarks":[]}')
            with self.assertRaisesRegex(ValueError, "undefined"):
                simulate(dag, CostDatabase(root), KernelResources(sm_count=8))

    def test_calibration_probe_schema_reports_residual_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            microbench = root / "microbench"
            calibration = root / "probe-data"
            microbench.mkdir()
            calibration.mkdir()
            database(microbench, {"fadd_rn_ftz_f32"})
            probe_dag = {
                "probe": "scalar_probe",
                "nodes": [
                    {"id": "a", "actor": "warp", "atom_id": "fadd_rn_ftz_f32",
                     "work_amount": 4},
                    {"id": "b", "actor": "warp", "atom_id": "fadd_rn_ftz_f32",
                     "work_amount": "num_splits dependent"},
                ],
                "dependencies": [["a.complete", "b.issue", "data"]],
            }
            (calibration / "probe.json").write_text(json.dumps(probe_dag))
            (calibration / "manifest.json").write_text(json.dumps({
                "probes": [{"id": "scalar_probe", "probe_dag": "probe.json"}]
            }))
            with (calibration / "result.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "probe", "params_json", "measured_latency_value",
                    "measured_latency_unit", "p50",
                ])
                writer.writeheader()
                writer.writerow({"probe": "scalar_probe", "params_json": '{"num_splits":2}',
                                 "measured_latency_value": 10, "measured_latency_unit": "cycles",
                                 "p50": 10})
            report = validate_calibration(microbench, calibration, KernelResources(sm_count=1))
            self.assertEqual(report["policy"], "diagnostic_only_no_prediction_correction")
            self.assertEqual(report["probes"][0]["probe"], "scalar_probe")
            self.assertIn("residual_cycles", report["probes"][0])

    def test_all_repository_probe_dags_parse_with_synthetic_parameters(self):
        probe_root = MODEL_ROOT.parent / "calibration" / "probe_dags"
        params = {
            "num_splits": 4, "max_splits": 32, "bucket": 32, "d_v": 512,
            "batch": 8, "num_sm_parts": 16, "actual_cta_records": 16,
            "prefix_iters": 10, "tail_iters": 20, "consumer_iters": 30,
        }
        paths = sorted(probe_root.glob("*.json"))
        self.assertEqual(len(paths), 14)
        for path in paths:
            dag = _probe_dag(json.loads(path.read_text()), params)
            self.assertEqual(len(dag.topological_nodes()), len(dag.nodes), path.name)
            for node in dag.nodes.values():
                if node.atom_id.startswith("m64n"):
                    self.assertEqual(node.work_unit, "committed_group")
                    self.assertIn(node.benchmark_params["group_size"], {4, 36})
                    self.assertEqual(
                        node.benchmark_params["depth"],
                        1 if node.benchmark_params["group_size"] == 36 else 4,
                    )
            if path.name.startswith("steady_score_"):
                score_edges = [
                    edge for edge in dag.dependencies
                    if edge.kind == "program"
                    and "score_" in edge.src and "score_" in edge.dst
                ]
                self.assertEqual(len(score_edges), 8)
                self.assertTrue(all(
                    edge.src_event == edge.dst_event == "issue"
                    for edge in score_edges
                ))


class InterfaceTests(unittest.TestCase):
    def test_launch_bounds_minimum_is_not_a_residency_cap(self):
        resources = KernelResources(
            sm_count=1, main_registers_per_thread=0, main_shared_bytes=0,
            main_min_blocks_per_sm=1,
        )
        self.assertEqual(resources.residency("main"), 8)
        legacy = load_kernel_resources({
            "combine_launch_bound": 2,
            "combine_registers_per_thread": 64,
            "combine_shared_bytes": 0,
            "combine_threads": 256,
        })
        self.assertEqual(legacy.combine_min_blocks_per_sm, 2)
        self.assertEqual(legacy.residency("combine"), 4)
        with self.assertRaisesRegex(ValueError, "below the __launch_bounds__"):
            KernelResources(
                combine_registers_per_thread=200,
                combine_min_blocks_per_sm=2,
            ).residency("combine")

    def test_compose_direct_script_help_from_outside_repo(self):
        compose = MODEL_ROOT.parent / "compose.py"
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [sys.executable, str(compose), "--help"], cwd=directory,
                text=True, capture_output=True, check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--microbench-root", completed.stdout)


if __name__ == "__main__":
    unittest.main()
