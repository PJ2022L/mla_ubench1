#!/usr/bin/env python3
"""Regenerate manifest.json from the explicit generic benchmark registry."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

WGMMA = {
    "m64n64k16_ss_bf16": (64, "ss", "bf16"),
    "m64n64k16_rs_bf16": (64, "rs", "bf16"),
    "m64n256k16_ss_bf16": (256, "ss", "bf16"),
    "m64n256k16_rs_bf16": (256, "rs", "bf16"),
    "m64n64k16_ss_fp16": (64, "ss", "fp16"),
    "m64n64k16_rs_fp16": (64, "rs", "fp16"),
    "m64n256k16_ss_fp16": (256, "ss", "fp16"),
    "m64n256k16_rs_fp16": (256, "rs", "fp16"),
}

TARGETS = {
    "ex2_approx_ftz_f32": (r"ex2\.approx\.ftz\.f32", r"MUFU\.EX2"),
    "lg2_approx_ftz_f32": (r"lg2\.approx\.ftz\.f32", r"MUFU\.LG2"),
    "rcp_approx_ftz_f32": (r"rcp\.approx\.ftz\.f32", r"MUFU\.RCP"),
    "ffma_rn_ftz_f32": (r"fma\.rn\.ftz\.f32", r"FFMA"),
    "fadd_rn_ftz_f32": (r"add\.rn\.ftz\.f32", r"FADD"),
    "fmul_rn_ftz_f32": (r"mul\.rn\.ftz\.f32", r"FMUL"),
    "max_ftz_f32": (r"max\.ftz\.f32", r"FMNMX"),
    "fsetp_lt_ftz_f32": (r"setp\.lt\.ftz\.f32", r"FSETP"),
    "fsel_f32": (r"selp\.f32", r"FSEL"),
    "cvt_rn_bf16_f32": (r"cvt\.rn\.bf16\.f32", r"F2F(P)?\.BF16\.F32"),
    "cvt_rn_f16_f32": (r"cvt\.rn\.f16\.f32", r"F2FP\.F16\.F32"),
    "shfl_sync_bfly_b32": (r"shfl\.sync\.bfly\.b32", r"SHFL\.BFLY"),
    "iadd3_u32": (r"add\.u32", r"IADD3"),
    "imad_lo_u32": (r"mad\.lo\.u32", r"IMAD"),
    "isetp_lt_u32": (r"setp\.lt\.u32", r"ISETP"),
    "div_u32": (r"div\.u32", r"(MUFU|IMAD|IADD3)"),
    "rem_u32": (r"rem\.u32", r"(MUFU|IMAD|IADD3)"),
    "bar_sync_128": (r"bar\.sync.*128", r"BAR\.SYNC"),
    "bar_sync_256": (r"bar\.sync.*256", r"BAR\.SYNC"),
    "bar_arrive_2wg": (r"bar\.arrive", r"BAR\.ARV"),
    "warp_sync": (r"bar\.warp\.sync", ""),
    "mbarrier_init": (r"mbarrier\.init", r"SYNCS"),
    "mbarrier_expect_tx": (r"mbarrier\.expect_tx", r"SYNCS"),
    "mbarrier_wait_128": (r"mbarrier\.try_wait", r"SYNCS"),
    "mbarrier_wait_256": (r"mbarrier\.try_wait", r"SYNCS"),
    "fence_proxy_async_shared_cta": (r"fence\.proxy\.async\.shared::cta", r"FENCE\.VIEW\.ASYNC"),
    "tensor_4d_64x64_bf16": (r"cp\.async\.bulk\.tensor\.4d.*shared::cluster\.global", r"UTMALDG\.4D"),
    "tensor_4d_64x576_bf16": (r"cp\.async\.bulk\.tensor\.4d.*shared::cluster\.global", r"UTMALDG\.4D"),
    "tensor_4d_64x512_bf16": (r"cp\.async\.bulk\.tensor\.4d.*global\.shared::cta", r"UTMASTG\.4D"),
    "tensor_4d_64x64_fp16": (r"cp\.async\.bulk\.tensor\.4d.*shared::cluster\.global", r"UTMALDG\.4D"),
    "tensor_4d_64x576_fp16": (r"cp\.async\.bulk\.tensor\.4d.*shared::cluster\.global", r"UTMALDG\.4D"),
    "tensor_4d_64x512_fp16": (r"cp\.async\.bulk\.tensor\.4d.*global\.shared::cta", r"UTMASTG\.4D"),
    "cp_async_bulk_s2g_64x512_f32": (r"cp\.async\.bulk\.global\.shared::cta", r"UBLKCP\.G\.S"),
    "stmatrix_m64n64_b16_x4": (r"stmatrix\.sync\.aligned\.x4", r"STSM"),
    "stmatrix_m64n256_b16_x4": (r"stmatrix\.sync\.aligned\.x4", r"STSM"),
    "ldmatrix_m64n64_b16_x4": (r"ldmatrix\.sync\.aligned\.x4", r"LDSM"),
    "ld_shared_u32_patterns": (r"ld\.shared\.u32", r"LDS"),
    "st_shared_u32": (r"st\.shared\.u32", r"STS"),
    "st_shared_v2_u32_stride520": (r"st\.shared\.v2\.u32", r"STS\.64"),
    "st_shared_u64_sw128": (r"st\.shared\.u64", r"STS\.64"),
    "ld_global_nc_u32": (r"ld\.global\.nc\.u32", r"LDG"),
    "ld_global_u32": (r"ld\.global\.u32", r"LDG"),
    "ld_global_v4_u32_32b": (r"ld\.global\.v4\.u32", r"LDG"),
    "ld_global_v4_f32": (r"ld\.global\.v4\.f32", r"LDG"),
    "ld_global_f32_strided": (r"ld\.global\.f32", r"LDG"),
    "st_global_f32": (r"st\.global\.f32", r"STG"),
    "st_global_v4_u32_32b": (r"st\.global\.v4\.u32", r"STG"),
    "st_global_u32": (r"st\.global\.u32", r"STG"),
    "st_global_u64": (r"st\.global\.u64", r"STG"),
    "prefetch_tensormap_rank4": (r"prefetch\.tensormap", r"UTMACCTL\.PF"),
    "griddepcontrol_launch_dependents": (r"griddepcontrol\.launch_dependents", r"PREEXIT"),
    "griddepcontrol_wait": (r"griddepcontrol\.wait", r"ACQBULK"),
    "griddepcontrol_producer_consumer": (
        r"griddepcontrol\.(launch_dependents|wait)",
        r"(PREEXIT|ACQBULK)",
    ),
    "ld_global_u64_saturation": (r"ld\.global\.u64", r"LDG"),
    "st_global_u64_saturation": (r"st\.global\.u64", r"STG"),
    "tma_load_4d_service": (
        r"cp\.async\.bulk\.tensor\.4d.*shared::cluster\.global",
        r"UTMALDG\.4D",
    ),
    "wgmma_mixed_shape_mode": (
        r"wgmma\.mma_async.*m64n(64|256)k16\.f32\.bf16",
        r"HGMMA\.64x(64|256)x16\.F32\.BF16",
    ),
    "wgmma_tma_interference": (
        r"(wgmma\.mma_async|cp\.async\.bulk\.tensor\.4d)",
        r"(HGMMA|UTMALDG)",
    ),
    "wgmma_sfu_shared_interference": (
        r"(wgmma\.mma_async|ex2\.approx\.ftz\.f32|ld\.shared\.u32)",
        r"(HGMMA|MUFU\.EX2|LDS)",
    ),
}

PARAMETERS = {
    "wgmma": ["warpgroups", "group_size", "depth", "blocks"],
    "sfu": ["blocks"],
    "fp32_alu": ["blocks"],
    "convert": ["blocks"],
    "integer": ["blocks"],
    "shuffle": ["delta", "blocks"],
    "synchronization": ["blocks"],
    "ordering": ["blocks"],
    "tma_load": ["depth", "working_set_pages", "pattern", "blocks"],
    "tma_store": ["depth", "working_set_tiles", "blocks"],
    "bulk_store": ["working_set_tiles", "pattern", "blocks"],
    "stmatrix": ["warpgroups", "blocks"],
    "ldmatrix": ["warpgroups", "blocks"],
    "matrix_movement": ["warpgroups", "blocks"],
    "shared_load": ["threads", "pattern", "working_set_words", "blocks"],
    "shared_store": ["blocks"],
    "global_load": ["blocks"],
    "global_store": ["blocks"],
    "tensormap_prefetch": ["mode", "working_pages", "working_tiles", "blocks"],
    "pdl": ["blocks"],
    "memory_service": ["working_set_bytes", "pattern", "cache_mode",
                       "outstanding_depth", "threads", "blocks"],
    "tma_service": ["depth", "working_set_pages", "pattern", "cache_mode", "blocks"],
    "interference": ["actors", "blocks"],
}

PARAMETERS_BY_ID = {
    "wgmma_tma_interference": ["actors", "working_set_pages", "blocks"],
    "st_shared_u32": [
        "threads", "producers", "topology", "working_set_words", "blocks"
    ],
    "st_shared_v2_u32_stride520": [
        "warpgroups", "stores_per_thread", "invalid_tokens", "blocks"
    ],
    "st_shared_u64_sw128": [
        "warpgroups", "stores_per_thread", "invalid_tokens", "blocks"
    ],
    "ld_global_nc_u32": [
        "threads", "issuers", "pattern", "working_set_entries", "blocks"
    ],
    "ld_global_u32": [
        "threads", "issuers", "pattern", "working_set_entries", "blocks"
    ],
    "ld_global_v4_u32_32b": [
        "threads", "issuers", "pattern", "working_set_records", "blocks"
    ],
    "ld_global_v4_f32": [
        "segments", "rowsets", "warps", "vectors_per_thread", "pattern",
        "blocks"
    ],
    "ld_global_f32_strided": [
        "segments", "split_stride", "rowsets", "warps", "pattern", "blocks"
    ],
    "st_global_f32": [
        "lane_mode", "working_set_records", "pattern", "blocks"
    ],
    "st_global_v4_u32_32b": [
        "working_set_records", "pattern", "blocks"
    ],
    "st_global_u32": [
        "producers", "working_set_records", "pattern", "blocks"
    ],
    "st_global_u64": [
        "dtype", "warps", "vectors_per_thread", "working_set_records",
        "pattern", "blocks"
    ],
    "griddepcontrol_launch_dependents": ["blocks"],
    "griddepcontrol_wait": ["blocks"],
    "griddepcontrol_producer_consumer": [
        "producer_blocks", "consumer_blocks", "prefix_iters", "suffix_iters",
        "consumer_iters"
    ],
}


def entry(source: Path) -> dict[str, object]:
    relative = source.relative_to(ROOT)
    category, family = relative.parts[:2]
    atom_id = source.stem
    if atom_id in WGMMA:
        n, mode, dtype = WGMMA[atom_id]
        ptx_dtype = "bf16" if dtype == "bf16" else "f16"
        ptx_opcode = (
            rf"wgmma\.mma_async[^\n]*m64n{n}k16\.f32\."
            rf"{ptx_dtype}\.{ptx_dtype}"
        )
        if mode == "rs":
            target_ptx = (
                ptx_opcode
                + r"[^\n]*\},\s*\{%r[0-9]+(?:,\s*%r[0-9]+){3}\},"
                + r"\s*%rd[0-9]+,\s*p"
            )
        else:
            target_ptx = (
                ptx_opcode
                + r"[^\n]*\},\s*%rd[0-9]+,\s*%rd[0-9]+,\s*p"
            )
        sass_opcode = rf"HGMMA\.64x{n}x16\.F32" + (
            r"\.BF16" if dtype == "bf16" else r"(?!\.BF16)"
        )
        target_sass = sass_opcode + (
            r"\s+R[0-9]+,\s*R[0-9]+,\s*gdesc\["
            if mode == "rs"
            else r"\s+R[0-9]+,\s*gdesc\["
        )
        protocol = {
            "latency_boundary": (
                "selected group_size issue + commit_group + wait_group 0; "
                "dependency depth 1; matched inline-PTX loop control subtracted"
            ),
            "initiation_interval_boundary": (
                "cycles per committed group at selected group_size and depth; "
                "matched inline-PTX loop control subtracted"
            ),
            "accumulator_dtype": "f32",
            "a_major": "k",
            "b_major": "k" if n == 64 else "transposed_k",
            "swizzle": "128B",
            "transpose": n == 256,
            "scale_modifier": "1x",
            "source_mode": mode,
            "m": 64,
            "n": n,
            "k": 16,
            "input_dtype": dtype,
            "supported_group_sizes": (
                [1, 4, 36] if n == 64 and mode == "ss" else [1, 4]
            ),
            "group_size_depth_constraints": (
                {"36": [1]} if n == 64 and mode == "ss" else {}
            ),
        }
        support_ptx = [r"wgmma\.fence", r"wgmma\.commit_group", r"wgmma\.wait_group", r"fence\.proxy\.async"]
    else:
        if atom_id not in TARGETS:
            raise SystemExit(f"unregistered .cu source: {relative}")
        target_ptx, target_sass = TARGETS[atom_id]
        protocol = {"latency_boundary": "defined by family harness"}
        support_ptx = []
        if family == "sfu":
            support_ptx = [r"fma\.rn\.ftz\.f32"]
            protocol["support_opcode_accounting"] = "FFMA stabilizer/baseline is excluded from target count"
        if family in {"synchronization", "ordering"}:
            support_ptx.append(r"bar\.sync")
        if atom_id == "mbarrier_init":
            support_ptx.append(r"mbarrier\.inval")
        elif atom_id == "mbarrier_expect_tx":
            support_ptx.extend([
                r"mbarrier\.init",
                r"mbarrier\.complete_tx",
            ])
        elif atom_id in {"mbarrier_wait_128", "mbarrier_wait_256"}:
            support_ptx.extend([
                r"mbarrier\.init",
                r"mbarrier\.arrive",
            ])
        if family in {"tma_load", "tma_service"}:
            support_ptx.extend([
                r"mbarrier\.init",
                r"mbarrier\.arrive\.expect_tx",
                r"mbarrier\.try_wait",
            ])
            protocol = {
                "latency_boundary": (
                    "TMA issue + expect_tx + completion wait; matched page/address "
                    "selection, loop, shared-consume, and sink baseline subtracted"
                ),
                "initiation_interval_boundary": (
                    "cycles per logical operation at selected depth and block count; "
                    "matched non-TMA baseline subtracted"
                ),
                "support_opcode_accounting": (
                    "mbarrier expect/wait belongs to the TMA load protocol"
                ),
            }
        elif family in {"tma_store", "bulk_store"}:
            support_ptx.extend([
                r"fence\.proxy\.async\.shared::cta",
                r"cp\.async\.bulk\.commit_group",
                r"cp\.async\.bulk\.wait_group",
            ])
            protocol = {
                "latency_boundary": (
                    "asynchronous store issue + commit_group + wait_group 0; "
                    "matched address/loop/sink baseline subtracted"
                ),
                "initiation_interval_boundary": (
                    "cycles per logical tile at selected depth/block count; "
                    "matched non-store baseline subtracted"
                ),
                "support_opcode_accounting": (
                    "commit/wait are timed protocol support; proxy fence is "
                    "untimed source-visibility setup"
                ),
            }
        if family == "tensormap_prefetch":
            protocol = {
                "latency_boundary": (
                    "selected descriptor-prefetch round; matched descriptor "
                    "selection and loop baseline subtracted"
                ),
                "initiation_interval_boundary": (
                    "baseline-subtracted cycles per selected prefetch round"
                ),
            }
        elif family == "ordering":
            protocol = {
                "latency_boundary": (
                    "one ordering fence; matched unroll and loop-control "
                    "baseline subtracted"
                ),
                "initiation_interval_boundary": (
                    "baseline-subtracted cycles per fence"
                ),
                "support_opcode_accounting": (
                    "bar.sync instructions are outside the clock64 boundary"
                ),
            }
        elif family == "synchronization":
            protocol = {
                "latency_boundary": (
                    "complete synchronization protocol; matched unroll and "
                    "loop-control baseline subtracted"
                ),
                "initiation_interval_boundary": (
                    "baseline-subtracted cycles per protocol operation"
                ),
                "support_opcode_accounting": (
                    "arrive/complete/barrier instructions required for a valid "
                    "generation remain in the target protocol"
                ),
            }
            if atom_id == "warp_sync":
                protocol["latency_boundary"] = (
                    "SM90a ptxas elides converged full-mask BAR.WARP.SYNC; "
                    "formal executable cost is zero"
                )
        elif family == "pdl":
            if atom_id == "griddepcontrol_producer_consumer":
                protocol = {
                    "latency_boundary": (
                        "grid-level producer suffix / consumer wait overlap; "
                        "resource curve only, not an operation latency"
                    ),
                    "resource_scope": "producer_consumer_grid_pair",
                }
            else:
                protocol = {
                    "latency_boundary": (
                        "ready-path griddepcontrol instruction; matched "
                        "programmatic-launch loop baseline subtracted"
                    ),
                    "initiation_interval_boundary": (
                        "baseline-subtracted cycles per instruction"
                    ),
                }
        if family == "matrix_movement":
            tile_n = 256 if "m64n256" in atom_id else 64
            instructions_per_warp = 16 if tile_n == 256 else 4
            protocol = {
                "work_unit": "m64_tile",
                "tile_m": 64,
                "tile_n": tile_n,
                "instructions_per_warp": instructions_per_warp,
                "warp_instructions_per_tile": 4 * instructions_per_warp,
                "latency_boundary": (
                    "complete m64 tile movement; LDSM uses dependency completion "
                    "plus remaining issue span, STSM uses issue span and leaves "
                    "visibility to the ordering atom"
                ),
                "initiation_interval_boundary": (
                    "cycles per complete m64 tile with matched inline-PTX loop "
                    "control subtracted"
                ),
            }
        elif family == "shared_load":
            protocol = {
                "latency_boundary": (
                    "target LDS loop minus a separately compiled matched "
                    "address/pattern/checksum/control baseline"
                ),
                "throughput_boundary": (
                    "CUDA-event target kernel only; baseline is never launched "
                    "inside the event interval"
                ),
                "latency_target_kernel": "shared_load_u32_target_kernel",
                "latency_baseline_kernel": "shared_load_u32_baseline_kernel",
                "throughput_target_kernel": "shared_load_u32_target_kernel",
                "baseline_required_sass_patterns": [r"LOP3", r"BAR\.SYNC"],
            }
        elif family == "shared_store":
            kernel_stem = (
                "shared_store_u32" if atom_id == "st_shared_u32"
                else "shared_store_8b"
            )
            protocol = {
                "latency_boundary": (
                    "target STS issue loop minus a separately compiled matched "
                    "layout/address/value/predicate/control baseline"
                ),
                "throughput_boundary": (
                    "CUDA-event target-only kernel with no baseline/checksum "
                    "work in the timed grid"
                ),
                "latency_target_kernel": f"{kernel_stem}_target_kernel",
                "latency_baseline_kernel": f"{kernel_stem}_baseline_kernel",
                "throughput_target_kernel": f"{kernel_stem}_throughput_kernel",
                "baseline_required_sass_patterns": [r"LOP3", r"BAR\.SYNC"],
            }
            if atom_id == "st_shared_u64_sw128":
                protocol["forbidden_target_ptx_patterns"] = [
                    r"st\.shared\.v2\.u32"
                ]
            elif atom_id == "st_shared_v2_u32_stride520":
                protocol["forbidden_target_ptx_patterns"] = [
                    r"st\.shared\.u64"
                ]
        if family in {"global_load", "global_store"}:
            kernel_markers = {
                "ld_global_nc_u32": "global_load_i32_cached_kernel",
                "ld_global_u32": "global_load_i32_cached_kernel",
                "ld_global_v4_u32_32b": "global_load_record_32b_kernel",
                "ld_global_v4_f32": "global_load_v4_f32_kernel",
                "ld_global_f32_strided": "global_load_f32_strided_kernel",
                "st_global_f32": "global_store_f32_kernel",
                "st_global_v4_u32_32b": "global_store_record_32b_kernel",
                "st_global_u32": "global_store_u32_kernel",
                "st_global_u64": "global_store_u64_kernel",
            }
            protocol = {
                "latency_boundary": (
                    "target global-memory issue loop minus a separately "
                    "compiled matched address/value/predicate/checksum/control "
                    "baseline"
                ),
                "throughput_boundary": (
                    "CUDA-event target specialization only; baseline is never "
                    "launched inside the event interval"
                ),
                "timed_kernel_marker": kernel_markers[atom_id],
                "baseline_required_sass_patterns": [
                    r"BAR\.SYNC", r"(IADD3|IMAD)"
                ],
            }
        if family in {"tma_load", "tma_service"}:
            protocol["timed_kernel_marker"] = "tma_load_kernel"
        if family == "tma_service":
            protocol["source_page_bytes"] = 64 * 576 * 2
        if family == "memory_service":
            protocol.update({
                "timed_kernel_marker": "memory_service6kernel",
                "resource_scope": "active_grid_memory_service",
            })
        if family == "interference":
            protocol.update({
                "timed_kernel_marker": {
                    "wgmma_mixed_shape_mode": "mixed_wgmma_kernel",
                    "wgmma_tma_interference": "wgmma_tma_kernel",
                    "wgmma_sfu_shared_interference":
                        "wgmma_sfu_shared_kernel",
                }[atom_id],
                "matched_actor_footprint": (
                    "actors=1 and actors=2 use identical 256-thread and "
                    "dynamic-shared-memory launch footprints"
                ),
            })
        if family == "pdl":
            if atom_id == "griddepcontrol_producer_consumer":
                protocol["producer_kernel_marker"] = "producer_kernel"
                protocol["consumer_kernel_marker"] = "consumer_kernel"
            else:
                protocol["timed_kernel_marker"] = "pdl_atomic6kernel"
    item: dict[str, object] = {
        "id": atom_id,
        "kind": ("resource_curve"
                 if atom_id in {"griddepcontrol_producer_consumer",
                                "ld_global_u64_saturation",
                                "st_global_u64_saturation",
                                "tma_load_4d_service",
                                "wgmma_mixed_shape_mode",
                                "wgmma_tma_interference",
                                "wgmma_sfu_shared_interference"}
                 else "operation"),
        "category": category,
        "family": family,
        "source": relative.as_posix(),
        "binary": atom_id,
        "result_csv": f"{category}/{family}/result.csv",
        "parameters": PARAMETERS_BY_ID.get(
            atom_id, PARAMETERS.get(family, ["blocks"])),
        "target_ptx_patterns": [target_ptx],
        "support_ptx_patterns": support_ptx,
        "target_sass_patterns": [target_sass] if target_sass else [],
        "protocol": protocol,
    }
    if atom_id in WGMMA:
        item["source_mode"] = mode
        item["fixed_modifiers"] = {
            "accumulator_dtype": "f32",
            "a_major": "register" if mode == "rs" else "k_major",
            "b_major": "k_major" if n == 64 else "transposed_k",
            "swizzle": "128B",
            "transpose_a": False,
            "transpose_b": n == 256,
            "scale_a": 1,
            "scale_b": 1,
            "scale_d": "predicate",
        }
    if atom_id == "warp_sync":
        item["sass_elided"] = True
    if atom_id == "wgmma_mixed_shape_mode":
        item["target_ptx_patterns"] = [
            r"wgmma\.mma_async.*m64n64k16\.f32\.bf16",
            r"wgmma\.mma_async.*m64n256k16\.f32\.bf16",
        ]
        item["target_sass_patterns"] = [
            r"HGMMA\.64x64x16\.F32\.BF16",
            r"HGMMA\.64x256x16\.F32\.BF16",
        ]
    elif atom_id == "wgmma_tma_interference":
        item["target_ptx_patterns"] = [
            r"wgmma\.mma_async.*m64n64k16\.f32\.bf16",
            r"cp\.async\.bulk\.tensor\.4d.*shared::cluster\.global",
        ]
        item["target_sass_patterns"] = [r"HGMMA", r"UTMALDG"]
    elif atom_id == "wgmma_sfu_shared_interference":
        item["target_ptx_patterns"] = [
            r"wgmma\.mma_async.*m64n64k16\.f32\.bf16",
            r"ex2\.approx\.ftz\.f32",
            r"ld\.shared\.u32",
        ]
        item["target_sass_patterns"] = [r"HGMMA", r"MUFU\.EX2", r"LDS"]
    elif atom_id == "griddepcontrol_producer_consumer":
        item["target_ptx_patterns"] = [
            r"griddepcontrol\.launch_dependents",
            r"griddepcontrol\.wait",
        ]
        item["target_sass_patterns"] = [r"PREEXIT", r"ACQBULK"]
    return item


def main() -> int:
    sources = sorted(
        source
        for category in ("compute", "memory", "resource")
        for source in (ROOT / category).glob("*/*.cu")
    )
    benchmarks = [entry(source) for source in sources]
    identifiers = [item["id"] for item in benchmarks]
    if len(identifiers) != len(set(identifiers)):
        raise SystemExit("manifest IDs must be globally unique")
    document = {
        "schema_version": 2,
        "target": {"gpu": "NVIDIA H800", "architecture": "sm_90a"},
        "identity_rule": "id == source stem == binary == JSON name",
        "result_policy": "latest complete accepted full sweep; atomic replacement",
        "benchmarks": benchmarks,
    }
    output = ROOT / "manifest.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n",
                         encoding="utf-8")
    temporary.replace(output)
    print(f"wrote {output} with {len(benchmarks)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
