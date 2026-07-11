#!/usr/bin/env python3
"""Compose instruction-level cycle measurements into the SM90 sparse-decode model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping


QK_ISSUES = 576 // 16
PV_ISSUES = 64 // 16


def _cost(cycles: Mapping[str, float], key: str) -> float:
    if key not in cycles:
        raise KeyError(f"missing cycle cost: {key}")
    value = float(cycles[key])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{key} must be a finite non-negative number")
    return value


def _optional(cycles: Mapping[str, float], key: str, default: float = 0.0) -> float:
    if key not in cycles:
        return default
    return _cost(cycles, key)


def _aliased_optional(
    cycles: Mapping[str, float], key: str, legacy_key: str, default: float = 0.0
) -> float:
    if key in cycles:
        return _cost(cycles, key)
    return _optional(cycles, legacy_key, default)


def producer_cycles(cycles: Mapping[str, float]) -> float:
    """Return one producer block in cycles from either a direct or decomposed value."""
    if "T_producer_direct" in cycles:
        return _cost(cycles, "T_producer_direct")
    if "T_producer" in cycles:
        return _cost(cycles, "T_producer")
    return sum(
        _cost(cycles, key)
        for key in (
            "T_ld_block",
            "T_cvt_block",
            "T_st_shared_block",
            "T_st_dsm_block",
        )
    )


def compose(
    cycles: Mapping[str, float], n_block: int, split_kv: bool = False
) -> dict[str, float | str | bool]:
    """Return one scheduler-segment prediction and diagnostics.

    Every input uses cycles per CTA at the granularity stated by its key. WGMMA
    inputs are cycles per instruction; producer inputs are cycles per 64-token block.
    This function does not scale a persistent partition or the independent combine grid.
    """
    if n_block < 1:
        raise ValueError("n_block must be at least 1")

    t_producer = producer_cycles(cycles)
    t_qk = QK_ISSUES * _cost(cycles, "T_wgmma_qk_ss")
    t_pv_local = PV_ISSUES * _cost(cycles, "T_wgmma_pv_rs")
    t_pv_score = (
        _aliased_optional(cycles, "T_handoff", "T_remote_handoff")
        + _aliased_optional(cycles, "T_score_scale", "T_remote_scale")
        + PV_ISSUES * _cost(cycles, "T_wgmma_pv_ss")
    )
    if "T_pv_dual_wg" in cycles:
        t_pv_stage = _cost(cycles, "T_pv_dual_wg")
    else:
        t_pv_stage = max(
            t_pv_local, t_pv_score, _optional(cycles, "T_pv_aggregate_floor")
        )
    t_consumer = t_qk + _cost(cycles, "T_softmax") + t_pv_stage
    t_qload = _optional(cycles, "T_qload")
    t_first_ready = max(t_qload, t_producer)
    t_steady = max(t_producer, t_consumer)
    if split_kv:
        try:
            t_epilogue = _cost(cycles, "T_partial_store")
        except KeyError as exc:
            raise KeyError("split-KV requires T_partial_store (FP32 bulk partial output)") from exc
        epilogue_kind = "split_fp32_partial"
    else:
        if "T_output_store" in cycles:
            t_epilogue = _cost(cycles, "T_output_store")
        elif "T_tma_store" in cycles:
            t_epilogue = _cost(cycles, "T_tma_store")
        else:
            raise KeyError("non-split requires T_output_store or legacy T_tma_store")
        epilogue_kind = "direct_bf16_output"

    t_model = t_first_ready + (n_block - 1) * t_steady + t_consumer + t_epilogue
    t_serial = (
        t_qload
        + n_block * (t_producer + t_consumer)
        + t_epilogue
    )
    result: dict[str, float | str | bool] = {
        "T_producer": t_producer,
        "T_qk": t_qk,
        "T_pv_local": t_pv_local,
        "T_pv_score": t_pv_score,
        "T_pv_stage": t_pv_stage,
        "T_consumer": t_consumer,
        "T_first_ready": t_first_ready,
        "T_steady": t_steady,
        "T_epilogue": t_epilogue,
        "epilogue_kind": epilogue_kind,
        "combine_included": False,
        "T_model": t_model,
        "T_serial": t_serial,
        "bottleneck": "producer(memory/convert)" if t_producer > t_consumer else "consumer(tensor/softmax)",
    }
    if "T_measured" in cycles:
        measured = _cost(cycles, "T_measured")
        if measured == 0:
            raise ValueError("T_measured must be positive")
        result["rho"] = t_model / measured
        result["within_bounds"] = t_model <= measured <= t_serial
    return result


def crossover_gain(cycles: Mapping[str, float]) -> float | None:
    if "T_producer_nocross" not in cycles:
        return None
    return _cost(cycles, "T_producer_nocross") - producer_cycles(cycles)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles-json", required=True, type=Path)
    parser.add_argument("--n-block", type=int, default=32)
    parser.add_argument(
        "--split-kv",
        action="store_true",
        help="select the FP32 partial-output epilogue; combine-grid time is not included",
    )
    args = parser.parse_args(argv)

    cycles = json.loads(args.cycles_json.read_text(encoding="utf-8"))
    if not isinstance(cycles, dict):
        raise ValueError("cycles JSON must contain an object")
    result = compose(cycles, args.n_block, args.split_kv)
    gain = crossover_gain(cycles)
    if gain is not None:
        result["crossover_gain"] = gain
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
