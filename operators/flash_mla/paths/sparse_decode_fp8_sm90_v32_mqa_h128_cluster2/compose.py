#!/usr/bin/env python3
"""Compose instruction-level cycle measurements into the SM90 sparse-decode model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


QK_ISSUES = 576 // 16
PV_ISSUES = 64 // 16


def producer_cycles(cycles: dict[str, float]) -> float:
    """Return one producer block in cycles from either a direct or decomposed value."""
    if "T_producer" in cycles:
        return cycles["T_producer"]
    return sum(
        cycles[key]
        for key in (
            "T_ld_block",
            "T_cvt_block",
            "T_st_shared_block",
            "T_st_dsm_block",
        )
    )


def compose(cycles: dict[str, float], n_block: int, split_kv: bool = False) -> dict[str, float | str | bool]:
    """Return the two-buffer pipeline prediction and diagnostics.

    Every input uses cycles per CTA at the granularity stated by its key. WGMMA
    inputs are cycles per instruction; producer inputs are cycles per 64-token block.
    """
    if n_block < 1:
        raise ValueError("n_block must be at least 1")

    t_producer = producer_cycles(cycles)
    t_qk = QK_ISSUES * cycles["T_wgmma_qk_ss"]
    t_pv_local = PV_ISSUES * cycles["T_wgmma_pv_rs"]
    t_pv_remote = (
        cycles.get("T_remote_handoff", 0.0)
        + cycles.get("T_remote_scale", 0.0)
        + PV_ISSUES * cycles["T_wgmma_pv_ss"]
    )
    t_consumer = t_qk + cycles["T_softmax"] + max(t_pv_local, t_pv_remote)
    t_first_ready = max(cycles.get("T_qload", 0.0), t_producer)
    t_steady = max(t_producer, t_consumer)
    t_tail = cycles["T_tma_store"]
    if split_kv:
        t_tail += cycles["T_splitkv_reduce"]

    t_model = t_first_ready + (n_block - 1) * t_steady + t_consumer + t_tail
    t_serial = (
        cycles.get("T_qload", 0.0)
        + n_block * (t_producer + t_consumer)
        + t_tail
    )
    result: dict[str, float | str | bool] = {
        "T_producer": t_producer,
        "T_qk": t_qk,
        "T_pv_local": t_pv_local,
        "T_pv_remote": t_pv_remote,
        "T_consumer": t_consumer,
        "T_first_ready": t_first_ready,
        "T_steady": t_steady,
        "T_tail": t_tail,
        "T_model": t_model,
        "T_serial": t_serial,
        "bottleneck": "producer(memory/convert)" if t_producer > t_consumer else "consumer(tensor/softmax)",
    }
    if "T_measured" in cycles:
        measured = cycles["T_measured"]
        result["rho"] = t_model / measured
        result["within_bounds"] = t_model <= measured <= t_serial
    return result


def crossover_gain(cycles: dict[str, float]) -> float | None:
    if "T_producer_nocross" not in cycles:
        return None
    return cycles["T_producer_nocross"] - producer_cycles(cycles)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles-json", required=True, type=Path)
    parser.add_argument("--n-block", type=int, default=32)
    parser.add_argument("--split-kv", action="store_true")
    args = parser.parse_args()

    cycles = json.loads(args.cycles_json.read_text(encoding="utf-8"))
    result = compose(cycles, args.n_block, args.split_kv)
    gain = crossover_gain(cycles)
    if gain is not None:
        result["crossover_gain"] = gain
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
