#!/usr/bin/env python3
"""First-order cycle model for FlashMLA dense BF16 decode on SM90."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def compose(c: dict[str, float], n_page: int, split_kv: bool = False) -> dict[str, float | str]:
    if n_page < 1:
        raise ValueError("n_page must be at least 1")

    qk_first = 36 * c["t_qk_ss"]
    qk_0a = 16 * c["t_qk_ss"]
    qk_0b = 16 * c["t_qk_ss"] + 4 * c["t_qk_rs"]
    qk_1 = 32 * c["t_qk_ss"] + 4 * c["t_qk_rs"]
    pv_local = 4 * c["t_pv_rs"]
    pv_remote = 4 * c["t_pv_ss"]
    k_page = 9 * c["T_tma_k_tile"]
    softmax = c["T_softmax"]
    stmatrix = c["T_stmatrix_p"]

    wg0 = softmax + max(pv_local, softmax) + stmatrix + max(pv_remote, qk_0a) + qk_0b
    wg1 = 2 * softmax + stmatrix + pv_local + pv_remote + qk_1
    pair_compute = max(wg0, wg1)
    pair_load = 2 * k_page
    pair_steady = max(pair_load, pair_compute)
    pair_prologue = max(c["T_qload"], pair_load) + max(qk_first, qk_1)
    pair_drain = max(
        softmax + max(pv_local, softmax) + stmatrix + pv_remote,
        2 * softmax + stmatrix + pv_local + pv_remote,
    )

    n_pair = n_page // 2
    single_kq = c.get("T_kq_page", k_page + qk_first)
    if n_pair:
        main = pair_prologue + max(n_pair - 1, 0) * pair_steady + pair_drain
    else:
        main = 0.0
    if n_page % 2:
        odd_compute = softmax + max(pv_local, stmatrix + pv_remote)
        if n_pair:
            main += single_kq + odd_compute
        else:
            main += max(c["T_qload"], single_kq) + odd_compute

    tail = c["T_output_store"] + c.get("T_reduce_l", 0.0)
    if split_kv:
        tail += c["T_combine"]
    model = main + tail
    serial = (
        c["T_qload"]
        + n_page * (k_page + qk_1 + softmax + pv_local + stmatrix + pv_remote)
        + tail
    )
    return {
        "N_page": float(n_page),
        "N_pair": float(n_pair),
        "T_K_page": k_page,
        "T_pair_load": pair_load,
        "T_pair_compute": pair_compute,
        "T_pair_steady": pair_steady,
        "T_model": model,
        "T_serial": serial,
        "bottleneck": "TMA/load" if pair_load > pair_compute else "compute/softmax",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles-json", required=True, type=Path)
    parser.add_argument("--seqlen-k", type=int, required=True)
    parser.add_argument("--split-kv", action="store_true")
    args = parser.parse_args()
    cycles = json.loads(args.cycles_json.read_text(encoding="utf-8"))
    result = compose(cycles, math.ceil(args.seqlen_k / 64), args.split_kv)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
