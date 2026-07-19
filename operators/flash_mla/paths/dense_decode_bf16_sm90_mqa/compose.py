#!/usr/bin/env python3
"""Compose per-CTA cycle costs for FlashMLA dense BF16 decode on SM90a."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Mapping


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _cost(costs: Mapping[str, float], key: str) -> float:
    if key not in costs:
        raise KeyError(f"missing cycle cost: {key}")
    value = float(costs[key])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{key} must be a finite non-negative number")
    return value


def _optional(costs: Mapping[str, float], key: str, default: float = 0.0) -> float:
    if key not in costs:
        return default
    return _cost(costs, key)


def _epilogue(costs: Mapping[str, float], split_kv: bool) -> tuple[float, float]:
    preferred = "T_output_store_split" if split_kv else "T_output_store_nosplit"
    if preferred in costs:
        store = _cost(costs, preferred)
    else:
        store = _cost(costs, "T_output_store")
    return store, _optional(costs, "T_reduce_l")


def _additive_e2e(
    costs: Mapping[str, float], main: float, split_kv: bool
) -> tuple[float | None, bool]:
    key = "T_combine" if split_kv else "T_combine_noop"
    if key not in costs:
        return None, False
    return main + _cost(costs, key), True


def _compose_schedule(
    costs: Mapping[str, float], n_page: int, split_kv: bool
) -> dict[str, float | str | bool | None]:
    n_pair = n_page // 2
    if n_page == 1:
        body = _cost(costs, "T_prologue_single") + _cost(costs, "T_single_drain")
        transition_count = 0
        tail_kind = "single"
    elif n_page % 2 == 0:
        transition_count = n_pair - 1
        transition = (
            transition_count * _cost(costs, "T_pair_transition")
            if transition_count
            else 0.0
        )
        body = (
            _cost(costs, "T_prologue_pair")
            + transition
            + _cost(costs, "T_pair_drain")
        )
        tail_kind = "pair"
    else:
        transition_count = n_pair - 1
        transition = (
            transition_count * _cost(costs, "T_pair_transition")
            if transition_count
            else 0.0
        )
        body = (
            _cost(costs, "T_prologue_pair")
            + transition
            + _cost(costs, "T_pair_to_single")
            + _cost(costs, "T_single_drain")
        )
        tail_kind = "pair_to_single"

    output_store, reduce_l = _epilogue(costs, split_kv)
    main = body + output_store + reduce_l
    e2e, combine_included = _additive_e2e(costs, main, split_kv)
    return {
        "model_kind": "measured_schedule",
        "N_page": float(n_page),
        "N_pair": float(n_pair),
        "N_pair_transition": float(transition_count),
        "tail_kind": tail_kind,
        "T_body": body,
        "T_output_store": output_store,
        "T_reduce_l": reduce_l,
        "T_main_model": main,
        "T_model": main,
        "T_e2e_additive": e2e,
        "combine_included": combine_included,
        "split_kv": split_kv,
    }


def _compose_atoms(
    costs: Mapping[str, float], n_page: int, split_kv: bool
) -> dict[str, float | str | bool | None]:
    qk_first = 36 * _cost(costs, "t_qk_ss")
    qk_steady = 32 * _cost(costs, "t_qk_ss") + 4 * _cost(costs, "t_qk_rs")
    pv_local = 4 * _cost(costs, "t_pv_rs")
    pv_remote = 4 * _cost(costs, "t_pv_ss")
    k_page = 9 * _cost(costs, "T_tma_k_tile")

    kq_first_page = _optional(costs, "T_kq_first_page", k_page + qk_first)
    kq_steady_page = _optional(costs, "T_kq_steady_page", k_page + qk_steady)
    softmax_legacy = _cost(costs, "T_softmax") if "T_softmax" in costs else None
    if "T_softmax_even" in costs:
        softmax_even = _cost(costs, "T_softmax_even")
    elif softmax_legacy is not None:
        softmax_even = softmax_legacy
    else:
        raise KeyError("missing cycle cost: T_softmax_even (or legacy T_softmax)")
    if "T_softmax_odd" in costs:
        softmax_odd = _cost(costs, "T_softmax_odd")
    elif softmax_legacy is not None:
        softmax_odd = softmax_legacy
    else:
        raise KeyError("missing cycle cost: T_softmax_odd (or legacy T_softmax)")
    softmax_empty_odd = _optional(costs, "T_softmax_empty_odd", softmax_odd)
    rescale_p_even = _optional(costs, "T_rescale_p_even")
    rescale_o_even = _optional(costs, "T_rescale_o_even")
    stmatrix = _cost(costs, "T_stmatrix_p")

    n_even = (n_page + 1) // 2
    n_odd = n_page // 2
    has_single_tail = n_page % 2
    kq_all = kq_first_page + (n_page - 1) * kq_steady_page
    compute_serial = (
        n_even * (softmax_even + rescale_p_even)
        + n_odd * (softmax_odd + rescale_o_even)
        + has_single_tail * softmax_empty_odd
        + n_page * (pv_local + stmatrix + pv_remote)
    )
    overlap_credit = n_odd * min(pv_local, softmax_odd)
    if has_single_tail:
        overlap_credit += min(pv_local, softmax_empty_odd)

    output_store, reduce_l = _epilogue(costs, split_kv)
    atom_sum = _cost(costs, "T_qload") + kq_all + compute_serial + output_store + reduce_l
    source_dag = atom_sum - overlap_credit
    e2e, combine_included = _additive_e2e(costs, source_dag, split_kv)
    missing_dense_modes = any(
        key not in costs
        for key in (
            "T_softmax_even",
            "T_softmax_odd",
            "T_softmax_empty_odd",
            "T_rescale_p_even",
            "T_rescale_o_even",
        )
    )
    return {
        "model_kind": "atom_fallback",
        "N_page": float(n_page),
        "N_pair": float(n_odd),
        "T_K_page": k_page,
        "T_KQ_first_page": kq_first_page,
        "T_KQ_steady_page": kq_steady_page,
        "T_atom_sum": atom_sum,
        "T_confirmed_overlap_credit": overlap_credit,
        "T_source_dag": source_dag,
        "T_main_model": source_dag,
        "T_model": source_dag,
        "T_e2e_additive": e2e,
        "combine_included": combine_included,
        "missing_dense_softmax_or_rescale_modes": missing_dense_modes,
        "split_kv": split_kv,
        "warning": "diagnostic fallback; cross-pair TMA/WGMMA overlap is not modeled",
    }


def compose(
    costs: Mapping[str, float], n_page: int, split_kv: bool = False
) -> dict[str, float | str | bool | None]:
    if n_page < 1:
        raise ValueError("n_page must be at least 1")
    schedule_keys = {
        "T_prologue_single",
        "T_prologue_pair",
        "T_pair_transition",
        "T_pair_to_single",
        "T_pair_drain",
        "T_single_drain",
    }
    if schedule_keys.intersection(costs):
        return _compose_schedule(costs, n_page, split_kv)
    return _compose_atoms(costs, n_page, split_kv)


def main(argv: list[str] | None = None) -> None:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    if "--profile" in arguments or "--workload" in arguments:
        from microbench.model.dense_decode.profile import load_profile
        from microbench.model.dense_decode.schema import load_workload
        from microbench.model.dense_decode.simulator import predict

        model_parser = argparse.ArgumentParser(
            description="Compatibility entry point for the microbench dense-decode model"
        )
        model_parser.add_argument("--profile", required=True, type=Path)
        model_parser.add_argument("--workload", required=True, type=Path)
        model_parser.add_argument("--bootstrap", type=int, default=0)
        model_args = model_parser.parse_args(arguments)
        if model_args.bootstrap < 0:
            model_parser.error("--bootstrap must be non-negative")
        profile = load_profile(model_args.profile)
        workload_value = json.loads(model_args.workload.read_text(encoding="utf-8"))
        if not isinstance(workload_value, dict):
            raise ValueError("workload JSON must contain an object")
        result = predict(
            profile, load_workload(workload_value), bootstrap=model_args.bootstrap
        ).result
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles-json", required=True, type=Path)
    pages = parser.add_mutually_exclusive_group(required=True)
    pages.add_argument("--n-page", type=int, help="pages assigned to this CTA request/split")
    pages.add_argument(
        "--seqlen-k",
        type=int,
        help="full-sequence convenience only; invalid when the scheduler splits a request",
    )
    parser.add_argument("--split-kv", action="store_true")
    args = parser.parse_args(arguments)

    if args.n_page is not None:
        n_page = args.n_page
    else:
        if args.seqlen_k < 1:
            parser.error("--seqlen-k must be at least 1")
        if args.split_kv:
            parser.error("use --n-page for split-KV; full seqlen does not determine CTA work")
        n_page = math.ceil(args.seqlen_k / 64)

    loaded = json.loads(args.cycles_json.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("cycles JSON must contain an object")
    result = compose(loaded, n_page, args.split_kv)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
