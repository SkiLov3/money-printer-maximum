#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.settings import get_combine_rules, get_optimizer_config, load_stack_config
from src.optimization.combine_rules import evaluate_combine_rules
from src.utils.backtester import Backtester


SYMBOLS = ("MNQM6", "MGCM6", "MCLM6")
SYMBOL_GLOBS = {
    "MNQM6": ("*mnqm6*.csv", "*MNQM6*.csv", "*mnq*.csv", "*MNQ*.csv"),
    "MGCM6": ("*mgcm6*.csv", "*MGCM6*.csv", "*mgc*.csv", "*MGC*.csv"),
    "MCLM6": ("*mclm6*.csv", "*MCLM6*.csv", "*mcl*.csv", "*MCL*.csv"),
}
SUMMARY_FIELDS = (
    "candidate_id",
    "candidate_family",
    "routing_mode",
    "total_pnl",
    "avg_daily_pnl_trade_dates",
    "avg_daily_pnl_calendar_span",
    "worst_daily_loss",
    "best_daily_pnl",
    "best_day_consistency_pct",
    "max_drawdown",
    "total_trades",
    "trades_per_trade_day",
    "trades_per_calendar_day",
    "win_rate",
    "profit_factor",
    "topstep_50k_passed",
    "topstep_50k_daily_enforced_passed",
    "topstep_50k_mll_gap",
    "topstep_50k_daily_mll_gap",
    "score",
)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    family: str
    strategy_config: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline portfolio optimizer for MNQM6, MGCM6, and MCLM6 strategy mixes. "
            "It reports combine-rule outcomes only; it does not add live execution gates."
        )
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing MNQM6/MGCM6/MCLM6 CSV files.")
    parser.add_argument("--output-dir", default="reports/portfolio_mix_optimizer")
    parser.add_argument("--max-candidates", type=int, default=0, help="Limit candidates after deterministic de-dupe; 0 means all.")
    parser.add_argument(
        "--candidate-id",
        action="append",
        default=[],
        help="Evaluate only the requested candidate id. May be provided multiple times.",
    )
    parser.add_argument("--tail-bars", type=int, default=0, help="Use only the most recent N bars per symbol; 0 means full CSV.")
    parser.add_argument("--top-n", type=int, default=12, help="Number of search finalists to include in reports.")
    parser.add_argument("--routing-mode", default="optimizer_fast", help="Backtester routing mode used for candidate search.")
    parser.add_argument(
        "--final-routing-mode",
        default="",
        help="Optional routing mode for finalists, usually full_live. Empty string skips finalist re-runs.",
    )
    parser.add_argument(
        "--family",
        action="append",
        default=[],
        help="Evaluate only candidates belonging to the requested family. May be provided multiple times.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    run_dir = Path(args.output_dir) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)

    stack_config = load_stack_config()
    combine_rules = {
        "topstep_50k": get_combine_rules(stack_config, "topstep_50k"),
        "topstep_50k_daily_enforced": get_combine_rules(stack_config, "topstep_50k_daily_enforced"),
    }
    data = load_symbol_data(Path(args.data_dir), int(args.tail_bars))
    candidates = build_candidates(stack_config)
    requested_families = {str(item) for item in args.family}
    if requested_families:
        candidates = [candidate for candidate in candidates if candidate.family in requested_families]
    requested_ids = {str(item) for item in args.candidate_id}
    if requested_ids:
        candidates = [candidate for candidate in candidates if candidate.candidate_id in requested_ids]
        missing = sorted(requested_ids - {candidate.candidate_id for candidate in candidates})
        if missing:
            raise ValueError(f"Unknown candidate id(s): {', '.join(missing)}")
    if int(args.max_candidates) > 0:
        candidates = candidates[: int(args.max_candidates)]

    print(f"Testing {len(candidates)} portfolio candidates across {', '.join(SYMBOLS)} -> {run_dir}", flush=True)
    write_json(run_dir / "candidates.json", [candidate_to_dict(item) for item in candidates])

    from concurrent.futures import ProcessPoolExecutor
    import os

    num_workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"Running optimization with {num_workers} processes...", flush=True)

    worker_args = [(c, data, combine_rules, str(args.routing_mode)) for c in candidates]
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(worker_evaluate, worker_args)
        for index, res in enumerate(results, start=1):
            rows.append(res["row"])
            details.append(res["evaluation"])
            row = res["row"]
            
            if index % 10 == 0 or index == len(candidates):
                write_csv(run_dir / "candidate_scores.csv", sorted_rows(rows))
            
            err = row.get("error", "")
            if err:
                print(f"{index}/{len(candidates)} {row['candidate_id']} ERROR: {err}", flush=True)
            else:
                print(
                    f"{index}/{len(candidates)} {row['candidate_id']} "
                    f"trades={row.get('total_trades', 0)} pnl={row.get('total_pnl', 0.0)} "
                    f"pf={row.get('profit_factor', 0.0)} pass50k={row.get('topstep_50k_passed', False)}",
                    flush=True,
                )

    ranked_rows = sorted_rows(rows)
    finalists = ranked_rows[: max(1, int(args.top_n))]
    write_csv(run_dir / "candidate_scores.csv", ranked_rows)
    write_json(run_dir / "candidate_details.json", details)

    final_rows: list[dict[str, Any]] = []
    final_details: list[dict[str, Any]] = []
    final_mode = str(args.final_routing_mode or "").strip()
    if final_mode:
        candidate_by_id = {item.candidate_id: item for item in candidates}
        for rank, row in enumerate(finalists, start=1):
            candidate = candidate_by_id[str(row["candidate_id"])]
            evaluation = evaluate_candidate(candidate, data, combine_rules, routing_mode=final_mode)
            final_row = flatten_evaluation(evaluation)
            final_row["search_rank"] = rank
            final_rows.append(final_row)
            final_details.append(evaluation)
            print(
                f"final {rank}/{len(finalists)} {candidate.candidate_id} "
                f"trades={final_row.get('total_trades', 0)} pnl={final_row.get('total_pnl', 0.0)} "
                f"pass50k={final_row.get('topstep_50k_passed', False)}",
                flush=True,
            )
        final_rows = sorted_rows(final_rows)
        write_csv(run_dir / "finalists_full.csv", final_rows)
        write_json(run_dir / "finalist_details.json", final_details)

    if finalists:
        best_id = str(finalists[0]["candidate_id"])
        best = next(item for item in details if item.get("candidate_id") == best_id)
        write_json(run_dir / "best_config.json", {"candidate_id": best_id, "strategy_config": best["strategy_config"]})
        write_csv(run_dir / "best_trades.csv", best.get("portfolio_trades", []))

    write_report(
        run_dir,
        ranked_rows,
        final_rows,
        args,
        data,
        elapsed_seconds=time.perf_counter() - started,
    )
    print(f"Report: {run_dir / 'analysis.md'}", flush=True)
    return 0


def load_symbol_data(data_dir: Path, tail_bars: int) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for symbol in SYMBOLS:
        path = find_data_file(data_dir, symbol)
        frame = pd.read_csv(path)
        frame.columns = [str(col).lower() for col in frame.columns]
        if "timestamp" in frame.columns:
            frame.loc[:, "timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        if tail_bars > 0 and len(frame) > tail_bars:
            frame = frame.tail(tail_bars).reset_index(drop=True)
        loaded[symbol] = {"path": str(path), "frame": frame}
    return loaded


def find_data_file(data_dir: Path, symbol: str) -> Path:
    for pattern in SYMBOL_GLOBS[symbol]:
        matches = sorted(data_dir.glob(pattern))
        if matches:
            return matches[-1]
    raise FileNotFoundError(f"No CSV found for {symbol} under {data_dir}")


def build_candidates(stack_config: dict[str, Any]) -> list[Candidate]:
    configured_candidates: list[Candidate] = []
    optimizer_config = get_optimizer_config(stack_config)
    for idx, configured in enumerate(optimizer_config.get("strategy_candidates") or [], start=1):
        configured_candidates.append(Candidate(f"configured_strategy_candidate_{idx}", "configured", normalize_config(configured)))

    groups: list[list[Candidate]] = []
    candidates: list[Candidate] = list(configured_candidates)

    price_action_group: list[Candidate] = []
    for strategy in price_action_candidates():
        price_action_group.append(Candidate(strategy["candidate_id"], "price_action_bs", strategy["strategy_config"]))
    groups.append(price_action_group)

    golden_zone_group: list[Candidate] = []
    for strategy in golden_zone_candidates():
        golden_zone_group.append(Candidate(strategy["candidate_id"], "golden_zone_scalper", strategy["strategy_config"]))
    groups.append(golden_zone_group)

    pa_gz_group: list[Candidate] = []
    for strategy in price_action_golden_zone_candidates():
        pa_gz_group.append(Candidate(strategy["candidate_id"], "price_action_plus_golden_zone", strategy["strategy_config"]))
    groups.append(pa_gz_group)

    orb_group: list[Candidate] = []
    for strategy in orb_vwap_candidates():
        orb_group.append(Candidate(strategy["candidate_id"], "orb_vwap", strategy["strategy_config"]))
    groups.append(orb_group)

    prev_breakout_group: list[Candidate] = []
    for strategy in previous_bar_breakout_candidates():
        prev_breakout_group.append(Candidate(strategy["candidate_id"], "previous_bar_breakout", strategy["strategy_config"]))
    groups.append(prev_breakout_group)

    otz_group: list[Candidate] = []
    for strategy in otz_continuation_candidates():
        otz_group.append(Candidate(strategy["candidate_id"], "otz_continuation", strategy["strategy_config"]))
    groups.append(otz_group)

    smm_group: list[Candidate] = []
    for strategy in simple_market_metrics_candidates():
        smm_group.append(Candidate(strategy["candidate_id"], "simple_market_metrics", strategy["strategy_config"]))
    groups.append(smm_group)

    reversion_group: list[Candidate] = []
    for strategy in prior_extreme_vwap_reversion_candidates():
        reversion_group.append(Candidate(strategy["candidate_id"], "prior_extreme_vwap_reversion", strategy["strategy_config"]))
    groups.append(reversion_group)

    resting_level_group: list[Candidate] = []
    for strategy in resting_level_breakout_candidates():
        resting_level_group.append(Candidate(strategy["candidate_id"], "resting_level_breakout", strategy["strategy_config"]))
    groups.append(resting_level_group)

    squeeze_group: list[Candidate] = []
    for strategy in volatility_squeeze_breakout_candidates():
        squeeze_group.append(Candidate(strategy["candidate_id"], "volatility_squeeze_breakout", strategy["strategy_config"]))
    groups.append(squeeze_group)

    htf_level_group: list[Candidate] = []
    for strategy in htf_level_retest_candidates():
        htf_level_group.append(Candidate(strategy["candidate_id"], "htf_level_retest", strategy["strategy_config"]))
    groups.append(htf_level_group)

    session_pivot_group: list[Candidate] = []
    for strategy in session_pivot_retest_candidates():
        session_pivot_group.append(Candidate(strategy["candidate_id"], "session_pivot_retest", strategy["strategy_config"]))
    groups.append(session_pivot_group)

    causal_mcl_blend_group: list[Candidate] = []
    for strategy in causal_mcl_blend_candidates():
        causal_mcl_blend_group.append(Candidate(strategy["candidate_id"], "causal_mcl_blend", strategy["strategy_config"]))
    groups.append(causal_mcl_blend_group)

    pa_orb_group: list[Candidate] = []
    for strategy in price_action_orb_candidates():
        pa_orb_group.append(Candidate(strategy["candidate_id"], "price_action_plus_orb_vwap", strategy["strategy_config"]))
    groups.append(pa_orb_group)

    legacy_group: list[Candidate] = []
    for strategy in legacy_candidates():
        legacy_group.append(Candidate(strategy["candidate_id"], "legacy_combo", strategy["strategy_config"]))
    groups.append(legacy_group)

    gz_legacy_group: list[Candidate] = []
    for strategy in golden_zone_legacy_candidates():
        gz_legacy_group.append(Candidate(strategy["candidate_id"], "golden_zone_plus_legacy", strategy["strategy_config"]))
    groups.append(gz_legacy_group)

    candidates.extend(interleave_candidate_groups(groups))

    seen: set[str] = set()
    unique: list[Candidate] = []
    for candidate in candidates:
        key = json.dumps(candidate.strategy_config, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def interleave_candidate_groups(groups: list[list[Candidate]]) -> list[Candidate]:
    ordered: list[Candidate] = []
    max_len = max((len(group) for group in groups), default=0)
    for idx in range(max_len):
        for group in groups:
            if idx < len(group):
                ordered.append(group[idx])
    return ordered


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    normalized.setdefault("blender", {"mode": "priority_first", "multi_agree_bonus": 0.0})
    normalized.setdefault("direction_filters", {})
    normalized.setdefault("strategies", [])
    return normalized


def strategy_config(strategies: list[dict[str, Any]], *, blender_mode: str = "priority_first", risk: dict[str, Any] | None = None) -> dict[str, Any]:
    direction_filters: dict[str, list[str]] = {}
    for spec in strategies:
        direction_filters[str(spec["name"])] = ["buy", "sell"]
    config = {
        "blender": {"mode": blender_mode, "multi_agree_bonus": 0.0 if blender_mode == "priority_first" else 2.0},
        "direction_filters": direction_filters,
        "strategies": strategies,
    }
    if risk:
        config["risk"] = copy.deepcopy(risk)
    return config


def spec(name: str, priority: int, params: dict[str, Any] | None = None, enabled: bool = True) -> dict[str, Any]:
    return {"name": name, "enabled": enabled, "priority": priority, "params": copy.deepcopy(params or {})}


def price_action_params(
    *,
    risk_reward: float,
    min_atr_ratio: float,
    ema_distance: float,
    max_stop_ticks: dict[str, dict[str, int]],
    allowed_hours: list[int],
    target_mode: str = "risk_multiple",
) -> dict[str, Any]:
    return {
        "setups_path": "pa_strat/setups.yaml",
        "source_files": [
            "pa_strat/playbook.md",
            "pa_strat/setups.yaml",
            "pa_strat/glossary.md",
            "pa_strat/open_questions.md",
        ],
        "enabled_setups": ["E-01", "E-02", "E-03", "E-04", "E-05", "E-06", "E-07", "E-08"],
        "disabled_setup_sides": [],
        "scalp_target_ticks": 10,
        "target_mode": target_mode,
        "risk_reward": float(risk_reward),
        "min_target_ticks": 10,
        "session_cutoff_enabled": True,
        "min_signal_bar_atr_ratio": float(min_atr_ratio),
        "min_close_ema_distance_ticks": float(ema_distance),
        "allowed_entry_hours_et": list(allowed_hours),
        "range_double_test_breakout_ticks": 1,
        "range_double_test_min_separation_bars": 3,
        "range_double_test_tolerance_ticks": 3,
        "range_double_test_visual_bars": 120,
        "use_candidate_price_targets": False,
        "symbol_params": copy.deepcopy(max_stop_ticks),
    }


def price_action_candidates() -> list[dict[str, Any]]:
    stop_profiles = {
        "live": {"MGC": {"max_stop_ticks": 40}, "MCL": {"max_stop_ticks": 0}, "MNQ": {"max_stop_ticks": 60}},
        "balanced": {"MGC": {"max_stop_ticks": 50}, "MCL": {"max_stop_ticks": 50}, "MNQ": {"max_stop_ticks": 70}},
        "wide": {"MGC": {"max_stop_ticks": 60}, "MCL": {"max_stop_ticks": 60}, "MNQ": {"max_stop_ticks": 90}},
    }
    hour_profiles = {
        "all_hours": [],
        "rth_core": [8, 9, 10, 11, 12, 13, 14, 15],
        "known_good": [0, 1, 8, 10, 13, 15],
    }
    rows: list[dict[str, Any]] = []
    for stops_name, stops in stop_profiles.items():
        for hours_name, hours in hour_profiles.items():
            for rr in (1.5, 2.0, 3.0):
                for atr in (1.0, 1.25):
                    for ema in (12.0, 20.0, 30.0):
                        params = price_action_params(
                            risk_reward=rr,
                            min_atr_ratio=atr,
                            ema_distance=ema,
                            max_stop_ticks=stops,
                            allowed_hours=hours,
                        )
                        rows.append(
                            {
                                "candidate_id": f"pa_{stops_name}_{hours_name}_rr{rr:g}_atr{atr:g}_ema{ema:g}",
                                "strategy_config": strategy_config([spec("price_action_bs", 100, params)]),
                            }
                        )
    return rows


def golden_zone_params(
    *,
    risk_reward: float,
    min_adx: float,
    lookback: int,
    max_stop_ticks: dict[str, dict[str, int]],
    require_engulfing: bool,
) -> dict[str, Any]:
    return {
        "lookback": int(lookback),
        "trigger_lookback": 3,
        "ema_fast": "ema_21",
        "ema_slow": "ema_50",
        "min_adx": float(min_adx),
        "min_ema_slope_ticks": 1.0,
        "zone_min": 0.382,
        "zone_max": 0.618,
        "breakout_buffer_ticks": 1,
        "stop_buffer_ticks": 2,
        "min_stop_ticks": 8,
        "risk_reward": float(risk_reward),
        "min_target_ticks": 10,
        "min_body_ratio": 0.25,
        "require_engulfing": bool(require_engulfing),
        "symbol_params": copy.deepcopy(max_stop_ticks),
    }


def golden_zone_candidates() -> list[dict[str, Any]]:
    stop_profiles = {
        "balanced": {"MGC": {"max_stop_ticks": 45}, "MCL": {"max_stop_ticks": 45}, "MNQ": {"max_stop_ticks": 65}},
        "wide": {"MGC": {"max_stop_ticks": 65}, "MCL": {"max_stop_ticks": 65}, "MNQ": {"max_stop_ticks": 90}},
    }
    rows: list[dict[str, Any]] = []
    for stops_name, stops in stop_profiles.items():
        for rr in (1.5, 2.0, 2.5):
            for adx in (10.0, 14.0, 18.0):
                for lookback in (36, 48):
                    params = golden_zone_params(
                        risk_reward=rr,
                        min_adx=adx,
                        lookback=lookback,
                        max_stop_ticks=stops,
                        require_engulfing=False,
                    )
                    rows.append(
                        {
                            "candidate_id": f"gz_{stops_name}_rr{rr:g}_adx{adx:g}_lb{lookback}",
                            "strategy_config": strategy_config([spec("golden_zone_scalper", 100, params)]),
                        }
                    )
    session_hours = [8, 9, 15, 19, 20, 21]
    for rr in (1.5, 2.0, 2.5):
        for adx in (14.0, 18.0, 22.0):
            params = golden_zone_params(
                risk_reward=rr,
                min_adx=adx,
                lookback=48,
                max_stop_ticks={"MGC": {"max_stop_ticks": 45}, "MCL": {"max_stop_ticks": 45}, "MNQ": {"max_stop_ticks": 65}},
                require_engulfing=False,
            )
            params["allowed_entry_hours_et"] = session_hours
            rows.append(
                {
                    "candidate_id": f"session_gz_core_8_9_15_19_20_21_rr{rr:g}_adx{adx:g}_lb48",
                    "strategy_config": strategy_config([spec("golden_zone_scalper", 100, params)]),
                }
            )
    return rows


def price_action_golden_zone_candidates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pa_stops = {"MGC": {"max_stop_ticks": 40}, "MCL": {"max_stop_ticks": 0}, "MNQ": {"max_stop_ticks": 60}}
    gz_stops = {"MGC": {"max_stop_ticks": 55}, "MCL": {"max_stop_ticks": 55}, "MNQ": {"max_stop_ticks": 80}}
    for rr in (1.5, 2.0, 3.0):
        for atr in (1.0, 1.25):
            pa = price_action_params(
                risk_reward=rr,
                min_atr_ratio=atr,
                ema_distance=20.0,
                max_stop_ticks=pa_stops,
                allowed_hours=[0, 1, 8, 10, 13, 15],
            )
            gz = golden_zone_params(
                risk_reward=2.0,
                min_adx=14.0,
                lookback=48,
                max_stop_ticks=gz_stops,
                require_engulfing=False,
            )
            rows.append(
                {
                    "candidate_id": f"pa_gz_priority_rr{rr:g}_atr{atr:g}",
                    "strategy_config": strategy_config(
                        [spec("price_action_bs", 100, pa), spec("golden_zone_scalper", 90, gz)]
                    ),
                }
            )
    return rows


def orb_vwap_params(
    *,
    mode: str,
    or_minutes: int,
    trade_end_hour_et: int,
    risk_reward: float,
    vwap_filter: bool,
    volume_ratio: float,
    min_or_ticks: int,
    max_or_ticks: int,
) -> dict[str, Any]:
    range_end_minutes = (9 * 60) + 30 + int(or_minutes)
    range_end = f"{range_end_minutes // 60:02d}:{range_end_minutes % 60:02d}"
    return {
        "direction_mode": mode,
        "opening_range_start_et": "09:30",
        "opening_range_end_et": range_end,
        "trade_window_end_et": f"{int(trade_end_hour_et):02d}:00",
        "use_vwap_filter": bool(vwap_filter),
        "use_ema200_filter": False,
        "min_volume_ratio": float(volume_ratio),
        "target_mode": "risk_multiple",
        "risk_reward": float(risk_reward),
        "fixed_target_ticks": 40,
        "stop_buffer_ticks": 2,
        "breakout_buffer_ticks": 1,
        "max_one_trade_per_session": True,
        "symbol_params": {
            "MNQ": {"min_or_ticks": min_or_ticks, "max_or_ticks": max_or_ticks},
            "MGC": {"min_or_ticks": max(4, min_or_ticks // 2), "max_or_ticks": max(20, max_or_ticks // 2)},
            "MCL": {"min_or_ticks": max(6, min_or_ticks // 3), "max_or_ticks": max(30, max_or_ticks // 2)},
        },
    }


def orb_vwap_candidates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in ("breakout", "fade", "both"):
        for or_minutes in (15, 30, 60):
            for rr in (1.25, 1.5, 2.0):
                params = orb_vwap_params(
                    mode=mode,
                    or_minutes=or_minutes,
                    trade_end_hour_et=12,
                    risk_reward=rr,
                    vwap_filter=True,
                    volume_ratio=1.0,
                    min_or_ticks=20,
                    max_or_ticks=240,
                )
                rows.append(
                    {
                        "candidate_id": f"orb_{mode}_or{or_minutes}_rr{rr:g}_vwap",
                        "strategy_config": strategy_config([spec("orb_vwap", 100, params)]),
                    }
                )
    for mode in ("breakout", "fade", "both"):
        for or_minutes in (5, 15, 30):
            for trade_end_hour in (11, 14):
                for rr in (0.8, 1.0, 1.25):
                    for filter_name, use_vwap, volume_ratio in (
                        ("nofilter", False, 0.0),
                        ("vwap", True, 0.0),
                        ("vwap_vol05", True, 0.5),
                    ):
                        params = orb_vwap_params(
                            mode=mode,
                            or_minutes=or_minutes,
                            trade_end_hour_et=trade_end_hour,
                            risk_reward=rr,
                            vwap_filter=use_vwap,
                            volume_ratio=volume_ratio,
                            min_or_ticks=4,
                            max_or_ticks=480,
                        )
                        params["symbol_params"] = {
                            "MNQ": {"min_or_ticks": 4, "max_or_ticks": 480},
                            "MGC": {"min_or_ticks": 2, "max_or_ticks": 240},
                            "MCL": {"min_or_ticks": 2, "max_or_ticks": 240},
                        }
                        rows.append(
                            {
                                "candidate_id": (
                                    f"orbx_{mode}_or{or_minutes}_to{trade_end_hour}_"
                                    f"rr{rr:g}_{filter_name}"
                                ),
                                "strategy_config": strategy_config([spec("orb_vwap", 100, params)]),
                            }
                        )
    return rows


def price_action_orb_candidates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pa_stops = {"MGC": {"max_stop_ticks": 40}, "MCL": {"max_stop_ticks": 0}, "MNQ": {"max_stop_ticks": 60}}
    for rr in (1.5, 2.0, 3.0):
        pa = price_action_params(
            risk_reward=rr,
            min_atr_ratio=1.25,
            ema_distance=12.0,
            max_stop_ticks=pa_stops,
            allowed_hours=[0, 1, 8, 10, 13, 15],
        )
        orb = orb_vwap_params(
            mode="breakout",
            or_minutes=30,
            trade_end_hour_et=12,
            risk_reward=1.5,
            vwap_filter=True,
            volume_ratio=1.0,
            min_or_ticks=20,
            max_or_ticks=240,
        )
        rows.append(
            {
                "candidate_id": f"pa_orb_priority_rr{rr:g}",
                "strategy_config": strategy_config([spec("price_action_bs", 100, pa), spec("orb_vwap", 90, orb)]),
            }
        )
    return rows


def previous_bar_breakout_params(
    *,
    direction_mode: str,
    lookback: int,
    stop_ticks: int,
    target_ticks: int,
    min_adx: float,
    min_body_ratio: float,
    min_range_ticks: int,
    ema_filter: bool,
    allowed_hours: list[int],
) -> dict[str, Any]:
    return {
        "direction_mode": str(direction_mode),
        "lookback": int(lookback),
        "breakout_buffer_ticks": 1,
        "min_range_ticks": int(min_range_ticks),
        "min_body_ratio": float(min_body_ratio),
        "min_adx": float(min_adx),
        "use_ema_filter": bool(ema_filter),
        "ema_column": "ema_21",
        "sl_ticks": int(stop_ticks),
        "tp_ticks": int(target_ticks),
        "min_signal_gap_bars": 3,
        "allowed_entry_hours_et": list(allowed_hours),
        "symbol_params": {
            "MNQ": {"min_range_ticks": int(min_range_ticks), "sl_ticks": int(stop_ticks), "tp_ticks": int(target_ticks)},
            "MGC": {
                "min_range_ticks": max(2, int(min_range_ticks // 2)),
                "sl_ticks": max(12, int(stop_ticks // 2)),
                "tp_ticks": max(24, int(target_ticks // 2)),
            },
            "MCL": {
                "min_range_ticks": max(2, int(min_range_ticks // 3)),
                "sl_ticks": max(12, int(stop_ticks // 2)),
                "tp_ticks": max(24, int(target_ticks // 2)),
            },
        },
    }


def previous_bar_breakout_candidates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hour_profiles = {
        "all": [],
        "rth": [8, 9, 10, 11, 12, 13, 14, 15],
        "active": [0, 1, 8, 9, 10, 13, 14, 15],
        "session_edge": [0, 8, 9, 15],
    }
    for direction_mode in ("fade", "continuation"):
        for hours_name, hours in hour_profiles.items():
            for lookback in (4, 8, 12, 20):
                for stop_ticks, target_ticks in ((24, 36), (30, 45), (40, 60), (40, 80)):
                    for min_adx in (0.0, 12.0, 18.0):
                        for body in (0.35, 0.55):
                            ema_filter = direction_mode == "continuation"
                            params = previous_bar_breakout_params(
                                direction_mode=direction_mode,
                                lookback=lookback,
                                stop_ticks=stop_ticks,
                                target_ticks=target_ticks,
                                min_adx=min_adx,
                                min_body_ratio=body,
                                min_range_ticks=8,
                                ema_filter=ema_filter,
                                allowed_hours=hours,
                            )
                            rows.append(
                                {
                                    "candidate_id": (
                                        f"pbb_{direction_mode}_{hours_name}_lb{lookback}_"
                                        f"sl{stop_ticks}_tp{target_ticks}_adx{min_adx:g}_body{body:g}"
                                    ),
                                    "strategy_config": strategy_config([spec("previous_bar_breakout", 100, params)]),
                                }
                            )
    return rows


def simple_market_metrics_params(
    *,
    stop_ticks: int,
    target_ticks: int,
    allowed_hours: list[int],
    require_stacked_emas: bool,
    use_chop_filter: bool,
    use_profit_wave_exit: bool,
    max_wick_ticks: float,
    enabled_symbols: list[str] | None = None,
    wick_floor_match: bool = True,
    signal_mode: str = "normal",
    exit_on_dashboard_reversal: bool = True,
    require_mfi_slope: bool = False,
    mfi_slope_lookback: int = 3,
    min_mfi_slope: float = 2.0,
    require_recent_ema_cross: bool = False,
    ema_cross_lookback: int = 6,
    require_linreg_slope: bool = False,
    linreg_length: int = 34,
    min_linreg_slope_ticks: float = 0.2,
) -> dict[str, Any]:
    return {
        "sl_ticks": int(stop_ticks),
        "tp_ticks": int(target_ticks),
        "allowed_entry_hours_et": list(allowed_hours),
        "require_stacked_emas": bool(require_stacked_emas),
        "use_ma_filter": True,
        "use_chop_filter": bool(use_chop_filter),
        "use_profit_wave_exit": bool(use_profit_wave_exit),
        "exit_on_dashboard_reversal": bool(exit_on_dashboard_reversal),
        "signal_mode": str(signal_mode),
        "wick_floor_match": bool(wick_floor_match),
        "max_wick_ticks": float(max_wick_ticks),
        "require_mfi_slope": bool(require_mfi_slope),
        "mfi_slope_lookback": int(mfi_slope_lookback),
        "min_mfi_slope": float(min_mfi_slope),
        "require_recent_ema_cross": bool(require_recent_ema_cross),
        "ema_cross_lookback": int(ema_cross_lookback),
        "require_linreg_slope": bool(require_linreg_slope),
        "linreg_length": int(linreg_length),
        "min_linreg_slope_ticks": float(min_linreg_slope_ticks),
        "enabled_symbols": list(enabled_symbols or []),
        "min_di": 45,
        "min_mfi_buy": 52,
        "max_mfi_sell": 52,
        "symbol_params": {
            "MNQ": {"sl_ticks": int(stop_ticks), "tp_ticks": int(target_ticks)},
            "NQ": {"sl_ticks": int(stop_ticks), "tp_ticks": int(target_ticks)},
            "MES": {"sl_ticks": max(8, int(stop_ticks // 5)), "tp_ticks": max(4, int(target_ticks // 5))},
            "ES": {"sl_ticks": max(8, int(stop_ticks // 5)), "tp_ticks": max(4, int(target_ticks // 5))},
            "MGC": {"sl_ticks": max(12, int(stop_ticks // 2)), "tp_ticks": max(12, int(target_ticks // 2))},
            "MCL": {"sl_ticks": max(12, int(stop_ticks // 2)), "tp_ticks": max(12, int(target_ticks // 2))},
        },
    }


def simple_market_metrics_candidates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hour_profiles = {
        "ny10_15": [10, 11, 12, 13, 14],
        "rth": [9, 10, 11, 12, 13, 14, 15],
        "open": [9, 10, 11],
    }
    for hours_name, hours in hour_profiles.items():
        for stop_ticks, target_ticks in ((16, 20), (20, 20), (24, 28), (30, 20), (30, 30), (40, 40), (50, 40), (30, 160)):
            for stacked in (True, False):
                for chop in (True, False):
                    for wick_floor_match in (True, False):
                        for signal_mode in ("normal", "invert"):
                            if signal_mode == "invert" and (not stacked or not chop or not wick_floor_match):
                                continue
                            params = simple_market_metrics_params(
                                stop_ticks=stop_ticks,
                                target_ticks=target_ticks,
                                allowed_hours=hours,
                                require_stacked_emas=stacked,
                                use_chop_filter=chop,
                                use_profit_wave_exit=True,
                                max_wick_ticks=1.0,
                                enabled_symbols=["MNQ"],
                                wick_floor_match=wick_floor_match,
                                signal_mode=signal_mode,
                                exit_on_dashboard_reversal=True,
                            )
                            rows.append(
                                {
                                    "candidate_id": (
                                        f"smm_mnq_{hours_name}_sl{stop_ticks}_tp{target_ticks}_"
                                        f"{'stack' if stacked else 'nostack'}_{'chop' if chop else 'nochop'}_"
                                        f"{'floor' if wick_floor_match else 'wick'}_{signal_mode}"
                                    ),
                                    "strategy_config": strategy_config([spec("simple_market_metrics", 115, params)]),
                                }
                            )
    strict_hour_profiles = {
        "s2open9": [9],
        "s2open910": [9, 10],
        "s2ny10": [10],
    }
    for hours_name, hours in strict_hour_profiles.items():
        for stop_ticks, target_ticks in ((24, 28), (30, 30), (40, 40), (50, 40), (30, 80)):
            for mfi_lookback, min_mfi_slope in ((2, 1.5), (3, 2.5), (5, 3.5)):
                for cross_lookback in (4, 8):
                    for linreg_length, slope_ticks in ((21, 0.15), (34, 0.1)):
                        params = simple_market_metrics_params(
                            stop_ticks=stop_ticks,
                            target_ticks=target_ticks,
                            allowed_hours=hours,
                            require_stacked_emas=True,
                            use_chop_filter=True,
                            use_profit_wave_exit=True,
                            max_wick_ticks=1.0,
                            enabled_symbols=["MNQ"],
                            wick_floor_match=True,
                            signal_mode="normal",
                            exit_on_dashboard_reversal=True,
                            require_mfi_slope=True,
                            mfi_slope_lookback=mfi_lookback,
                            min_mfi_slope=min_mfi_slope,
                            require_recent_ema_cross=True,
                            ema_cross_lookback=cross_lookback,
                            require_linreg_slope=True,
                            linreg_length=linreg_length,
                            min_linreg_slope_ticks=slope_ticks,
                        )
                        rows.append(
                            {
                                "candidate_id": (
                                    f"smm_mnq_{hours_name}_sl{stop_ticks}_tp{target_ticks}_"
                                    f"mf{mfi_lookback}x{min_mfi_slope:g}_cross{cross_lookback}_"
                                    f"lr{linreg_length}x{slope_ticks:g}"
                                ),
                                "strategy_config": strategy_config([spec("simple_market_metrics", 115, params)]),
                            }
                        )
    return rows


def otz_continuation_params(
    *,
    stop_ticks: int,
    target_ticks: int,
    allowed_hours: list[int],
    htf_minutes: int,
    zone_mode: str,
    confirmation: str,
    min_htf_impulse_atr: float,
    min_htf_body_pct: float,
    require_ema_context: bool,
    enabled_symbols: list[str],
    htf_lookback: int = 24,
    engulf_mode: str = "body",
    ob_wick_area_mode: str = "recent_bars",
    ob_wick_area_bars: int = 3,
    ob_wick_cluster_tolerance_ticks: int = 120,
    structure_123_mode: str = "close_break",
    trendline_source: str = "wick",
    orderflow_mode: str = "ema_any",
    stop_mode: str = "fixed_ticks",
    stop_padding_ticks: int = 2,
    target_mode: str = "fixed_ticks",
    target_r_multiple: float = 2.0,
    trigger_zone_mode: str = "wick_touch_any_close",
) -> dict[str, Any]:
    return {
        "sl_ticks": int(stop_ticks),
        "tp_ticks": int(target_ticks),
        "allowed_entry_hours_et": list(allowed_hours),
        "htf_minutes": int(htf_minutes),
        "htf_lookback": int(htf_lookback),
        "zone_mode": str(zone_mode),
        "engulf_mode": str(engulf_mode),
        "ob_wick_area_mode": str(ob_wick_area_mode),
        "ob_wick_area_bars": int(ob_wick_area_bars),
        "ob_wick_cluster_tolerance_ticks": int(ob_wick_cluster_tolerance_ticks),
        "ob_zone_padding_ticks": 0,
        "invalidate_on_close_through_otz": True,
        "confirmation": str(confirmation),
        "structure_123_mode": str(structure_123_mode),
        "trendline_source": str(trendline_source),
        "min_htf_impulse_atr": float(min_htf_impulse_atr),
        "min_htf_body_pct": float(min_htf_body_pct),
        "zone_tolerance_ticks": 6,
        "trigger_zone_mode": str(trigger_zone_mode),
        "min_reject_wick_ratio": 0.45,
        "min_confirmation_body_ticks": 4,
        "require_ema_context": bool(require_ema_context),
        "orderflow_mode": str(orderflow_mode),
        "stop_mode": str(stop_mode),
        "stop_padding_ticks": int(stop_padding_ticks),
        "target_mode": str(target_mode),
        "target_r_multiple": float(target_r_multiple),
        "min_signal_gap_bars": 8,
        "enabled_symbols": list(enabled_symbols),
        "symbol_params": {
            "MGC": {"sl_ticks": int(stop_ticks), "tp_ticks": int(target_ticks), "tick_size": 0.1},
            "MNQ": {"sl_ticks": max(20, int(stop_ticks * 2)), "tp_ticks": max(24, int(target_ticks * 2))},
            "MCL": {"sl_ticks": max(16, int(stop_ticks // 2)), "tp_ticks": max(16, int(target_ticks // 2))},
        },
    }


def otz_continuation_candidates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hour_profiles = {
        "all": [],
        "ny": [8, 9, 10, 11, 12, 13, 14],
        "rth": [9, 10, 11, 12, 13, 14, 15],
    }
    # The Notion examples are mostly XAU/US30/GBPJPY. In this combine repo, MGC is
    # the closest tradable proxy to XAU, while MNQ/MCL let us test transferability.
    for symbols_name, enabled_symbols in {"mgc": ["MGC"], "mgc_mnq": ["MGC", "MNQ"], "all3": ["MGC", "MNQ", "MCL"]}.items():
        for hours_name, hours in hour_profiles.items():
            for stop_ticks, target_ticks in ((24, 36), (30, 45), (40, 60), (50, 80)):
                for htf_minutes in (15, 30):
                    for zone_mode in ("engulfed_wick_ob", "impulse_body_50", "full_body", "wick"):
                        for confirmation in ("engulf_or_123", "engulf", "123", "wick"):
                            for impulse_atr in (1.2, 1.6):
                                params = otz_continuation_params(
                                    stop_ticks=stop_ticks,
                                    target_ticks=target_ticks,
                                    allowed_hours=hours,
                                    htf_minutes=htf_minutes,
                                    zone_mode=zone_mode,
                                    confirmation=confirmation,
                                    min_htf_impulse_atr=impulse_atr,
                                    min_htf_body_pct=0.55,
                                    require_ema_context=True,
                                    enabled_symbols=enabled_symbols,
                                )
                                rows.append(
                                    {
                                        "candidate_id": (
                                            f"otz_{symbols_name}_{hours_name}_h{htf_minutes}_"
                                            f"{zone_mode}_{confirmation}_sl{stop_ticks}_tp{target_ticks}_atr{impulse_atr:g}"
                                        ),
                                        "strategy_config": strategy_config([spec("otz_continuation", 120, params)]),
                                    }
                                )
    daily_locks = (0, 1000, 1500, 2000)
    htf_profiles = {
        "h5": [5],
        "h15": [15],
        "h60": [60],
        "h5_15_60": [5, 15, 60],
    }
    target_profiles = (
        ("r2", 30, 60, "risk_multiple", 2.0),
        ("r3", 30, 90, "risk_multiple", 3.0),
        ("fixed45", 30, 45, "fixed_ticks", 2.0),
        ("oppotz", 30, 45, "opposite_otz", 2.0),
    )
    for daily_lock in daily_locks:
        for htf_name, htf_minutes_values in htf_profiles.items():
            for target_name, stop_ticks, target_ticks, target_mode, target_r in target_profiles:
                for confirmation in ("123_or_trendline", "engulf_or_123", "fib382_plus1", "confluence2"):
                    for orderflow_mode in ("htf_structure", "ema21", "ema200", "adaptive_any"):
                        for engulf_mode in ("body", "full"):
                            for structure_123_mode in ("close_break", "wick_break"):
                                for stop_mode, padding in (("signal_candle", 2), ("signal_candle", 5), ("signal_or_otz", 5)):
                                    for trigger_zone_mode in ("wick_touch_any_close", "close_inside"):
                                        strategy_specs = []
                                        for priority_offset, htf_minutes in enumerate(htf_minutes_values):
                                            params = otz_continuation_params(
                                                stop_ticks=stop_ticks,
                                                target_ticks=target_ticks,
                                                allowed_hours=[],
                                                htf_minutes=htf_minutes,
                                                htf_lookback=8640 if htf_minutes == 5 else (2880 if htf_minutes == 15 else 720),
                                                zone_mode="engulfed_wick_ob",
                                                confirmation=confirmation,
                                                min_htf_impulse_atr=1.2,
                                                min_htf_body_pct=0.55,
                                                require_ema_context=True,
                                                enabled_symbols=["MGC", "MNQ", "MCL"],
                                                engulf_mode=engulf_mode,
                                                ob_wick_area_mode="lookback_cluster",
                                                ob_wick_area_bars=3,
                                                ob_wick_cluster_tolerance_ticks=120,
                                                structure_123_mode=structure_123_mode,
                                                trendline_source="wick",
                                                orderflow_mode=orderflow_mode,
                                                stop_mode=stop_mode,
                                                stop_padding_ticks=padding,
                                                target_mode=target_mode,
                                                target_r_multiple=target_r,
                                                trigger_zone_mode=trigger_zone_mode,
                                            )
                                            strategy_specs.append(spec("otz_continuation", 120 - priority_offset, params))
                                        risk = {"daily_profit_lock": daily_lock} if daily_lock > 0 else None
                                    rows.append(
                                        {
                                            "candidate_id": (
                                                f"otz_v2_all3_all_{htf_name}_obcluster_{confirmation}_{orderflow_mode}_"
                                                f"{engulf_mode}_{structure_123_mode}_{trigger_zone_mode}_"
                                                f"{stop_mode}p{padding}_{target_name}_day{daily_lock}"
                                            ),
                                            "strategy_config": strategy_config(strategy_specs, risk=risk),
                                        }
                                    )
    return rows


def prior_extreme_vwap_reversion_params(
    *,
    lookback: int,
    stop_ticks: int,
    target_ticks: int,
    min_vwap_distance_atr: float,
    rsi_pair: tuple[int, int],
    min_wick_ratio: float,
    max_signal_range_atr: float,
    allowed_hours: list[int],
) -> dict[str, Any]:
    return {
        "lookback": int(lookback),
        "breakout_buffer_ticks": 1,
        "reclaim_ticks": 0,
        "min_vwap_distance_atr": float(min_vwap_distance_atr),
        "rsi_oversold": int(rsi_pair[0]),
        "rsi_overbought": int(rsi_pair[1]),
        "min_wick_ratio": float(min_wick_ratio),
        "max_signal_range_atr": float(max_signal_range_atr),
        "sl_ticks": int(stop_ticks),
        "tp_ticks": int(target_ticks),
        "min_signal_gap_bars": 6,
        "allowed_entry_hours_et": list(allowed_hours),
        "symbol_params": {
            "MNQ": {"sl_ticks": int(stop_ticks), "tp_ticks": int(target_ticks)},
            "MGC": {"sl_ticks": max(12, int(stop_ticks // 2)), "tp_ticks": max(10, int(target_ticks // 2))},
            "MCL": {"sl_ticks": max(12, int(stop_ticks // 2)), "tp_ticks": max(10, int(target_ticks // 2))},
        },
    }


def prior_extreme_vwap_reversion_candidates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hour_profiles = {
        "rth": [9, 10, 11, 12, 13, 14, 15],
        "edge": [8, 9, 15, 19, 20, 21],
        "lunch": [11, 12, 13],
        "active": [0, 8, 9, 15],
    }
    for hours_name, hours in hour_profiles.items():
        for lookback in (8, 12, 20, 30):
            for stop_ticks, target_ticks in ((30, 24), (40, 32), (50, 40), (40, 60)):
                for vwap_dist in (0.5, 0.75, 1.0, 1.25):
                    for rsi_pair in ((25, 75), (30, 70), (35, 65)):
                        for wick in (0.35, 0.5):
                            params = prior_extreme_vwap_reversion_params(
                                lookback=lookback,
                                stop_ticks=stop_ticks,
                                target_ticks=target_ticks,
                                min_vwap_distance_atr=vwap_dist,
                                rsi_pair=rsi_pair,
                                min_wick_ratio=wick,
                                max_signal_range_atr=2.5,
                                allowed_hours=hours,
                            )
                            rows.append(
                                {
                                    "candidate_id": (
                                        f"pevwap_{hours_name}_lb{lookback}_sl{stop_ticks}_tp{target_ticks}_"
                                        f"vd{vwap_dist:g}_rsi{rsi_pair[0]}x{rsi_pair[1]}_wick{wick:g}"
                                    ),
                                    "strategy_config": strategy_config([spec("prior_extreme_vwap_reversion", 100, params)]),
                                }
                            )
    return rows


def resting_level_breakout_params(
    *,
    level_source: str,
    lookback: int,
    side_mode: str,
    stop_ticks: int,
    target_ticks: int,
    min_adx: float,
    allowed_hours: list[int],
    opening_range_end_et: str = "09:45",
) -> dict[str, Any]:
    return {
        "level_source": level_source,
        "lookback": int(lookback),
        "side_mode": side_mode,
        "breakout_buffer_ticks": 1,
        "sl_ticks": int(stop_ticks),
        "tp_ticks": int(target_ticks),
        "min_range_ticks": 8,
        "max_range_ticks": 500,
        "min_adx": float(min_adx),
        "min_signal_gap_bars": 5,
        "allowed_entry_hours_et": list(allowed_hours),
        "opening_range_start_et": "09:30",
        "opening_range_end_et": opening_range_end_et,
        "trade_window_end_et": "12:00",
        "symbol_params": {
            "MNQ": {"sl_ticks": int(stop_ticks), "tp_ticks": int(target_ticks), "min_range_ticks": 8, "max_range_ticks": 500},
            "MGC": {
                "sl_ticks": max(12, int(stop_ticks // 2)),
                "tp_ticks": max(18, int(target_ticks // 2)),
                "min_range_ticks": 4,
                "max_range_ticks": 250,
            },
            "MCL": {
                "sl_ticks": max(12, int(stop_ticks // 2)),
                "tp_ticks": max(18, int(target_ticks // 2)),
                "min_range_ticks": 4,
                "max_range_ticks": 250,
            },
        },
    }


def resting_level_breakout_candidates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lookback in (12, 20, 30, 48):
        for stop_ticks, rr in ((24, 1.5), (30, 1.5), (40, 2.0), (55, 2.0)):
            target_ticks = int(round(stop_ticks * rr))
            for min_adx in (0.0, 12.0, 18.0):
                params = resting_level_breakout_params(
                    level_source="rolling_donchian",
                    lookback=lookback,
                    side_mode="ema_bias",
                    stop_ticks=stop_ticks,
                    target_ticks=target_ticks,
                    min_adx=min_adx,
                    allowed_hours=[0, 1, 8, 9, 10, 13, 14, 15],
                )
                rows.append(
                    {
                        "candidate_id": f"rlb_donch_active_lb{lookback}_sl{stop_ticks}_tp{target_ticks}_adx{min_adx:g}",
                        "strategy_config": strategy_config([spec("resting_level_breakout", 100, params)]),
                    }
                )
    for end_name, end_time in (("or5", "09:35"), ("or15", "09:45"), ("or30", "10:00")):
        for stop_ticks, target_ticks in ((24, 36), (30, 45), (40, 60), (40, 80)):
            params = resting_level_breakout_params(
                level_source="opening_range",
                lookback=20,
                side_mode="vwap_bias",
                stop_ticks=stop_ticks,
                target_ticks=target_ticks,
                min_adx=0.0,
                allowed_hours=[],
                opening_range_end_et=end_time,
            )
            rows.append(
                {
                    "candidate_id": f"rlb_or_mgc_first_{end_name}_sl{stop_ticks}_tp{target_ticks}",
                    "strategy_config": strategy_config([spec("resting_level_breakout", 100, params)]),
                }
            )
    return rows


def volatility_squeeze_breakout_params(
    *,
    bb_period: int,
    bb_mult: float,
    squeeze_lookback: int,
    squeeze_quantile: float,
    stop_ticks: int,
    target_ticks: int,
    volume_ratio: float,
    allowed_hours: list[int],
) -> dict[str, Any]:
    return {
        "bb_period": int(bb_period),
        "bb_mult": float(bb_mult),
        "squeeze_lookback": int(squeeze_lookback),
        "squeeze_quantile": float(squeeze_quantile),
        "breakout_buffer_ticks": 1,
        "min_volume_ratio": float(volume_ratio),
        "volume_lookback": 20,
        "side_mode": "ema_bias",
        "sl_ticks": int(stop_ticks),
        "tp_ticks": int(target_ticks),
        "min_signal_gap_bars": 8,
        "allowed_entry_hours_et": list(allowed_hours),
        "symbol_params": {
            "MNQ": {"sl_ticks": int(stop_ticks), "tp_ticks": int(target_ticks)},
            "MGC": {"sl_ticks": max(12, int(stop_ticks // 2)), "tp_ticks": max(18, int(target_ticks // 2))},
            "MCL": {"sl_ticks": max(12, int(stop_ticks // 2)), "tp_ticks": max(18, int(target_ticks // 2))},
        },
    }


def volatility_squeeze_breakout_candidates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hour_profiles = {
        "all": [],
        "edge": [8, 9, 15, 19, 20, 21],
        "active": [0, 1, 8, 9, 10, 13, 14, 15],
    }
    for hours_name, hours in hour_profiles.items():
        for bb_period in (20, 30):
            for bb_mult in (1.5, 2.0):
                for squeeze_quantile in (0.15, 0.25):
                    for stop_ticks, target_ticks in ((30, 45), (40, 60), (40, 80), (55, 110)):
                        for volume_ratio in (0.0, 1.2):
                            params = volatility_squeeze_breakout_params(
                                bb_period=bb_period,
                                bb_mult=bb_mult,
                                squeeze_lookback=80,
                                squeeze_quantile=squeeze_quantile,
                                stop_ticks=stop_ticks,
                                target_ticks=target_ticks,
                                volume_ratio=volume_ratio,
                                allowed_hours=hours,
                            )
                            rows.append(
                                {
                                    "candidate_id": (
                                        f"vsq_{hours_name}_bb{bb_period}_m{bb_mult:g}_q{squeeze_quantile:g}_"
                                        f"sl{stop_ticks}_tp{target_ticks}_vol{volume_ratio:g}"
                                    ),
                                    "strategy_config": strategy_config([spec("volatility_squeeze_breakout", 100, params)]),
                                }
                            )
    return rows


def htf_level_retest_params(
    *,
    mode: str,
    htf_minutes: int,
    level_lookback: int,
    stop_ticks: int,
    target_ticks: int,
    allowed_hours: list[int],
    enabled_symbols: list[str] | None = None,
    allowed_sides: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "htf_minutes": int(htf_minutes),
        "level_lookback": int(level_lookback),
        "break_buffer_ticks": 1,
        "retest_tolerance_ticks": 4,
        "reclaim_ticks": 1,
        "sl_ticks": int(stop_ticks),
        "tp_ticks": int(target_ticks),
        "min_signal_gap_bars": 5,
        "allowed_entry_hours_et": list(allowed_hours),
        "enabled_symbols": list(enabled_symbols or []),
        "allowed_sides": list(allowed_sides or []),
        "symbol_params": {
            "MNQ": {"sl_ticks": int(stop_ticks), "tp_ticks": int(target_ticks)},
            "MGC": {"sl_ticks": max(12, int(stop_ticks // 2)), "tp_ticks": max(18, int(target_ticks // 2))},
            "MCL": {"sl_ticks": max(12, int(stop_ticks // 2)), "tp_ticks": max(18, int(target_ticks // 2))},
        },
    }


def htf_level_retest_candidates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hour_profiles = {
        "edge": [8, 9, 15, 19, 20, 21],
        "rth": [8, 9, 10, 11, 12, 13, 14, 15],
        "active": [0, 1, 8, 9, 10, 13, 14, 15],
    }
    for hours_name, hours in hour_profiles.items():
        for mode in ("break_retest", "gap_fill"):
            for htf_minutes in (15, 30):
                for level_lookback in (8, 16, 24):
                    for stop_ticks, target_ticks in ((30, 45), (40, 60), (40, 80), (55, 110)):
                        params = htf_level_retest_params(
                            mode=mode,
                            htf_minutes=htf_minutes,
                            level_lookback=level_lookback,
                            stop_ticks=stop_ticks,
                            target_ticks=target_ticks,
                            allowed_hours=hours,
                        )
                        rows.append(
                            {
                                "candidate_id": (
                                    f"htflvl_{mode}_{hours_name}_{htf_minutes}m_lb{level_lookback}_"
                                    f"sl{stop_ticks}_tp{target_ticks}"
                                ),
                                "strategy_config": strategy_config([spec("htf_level_retest", 100, params)]),
                            }
                        )
    focus_hours = {
        "open": [8, 9, 10],
        "h9": [9],
        "rth": [8, 9, 10, 11, 12, 13, 14, 15],
    }
    symbol_profiles = {
        "mcl": ["MCL"],
        "mnq": ["MNQ"],
        "mgc": ["MGC"],
        "mnq_mcl": ["MNQ", "MCL"],
    }
    side_profiles = {
        "both": [],
        "buy": ["buy"],
        "sell": ["sell"],
    }
    for hours_name, hours in focus_hours.items():
        for symbols_name, symbols in symbol_profiles.items():
            for side_name, sides in side_profiles.items():
                for htf_minutes in (15, 30):
                    for level_lookback in (4, 8, 16):
                        for stop_ticks, target_ticks in ((30, 45), (40, 60), (45, 70), (55, 110)):
                            params = htf_level_retest_params(
                                mode="gap_fill",
                                htf_minutes=htf_minutes,
                                level_lookback=level_lookback,
                                stop_ticks=stop_ticks,
                                target_ticks=target_ticks,
                                allowed_hours=hours,
                                enabled_symbols=symbols,
                                allowed_sides=sides,
                            )
                            rows.append(
                                {
                                    "candidate_id": (
                                        f"htfgapfocus_{hours_name}_{symbols_name}_{side_name}_{htf_minutes}m_"
                                        f"lb{level_lookback}_sl{stop_ticks}_tp{target_ticks}"
                                    ),
                                    "strategy_config": strategy_config([spec("htf_level_retest", 100, params)]),
                                }
                            )
    for hours_name, hours in focus_hours.items():
        for symbols_name, symbols in symbol_profiles.items():
            for side_name, sides in side_profiles.items():
                for htf_minutes in (15, 30):
                    for level_lookback in (4, 8, 16, 24):
                        for stop_ticks, target_ticks in ((30, 45), (40, 60), (45, 70), (55, 110)):
                            params = htf_level_retest_params(
                                mode="break_retest",
                                htf_minutes=htf_minutes,
                                level_lookback=level_lookback,
                                stop_ticks=stop_ticks,
                                target_ticks=target_ticks,
                                allowed_hours=hours,
                                enabled_symbols=symbols,
                                allowed_sides=sides,
                            )
                            rows.append(
                                {
                                    "candidate_id": (
                                        f"htfretestfocus_{hours_name}_{symbols_name}_{side_name}_{htf_minutes}m_"
                                        f"lb{level_lookback}_sl{stop_ticks}_tp{target_ticks}"
                                    ),
                                    "strategy_config": strategy_config([spec("htf_level_retest", 100, params)]),
                                }
                            )
    return rows


def session_pivot_retest_params(
    *,
    level_source: str,
    mode: str,
    stop_ticks: int,
    target_ticks: int,
    allowed_hours: list[int],
    enabled_symbols: list[str],
    retest_tolerance_ticks: int = 4,
    allowed_sides: list[str] | None = None,
    enabled_level_names: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "level_source": level_source,
        "mode": mode,
        "break_buffer_ticks": 1,
        "retest_tolerance_ticks": int(retest_tolerance_ticks),
        "reclaim_ticks": 1,
        "sl_ticks": int(stop_ticks),
        "tp_ticks": int(target_ticks),
        "min_signal_gap_bars": 8,
        "allowed_entry_hours_et": list(allowed_hours),
        "allowed_sides": list(allowed_sides or []),
        "enabled_level_names": list(enabled_level_names or []),
        "enabled_symbols": list(enabled_symbols),
        "symbol_params": {
            "MNQ": {"sl_ticks": int(stop_ticks), "tp_ticks": int(target_ticks)},
            "MGC": {"sl_ticks": max(12, int(stop_ticks // 2)), "tp_ticks": max(18, int(target_ticks // 2))},
            "MCL": {"sl_ticks": max(12, int(stop_ticks // 2)), "tp_ticks": max(18, int(target_ticks // 2))},
        },
    }


def session_pivot_retest_candidates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for symbols_name, symbols in {
        "mcl": ["MCL"],
        "mnq": ["MNQ"],
        "mnq_mcl": ["MNQ", "MCL"],
        "all": [],
    }.items():
        for stop_ticks, target_ticks in ((35, 55), (40, 60), (45, 70), (50, 75), (55, 110)):
            for tolerance in (2, 4, 6):
                params = session_pivot_retest_params(
                    level_source="previous_rth",
                    mode="retest",
                    stop_ticks=stop_ticks,
                    target_ticks=target_ticks,
                    allowed_hours=[8, 9, 10],
                    enabled_symbols=symbols,
                    retest_tolerance_ticks=tolerance,
                )
                rows.append(
                    {
                        "candidate_id": (
                            f"spivtune_prev_rth_retest_open_{symbols_name}_"
                            f"sl{stop_ticks}_tp{target_ticks}_tol{tolerance}"
                        ),
                        "strategy_config": strategy_config([spec("session_pivot_retest", 100, params)]),
                    }
                )
    hour_profiles = {
        "open": [8, 9, 10],
        "edge": [8, 9, 15, 19, 20, 21],
        "rth": [8, 9, 10, 11, 12, 13, 14, 15],
    }
    symbol_profiles = {
        "all": [],
        "mcl": ["MCL"],
        "mnq": ["MNQ"],
        "mnq_mcl": ["MNQ", "MCL"],
        "mnq_mgc": ["MNQ", "MGC"],
        "mgc": ["MGC"],
    }
    for hours_name, hours in hour_profiles.items():
        for symbols_name, symbols in symbol_profiles.items():
            for level_source in ("previous_rth", "overnight", "floor_pivots"):
                for mode in ("retest", "reclaim"):
                    for stop_ticks, target_ticks in ((30, 45), (40, 60), (40, 80), (55, 110)):
                        params = session_pivot_retest_params(
                            level_source=level_source,
                            mode=mode,
                            stop_ticks=stop_ticks,
                            target_ticks=target_ticks,
                            allowed_hours=hours,
                            enabled_symbols=symbols,
                        )
                        rows.append(
                            {
                                "candidate_id": (
                                    f"spiv_{level_source}_{mode}_{hours_name}_{symbols_name}_"
                                    f"sl{stop_ticks}_tp{target_ticks}"
                                ),
                                "strategy_config": strategy_config([spec("session_pivot_retest", 100, params)]),
                            }
                        )
    focus_hours = {
        "h9": [9],
        "h8_9": [8, 9],
        "h9_10": [9, 10],
        "open": [8, 9, 10],
    }
    side_profiles = {
        "both": [],
        "sell": ["sell"],
        "buy": ["buy"],
    }
    level_profiles = {
        "all": [],
        "high": ["high"],
        "low": ["low"],
    }
    for hours_name, hours in focus_hours.items():
        for side_name, sides in side_profiles.items():
            for level_name, level_names in level_profiles.items():
                for stop_ticks, target_ticks in ((32, 50), (35, 55), (40, 60), (45, 70), (50, 75)):
                    for tolerance in (4, 6):
                        params = session_pivot_retest_params(
                            level_source="previous_rth",
                            mode="retest",
                            stop_ticks=stop_ticks,
                            target_ticks=target_ticks,
                            allowed_hours=hours,
                            enabled_symbols=["MCL"],
                            retest_tolerance_ticks=tolerance,
                            allowed_sides=sides,
                            enabled_level_names=level_names,
                        )
                        rows.append(
                            {
                                "candidate_id": (
                                    f"spivfocus_prev_rth_retest_{hours_name}_mcl_{side_name}_{level_name}_"
                                    f"sl{stop_ticks}_tp{target_ticks}_tol{tolerance}"
                                ),
                                "strategy_config": strategy_config([spec("session_pivot_retest", 100, params)]),
                            }
                        )
    return rows


def causal_mcl_blend_candidates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    htf_core = htf_level_retest_params(
        mode="break_retest",
        htf_minutes=15,
        level_lookback=16,
        stop_ticks=40,
        target_ticks=60,
        allowed_hours=[9],
        enabled_symbols=["MCL"],
    )
    htf_wide = htf_level_retest_params(
        mode="break_retest",
        htf_minutes=15,
        level_lookback=16,
        stop_ticks=45,
        target_ticks=70,
        allowed_hours=[8, 9, 10],
        enabled_symbols=["MCL"],
    )
    pivot_h9 = session_pivot_retest_params(
        level_source="previous_rth",
        mode="retest",
        stop_ticks=45,
        target_ticks=70,
        allowed_hours=[9],
        enabled_symbols=["MCL"],
        retest_tolerance_ticks=6,
    )
    pivot_open = session_pivot_retest_params(
        level_source="previous_rth",
        mode="retest",
        stop_ticks=35,
        target_ticks=55,
        allowed_hours=[8, 9, 10],
        enabled_symbols=["MCL"],
        retest_tolerance_ticks=6,
    )
    specs = {
        "htf_h9_lb16_sl40_tp60": [spec("htf_level_retest", 110, htf_core)],
        "htf_h9_lb16_sl40_tp60_pivot_h9_sl45_tp70": [
            spec("htf_level_retest", 110, htf_core),
            spec("session_pivot_retest", 100, pivot_h9),
        ],
        "htf_h9_lb16_sl40_tp60_pivot_open_sl35_tp55": [
            spec("htf_level_retest", 110, htf_core),
            spec("session_pivot_retest", 100, pivot_open),
        ],
        "htf_open_lb16_sl45_tp70_pivot_h9_sl45_tp70": [
            spec("htf_level_retest", 110, htf_wide),
            spec("session_pivot_retest", 100, pivot_h9),
        ],
        "htf_open_lb16_sl45_tp70_pivot_open_sl35_tp55": [
            spec("htf_level_retest", 110, htf_wide),
            spec("session_pivot_retest", 100, pivot_open),
        ],
    }
    for name, strategy_specs in specs.items():
        rows.append({"candidate_id": f"causalmcl_{name}", "strategy_config": strategy_config(strategy_specs)})

    htf_sell = htf_level_retest_params(
        mode="break_retest",
        htf_minutes=15,
        level_lookback=4,
        stop_ticks=30,
        target_ticks=45,
        allowed_hours=[8, 9, 10],
        enabled_symbols=["MCL"],
        allowed_sides=["sell"],
    )
    for hours_name, hours in {
        "h9": [9],
        "h8_9": [8, 9],
        "h9_10": [9, 10],
        "open": [8, 9, 10],
    }.items():
        for stop_ticks, target_ticks in ((35, 55), (40, 60)):
            pivot = session_pivot_retest_params(
                level_source="previous_rth",
                mode="retest",
                stop_ticks=stop_ticks,
                target_ticks=target_ticks,
                allowed_hours=hours,
                enabled_symbols=["MCL"],
                retest_tolerance_ticks=6,
            )
            rows.append(
                {
                    "candidate_id": (
                        f"causalmcl_htf_h9_lb16_sl40_tp60_pivot_{hours_name}_"
                        f"sl{stop_ticks}_tp{target_ticks}_tol6"
                    ),
                    "strategy_config": strategy_config(
                        [
                            spec("htf_level_retest", 110, htf_core),
                            spec("session_pivot_retest", 100, pivot),
                        ]
                    ),
                }
            )
            rows.append(
                {
                    "candidate_id": (
                        f"causalmcl_htfsell_open_lb4_sl30_tp45_pivot_{hours_name}_"
                        f"sl{stop_ticks}_tp{target_ticks}_tol6"
                    ),
                    "strategy_config": strategy_config(
                        [
                            spec("htf_level_retest", 110, htf_sell),
                            spec("session_pivot_retest", 100, pivot),
                        ]
                    ),
                }
            )
    return rows


def legacy_candidates() -> list[dict[str, Any]]:
    enabled_sets = {
        "all_legacy": {
            "ichimoku": True,
            "fx_at_one_glance": True,
            "ema_pullback": True,
            "asian_breakout": True,
            "sss": True,
            "trend_following": False,
        },
        "trend_breakout": {
            "ichimoku": False,
            "fx_at_one_glance": False,
            "ema_pullback": True,
            "asian_breakout": True,
            "sss": True,
            "trend_following": True,
        },
        "signal_stack": {
            "ichimoku": True,
            "fx_at_one_glance": True,
            "ema_pullback": True,
            "asian_breakout": False,
            "sss": False,
            "trend_following": True,
        },
    }
    rows: list[dict[str, Any]] = []
    for set_name, enabled in enabled_sets.items():
        for stop_ticks, target_ticks in ((30, 60), (40, 80), (55, 110)):
            for asian_rr in (1.25, 1.75):
                for confluence in (2, 3):
                    strategies = [
                        spec("ichimoku", 100, {"min_confluence": max(4, confluence + 2), "sl_ticks": stop_ticks, "tp_ticks": target_ticks}, enabled["ichimoku"]),
                        spec(
                            "fx_at_one_glance",
                            95,
                            {
                                "min_confluence": max(6, confluence + 4),
                                "min_signal_gap_bars": 24,
                                "min_score_edge": 2,
                                "require_breakout": True,
                                "sl_ticks": stop_ticks,
                                "tp_ticks": target_ticks,
                            },
                            enabled["fx_at_one_glance"],
                        ),
                        spec("ema_pullback", 90, {"sl_ticks": stop_ticks, "tp_ticks": target_ticks}, enabled["ema_pullback"]),
                        spec(
                            "asian_breakout",
                            85,
                            {"sl_ticks": stop_ticks, "tp_ticks": target_ticks, "rr_ratio": asian_rr},
                            enabled["asian_breakout"],
                        ),
                        spec(
                            "sss",
                            80,
                            {"sl_ticks": stop_ticks, "tp_ticks": target_ticks, "min_confluence": confluence},
                            enabled["sss"],
                        ),
                        spec(
                            "trend_following",
                            70,
                            {"sl_ticks": stop_ticks, "tp_ticks": target_ticks, "adx_threshold_delta": 0},
                            enabled["trend_following"],
                        ),
                    ]
                    rows.append(
                        {
                            "candidate_id": f"legacy_{set_name}_sl{stop_ticks}_tp{target_ticks}_rr{asian_rr:g}_conf{confluence}",
                            "strategy_config": strategy_config(strategies, blender_mode="priority_first"),
                        }
                    )
    return rows


def legacy_strategy_specs(
    *,
    enabled: dict[str, bool],
    stop_ticks: int,
    target_ticks: int,
    asian_rr: float,
    confluence: int,
    priority_offset: int = 0,
    fx_signal_gap_bars: int = 24,
    fx_score_edge: int = 2,
    trend_adx_delta: int = 0,
) -> list[dict[str, Any]]:
    return [
        spec(
            "ichimoku",
            100 - priority_offset,
            {"min_confluence": max(4, confluence + 2), "sl_ticks": stop_ticks, "tp_ticks": target_ticks},
            enabled.get("ichimoku", False),
        ),
        spec(
            "fx_at_one_glance",
            95 - priority_offset,
            {
                "min_confluence": max(6, confluence + 4),
                "min_signal_gap_bars": fx_signal_gap_bars,
                "min_score_edge": fx_score_edge,
                "require_breakout": True,
                "sl_ticks": stop_ticks,
                "tp_ticks": target_ticks,
            },
            enabled.get("fx_at_one_glance", False),
        ),
        spec("ema_pullback", 90 - priority_offset, {"sl_ticks": stop_ticks, "tp_ticks": target_ticks}, enabled.get("ema_pullback", False)),
        spec(
            "asian_breakout",
            85 - priority_offset,
            {"sl_ticks": stop_ticks, "tp_ticks": target_ticks, "rr_ratio": asian_rr},
            enabled.get("asian_breakout", False),
        ),
        spec(
            "sss",
            80 - priority_offset,
            {"sl_ticks": stop_ticks, "tp_ticks": target_ticks, "min_confluence": confluence},
            enabled.get("sss", False),
        ),
        spec(
            "trend_following",
            70 - priority_offset,
            {"sl_ticks": stop_ticks, "tp_ticks": target_ticks, "adx_threshold_delta": trend_adx_delta},
            enabled.get("trend_following", False),
        ),
    ]


def golden_zone_legacy_candidates() -> list[dict[str, Any]]:
    gz_stop_profiles = {
        "balanced": {"MGC": {"max_stop_ticks": 45}, "MCL": {"max_stop_ticks": 45}, "MNQ": {"max_stop_ticks": 65}},
        "wide": {"MGC": {"max_stop_ticks": 65}, "MCL": {"max_stop_ticks": 65}, "MNQ": {"max_stop_ticks": 90}},
    }
    legacy_sets = {
        "all_no_trend": {
            "ichimoku": True,
            "fx_at_one_glance": True,
            "ema_pullback": True,
            "asian_breakout": True,
            "sss": True,
            "trend_following": False,
        },
        "breakout_sss": {
            "ichimoku": False,
            "fx_at_one_glance": True,
            "ema_pullback": False,
            "asian_breakout": True,
            "sss": True,
            "trend_following": False,
        },
        "pullback_sss": {
            "ichimoku": False,
            "fx_at_one_glance": False,
            "ema_pullback": True,
            "asian_breakout": False,
            "sss": True,
            "trend_following": False,
        },
        "signal_quality": {
            "ichimoku": True,
            "fx_at_one_glance": True,
            "ema_pullback": True,
            "asian_breakout": False,
            "sss": False,
            "trend_following": False,
        },
    }
    rows: list[dict[str, Any]] = []
    for gz_stop_name, gz_stops in gz_stop_profiles.items():
        for gz_rr in (1.5, 2.0):
            for gz_adx in (10.0, 14.0):
                gz = golden_zone_params(
                    risk_reward=gz_rr,
                    min_adx=gz_adx,
                    lookback=48,
                    max_stop_ticks=gz_stops,
                    require_engulfing=False,
                )
                for legacy_name, enabled in legacy_sets.items():
                    for stop_ticks, target_ticks in ((40, 80), (55, 110)):
                        for confluence in (3, 4):
                            legacy = legacy_strategy_specs(
                                enabled=enabled,
                                stop_ticks=stop_ticks,
                                target_ticks=target_ticks,
                                asian_rr=1.25,
                                confluence=confluence,
                                priority_offset=20,
                                fx_signal_gap_bars=36 if confluence >= 4 else 24,
                                fx_score_edge=3 if confluence >= 4 else 2,
                            )
                            strategies = [spec("golden_zone_scalper", 120, gz)] + legacy
                            rows.append(
                                {
                                    "candidate_id": (
                                        f"gzleg_{gz_stop_name}_gzrr{gz_rr:g}_adx{gz_adx:g}_"
                                        f"{legacy_name}_sl{stop_ticks}_tp{target_ticks}_conf{confluence}"
                                    ),
                                    "strategy_config": strategy_config(strategies, blender_mode="priority_first"),
                                }
                            )
    return rows


def evaluate_candidate(
    candidate: Candidate,
    data: dict[str, dict[str, Any]],
    combine_rules: dict[str, dict[str, Any]],
    *,
    routing_mode: str,
) -> dict[str, Any]:
    by_symbol: dict[str, Any] = {}
    all_trades: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        frame = data[symbol]["frame"]
        bt = Backtester(
            symbol,
            initial_balance=50000,
            strategy_config=copy.deepcopy(candidate.strategy_config),
            telemetry_enabled=False,
            screenshot_config={"enabled": False},
            run_id=f"portfolio_{candidate.candidate_id}_{symbol}",
            routing_mode=routing_mode,
        )
        prepared = bt.strategy._calculate_indicators(bt._normalize_dataframe(frame))
        result = bt.run_dataframe(frame, dataset_label=str(data[symbol]["path"]), prepared_indicators=prepared)
        symbol_trades = []
        for trade in result.get("trades", []):
            normalized = normalize_portfolio_trade(trade, symbol, bt.commission_per_contract)
            symbol_trades.append(normalized)
            all_trades.append(normalized)
        corrected_pnl = round(sum(float(trade.get("pnl", 0.0)) for trade in symbol_trades), 2)
        raw_trade_pnl = round(sum(float(trade.get("raw_pnl", trade.get("pnl", 0.0))) for trade in symbol_trades), 2)
        final_balance_net = round(float(result.get("summary", {}).get("final_balance", 50000.0) or 50000.0) - 50000.0, 2)
        by_symbol[symbol] = {
            "csv_path": data[symbol]["path"],
            "summary": result.get("summary", {}),
            "by_strategy": result.get("by_strategy", {}),
            "trade_count": len(symbol_trades),
            "pnl": corrected_pnl,
            "raw_trade_pnl": raw_trade_pnl,
            "final_balance_net": final_balance_net,
            "entry_commission_adjustment": round(raw_trade_pnl - corrected_pnl, 2),
        }

    portfolio_result = build_portfolio_result(all_trades)
    combine = {key: evaluate_combine_rules(portfolio_result, rules) for key, rules in combine_rules.items()}
    metrics = portfolio_metrics(portfolio_result, data)
    return {
        "candidate_id": candidate.candidate_id,
        "candidate_family": candidate.family,
        "routing_mode": routing_mode,
        "strategy_config": candidate.strategy_config,
        "metrics": metrics,
        "combine": combine,
        "per_symbol": by_symbol,
        "portfolio_trades": portfolio_result["trades"],
    }


def worker_evaluate(args_tuple):
    candidate, data, combine_rules, routing_mode = args_tuple
    try:
        evaluation = evaluate_candidate(candidate, data, combine_rules, routing_mode=routing_mode)
        row = flatten_evaluation(evaluation)
    except Exception as exc:
        evaluation = {
            "candidate_id": candidate.candidate_id,
            "candidate_family": candidate.family,
            "routing_mode": routing_mode,
            "error": f"{type(exc).__name__}: {exc}",
            "strategy_config": candidate.strategy_config,
        }
        row = {
            "candidate_id": candidate.candidate_id,
            "candidate_family": candidate.family,
            "routing_mode": routing_mode,
            "error": evaluation["error"],
            "total_pnl": 0.0,
            "total_trades": 0,
            "score": -1_000_000.0,
        }
    return {"evaluation": evaluation, "row": row}


def normalize_portfolio_trade(trade: dict[str, Any], symbol: str, commission_per_contract: float) -> dict[str, Any]:
    normalized = copy.deepcopy(trade)
    raw_pnl = float(normalized.get("pnl", 0.0) or 0.0)
    entry_commission = abs(float(normalized.get("qty", 1) or 1)) * float(commission_per_contract)
    normalized["symbol"] = symbol
    normalized["raw_pnl"] = round(raw_pnl, 2)
    normalized["entry_commission_adjustment"] = round(entry_commission, 2)
    normalized["pnl"] = round(raw_pnl - entry_commission, 2)
    return normalized


def build_portfolio_result(trades: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(trades, key=lambda trade: str(trade.get("time_close") or trade.get("exit_time") or ""))
    balance = 50000.0
    equity_history = [balance]
    equity_curve: list[dict[str, Any]] = []
    wins = losses = scratches = 0
    gross_profit = gross_loss = 0.0
    for trade in ordered:
        pnl = float(trade.get("pnl", 0.0) or 0.0)
        balance += pnl
        equity_history.append(balance)
        equity_curve.append({"timestamp": trade.get("time_close") or trade.get("exit_time"), "equity": round(balance, 2)})
        if pnl > 0:
            wins += 1
            gross_profit += pnl
        elif pnl < 0:
            losses += 1
            gross_loss += abs(pnl)
        else:
            scratches += 1
    total_pnl = sum(float(trade.get("pnl", 0.0) or 0.0) for trade in ordered)
    total_trades = len(ordered)
    drawdown = max_drawdown(equity_history)
    profit_factor: float | str
    if gross_loss > 0:
        profit_factor = round(gross_profit / gross_loss, 2)
    else:
        profit_factor = "Inf" if gross_profit > 0 else 0.0
    return {
        "summary": {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "scratches": scratches,
            "win_rate": round((wins / total_trades) * 100.0, 2) if total_trades else 0.0,
            "total_pnl": round(total_pnl, 2),
            "final_balance": round(balance, 2),
            "profit_factor": profit_factor,
            "max_drawdown": round(drawdown, 2),
            "avg_trade": round(total_pnl / total_trades, 2) if total_trades else 0.0,
        },
        "trades": ordered,
        "equity_history": equity_history,
        "equity_curve": equity_curve,
    }


def max_drawdown(equity_history: list[float]) -> float:
    peak = -math.inf
    worst = 0.0
    for value in equity_history:
        peak = max(peak, float(value))
        worst = min(worst, float(value) - peak)
    return worst


def portfolio_metrics(portfolio_result: dict[str, Any], data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    trades = portfolio_result.get("trades", [])
    summary = dict(portfolio_result.get("summary", {}))
    daily = daily_pnl(trades)
    data_start, data_end = data_span(data)
    if daily:
        trade_days = len(daily)
        span_days = max(1, (max(daily) - min(daily)).days + 1)
    elif data_start is not None and data_end is not None:
        trade_days = 0
        span_days = max(1, (data_end - data_start).days + 1)
    else:
        trade_days = 0
        span_days = 1
    if data_start is not None and data_end is not None:
        calendar_span_days = max(1, (data_end - data_start).days + 1)
    else:
        calendar_span_days = span_days

    total_pnl = float(summary.get("total_pnl", 0.0) or 0.0)
    best_day = max(daily.values(), default=0.0)
    worst_day = min(daily.values(), default=0.0)
    total_profit = max(total_pnl, 0.0)
    best_day_consistency = (best_day / total_profit * 100.0) if total_profit > 0 and best_day > 0 else 0.0
    summary.update(
        {
            "avg_daily_pnl_trade_dates": round(total_pnl / trade_days, 2) if trade_days else 0.0,
            "avg_daily_pnl_calendar_span": round(total_pnl / calendar_span_days, 2) if calendar_span_days else 0.0,
            "worst_daily_loss": round(worst_day, 2),
            "best_daily_pnl": round(best_day, 2),
            "best_day_consistency_pct": round(best_day_consistency, 2),
            "trade_days": trade_days,
            "calendar_span_days": calendar_span_days,
            "trades_per_trade_day": round(len(trades) / trade_days, 2) if trade_days else 0.0,
            "trades_per_calendar_day": round(len(trades) / calendar_span_days, 2) if calendar_span_days else 0.0,
            "daily_pnl": {day.isoformat(): round(value, 2) for day, value in sorted(daily.items())},
        }
    )
    return summary


def daily_pnl(trades: list[dict[str, Any]]) -> dict[Any, float]:
    daily: dict[Any, float] = {}
    for trade in trades:
        raw_ts = trade.get("time_close") or trade.get("exit_time")
        if not raw_ts:
            continue
        ts = pd.to_datetime(str(raw_ts), utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        day = ts.date()
        daily[day] = daily.get(day, 0.0) + float(trade.get("pnl", 0.0) or 0.0)
    return daily


def data_span(data: dict[str, dict[str, Any]]):
    starts = []
    ends = []
    for item in data.values():
        frame = item["frame"]
        if "timestamp" not in frame.columns or frame.empty:
            continue
        stamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
        if stamps.empty:
            continue
        starts.append(stamps.min().date())
        ends.append(stamps.max().date())
    if not starts or not ends:
        return None, None
    return min(starts), max(ends)


def flatten_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(evaluation.get("metrics", {}))
    combine = evaluation.get("combine", {})
    topstep = combine.get("topstep_50k", {})
    daily = combine.get("topstep_50k_daily_enforced", {})
    row = {
        "candidate_id": evaluation.get("candidate_id"),
        "candidate_family": evaluation.get("candidate_family"),
        "routing_mode": evaluation.get("routing_mode"),
        "total_pnl": metrics.get("total_pnl", 0.0),
        "avg_daily_pnl_trade_dates": metrics.get("avg_daily_pnl_trade_dates", 0.0),
        "avg_daily_pnl_calendar_span": metrics.get("avg_daily_pnl_calendar_span", 0.0),
        "worst_daily_loss": metrics.get("worst_daily_loss", 0.0),
        "best_daily_pnl": metrics.get("best_daily_pnl", 0.0),
        "best_day_consistency_pct": metrics.get("best_day_consistency_pct", 0.0),
        "max_drawdown": metrics.get("max_drawdown", 0.0),
        "total_trades": metrics.get("total_trades", 0),
        "trades_per_trade_day": metrics.get("trades_per_trade_day", 0.0),
        "trades_per_calendar_day": metrics.get("trades_per_calendar_day", 0.0),
        "win_rate": metrics.get("win_rate", 0.0),
        "profit_factor": metrics.get("profit_factor", 0.0),
        "topstep_50k_passed": bool(topstep.get("passed", False)),
        "topstep_50k_daily_enforced_passed": bool(daily.get("passed", False)),
        "topstep_50k_mll_gap": nested(topstep, "checks", "max_loss_limit", "min_mll_gap", default=0.0),
        "topstep_50k_daily_mll_gap": nested(daily, "checks", "max_loss_limit", "min_mll_gap", default=0.0),
        "topstep_50k_best_day_pct": nested(topstep, "checks", "consistency", "actual_best_day_pct", default=0.0),
        "topstep_50k_worst_day_loss": nested(topstep, "checks", "daily_loss_limit", "worst_day_loss", default=0.0),
        "score": score_metrics(metrics, topstep, daily),
        "per_symbol_json": json.dumps(evaluation.get("per_symbol", {}), sort_keys=True, default=str),
        "config_json": json.dumps(evaluation.get("strategy_config", {}), sort_keys=True, default=str),
    }
    if evaluation.get("error"):
        row["error"] = evaluation["error"]
    return row


def nested(value: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def profit_factor_number(value: Any) -> float:
    if value == "Inf":
        return 10.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def score_metrics(metrics: dict[str, Any], topstep: dict[str, Any], daily: dict[str, Any]) -> float:
    trades = float(metrics.get("total_trades", 0) or 0)
    pnl = float(metrics.get("total_pnl", 0.0) or 0.0)
    pf = profit_factor_number(metrics.get("profit_factor", 0.0))
    dd = abs(float(metrics.get("max_drawdown", 0.0) or 0.0))
    consistency = float(metrics.get("best_day_consistency_pct", 100.0) or 0.0)
    worst_day = abs(min(0.0, float(metrics.get("worst_daily_loss", 0.0) or 0.0)))
    pass_bonus = 12_000.0 if topstep.get("passed") else 0.0
    daily_bonus = 4_000.0 if daily.get("passed") else 0.0
    trade_penalty = max(0.0, trades - 450.0) * 3.0
    consistency_penalty = max(0.0, consistency - 50.0) * 30.0
    if trades < 10:
        return round(-500_000.0 + pnl + trades, 3)
    return round(pass_bonus + daily_bonus + pnl + pf * 900.0 - dd * 0.6 - worst_day * 0.25 - consistency_penalty - trade_penalty, 3)


def sorted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            bool(row.get("topstep_50k_passed", False)),
            bool(row.get("topstep_50k_daily_enforced_passed", False)),
            float(row.get("score", -1_000_000.0) or -1_000_000.0),
            float(row.get("total_pnl", 0.0) or 0.0),
        ),
        reverse=True,
    )


def candidate_to_dict(candidate: Candidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "candidate_family": candidate.family,
        "strategy_config": candidate.strategy_config,
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(SUMMARY_FIELDS)
    extras = sorted({key for row in rows for key in row.keys()} - set(fields))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + extras, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    run_dir: Path,
    ranked_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    data: dict[str, dict[str, Any]],
    *,
    elapsed_seconds: float,
) -> None:
    lines = [
        "# Portfolio Mix Optimization",
        "",
        "Offline search report. Combine-rule pass/fail values are reporting metrics and do not alter live routing.",
        "Portfolio PnL is commission-corrected: trade records keep `raw_pnl`, and reported portfolio `pnl` subtracts entry-side commission because the backtester trade record already includes exit-side commission only.",
        "",
        f"- Data dir: `{args.data_dir}`",
        f"- Output dir: `{run_dir}`",
        f"- Routing mode: `{args.routing_mode}`",
        f"- Final routing mode: `{args.final_routing_mode or 'disabled'}`",
        f"- Tail bars: `{args.tail_bars}`",
        f"- Candidates evaluated: `{len(ranked_rows)}`",
        f"- Elapsed seconds: `{elapsed_seconds:.1f}`",
        "",
        "## Data Files",
        "",
    ]
    for symbol in SYMBOLS:
        frame = data[symbol]["frame"]
        start = frame["timestamp"].min() if "timestamp" in frame.columns and not frame.empty else ""
        end = frame["timestamp"].max() if "timestamp" in frame.columns and not frame.empty else ""
        lines.append(f"- `{symbol}`: `{data[symbol]['path']}` ({len(frame)} bars, {start} to {end})")
    lines.extend(
        [
            "",
            "## Top Search Candidates",
            "",
            markdown_table(ranked_rows[: min(25, len(ranked_rows))]),
            "",
        ]
    )
    if final_rows:
        lines.extend(["## Finalist Re-Runs", "", markdown_table(final_rows[: min(25, len(final_rows))]), ""])
    lines.extend(
        [
            "## Artifacts",
            "",
            "- `candidate_scores.csv`: ranked flat metrics.",
            "- `candidate_details.json`: full metrics, combine evaluations, per-symbol summaries, and configs.",
            "- `candidates.json`: generated candidate configs.",
            "- `best_config.json`: highest-ranked search candidate config.",
            "- `best_trades.csv`: closed trades for the highest-ranked search candidate.",
        ]
    )
    (run_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    cols = [
        "candidate_id",
        "candidate_family",
        "routing_mode",
        "total_pnl",
        "avg_daily_pnl_trade_dates",
        "worst_daily_loss",
        "best_day_consistency_pct",
        "max_drawdown",
        "total_trades",
        "trades_per_trade_day",
        "profit_factor",
        "topstep_50k_passed",
        "topstep_50k_daily_enforced_passed",
        "score",
    ]
    header = "| " + " | ".join(cols) + " |"
    divider = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = []
    for row in rows:
        values = [format_md_value(row.get(col, "")) for col in cols]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider] + body)


def format_md_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())
