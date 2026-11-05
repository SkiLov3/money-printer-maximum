#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.optimize_portfolio_mix import build_candidates, build_portfolio_result, evaluate_candidate, load_symbol_data
from src.config.settings import get_combine_rules, load_stack_config
from src.optimization.combine_rules import evaluate_combine_rules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trade-order permutation test for one portfolio optimizer candidate.")
    parser.add_argument("--config", default="config/trading_stack.yaml")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument(
        "--candidate-details",
        default="",
        help="Optional candidate_details.json to reuse cached portfolio_trades instead of rerunning the backtest.",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="reports/permutation_tests")
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--routing-mode", default="optimizer_fast")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    out_dir = Path(args.output_dir) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)

    previous_stack = os.environ.get("STACK_CONFIG_PATH")
    os.environ["STACK_CONFIG_PATH"] = str(Path(args.config).resolve())
    try:
        stack_config = load_stack_config(args.config)
        if args.candidate_details:
            real = load_cached_evaluation(Path(args.candidate_details), str(args.candidate_id), stack_config)
        else:
            candidate = find_candidate(stack_config, str(args.candidate_id))
            data = load_symbol_data(Path(args.data_dir), tail_bars=0)
            combine_rules = {
                "topstep_50k": get_combine_rules(stack_config, "topstep_50k"),
                "topstep_50k_daily_enforced": get_combine_rules(stack_config, "topstep_50k_daily_enforced"),
            }
            real = evaluate_candidate(candidate, data, combine_rules, routing_mode=str(args.routing_mode))
        rows = run_permutations(real["portfolio_trades"], stack_config, int(args.reps), int(args.seed))
    finally:
        if previous_stack is None:
            os.environ.pop("STACK_CONFIG_PATH", None)
        else:
            os.environ["STACK_CONFIG_PATH"] = previous_stack

    payload = {
        "candidate_id": args.candidate_id,
        "config": args.config,
        "routing_mode": args.routing_mode,
        "mode": "trade_pnl_shuffle_across_original_close_timestamps",
        "reps": int(args.reps),
        "real": summarize_real(real),
        "permutation": summarize_rows(rows),
        "rows": rows,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    (out_dir / "permutation.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    write_report(out_dir, payload)
    print(json.dumps({k: payload[k] for k in ("candidate_id", "real", "permutation")}, indent=2, default=str))
    print(f"Report: {out_dir / 'analysis.md'}")
    return 0


def find_candidate(stack_config: dict[str, Any], candidate_id: str):
    matches = [candidate for candidate in build_candidates(stack_config) if candidate.candidate_id == candidate_id]
    if not matches:
        raise ValueError(f"Unknown candidate id: {candidate_id}")
    return matches[0]


def load_cached_evaluation(path: Path, candidate_id: str, stack_config: dict[str, Any]) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    matches = [item for item in data if item.get("candidate_id") == candidate_id]
    if not matches:
        raise ValueError(f"Candidate {candidate_id!r} not found in {path}")
    evaluation = copy.deepcopy(matches[0])
    portfolio = build_portfolio_result(evaluation.get("portfolio_trades", []))
    rules = {
        "topstep_50k": get_combine_rules(stack_config, "topstep_50k"),
        "topstep_50k_daily_enforced": get_combine_rules(stack_config, "topstep_50k_daily_enforced"),
    }
    evaluation["combine"] = {key: evaluate_combine_rules(portfolio, rule) for key, rule in rules.items()}
    metrics = dict(evaluation.get("metrics", {}))
    metrics.update(portfolio.get("summary", {}))
    evaluation["metrics"] = metrics
    evaluation["portfolio_trades"] = portfolio["trades"]
    return evaluation


def run_permutations(trades: list[dict[str, Any]], stack_config: dict[str, Any], reps: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    ordered = sorted(trades, key=lambda trade: str(trade.get("time_close") or trade.get("exit_time") or ""))
    pnls = [float(trade.get("pnl", 0.0) or 0.0) for trade in ordered]
    rules = get_combine_rules(stack_config, "topstep_50k_daily_enforced")
    day_keys = [topstep_trading_day(trade.get("time_close") or trade.get("exit_time") or trade.get("timestamp")) for trade in ordered]
    rows: list[dict[str, Any]] = []
    for rep in range(1, reps + 1):
        shuffled = list(pnls)
        rng.shuffle(shuffled)
        result = fast_combine_metrics(shuffled, day_keys, rules)
        rows.append(
            {
                "rep": rep,
                "passed": result["passed"],
                "breached": result["breached"],
                "total_pnl": result["total_pnl"],
                "max_drawdown": result["max_drawdown"],
                "min_mll_gap": result["min_mll_gap"],
                "best_day_pct": result["best_day_pct"],
                "worst_day_loss": result["worst_day_loss"],
            }
        )
    return rows


def topstep_trading_day(raw_timestamp: Any) -> str:
    ts = pd.to_datetime(raw_timestamp, utc=True, errors="coerce")
    if pd.isna(ts):
        return "unknown"
    local = ts.tz_convert("America/Chicago")
    if (local.hour, local.minute) >= (17, 0):
        local = local + pd.Timedelta(days=1)
    return local.date().isoformat()


def fast_combine_metrics(pnls: list[float], day_keys: list[str], rules: dict[str, Any]) -> dict[str, Any]:
    starting_balance = float(rules.get("starting_balance", 50000))
    profit_target = float(rules.get("profit_target", 3000))
    max_loss_limit = float(rules.get("max_loss_limit", 2000))
    consistency_limit = float(rules.get("consistency_max_best_day_pct", 50))
    daily_loss_limit = float(rules.get("daily_loss_limit", 1000))
    evaluate_daily_loss = bool(rules.get("evaluate_daily_loss", False))

    balance = starting_balance
    peak = starting_balance
    max_dd = 0.0
    mll = starting_balance - max_loss_limit
    current_day = None
    day_end_equity = starting_balance
    min_balance = starting_balance
    min_mll_gap = float("inf")
    breached = False
    daily: dict[str, float] = {}

    for pnl, day in zip(pnls, day_keys):
        if current_day is None:
            current_day = day
        elif day != current_day:
            mll = min(starting_balance, max(mll, day_end_equity - max_loss_limit))
            current_day = day
        balance += float(pnl)
        peak = max(peak, balance)
        max_dd = min(max_dd, balance - peak)
        min_balance = min(min_balance, balance)
        min_mll_gap = min(min_mll_gap, balance - mll)
        if balance <= mll:
            breached = True
        day_end_equity = balance
        daily[day] = daily.get(day, 0.0) + float(pnl)

    total_pnl = round(balance - starting_balance, 2)
    total_profit = max(total_pnl, 0.0)
    best_day_profit = max([value for value in daily.values() if value > 0], default=0.0)
    best_day_pct = round((best_day_profit / total_profit * 100.0) if total_profit > 0 else 0.0, 2)
    worst_day_loss = abs(min(daily.values())) if daily else 0.0
    passed = (
        not breached
        and total_pnl >= profit_target
        and best_day_pct <= consistency_limit
        and (worst_day_loss < daily_loss_limit if evaluate_daily_loss else True)
    )
    return {
        "passed": passed,
        "breached": breached,
        "total_pnl": total_pnl,
        "max_drawdown": round(max_dd, 2),
        "min_balance_seen": round(min_balance, 2),
        "min_mll_gap": round(min_mll_gap, 2),
        "best_day_pct": best_day_pct,
        "worst_day_loss": round(worst_day_loss, 2),
    }


def summarize_real(evaluation: dict[str, Any]) -> dict[str, Any]:
    metrics = evaluation["metrics"]
    checks = evaluation["combine"]["topstep_50k_daily_enforced"]["checks"]
    return {
        "passed": evaluation["combine"]["topstep_50k_daily_enforced"]["passed"],
        "total_pnl": metrics["total_pnl"],
        "avg_daily_pnl_trade_dates": metrics["avg_daily_pnl_trade_dates"],
        "total_trades": metrics["total_trades"],
        "profit_factor": metrics["profit_factor"],
        "max_drawdown": metrics["max_drawdown"],
        "min_mll_gap": checks["max_loss_limit"].get("min_mll_gap", 0.0),
        "best_day_pct": checks["consistency"].get("actual_best_day_pct", 0.0),
        "worst_day_loss": checks["daily_loss_limit"].get("worst_day_loss", 0.0),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return {
        "pass_rate": mean_bool(row["passed"] for row in rows),
        "breach_rate": mean_bool(row["breached"] for row in rows),
        "p05_mll_gap": quantile([float(row["min_mll_gap"]) for row in rows], 0.05),
        "median_mll_gap": statistics.median(float(row["min_mll_gap"]) for row in rows),
        "p95_best_day_pct": quantile([float(row["best_day_pct"]) for row in rows], 0.95),
        "median_best_day_pct": statistics.median(float(row["best_day_pct"]) for row in rows),
        "p95_worst_day_loss": quantile([float(row["worst_day_loss"]) for row in rows], 0.95),
        "median_worst_day_loss": statistics.median(float(row["worst_day_loss"]) for row in rows),
        "total_pnl": rows[0]["total_pnl"],
    }


def mean_bool(values) -> float:
    values = list(values)
    return round(sum(1 for value in values if value) / len(values), 4) if values else 0.0


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return round(float(ordered[idx]), 2)


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    real = payload["real"]
    perm = payload["permutation"]
    lines = [
        "# Candidate Permutation Test",
        "",
        f"- Candidate: `{payload['candidate_id']}`",
        f"- Config: `{payload['config']}`",
        f"- Mode: `{payload['mode']}`",
        f"- Reps: `{payload['reps']}`",
        f"- Elapsed seconds: `{payload['elapsed_seconds']}`",
        "",
        "## Real Path",
        "",
        f"- Passed: `{real['passed']}`",
        f"- Total PnL: `{real['total_pnl']}`",
        f"- Avg daily PnL: `{real['avg_daily_pnl_trade_dates']}`",
        f"- Trades: `{real['total_trades']}`",
        f"- Profit factor: `{real['profit_factor']}`",
        f"- Max drawdown: `{real['max_drawdown']}`",
        f"- Min MLL gap: `{real['min_mll_gap']}`",
        f"- Best-day pct: `{real['best_day_pct']}`",
        f"- Worst day loss: `{real['worst_day_loss']}`",
        "",
        "## Permutations",
        "",
        f"- Pass rate: `{perm.get('pass_rate')}`",
        f"- Breach rate: `{perm.get('breach_rate')}`",
        f"- p05 MLL gap: `{perm.get('p05_mll_gap')}`",
        f"- Median MLL gap: `{perm.get('median_mll_gap')}`",
        f"- p95 best-day pct: `{perm.get('p95_best_day_pct')}`",
        f"- Median best-day pct: `{perm.get('median_best_day_pct')}`",
        f"- p95 worst day loss: `{perm.get('p95_worst_day_loss')}`",
        f"- Median worst day loss: `{perm.get('median_worst_day_loss')}`",
    ]
    (out_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
