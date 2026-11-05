#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.optimize_portfolio_mix import (
    Candidate,
    SYMBOLS,
    build_candidates,
    build_portfolio_result,
    evaluate_candidate,
    find_data_file,
    flatten_evaluation,
)
from src.config.settings import get_combine_rules, load_stack_config
from src.optimization.combine_rules import evaluate_combine_rules
from src.utils.backtester import Backtester


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward and holdout validation for one optimizer candidate.")
    parser.add_argument("--candidate-id", default="", help="Candidate id to regenerate from the optimizer search space.")
    parser.add_argument(
        "--candidate-config",
        default="",
        help="Path to a saved best_config.json artifact. Prefer this for exact promoted-candidate validation.",
    )
    parser.add_argument("--config", default="config/trading_stack.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="reports/walk_forward")
    parser.add_argument("--train-pct", type=float, default=0.60)
    parser.add_argument("--validation-pct", type=float, default=0.20)
    parser.add_argument("--risk", type=float, default=0.18, help="Risk percent to apply during this validation.")
    parser.add_argument("--scenario-set", choices=("baseline", "core"), default="core")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    out_dir = Path(args.output_dir) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_stack_config(args.config)
    config.setdefault("routing", {})["risk_per_trade_pct"] = float(args.risk)
    candidate = load_candidate(config, str(args.candidate_id), str(args.candidate_config))
    args.candidate_id = candidate.candidate_id
    full_data = load_full_data(Path(args.data_dir))
    windows = build_windows(full_data, float(args.train_pct), float(args.validation_pct))
    scenarios = scenario_definitions(float(args.risk), str(args.scenario_set))

    previous = os.environ.get("STACK_CONFIG_PATH")
    tmp = Path(tempfile.gettempdir()) / f"walk_forward_stack_{os.getpid()}.yaml"
    tmp.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    os.environ["STACK_CONFIG_PATH"] = str(tmp)
    try:
        rows: list[dict[str, Any]] = []
        details: list[dict[str, Any]] = []
        for window in windows:
            window_data = slice_data(full_data, window["start"], window["end"])
            combine_rules = {
                "topstep_50k": get_combine_rules(config, "topstep_50k"),
                "topstep_50k_daily_enforced": get_combine_rules(config, "topstep_50k_daily_enforced"),
            }
            evaluation = evaluate_candidate(candidate, window_data, combine_rules, routing_mode="optimizer_fast")
            baseline_row = flatten_evaluation(evaluation)
            baseline_row.update(
                {
                    "window": window["name"],
                    "window_start": window["start"].isoformat(),
                    "window_end": window["end"].isoformat(),
                    "scenario": "baseline",
                    "risk_per_trade_pct": float(args.risk),
                }
            )
            rows.append(baseline_row)
            details.append({"window": window, "scenario": "baseline", "evaluation": evaluation})
            print(
                f"{window['name']} baseline pnl={baseline_row['total_pnl']} "
                f"avg_day={baseline_row['avg_daily_pnl_trade_dates']} "
                f"trades={baseline_row['total_trades']} pass={baseline_row['topstep_50k_daily_enforced_passed']}",
                flush=True,
            )

            for scenario in scenarios:
                if scenario["name"] == "baseline":
                    continue
                stress = run_stress_scenario(
                    candidate.strategy_config,
                    config,
                    window_data,
                    scenario,
                )
                stress.update(
                    {
                        "candidate_id": candidate.candidate_id,
                        "candidate_family": candidate.family,
                        "routing_mode": "optimizer_fast",
                        "window": window["name"],
                        "window_start": window["start"].isoformat(),
                        "window_end": window["end"].isoformat(),
                        "scenario": scenario["name"],
                    }
                )
                rows.append(stress)
                details.append({"window": window, "scenario": scenario["name"], "metrics": stress})
                print(
                    f"{window['name']} {scenario['name']} pnl={stress['total_pnl']} "
                    f"avg_day={stress['avg_daily_pnl_trade_dates']} trades={stress['total_trades']} "
                    f"pass={stress['topstep_50k_daily_enforced_passed']}",
                    flush=True,
                )

        write_outputs(out_dir, rows, details, args, windows, elapsed=time.perf_counter() - started)
    finally:
        if previous is None:
            os.environ.pop("STACK_CONFIG_PATH", None)
        else:
            os.environ["STACK_CONFIG_PATH"] = previous
        try:
            tmp.unlink()
        except OSError:
            pass

    print(f"Report: {out_dir / 'analysis.md'}")
    return 0


def load_candidate(config: dict[str, Any], candidate_id: str, candidate_config: str) -> Candidate:
    if candidate_config:
        payload = json.loads(Path(candidate_config).read_text(encoding="utf-8"))
        artifact_id = str(payload.get("candidate_id") or candidate_id or Path(candidate_config).stem)
        if candidate_id and artifact_id != candidate_id:
            raise ValueError(f"Candidate config id {artifact_id!r} does not match requested id {candidate_id!r}")
        strategy_config = payload.get("strategy_config")
        if not isinstance(strategy_config, dict):
            raise ValueError(f"Candidate config must contain a strategy_config object: {candidate_config}")
        return Candidate(artifact_id, str(payload.get("candidate_family", "artifact")), strategy_config)
    if not candidate_id:
        raise ValueError("Either --candidate-id or --candidate-config is required.")
    return find_candidate(config, candidate_id)


def find_candidate(config: dict[str, Any], candidate_id: str) -> Candidate:
    matches = [candidate for candidate in build_candidates(config) if candidate.candidate_id == candidate_id]
    if not matches:
        raise ValueError(f"Unknown candidate id: {candidate_id}")
    return matches[0]


def load_full_data(data_dir: Path) -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = {}
    for symbol in SYMBOLS:
        path = find_data_file(data_dir, symbol)
        frame = pd.read_csv(path)
        frame.columns = [str(col).lower() for col in frame.columns]
        frame.loc[:, "timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["timestamp"]).reset_index(drop=True)
        data[symbol] = {"path": str(path), "frame": frame}
    return data


def build_windows(data: dict[str, dict[str, Any]], train_pct: float, validation_pct: float) -> list[dict[str, Any]]:
    dates = sorted(
        {
            stamp.date()
            for item in data.values()
            for stamp in pd.to_datetime(item["frame"]["timestamp"], utc=True, errors="coerce").dropna()
        }
    )
    if len(dates) < 5:
        raise ValueError("Need at least 5 dates for walk-forward validation.")
    train_end_idx = max(1, min(len(dates) - 3, int(len(dates) * train_pct) - 1))
    validation_end_idx = max(train_end_idx + 1, min(len(dates) - 2, int(len(dates) * (train_pct + validation_pct)) - 1))
    return [
        {"name": "train", "start": dates[0], "end": dates[train_end_idx]},
        {"name": "validation", "start": dates[train_end_idx + 1], "end": dates[validation_end_idx]},
        {"name": "holdout", "start": dates[validation_end_idx + 1], "end": dates[-1]},
    ]


def slice_data(data: dict[str, dict[str, Any]], start, end) -> dict[str, dict[str, Any]]:
    sliced: dict[str, dict[str, Any]] = {}
    for symbol, item in data.items():
        frame = item["frame"]
        dates = frame["timestamp"].dt.date
        filtered = frame[(dates >= start) & (dates <= end)].reset_index(drop=True)
        sliced[symbol] = {"path": item["path"], "frame": filtered}
    return sliced


def scenario_definitions(risk: float, scenario_set: str) -> list[dict[str, Any]]:
    scenarios = [{"name": "baseline", "risk": risk, "slip": 1, "commission": 0.62, "delay": 0}]
    if scenario_set == "core":
        scenarios.extend(
            [
                {"name": "slip_2_ticks", "risk": risk, "slip": 2, "commission": 0.62, "delay": 0},
                {"name": "slip_3_ticks", "risk": risk, "slip": 3, "commission": 0.62, "delay": 0},
                {"name": "double_commission", "risk": risk, "slip": 1, "commission": 1.24, "delay": 0},
                {"name": "slip_2_double_commission", "risk": risk, "slip": 2, "commission": 1.24, "delay": 0},
                {"name": "delay_1_bar", "risk": risk, "slip": 1, "commission": 0.62, "delay": 1},
            ]
        )
    return scenarios


def run_stress_scenario(
    strategy_config: dict[str, Any],
    base_config: dict[str, Any],
    data: dict[str, dict[str, Any]],
    scenario: dict[str, Any],
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config.setdefault("routing", {})["risk_per_trade_pct"] = float(scenario["risk"])
    tmp = Path(tempfile.gettempdir()) / f"walk_forward_scenario_{os.getpid()}_{scenario['name']}.yaml"
    tmp.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    previous = os.environ.get("STACK_CONFIG_PATH")
    os.environ["STACK_CONFIG_PATH"] = str(tmp)
    try:
        all_trades: list[dict[str, Any]] = []
        for symbol in SYMBOLS:
            frame = data[symbol]["frame"].copy()
            if frame.empty:
                continue
            if int(scenario["delay"]) > 0:
                frame = delay_ohlc(frame, int(scenario["delay"]))
            bt = Backtester(
                symbol,
                initial_balance=50000,
                strategy_config=copy.deepcopy(strategy_config),
                telemetry_enabled=False,
                screenshot_config={"enabled": False},
                routing_mode="optimizer_fast",
            )
            bt.slippage_ticks = int(scenario["slip"])
            bt.commission_per_contract = float(scenario["commission"])
            prepared = bt.strategy._calculate_indicators(bt._normalize_dataframe(frame))
            result = bt.run_dataframe(frame, dataset_label=data[symbol]["path"], prepared_indicators=prepared)
            for trade in result.get("trades", []):
                item = copy.deepcopy(trade)
                entry_commission = abs(float(item.get("qty", 1))) * float(scenario["commission"])
                item["pnl"] = round(float(item.get("pnl", 0.0)) - entry_commission, 2)
                item["entry_commission_adjustment"] = round(entry_commission, 2)
                item["symbol"] = symbol
                all_trades.append(item)
        portfolio = build_portfolio_result(all_trades)
        stack = load_stack_config()
        combine = {
            "topstep_50k": evaluate_combine_rules(portfolio, get_combine_rules(stack, "topstep_50k")),
            "topstep_50k_daily_enforced": evaluate_combine_rules(portfolio, get_combine_rules(stack, "topstep_50k_daily_enforced")),
        }
        row = flatten_stress(portfolio, combine)
        row.update(
            {
                "risk_per_trade_pct": float(scenario["risk"]),
                "slippage_ticks": int(scenario["slip"]),
                "commission_per_side": float(scenario["commission"]),
                "delay_bars": int(scenario["delay"]),
            }
        )
        return row
    finally:
        if previous is None:
            os.environ.pop("STACK_CONFIG_PATH", None)
        else:
            os.environ["STACK_CONFIG_PATH"] = previous
        try:
            tmp.unlink()
        except OSError:
            pass


def delay_ohlc(frame: pd.DataFrame, bars: int) -> pd.DataFrame:
    delayed = frame.copy()
    for col in ("open", "high", "low", "close", "price"):
        if col in delayed.columns:
            delayed.loc[:, col] = delayed[col].shift(-bars)
    delayed = delayed.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    if "price" not in delayed.columns:
        delayed.loc[:, "price"] = delayed["close"]
    return delayed


def flatten_stress(portfolio: dict[str, Any], combine: dict[str, Any]) -> dict[str, Any]:
    summary = portfolio["summary"]
    topstep = combine["topstep_50k"]
    daily = combine["topstep_50k_daily_enforced"]
    daily_pnl = daily.get("daily_pnl", {})
    trade_days = len(daily_pnl)
    return {
        "total_pnl": summary["total_pnl"],
        "avg_daily_pnl_trade_dates": round(summary["total_pnl"] / trade_days, 2) if trade_days else 0.0,
        "worst_daily_loss": daily["checks"]["daily_loss_limit"]["worst_day_loss"],
        "best_day_consistency_pct": daily["checks"]["consistency"]["actual_best_day_pct"],
        "max_drawdown": summary["max_drawdown"],
        "total_trades": summary["total_trades"],
        "profit_factor": summary["profit_factor"],
        "topstep_50k_passed": topstep["passed"],
        "topstep_50k_daily_enforced_passed": daily["passed"],
        "topstep_50k_daily_mll_gap": daily["checks"]["max_loss_limit"].get("min_mll_gap", 0.0),
        "daily_pnl": daily_pnl,
    }


def write_outputs(
    out_dir: Path,
    rows: list[dict[str, Any]],
    details: list[dict[str, Any]],
    args: argparse.Namespace,
    windows: list[dict[str, Any]],
    *,
    elapsed: float,
) -> None:
    pd.DataFrame(rows).to_csv(out_dir / "walk_forward_results.csv", index=False)
    (out_dir / "walk_forward_details.json").write_text(json.dumps(details, indent=2, default=str), encoding="utf-8")
    lines = [
        "# Walk-Forward Candidate Validation",
        "",
        f"- Candidate: `{args.candidate_id}`",
        f"- Config: `{args.config}`",
        f"- Risk per trade: `{args.risk}`",
        f"- Scenario set: `{args.scenario_set}`",
        f"- Elapsed seconds: `{elapsed:.1f}`",
        "",
        "## Windows",
        "",
    ]
    for window in windows:
        lines.append(f"- `{window['name']}`: {window['start']} to {window['end']}")
    lines.extend(["", "## Results", "", markdown_table(rows), ""])
    lines.extend(
        [
            "## Files",
            "",
            f"- Results: `{out_dir / 'walk_forward_results.csv'}`",
            f"- Details: `{out_dir / 'walk_forward_details.json'}`",
        ]
    )
    (out_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    cols = [
        "window",
        "scenario",
        "total_pnl",
        "avg_daily_pnl_trade_dates",
        "total_trades",
        "profit_factor",
        "best_day_consistency_pct",
        "worst_daily_loss",
        "max_drawdown",
        "topstep_50k_daily_enforced_passed",
    ]
    out = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(format_value(row.get(col, "")) for col in cols) + " |")
    return "\n".join(out)


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())
