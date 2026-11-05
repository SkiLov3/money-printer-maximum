#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import get_combine_rules, load_stack_config
from src.optimization.combine_rules import evaluate_combine_rules
from src.utils.backtester import Backtester


SYMBOL_FILES = {
    "MNQM6": "data/mnqm6_1m_20260501_20260610.csv",
    "MGCM6": "data/mgcq6_1m_20260501_20260610.csv",
    "MCLM6": "data/mcln6_1m_20260501_20260610.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stress-test a promoted strategy config under harsher execution assumptions.")
    parser.add_argument("--config", default="config/trading_stack.yaml")
    parser.add_argument("--candidate-id", default="", help="Optional optimize_portfolio_mix candidate id to stress instead of config strategies.")
    parser.add_argument("--output-dir", default="reports/stress_tests")
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--reps", type=int, default=200, help="Monte Carlo reps for randomized scenarios.")
    parser.add_argument(
        "--scenario-set",
        choices=("core", "full"),
        default="full",
        help="Use 'core' for deterministic baseline/slippage/commission scenarios only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.getLogger("src.agents.risk_agent").setLevel(logging.ERROR)
    started = time.perf_counter()
    out_dir = Path(args.output_dir) / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_stack_config(args.config)
    strategy_config = resolve_strategy_config(config, str(args.candidate_id or ""))
    base_risk = float(config.get("routing", {}).get("risk_per_trade_pct", 0.18))

    scenarios = core_scenarios(base_risk)
    if args.scenario_set == "full":
        scenarios.extend(
            [
                {"name": "skip_10pct", "risk": base_risk, "slip": 1, "commission": 0.62, "skip": 0.10, "delay": 0, "reps": args.reps},
                {"name": "delay_1_bar", "risk": base_risk, "slip": 1, "commission": 0.62, "skip": 0.0, "delay": 1, "reps": 1},
                {"name": "delay_2_bars", "risk": base_risk, "slip": 1, "commission": 0.62, "skip": 0.0, "delay": 2, "reps": 1},
                {"name": "slip_2_skip_10pct", "risk": base_risk, "slip": 2, "commission": 0.62, "skip": 0.10, "delay": 0, "reps": args.reps},
                {"name": "slip_2_delay_1_double_commission", "risk": base_risk, "slip": 2, "commission": 1.24, "skip": 0.0, "delay": 1, "reps": 1},
                {"name": "risk_0_12_baseline", "risk": 0.12, "slip": 1, "commission": 0.62, "skip": 0.0, "delay": 0, "reps": 1},
            ]
        )

    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for scenario in scenarios:
        reps = int(scenario.pop("reps"))
        scenario_rows = []
        for rep in range(max(1, reps)):
            row, detail = run_scenario(
                scenario,
                strategy_config=strategy_config,
                base_config=config,
                seed=int(args.seed) + rep,
            )
            row["scenario"] = scenario["name"]
            row["rep"] = rep + 1
            scenario_rows.append(row)
            if rep == 0:
                detail["scenario"] = scenario["name"]
                details.append(detail)
        rows.extend(scenario_rows)
        frame = pd.DataFrame(rows)
        frame.to_csv(out_dir / "stress_results.csv", index=False)
        (out_dir / "stress_details.json").write_text(json.dumps(details, indent=2, default=str), encoding="utf-8")
        write_report(out_dir, frame, args, elapsed=time.perf_counter() - started)
        print(summarize_scenario(scenario["name"], scenario_rows), flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "stress_results.csv", index=False)
    (out_dir / "stress_details.json").write_text(json.dumps(details, indent=2, default=str), encoding="utf-8")
    write_report(out_dir, frame, args, elapsed=time.perf_counter() - started)
    print(f"Report: {out_dir / 'analysis.md'}")
    return 0


def resolve_strategy_config(config: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    if not candidate_id:
        return config["strategies"]
    from scripts.optimize_portfolio_mix import build_candidates

    matches = [candidate for candidate in build_candidates(config) if candidate.candidate_id == candidate_id]
    if not matches:
        raise ValueError(f"Unknown candidate id: {candidate_id}")
    return matches[0].strategy_config


def core_scenarios(base_risk: float) -> list[dict[str, Any]]:
    return [
        {"name": "baseline", "risk": base_risk, "slip": 1, "commission": 0.62, "skip": 0.0, "delay": 0, "reps": 1},
        {"name": "slip_2_ticks", "risk": base_risk, "slip": 2, "commission": 0.62, "skip": 0.0, "delay": 0, "reps": 1},
        {"name": "slip_3_ticks", "risk": base_risk, "slip": 3, "commission": 0.62, "skip": 0.0, "delay": 0, "reps": 1},
        {"name": "double_commission", "risk": base_risk, "slip": 1, "commission": 1.24, "skip": 0.0, "delay": 0, "reps": 1},
        {"name": "slip_2_double_commission", "risk": base_risk, "slip": 2, "commission": 1.24, "skip": 0.0, "delay": 0, "reps": 1},
    ]


def run_scenario(
    scenario: dict[str, Any],
    *,
    strategy_config: dict[str, Any],
    base_config: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = copy.deepcopy(base_config)
    config.setdefault("routing", {})["risk_per_trade_pct"] = float(scenario["risk"])
    tmp = Path(tempfile.gettempdir()) / f"stress_stack_{os.getpid()}_{seed}.yaml"
    tmp.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    previous = os.environ.get("STACK_CONFIG_PATH")
    os.environ["STACK_CONFIG_PATH"] = str(tmp)
    rng = random.Random(seed)

    try:
        all_trades: list[dict[str, Any]] = []
        per_symbol: dict[str, Any] = {}
        for symbol, path in SYMBOL_FILES.items():
            frame = pd.read_csv(path)
            frame.columns = [str(col).lower() for col in frame.columns]
            if "timestamp" in frame.columns:
                frame.loc[:, "timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
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
            result = bt.run_dataframe(frame, dataset_label=path, prepared_indicators=prepared)
            per_symbol[symbol] = result["summary"]
            for trade in result.get("trades", []):
                if float(scenario["skip"]) > 0 and rng.random() < float(scenario["skip"]):
                    continue
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
        metrics = summarize(portfolio, combine)
        metrics.update(
            {
                "risk_per_trade_pct": float(scenario["risk"]),
                "slippage_ticks": int(scenario["slip"]),
                "commission_per_side": float(scenario["commission"]),
                "skip_rate": float(scenario["skip"]),
                "delay_bars": int(scenario["delay"]),
            }
        )
        return metrics, {"scenario": scenario, "metrics": metrics, "per_symbol": per_symbol, "combine": combine}
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


def build_portfolio_result(trades: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(trades, key=lambda trade: str(trade.get("time_close", trade.get("time_open", ""))))
    balance = 50000.0
    equity = [balance]
    equity_curve: list[dict[str, Any]] = []
    for trade in ordered:
        balance += float(trade.get("pnl", 0.0))
        equity.append(balance)
        equity_curve.append({"timestamp": trade.get("time_close") or trade.get("exit_time"), "equity": round(balance, 2)})
    wins = sum(1 for trade in ordered if float(trade.get("pnl", 0.0)) > 0)
    losses = sum(1 for trade in ordered if float(trade.get("pnl", 0.0)) < 0)
    gross_profit = sum(float(trade.get("pnl", 0.0)) for trade in ordered if float(trade.get("pnl", 0.0)) > 0)
    gross_loss = abs(sum(float(trade.get("pnl", 0.0)) for trade in ordered if float(trade.get("pnl", 0.0)) < 0))
    equity_series = pd.Series(equity, dtype="float64")
    max_drawdown = float((equity_series - equity_series.cummax()).min()) if not equity_series.empty else 0.0
    return {
        "summary": {
            "total_trades": len(ordered),
            "wins": wins,
            "losses": losses,
            "scratches": len(ordered) - wins - losses,
            "win_rate": round((wins / len(ordered) * 100.0) if ordered else 0.0, 2),
            "total_pnl": round(balance - 50000.0, 2),
            "final_balance": round(balance, 2),
            "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss else "Inf",
            "max_drawdown": round(max_drawdown, 2),
            "avg_trade": round((balance - 50000.0) / len(ordered), 2) if ordered else 0.0,
            "sharpe_ratio": 0.0,
        },
        "trades": ordered,
        "equity_history": equity,
        "equity_curve": equity_curve,
    }


def summarize(portfolio: dict[str, Any], combine: dict[str, Any]) -> dict[str, Any]:
    summary = portfolio["summary"]
    daily = combine["topstep_50k_daily_enforced"].get("daily_pnl", {})
    avg_day = sum(daily.values()) / len(daily) if daily else 0.0
    daily_eval = combine["topstep_50k_daily_enforced"]["checks"]
    return {
        "total_pnl": summary["total_pnl"],
        "avg_daily_pnl": round(avg_day, 2),
        "total_trades": summary["total_trades"],
        "profit_factor": summary["profit_factor"],
        "max_drawdown": summary["max_drawdown"],
        "topstep_50k_passed": combine["topstep_50k"]["passed"],
        "topstep_50k_daily_passed": combine["topstep_50k_daily_enforced"]["passed"],
        "best_day_pct": daily_eval["consistency"]["actual_best_day_pct"],
        "worst_day_loss": daily_eval["daily_loss_limit"]["worst_day_loss"],
        "min_balance_seen": round(daily_eval["max_loss_limit"]["min_balance_seen"], 2),
    }


def summarize_scenario(name: str, rows: list[dict[str, Any]]) -> str:
    frame = pd.DataFrame(rows)
    return (
        f"{name}: pass_rate={frame['topstep_50k_daily_passed'].mean():.2%} "
        f"median_pnl={frame['total_pnl'].median():.2f} "
        f"p05_pnl={frame['total_pnl'].quantile(0.05):.2f} "
        f"median_avg_day={frame['avg_daily_pnl'].median():.2f} "
        f"p05_mll_balance={frame['min_balance_seen'].quantile(0.05):.2f}"
    )


def write_report(out_dir: Path, frame: pd.DataFrame, args: argparse.Namespace, *, elapsed: float) -> None:
    grouped = (
        frame.groupby("scenario")
        .agg(
            reps=("rep", "count"),
            pass_rate=("topstep_50k_daily_passed", "mean"),
            median_pnl=("total_pnl", "median"),
            p05_pnl=("total_pnl", lambda s: s.quantile(0.05)),
            median_avg_day=("avg_daily_pnl", "median"),
            p05_min_balance=("min_balance_seen", lambda s: s.quantile(0.05)),
            median_trades=("total_trades", "median"),
        )
        .reset_index()
    )
    lines = [
        "# Execution Stress Test",
        "",
        f"- Config: `{args.config}`",
        f"- Candidate ID: `{args.candidate_id or 'config strategies'}`",
        f"- Scenario set: `{args.scenario_set}`",
        f"- Reps: `{args.reps}` randomized scenarios",
        f"- Elapsed seconds: `{elapsed:.1f}`",
        "",
        "## Scenario Summary",
        "",
        markdown_table(grouped),
        "",
        "## Files",
        "",
        f"- Results: `{out_dir / 'stress_results.csv'}`",
        f"- Details: `{out_dir / 'stress_details.json'}`",
    ]
    (out_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: f"{value:.4f}")
    headers = [str(col) for col in display.columns]
    rows = [[str(value) for value in row] for row in display.to_numpy().tolist()]
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    out = [
        "| " + " | ".join(header.ljust(width) for header, width in zip(headers, widths)) + " |",
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(value.ljust(width) for value, width in zip(row, widths)) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
