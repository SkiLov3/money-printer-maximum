#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.optimize_portfolio_mix import (  # noqa: E402, I001
    SYMBOLS,
    build_portfolio_result,
    find_data_file,
    flatten_evaluation,
    normalize_portfolio_trade,
    portfolio_metrics,
)
from src.config.settings import get_combine_rules, load_stack_config  # noqa: E402
from src.optimization.combine_rules import evaluate_combine_rules  # noqa: E402
from src.utils.backtester import Backtester  # noqa: E402


TOPSTEPX_MICRO_COMMISSION_PER_SIDE = {
    "MNQM6": 0.62,  # $1.24 round turn
    "MGCM6": 0.87,  # $1.74 round turn
    "MCLM6": 0.77,  # $1.54 round turn
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strict no-lookahead validation. Signals are generated from a completed bar, "
            "then executed only at the next bar open."
        )
    )
    parser.add_argument(
        "--candidate-config",
        default="reports/portfolio_mix_optimizer_corrected_full/20260502T235622Z/best_config.json",
        help="Saved best_config.json artifact containing the promoted strategy_config.",
    )
    parser.add_argument("--config", default="config/trading_stack.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="reports/no_lookahead")
    parser.add_argument("--risk", type=float, default=0.12)
    parser.add_argument("--scenario-set", choices=("baseline", "core"), default="core")
    parser.add_argument(
        "--execution-model",
        choices=("next_bar_open", "resting_stop"),
        default="next_bar_open",
        help="Use next-bar market/open fills or resting stop orders from signal metadata.",
    )
    parser.add_argument("--max-pending-bars", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    out_dir = Path(args.output_dir) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_stack_config(args.config)
    config.setdefault("routing", {})["risk_per_trade_pct"] = float(args.risk)
    try:
        candidate_payload = json.loads(Path(args.candidate_config).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load candidate config {args.candidate_config}: {exc}") from exc
    strategy_config = candidate_payload.get("strategy_config")
    if not isinstance(strategy_config, dict):
        raise ValueError(f"Missing strategy_config object in {args.candidate_config}")
    candidate_id = str(candidate_payload.get("candidate_id") or Path(args.candidate_config).stem)
    data = load_data(Path(args.data_dir))
    combine_rules = {
        "topstep_50k": get_combine_rules(config, "topstep_50k"),
        "topstep_50k_daily_enforced": get_combine_rules(config, "topstep_50k_daily_enforced"),
    }

    scenarios = scenario_definitions(args.scenario_set)
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    best_trades: list[dict[str, Any]] = []
    for scenario in scenarios:
        if args.execution_model == "resting_stop":
            evaluation = evaluate_resting_stop_orders(
                candidate_id=candidate_id,
                strategy_config=strategy_config,
                config=config,
                data=data,
                combine_rules=combine_rules,
                scenario=scenario,
                max_pending_bars=int(args.max_pending_bars),
            )
        else:
            evaluation = evaluate_next_bar_open(
                candidate_id=candidate_id,
                strategy_config=strategy_config,
                config=config,
                data=data,
                combine_rules=combine_rules,
                scenario=scenario,
            )
        row = flatten_evaluation(evaluation)
        row.update(
            {
                "scenario": scenario["name"],
                "execution_model": args.execution_model,
                "slippage_ticks": scenario["slippage_ticks"],
                "commission_model": scenario["commission_model"],
            }
        )
        rows.append(row)
        details.append(evaluation)
        if not best_trades:
            best_trades = evaluation.get("portfolio_trades", [])
        print(
            f"{scenario['name']} pnl={row['total_pnl']} avg_day={row['avg_daily_pnl_trade_dates']} "
            f"trades={row['total_trades']} pf={row['profit_factor']} "
            f"pass={row['topstep_50k_daily_enforced_passed']}",
            flush=True,
        )

    write_csv(out_dir / "no_lookahead_results.csv", rows)
    write_csv(out_dir / "no_lookahead_trades_first_scenario.csv", best_trades)
    (out_dir / "no_lookahead_details.json").write_text(json.dumps(details, indent=2, default=str), encoding="utf-8")
    write_report(out_dir, rows, args, candidate_id, elapsed=time.perf_counter() - started)
    print(f"Report: {out_dir / 'analysis.md'}")
    return 0


def load_data(data_dir: Path) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for symbol in SYMBOLS:
        path = find_data_file(data_dir, symbol)
        frame = pd.read_csv(path)
        frame.columns = [str(col).lower() for col in frame.columns]
        if "timestamp" in frame.columns:
            frame.loc[:, "timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        loaded[symbol] = {"path": str(path), "frame": frame}
    return loaded


def scenario_definitions(scenario_set: str) -> list[dict[str, Any]]:
    scenarios = [
        {
            "name": "next_bar_open_repo_fee_slip1",
            "slippage_ticks": 1,
            "commission_model": "repo_uniform_micro_1.24rt",
            "commission_per_side": {},
            "commission_multiplier": 1.0,
        },
        {
            "name": "next_bar_open_topstep_fee_slip1",
            "slippage_ticks": 1,
            "commission_model": "topstepx_product_specific",
            "commission_per_side": TOPSTEPX_MICRO_COMMISSION_PER_SIDE,
            "commission_multiplier": 1.0,
        },
    ]
    if scenario_set == "core":
        scenarios.extend(
            [
                {
                    "name": "next_bar_open_topstep_fee_slip2",
                    "slippage_ticks": 2,
                    "commission_model": "topstepx_product_specific",
                    "commission_per_side": TOPSTEPX_MICRO_COMMISSION_PER_SIDE,
                    "commission_multiplier": 1.0,
                },
                {
                    "name": "next_bar_open_topstep_fee_slip3",
                    "slippage_ticks": 3,
                    "commission_model": "topstepx_product_specific",
                    "commission_per_side": TOPSTEPX_MICRO_COMMISSION_PER_SIDE,
                    "commission_multiplier": 1.0,
                },
                {
                    "name": "next_bar_open_topstep_fee_slip2_double_commission",
                    "slippage_ticks": 2,
                    "commission_model": "topstepx_product_specific_x2",
                    "commission_per_side": TOPSTEPX_MICRO_COMMISSION_PER_SIDE,
                    "commission_multiplier": 2.0,
                },
            ]
        )
    return scenarios


def evaluate_next_bar_open(
    *,
    candidate_id: str,
    strategy_config: dict[str, Any],
    config: dict[str, Any],
    data: dict[str, dict[str, Any]],
    combine_rules: dict[str, dict[str, Any]],
    scenario: dict[str, Any],
) -> dict[str, Any]:
    previous = os.environ.get("STACK_CONFIG_PATH")
    tmp = Path(tempfile.gettempdir()) / f"no_lookahead_stack_{os.getpid()}_{scenario['name']}.yaml"
    tmp.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    os.environ["STACK_CONFIG_PATH"] = str(tmp)
    try:
        by_symbol: dict[str, Any] = {}
        all_trades: list[dict[str, Any]] = []
        for symbol in SYMBOLS:
            commission = commission_for_symbol(symbol, scenario)
            bt = Backtester(
                symbol,
                initial_balance=50000,
                strategy_config=copy.deepcopy(strategy_config),
                telemetry_enabled=False,
                screenshot_config={"enabled": False},
                run_id=f"no_lookahead_{candidate_id}_{scenario['name']}_{symbol}",
                routing_mode="optimizer_fast",
            )
            bt.slippage_ticks = int(scenario["slippage_ticks"])
            bt.commission_per_contract = commission
            result = run_next_bar_open(
                bt,
                data[symbol]["frame"],
                dataset_label=str(data[symbol]["path"]),
            )
            symbol_trades = []
            for trade in result.get("trades", []):
                normalized = normalize_portfolio_trade(trade, symbol, commission)
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
                "commission_per_side": commission,
            }

        portfolio_result = build_portfolio_result(all_trades)
        combine = {key: evaluate_combine_rules(portfolio_result, rules) for key, rules in combine_rules.items()}
        metrics = portfolio_metrics(portfolio_result, data)
        return {
            "candidate_id": candidate_id,
            "candidate_family": "artifact",
            "routing_mode": "optimizer_fast_no_lookahead_next_bar_open",
            "strategy_config": strategy_config,
            "scenario": scenario,
            "metrics": metrics,
            "combine": combine,
            "per_symbol": by_symbol,
            "portfolio_trades": portfolio_result["trades"],
        }
    finally:
        if previous is None:
            os.environ.pop("STACK_CONFIG_PATH", None)
        else:
            os.environ["STACK_CONFIG_PATH"] = previous
        with contextlib.suppress(OSError):
            tmp.unlink()


def evaluate_resting_stop_orders(
    *,
    candidate_id: str,
    strategy_config: dict[str, Any],
    config: dict[str, Any],
    data: dict[str, dict[str, Any]],
    combine_rules: dict[str, dict[str, Any]],
    scenario: dict[str, Any],
    max_pending_bars: int = 5,
) -> dict[str, Any]:
    previous = os.environ.get("STACK_CONFIG_PATH")
    tmp = Path(tempfile.gettempdir()) / f"resting_stop_stack_{os.getpid()}_{scenario['name']}.yaml"
    tmp.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    os.environ["STACK_CONFIG_PATH"] = str(tmp)
    try:
        by_symbol: dict[str, Any] = {}
        all_trades: list[dict[str, Any]] = []
        for symbol in SYMBOLS:
            commission = commission_for_symbol(symbol, scenario)
            bt = Backtester(
                symbol,
                initial_balance=50000,
                strategy_config=copy.deepcopy(strategy_config),
                telemetry_enabled=False,
                screenshot_config={"enabled": False},
                run_id=f"resting_stop_{candidate_id}_{scenario['name']}_{symbol}",
                routing_mode="optimizer_fast",
            )
            bt.slippage_ticks = int(scenario["slippage_ticks"])
            bt.commission_per_contract = commission
            result = run_resting_stop_orders(
                bt,
                data[symbol]["frame"],
                dataset_label=str(data[symbol]["path"]),
                max_pending_bars=max_pending_bars,
            )
            symbol_trades = []
            for trade in result.get("trades", []):
                normalized = normalize_portfolio_trade(trade, symbol, commission)
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
                "commission_per_side": commission,
            }

        portfolio_result = build_portfolio_result(all_trades)
        combine = {key: evaluate_combine_rules(portfolio_result, rules) for key, rules in combine_rules.items()}
        metrics = portfolio_metrics(portfolio_result, data)
        return {
            "candidate_id": candidate_id,
            "candidate_family": "artifact",
            "routing_mode": "optimizer_fast_no_lookahead_resting_stop",
            "strategy_config": strategy_config,
            "scenario": scenario,
            "metrics": metrics,
            "combine": combine,
            "per_symbol": by_symbol,
            "portfolio_trades": portfolio_result["trades"],
        }
    finally:
        if previous is None:
            os.environ.pop("STACK_CONFIG_PATH", None)
        else:
            os.environ["STACK_CONFIG_PATH"] = previous
        with contextlib.suppress(OSError):
            tmp.unlink()


def commission_for_symbol(symbol: str, scenario: dict[str, Any]) -> float:
    overrides = scenario.get("commission_per_side") or {}
    base = float(overrides.get(symbol, 0.62))
    return base * float(scenario.get("commission_multiplier", 1.0))


def run_next_bar_open(bt: Backtester, df: pd.DataFrame, *, dataset_label: str) -> dict[str, Any]:
    bt._reset_state()
    frame = bt._normalize_dataframe(df)
    indicators = bt.strategy._calculate_indicators(frame)
    bars = pd.DataFrame(indicators[["timestamp", "open", "high", "low", "close", "volume", "price"]]).copy()

    bt.equity_history = [bt.balance]
    bt.equity_curve = []
    pending_signal: dict[str, Any] | None = None

    for index in range(len(indicators)):
        row = indicators.iloc[index]
        current_time = pd.to_datetime(row.get("timestamp", pd.Timestamp.now("UTC")))
        current_time_str = current_time.isoformat()

        if pending_signal is not None:
            execute_pending_signal(bt, pending_signal, row, current_time)
            pending_signal = None

        bt._update_active_trades(row)
        bt._sync_risk_state()

        signal, _ = bt.strategy.engine.evaluate_index(bars, indicators, index)
        if signal:
            pending_signal = copy.deepcopy(signal)

        floating_pnl = sum(bt._calculate_pnl(trade, float(row["close"])) for trade in bt.active_trades)
        unrealized_commission = sum(trade["qty"] * bt.commission_per_contract for trade in bt.active_trades)
        equity_value = bt.balance + floating_pnl - unrealized_commission
        bt.equity_history.append(equity_value)
        bt.equity_curve.append({"timestamp": current_time_str, "equity": equity_value})
        if not bt.active_trades and pending_signal is None:
            risk_hit, _ = bt.risk_agent.is_risk_limit_hit()
            if risk_hit:
                break

    result = bt._get_results()
    return result


def run_resting_stop_orders(
    bt: Backtester,
    df: pd.DataFrame,
    *,
    dataset_label: str,
    max_pending_bars: int,
) -> dict[str, Any]:
    bt._reset_state()
    frame = bt._normalize_dataframe(df)
    indicators = bt.strategy._calculate_indicators(frame)
    bars = pd.DataFrame(indicators[["timestamp", "open", "high", "low", "close", "volume", "price"]]).copy()

    bt.equity_history = [bt.balance]
    bt.equity_curve = []
    pending_order: dict[str, Any] | None = None

    for index in range(len(indicators)):
        row = indicators.iloc[index]
        current_time = pd.to_datetime(row.get("timestamp", pd.Timestamp.now("UTC")))
        current_time_str = current_time.isoformat()

        bt._update_active_trades(row)
        bt._sync_risk_state()

        if pending_order is not None:
            if index - int(pending_order["created_index"]) > int(max_pending_bars):
                pending_order = None
            elif not bt.active_trades and resting_stop_triggered(pending_order, row):
                if execute_resting_stop_order(bt, pending_order, row, current_time):
                    resolve_entry_bar_bracket_conservatively(bt, row, current_time)
                pending_order = None

        signal, _ = bt.strategy.engine.evaluate_index(bars, indicators, index)
        if signal and not bt.active_trades:
            pending_order = build_resting_stop_order(signal, row, index, bt.tick_size)

        floating_pnl = sum(bt._calculate_pnl(trade, float(row["close"])) for trade in bt.active_trades)
        unrealized_commission = sum(trade["qty"] * bt.commission_per_contract for trade in bt.active_trades)
        equity_value = bt.balance + floating_pnl - unrealized_commission
        bt.equity_history.append(equity_value)
        bt.equity_curve.append({"timestamp": current_time_str, "equity": equity_value})
        if not bt.active_trades and pending_order is None:
            risk_hit, _ = bt.risk_agent.is_risk_limit_hit()
            if risk_hit:
                break

    return bt._get_results()


def build_resting_stop_order(signal: dict[str, Any], row: pd.Series, index: int, tick_size: float) -> dict[str, Any]:
    order = copy.deepcopy(signal)
    metadata = dict(order.get("metadata", {}) or {})
    side = str(order.get("side", "Buy"))
    fallback = float(row["high"]) + tick_size if side == "Buy" else float(row["low"]) - tick_size
    trigger = metadata.get("entry_trigger_price", metadata.get("entry_trigger", metadata.get("entry_reference", fallback)))
    try:
        trigger_price = float(trigger)
    except (TypeError, ValueError):
        trigger_price = fallback
    metadata["resting_order_created_index"] = index
    metadata["resting_stop_trigger"] = trigger_price
    metadata["execution_model"] = "resting_stop_after_completed_signal_bar"
    order["metadata"] = metadata
    order["trigger_price"] = trigger_price
    order["created_index"] = index
    return order


def resting_stop_triggered(order: dict[str, Any], row: pd.Series) -> bool:
    trigger = float(order["trigger_price"])
    if order["side"] == "Buy":
        return float(row["high"]) >= trigger
    return float(row["low"]) <= trigger


def execute_resting_stop_order(bt: Backtester, order: dict[str, Any], row: pd.Series, current_time: pd.Timestamp) -> bool:
    trigger = float(order["trigger_price"])
    if order["side"] == "Buy":
        base = max(trigger, float(row["open"])) if float(row["open"]) >= trigger else trigger
        entry_price = base + (bt.slippage_ticks * bt.tick_size)
    else:
        base = min(trigger, float(row["open"])) if float(row["open"]) <= trigger else trigger
        entry_price = base - (bt.slippage_ticks * bt.tick_size)
    order["entry_price"] = entry_price
    max_pos = bt.risk_agent.position_limits.get(bt.base_symbol, {}).get("max_positions", 1)
    route_context = {
        "now_dt": current_time.to_pydatetime() if hasattr(current_time, "to_pydatetime") else current_time,
        "market_open": True,
        "news_standdown": False,
        "near_close": False,
        "active_count": len(bt.active_trades),
        "max_positions": max_pos,
        "pending_order": bt.symbol in bt.risk_agent.pending_orders,
        "last_signal_time": bt.last_signal_time,
        "cooldown_seconds": bt.WIN_COOLDOWN_SECONDS if bt._last_trade_won is True else bt.SIGNAL_COOLDOWN_SECONDS,
        "current_price": entry_price,
    }
    routed = bt._route_signal(order, route_context)
    if getattr(routed, "accepted", False):
        order["quantity"] = getattr(routed, "quantity", order.get("quantity", 1))
        entry_commission = order["quantity"] * bt.commission_per_contract
        bt.balance -= entry_commission
        bt._sync_risk_state()
        bt._open_trade(order, current_time.isoformat(), entry_price)
        bt.last_signal_time = current_time.timestamp()
        return True
    return False


def resolve_entry_bar_bracket_conservatively(bt: Backtester, row: pd.Series, current_time: pd.Timestamp) -> None:
    if not bt.active_trades:
        return
    high = float(row["high"])
    low = float(row["low"])
    timestamp_str = current_time.isoformat()
    to_remove: list[dict[str, Any]] = []
    for trade in list(bt.active_trades):
        is_sl_hit = (trade["side"] == "Buy" and low <= trade["sl_price"]) or (
            trade["side"] == "Sell" and high >= trade["sl_price"]
        )
        is_tp_hit = (trade["side"] == "Buy" and high >= trade["tp_price"]) or (
            trade["side"] == "Sell" and low <= trade["tp_price"]
        )
        if not is_sl_hit and not is_tp_hit:
            continue
        if is_sl_hit:
            exit_price = (
                trade["sl_price"] - (bt.slippage_ticks * bt.tick_size)
                if trade["side"] == "Buy"
                else trade["sl_price"] + (bt.slippage_ticks * bt.tick_size)
            )
            result = "SL"
        else:
            exit_price = (
                trade["tp_price"] - (bt.slippage_ticks * bt.tick_size)
                if trade["side"] == "Buy"
                else trade["tp_price"] + (bt.slippage_ticks * bt.tick_size)
            )
            result = "TP"
        trade.setdefault("metadata", {})["entry_bar_resolution"] = "conservative_sl_first"
        pnl = bt._calculate_final_pnl(trade, exit_price)
        bt._record_trade(trade, exit_price, pnl, result, timestamp_str)
        bt.risk_agent.record_trade_result(pnl, symbol=bt.symbol)
        bt._last_trade_won = pnl > 0
        to_remove.append(trade)
    for trade in to_remove:
        if trade in bt.active_trades:
            bt.active_trades.remove(trade)
            bt.risk_agent.record_position_close(bt.symbol)
    if to_remove:
        bt._sync_risk_state()


def execute_pending_signal(bt: Backtester, signal: dict[str, Any], row: pd.Series, current_time: pd.Timestamp) -> None:
    current_time_str = current_time.isoformat()
    entry_price_base = float(row["open"])
    entry_price = (
        entry_price_base + (bt.slippage_ticks * bt.tick_size)
        if signal["side"] == "Buy"
        else entry_price_base - (bt.slippage_ticks * bt.tick_size)
    )
    signal["entry_price"] = entry_price
    max_pos = bt.risk_agent.position_limits.get(bt.base_symbol, {}).get("max_positions", 1)
    route_context = {
        "now_dt": current_time.to_pydatetime() if hasattr(current_time, "to_pydatetime") else current_time,
        "market_open": True,
        "news_standdown": False,
        "near_close": False,
        "active_count": len(bt.active_trades),
        "max_positions": max_pos,
        "pending_order": bt.symbol in bt.risk_agent.pending_orders,
        "last_signal_time": bt.last_signal_time,
        "cooldown_seconds": bt.WIN_COOLDOWN_SECONDS if bt._last_trade_won is True else bt.SIGNAL_COOLDOWN_SECONDS,
        "current_price": float(row.get("open", row.get("price", 0.0)) or 0.0),
    }
    routed = bt._route_signal(signal, route_context)
    if getattr(routed, "accepted", False):
        signal["quantity"] = getattr(routed, "quantity", signal.get("quantity", 1))
        entry_commission = signal["quantity"] * bt.commission_per_contract
        bt.balance -= entry_commission
        bt._sync_risk_state()
        bt._open_trade(signal, current_time_str, entry_price)
        bt.last_signal_time = current_time.timestamp()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(out_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace, candidate_id: str, *, elapsed: float) -> None:
    lines = [
        "# No-Lookahead Next-Bar-Open Validation",
        "",
        f"- Candidate: `{candidate_id}`",
        f"- Candidate config: `{args.candidate_config}`",
        f"- Runtime config: `{args.config}`",
        f"- Risk per trade: `{args.risk}`",
        f"- Execution model: `{args.execution_model}`",
        f"- Max pending bars: `{args.max_pending_bars}`",
        f"- Elapsed seconds: `{elapsed:.1f}`",
        "",
        "## Results",
        "",
        markdown_table(rows),
        "",
        "## Files",
        "",
        f"- Results: `{out_dir / 'no_lookahead_results.csv'}`",
        f"- First scenario trades: `{out_dir / 'no_lookahead_trades_first_scenario.csv'}`",
        f"- Details: `{out_dir / 'no_lookahead_details.json'}`",
    ]
    (out_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    cols = [
        "scenario",
        "total_pnl",
        "avg_daily_pnl_trade_dates",
        "total_trades",
        "trades_per_trade_day",
        "win_rate",
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
