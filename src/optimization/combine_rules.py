from __future__ import annotations

from typing import Any

import pandas as pd


def _trade_timestamp(trade: dict[str, Any]) -> pd.Timestamp:
    raw = trade.get("time_close") or trade.get("exit_time") or trade.get("timestamp")
    ts = pd.to_datetime(raw, utc=True, errors="coerce")
    return ts


def _topstep_trading_day(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return "unknown"
    local = ts.tz_convert("America/Chicago")
    # Topstep trading day begins at 5:00 PM CT and is attributed to the next
    # calendar date. This is reporting-only combine evaluation, not a live gate.
    if (local.hour, local.minute) >= (17, 0):
        local = local + pd.Timedelta(days=1)
    return local.date().isoformat()


def _daily_pnl_from_trades(trades: list[dict[str, Any]]) -> dict[str, float]:
    daily_pnl: dict[str, float] = {}
    for trade in trades:
        key = _topstep_trading_day(_trade_timestamp(trade))
        daily_pnl[key] = daily_pnl.get(key, 0.0) + float(trade.get("pnl", 0.0))
    return daily_pnl


def _is_micro_symbol(symbol: Any) -> bool:
    symbol_upper = str(symbol or "").upper()
    return symbol_upper.startswith(("MNQ", "MES", "MYM", "MCL", "MGC", "M2K"))


def _position_size_check(trades: list[dict[str, Any]], combine_rules: dict[str, Any]) -> dict[str, Any]:
    max_contracts = combine_rules.get("max_contracts")
    max_micros = combine_rules.get("max_micros")
    if max_contracts is None and max_micros is None:
        return {"ok": True, "evaluated": False}

    events = []
    for sequence, trade in enumerate(trades):
        qty = abs(int(float(trade.get("qty", 0) or 0)))
        if qty <= 0:
            continue
        symbol = trade.get("symbol") or trade.get("contract") or trade.get("instrument") or ""
        is_micro = _is_micro_symbol(symbol)
        open_ts = pd.to_datetime(trade.get("time_open") or trade.get("entry_time") or trade.get("timestamp"), utc=True, errors="coerce")
        close_ts = pd.to_datetime(trade.get("time_close") or trade.get("exit_time"), utc=True, errors="coerce")
        if pd.isna(open_ts):
            continue
        events.append((open_ts, 0, sequence, qty, is_micro, "open", symbol))
        if not pd.isna(close_ts):
            events.append((close_ts, 1, sequence, qty, is_micro, "close", symbol))

    open_contracts = 0
    open_micros = 0
    max_seen_contracts = 0
    max_seen_micros = 0
    breach = None
    for ts, _, _, qty, is_micro, action, symbol in sorted(events):
        if action == "open":
            if is_micro:
                open_micros += qty
            else:
                open_contracts += qty
            max_seen_contracts = max(max_seen_contracts, open_contracts)
            max_seen_micros = max(max_seen_micros, open_micros)
            contracts_ok = max_contracts is None or open_contracts <= int(max_contracts)
            micros_ok = max_micros is None or open_micros <= int(max_micros)
            if breach is None and not (contracts_ok and micros_ok):
                breach = {
                    "timestamp": ts.isoformat(),
                    "symbol": symbol,
                    "open_contracts": open_contracts,
                    "open_micros": open_micros,
                    "max_contracts": max_contracts,
                    "max_micros": max_micros,
                }
        elif is_micro:
            open_micros = max(0, open_micros - qty)
        else:
            open_contracts = max(0, open_contracts - qty)

    return {
        "ok": breach is None,
        "evaluated": True,
        "max_contracts": max_contracts,
        "max_micros": max_micros,
        "max_open_contracts_seen": max_seen_contracts,
        "max_open_micros_seen": max_seen_micros,
        "breach": breach,
    }


def _equity_points(results: dict[str, Any], starting_balance: float) -> list[dict[str, Any]]:
    curve = results.get("equity_curve") or []
    points: list[dict[str, Any]] = []
    for point in curve:
        if not isinstance(point, dict):
            continue
        ts = pd.to_datetime(point.get("timestamp"), utc=True, errors="coerce")
        equity = point.get("equity")
        if pd.isna(ts) or equity is None:
            continue
        points.append({"timestamp": ts, "equity": float(equity)})
    if points:
        return points

    equity = results.get("equity_history", [])
    return [{"timestamp": pd.NaT, "equity": float(value)} for value in equity] or [
        {"timestamp": pd.NaT, "equity": starting_balance + float(results["summary"].get("total_pnl", 0))}
    ]


def _max_loss_check(results: dict[str, Any], combine_rules: dict[str, Any], starting_balance: float) -> dict[str, Any]:
    max_loss_limit = float(combine_rules.get("max_loss_limit", 2000))
    balance_floor = starting_balance - max_loss_limit
    trailing_max_loss = bool(combine_rules.get("trailing_max_loss", False))
    points = _equity_points(results, starting_balance)

    if not trailing_max_loss:
        min_balance = min(point["equity"] for point in points)
        return {
            "ok": min_balance > balance_floor,
            "mode": "static_floor",
            "starting_balance": starting_balance,
            "balance_floor": balance_floor,
            "min_balance_seen": min_balance,
        }

    mll = balance_floor
    day_key = None
    day_end_equity = starting_balance
    min_balance = starting_balance
    min_mll_gap = float("inf")
    breach = None
    used_timestamps = all(not pd.isna(point["timestamp"]) for point in points)

    for point in points:
        ts = point["timestamp"]
        equity = float(point["equity"])
        if not pd.isna(ts):
            point_day = _topstep_trading_day(ts)
            if day_key is None:
                day_key = point_day
            elif point_day != day_key:
                mll = min(starting_balance, max(mll, day_end_equity - max_loss_limit))
                day_key = point_day
        min_balance = min(min_balance, equity)
        min_mll_gap = min(min_mll_gap, equity - mll)
        if breach is None and equity <= mll:
            breach = {
                "timestamp": None if pd.isna(ts) else ts.isoformat(),
                "equity": round(equity, 2),
                "mll": round(mll, 2),
            }
        day_end_equity = equity

    return {
        "ok": breach is None,
        "mode": "topstep_trailing_eod" if used_timestamps else "topstep_trailing_eod_without_timestamps",
        "starting_balance": starting_balance,
        "initial_floor": balance_floor,
        "final_mll": round(mll, 2),
        "min_balance_seen": round(min_balance, 2),
        "min_mll_gap": round(min_mll_gap, 2),
        "breach": breach,
    }


def evaluate_combine_rules(results: dict[str, Any], combine_rules: dict[str, Any]) -> dict[str, Any]:
    starting_balance = float(combine_rules.get("starting_balance", 50000))
    profit_target = float(combine_rules.get("profit_target", 3000))
    consistency_max_best_day_pct = float(combine_rules.get("consistency_max_best_day_pct", 50))
    evaluate_daily_loss = bool(combine_rules.get("evaluate_daily_loss", False))
    daily_loss_limit = float(combine_rules.get("daily_loss_limit", 1000))

    summary = results["summary"]
    trades = results.get("trades", [])
    max_loss = _max_loss_check(results, combine_rules, starting_balance)
    rule_max_loss_ok = bool(max_loss["ok"])
    objective_profit_ok = float(summary.get("total_pnl", 0)) >= profit_target

    daily_pnl = _daily_pnl_from_trades(trades)
    position_size = _position_size_check(trades, combine_rules)
    best_day_profit = max([v for v in daily_pnl.values() if v > 0], default=0.0)
    total_profit = max(float(summary.get("total_pnl", 0)), 0.0)
    best_day_pct = ((best_day_profit / total_profit) * 100.0) if total_profit > 0 else 0.0
    objective_consistency_ok = total_profit > 0 and best_day_pct <= consistency_max_best_day_pct

    max_daily_loss = abs(min(daily_pnl.values())) if daily_pnl else 0.0
    objective_daily_ok = (max_daily_loss < daily_loss_limit) if evaluate_daily_loss else True

    passed = (
        rule_max_loss_ok
        and objective_profit_ok
        and objective_consistency_ok
        and objective_daily_ok
        and bool(position_size["ok"])
    )
    return {
        "passed": passed,
        "checks": {
            "profit_target": {
                "ok": objective_profit_ok,
                "target": profit_target,
                "actual": float(summary.get("total_pnl", 0)),
            },
            "max_loss_limit": max_loss,
            "consistency": {
                "ok": objective_consistency_ok,
                "max_best_day_pct": consistency_max_best_day_pct,
                "actual_best_day_pct": round(best_day_pct, 2),
                "best_day_profit": round(best_day_profit, 2),
                "total_profit": round(total_profit, 2),
            },
            "daily_loss_limit": {
                "evaluated": evaluate_daily_loss,
                "ok": objective_daily_ok,
                "limit": daily_loss_limit,
                "worst_day_loss": round(max_daily_loss, 2),
            },
            "position_size": position_size,
        },
        "daily_pnl": {k: round(v, 2) for k, v in sorted(daily_pnl.items())},
    }
