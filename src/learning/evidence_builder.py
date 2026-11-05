from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.learning.session_capture import (
    _normalize_bars,
    capture_trade_context_artifacts,
)


def build_evidence_pack(
    *,
    role: str,
    symbol: str | None = None,
    batch_id: str | None = None,
    epoch: int | None = None,
    optimizer_result: dict[str, Any] | None = None,
    trades: Sequence[dict[str, Any]] | None = None,
    market_bars: pd.DataFrame | None = None,
    strategy_config: dict[str, Any] | None = None,
    routing_config: dict[str, Any] | None = None,
    evidence_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = evidence_config or {}
    max_samples = int(cfg.get("max_trade_samples", 12) or 12)
    timeframes = [str(item) for item in (cfg.get("timeframes") or ["1m", "5m", "15m"])]
    max_bars = int(cfg.get("max_bars_per_sample", cfg.get("max_chart_bars", 120)) or 120)
    artifact_dir = Path(str(cfg.get("artifact_dir") or "reports/llm_analysis/evidence"))

    optimizer_result = optimizer_result or {}
    source_trades = list(trades if trades is not None else _best_trial_trades(optimizer_result))
    samples = select_trade_samples(source_trades, max_samples=max_samples)
    if market_bars is not None and not market_bars.empty:
        _attach_generated_artifacts(
            samples=samples,
            symbol=symbol or _infer_symbol(samples),
            market_bars=market_bars,
            artifact_dir=artifact_dir,
            timeframes=timeframes,
            max_bars=max_bars,
        )

    return {
        "schema": "structured_llm_evidence.v1",
        "role": role,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run": {
            "symbol": symbol,
            "batch_id": batch_id,
            "epoch": epoch,
        },
        "run_metrics": _run_metrics(optimizer_result),
        "top_trials": _top_trials(optimizer_result),
        "best_config_summary": _best_config_summary(optimizer_result),
        "strategy_breakdown": _strategy_breakdown(optimizer_result),
        "strategy_config": strategy_config or {},
        "routing_config": routing_config or {},
        "trade_samples": [_compact_trade_sample(trade, timeframes=timeframes, max_bars=max_bars) for trade in samples],
        "selection_policy": {
            "max_trade_samples": max_samples,
            "order": "worst_losses_then_best_wins_then_near_breakeven",
            "timeframes": timeframes,
            "max_bars_per_sample": max_bars,
        },
        "live_trading_invariant": (
            "Evidence is advisory/offline only. It must not block, veto, resize, override, "
            "or alter live trades outside validated strategy, risk, routing, broker, and account-limit parameters."
        ),
    }


def select_trade_samples(trades: Sequence[dict[str, Any]], *, max_samples: int = 12) -> list[dict[str, Any]]:
    indexed = [(idx, trade) for idx, trade in enumerate(trades or [])]

    def pnl(item: tuple[int, dict[str, Any]]) -> float:
        return _trade_pnl(item[1])

    losses = sorted([item for item in indexed if pnl(item) < 0], key=pnl)
    wins = sorted([item for item in indexed if pnl(item) > 0], key=pnl, reverse=True)
    near = sorted(indexed, key=lambda item: (abs(pnl(item)), item[0]))
    selected: list[dict[str, Any]] = []
    seen = set()
    for group in (losses, wins, near):
        for idx, trade in group:
            key = _trade_key(trade, idx)
            if key in seen:
                continue
            selected.append(dict(trade))
            seen.add(key)
            if len(selected) >= max_samples:
                return selected
    return selected


def prompt_with_evidence(
    *,
    heading: str,
    evidence_pack: dict[str, Any],
    instructions: Sequence[str],
    prior_analysis: Sequence[dict[str, Any]] | None = None,
    strategy_code: str = "",
    max_prompt_chars: int = 12000,
) -> str:
    pack = _fit_evidence_pack(evidence_pack, max_prompt_chars=max_prompt_chars, reserve_chars=2200 + len(strategy_code))
    evidence_json = json.dumps(pack, indent=2, default=str)
    prior = _compact_prior(prior_analysis or [])
    prompt = f"""{heading}

Instructions:
{chr(10).join(f"- {item}" for item in instructions)}

Structured evidence JSON:
```json
{evidence_json}
```

Prior analysis:
{prior}

Current strategy code excerpt:
{strategy_code}
"""
    if len(prompt) <= max_prompt_chars:
        return prompt
    overflow = len(prompt) - max_prompt_chars
    if strategy_code and overflow > 0:
        strategy_code = strategy_code[: max(0, len(strategy_code) - overflow - 200)] + "\n...truncated..."
        prompt = f"""{heading}

Instructions:
{chr(10).join(f"- {item}" for item in instructions)}

Structured evidence JSON:
```json
{evidence_json}
```

Prior analysis:
{prior}

Current strategy code excerpt:
{strategy_code}
"""
    return prompt[:max_prompt_chars]


def _best_trial_trades(optimizer_result: dict[str, Any]) -> list[dict[str, Any]]:
    best = ((optimizer_result.get("status") or {}).get("best_trial") or {})
    return list((best.get("result") or {}).get("trades") or [])


def _run_metrics(optimizer_result: dict[str, Any]) -> dict[str, Any]:
    status = optimizer_result.get("status") or {}
    best = status.get("best_trial") or {}
    return {
        "epoch": status.get("epoch"),
        "trials_attempted": status.get("trials_attempted"),
        "trials_recorded": status.get("trials_recorded"),
        "best_score": status.get("best_score"),
        "consecutive_passes": status.get("consecutive_passes"),
        "permutation_significant": status.get("permutation_significant"),
        "best_summary": (best.get("summary") or {}),
        "best_combine_evaluation": (best.get("combine_evaluation") or {}),
    }


def _top_trials(optimizer_result: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    out = []
    trials = sorted(
        optimizer_result.get("trials") or [],
        key=lambda item: float(item.get("score", float("-inf"))),
        reverse=True,
    )
    for trial in trials[:limit]:
        out.append(
            {
                "trial": trial.get("trial"),
                "score": trial.get("score"),
                "passed_combine": trial.get("passed_combine"),
                "summary": trial.get("summary") or {},
                "config": _config_summary(trial.get("config") or {}),
            }
        )
    return out


def _best_config_summary(optimizer_result: dict[str, Any]) -> dict[str, Any]:
    best = ((optimizer_result.get("status") or {}).get("best_trial") or {})
    return _config_summary(best.get("config") or {})


def _strategy_breakdown(optimizer_result: dict[str, Any]) -> dict[str, Any]:
    best = ((optimizer_result.get("status") or {}).get("best_trial") or {})
    result = best.get("result") or {}
    return (best.get("summary") or {}).get("by_strategy") or result.get("by_strategy") or {}


def _config_summary(config: dict[str, Any]) -> dict[str, Any]:
    strategies = []
    for item in config.get("strategies") or []:
        strategies.append(
            {
                "name": item.get("name"),
                "enabled": item.get("enabled"),
                "priority": item.get("priority"),
                "params": item.get("params") or {},
            }
        )
    return {"direction_filters": config.get("direction_filters") or {}, "strategies": strategies}


def _compact_trade_sample(trade: dict[str, Any], *, timeframes: Sequence[str], max_bars: int) -> dict[str, Any]:
    outcome = dict(trade.get("outcome") or {})
    if "pnl" not in outcome and "pnl" in trade:
        outcome["pnl"] = trade.get("pnl")
    if "close_type" not in outcome and "result" in trade:
        outcome["close_type"] = trade.get("result")
    sample = {
        "trade_key": trade.get("trade_key") or trade.get("order_id") or trade.get("time_open"),
        "symbol": trade.get("symbol"),
        "strategy_name": trade.get("strategy_name"),
        "side": trade.get("side"),
        "quantity": trade.get("quantity", trade.get("qty")),
        "entry_price": trade.get("entry_price", trade.get("entry")),
        "exit_price": trade.get("exit"),
        "opened_at": trade.get("opened_at", trade.get("time_open")),
        "closed_at": outcome.get("closed_at", trade.get("time_close")),
        "outcome": outcome,
        "selected_signal": _limit_json(trade.get("selected_signal") or trade.get("metadata") or {}, 1800),
        "candidates": _limit_list(trade.get("candidates") or [], 8),
        "route_context": _limit_json(trade.get("route_context") or {}, 1200),
        "discovery_context": _limit_json(trade.get("discovery_context") or {}, 1200),
        "account_snapshot": _limit_json(trade.get("account_snapshot") or {}, 800),
        "chart_artifacts": _compact_chart_artifacts(trade.get("chart_artifacts") or {}, timeframes=timeframes, max_bars=max_bars),
    }
    if trade.get("entry_screenshot"):
        sample["chart_artifacts"].setdefault("screenshots", []).append(str(trade.get("entry_screenshot")))
    return sample


def _compact_chart_artifacts(chart_artifacts: dict[str, Any], *, timeframes: Sequence[str], max_bars: int) -> dict[str, Any]:
    out: dict[str, Any] = {"timeframes": {}, "screenshots": []}
    for timeframe, artifact in (chart_artifacts.get("timeframes") or {}).items():
        if str(timeframe) not in timeframes:
            continue
        compact = {key: artifact.get(key) for key in ("status", "bars", "svg_path", "data_path", "error") if key in artifact}
        data_path = artifact.get("data_path")
        if data_path:
            compact["candles"] = _read_candles(data_path, max_bars=max_bars)
        out["timeframes"][str(timeframe)] = compact
    return out


def _attach_generated_artifacts(
    *,
    samples: list[dict[str, Any]],
    symbol: str,
    market_bars: pd.DataFrame,
    artifact_dir: Path,
    timeframes: Sequence[str],
    max_bars: int,
) -> None:
    base = _normalize_bars(market_bars)
    for idx, sample in enumerate(samples):
        opened_at = pd.to_datetime(sample.get("opened_at") or sample.get("time_open"), errors="coerce")
        if pd.isna(opened_at):
            opened_at = base["timestamp"].iloc[-1] if not base.empty else pd.Timestamp.now(tz="UTC")
        window = base[base["timestamp"] <= opened_at].tail(max_bars) if not base.empty else base
        trade_key = str(sample.get("trade_key") or sample.get("order_id") or f"sample-{idx}")
        sample["chart_artifacts"] = capture_trade_context_artifacts(
            bars=window,
            trade_key=_safe_key(trade_key),
            symbol=symbol,
            opened_at=opened_at.to_pydatetime() if hasattr(opened_at, "to_pydatetime") else datetime.now(timezone.utc),
            config={
                "enabled": True,
                "artifact_dir": str(artifact_dir),
                "timeframes": list(timeframes),
                "min_chart_bars": 1,
                "max_chart_bars": max_bars,
            },
        )


def _read_candles(data_path: str, *, max_bars: int) -> list[dict[str, Any]]:
    path = Path(data_path)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    bars = payload.get("bars") or []
    return bars[-max_bars:]


def _fit_evidence_pack(evidence_pack: dict[str, Any], *, max_prompt_chars: int, reserve_chars: int) -> dict[str, Any]:
    pack = json.loads(json.dumps(evidence_pack, default=str))
    budget = max(2000, max_prompt_chars - reserve_chars)
    while len(json.dumps(pack, default=str)) > budget:
        changed = False
        for sample in pack.get("trade_samples", []):
            for artifact in (sample.get("chart_artifacts", {}).get("timeframes") or {}).values():
                candles = artifact.get("candles")
                if isinstance(candles, list) and len(candles) > 20:
                    artifact["candles"] = candles[-max(20, len(candles) // 2) :]
                    changed = True
        if not changed:
            if len(pack.get("trade_samples", [])) > 3:
                pack["trade_samples"] = pack["trade_samples"][:-1]
                changed = True
            elif len(pack.get("top_trials", [])) > 3:
                pack["top_trials"] = pack["top_trials"][:3]
                changed = True
        if not changed:
            break
    return pack


def _compact_prior(prior_analysis: Sequence[dict[str, Any]]) -> str:
    if not prior_analysis:
        return "No prior analysis history."
    lines = []
    for item in prior_analysis[:5]:
        reasoning = str(item.get("reasoning", "")).replace("\n", " ")[:300]
        lines.append(
            f"Epoch {item.get('epoch')}: {reasoning} | config_changes={len(item.get('config_changes', {}))} "
            f"| code_patches={len(item.get('code_patches', []))}"
        )
    return "\n".join(lines)


def _trade_pnl(trade: dict[str, Any]) -> float:
    outcome = trade.get("outcome") or {}
    try:
        return float(outcome.get("pnl", trade.get("pnl", 0.0)) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _trade_key(trade: dict[str, Any], fallback: int) -> str:
    return str(trade.get("trade_key") or trade.get("order_id") or trade.get("time_open") or fallback)


def _infer_symbol(samples: Sequence[dict[str, Any]]) -> str:
    for sample in samples:
        if sample.get("symbol"):
            return str(sample["symbol"])
    return "UNKNOWN"


def _safe_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "trade"


def _limit_json(value: Any, max_chars: int) -> Any:
    text = json.dumps(value, default=str)
    if len(text) <= max_chars:
        return value
    return {"truncated": True, "preview": text[:max_chars]}


def _limit_list(value: Sequence[Any], limit: int) -> list[Any]:
    return list(value)[:limit]
