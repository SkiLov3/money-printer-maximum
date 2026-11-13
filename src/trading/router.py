from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.trading.filters import FilterManager, TradingFilterContext


@dataclass
class RouterRejection:
    reason: str
    stage: str
    filter_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RouterResult:
    accepted: bool
    quantity: int = 0
    result: dict[str, Any] | None = None
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class TradeRouter:
    def __init__(self, risk_agent, execution_agent, telemetry=None, filter_manager: FilterManager | None = None):
        self.risk_agent = risk_agent
        self.execution_agent = execution_agent
        self.telemetry = telemetry
        self.filter_manager = filter_manager or FilterManager()

    def route_trade(self, trade_intent: dict[str, Any], route_context: dict[str, Any]) -> RouterResult | RouterRejection:
        now_dt = route_context.get("now_dt") or datetime.now(timezone.utc)
        filter_context = TradingFilterContext(
            symbol=trade_intent["symbol"],
            now_dt=now_dt,
            trade_intent=trade_intent,
            market_open=bool(route_context.get("market_open", True)),
            news_standdown=bool(route_context.get("news_standdown", False)),
            near_close=bool(route_context.get("near_close", False)),
            active_count=int(route_context.get("active_count", 0)),
            max_positions=int(route_context.get("max_positions", 1)),
            pending_order=bool(route_context.get("pending_order", False)),
            last_signal_time=float(route_context.get("last_signal_time", 0.0)),
            cooldown_seconds=int(route_context.get("cooldown_seconds", 0)),
            existing_side=str(route_context.get("existing_side", "") or ""),
        )
        allowed, decisions = self.filter_manager.evaluate_entry(filter_context)
        if not allowed:
            blocker = decisions[-1]
            return RouterRejection(blocker.reason, "filter", blocker.name, {"decisions": [d.__dict__ for d in decisions]})

        metadata = trade_intent.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}

        effective_sl_ticks = trade_intent.get("sl_ticks")
        if (
            metadata.get("stop_price") is not None
            and route_context.get("current_price") is not None
        ):
            meta_fn = getattr(getattr(self.execution_agent, "client", None), "get_contract_meta", None)
            c_meta = meta_fn(trade_intent["symbol"]) if callable(meta_fn) else None
            tick_sz = float(c_meta.get("tickSize") or 0.0) if isinstance(c_meta, dict) else 0.0
            if tick_sz > 0:
                try:
                    ep = float(trade_intent.get("entry_price") or route_context["current_price"])
                    sp = float(metadata["stop_price"])
                    side_norm = str(trade_intent.get("side") or "").strip().lower()
                    if side_norm in ("buy", "long") and sp < ep:
                        effective_sl_ticks = max(1, int(-(-(ep - sp) // tick_sz)))
                    elif side_norm in ("sell", "short") and sp > ep:
                        effective_sl_ticks = max(1, int(-(-(sp - ep) // tick_sz)))
                except (ValueError, TypeError):
                    effective_sl_ticks = trade_intent.get("sl_ticks")

        quantity = self.risk_agent.get_entry_quantity(
            trade_intent["symbol"],
            int(trade_intent["quantity"]),
            sl_ticks=effective_sl_ticks,
            current_price=route_context.get("current_price"),
            current_atr=metadata.get("atr"),
        )
        if quantity <= 0:
            return RouterRejection("quantity sizing returned 0", "size")

        if not self.risk_agent.validate_trade(trade_intent["symbol"], quantity, trade_intent["side"]):
            return RouterRejection("risk validation rejected trade", "risk")

        routed_intent = dict(trade_intent)
        routed_intent["quantity"] = quantity
        if routed_intent.get("entry_price") is None and route_context.get("current_price"):
            routed_intent["entry_price"] = route_context.get("current_price")
        self.risk_agent.set_pending(trade_intent["symbol"], True)
        try:
            result = self.execution_agent.execute_trade(routed_intent)
        except Exception as exc:
            self.risk_agent.set_pending(trade_intent["symbol"], False)
            return RouterRejection(f"execution exception: {exc}", "execution")

        if not result or result.get("status") in ("rejected", "error"):
            self.risk_agent.set_pending(trade_intent["symbol"], False)
            return RouterRejection(result.get("message", "execution rejected") if isinstance(result, dict) else "execution rejected", "execution")

        if self.telemetry:
            self.telemetry.record_signal(
                trade_intent["symbol"],
                trade_intent.get("strategy_name", "unknown"),
                "router_accepted",
                trade_intent.get("side"),
                now_dt,
                route_context.get("current_price"),
                metadata={"quantity": quantity, "source": trade_intent.get("source", "strategy")},
            )
        return RouterResult(True, quantity, result=result, reason="accepted")
