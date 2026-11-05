from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class TradingFilterContext:
    symbol: str
    now_dt: datetime
    trade_intent: dict[str, Any]
    market_open: bool
    news_standdown: bool
    near_close: bool
    active_count: int
    max_positions: int
    pending_order: bool
    last_signal_time: float
    cooldown_seconds: int
    existing_side: str = ""  # current open position side ("Buy"/"Sell") or "" when flat


@dataclass
class FilterDecision:
    name: str
    allowed: bool
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class _BaseFilter:
    name = "base"

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def evaluate(self, context: TradingFilterContext) -> FilterDecision:
        raise NotImplementedError


class MarketHoursFilter(_BaseFilter):
    name = "market_hours"

    def evaluate(self, context: TradingFilterContext) -> FilterDecision:
        allowed = context.market_open
        reason = "" if allowed else "outside market hours"
        return FilterDecision(self.name, allowed, reason)


class NewsStanddownFilter(_BaseFilter):
    name = "news_standdown"

    def evaluate(self, context: TradingFilterContext) -> FilterDecision:
        allowed = not context.news_standdown
        reason = "" if allowed else "news stand-down active"
        return FilterDecision(self.name, allowed, reason)


class NearCloseFilter(_BaseFilter):
    name = "near_close"

    def evaluate(self, context: TradingFilterContext) -> FilterDecision:
        allowed = not context.near_close
        reason = "" if allowed else "market close approaching"
        return FilterDecision(self.name, allowed, reason)


class CooldownFilter(_BaseFilter):
    name = "cooldown"

    def evaluate(self, context: TradingFilterContext) -> FilterDecision:
        elapsed = context.now_dt.timestamp() - float(context.last_signal_time or 0.0)
        allowed = elapsed >= context.cooldown_seconds
        reason = "" if allowed else f"cooldown active ({int(context.cooldown_seconds - elapsed)}s left)"
        return FilterDecision(self.name, allowed, reason, {"elapsed": elapsed})


class MaxPositionsFilter(_BaseFilter):
    name = "max_positions"

    def __init__(self, enabled: bool = True, allow_flips: bool = False):
        super().__init__(enabled)
        self.allow_flips = allow_flips

    def evaluate(self, context: TradingFilterContext) -> FilterDecision:
        # Flip exemption: an opposite-side intent against an existing position
        # is a net-position flip (one broker fill closes the existing leg and
        # opens the new one). It is not a second concurrent position.
        intent_side = str(context.trade_intent.get("side", "")).lower()
        existing_side = str(context.existing_side or "").lower()
        is_flip = bool(intent_side and existing_side and intent_side != existing_side)
        if is_flip and self.allow_flips:
            return FilterDecision(self.name, True, "", {"flip": True})
        allowed = context.active_count < context.max_positions
        reason = "" if allowed else f"already at max positions ({context.active_count}/{context.max_positions})"
        return FilterDecision(self.name, allowed, reason)


class PendingOrderFilter(_BaseFilter):
    name = "pending_order"

    def evaluate(self, context: TradingFilterContext) -> FilterDecision:
        allowed = not context.pending_order
        reason = "" if allowed else "pending order already exists"
        return FilterDecision(self.name, allowed, reason)


class FilterManager:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        enabled = self.config.get("enabled", True)
        filter_cfg = self.config.get("entry_filters", {})
        self.filters: list[_BaseFilter] = []
        if enabled:
            self.filters = [
                MarketHoursFilter(filter_cfg.get("market_hours", True)),
                NewsStanddownFilter(filter_cfg.get("news_standdown", True)),
                NearCloseFilter(filter_cfg.get("near_close", True)),
                CooldownFilter(filter_cfg.get("cooldown", True)),
                MaxPositionsFilter(
                    filter_cfg.get("max_positions", True),
                    allow_flips=filter_cfg.get("allow_flips", False)
                ),
                PendingOrderFilter(filter_cfg.get("pending_order", True)),
            ]

    def evaluate_entry(self, context: TradingFilterContext) -> tuple[bool, list[FilterDecision]]:
        decisions: list[FilterDecision] = []
        for trading_filter in self.filters:
            if not trading_filter.enabled:
                continue
            decision = trading_filter.evaluate(context)
            decisions.append(decision)
            if not decision.allowed:
                return False, decisions
        return True, decisions
