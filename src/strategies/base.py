from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StrategySignal:
    strategy_name: str
    side: str
    quantity: int
    sl_ticks: int
    tp_ticks: int
    confidence: float = 0.0
    reason: str = ""
    trail_stop_atr_mult: float = 0.0
    trail_trigger_ticks: int = 999
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_trade_intent(self, symbol: str) -> dict[str, Any]:
        return {
            "side": self.side,
            "quantity": self.quantity,
            "symbol": symbol,
            "sl_ticks": self.sl_ticks,
            "tp_ticks": self.tp_ticks,
            "trail_stop_atr_mult": self.trail_stop_atr_mult,
            "trail_trigger_ticks": self.trail_trigger_ticks,
            "strategy_name": self.strategy_name,
            "confidence": self.confidence,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


class BaseStrategy(ABC):
    name = "base"
    warmup_bars = 0

    def __init__(self, symbol: str, params: dict[str, Any] | None = None):
        self.symbol = symbol
        self.params = params or {}
        self.priority = int(self.params.get("priority", 0))
        self.warmup_bars = int(getattr(self, "warmup_bars", 0))
        warmup_val = self.params.get("warmup_bars")
        if warmup_val is not None:
            try:
                self.warmup_bars = max(0, int(warmup_val))
            except (ValueError, TypeError):
                pass
        self.enabled = bool(self.params.get("enabled", True))
        enabled_syms = self.params.get("enabled_symbols")
        if self.enabled and enabled_syms is not None and isinstance(enabled_syms, list):
            base_sym = self._base_symbol(self.symbol)
            if self.symbol not in enabled_syms and base_sym not in enabled_syms and self.symbol.upper() not in enabled_syms and base_sym.upper() not in enabled_syms:
                self.enabled = False
        self.logger = logging.getLogger(f"Strategy-{self.name}-{symbol}")
        self.is_priming = False

    @staticmethod
    def _base_symbol(symbol: str) -> str:
        symbol_upper = str(symbol).upper()
        known = ("MNQ", "MES", "MCL", "MGC", "MYM", "NQ", "ES", "CL", "GC", "YM", "M6E", "6E", "MNK", "NKD")
        for prefix in known:
            if symbol_upper.startswith(prefix):
                return prefix
        s = "".join([c for c in symbol_upper if c.isalpha()])
        futures_months = ("F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z")
        if len(s) > 2 and s[-1] in futures_months:
            return s[:-1]
        return s

    def reset(self) -> None:
        pass

    @abstractmethod
    def get_description(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def on_bar(self, index, bars, indicators) -> StrategySignal | None:
        raise NotImplementedError

    def get_snapshot(self, index, bars, indicators) -> dict[str, Any]:
        return {"status": "active", "strategy": self.name}

    def notify_stop_exit(self, fill_time) -> None:
        """Hook for the supervisor to notify a strategy that its position
        was closed by a broker-side stop (not an opposite-side intent we
        dispatched). Default is a no-op; strategies that need to gate
        re-entry after a stop-out (e.g. always-in bias systems) override
        this. `fill_time` is a datetime in UTC."""
        return

    def current_bias(self, index, bars, indicators) -> str:
        """Return 'long', 'short', or 'neutral' describing the direction this
        strategy's CURRENT STATE supports — even when no fresh entry trigger
        has fired this bar. Used by the always-in vote meta-strategy to
        aggregate continuous directional bias across the strategy library.

        Default is 'neutral' so strategies that have no natural continuous-bias
        notion don't accidentally vote. Each concrete strategy that DOES have a
        natural bias should override this with a deterministic function of the
        indicator state at `index`."""
        return "neutral"
