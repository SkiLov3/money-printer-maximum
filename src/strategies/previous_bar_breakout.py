from __future__ import annotations

from typing import Any

import pandas as pd

from .base import BaseStrategy, StrategySignal


class PreviousBarBreakoutStrategy(BaseStrategy):
    name = "previous_bar_breakout"
    warmup_bars = 30

    def __init__(self, symbol: str, params: dict[str, Any] | None = None):
        super().__init__(symbol, params)
        self.params = self._apply_symbol_overrides(self.params)
        self.tick_size = float(self.params.get("tick_size", 0.25))
        self.quantity = int(self.params.get("quantity", 1))
        self.lookback = int(self.params.get("lookback", 12))
        self.direction_mode = str(self.params.get("direction_mode", "continuation")).strip().lower()
        if self.direction_mode not in {"continuation", "fade"}:
            self.direction_mode = "continuation"
        self.breakout_buffer_ticks = int(self.params.get("breakout_buffer_ticks", 1))
        self.min_range_ticks = float(self.params.get("min_range_ticks", 0.0))
        self.min_body_ratio = float(self.params.get("min_body_ratio", 0.45))
        self.min_adx = float(self.params.get("min_adx", 0.0))
        self.use_ema_filter = bool(self.params.get("use_ema_filter", True))
        self.ema_column = str(self.params.get("ema_column", "ema_21"))
        self.sl_ticks = int(self.params.get("sl_ticks", 40))
        self.tp_ticks = int(self.params.get("tp_ticks", 80))
        self.min_signal_gap_bars = int(self.params.get("min_signal_gap_bars", 0))
        self.allowed_entry_hours_et = [int(hour) for hour in self.params.get("allowed_entry_hours_et", []) or []]
        self.reset()

    def _apply_symbol_overrides(self, params: dict[str, Any]) -> dict[str, Any]:
        merged = dict(params or {})
        overrides = merged.get("symbol_params", {}) or {}
        if not isinstance(overrides, dict):
            return merged
        symbol_upper = self.symbol.upper()
        for key in (symbol_upper, self._base_symbol(symbol_upper)):
            override = overrides.get(key)
            if isinstance(override, dict):
                merged.update(override)
        return merged

    def _base_symbol(self, symbol_upper: str) -> str:
        for prefix in ("MNQ", "MES", "MCL", "MGC", "MYM", "NQ", "ES", "CL", "GC", "YM"):
            if symbol_upper.startswith(prefix):
                return prefix
        return symbol_upper

    def reset(self) -> None:
        self._last_signal_index: int | None = None
        self._last_reject_reason: str | None = None

    def get_description(self) -> str:
        return "Previous-bar breakout: completed bar closes outside a prior range, then fills at the next bar open."

    def _timestamp_et(self, value) -> pd.Timestamp | None:
        if value is None or pd.isna(value):
            return None
        ts = pd.to_datetime(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.tz_convert("America/New_York")

    def _hour_allowed(self, row) -> bool:
        if not self.allowed_entry_hours_et:
            return True
        et = self._timestamp_et(row.get("timestamp")) if "timestamp" in row else None
        return et is not None and int(et.hour) in self.allowed_entry_hours_et

    def _body_ratio(self, row) -> float:
        high = float(row.get("high", row.get("close", 0.0)))
        low = float(row.get("low", row.get("close", 0.0)))
        spread = max(self.tick_size, high - low)
        return abs(float(row.get("close", 0.0)) - float(row.get("open", row.get("close", 0.0)))) / spread

    def _passes_trend_filter(self, side: str, row, indicators_row) -> bool:
        if not self.use_ema_filter:
            return True
        ema = indicators_row.get(self.ema_column)
        if ema is None or pd.isna(ema):
            return True
        close = float(row.get("close", row.get("price", 0.0)))
        if side == "Buy":
            return close >= float(ema)
        return close <= float(ema)

    def get_snapshot(self, index, bars, indicators) -> dict[str, Any]:
        return {
            "status": "Active" if index >= self.lookback else f"Warming up ({index}/{self.lookback})",
            "strategy": self.name,
            "lookback": self.lookback,
            "last_signal_index": self._last_signal_index,
            "last_reject_reason": self._last_reject_reason,
        }

    def on_bar(self, index, bars, indicators):
        if index < max(self.lookback, self.warmup_bars):
            return None
        if self._last_signal_index is not None and index - self._last_signal_index < self.min_signal_gap_bars:
            return None

        row = bars.iloc[index]
        indicators_row = indicators.iloc[index]
        if not self._hour_allowed(row):
            self._last_reject_reason = "hour outside allowed set"
            return None

        high = float(row.get("high", row.get("close", 0.0)))
        low = float(row.get("low", row.get("close", 0.0)))
        close = float(row.get("close", row.get("price", 0.0)))
        range_ticks = (high - low) / self.tick_size
        if range_ticks < self.min_range_ticks:
            self._last_reject_reason = "range below threshold"
            return None
        if self._body_ratio(row) < self.min_body_ratio:
            self._last_reject_reason = "body ratio below threshold"
            return None

        adx = indicators_row.get("adx")
        if self.min_adx > 0 and adx is not None and not pd.isna(adx) and float(adx) < self.min_adx:
            self._last_reject_reason = "adx below threshold"
            return None

        prior = bars.iloc[index - self.lookback : index]
        prior_high = float(prior["high"].astype(float).max())
        prior_low = float(prior["low"].astype(float).min())
        buffer = self.breakout_buffer_ticks * self.tick_size

        side = None
        reason = ""
        if close > prior_high + buffer:
            side = "Buy" if self.direction_mode == "continuation" else "Sell"
            reason = (
                "Completed bar closed above prior range"
                if self.direction_mode == "continuation"
                else "Completed upside breakout faded at next open"
            )
        elif close < prior_low - buffer:
            side = "Sell" if self.direction_mode == "continuation" else "Buy"
            reason = (
                "Completed bar closed below prior range"
                if self.direction_mode == "continuation"
                else "Completed downside breakout faded at next open"
            )
        else:
            self._last_reject_reason = "no prior-range breakout"
            return None

        if not self._passes_trend_filter(side, row, indicators_row):
            self._last_reject_reason = "ema filter rejected signal"
            return None

        self._last_signal_index = index
        self._last_reject_reason = None
        confidence = min(0.9, 0.45 + min(0.25, range_ticks / 120.0) + min(0.15, self._body_ratio(row) * 0.15))
        return StrategySignal(
            strategy_name=self.name,
            side=side,
            quantity=self.quantity,
            sl_ticks=self.sl_ticks,
            tp_ticks=self.tp_ticks,
            confidence=confidence,
            reason=reason,
            metadata={
                "lookback": self.lookback,
                "direction_mode": self.direction_mode,
                "prior_high": prior_high,
                "prior_low": prior_low,
                "breakout_buffer_ticks": self.breakout_buffer_ticks,
                "range_ticks": round(range_ticks, 2),
                "body_ratio": round(self._body_ratio(row), 3),
                "signal_index": index,
                "execution_model": "next_bar_open_required",
            },
        )
