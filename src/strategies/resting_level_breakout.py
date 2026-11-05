from __future__ import annotations

from datetime import time
from typing import Any

import pandas as pd

from .base import BaseStrategy, StrategySignal


class RestingLevelBreakoutStrategy(BaseStrategy):
    name = "resting_level_breakout"
    warmup_bars = 50

    def __init__(self, symbol: str, params: dict[str, Any] | None = None):
        super().__init__(symbol, params)
        self.params = self._apply_symbol_overrides(self.params)
        self.tick_size = float(self.params.get("tick_size", 0.25))
        self.quantity = int(self.params.get("quantity", 1))
        self.level_source = str(self.params.get("level_source", "rolling_donchian")).strip().lower()
        self.lookback = int(self.params.get("lookback", 20))
        self.side_mode = str(self.params.get("side_mode", "ema_bias")).strip().lower()
        self.breakout_buffer_ticks = int(self.params.get("breakout_buffer_ticks", 1))
        self.sl_ticks = int(self.params.get("sl_ticks", 40))
        self.tp_ticks = int(self.params.get("tp_ticks", 80))
        self.min_range_ticks = int(self.params.get("min_range_ticks", 8))
        self.max_range_ticks = int(self.params.get("max_range_ticks", 500))
        self.min_adx = float(self.params.get("min_adx", 0.0))
        self.min_signal_gap_bars = int(self.params.get("min_signal_gap_bars", 5))
        self.allowed_entry_hours_et = [int(hour) for hour in self.params.get("allowed_entry_hours_et", []) or []]
        self.opening_range_start = self._parse_et_time(self.params.get("opening_range_start_et", "09:30"))
        self.opening_range_end = self._parse_et_time(self.params.get("opening_range_end_et", "09:45"))
        self.trade_window_end = self._parse_et_time(self.params.get("trade_window_end_et", "12:00"))
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

    def _parse_et_time(self, value: Any) -> time:
        parts = str(value).split(":")
        return time(hour=int(parts[0]), minute=int(parts[1]))

    def reset(self) -> None:
        self._last_signal_index: int | None = None
        self._session_key: str | None = None
        self._or_high: float | None = None
        self._or_low: float | None = None
        self._or_fired_session: str | None = None
        self._last_reject_reason: str | None = None

    def get_description(self) -> str:
        return "Resting level breakout: arm stop orders from known prior levels and fill only on later trade-through."

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

    def _session_for_timestamp(self, et: pd.Timestamp) -> str:
        return et.date().isoformat()

    def _in_opening_range(self, et: pd.Timestamp) -> bool:
        return self.opening_range_start <= et.time() < self.opening_range_end

    def _in_trade_window(self, et: pd.Timestamp) -> bool:
        return self.opening_range_end <= et.time() < self.trade_window_end

    def get_snapshot(self, index, bars, indicators) -> dict[str, Any]:
        return {
            "status": "Active" if index >= self.lookback else f"Warming up ({index}/{self.lookback})",
            "strategy": self.name,
            "level_source": self.level_source,
            "last_signal_index": self._last_signal_index,
            "opening_range_high": self._or_high,
            "opening_range_low": self._or_low,
            "last_reject_reason": self._last_reject_reason,
        }

    def on_bar(self, index, bars, indicators):
        row = bars.iloc[index]
        indicators_row = indicators.iloc[index]
        if self.level_source == "opening_range":
            return self._opening_range_signal(index, row, indicators_row)
        return self._rolling_donchian_signal(index, bars, row, indicators_row)

    def _rolling_donchian_signal(self, index, bars, row, indicators_row):
        if index < max(self.lookback, self.warmup_bars):
            return None
        if self._last_signal_index is not None and index - self._last_signal_index < self.min_signal_gap_bars:
            return None
        if not self._hour_allowed(row):
            self._last_reject_reason = "hour outside allowed set"
            return None
        adx = float(indicators_row.get("adx", 0.0) or 0.0)
        if adx < self.min_adx:
            self._last_reject_reason = "adx below threshold"
            return None

        prior = bars.iloc[index - self.lookback : index]
        level_high = float(prior["high"].astype(float).max())
        level_low = float(prior["low"].astype(float).min())
        range_ticks = int(round((level_high - level_low) / self.tick_size))
        if range_ticks < self.min_range_ticks or range_ticks > self.max_range_ticks:
            self._last_reject_reason = "range outside bounds"
            return None

        close = float(row.get("close", row.get("price", 0.0)))
        ema = float(indicators_row.get("ema_21", close) or close)
        if self.side_mode == "long_only":
            side = "Buy"
        elif self.side_mode == "short_only":
            side = "Sell"
        elif close >= ema:
            side = "Buy"
        else:
            side = "Sell"
        trigger = level_high + self.breakout_buffer_ticks * self.tick_size if side == "Buy" else level_low - self.breakout_buffer_ticks * self.tick_size
        self._last_signal_index = index
        self._last_reject_reason = None
        return self._signal(side, trigger, "rolling_donchian", index, range_ticks, level_high, level_low)

    def _opening_range_signal(self, index, row, indicators_row):
        et = self._timestamp_et(row.get("timestamp")) if "timestamp" in row else None
        if et is None:
            return None
        session_key = self._session_for_timestamp(et)
        if session_key != self._session_key:
            self._session_key = session_key
            self._or_high = None
            self._or_low = None
            self._or_fired_session = None
        if self._in_opening_range(et):
            high = float(row.get("high", row.get("close", 0.0)))
            low = float(row.get("low", row.get("close", 0.0)))
            self._or_high = high if self._or_high is None else max(self._or_high, high)
            self._or_low = low if self._or_low is None else min(self._or_low, low)
            return None
        if self._or_high is None or self._or_low is None or self._or_fired_session == session_key or not self._in_trade_window(et):
            return None
        range_ticks = int(round((self._or_high - self._or_low) / self.tick_size))
        if range_ticks < self.min_range_ticks or range_ticks > self.max_range_ticks:
            self._last_reject_reason = "opening range outside bounds"
            return None
        close = float(row.get("close", row.get("price", 0.0)))
        vwap = float(indicators_row.get("vwap", close) or close)
        side = "Buy" if close >= vwap else "Sell"
        trigger = self._or_high + self.breakout_buffer_ticks * self.tick_size if side == "Buy" else self._or_low - self.breakout_buffer_ticks * self.tick_size
        if not self.is_priming:
            self._or_fired_session = session_key
        self._last_signal_index = index
        self._last_reject_reason = None
        return self._signal(side, trigger, "opening_range", index, range_ticks, self._or_high, self._or_low)

    def _signal(self, side: str, trigger: float, source: str, index: int, range_ticks: int, level_high: float, level_low: float):
        return StrategySignal(
            strategy_name=self.name,
            side=side,
            quantity=self.quantity,
            sl_ticks=self.sl_ticks,
            tp_ticks=self.tp_ticks,
            confidence=0.58,
            reason=f"{side} resting {source} breakout stop",
            metadata={
                "entry_order_type": "stop_market",
                "entry_trigger_price": float(trigger),
                "trigger_source": source,
                "active_from_index": index + 1,
                "expires_after_bars": self.params.get("expires_after_bars", None),
                "level_high": level_high,
                "level_low": level_low,
                "range_ticks": range_ticks,
                "signal_index": index,
                "execution_model": "resting_stop_after_completed_level",
            },
        )
