from __future__ import annotations

import math
from typing import Any

import pandas as pd

from .base import BaseStrategy, StrategySignal


class GoldenZoneScalperStrategy(BaseStrategy):
    name = "golden_zone_scalper"
    warmup_bars = 120

    def __init__(self, symbol: str, params: dict[str, Any] | None = None):
        super().__init__(symbol, self._apply_symbol_overrides_for_symbol(symbol, params or {}))
        self.tick_size = float(self.params.get("tick_size", 0.25))
        self.quantity = int(self.params.get("quantity", 1))
        self.lookback = int(self.params.get("lookback", 48))
        self.trigger_lookback = int(self.params.get("trigger_lookback", 3))
        self.ema_fast = str(self.params.get("ema_fast", "ema_21"))
        self.ema_slow = str(self.params.get("ema_slow", "ema_50"))
        self.min_adx = float(self.params.get("min_adx", 12.0))
        self.min_ema_slope_ticks = float(self.params.get("min_ema_slope_ticks", 1.0))
        self.zone_min = float(self.params.get("zone_min", 0.382))
        self.zone_max = float(self.params.get("zone_max", 0.618))
        self.breakout_buffer_ticks = int(self.params.get("breakout_buffer_ticks", 1))
        self.stop_buffer_ticks = int(self.params.get("stop_buffer_ticks", 2))
        self.min_stop_ticks = int(self.params.get("min_stop_ticks", 8))
        self.max_stop_ticks = int(self.params.get("max_stop_ticks", 80))
        self.risk_reward = float(self.params.get("risk_reward", 2.0))
        self.min_target_ticks = int(self.params.get("min_target_ticks", 10))
        self.max_target_ticks = int(self.params.get("max_target_ticks", 0) or 0)
        self.min_body_ratio = float(self.params.get("min_body_ratio", 0.25))
        self.require_engulfing = bool(self.params.get("require_engulfing", False))
        self.allowed_sides = [str(s).lower() for s in self.params.get("allowed_sides", ["buy", "sell"])]
        self.allowed_entry_hours_et = self.params.get("allowed_entry_hours_et", None)
        self._last_signal_key: tuple[Any, ...] | None = None

    @classmethod
    def _apply_symbol_overrides_for_symbol(cls, symbol: str, params: dict[str, Any]) -> dict[str, Any]:
        merged = dict(params or {})
        overrides = merged.get("symbol_params", {}) or {}
        if not isinstance(overrides, dict):
            return merged
        symbol_upper = symbol.upper()
        keys = [symbol_upper, cls._base_symbol(symbol_upper)]
        for key in keys:
            override = overrides.get(key)
            if isinstance(override, dict):
                merged.update(override)
        return merged

    @staticmethod
    def _base_symbol(symbol_upper: str) -> str:
        known = ("MNQ", "MES", "MCL", "MGC", "MYM", "NQ", "ES", "CL", "GC", "YM")
        for prefix in known:
            if symbol_upper.startswith(prefix):
                return prefix
        return symbol_upper

    def reset(self) -> None:
        self._last_signal_key = None

    def get_description(self) -> str:
        return (
            "Golden-zone trend scalper: trade 38.2-61.8% pullbacks in EMA trend "
            "after a directional trigger bar breaks the short pullback."
        )

    def get_snapshot(self, index, bars, indicators) -> dict[str, Any]:
        row = indicators.iloc[index]
        close = self._num(row.get("close"), 0.0)
        fast = self._num(row.get(self.ema_fast), close)
        slow = self._num(row.get(self.ema_slow), close)
        return {
            "status": "Active" if index + 1 >= self.warmup_bars else f"Warming up ({index + 1}/{self.warmup_bars})",
            "strategy": self.name,
            "lookback": self.lookback,
            "zone": [self.zone_min, self.zone_max],
            "ema_fast": self.ema_fast,
            "ema_slow": self.ema_slow,
            "trend": "bullish" if close > fast > slow else ("bearish" if close < fast < slow else "neutral"),
            "adx": self._clean(row.get("adx")),
        }

    def on_bar(self, index, bars, indicators):
        if not self.enabled or index < max(self.warmup_bars, self.lookback + 2) or self.tick_size <= 0:
            return None

        row = indicators.iloc[index]
        prev = indicators.iloc[index - 1]
        close = self._num(row.get("close"), 0.0)
        open_ = self._num(row.get("open"), close)
        high = self._num(row.get("high"), close)
        low = self._num(row.get("low"), close)
        adx = self._num(row.get("adx"), 0.0)
        if adx < self.min_adx:
            return None

        if self.allowed_entry_hours_et:
            ts = row.get("timestamp")
            if ts is not None and not pd.isna(ts):
                from src.utils.session_utils import parse_timestamp_to_utc
                dt_utc = parse_timestamp_to_utc(ts, fallback_to_now=False)
                if dt_utc is not None:
                    dt_et = pd.Timestamp(dt_utc).tz_convert("America/New_York")
                    if dt_et.hour not in self.allowed_entry_hours_et:
                        return None

        fast = self._num(row.get(self.ema_fast), close)
        slow = self._num(row.get(self.ema_slow), close)
        prior_fast = self._num(indicators.iloc[index - 5].get(self.ema_fast), fast)
        ema_slope_ticks = (fast - prior_fast) / self.tick_size

        recent = bars.iloc[index - self.lookback : index + 1]
        swing_high = float(recent["high"].max())
        swing_low = float(recent["low"].min())
        impulse_ticks = (swing_high - swing_low) / self.tick_size
        if impulse_ticks < max(12, self.min_stop_ticks):
            return None

        trigger_start = max(0, index - self.trigger_lookback)
        trigger_window = bars.iloc[trigger_start:index]
        if trigger_window.empty:
            return None
        body_ratio = abs(close - open_) / max(self.tick_size, high - low)
        if body_ratio < self.min_body_ratio:
            return None

        bullish = close > fast > slow and ema_slope_ticks >= self.min_ema_slope_ticks
        bearish = close < fast < slow and ema_slope_ticks <= -self.min_ema_slope_ticks
        if bullish and "buy" in self.allowed_sides:
            return self._maybe_long(index, row, prev, trigger_window, swing_high, swing_low)
        if bearish and "sell" in self.allowed_sides:
            return self._maybe_short(index, row, prev, trigger_window, swing_high, swing_low)
        return None

    def _maybe_long(self, index, row, prev, trigger_window, swing_high: float, swing_low: float):
        close = self._num(row.get("close"), 0.0)
        open_ = self._num(row.get("open"), close)
        low = self._num(row.get("low"), close)
        zone_high = swing_high - (swing_high - swing_low) * self.zone_min   # ~61.8% retracement
        zone_low = swing_high - (swing_high - swing_low) * self.zone_max    # ~38.2% retracement
        window_low = min(float(trigger_window["low"].min()), low)
        touched_zone = (low <= zone_high and low >= zone_low) or (window_low <= zone_high and window_low >= zone_low)
        if not touched_zone or close < zone_low:
            return None
        if close <= open_:
            return None
        if self.require_engulfing and not self._bullish_engulfing(row, prev):
            return None
        pullback_high = float(trigger_window["high"].max())
        if close < pullback_high + self.breakout_buffer_ticks * self.tick_size:
            return None
        stop_price = window_low - self.stop_buffer_ticks * self.tick_size
        return self._build_signal(index, "Buy", close, stop_price, zone_low, zone_high, swing_low, swing_high)

    def _maybe_short(self, index, row, prev, trigger_window, swing_high: float, swing_low: float):
        close = self._num(row.get("close"), 0.0)
        open_ = self._num(row.get("open"), close)
        high = self._num(row.get("high"), close)
        zone_low = swing_low + (swing_high - swing_low) * self.zone_min    # ~38.2% bounce
        zone_high = swing_low + (swing_high - swing_low) * self.zone_max   # ~61.8% bounce
        window_high = max(float(trigger_window["high"].max()), high)
        touched_zone = (high >= zone_low and high <= zone_high) or (window_high >= zone_low and window_high <= zone_high)
        if not touched_zone or close > zone_high:
            return None
        if close >= open_:
            return None
        if self.require_engulfing and not self._bearish_engulfing(row, prev):
            return None
        pullback_low = float(trigger_window["low"].min())
        if close > pullback_low - self.breakout_buffer_ticks * self.tick_size:
            return None
        stop_price = window_high + self.stop_buffer_ticks * self.tick_size
        return self._build_signal(index, "Sell", close, stop_price, zone_low, zone_high, swing_low, swing_high)

    def _build_signal(
        self,
        index: int,
        side: str,
        entry_reference: float,
        stop_price: float,
        zone_low: float,
        zone_high: float,
        swing_low: float,
        swing_high: float,
    ):
        sl_ticks = int(math.ceil(round(abs(entry_reference - stop_price) / self.tick_size, 4)))
        if sl_ticks < self.min_stop_ticks:
            sl_ticks = self.min_stop_ticks
            stop_price = entry_reference - (sl_ticks * self.tick_size) if side == "Buy" else entry_reference + (sl_ticks * self.tick_size)
        if self.max_stop_ticks > 0 and sl_ticks > self.max_stop_ticks:
            return None
        tp_ticks = int(math.ceil(sl_ticks * max(0.1, self.risk_reward)))
        tp_ticks = max(tp_ticks, self.min_target_ticks)
        if self.max_target_ticks > 0:
            tp_ticks = min(tp_ticks, self.max_target_ticks)
        signal_key = (side, index)
        if signal_key == self._last_signal_key:
            return None
        self._last_signal_key = signal_key
        return StrategySignal(
            strategy_name=self.name,
            side=side,
            quantity=self.quantity,
            sl_ticks=sl_ticks,
            tp_ticks=tp_ticks,
            confidence=0.58,
            reason=f"{side} golden-zone trend pullback",
            metadata={
                "setup_id": "GZ-01",
                "setup_name": "Golden Zone Trend Pullback",
                "entry_reference": entry_reference,
                "stop_price": stop_price,
                "stop_distance_ticks": sl_ticks,
                "target_ticks": tp_ticks,
                "risk_reward": self.risk_reward,
                "zone_low": zone_low,
                "zone_high": zone_high,
                "swing_low": swing_low,
                "swing_high": swing_high,
            },
        )

    @staticmethod
    def _bullish_engulfing(row, prev) -> bool:
        return (
            float(row["close"]) > float(row["open"])
            and float(prev["close"]) < float(prev["open"])
            and float(row["close"]) >= float(prev["open"])
            and float(row["open"]) <= float(prev["close"])
        )

    @staticmethod
    def _bearish_engulfing(row, prev) -> bool:
        return (
            float(row["close"]) < float(row["open"])
            and float(prev["close"]) > float(prev["open"])
            and float(row["close"]) <= float(prev["open"])
            and float(row["open"]) >= float(prev["close"])
        )

    @staticmethod
    def _num(value: Any, default: float) -> float:
        try:
            if value is None or pd.isna(value):
                return float(default)
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _clean(value: Any):
        try:
            if value is None or pd.isna(value):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None
