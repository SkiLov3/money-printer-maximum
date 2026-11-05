from __future__ import annotations

from typing import Any

import pandas as pd

from .base import BaseStrategy, StrategySignal


class VolatilitySqueezeBreakoutStrategy(BaseStrategy):
    name = "volatility_squeeze_breakout"
    warmup_bars = 120

    def __init__(self, symbol: str, params: dict[str, Any] | None = None):
        super().__init__(symbol, params)
        self.params = self._apply_symbol_overrides(self.params)
        self.tick_size = float(self.params.get("tick_size", 0.25))
        self.quantity = int(self.params.get("quantity", 1))
        self.bb_period = int(self.params.get("bb_period", 20))
        self.bb_mult = float(self.params.get("bb_mult", 2.0))
        self.squeeze_lookback = int(self.params.get("squeeze_lookback", 80))
        self.squeeze_quantile = float(self.params.get("squeeze_quantile", 0.25))
        self.breakout_buffer_ticks = int(self.params.get("breakout_buffer_ticks", 1))
        self.min_volume_ratio = float(self.params.get("min_volume_ratio", 0.0))
        self.volume_lookback = int(self.params.get("volume_lookback", 20))
        self.side_mode = str(self.params.get("side_mode", "ema_bias")).strip().lower()
        self.sl_ticks = int(self.params.get("sl_ticks", 40))
        self.tp_ticks = int(self.params.get("tp_ticks", 80))
        self.min_signal_gap_bars = int(self.params.get("min_signal_gap_bars", 8))
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

    @staticmethod
    def _base_symbol(symbol: str) -> str:
        symbol_upper = str(symbol or "").upper()
        for prefix in ("MNQ", "MES", "MCL", "MGC", "MYM", "NQ", "ES", "CL", "GC", "YM"):
            if symbol_upper.startswith(prefix):
                return prefix
        return symbol_upper

    def reset(self) -> None:
        self._last_signal_index: int | None = None
        self._last_reject_reason: str | None = None

    def get_description(self) -> str:
        return "Volatility squeeze breakout: low Bollinger bandwidth arms a resting stop in the biased direction."

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

    def get_snapshot(self, index, bars, indicators) -> dict[str, Any]:
        return {
            "status": "Active" if index >= self.warmup_bars else f"Warming up ({index}/{self.warmup_bars})",
            "strategy": self.name,
            "last_signal_index": self._last_signal_index,
            "last_reject_reason": self._last_reject_reason,
        }

    def on_bar(self, index, bars, indicators):
        min_index = max(self.warmup_bars, self.bb_period + self.squeeze_lookback)
        if index < min_index:
            return None
        if self._last_signal_index is not None and index - self._last_signal_index < self.min_signal_gap_bars:
            return None
        row = bars.iloc[index]
        if not self._hour_allowed(row):
            self._last_reject_reason = "hour outside allowed set"
            return None

        close_series = bars.iloc[: index + 1]["close"].astype(float)
        basis = close_series.rolling(self.bb_period).mean()
        stdev = close_series.rolling(self.bb_period).std(ddof=0)
        upper = basis + stdev * self.bb_mult
        lower = basis - stdev * self.bb_mult
        bandwidth = (upper - lower) / basis.abs().clip(lower=self.tick_size)
        current_bw = float(bandwidth.iloc[index])
        history = bandwidth.iloc[index - self.squeeze_lookback : index].dropna()
        if history.empty:
            return None
        threshold = float(history.quantile(self.squeeze_quantile))
        if current_bw > threshold:
            self._last_reject_reason = "bandwidth not compressed"
            return None

        volume_ratio = self._volume_ratio(index, bars)
        if self.min_volume_ratio > 0 and volume_ratio is not None and volume_ratio < self.min_volume_ratio:
            self._last_reject_reason = "volume ratio below threshold"
            return None

        close = float(row.get("close", row.get("price", 0.0)))
        ema = float(indicators.iloc[index].get("ema_21", close) or close)
        if self.side_mode == "long_only":
            side = "Buy"
        elif self.side_mode == "short_only":
            side = "Sell"
        elif close >= ema:
            side = "Buy"
        else:
            side = "Sell"
        trigger = float(upper.iloc[index]) + self.breakout_buffer_ticks * self.tick_size if side == "Buy" else float(lower.iloc[index]) - self.breakout_buffer_ticks * self.tick_size
        self._last_signal_index = index
        self._last_reject_reason = None
        return StrategySignal(
            strategy_name=self.name,
            side=side,
            quantity=self.quantity,
            sl_ticks=self.sl_ticks,
            tp_ticks=self.tp_ticks,
            confidence=0.56,
            reason=f"{side} volatility squeeze resting breakout",
            metadata={
                "entry_order_type": "stop_market",
                "entry_trigger_price": round(trigger, 8),
                "trigger_source": "bollinger_squeeze",
                "active_from_index": index + 1,
                "bb_period": self.bb_period,
                "bb_mult": self.bb_mult,
                "bandwidth": round(current_bw, 6),
                "bandwidth_threshold": round(threshold, 6),
                "volume_ratio": volume_ratio,
                "signal_index": index,
                "execution_model": "resting_stop_after_completed_squeeze",
            },
        )

    def _volume_ratio(self, index: int, bars: pd.DataFrame) -> float | None:
        if self.volume_lookback <= 0 or "volume" not in bars.columns:
            return None
        history = bars.iloc[max(0, index - self.volume_lookback) : index]["volume"].astype(float)
        if history.empty:
            return None
        avg = float(history.mean())
        if avg <= 0:
            return None
        return float(bars.iloc[index].get("volume", 0.0)) / avg
