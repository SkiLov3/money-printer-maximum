from __future__ import annotations

from typing import Any

import pandas as pd

from .base import BaseStrategy, StrategySignal


class PriorExtremeVWAPReversionStrategy(BaseStrategy):
    name = "prior_extreme_vwap_reversion"
    warmup_bars = 50

    def __init__(self, symbol: str, params: dict[str, Any] | None = None):
        super().__init__(symbol, params)
        self.params = self._apply_symbol_overrides(self.params)
        self.tick_size = float(self.params.get("tick_size", 0.25))
        self.quantity = int(self.params.get("quantity", 1))
        self.lookback = int(self.params.get("lookback", 20))
        self.breakout_buffer_ticks = int(self.params.get("breakout_buffer_ticks", 1))
        self.reclaim_ticks = int(self.params.get("reclaim_ticks", 0))
        self.min_vwap_distance_atr = float(self.params.get("min_vwap_distance_atr", 0.75))
        self.rsi_oversold = float(self.params.get("rsi_oversold", 30.0))
        self.rsi_overbought = float(self.params.get("rsi_overbought", 70.0))
        self.min_wick_ratio = float(self.params.get("min_wick_ratio", 0.35))
        self.max_signal_range_atr = float(self.params.get("max_signal_range_atr", 3.0))
        self.sl_ticks = int(self.params.get("sl_ticks", 40))
        self.tp_ticks = int(self.params.get("tp_ticks", 32))
        self.min_signal_gap_bars = int(self.params.get("min_signal_gap_bars", 6))
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
        return "Prior extreme VWAP reversion: fade completed-bar stop-run/reclaim extremes at next bar open."

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

    def _wick_ratio(self, side: str, row) -> float:
        open_ = float(row.get("open", row.get("close", 0.0)))
        high = float(row.get("high", row.get("close", 0.0)))
        low = float(row.get("low", row.get("close", 0.0)))
        close = float(row.get("close", row.get("price", 0.0)))
        spread = max(self.tick_size, high - low)
        if side == "Buy":
            wick = min(open_, close) - low
        else:
            wick = high - max(open_, close)
        return max(0.0, wick / spread)

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
        ind = indicators.iloc[index]
        if not self._hour_allowed(row):
            self._last_reject_reason = "hour outside allowed set"
            return None

        close = float(row.get("close", row.get("price", 0.0)))
        high = float(row.get("high", close))
        low = float(row.get("low", close))
        atr = float(ind.get("atr", 0.0) or 0.0)
        vwap = float(ind.get("vwap", close) or close)
        rsi = float(ind.get("rsi", 50.0) or 50.0)
        if atr <= 0:
            self._last_reject_reason = "missing atr"
            return None
        if (high - low) / atr > self.max_signal_range_atr:
            self._last_reject_reason = "signal range too large"
            return None

        prior = bars.iloc[index - self.lookback : index]
        prior_high = float(prior["high"].astype(float).max())
        prior_low = float(prior["low"].astype(float).min())
        breakout_buffer = self.breakout_buffer_ticks * self.tick_size
        reclaim = self.reclaim_ticks * self.tick_size

        side = None
        reason = ""
        vwap_distance_atr = 0.0
        if low < prior_low - breakout_buffer and close > prior_low + reclaim:
            side = "Buy"
            reason = "Prior low stop-run reclaimed below VWAP"
            vwap_distance_atr = (vwap - close) / atr
            if rsi > self.rsi_oversold or vwap_distance_atr < self.min_vwap_distance_atr:
                self._last_reject_reason = "long exhaustion filters rejected"
                return None
        elif high > prior_high + breakout_buffer and close < prior_high - reclaim:
            side = "Sell"
            reason = "Prior high stop-run reclaimed above VWAP"
            vwap_distance_atr = (close - vwap) / atr
            if rsi < self.rsi_overbought or vwap_distance_atr < self.min_vwap_distance_atr:
                self._last_reject_reason = "short exhaustion filters rejected"
                return None
        else:
            self._last_reject_reason = "no stop-run reclaim"
            return None

        wick_ratio = self._wick_ratio(side, row)
        if wick_ratio < self.min_wick_ratio:
            self._last_reject_reason = "wick ratio below threshold"
            return None

        self._last_signal_index = index
        self._last_reject_reason = None
        confidence = min(0.9, 0.45 + min(0.2, vwap_distance_atr * 0.08) + min(0.15, wick_ratio * 0.15))
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
                "prior_high": prior_high,
                "prior_low": prior_low,
                "vwap": vwap,
                "rsi": round(rsi, 2),
                "vwap_distance_atr": round(vwap_distance_atr, 3),
                "wick_ratio": round(wick_ratio, 3),
                "signal_index": index,
                "execution_model": "next_bar_open_required",
            },
        )
