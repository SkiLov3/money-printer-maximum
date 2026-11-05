from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import BaseStrategy, StrategySignal


class SimpleMarketMetricsStrategy(BaseStrategy):
    name = "simple_market_metrics"
    warmup_bars = 220

    def __init__(self, symbol: str, params: dict[str, Any] | None = None):
        super().__init__(symbol, params)
        self.params = self._apply_symbol_overrides(self.params)
        self.tick_size = float(self.params.get("tick_size", 0.25))
        self.quantity = int(self.params.get("quantity", 1))
        self.sl_ticks = int(self.params.get("sl_ticks", 30))
        self.tp_ticks = int(self.params.get("tp_ticks", self._default_target_ticks()))
        self.require_stacked_emas = bool(self.params.get("require_stacked_emas", True))
        self.use_ma_filter = bool(self.params.get("use_ma_filter", True))
        self.use_chop_filter = bool(self.params.get("use_chop_filter", True))
        self.use_profit_wave_exit = bool(self.params.get("use_profit_wave_exit", True))
        self.exit_on_dashboard_reversal = bool(self.params.get("exit_on_dashboard_reversal", True))
        self.signal_mode = str(self.params.get("signal_mode", "normal")).strip().lower()
        self.wick_floor_match = bool(self.params.get("wick_floor_match", True))
        self.max_wick_ticks = float(self.params.get("max_wick_ticks", 1.0))
        self.min_mfi_buy = float(self.params.get("min_mfi_buy", 52.0))
        self.max_mfi_sell = float(self.params.get("max_mfi_sell", 52.0))
        self.min_di = float(self.params.get("min_di", 45.0))
        self.require_mfi_slope = bool(self.params.get("require_mfi_slope", False))
        self.mfi_slope_lookback = max(1, int(self.params.get("mfi_slope_lookback", 3)))
        self.min_mfi_slope = float(self.params.get("min_mfi_slope", 2.0))
        self.require_recent_ema_cross = bool(self.params.get("require_recent_ema_cross", False))
        self.ema_cross_lookback = max(1, int(self.params.get("ema_cross_lookback", 6)))
        self.require_linreg_slope = bool(self.params.get("require_linreg_slope", False))
        self.linreg_length = max(3, int(self.params.get("linreg_length", 34)))
        self.min_linreg_slope_ticks = float(self.params.get("min_linreg_slope_ticks", 0.2))
        self.enabled_symbols = [str(symbol).upper() for symbol in self.params.get("enabled_symbols", []) or []]
        self.allowed_entry_hours_et = [int(hour) for hour in self.params.get("allowed_entry_hours_et", [10, 11, 12, 13, 14]) or []]
        self.reset()

    def _base_symbol(self) -> str:
        symbol_upper = self.symbol.upper()
        for prefix in ("MNQ", "MES", "MCL", "MGC", "MYM", "NQ", "ES", "CL", "GC", "YM"):
            if symbol_upper.startswith(prefix):
                return prefix
        return symbol_upper

    def _default_target_ticks(self) -> int:
        base = self._base_symbol()
        if base in {"ES", "MES"}:
            return 4
        if base in {"NQ", "MNQ"}:
            return 20
        return 20

    def _apply_symbol_overrides(self, params: dict[str, Any]) -> dict[str, Any]:
        merged = dict(params or {})
        overrides = merged.get("symbol_params", {}) or {}
        if not isinstance(overrides, dict):
            return merged
        symbol_upper = self.symbol.upper()
        for key in (symbol_upper, self._base_symbol()):
            override = overrides.get(key)
            if isinstance(override, dict):
                merged.update(override)
        return merged

    def reset(self) -> None:
        self._current_mode = "none"
        self._last_trend_switch: int | None = None
        self._last_signal_index: int | None = None
        self._last_reject_reason: str | None = None

    def get_description(self) -> str:
        return "Simple Market Metrics-style HA trend, stacked profit-wave EMA, MFI, chop, and 200 EMA signal."

    def get_snapshot(self, index, bars, indicators) -> dict[str, Any]:
        return {
            "status": "Active" if index >= self.warmup_bars else f"Warming up ({index}/{self.warmup_bars})",
            "strategy": self.name,
            "mode": self._current_mode,
            "last_reject_reason": self._last_reject_reason,
        }

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

    def _reset_mode_if_needed(self, row) -> None:
        trend_switch = int(row.get("smm_trend_switch", 0) or 0)
        if self._last_trend_switch is not None and trend_switch != self._last_trend_switch:
            self._current_mode = "none"
        self._last_trend_switch = trend_switch

        real_close = float(row.get("close", row.get("price", 0.0)))
        ha_close = float(row.get("smm_ha_close", row.get("close", 0.0)))
        ema21 = float(row.get("smm_ema21", row.get("ema_21", row.get("close", 0.0))))
        if (
            (self._current_mode == "buy" and real_close < ema21 and ha_close < ema21)
            or (self._current_mode == "sell" and real_close > ema21 and ha_close > ema21)
        ):
            self._current_mode = "none"

    def _strong_candle(self, side: str, row) -> bool:
        ha_open = float(row.get("smm_ha_open", row.get("open", 0.0)))
        ha_high = float(row.get("smm_ha_high", row.get("high", 0.0)))
        ha_low = float(row.get("smm_ha_low", row.get("low", 0.0)))
        ha_close = float(row.get("smm_ha_close", row.get("close", 0.0)))
        real_close = float(row.get("close", row.get("price", 0.0)))
        ema8 = float(row.get("smm_ema8", row.get("close", 0.0)))
        ema21 = float(row.get("smm_ema21", row.get("close", 0.0)))
        if side == "Buy":
            lower_wick_ticks = max(0.0, min(ha_open, ha_close) - ha_low) / max(1e-9, self.tick_size)
            if self.wick_floor_match and lower_wick_ticks > self.max_wick_ticks:
                return False
            return ha_close > ha_open and lower_wick_ticks <= self.max_wick_ticks and real_close > ema8 and real_close > ema21
        upper_wick_ticks = max(0.0, ha_high - max(ha_open, ha_close)) / max(1e-9, self.tick_size)
        if self.wick_floor_match and upper_wick_ticks > self.max_wick_ticks:
            return False
        return ha_close < ha_open and upper_wick_ticks <= self.max_wick_ticks and real_close < ema8 and real_close < ema21

    def _mfi_slope_ok(self, side: str, index: int, indicators: pd.DataFrame) -> bool:
        if not self.require_mfi_slope:
            return True
        prior_index = index - self.mfi_slope_lookback
        if prior_index < 0 or "smm_mfi" not in indicators.columns:
            self._last_reject_reason = "mfi slope unavailable"
            return False
        current = float(indicators.iloc[index].get("smm_mfi", 50.0) or 50.0)
        prior = float(indicators.iloc[prior_index].get("smm_mfi", 50.0) or 50.0)
        slope = current - prior
        if side == "Buy" and slope < self.min_mfi_slope:
            self._last_reject_reason = "money flow not stepping up"
            return False
        if side == "Sell" and slope > -self.min_mfi_slope:
            self._last_reject_reason = "money flow not stepping down"
            return False
        return True

    def _recent_ema_cross_ok(self, side: str, index: int, indicators: pd.DataFrame) -> bool:
        if not self.require_recent_ema_cross:
            return True
        if index < 1 or "smm_ema8" not in indicators.columns or "smm_ema13" not in indicators.columns:
            self._last_reject_reason = "ema cross unavailable"
            return False
        start = max(1, index - self.ema_cross_lookback + 1)
        window = indicators.iloc[start : index + 1]
        prev = indicators.iloc[start - 1 : index]
        ema8 = window["smm_ema8"].astype(float).to_numpy()
        ema13 = window["smm_ema13"].astype(float).to_numpy()
        prev_ema8 = prev["smm_ema8"].astype(float).to_numpy()
        prev_ema13 = prev["smm_ema13"].astype(float).to_numpy()
        if side == "Buy" and bool(np.any((prev_ema8 <= prev_ema13) & (ema8 > ema13))):
            return True
        if side == "Sell" and bool(np.any((prev_ema8 >= prev_ema13) & (ema8 < ema13))):
            return True
        self._last_reject_reason = "no recent profit wave cross"
        return False

    def _linreg_slope_ok(self, side: str, index: int, indicators: pd.DataFrame) -> bool:
        if not self.require_linreg_slope:
            return True
        start = index - self.linreg_length + 1
        if start < 0 or "smm_ha_close" not in indicators.columns:
            self._last_reject_reason = "linear regression slope unavailable"
            return False
        closes = indicators.iloc[start : index + 1]["smm_ha_close"].astype(float).to_numpy()
        if len(closes) < self.linreg_length or np.isnan(closes).any():
            self._last_reject_reason = "linear regression slope unavailable"
            return False
        x = np.arange(len(closes), dtype=float)
        slope = float(np.polyfit(x, closes, 1)[0])
        slope_ticks = slope / self.tick_size
        if side == "Buy" and slope_ticks < self.min_linreg_slope_ticks:
            self._last_reject_reason = "linear regression rejected buy"
            return False
        if side == "Sell" and slope_ticks > -self.min_linreg_slope_ticks:
            self._last_reject_reason = "linear regression rejected sell"
            return False
        return True

    def _passes_filters(self, side: str, row, index: int, indicators: pd.DataFrame) -> bool:
        real_close = float(row.get("close", row.get("price", 0.0)))
        ha_close = float(row.get("smm_ha_close", row.get("close", 0.0)))
        ema8 = float(row.get("smm_ema8", row.get("close", 0.0)))
        ema13 = float(row.get("smm_ema13", row.get("close", 0.0)))
        ema21 = float(row.get("smm_ema21", row.get("close", 0.0)))
        ema200 = float(row.get("smm_ema200", row.get("close", 0.0)))
        mfi = float(row.get("smm_mfi", 50.0) or 50.0)
        plus_di = float(row.get("smm_plus_di", 0.0) or 0.0)
        minus_di = float(row.get("smm_minus_di", 0.0) or 0.0)

        if self.require_stacked_emas:
            if side == "Buy" and not (ema8 > ema13 > ema21):
                self._last_reject_reason = "profit wave not stacked bullish"
                return False
            if side == "Sell" and not (ema8 < ema13 < ema21):
                self._last_reject_reason = "profit wave not stacked bearish"
                return False

        if self.use_ma_filter:
            if side == "Buy" and not (ha_close > ema200 and real_close > ema200):
                self._last_reject_reason = "200 ema rejected buy"
                return False
            if side == "Sell" and not (ha_close < ema200 and real_close < ema200):
                self._last_reject_reason = "200 ema rejected sell"
                return False

        if self.use_chop_filter:
            if side == "Buy" and not (int(plus_di) > int(minus_di) and int(plus_di) >= self.min_di and mfi > self.min_mfi_buy):
                self._last_reject_reason = "chop/mfi rejected buy"
                return False
            if side == "Sell" and not (int(minus_di) > int(plus_di) and int(minus_di) >= self.min_di and mfi < self.max_mfi_sell):
                self._last_reject_reason = "chop/mfi rejected sell"
                return False
        if not self._mfi_slope_ok(side, index, indicators):
            return False
        if not self._recent_ema_cross_ok(side, index, indicators):
            return False
        if not self._linreg_slope_ok(side, index, indicators):
            return False
        return True

    def on_bar(self, index, bars, indicators):
        if index < self.warmup_bars:
            return None
        if self.enabled_symbols and self._base_symbol() not in self.enabled_symbols:
            return None
        row = indicators.iloc[index]
        self._reset_mode_if_needed(row)
        if not self._hour_allowed(row):
            self._last_reject_reason = "hour outside preferred session"
            return None

        trend_switch = int(row.get("smm_trend_switch", 0) or 0)
        side = None
        if trend_switch == 1 and self._current_mode != "buy" and self._strong_candle("Buy", row):
            side = "Buy"
        elif trend_switch == -1 and self._current_mode != "sell" and self._strong_candle("Sell", row):
            side = "Sell"
        else:
            self._last_reject_reason = "no fresh strong SMM signal"
            return None

        if not self._passes_filters(side, row, index, indicators):
            return None

        raw_side = side
        if self.signal_mode in {"invert", "inverse", "fade"}:
            side = "Sell" if side == "Buy" else "Buy"

        self._current_mode = "buy" if side == "Buy" else "sell"
        self._last_signal_index = index
        self._last_reject_reason = None
        return StrategySignal(
            strategy_name=self.name,
            side=side,
            quantity=self.quantity,
            sl_ticks=self.sl_ticks,
            tp_ticks=self.tp_ticks,
            confidence=0.75,
            reason="SMM all-colors-match signal on completed Heikin Ashi bar",
            metadata={
                "signal_index": int(index),
                "execution_model": "next_bar_open_required",
                "profit_wave_exit": self.use_profit_wave_exit,
                "smm_exit_on_dashboard_reversal": self.exit_on_dashboard_reversal,
                "profit_wave_slow_column": "smm_ema21",
                "smm_raw_side": raw_side,
                "smm_signal_mode": self.signal_mode,
                "smm_mfi": round(float(row.get("smm_mfi", 50.0) or 50.0), 2),
                "smm_plus_di": round(float(row.get("smm_plus_di", 0.0) or 0.0), 2),
                "smm_minus_di": round(float(row.get("smm_minus_di", 0.0) or 0.0), 2),
                "smm_stacked_emas": self.require_stacked_emas,
                "smm_mfi_slope_filter": self.require_mfi_slope,
                "smm_recent_ema_cross_filter": self.require_recent_ema_cross,
                "smm_linreg_slope_filter": self.require_linreg_slope,
            },
        )
