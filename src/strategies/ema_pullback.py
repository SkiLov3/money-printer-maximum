from __future__ import annotations

from typing import Any

import pandas as pd

from .base import BaseStrategy, StrategySignal


class EMAPullbackStrategy(BaseStrategy):
    name = "ema_pullback"
    warmup_bars = 60

    def __init__(self, symbol: str, params: dict[str, Any] | None = None):
        params = self._apply_symbol_overrides(symbol, params or {})
        super().__init__(symbol, params)
        self.quantity = int(self.params.get("quantity", 1))
        self.adx_threshold = float(self.params.get("adx_threshold", 18))
        self.pullback_window = int(self.params.get("pullback_window", 3))
        
        # Dynamic risk params
        self.atr_multiplier = float(self.params.get("atr_multiplier", 2.0))
        self.min_sl_ticks = int(self.params.get("min_sl_ticks", self.params.get("sl_ticks", 100)))
        self.min_trend_strength = float(self.params.get("min_trend_strength", 1.5))
        self.tp_sl_ratio = float(self.params.get("tp_sl_ratio", 2.0))
        
        # Helper tick multiplier calculated dynamically from tick size.
        self.tick_size = float(self.params.get("tick_size", 0.25))
        self.tick_conv = 1.0 / self.tick_size if self.tick_size > 0 else 4.0

    @classmethod
    def _apply_symbol_overrides(cls, symbol: str, params: dict[str, Any]) -> dict[str, Any]:
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

    def get_description(self) -> str:
        return "EMA pullback state machine: 14/18/24 alignment, controlled pullback, breakout continuation."

    def get_snapshot(self, index, bars, indicators) -> dict[str, Any]:
        row = indicators.iloc[index]
        return {
            "status": "Active" if index + 1 >= self.warmup_bars else f"Warming up ({index + 1}/{self.warmup_bars})",
            "strategy": self.name,
            "reasoning": "EMA pullback window tracking",
            "ema_14": self._clean(row.get("ema_14")),
            "ema_18": self._clean(row.get("ema_18")),
            "ema_24": self._clean(row.get("ema_24")),
            "adx": self._clean(row.get("adx")),
        }

    def on_bar(self, index, bars, indicators):
        if index < self.warmup_bars:
            return None

        row = indicators.iloc[index]
        prev = indicators.iloc[index - 1]
        recent_prior = indicators.iloc[max(0, index - self.pullback_window): index]
        price = float(row.get("close", row.get("price", 0.0)) or 0.0)
        adx = float(row.get("adx", 0.0) or 0.0)
        if adx < self.adx_threshold:
            return None

        long_aligned = float(row["ema_14"]) > float(row["ema_18"]) > float(row["ema_24"])
        short_aligned = float(row["ema_14"]) < float(row["ema_18"]) < float(row["ema_24"])
        ema_slope = float(row["ema_14"]) - float(prev["ema_14"])
        breakout_high = float(recent_prior["high"].max())
        breakout_low = float(recent_prior["low"].min())

        raw_atr = row.get("atr")
        atr_val = float(raw_atr) if raw_atr is not None and not pd.isna(raw_atr) else 0.0
        atr_ticks = atr_val * self.tick_conv
        dynamic_sl_ticks = max(int(self.atr_multiplier * atr_ticks), self.min_sl_ticks)
        dynamic_tp_ticks = int(dynamic_sl_ticks * self.tp_sl_ratio)

        # Normalize slope to remove scale dependency & filter weak trends
        ema_slope_norm = ema_slope / atr_val if atr_val > 0 else 0.0

        if long_aligned and ema_slope_norm >= self.min_trend_strength:
            # Strict pullback: shallow dip into EMA zone, not a breakdown
            pullback_red = recent_prior["close"] < recent_prior["open"]
            shallow = recent_prior["close"] > (float(row["ema_14"]) - atr_val * 0.5)
            pullback_seen = bool(pullback_red.any() and shallow.any())
            
            if pullback_seen and price >= breakout_high:
                return StrategySignal(
                    strategy_name=self.name,
                    side="Buy",
                    quantity=self.quantity,
                    sl_ticks=dynamic_sl_ticks,
                    tp_ticks=dynamic_tp_ticks,
                    confidence=min(0.9, 0.45 + adx / 100.0),
                    reason=f"EMA bullish pullback breakout (SL={dynamic_sl_ticks})",
                    metadata={"confluence_score": 2, "quality_tier": "B", "ema_slope_norm": round(ema_slope_norm, 2)},
                )

        if short_aligned and ema_slope_norm <= -self.min_trend_strength:
            pullback_green = recent_prior["close"] > recent_prior["open"]
            shallow = recent_prior["close"] < (float(row["ema_14"]) + atr_val * 0.5)
            pullback_seen = bool(pullback_green.any() and shallow.any())
            
            if pullback_seen and price <= breakout_low:
                return StrategySignal(
                    strategy_name=self.name,
                    side="Sell",
                    quantity=self.quantity,
                    sl_ticks=dynamic_sl_ticks,
                    tp_ticks=dynamic_tp_ticks,
                    confidence=min(0.9, 0.45 + adx / 100.0),
                    reason=f"EMA bearish pullback breakout (SL={dynamic_sl_ticks})",
                    metadata={"confluence_score": 2, "quality_tier": "B", "ema_slope_norm": round(ema_slope_norm, 2)},
                )
        return None

    @staticmethod
    def _clean(value):
        if value is None or pd.isna(value):
            return None
        return float(value)
