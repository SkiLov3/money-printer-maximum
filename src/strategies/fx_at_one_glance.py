from __future__ import annotations

from typing import Any

import pandas as pd

from .base import BaseStrategy, StrategySignal


class FXAtOneGlanceStrategy(BaseStrategy):
    name = "fx_at_one_glance"
    warmup_bars = 120

    def __init__(self, symbol: str, params: dict[str, Any] | None = None):
        super().__init__(symbol, params)
        self.quantity = int(self.params.get("quantity", 1))
        self.sl_ticks = int(self.params.get("sl_ticks", 32))
        self.tp_ticks = int(self.params.get("tp_ticks", 64))
        self.min_confluence = int(self.params.get("min_confluence", 6))
        self.max_kijun_distance_atr = float(self.params.get("max_kijun_distance_atr", 1.0))
        self.adx_threshold = float(self.params.get("adx_threshold", 20))
        self.min_score_edge = int(self.params.get("min_score_edge", 2))
        self.min_signal_gap_bars = int(self.params.get("min_signal_gap_bars", 12))
        self.require_breakout = bool(self.params.get("require_breakout", True))
        self._last_signal_index: int | None = None

    def get_description(self) -> str:
        return "FX At One Glance: bias + entry checklist using cloud position, kijun proximity, VWAP, RSI, ADX, and breakout context."

    def reset(self) -> None:
        self._last_signal_index = None

    def get_snapshot(self, index, bars, indicators) -> dict[str, Any]:
        row = indicators.iloc[index]
        buy_score, _ = self._score(row, "Buy")
        sell_score, _ = self._score(row, "Sell")
        return {
            "status": "Active" if index + 1 >= self.warmup_bars else f"Warming up ({index + 1}/{self.warmup_bars})",
            "strategy": self.name,
            "reasoning": f"buy_score={buy_score} sell_score={sell_score} min={self.min_confluence}",
            "kijun": self._clean(row.get("kijun")),
            "vwap": self._clean(row.get("vwap")),
            "rsi": self._clean(row.get("rsi")),
            "cloud_thickness": self._clean(row.get("cloud_thickness")),
        }

    def on_bar(self, index, bars, indicators):
        if index < self.warmup_bars - 1:
            return None
        row = indicators.iloc[index]
        buy_score, buy_reasons = self._score(row, "Buy")
        sell_score, sell_reasons = self._score(row, "Sell")
        if self._last_signal_index is not None and index - self._last_signal_index < self.min_signal_gap_bars:
            return None
        if self._passes_quality_gate(buy_score, sell_score, buy_reasons):
            self._last_signal_index = index
            return StrategySignal(
                strategy_name=self.name,
                side="Buy",
                quantity=self.quantity,
                sl_ticks=self.sl_ticks,
                tp_ticks=self.tp_ticks,
                confidence=min(0.95, 0.3 + buy_score * 0.08),
                reason="FXAOG bullish checklist",
                metadata={"confluence_score": buy_score, "confluence_reasons": buy_reasons, "quality_tier": self._tier(buy_score)},
            )
        if self._passes_quality_gate(sell_score, buy_score, sell_reasons):
            self._last_signal_index = index
            return StrategySignal(
                strategy_name=self.name,
                side="Sell",
                quantity=self.quantity,
                sl_ticks=self.sl_ticks,
                tp_ticks=self.tp_ticks,
                confidence=min(0.95, 0.3 + sell_score * 0.08),
                reason="FXAOG bearish checklist",
                metadata={"confluence_score": sell_score, "confluence_reasons": sell_reasons, "quality_tier": self._tier(sell_score)},
            )
        return None

    def _score(self, row, side: str):
        score = 0
        reasons = []
        price = float(row.get("price", row.get("close", 0.0)) or 0.0)
        vwap = float(row.get("vwap", price) or price)
        ema_200 = float(row.get("ema_200", price) or price)
        kijun = row.get("kijun")
        atr = float(row.get("atr", 0.0) or 0.0)
        rsi = float(row.get("rsi", 50.0) or 50.0)
        adx = float(row.get("adx", 0.0) or 0.0)
        cloud_thickness = float(row.get("cloud_thickness", 0.0) or 0.0)
        rolling_high = float(row.get("rolling_high_20", price) or price)
        rolling_low = float(row.get("rolling_low_20", price) or price)
        bullish = side == "Buy"

        if bullish and price > ema_200:
            score += 1
            reasons.append("bias_above_ema200")
        if (not bullish) and price < ema_200:
            score += 1
            reasons.append("bias_below_ema200")

        if bullish and price > vwap:
            score += 1
            reasons.append("above_vwap")
        if (not bullish) and price < vwap:
            score += 1
            reasons.append("below_vwap")

        if kijun is not None and not pd.isna(kijun) and atr > 0:
            kijun_distance_atr = abs(price - float(kijun)) / atr
            if kijun_distance_atr <= self.max_kijun_distance_atr:
                score += 1
                reasons.append("kijun_proximity")

        if adx >= self.adx_threshold:
            score += 1
            reasons.append("adx_ok")

        if cloud_thickness > 0:
            score += 1
            reasons.append("cloud_thickness")

        if bullish and rsi > 50:
            score += 1
            reasons.append("rsi_bullish")
        if (not bullish) and rsi < 50:
            score += 1
            reasons.append("rsi_bearish")

        if bullish and price >= rolling_high:
            score += 1
            reasons.append("breakout_high")
        if (not bullish) and price <= rolling_low:
            score += 1
            reasons.append("breakout_low")

        return score, reasons

    def _passes_quality_gate(self, score: int, opposite_score: int, reasons: list[str]) -> bool:
        if score < self.min_confluence:
            return False
        if score - opposite_score < self.min_score_edge:
            return False
        if self.require_breakout and not any(reason in ("breakout_high", "breakout_low") for reason in reasons):
            return False
        return True

    @staticmethod
    def _tier(score: int) -> str:
        if score >= 7:
            return "A+"
        if score >= 6:
            return "A"
        if score >= 5:
            return "B"
        return "C"

    @staticmethod
    def _clean(value):
        if value is None or pd.isna(value):
            return None
        return float(value)
