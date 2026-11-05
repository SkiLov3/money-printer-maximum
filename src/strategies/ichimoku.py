from __future__ import annotations

from typing import Any

import pandas as pd

from .base import BaseStrategy, StrategySignal


class IchimokuStrategy(BaseStrategy):
    name = "ichimoku"
    warmup_bars = 120

    def __init__(self, symbol: str, params: dict[str, Any] | None = None):
        super().__init__(symbol, params)
        self.quantity = int(self.params.get("quantity", 1))
        self.sl_ticks = int(self.params.get("sl_ticks", 36))
        self.tp_ticks = int(self.params.get("tp_ticks", 72))
        self.min_confluence = int(self.params.get("min_confluence", 4))
        self.adx_threshold = float(self.params.get("adx_threshold", 20))
        self.require_chikou = bool(self.params.get("require_chikou", True))

    def get_description(self) -> str:
        return (
            "Ichimoku cascade: cloud direction, TK alignment, chikou confirmation, "
            "ADX trend strength, and VWAP agreement."
        )

    def get_snapshot(self, index, bars, indicators) -> dict[str, Any]:
        row = indicators.iloc[index]
        score_buy = self._score_side(row, "Buy")[0]
        score_sell = self._score_side(row, "Sell")[0]
        return {
            "status": "Active" if index + 1 >= self.warmup_bars else f"Warming up ({index + 1}/{self.warmup_bars})",
            "strategy": self.name,
            "reasoning": f"buy_score={score_buy} sell_score={score_sell} min={self.min_confluence}",
            "tenkan": self._clean(row.get("tenkan")),
            "kijun": self._clean(row.get("kijun")),
            "senkou_a": self._clean(row.get("senkou_a")),
            "senkou_b": self._clean(row.get("senkou_b")),
            "adx": self._clean(row.get("adx")),
        }

    def on_bar(self, index, bars, indicators):
        if index < self.warmup_bars - 1:
            return None
        row = indicators.iloc[index]

        # BAR MOMENTUM FILTER: Skip entries when the triggering bar is an
        # exhaustion spike (body > 1.5× ATR).  Entering after a large candle
        # means chasing the tail of a move — the pullback clips our stop.
        atr = float(row.get("atr", 0) or 0)
        if atr > 0:
            bar_body = abs(float(row.get("close", 0)) - float(row.get("open", 0)))
            if bar_body > 1.5 * atr:
                return None

        buy_score, buy_reasons = self._score_side(row, "Buy")
        sell_score, sell_reasons = self._score_side(row, "Sell")

        if buy_score >= self.min_confluence and buy_score > sell_score:
            return StrategySignal(
                strategy_name=self.name,
                side="Buy",
                quantity=self.quantity,
                sl_ticks=self.sl_ticks,
                tp_ticks=self.tp_ticks,
                confidence=min(0.95, 0.35 + (buy_score * 0.1)),
                reason="Ichimoku bullish confluence",
                metadata={"confluence_score": buy_score, "confluence_reasons": buy_reasons, "quality_tier": self._tier(buy_score)},
            )
        if sell_score >= self.min_confluence and sell_score > buy_score:
            return StrategySignal(
                strategy_name=self.name,
                side="Sell",
                quantity=self.quantity,
                sl_ticks=self.sl_ticks,
                tp_ticks=self.tp_ticks,
                confidence=min(0.95, 0.35 + (sell_score * 0.1)),
                reason="Ichimoku bearish confluence",
                metadata={"confluence_score": sell_score, "confluence_reasons": sell_reasons, "quality_tier": self._tier(sell_score)},
            )
        return None

    def _score_side(self, row, side: str):
        score = 0
        reasons = []
        price = float(row.get("price", row.get("close", 0.0)) or 0.0)
        tenkan = row.get("tenkan")
        kijun = row.get("kijun")
        senkou_a = row.get("senkou_a")
        senkou_b = row.get("senkou_b")
        adx = float(row.get("adx", 0.0) or 0.0)
        vwap = float(row.get("vwap", price) or price)
        chikou_ref = row.get("close_lag_26")

        if any(pd.isna(v) for v in (tenkan, kijun, senkou_a, senkou_b)):
            return 0, reasons
        cloud_top = max(float(senkou_a), float(senkou_b))
        cloud_bottom = min(float(senkou_a), float(senkou_b))
        bullish = side == "Buy"

        if bullish and price > cloud_top:
            score += 1
            reasons.append("above_cloud")
        if (not bullish) and price < cloud_bottom:
            score += 1
            reasons.append("below_cloud")

        if bullish and float(tenkan) > float(kijun):
            score += 1
            reasons.append("tk_bullish")
        if (not bullish) and float(tenkan) < float(kijun):
            score += 1
            reasons.append("tk_bearish")

        if bullish and price > vwap:
            score += 1
            reasons.append("above_vwap")
        if (not bullish) and price < vwap:
            score += 1
            reasons.append("below_vwap")

        if adx >= self.adx_threshold:
            score += 1
            reasons.append("adx_trending")

        if not self.require_chikou or not pd.isna(chikou_ref):
            if bullish and (pd.isna(chikou_ref) or price > float(chikou_ref)):
                score += 1
                reasons.append("chikou_confirmed")
            if (not bullish) and (pd.isna(chikou_ref) or price < float(chikou_ref)):
                score += 1
                reasons.append("chikou_confirmed")

        return score, reasons

    @staticmethod
    def _tier(score: int) -> str:
        if score >= 5:
            return "A"
        if score >= 4:
            return "B"
        return "C"

    @staticmethod
    def _clean(value):
        if value is None or pd.isna(value):
            return None
        return float(value)
