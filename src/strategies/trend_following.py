from __future__ import annotations

from typing import Any

import pandas as pd

from .base import BaseStrategy, StrategySignal


class TrendFollowingStrategy(BaseStrategy):
    name = "trend_following"
    warmup_bars = 200

    def __init__(self, symbol: str, params: dict[str, Any] | None = None):
        super().__init__(symbol, params)
        self.sl_ticks = int(self.params.get("sl_ticks", 40))
        self.tp_ticks = int(self.params.get("tp_ticks", 80))
        self.min_atr = float(self.params.get("min_atr", 1.0))
        self.max_atr = float(self.params.get("max_atr", 100.0))
        self.adx_threshold = float(self.params.get("adx_threshold", 20))
        self.rsi_overbought = float(self.params.get("rsi_overbought", 70))
        self.rsi_oversold = float(self.params.get("rsi_oversold", 30))
        self.quantity = int(self.params.get("quantity", 1))

    def get_description(self) -> str:
        return (
            "Trend continuation: EMA 9/21 Alignment (Fanned) with EMA-200, VWAP, Ichimoku cloud, "
            f"ADX>{self.adx_threshold:.0f}, RSI {self.rsi_oversold:.0f}/{self.rsi_overbought:.0f}."
        )

    def _reasoning(self, last, prev) -> str:
        if pd.isna(last["atr"]):
            return "Calculating indicators..."
        if last["atr"] < self.min_atr:
            return f"Vol Low (ATR {last['atr']:.2f} < {self.min_atr})"
        if last["atr"] > self.max_atr:
            return f"Vol High (ATR {last['atr']:.2f} > {self.max_atr})"
        if last["adx"] < self.adx_threshold:
            return f"Trend Weak (ADX {last['adx']:.1f} < {self.adx_threshold:.0f})"

        price = last["price"]
        vwap = last.get("vwap", price)
        ema200 = last.get("ema_200", price)
        rsi = last.get("rsi", 50)
        senkou_a = last.get("senkou_a", 0)
        senkou_b = last.get("senkou_b", 0)
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)

        long_reason = ""
        momentum_override = last["adx"] > 35
        if price <= ema200:
            long_reason = "Price < EMA 200"
        elif price <= vwap:
            long_reason = "Price < VWAP"
        elif price <= cloud_top:
            long_reason = "Price in/below Cloud"
        elif rsi >= self.rsi_overbought and not momentum_override:
            long_reason = "RSI Overbought"
        elif not (last["ema_9"] > last["ema_21"] and last["ema_21"] > ema200):
            long_reason = "EMAs not aligned"
        else:
            long_reason = "Ready"

        short_reason = ""
        if price >= ema200:
            short_reason = "Price > EMA 200"
        elif price >= vwap:
            short_reason = "Price > VWAP"
        elif price >= cloud_bottom:
            short_reason = "Price in/above Cloud"
        elif rsi <= self.rsi_oversold and not momentum_override:
            short_reason = "RSI Oversold"
        elif not (last["ema_9"] < last["ema_21"] and last["ema_21"] < ema200):
            short_reason = "EMAs not aligned"
        else:
            short_reason = "Ready"

        if price > ema200:
            return f"LONG: {long_reason} | SHORT: Trend Bullish"
        if price < ema200:
            return f"LONG: Trend Bearish | SHORT: {short_reason}"
        return "Neutral (At EMA 200)"

    def get_snapshot(self, index, bars, indicators) -> dict[str, Any]:
        last = indicators.iloc[index]
        prev = indicators.iloc[index - 1] if index > 0 else last

        def clean(value):
            if value is None or pd.isna(value):
                return None
            return float(value)

        bar_count = index + 1
        return {
            "status": "Active" if bar_count >= self.warmup_bars else f"Warming up ({bar_count}/{self.warmup_bars})",
            "strategy": self.name,
            "bar_count": bar_count,
            "reasoning": self._reasoning(last, prev),
            "price": clean(last.get("price")),
            "ema_200": clean(last.get("ema_200")),
            "rsi": clean(last.get("rsi")),
            "adx": clean(last.get("adx")),
            "atr": clean(last.get("atr")),
            "vwap": clean(last.get("vwap")),
            "tenkan": clean(last.get("tenkan")),
            "kijun": clean(last.get("kijun")),
            "senkou_a": clean(last.get("senkou_a")),
            "senkou_b": clean(last.get("senkou_b")),
        }

    def on_bar(self, index, bars, indicators):
        if index < self.warmup_bars - 1:
            return None

        last = indicators.iloc[index]

        if pd.isna(last["atr"]) or last["atr"] < self.min_atr or last["atr"] > self.max_atr:
            return None
        if pd.isna(last["adx"]) or last["adx"] < self.adx_threshold:
            return None

        rsi_val = last.get("rsi", 50)
        if pd.isna(rsi_val):
            rsi_val = 50

        price = last["price"]
        vwap = last.get("vwap", price)
        senkou_a = last.get("senkou_a", 0)
        senkou_b = last.get("senkou_b", 0)

        bullish_cloud = not pd.isna(senkou_a) and price > max(senkou_a, senkou_b)
        bearish_cloud = not pd.isna(senkou_a) and price < min(senkou_a, senkou_b)

        momentum_override = last["adx"] > 35
        if price > last["ema_200"] and price > vwap and bullish_cloud:
            if rsi_val < self.rsi_overbought or momentum_override:
                if last["ema_9"] > last["ema_21"] and last["ema_21"] > last["ema_200"]:
                    return StrategySignal(
                        strategy_name=self.name,
                        side="Buy",
                        quantity=self.quantity,
                        sl_ticks=self.sl_ticks,
                        tp_ticks=self.tp_ticks,
                        confidence=min(0.95, max(0.45, float(last["adx"]) / 100.0)),
                        reason="Bullish trend alignment" + (" (Momentum Override)" if momentum_override else ""),
                        metadata={"adx": float(last["adx"]), "rsi": float(rsi_val)},
                    )

        if price < last["ema_200"] and price < vwap and bearish_cloud:
            if rsi_val > self.rsi_oversold or momentum_override:
                if last["ema_9"] < last["ema_21"] and last["ema_21"] < last["ema_200"]:
                    return StrategySignal(
                        strategy_name=self.name,
                        side="Sell",
                        quantity=self.quantity,
                        sl_ticks=self.sl_ticks,
                        tp_ticks=self.tp_ticks,
                        confidence=min(0.95, max(0.45, float(last["adx"]) / 100.0)),
                        reason="Bearish trend alignment" + (" (Momentum Override)" if momentum_override else ""),
                        metadata={"adx": float(last["adx"]), "rsi": float(rsi_val)},
                    )

        return None
