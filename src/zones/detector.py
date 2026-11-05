from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class PriceZone:
    zone_type: str
    price_low: float
    price_high: float
    strength: float
    touch_count: int = 0
    status: str = "active"


class ZoneDetector:
    def __init__(self, lookback: int = 80, pivot_window: int = 4, zone_half_atr: float = 0.5):
        self.lookback = lookback
        self.pivot_window = pivot_window
        self.zone_half_atr = zone_half_atr

    def detect(self, df: pd.DataFrame) -> list[PriceZone]:
        if df.empty or len(df) < max(self.lookback, self.pivot_window * 2 + 1):
            return []
        bars = df.tail(self.lookback).reset_index(drop=True)
        atr = float(bars.iloc[-1].get("atr", 0.0) or 0.0)
        half_width = atr * self.zone_half_atr if atr > 0 else max(float(bars["close"].std() or 0.0), 0.25)
        zones: list[PriceZone] = []
        for idx in range(self.pivot_window, len(bars) - self.pivot_window):
            window = bars.iloc[idx - self.pivot_window : idx + self.pivot_window + 1]
            high = float(bars.iloc[idx]["high"])
            low = float(bars.iloc[idx]["low"])
            if high >= float(window["high"].max()):
                zones.append(PriceZone("supply", high - half_width, high + half_width, strength=1.0))
            if low <= float(window["low"].min()):
                zones.append(PriceZone("demand", low - half_width, low + half_width, strength=1.0))
        return zones[-12:]
