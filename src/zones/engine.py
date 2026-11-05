from __future__ import annotations

from typing import Any

import pandas as pd

from src.zones.detector import ZoneDetector
from src.zones.manager import ZoneManager


class ZoneSignalEngine:
    def __init__(self, symbol: str, config: dict[str, Any] | None = None, database=None):
        self.symbol = symbol
        self.config = config or {}
        self.database = database
        self.detector = ZoneDetector(
            lookback=int(self.config.get("lookback", 80)),
            pivot_window=int(self.config.get("pivot_window", 4)),
            zone_half_atr=float(self.config.get("zone_half_atr", 0.5)),
        )
        self.manager = ZoneManager(merge_overlap=bool(self.config.get("merge_overlap", True)))

    def analyze(self, df: pd.DataFrame) -> dict[str, Any] | None:
        if df.empty or len(df) < int(self.config.get("minimum_bars", 80)):
            return None
        detected = self.detector.detect(df)
        for zone in detected:
            self.manager.add_zone(zone)
            if self.database and self.database.connected:
                self.database.persist_zone(self.symbol, zone)
        self.manager.maintenance()

        row = df.iloc[-1]
        close = float(row["close"])
        atr = float(row.get("atr", 0.0) or 0.0)
        if atr <= 0:
            return None

        nearby = self.manager.get_nearby_zones(close, atr, max_distance_atr=float(self.config.get("max_distance_atr", 1.0)))
        if not nearby:
            return None

        nearest = nearby[0]
        mid = (float(nearest.price_low) + float(nearest.price_high)) / 2.0
        direction_bias = "Buy" if nearest.zone_type == "demand" else "Sell"
        sl_ticks = int(self.config.get("sl_ticks", 40))
        tp_ticks = int(self.config.get("tp_ticks", 80))

        trend_up = bool(row.get("ema_50", close) >= row.get("ema_200", close))
        if direction_bias == "Buy" and not trend_up:
            return None
        if direction_bias == "Sell" and trend_up:
            return None

        distance = abs(close - mid)
        if distance > atr * float(self.config.get("entry_distance_atr", 0.35)):
            return None

        return {
            "symbol": self.symbol,
            "side": direction_bias,
            "quantity": int(self.config.get("quantity", 1)),
            "sl_ticks": sl_ticks,
            "tp_ticks": tp_ticks,
            "strategy_name": "zone_confluence",
            "source": "zone_engine",
            "metadata": {
                "zone_type": nearest.zone_type,
                "zone_low": float(nearest.price_low),
                "zone_high": float(nearest.price_high),
                "zone_strength": float(getattr(nearest, "strength", 0.0)),
            },
        }
