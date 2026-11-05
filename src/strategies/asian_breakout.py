from __future__ import annotations

from datetime import timedelta
from typing import Any

import pandas as pd

from .base import BaseStrategy, StrategySignal


class AsianBreakoutStrategy(BaseStrategy):
    name = "asian_breakout"
    warmup_bars = 20

    def __init__(self, symbol: str, params: dict[str, Any] | None = None):
        super().__init__(symbol, params)
        self.tick_size = float(self.params.get("tick_size", 0.25))
        self.quantity = int(self.params.get("quantity", 1))
        self.base_sl_ticks = int(self.params.get("sl_ticks", 40))
        self.base_tp_ticks = int(self.params.get("tp_ticks", 80))
        self.min_range_ticks = int(self.params.get("min_range_ticks", max(8, self.base_sl_ticks // 2)))
        self.max_range_ticks = int(self.params.get("max_range_ticks", max(self.base_sl_ticks * 4, self.base_tp_ticks * 2)))
        self.rr_ratio = float(self.params.get("rr_ratio", 1.5))
        self.breakout_buffer_ticks = int(self.params.get("breakout_buffer_ticks", 2))
        self.asian_start_hour = int(self.params.get("asian_start_hour_utc", 21))
        self.asian_end_hour = int(self.params.get("asian_end_hour_utc", 6))
        self.london_start_hour = int(self.params.get("london_start_hour_utc", 6))
        self.london_end_hour = int(self.params.get("london_end_hour_utc", 10))
        self.reset()

    def reset(self) -> None:
        self._session_key: str | None = None
        self._asian_high: float | None = None
        self._asian_low: float | None = None
        self._range_locked = False
        self._fired_session: str | None = None

    def get_description(self) -> str:
        return (
            "Session breakout: build the Asian range, then trade the London breakout once per session "
            f"when range is between {self.min_range_ticks} and {self.max_range_ticks} ticks."
        )

    def _timestamp(self, value):
        ts = pd.to_datetime(value)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return ts

    def _session_for_timestamp(self, ts):
        if ts.hour >= self.asian_start_hour:
            return (ts + timedelta(days=1)).date().isoformat()
        return ts.date().isoformat()

    def _in_asian_window(self, ts):
        return ts.hour >= self.asian_start_hour or ts.hour < self.asian_end_hour

    def _in_london_window(self, ts):
        return self.london_start_hour <= ts.hour < self.london_end_hour

    def _range_ticks(self):
        if self._asian_high is None or self._asian_low is None:
            return None
        return int(round((self._asian_high - self._asian_low) / self.tick_size))

    def get_snapshot(self, index, bars, indicators) -> dict[str, Any]:
        row = bars.iloc[index]
        ts = self._timestamp(row["timestamp"]) if "timestamp" in row else None
        return {
            "status": "Active" if index + 1 >= self.warmup_bars else f"Warming up ({index + 1}/{self.warmup_bars})",
            "strategy": self.name,
            "reasoning": (
                f"Asian range locked: {self._range_locked}, fired: {self._fired_session == self._session_key}, "
                f"range_ticks: {self._range_ticks()}"
            ),
            "session": self._session_key,
            "timestamp": ts.isoformat() if ts is not None else None,
            "asian_high": self._asian_high,
            "asian_low": self._asian_low,
            "range_ticks": self._range_ticks(),
        }

    def on_bar(self, index, bars, indicators):
        row = bars.iloc[index]
        ts = self._timestamp(row["timestamp"]) if "timestamp" in row else None
        if ts is None:
            return None

        session_key = self._session_for_timestamp(ts)
        in_asian = self._in_asian_window(ts)

        if session_key != self._session_key and in_asian:
            self._session_key = session_key
            self._asian_high = None
            self._asian_low = None
            self._range_locked = False
            self._fired_session = None

        if in_asian and not self._range_locked:
            high = float(row["high"])
            low = float(row["low"])
            self._asian_high = high if self._asian_high is None else max(self._asian_high, high)
            self._asian_low = low if self._asian_low is None else min(self._asian_low, low)
            return None

        if not in_asian and not self._range_locked and self._asian_high is not None and self._asian_low is not None:
            self._range_locked = True

        if not self._range_locked or self._fired_session == session_key or not self._in_london_window(ts):
            return None

        range_ticks = self._range_ticks()
        if range_ticks is None or range_ticks < self.min_range_ticks or range_ticks > self.max_range_ticks:
            return None

        close = float(row["close"])
        breakout_buffer = self.breakout_buffer_ticks * self.tick_size
        side = None
        reason = ""

        if close > self._asian_high + breakout_buffer:
            side = "Buy"
            reason = "London breakout above Asian high"
        elif close < self._asian_low - breakout_buffer:
            side = "Sell"
            reason = "London breakout below Asian low"

        if side is None:
            return None

        if not self.is_priming:
            self._fired_session = session_key
        sl_ticks = min(max(1, range_ticks), self.base_sl_ticks)
        tp_ticks = max(int(round(sl_ticks * self.rr_ratio)), 1, self.base_tp_ticks // 2)
        confidence = min(0.9, 0.45 + (range_ticks / max(self.max_range_ticks, 1)) * 0.3)

        return StrategySignal(
            strategy_name=self.name,
            side=side,
            quantity=self.quantity,
            sl_ticks=sl_ticks,
            tp_ticks=tp_ticks,
            confidence=confidence,
            reason=reason,
            metadata={
                "asian_high": self._asian_high,
                "asian_low": self._asian_low,
                "range_ticks": range_ticks,
                "session": session_key,
            },
        )
