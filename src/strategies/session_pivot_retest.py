from __future__ import annotations

from datetime import time
from typing import Any

import pandas as pd

from .base import BaseStrategy, StrategySignal


class SessionPivotRetestStrategy(BaseStrategy):
    name = "session_pivot_retest"
    warmup_bars = 300

    def __init__(self, symbol: str, params: dict[str, Any] | None = None):
        super().__init__(symbol, params)
        self.params = self._apply_symbol_overrides(self.params)
        self.tick_size = float(self.params.get("tick_size", 0.25))
        self.quantity = int(self.params.get("quantity", 1))
        self.level_source = str(self.params.get("level_source", "previous_rth")).strip().lower()
        self.mode = str(self.params.get("mode", "retest")).strip().lower()
        self.break_buffer_ticks = int(self.params.get("break_buffer_ticks", 1))
        self.retest_tolerance_ticks = int(self.params.get("retest_tolerance_ticks", 4))
        self.reclaim_ticks = int(self.params.get("reclaim_ticks", 1))
        self.sl_ticks = int(self.params.get("sl_ticks", 40))
        self.tp_ticks = int(self.params.get("tp_ticks", 80))
        self.min_signal_gap_bars = int(self.params.get("min_signal_gap_bars", 8))
        self.allowed_entry_hours_et = [int(hour) for hour in self.params.get("allowed_entry_hours_et", []) or []]
        self.allowed_sides = {str(side).strip().lower() for side in self.params.get("allowed_sides", []) or []}
        self.enabled_level_names = {str(name).strip().lower() for name in self.params.get("enabled_level_names", []) or []}
        self.enabled_symbols = {str(item).upper() for item in self.params.get("enabled_symbols", []) or []}
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
        self._level_cache: dict[tuple[str, object], dict[str, float]] = {}

    def get_description(self) -> str:
        return "Previous-session pivot/high-low retest and reclaim strategy using only completed session levels."

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

    def _symbol_allowed(self) -> bool:
        if not self.enabled_symbols:
            return True
        base = self._base_symbol(self.symbol.upper())
        return self.symbol.upper() in self.enabled_symbols or base in self.enabled_symbols

    def _side_allowed(self, side: str) -> bool:
        return not self.allowed_sides or side.strip().lower() in self.allowed_sides

    def get_snapshot(self, index, bars, indicators) -> dict[str, Any]:
        return {
            "status": "Active" if index >= self.warmup_bars else f"Warming up ({index}/{self.warmup_bars})",
            "strategy": self.name,
            "level_source": self.level_source,
            "mode": self.mode,
            "last_signal_index": self._last_signal_index,
            "last_reject_reason": self._last_reject_reason,
        }

    def on_bar(self, index, bars, indicators):
        if index < self.warmup_bars:
            return None
        if not self._symbol_allowed():
            return None
        if self._last_signal_index is not None and index - self._last_signal_index < self.min_signal_gap_bars:
            return None
        row = bars.iloc[index]
        if not self._hour_allowed(row):
            self._last_reject_reason = "hour outside allowed set"
            return None
        levels = self._levels(bars.iloc[: index + 1])
        if not levels:
            self._last_reject_reason = "levels unavailable"
            return None
        if self.mode == "reclaim":
            return self._reclaim_signal(index, bars, row, levels)
        return self._retest_signal(index, bars, row, levels)

    def _levels(self, bars: pd.DataFrame) -> dict[str, float]:
        if bars.empty:
            return {}
            
        last_ts = bars.iloc[-1]["timestamp"]
        current_et = pd.to_datetime(last_ts, utc=True).tz_convert("America/New_York")
        current_date = current_et.date()
        cache_key = (self.level_source, current_date)
        
        if cache_key in self._level_cache:
            return dict(self._level_cache[cache_key])
            
        # Find prev_date by scanning backwards
        prev_date = None
        for i in range(len(bars) - 1, -1, -1):
            t = pd.to_datetime(bars.iloc[i]["timestamp"], utc=True).tz_convert("America/New_York")
            d = t.date()
            if d < current_date:
                prev_date = d
                break
                
        if not prev_date:
            self._level_cache[cache_key] = {}
            return {}
            
        # Timezone convert only the tail slice (last 2 days, max 3000 bars)
        slice_df = bars.tail(3000).copy()
        slice_df = slice_df.assign(
            timestamp=pd.to_datetime(slice_df["timestamp"], utc=True).dt.tz_convert("America/New_York")
        )
        
        if self.level_source == "overnight":
            overnight = slice_df[
                ((slice_df["timestamp"].dt.date == prev_date) & (slice_df["timestamp"].dt.time >= time(18, 0)))
                | ((slice_df["timestamp"].dt.date == current_date) & (slice_df["timestamp"].dt.time < time(9, 30)))
            ]
            levels = self._high_low_levels(overnight, "overnight")
            self._level_cache[cache_key] = dict(levels)
            return levels
            
        rth = slice_df[
            (slice_df["timestamp"].dt.date == prev_date)
            & (slice_df["timestamp"].dt.time >= time(9, 30))
            & (slice_df["timestamp"].dt.time < time(16, 0))
        ]
        levels = self._high_low_levels(rth, "prev_rth")
        if not levels:
            self._level_cache[cache_key] = {}
            return {}
            
        if self.level_source == "floor_pivots":
            high = levels["high"]
            low = levels["low"]
            close = float(rth.iloc[-1]["close"])
            pivot = (high + low + close) / 3.0
            levels.update({"pivot": pivot, "r1": 2 * pivot - low, "s1": 2 * pivot - high})
            
        self._level_cache[cache_key] = dict(levels)
        return levels

    def _high_low_levels(self, frame: pd.DataFrame, prefix: str) -> dict[str, float]:
        if frame.empty:
            return {}
        return {
            "high": float(frame["high"].astype(float).max()),
            "low": float(frame["low"].astype(float).min()),
            "source": prefix,
        }

    def _candidate_levels(self, levels: dict[str, float]) -> list[tuple[str, float]]:
        if self.level_source == "floor_pivots":
            candidates = [("r1", levels["r1"]), ("pivot", levels["pivot"]), ("s1", levels["s1"])]
        else:
            candidates = [("high", levels["high"]), ("low", levels["low"])]
        if self.enabled_level_names:
            return [(name, level) for name, level in candidates if name.lower() in self.enabled_level_names]
        return candidates

    def _retest_signal(self, index, bars, row, levels):
        close = float(row.get("close", row.get("price", 0.0)))
        high = float(row.get("high", close))
        low = float(row.get("low", close))
        prior_close = float(bars.iloc[index - 1]["close"])
        buffer = self.break_buffer_ticks * self.tick_size
        tolerance = self.retest_tolerance_ticks * self.tick_size
        reclaim = self.reclaim_ticks * self.tick_size
        for name, level in self._candidate_levels(levels):
            if (
                self._side_allowed("Buy")
                and prior_close > level + buffer
                and low <= level + tolerance
                and close >= level + reclaim
            ):
                return self._signal("Buy", f"retest {name} support", index, level, level + buffer, levels)
            if (
                self._side_allowed("Sell")
                and prior_close < level - buffer
                and high >= level - tolerance
                and close <= level - reclaim
            ):
                return self._signal("Sell", f"retest {name} resistance", index, level, level - buffer, levels)
        self._last_reject_reason = "no session level retest"
        return None

    def _reclaim_signal(self, index, bars, row, levels):
        close = float(row.get("close", row.get("price", 0.0)))
        high = float(row.get("high", close))
        low = float(row.get("low", close))
        buffer = self.break_buffer_ticks * self.tick_size
        reclaim = self.reclaim_ticks * self.tick_size
        for name, level in self._candidate_levels(levels):
            if self._side_allowed("Buy") and low < level - buffer and close > level + reclaim:
                return self._signal("Buy", f"reclaim below {name}", index, level, level + buffer, levels)
            if self._side_allowed("Sell") and high > level + buffer and close < level - reclaim:
                return self._signal("Sell", f"reject above {name}", index, level, level - buffer, levels)
        self._last_reject_reason = "no session level reclaim"
        return None

    def _signal(self, side: str, reason: str, index: int, level: float, trigger: float, levels: dict[str, float]):
        self._last_signal_index = index
        self._last_reject_reason = None
        return StrategySignal(
            strategy_name=self.name,
            side=side,
            quantity=self.quantity,
            sl_ticks=self.sl_ticks,
            tp_ticks=self.tp_ticks,
            confidence=0.62,
            reason=reason,
            metadata={
                "entry_order_type": "stop_market",
                "entry_trigger_price": float(trigger),
                "trigger_source": self.level_source,
                "level": float(level),
                "levels": {key: value for key, value in levels.items() if isinstance(value, (int, float))},
                "active_from_index": index + 1,
                "signal_index": index,
                "execution_model": "resting_stop_after_completed_session_level",
            },
        )
