from __future__ import annotations

from typing import Any

import pandas as pd

from .base import BaseStrategy, StrategySignal


class HTFLevelRetestStrategy(BaseStrategy):
    name = "htf_level_retest"
    warmup_bars = 120

    def __init__(self, symbol: str, params: dict[str, Any] | None = None):
        super().__init__(symbol, params)
        self.params = self._apply_symbol_overrides(self.params)
        self.tick_size = float(self.params.get("tick_size", 0.25))
        self.quantity = int(self.params.get("quantity", 1))
        self.htf_minutes = int(self.params.get("htf_minutes", 15))
        self.level_lookback = int(self.params.get("level_lookback", 16))
        self.mode = str(self.params.get("mode", "break_retest")).strip().lower()
        self.break_buffer_ticks = int(self.params.get("break_buffer_ticks", 1))
        self.retest_tolerance_ticks = int(self.params.get("retest_tolerance_ticks", 4))
        self.reclaim_ticks = int(self.params.get("reclaim_ticks", 1))
        self.sl_ticks = int(self.params.get("sl_ticks", 40))
        self.tp_ticks = int(self.params.get("tp_ticks", 80))
        self.min_signal_gap_bars = int(self.params.get("min_signal_gap_bars", 5))
        self.allowed_entry_hours_et = [int(hour) for hour in self.params.get("allowed_entry_hours_et", []) or []]
        self.allowed_sides = {str(side).strip().lower() for side in self.params.get("allowed_sides", []) or []}
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
        self._htf_cache: dict[pd.Timestamp, pd.DataFrame] = {}

    def get_description(self) -> str:
        return "Completed higher-timeframe level break/retest or gap-fill entries on 1m bars."

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
        htf = self._completed_htf(bars.iloc[: index + 1])
        if len(htf) < self.level_lookback + 2:
            return None
        if self.mode == "gap_fill":
            return self._gap_fill_signal(index, row, htf)
        return self._break_retest_signal(index, bars, row, htf)

    def _completed_htf(self, bars: pd.DataFrame) -> pd.DataFrame:
        current_ts = pd.to_datetime(bars.iloc[-1]["timestamp"], utc=True)
        rule = f"{self.htf_minutes}min"
        current_floor = current_ts.floor(rule)
        completed_cutoff = current_floor - pd.Timedelta(minutes=self.htf_minutes) if current_ts == current_floor else current_floor
        if completed_cutoff in self._htf_cache:
            return self._htf_cache[completed_cutoff].copy()
        frame = bars.assign(timestamp=pd.to_datetime(bars["timestamp"], utc=True)).set_index("timestamp")
        htf = frame.resample(rule, label="right", closed="right").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        )
        htf = htf.dropna().reset_index()
        completed = htf[htf["timestamp"] <= completed_cutoff].copy()
        self._htf_cache[completed_cutoff] = completed
        return completed.copy()

    def _break_retest_signal(self, index, bars, row, htf):
        recent = htf.iloc[-self.level_lookback - 1 : -1]
        level_high = float(recent["high"].max())
        level_low = float(recent["low"].min())
        close = float(row.get("close", row.get("price", 0.0)))
        high = float(row.get("high", close))
        low = float(row.get("low", close))
        buffer = self.break_buffer_ticks * self.tick_size
        tolerance = self.retest_tolerance_ticks * self.tick_size
        reclaim = self.reclaim_ticks * self.tick_size
        prior_close = float(bars.iloc[index - 1]["close"])

        if (
            self._side_allowed("Buy")
            and prior_close > level_high + buffer
            and low <= level_high + tolerance
            and close >= level_high + reclaim
        ):
            return self._signal("Buy", "15m high break retest", index, level_high, level_low, level_high + buffer)
        if (
            self._side_allowed("Sell")
            and prior_close < level_low - buffer
            and high >= level_low - tolerance
            and close <= level_low - reclaim
        ):
            return self._signal("Sell", "15m low break retest", index, level_high, level_low, level_low - buffer)
        self._last_reject_reason = "no htf break retest"
        return None

    def _gap_fill_signal(self, index, row, htf):
        prev = htf.iloc[-2]
        last = htf.iloc[-1]
        gap_up = float(last["low"]) > float(prev["high"])
        gap_down = float(last["high"]) < float(prev["low"])
        close = float(row.get("close", row.get("price", 0.0)))
        high = float(row.get("high", close))
        low = float(row.get("low", close))
        tolerance = self.retest_tolerance_ticks * self.tick_size
        if self._side_allowed("Buy") and gap_up and low <= float(prev["high"]) + tolerance and close > float(prev["high"]):
            return self._signal("Buy", "15m gap fill support reclaim", index, float(last["high"]), float(prev["high"]), float(prev["high"]))
        if self._side_allowed("Sell") and gap_down and high >= float(prev["low"]) - tolerance and close < float(prev["low"]):
            return self._signal("Sell", "15m gap fill resistance reject", index, float(prev["low"]), float(last["low"]), float(prev["low"]))
        self._last_reject_reason = "no gap fill reclaim"
        return None

    def _signal(self, side: str, reason: str, index: int, level_high: float, level_low: float, trigger: float):
        self._last_signal_index = index
        self._last_reject_reason = None
        return StrategySignal(
            strategy_name=self.name,
            side=side,
            quantity=self.quantity,
            sl_ticks=self.sl_ticks,
            tp_ticks=self.tp_ticks,
            confidence=0.6,
            reason=reason,
            metadata={
                "entry_order_type": "stop_market",
                "entry_trigger_price": float(trigger),
                "trigger_source": self.mode,
                "htf_minutes": self.htf_minutes,
                "level_high": level_high,
                "level_low": level_low,
                "active_from_index": index + 1,
                "signal_index": index,
                "execution_model": "resting_stop_after_completed_htf_level",
            },
        )
