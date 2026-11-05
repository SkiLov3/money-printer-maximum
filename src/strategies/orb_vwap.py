from __future__ import annotations

import zoneinfo
from datetime import time, timedelta, timezone
from typing import Any

import pandas as pd

from .base import BaseStrategy, StrategySignal


class ORBVWAPStrategy(BaseStrategy):
    name = "orb_vwap"
    warmup_bars = 0

    def __init__(self, symbol: str, params: dict[str, Any] | None = None):
        super().__init__(symbol, params)
        self.params = self._apply_symbol_overrides(self.params)
        self.warmup_bars = int(self.params.get("warmup_bars", 0))
        self.tick_size = float(self.params.get("tick_size", 0.25))
        self.quantity = int(self.params.get("quantity", 1))
        self.base_sl_ticks = int(self.params.get("sl_ticks", 40))
        self.base_tp_ticks = int(self.params.get("tp_ticks", 80))

        # ── NY Session Parameters ──
        self.opening_range_enabled = bool(self.params.get("opening_range_enabled", True))
        self.opening_range_start = self._parse_et_time(self.params.get("opening_range_start_et", "09:30"))
        self.opening_range_end = self._parse_et_time(self.params.get("opening_range_end_et", "09:45"))
        self.trade_window_end = self._parse_et_time(self.params.get("trade_window_end_et", "11:00"))
        self.direction_mode = str(self.params.get("direction_mode", "breakout")).strip().lower()
        if self.direction_mode not in {"breakout", "fade", "both"}:
            self.direction_mode = "breakout"
        self.use_vwap_filter = bool(self.params.get("use_vwap_filter", True))
        self.use_ema200_filter = bool(self.params.get("use_ema200_filter", False))
        self.min_or_ticks = int(self.params.get("min_or_ticks", 8))
        self.max_or_ticks = int(self.params.get("max_or_ticks", 120))
        self.volume_lookback_bars = int(self.params.get("volume_lookback_bars", 20))
        self.min_volume_ratio = float(self.params.get("min_volume_ratio", 1.0))
        self.breakout_buffer_ticks = int(self.params.get("breakout_buffer_ticks", 1))
        self.fade_reclaim_ticks = int(self.params.get("fade_reclaim_ticks", 0))
        self.stop_buffer_ticks = int(self.params.get("stop_buffer_ticks", self.breakout_buffer_ticks))
        self.target_mode = str(self.params.get("target_mode", "risk_multiple")).strip().lower()
        self.risk_reward = float(self.params.get("risk_reward", self.params.get("target_r_multiple", 2.0)))
        self.fixed_target_ticks = int(self.params.get("fixed_target_ticks", self.params.get("target_ticks", self.base_tp_ticks)))
        self.min_target_ticks = int(self.params.get("min_target_ticks", 1))
        self.max_target_ticks = int(self.params.get("max_target_ticks", 0) or 0)
        self.max_one_trade_per_session = bool(self.params.get("max_one_trade_per_session", True))
        self.holiday_standdown_dates = set(self.params.get("holiday_standdown_dates", []))

        # ── Asian Session Parameters ──
        self.asian_enabled = bool(self.params.get("asian_enabled", False))
        self.asian_or_start = self._parse_et_time(self.params.get("asian_opening_range_start_et", "18:00"))
        self.asian_or_end = self._parse_et_time(self.params.get("asian_opening_range_end_et", "18:10"))
        self.asian_trade_end = self._parse_et_time(self.params.get("asian_trade_window_end_et", "23:00"))
        asian_dir = str(self.params.get("asian_direction_mode", self.direction_mode)).strip().lower()
        self.asian_direction_mode = asian_dir if asian_dir in {"breakout", "fade", "both"} else self.direction_mode
        self.asian_risk_reward = float(self.params.get("asian_risk_reward", self.risk_reward))
        self.asian_min_or_ticks = int(self.params.get("asian_min_or_ticks", self.min_or_ticks))
        self.asian_max_or_ticks = int(self.params.get("asian_max_or_ticks", self.max_or_ticks))
        self.asian_min_volume_ratio = float(self.params.get("asian_min_volume_ratio", self.min_volume_ratio))
        self.asian_stop_buffer_ticks = int(self.params.get("asian_stop_buffer_ticks", self.stop_buffer_ticks))
        self.asian_breakout_buffer_ticks = int(self.params.get("asian_breakout_buffer_ticks", self.breakout_buffer_ticks))
        self.asian_fade_reclaim_ticks = int(self.params.get("asian_fade_reclaim_ticks", self.fade_reclaim_ticks))
        self.asian_use_vwap_filter = bool(self.params.get("asian_use_vwap_filter", False))
        self.asian_use_ema200_filter = bool(self.params.get("asian_use_ema200_filter", False))

        self.reset()

    def _apply_symbol_overrides(self, params: dict[str, Any]) -> dict[str, Any]:
        merged = dict(params or {})
        overrides = merged.get("symbol_params", {}) or {}
        if not isinstance(overrides, dict):
            return merged
        symbol_upper = self.symbol.upper()
        keys = [symbol_upper, self._base_symbol(symbol_upper)]
        for key in keys:
            override = overrides.get(key)
            if isinstance(override, dict):
                merged.update(override)
        return merged

    def _parse_et_time(self, value: Any) -> time:
        if isinstance(value, time):
            return value
        parts = str(value).strip().split(":")
        if len(parts) < 2:
            raise ValueError(f"Expected HH:MM ET time, got {value!r}")
        return time(hour=int(parts[0]), minute=int(parts[1]))

    def reset(self) -> None:
        # NY state
        self._session_key: str | None = None
        self._or_high: float | None = None
        self._or_low: float | None = None
        self._range_locked = False
        self._fired_session: str | None = None
        self._last_reject_reason: str | None = None
        # Asian state
        self._asian_session_key: str | None = None
        self._asian_or_high: float | None = None
        self._asian_or_low: float | None = None
        self._asian_range_locked = False
        self._asian_fired_session: str | None = None
        self._asian_last_reject_reason: str | None = None

    def get_description(self) -> str:
        desc = (
            "Opening Range Breakout/VWAP: build the ET opening range, then take one "
            f"{self.direction_mode} setup before {self.trade_window_end.strftime('%H:%M')} ET "
            "when range, VWAP/EMA, and volume filters align."
        )
        if self.asian_enabled:
            desc += (
                f" Asian session: {self.asian_direction_mode} mode, "
                f"{self.asian_or_start.strftime('%H:%M')}-{self.asian_trade_end.strftime('%H:%M')} ET."
            )
        return desc

    # ── Timestamp Helpers ──

    def _timestamp_et(self, value) -> pd.Timestamp | None:
        if value is None or pd.isna(value):
            return None
        try:
            val = None
            try:
                val = float(value)
            except (ValueError, TypeError):
                pass
            if val is not None:
                if val > 1e16:
                    ts = pd.to_datetime(val, unit='ns', utc=True)
                elif val > 1e13:
                    ts = pd.to_datetime(val, unit='us', utc=True)
                elif val > 1e10:
                    ts = pd.to_datetime(val, unit='ms', utc=True)
                elif val > 1e8:
                    ts = pd.to_datetime(val, unit='s', utc=True)
                else:
                    return None
            else:
                ts = pd.to_datetime(value)
                if ts is None or pd.isna(ts):
                    return None
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                else:
                    ts = ts.tz_convert("UTC")
            return ts.tz_convert("America/New_York")
        except (TypeError, ValueError, OverflowError):
            return None

    def _session_for_timestamp(self, et: pd.Timestamp) -> str:
        return et.date().isoformat()

    def _asian_session_for_timestamp(self, et: pd.Timestamp) -> str:
        """Asian session key that groups evening + early morning into one session.
        Bars at 18:00+ ET use current date; bars before 06:00 ET use previous date."""
        if et.hour < 6:
            return (et.date() - timedelta(days=1)).isoformat() + "_asian"
        return et.date().isoformat() + "_asian"

    def _get_session_regime(self, row) -> str:
        """Determine the current session regime from a bar's timestamp."""
        raw_ts = row.get("timestamp")
        if raw_ts is None or pd.isna(raw_ts):
            return "OUT"
        try:
            val = None
            try:
                val = float(raw_ts)
            except (ValueError, TypeError):
                pass
            if val is not None:
                if val > 1e16:
                    ts = pd.to_datetime(val, unit='ns', utc=True)
                elif val > 1e13:
                    ts = pd.to_datetime(val, unit='us', utc=True)
                elif val > 1e10:
                    ts = pd.to_datetime(val, unit='ms', utc=True)
                elif val > 1e8:
                    ts = pd.to_datetime(val, unit='s', utc=True)
                else:
                    return "OUT"
            else:
                ts = pd.to_datetime(raw_ts)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            et = ts.astimezone(zoneinfo.ZoneInfo("America/New_York"))
            if et.year < 2000:
                return "OUT"
            mins = et.hour * 60 + et.minute
            # NY Window: 08:00 - 15:30 ET
            if 480 <= mins < 930:
                return "NY"
            # Asian Window: dynamic check based on strategy configuration
            if self.asian_enabled:
                current_time = et.time()
                if self.asian_or_start <= self.asian_trade_end:
                    if self.asian_or_start <= current_time < self.asian_trade_end:
                        return "ASIAN"
                else:
                    if current_time >= self.asian_or_start or current_time < self.asian_trade_end:
                        return "ASIAN"
            return "OUT"
        except (TypeError, ValueError):
            return "OUT"

    # ── NY Session Time Checks ──

    def _in_opening_range(self, et: pd.Timestamp) -> bool:
        current = et.time()
        return self.opening_range_start <= current < self.opening_range_end

    def _in_trade_window(self, et: pd.Timestamp) -> bool:
        current = et.time()
        return self.opening_range_end <= current < self.trade_window_end

    # ── Asian Session Time Checks ──

    def _in_asian_opening_range(self, et: pd.Timestamp) -> bool:
        current = et.time()
        if self.asian_or_start <= self.asian_or_end:
            return self.asian_or_start <= current < self.asian_or_end
        return current >= self.asian_or_start or current < self.asian_or_end

    def _in_asian_trade_window(self, et: pd.Timestamp) -> bool:
        current = et.time()
        if self.asian_or_end <= self.asian_trade_end:
            return self.asian_or_end <= current < self.asian_trade_end
        return current >= self.asian_or_end or current < self.asian_trade_end

    # ── Range Helpers ──

    def _range_ticks(self) -> int | None:
        if self._or_high is None or self._or_low is None or self.tick_size <= 0:
            return None
        return int(round((self._or_high - self._or_low) / self.tick_size))

    def _asian_range_ticks(self) -> int | None:
        if self._asian_or_high is None or self._asian_or_low is None or self.tick_size <= 0:
            return None
        return int(round((self._asian_or_high - self._asian_or_low) / self.tick_size))

    def _volume_ratio(self, index: int, bars: pd.DataFrame) -> float | None:
        if self.volume_lookback_bars <= 0 or "volume" not in bars.columns:
            return None
        start = max(0, index - self.volume_lookback_bars)
        history = bars.iloc[start:index]["volume"].astype(float)
        if history.empty:
            return None
        avg_volume = float(history.mean())
        if avg_volume <= 0:
            return None
        return float(bars.iloc[index].get("volume", 0.0)) / avg_volume

    def _passes_bias_filters(self, side: str, row, indicators_row,
                              use_vwap: bool = True, use_ema200: bool = False) -> tuple[bool, str | None]:
        close = float(row.get("close", row.get("price", 0.0)))
        if use_vwap:
            vwap = indicators_row.get("vwap")
            if vwap is not None and not pd.isna(vwap):
                if side == "Buy" and close < float(vwap):
                    return False, "buy below VWAP"
                if side == "Sell" and close > float(vwap):
                    return False, "sell above VWAP"
        if use_ema200:
            ema200 = indicators_row.get("ema_200")
            if ema200 is not None and not pd.isna(ema200):
                if side == "Buy" and close < float(ema200):
                    return False, "buy below EMA200"
                if side == "Sell" and close > float(ema200):
                    return False, "sell above EMA200"
        return True, None

    def _target_ticks(self, sl_ticks: int, rr_override: float | None = None) -> int:
        rr = rr_override if rr_override is not None else self.risk_reward
        if self.target_mode in {"fixed", "fixed_ticks"}:
            target = self.fixed_target_ticks
        else:
            target = int(round(sl_ticks * rr))
        target = max(self.min_target_ticks, target)
        if self.max_target_ticks > 0:
            target = min(self.max_target_ticks, target)
        return max(1, target)

    # ── Candidate Detection ──

    def _candidate(self, row, or_high: float, or_low: float,
                   direction_mode: str, breakout_buffer_ticks: int,
                   fade_reclaim_ticks: int) -> tuple[str | None, str | None, str | None]:
        """Evaluate whether the current bar triggers a trade candidate."""
        close = float(row.get("close", row.get("price", 0.0)))
        high = float(row.get("high", close))
        low = float(row.get("low", close))
        breakout_buffer = breakout_buffer_ticks * self.tick_size
        fade_reclaim = fade_reclaim_ticks * self.tick_size

        if direction_mode in {"breakout", "both"}:
            if close > or_high + breakout_buffer:
                return "Buy", "breakout", "Opening range breakout above high"
            if close < or_low - breakout_buffer:
                return "Sell", "breakout", "Opening range breakout below low"

        if direction_mode in {"fade", "both"}:
            if low < or_low - breakout_buffer and close > or_low + fade_reclaim:
                return "Buy", "fade", "Opening range low reclaim fade"
            if high > or_high + breakout_buffer and close < or_high - fade_reclaim:
                return "Sell", "fade", "Opening range high rejection fade"

        return None, None, None

    # ── Snapshot ──

    def get_snapshot(self, index, bars, indicators) -> dict[str, Any]:
        row = bars.iloc[index]
        et = self._timestamp_et(row.get("timestamp")) if "timestamp" in row else None
        session_regime = self._get_session_regime(row)
        snap: dict[str, Any] = {
            "status": "Active" if index + 1 >= self.warmup_bars else f"Warming up ({index + 1}/{self.warmup_bars})",
            "strategy": self.name,
            "session": self._session_key,
            "session_regime": session_regime,
            "timestamp_et": et.isoformat() if et is not None else None,
            "opening_range_high": self._or_high,
            "opening_range_low": self._or_low,
            "range_locked": self._range_locked,
            "range_ticks": self._range_ticks(),
            "fired_session": self._fired_session,
            "direction_mode": self.direction_mode,
            "opening_range_enabled": self.opening_range_enabled,
            "last_reject_reason": self._last_reject_reason,
        }
        if self.asian_enabled:
            snap.update({
                "asian_enabled": True,
                "asian_session": self._asian_session_key,
                "asian_opening_range_high": self._asian_or_high,
                "asian_opening_range_low": self._asian_or_low,
                "asian_range_locked": self._asian_range_locked,
                "asian_range_ticks": self._asian_range_ticks(),
                "asian_fired_session": self._asian_fired_session,
                "asian_direction_mode": self.asian_direction_mode,
                "asian_last_reject_reason": self._asian_last_reject_reason,
            })
        return snap

    # ── Main Bar Handler ──

    def on_bar(self, index, bars, indicators):
        if not self.enabled:
            return None
        row = bars.iloc[index]
        indicators_row = indicators.iloc[index]
        et = self._timestamp_et(row.get("timestamp")) if "timestamp" in row else None
        if et is None:
            return None

        # ── NY Session ──
        if self.opening_range_enabled:
            ny_signal = self._process_ny(et, index, row, indicators_row, bars)
            if ny_signal is not None:
                return ny_signal

        # ── Asian Session ──
        if self.asian_enabled:
            asian_signal = self._process_asian(et, index, row, indicators_row, bars)
            if asian_signal is not None:
                return asian_signal

        return None

    def _process_ny(self, et, index, row, indicators_row, bars):
        """Process the NY session opening range logic."""
        session_key = self._session_for_timestamp(et)
        if session_key != self._session_key:
            self._session_key = session_key
            self._or_high = None
            self._or_low = None
            self._range_locked = False
            self._fired_session = None
            self._last_reject_reason = None

        if self._in_opening_range(et):
            high = float(row.get("high", row.get("close", 0.0)))
            low = float(row.get("low", row.get("close", 0.0)))
            self._or_high = high if self._or_high is None else max(self._or_high, high)
            self._or_low = low if self._or_low is None else min(self._or_low, low)
            return None

        if not self._range_locked and et.time() >= self.opening_range_end and self._or_high is not None and self._or_low is not None:
            self._range_locked = True

        if not self._range_locked or not self._in_trade_window(et) or index + 1 < self.warmup_bars:
            return None
        if self.max_one_trade_per_session and self._fired_session == session_key:
            return None

        range_ticks = self._range_ticks()
        if range_ticks is None or range_ticks < self.min_or_ticks or range_ticks > self.max_or_ticks:
            self._last_reject_reason = "opening range outside configured tick bounds"
            return None

        if self._or_high is None or self._or_low is None:
            return None

        date_str = et.strftime("%Y-%m-%d")
        is_holiday = date_str in self.holiday_standdown_dates
        dir_mode = "fade" if is_holiday else self.direction_mode

        side, setup_type, reason = self._candidate(
            row, self._or_high, self._or_low,
            dir_mode, self.breakout_buffer_ticks, self.fade_reclaim_ticks,
        )
        if side is not None and is_holiday:
            reason = f"[HOLIDAY FADE] {reason}"
        if side is None:
            return None

        volume_ratio = self._volume_ratio(index, bars)
        if self.min_volume_ratio > 0 and volume_ratio is not None and volume_ratio < self.min_volume_ratio:
            self._last_reject_reason = "volume ratio below threshold"
            return None

        passes_bias, reject_reason = self._passes_bias_filters(
            side, row, indicators_row, self.use_vwap_filter, self.use_ema200_filter,
        )
        if not passes_bias:
            self._last_reject_reason = reject_reason
            return None

        self._fired_session = session_key
        sl_ticks = max(1, range_ticks + self.stop_buffer_ticks)
        tp_ticks = self._target_ticks(sl_ticks)
        confidence = min(0.92, 0.50 + min(0.25, (volume_ratio or 1.0) * 0.08) + (0.08 if self.use_vwap_filter else 0.0))
        self._last_reject_reason = None

        return StrategySignal(
            strategy_name=self.name,
            side=side,
            quantity=self.quantity,
            sl_ticks=sl_ticks,
            tp_ticks=tp_ticks,
            confidence=confidence,
            reason=reason or "Opening range setup",
            metadata={
                "session": session_key,
                "session_window": "NY",
                "setup_type": setup_type,
                "opening_range_high": self._or_high,
                "opening_range_low": self._or_low,
                "range_ticks": range_ticks,
                "volume_ratio": volume_ratio,
                "vwap_filter": self.use_vwap_filter,
                "ema200_filter": self.use_ema200_filter,
                "target_mode": self.target_mode,
                "trade_time_et": et.isoformat(),
                "atr": float(indicators_row.get("atr", 0.0)),
            },
        )

    def _process_asian(self, et, index, row, indicators_row, bars):
        """Process the Asian session opening range logic."""
        asian_key = self._asian_session_for_timestamp(et)
        if asian_key != self._asian_session_key:
            self._asian_session_key = asian_key
            self._asian_or_high = None
            self._asian_or_low = None
            self._asian_range_locked = False
            self._asian_fired_session = None
            self._asian_last_reject_reason = None

        if self._in_asian_opening_range(et):
            high = float(row.get("high", row.get("close", 0.0)))
            low = float(row.get("low", row.get("close", 0.0)))
            self._asian_or_high = high if self._asian_or_high is None else max(self._asian_or_high, high)
            self._asian_or_low = low if self._asian_or_low is None else min(self._asian_or_low, low)
            return None

        current_t = et.time()
        if self.asian_or_start <= self.asian_or_end:
            passed_end = current_t >= self.asian_or_end or (self.asian_trade_end < self.asian_or_end and current_t < self.asian_trade_end)
        else:
            passed_end = current_t >= self.asian_or_end and current_t < self.asian_or_start
        if not self._asian_range_locked and passed_end and self._asian_or_high is not None and self._asian_or_low is not None:
            self._asian_range_locked = True

        if not self._asian_range_locked or not self._in_asian_trade_window(et) or index + 1 < self.warmup_bars:
            return None
        if self.max_one_trade_per_session and self._asian_fired_session == asian_key:
            return None

        range_ticks = self._asian_range_ticks()
        if range_ticks is None or range_ticks < self.asian_min_or_ticks or range_ticks > self.asian_max_or_ticks:
            self._asian_last_reject_reason = "asian opening range outside configured tick bounds"
            return None

        if self._asian_or_high is None or self._asian_or_low is None:
            return None

        date_str = asian_key.removesuffix("_asian")
        is_holiday = date_str in self.holiday_standdown_dates
        asian_dir = "fade" if is_holiday else self.asian_direction_mode

        side, setup_type, reason = self._candidate(
            row, self._asian_or_high, self._asian_or_low,
            asian_dir, self.asian_breakout_buffer_ticks, self.asian_fade_reclaim_ticks,
        )
        if side is not None and is_holiday:
            reason = f"[HOLIDAY FADE] {reason}"
        if side is None:
            return None

        volume_ratio = self._volume_ratio(index, bars)
        if self.asian_min_volume_ratio > 0 and volume_ratio is not None and volume_ratio < self.asian_min_volume_ratio:
            self._asian_last_reject_reason = "asian volume ratio below threshold"
            return None

        passes_bias, reject_reason = self._passes_bias_filters(
            side, row, indicators_row, self.asian_use_vwap_filter, self.asian_use_ema200_filter,
        )
        if not passes_bias:
            self._asian_last_reject_reason = reject_reason
            return None

        self._asian_fired_session = asian_key
        sl_ticks = max(1, range_ticks + self.asian_stop_buffer_ticks)
        tp_ticks = self._target_ticks(sl_ticks, rr_override=self.asian_risk_reward)
        confidence = min(0.88, 0.45 + min(0.20, (volume_ratio or 1.0) * 0.06))
        self._asian_last_reject_reason = None

        return StrategySignal(
            strategy_name=self.name,
            side=side,
            quantity=self.quantity,
            sl_ticks=sl_ticks,
            tp_ticks=tp_ticks,
            confidence=confidence,
            reason=f"[ASIAN] {reason}" if reason else "Asian opening range setup",
            metadata={
                "session": asian_key,
                "session_window": "ASIAN",
                "setup_type": setup_type,
                "opening_range_high": self._asian_or_high,
                "opening_range_low": self._asian_or_low,
                "range_ticks": range_ticks,
                "volume_ratio": volume_ratio,
                "vwap_filter": self.asian_use_vwap_filter,
                "ema200_filter": self.asian_use_ema200_filter,
                "target_mode": self.target_mode,
                "trade_time_et": et.isoformat(),
                "atr": float(indicators_row.get("atr", 0.0)),
            },
        )
