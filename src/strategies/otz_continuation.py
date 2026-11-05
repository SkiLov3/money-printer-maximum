from __future__ import annotations

from typing import Any

import pandas as pd

from .base import BaseStrategy, StrategySignal


class OTZContinuationStrategy(BaseStrategy):
    name = "otz_continuation"
    warmup_bars = 260

    def __init__(self, symbol: str, params: dict[str, Any] | None = None):
        super().__init__(symbol, params)
        self.params = self._apply_symbol_overrides(self.params)
        self.tick_size = float(self.params.get("tick_size", 0.25))
        self.quantity = int(self.params.get("quantity", 1))
        self.htf_minutes = int(self.params.get("htf_minutes", 15))
        self.htf_lookback = int(self.params.get("htf_lookback", 24))
        self.ema_fast_period = int(self.params.get("ema_fast_period", 21))
        self.ema_slow_period = int(self.params.get("ema_slow_period", 200))
        self.zone_mode = str(self.params.get("zone_mode", "impulse_body_50")).strip().lower()
        self.min_htf_impulse_atr = float(self.params.get("min_htf_impulse_atr", 1.4))
        self.min_htf_body_pct = float(self.params.get("min_htf_body_pct", 0.58))
        self.engulf_mode = str(self.params.get("engulf_mode", "body")).strip().lower()
        self.ob_wick_area_bars = max(1, int(self.params.get("ob_wick_area_bars", 3)))
        self.ob_wick_area_mode = str(self.params.get("ob_wick_area_mode", "recent_bars")).strip().lower()
        self.ob_wick_cluster_tolerance_ticks = int(self.params.get("ob_wick_cluster_tolerance_ticks", 120))
        self.ob_zone_padding_ticks = int(self.params.get("ob_zone_padding_ticks", 0))
        self.invalidate_on_close_through_otz = bool(self.params.get("invalidate_on_close_through_otz", True))
        self.zone_tolerance_ticks = int(self.params.get("zone_tolerance_ticks", 4))
        self.trigger_zone_mode = str(self.params.get("trigger_zone_mode", "wick_touch_any_close")).strip().lower()
        self.confirmation = str(self.params.get("confirmation", "engulf_or_123")).strip().lower()
        self.structure_123_mode = str(self.params.get("structure_123_mode", "close_break")).strip().lower()
        self.trendline_source = str(self.params.get("trendline_source", "wick")).strip().lower()
        self.trendline_lookback_bars = int(self.params.get("trendline_lookback_bars", 40))
        self.trendline_break_buffer_ticks = int(self.params.get("trendline_break_buffer_ticks", 1))
        self.min_reject_wick_ratio = float(self.params.get("min_reject_wick_ratio", 0.45))
        self.min_confirmation_body_ticks = float(self.params.get("min_confirmation_body_ticks", 4.0))
        self.require_ema_context = bool(self.params.get("require_ema_context", True))
        self.orderflow_mode = str(self.params.get("orderflow_mode", "ema_any")).strip().lower()
        self.htf_structure_lookback = int(self.params.get("htf_structure_lookback", 6))
        self.stop_mode = str(self.params.get("stop_mode", "fixed_ticks")).strip().lower()
        self.stop_padding_ticks = int(self.params.get("stop_padding_ticks", 2))
        self.target_mode = str(self.params.get("target_mode", "fixed_ticks")).strip().lower()
        self.target_r_multiple = float(self.params.get("target_r_multiple", 2.0))
        self.skip_overlapping_otz = bool(self.params.get("skip_overlapping_otz", True))
        self.allowed_entry_hours_et = [int(hour) for hour in self.params.get("allowed_entry_hours_et", []) or []]
        self.enabled_symbols = {str(item).upper() for item in self.params.get("enabled_symbols", []) or []}
        self.allowed_sides = {str(side).strip().lower() for side in self.params.get("allowed_sides", []) or []}
        self.min_signal_gap_bars = int(self.params.get("min_signal_gap_bars", 8))
        self.sl_ticks = int(self.params.get("sl_ticks", 40))
        self.tp_ticks = int(self.params.get("tp_ticks", 80))
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
        return "OTZ continuation: completed HTF order-block wick zone plus closed-candle rejection/engulf/1-2-3 confirmation."

    def get_snapshot(self, index, bars, indicators) -> dict[str, Any]:
        return {
            "status": "Active" if index >= self.warmup_bars else f"Warming up ({index}/{self.warmup_bars})",
            "strategy": self.name,
            "zone_mode": self.zone_mode,
            "confirmation": self.confirmation,
            "last_reject_reason": self._last_reject_reason,
        }

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
        symbol_upper = self.symbol.upper()
        return symbol_upper in self.enabled_symbols or self._base_symbol(symbol_upper) in self.enabled_symbols

    def _side_allowed(self, side: str) -> bool:
        return not self.allowed_sides or side.strip().lower() in self.allowed_sides

    def _completed_htf(self, bars: pd.DataFrame) -> pd.DataFrame:
        current_ts = pd.to_datetime(bars.iloc[-1]["timestamp"], utc=True)
        rule = f"{self.htf_minutes}min"
        current_floor = current_ts.floor(rule)
        completed_cutoff = current_floor - pd.Timedelta(minutes=self.htf_minutes) if current_ts == current_floor else current_floor
        if completed_cutoff in self._htf_cache:
            return self._htf_cache[completed_cutoff].copy()
        frame = bars.copy()
        frame.loc[:, "timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.set_index("timestamp")
        htf = frame.resample(rule, label="right", closed="right").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        )
        htf = htf.dropna().reset_index()
        htf.loc[:, "range"] = htf["high"].astype(float) - htf["low"].astype(float)
        htf.loc[:, "body"] = (htf["close"].astype(float) - htf["open"].astype(float)).abs()
        htf.loc[:, "atr"] = htf["range"].rolling(14).mean()
        completed = htf[htf["timestamp"] <= completed_cutoff].copy()
        self._htf_cache[completed_cutoff] = completed
        return completed.copy()

    def _active_zone(self, htf: pd.DataFrame) -> dict[str, Any] | None:
        if len(htf) < max(16, self.htf_lookback):
            return None
        recent = htf.iloc[-self.htf_lookback :].copy()
        recent = recent[recent["atr"].notna()]
        if recent.empty:
            return None
        if self.zone_mode in {"engulfed_wick_ob", "ob_wicks", "orderblock_wicks"}:
            return self._engulfed_wick_orderblock_zone(recent)
        candidates = recent[
            (recent["range"] >= self.min_htf_impulse_atr * recent["atr"])
            & ((recent["body"] / recent["range"].replace(0, pd.NA)) >= self.min_htf_body_pct)
        ]
        if candidates.empty:
            return None
        impulse = candidates.iloc[-1]
        open_price = float(impulse["open"])
        close_price = float(impulse["close"])
        high = float(impulse["high"])
        low = float(impulse["low"])
        midpoint = (open_price + close_price) / 2.0
        side = "Sell" if close_price < open_price else "Buy"
        if self.zone_mode == "wick":
            zone_low, zone_high = (close_price, high) if side == "Sell" else (low, close_price)
        elif self.zone_mode == "full_body":
            zone_low, zone_high = sorted((open_price, close_price))
        else:
            zone_low, zone_high = sorted((midpoint, open_price))
        return {
            "side": side,
            "zone_low": float(zone_low),
            "zone_high": float(zone_high),
            "impulse_time": str(impulse["timestamp"]),
            "impulse_open": open_price,
            "impulse_close": close_price,
            "impulse_high": high,
                "impulse_low": low,
        }

    def _engulfed_wick_orderblock_zone(self, recent: pd.DataFrame) -> dict[str, Any] | None:
        zones = self._engulfed_wick_orderblock_zones(recent)
        return zones[-1] if zones else None

    def _engulfed_wick_orderblock_zones(self, recent: pd.DataFrame) -> list[dict[str, Any]]:
        zones: list[dict[str, Any]] = []
        padding = self.ob_zone_padding_ticks * self.tick_size
        rows = list(recent.to_dict("records"))
        for idx in range(1, len(rows)):
            engulfed = rows[idx - 1]
            engulfing = rows[idx]
            prior_open = float(engulfed["open"])
            prior_close = float(engulfed["close"])
            prior_high = float(engulfed["high"])
            prior_low = float(engulfed["low"])
            current_open = float(engulfing["open"])
            current_close = float(engulfing["close"])
            current_high = float(engulfing["high"])
            current_low = float(engulfing["low"])
            current_range = max(float(engulfing["range"]), self.tick_size)
            current_body = float(engulfing["body"])
            current_atr = float(engulfing["atr"])
            if current_atr <= 0:
                continue
            displacement_ok = (
                current_range >= self.min_htf_impulse_atr * current_atr
                and current_body / current_range >= self.min_htf_body_pct
            )
            body_bearish_engulf = (
                prior_close > prior_open
                and current_close < current_open
                and current_open >= prior_close
                and current_close <= prior_open
            )
            body_bullish_engulf = (
                prior_close < prior_open
                and current_close > current_open
                and current_open <= prior_close
                and current_close >= prior_open
            )
            full_bearish_engulf = (
                body_bearish_engulf
                and current_high >= prior_high
                and current_low <= prior_low
            )
            full_bullish_engulf = (
                body_bullish_engulf
                and current_high >= prior_high
                and current_low <= prior_low
            )
            if self.engulf_mode == "full":
                bearish_engulf = full_bearish_engulf
                bullish_engulf = full_bullish_engulf
            else:
                bearish_engulf = body_bearish_engulf
                bullish_engulf = body_bullish_engulf
            if not displacement_ok or not (bearish_engulf or bullish_engulf):
                continue
            side = "Sell" if bearish_engulf else "Buy"
            wick_area = self._wick_area_rows(rows, idx, side, prior_open, prior_close)
            wick_high = max(float(item["high"]) for item in wick_area)
            wick_low = min(float(item["low"]) for item in wick_area)
            if side == "Sell":
                zone_low = max(prior_open, prior_close) - padding
                zone_high = wick_high + padding
            else:
                zone_low = wick_low - padding
                zone_high = min(prior_open, prior_close) + padding
            candidate = {
                "side": side,
                "zone_low": float(min(zone_low, zone_high)),
                "zone_high": float(max(zone_low, zone_high)),
                "orderblock_time": str(engulfed["timestamp"]),
                "engulfing_time": str(engulfing["timestamp"]),
                "orderblock_open": prior_open,
                "orderblock_high": prior_high,
                "orderblock_low": prior_low,
                "orderblock_close": prior_close,
                "impulse_open": current_open,
                "impulse_close": current_close,
                "impulse_high": current_high,
                "impulse_low": current_low,
                "engulf_mode": self.engulf_mode,
                "ob_wick_area_mode": self.ob_wick_area_mode,
            }
            if self.skip_overlapping_otz and any(self._zones_overlap(candidate, zone) for zone in zones):
                continue
            zones.append(candidate)
        return zones

    def _zones_overlap(self, first: dict[str, Any], second: dict[str, Any]) -> bool:
        if str(first.get("side")) != str(second.get("side")):
            return False
        return float(first["zone_low"]) <= float(second["zone_high"]) and float(second["zone_low"]) <= float(first["zone_high"])

    def _wick_area_rows(self, rows: list[dict[str, Any]], engulfing_index: int, side: str, prior_open: float, prior_close: float) -> list[dict[str, Any]]:
        if self.ob_wick_area_mode not in {"cluster", "lookback_cluster", "screen_cluster"}:
            return rows[max(0, engulfing_index - self.ob_wick_area_bars) : engulfing_index]
        anchor = max(prior_open, prior_close) if side == "Sell" else min(prior_open, prior_close)
        tolerance = self.ob_wick_cluster_tolerance_ticks * self.tick_size
        candidates = rows[:engulfing_index]
        if side == "Sell":
            clustered = [item for item in candidates if float(item["high"]) >= anchor - tolerance]
        else:
            clustered = [item for item in candidates if float(item["low"]) <= anchor + tolerance]
        return clustered or rows[max(0, engulfing_index - self.ob_wick_area_bars) : engulfing_index]

    def _touches_zone(self, row: pd.Series, zone: dict[str, Any]) -> bool:
        tolerance = self.zone_tolerance_ticks * self.tick_size
        high = float(row.get("high", row.get("close", 0.0)))
        low = float(row.get("low", row.get("close", 0.0)))
        return high >= float(zone["zone_low"]) - tolerance and low <= float(zone["zone_high"]) + tolerance

    def _trigger_location_ok(self, row: pd.Series, zone: dict[str, Any]) -> bool:
        if not self._touches_zone(row, zone):
            return False
        if self.trigger_zone_mode in {"close_inside", "close_in_zone", "trigger_close_inside"}:
            close = float(row.get("close", row.get("price", 0.0)))
            return float(zone["zone_low"]) <= close <= float(zone["zone_high"])
        return True

    def _zone_invalidated_since(self, bars: pd.DataFrame, index: int, zone: dict[str, Any]) -> bool:
        if not self.invalidate_on_close_through_otz:
            return False
        event_time = zone.get("engulfing_time") or zone.get("impulse_time")
        if not event_time:
            return False
        start_time = pd.to_datetime(event_time, utc=True, errors="coerce")
        if pd.isna(start_time):
            return False
        frame = bars.iloc[: index + 1].copy()
        frame.loc[:, "timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        after_event = frame[frame["timestamp"] > start_time]
        if after_event.empty:
            return False
        closes = after_event["close"].astype(float)
        if str(zone["side"]) == "Sell":
            return bool((closes > float(zone["zone_high"])).any())
        return bool((closes < float(zone["zone_low"])).any())

    def _orderflow_ok(self, side: str, row: pd.Series, htf: pd.DataFrame) -> bool:
        if not self.require_ema_context or self.orderflow_mode in {"none", "off", "false"}:
            return True
        mode = self.orderflow_mode
        checks = {
            "ema21": self._ema_single_ok(side, row, self.ema_fast_period),
            "ema200": self._ema_single_ok(side, row, self.ema_slow_period),
            "ema_any": self._ema_any_ok(side, row),
            "ema_both": self._ema_both_ok(side, row),
            "htf_structure": self._htf_structure_ok(side, htf),
        }
        if mode in {"adaptive", "adaptive_any", "any"}:
            return checks["ema21"] or checks["ema200"] or checks["htf_structure"]
        ok = checks.get(mode, checks["ema_any"])
        if not ok:
            self._last_reject_reason = f"orderflow rejected by {mode}"
        return ok

    def _ema_values(self, row: pd.Series) -> tuple[float, float, float] | None:
        close = float(row.get("close", row.get("price", 0.0)))
        fast = row.get(f"ema_{self.ema_fast_period}")
        slow = row.get(f"ema_{self.ema_slow_period}")
        if fast is None or slow is None or pd.isna(fast) or pd.isna(slow):
            self._last_reject_reason = "ema context unavailable"
            return None
        return close, float(fast), float(slow)

    def _ema_single_ok(self, side: str, row: pd.Series, period: int) -> bool:
        close = float(row.get("close", row.get("price", 0.0)))
        value = row.get(f"ema_{period}")
        if value is None or pd.isna(value):
            return False
        if side == "Buy":
            return close > float(value)
        return close < float(value)

    def _ema_any_ok(self, side: str, row: pd.Series) -> bool:
        values = self._ema_values(row)
        if values is None:
            return False
        close, fast_value, slow_value = values
        if side == "Buy":
            return close > fast_value or close > slow_value or fast_value > slow_value
        return close < fast_value or close < slow_value or fast_value < slow_value

    def _ema_both_ok(self, side: str, row: pd.Series) -> bool:
        values = self._ema_values(row)
        if values is None:
            return False
        close, fast_value, slow_value = values
        if side == "Buy":
            return close > fast_value and close > slow_value and fast_value >= slow_value
        return close < fast_value and close < slow_value and fast_value <= slow_value

    def _htf_structure_ok(self, side: str, htf: pd.DataFrame) -> bool:
        if len(htf) <= self.htf_structure_lookback + 1:
            return False
        latest = htf.iloc[-1]
        prior = htf.iloc[-self.htf_structure_lookback - 1 : -1]
        if side == "Buy":
            return float(latest["close"]) > float(prior["high"].max())
        return float(latest["close"]) < float(prior["low"].min())

    def _confirmation_ok(self, side: str, index: int, bars: pd.DataFrame) -> bool:
        if index < 2:
            return False
        row = bars.iloc[index]
        prev = bars.iloc[index - 1]
        prior = bars.iloc[index - 2]
        open_price = float(row["open"])
        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        prev_open = float(prev["open"])
        prev_close = float(prev["close"])
        body_ticks = abs(close - open_price) / self.tick_size
        if body_ticks < self.min_confirmation_body_ticks:
            self._last_reject_reason = "confirmation body too small"
            return False
        body = max(abs(close - open_price), self.tick_size)
        upper_wick_ratio = (high - max(open_price, close)) / body
        lower_wick_ratio = (min(open_price, close) - low) / body
        bullish_engulf = close > open_price and prev_close < prev_open and close > prev_open and open_price <= prev_close
        bearish_engulf = close < open_price and prev_close > prev_open and close < prev_open and open_price >= prev_close
        bullish_123 = close > open_price and close > float(prev["high"]) and float(prev["low"]) > float(prior["low"])
        bearish_123 = close < open_price and close < float(prev["low"]) and float(prev["high"]) < float(prior["high"])
        if self.structure_123_mode in {"wick_break", "wick"}:
            bullish_123 = close > open_price and high > float(prev["high"]) and close > float(prev["close"]) and float(prev["low"]) > float(prior["low"])
            bearish_123 = close < open_price and low < float(prev["low"]) and close < float(prev["close"]) and float(prev["high"]) < float(prior["high"])
        bullish_trendline = self._trendline_break_ok("Buy", index, bars)
        bearish_trendline = self._trendline_break_ok("Sell", index, bars)
        bullish_fib382 = self._fib382_plus_one_ok("Buy", index, bars)
        bearish_fib382 = self._fib382_plus_one_ok("Sell", index, bars)
        bullish_wick = close > open_price and lower_wick_ratio >= self.min_reject_wick_ratio
        bearish_wick = close < open_price and upper_wick_ratio >= self.min_reject_wick_ratio
        if self.confirmation == "engulf":
            return bullish_engulf if side == "Buy" else bearish_engulf
        if self.confirmation == "123":
            return bullish_123 if side == "Buy" else bearish_123
        if self.confirmation == "engulf_or_123":
            return (bullish_engulf or bullish_123) if side == "Buy" else (bearish_engulf or bearish_123)
        if self.confirmation == "trendline":
            return bullish_trendline if side == "Buy" else bearish_trendline
        if self.confirmation in {"123_or_trendline", "trendline_or_123"}:
            return (bullish_123 or bullish_trendline) if side == "Buy" else (bearish_123 or bearish_trendline)
        if self.confirmation in {"fib382", "fib382_plus1"}:
            return bullish_fib382 if side == "Buy" else bearish_fib382
        if self.confirmation == "wick":
            return bullish_wick if side == "Buy" else bearish_wick
        if self.confirmation in {"confluence2", "any2"}:
            if side == "Buy":
                return sum([bullish_engulf, bullish_123, bullish_trendline, bullish_fib382, bullish_wick]) >= 2
            return sum([bearish_engulf, bearish_123, bearish_trendline, bearish_fib382, bearish_wick]) >= 2
        if side == "Buy":
            return bullish_engulf or bullish_123 or bullish_trendline or bullish_fib382 or bullish_wick
        return bearish_engulf or bearish_123 or bearish_trendline or bearish_fib382 or bearish_wick

    def _fib382_plus_one_ok(self, side: str, index: int, bars: pd.DataFrame) -> bool:
        if index < 1:
            return False
        setup = bars.iloc[index - 1]
        confirm = bars.iloc[index]
        setup_open = float(setup["open"])
        setup_close = float(setup["close"])
        setup_high = float(setup["high"])
        setup_low = float(setup["low"])
        setup_range = max(setup_high - setup_low, self.tick_size)
        confirm_open = float(confirm["open"])
        confirm_close = float(confirm["close"])
        if side == "Buy":
            fib382 = setup_low + (0.382 * setup_range)
            body_holds = min(setup_open, setup_close) >= fib382
            return body_holds and setup_close > setup_open and confirm_close > confirm_open and confirm_close >= setup_close
        fib382 = setup_high - (0.382 * setup_range)
        body_holds = max(setup_open, setup_close) <= fib382
        return body_holds and setup_close < setup_open and confirm_close < confirm_open and confirm_close <= setup_close

    def _trendline_break_ok(self, side: str, index: int, bars: pd.DataFrame) -> bool:
        if index < 8:
            return False
        source_high = bars[["open", "close"]].max(axis=1) if self.trendline_source == "body" else bars["high"]
        source_low = bars[["open", "close"]].min(axis=1) if self.trendline_source == "body" else bars["low"]
        start = max(2, index - self.trendline_lookback_bars)
        end = index - 2
        pivots: list[tuple[int, float]] = []
        if side == "Buy":
            for pos in range(start, end + 1):
                value = float(source_high.iloc[pos])
                if value >= float(source_high.iloc[pos - 1]) and value >= float(source_high.iloc[pos - 2]) and value >= float(source_high.iloc[pos + 1]) and value >= float(source_high.iloc[pos + 2]):
                    pivots.append((pos, value))
        else:
            for pos in range(start, end + 1):
                value = float(source_low.iloc[pos])
                if value <= float(source_low.iloc[pos - 1]) and value <= float(source_low.iloc[pos - 2]) and value <= float(source_low.iloc[pos + 1]) and value <= float(source_low.iloc[pos + 2]):
                    pivots.append((pos, value))
        if len(pivots) < 2:
            return False
        p1, p2 = pivots[-2], pivots[-1]
        if p2[0] == p1[0]:
            return False
        slope = (p2[1] - p1[1]) / float(p2[0] - p1[0])
        projected = p1[1] + slope * float(index - p1[0])
        close = float(bars.iloc[index]["close"])
        buffer = self.trendline_break_buffer_ticks * self.tick_size
        if side == "Buy":
            return close > projected + buffer
        return close < projected - buffer

    def _stop_price(self, side: str, row: pd.Series, zone: dict[str, Any]) -> float | None:
        if self.stop_mode == "fixed_ticks":
            return None
        padding = self.stop_padding_ticks * self.tick_size
        if side == "Buy":
            signal_stop = float(row["low"]) - padding
            otz_stop = float(zone["zone_low"]) - padding
            return min(signal_stop, otz_stop) if self.stop_mode in {"signal_or_otz", "otz"} else signal_stop
        signal_stop = float(row["high"]) + padding
        otz_stop = float(zone["zone_high"]) + padding
        return max(signal_stop, otz_stop) if self.stop_mode in {"signal_or_otz", "otz"} else signal_stop

    def _target_price(self, side: str, row: pd.Series, htf: pd.DataFrame, active_zone: dict[str, Any]) -> float | None:
        if self.target_mode not in {"opposite_otz", "nearest_opposite_otz"}:
            return None
        recent = htf.iloc[-self.htf_lookback :].copy()
        recent = recent[recent["atr"].notna()]
        if recent.empty:
            return None
        zones = self._engulfed_wick_orderblock_zones(recent)
        current_close = float(row.get("close", row.get("price", 0.0)))
        if side == "Buy":
            candidates = [zone for zone in zones if zone["side"] == "Sell" and float(zone["zone_low"]) > current_close]
            if not candidates:
                return None
            target = min(candidates, key=lambda zone: float(zone["zone_low"]) - current_close)
            return float(target["zone_low"])
        candidates = [zone for zone in zones if zone["side"] == "Buy" and float(zone["zone_high"]) < current_close]
        if not candidates:
            return None
        target = min(candidates, key=lambda zone: current_close - float(zone["zone_high"]))
        return float(target["zone_high"])

    def on_bar(self, index, bars, indicators):
        if index < self.warmup_bars:
            return None
        if not self._symbol_allowed():
            return None
        if self._last_signal_index is not None and index - self._last_signal_index < self.min_signal_gap_bars:
            return None
        row = indicators.iloc[index]
        if not self._hour_allowed(row):
            self._last_reject_reason = "hour outside allowed set"
            return None
        htf = self._completed_htf(bars.iloc[: index + 1])
        zone = self._active_zone(htf)
        if not zone:
            self._last_reject_reason = "no active completed HTF impulse zone"
            return None
        side = str(zone["side"])
        if self._zone_invalidated_since(bars, index, zone):
            self._last_reject_reason = "otz invalidated by close through zone"
            return None
        if not self._side_allowed(side):
            return None
        if not self._trigger_location_ok(row, zone):
            self._last_reject_reason = "zone not touched"
            return None
        if not self._orderflow_ok(side, row, htf):
            return None
        if not self._confirmation_ok(side, index, bars):
            return None
        self._last_signal_index = index
        self._last_reject_reason = None
        return StrategySignal(
            strategy_name=self.name,
            side=side,
            quantity=self.quantity,
            sl_ticks=self.sl_ticks,
            tp_ticks=self.tp_ticks,
            confidence=0.68,
            reason=f"OTZ {side.lower()} continuation after completed HTF impulse zone touch",
            metadata={
                **zone,
                "signal_index": int(index),
                "execution_model": "next_bar_open_required",
                "htf_minutes": self.htf_minutes,
                "zone_mode": self.zone_mode,
                "confirmation": self.confirmation,
                "engulf_mode": self.engulf_mode,
                "structure_123_mode": self.structure_123_mode,
                "trigger_zone_mode": self.trigger_zone_mode,
                "orderflow_mode": self.orderflow_mode,
                "stop_mode": self.stop_mode,
                "stop_price": self._stop_price(side, row, zone),
                "target_mode": self.target_mode,
                "target_price": self._target_price(side, row, htf, zone),
                "target_r_multiple": self.target_r_multiple if self.target_mode in {"risk_multiple", "r_multiple"} else None,
            },
        )
