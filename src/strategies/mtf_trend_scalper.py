from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseStrategy, StrategySignal

logger = logging.getLogger(__name__)


class MTFTrendScalperStrategy(BaseStrategy):
    """Institutional 5-Minute Multi-Timeframe (MTF) Trend & VWAP Reversion System.
    
    Architecture:
    1. 15-Minute Macro Bias: Requires 15m Close > 15m EMA 21 > 15m EMA 50 for Longs (inverse for Shorts).
    2. 5-Minute Execution: Requires 5m ADX >= 20 and a 5m pullback to 5m EMA 21 / VWAP value zone.
    3. ATR Volatility Normalization: Dynamic SL = 1.5 * 5m ATR(14), TP = 3.0 * 5m ATR(14) (2:1 R:R minimum).
    """

    name = "mtf_trend_scalper"
    warmup_bars = 180

    def __init__(self, symbol: str, params: dict[str, Any] | None = None):
        super().__init__(symbol, self._apply_symbol_overrides_for_symbol(symbol, params or {}))
        self.tick_size = float(self.params.get("tick_size", 0.25))
        self.quantity = int(self.params.get("quantity", 1))
        self.htf_minutes = int(self.params.get("htf_minutes", 15))
        self.ltf_minutes = int(self.params.get("ltf_minutes", 5))
        self.min_adx = float(self.params.get("min_adx", 20.0))
        self.sl_atr_mult = float(self.params.get("sl_atr_mult", 1.5))
        self.tp_atr_mult = float(self.params.get("tp_atr_mult", 3.0))
        self.min_stop_ticks = int(self.params.get("min_stop_ticks", 16))
        self.max_stop_ticks = int(self.params.get("max_stop_ticks", 80))
        self.allowed_entry_hours_et = self.params.get("allowed_entry_hours_et", [9, 10, 11, 12, 13, 14, 15])
        self.allowed_sides = [str(s).lower() for s in self.params.get("allowed_sides", ["buy", "sell"])]
        self.skip_open_minutes = int(self.params.get("skip_open_minutes", 5))
        self.skip_close_minutes = int(self.params.get("skip_close_minutes", 15))
        self._last_signal_index: Any = None
        self._last_evaluated_ltf_ts: Any = None

    @classmethod
    def _apply_symbol_overrides_for_symbol(cls, symbol: str, params: dict[str, Any]) -> dict[str, Any]:
        merged = dict(params or {})
        overrides = merged.get("symbol_params", {}) or {}
        if not isinstance(overrides, dict):
            return merged
        symbol_upper = symbol.upper()
        keys = [symbol_upper, cls._base_symbol(symbol_upper)]
        for key in keys:
            override = overrides.get(key)
            if isinstance(override, dict):
                merged.update(override)
        return merged

    def reset(self) -> None:
        self._last_signal_index = None
        self._last_evaluated_ltf_ts = None

    def get_description(self) -> str:
        return (
            f"MTF 5m/15m Trend Scalper ({self.symbol}) - "
            f"HTF: {self.htf_minutes}m EMA 21/50, LTF: {self.ltf_minutes}m ADX>={self.min_adx}, "
            f"SL: {self.sl_atr_mult}x ATR, TP: {self.tp_atr_mult}x ATR"
        )

    def get_snapshot(self, index: int, bars: pd.DataFrame, indicators: pd.DataFrame) -> dict[str, Any]:
        row = indicators.iloc[index] if index < len(indicators) else {}
        raw_ts = row.get("timestamp") if hasattr(row, "get") else None
        session_regime = "UNKNOWN"
        if raw_ts is not None and not pd.isna(raw_ts):
            try:
                import datetime
                import zoneinfo
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
                        ts = None
                else:
                    ts = pd.to_datetime(raw_ts)
                if ts is not None:
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=datetime.timezone.utc)
                    et = ts.astimezone(zoneinfo.ZoneInfo("America/New_York"))
                    if et.year >= 2000:
                        mins = et.hour * 60 + et.minute
                        if 570 <= mins < 960:
                            session_regime = "NY RTH"
                        elif 480 <= mins < 570:
                            session_regime = "NY PRE"
                        elif 1080 <= mins or mins < 120:
                            session_regime = "ASIAN"
                        else:
                            session_regime = "GLOBEX"
            except Exception:
                session_regime = "UNKNOWN"
        return {
            "status": "active" if self.enabled else "disabled",
            "strategy": self.name,
            "session_regime": session_regime,
        }

    def on_bar(self, index: int, bars: pd.DataFrame, indicators: pd.DataFrame) -> StrategySignal | None:
        if not self.enabled or index < self.warmup_bars:
            return None

        row = indicators.iloc[index]
        close = float(row.get("close", 0.0))
        open_ = float(row.get("open", close))
        high = float(row.get("high", close))
        low = float(row.get("low", close))
        ts = row.get("timestamp")

        dt_et = None
        if ts is not None and not pd.isna(ts):
            try:
                val = None
                try:
                    val = float(ts)
                except (ValueError, TypeError):
                    pass
                if val is not None:
                    if val > 1e16:
                        dt_et = pd.to_datetime(val, unit='ns', utc=True).tz_convert("America/New_York")
                    elif val > 1e13:
                        dt_et = pd.to_datetime(val, unit='us', utc=True).tz_convert("America/New_York")
                    elif val > 1e10:
                        dt_et = pd.to_datetime(val, unit='ms', utc=True).tz_convert("America/New_York")
                    elif val > 1e8:
                        dt_et = pd.to_datetime(val, unit='s', utc=True).tz_convert("America/New_York")
                else:
                    dt_et = pd.to_datetime(ts, utc=True).tz_convert("America/New_York")
            except Exception:
                dt_et = None

        # 1. RTH Time Filter
        if self.allowed_entry_hours_et and dt_et is not None:
            if dt_et.hour not in self.allowed_entry_hours_et:
                return None
            # Skip first N mins of market open (09:30 - 09:35 ET) and last M mins (15:45 - 16:00 ET)
            if self.skip_open_minutes > 0 and dt_et.hour == 9 and dt_et.minute < (30 + self.skip_open_minutes):
                return None
            if self.skip_close_minutes > 0 and dt_et.hour == 15 and dt_et.minute >= (60 - self.skip_close_minutes):
                return None

        if self.tick_size <= 0:
            return None

        # Fast path: if LTF > 1m and this 1m bar is not the closing minute of an LTF bar,
        # skip expensive resampling when the prior LTF bucket has already been evaluated.
        is_ltf_closing_min = (dt_et is not None) and ((dt_et.minute + 1) % self.ltf_minutes == 0)
        is_htf_closing_min = (dt_et is not None) and ((dt_et.minute + 1) % self.htf_minutes == 0)
        if self.ltf_minutes > 1 and dt_et is not None and not is_ltf_closing_min and self._last_evaluated_ltf_ts is not None:
            dt_utc = dt_et.tz_convert("UTC")
            prev_ltf_bucket = dt_utc.floor(f"{self.ltf_minutes}min") - pd.Timedelta(minutes=self.ltf_minutes)
            if self._last_evaluated_ltf_ts >= prev_ltf_bucket:
                return None

        # 2. Resample trailing bars for HTF (15m) bias and LTF (5m) execution
        lookback_1m = max(0, index - 2000)
        trailing_df = bars.iloc[lookback_1m : index + 1].copy()
        if "timestamp" not in trailing_df.columns:
            return None

        trailing_df = trailing_df.assign(dt=pd.to_datetime(trailing_df["timestamp"], utc=True))
        trailing_df = trailing_df.set_index("dt")

        # 2a. HTF (15m) Trend Bias
        rule_htf = f"{self.htf_minutes}min"
        df_htf = trailing_df.resample(rule_htf, closed="left", label="left").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last"
        }).dropna()

        if len(df_htf) < 15:
            return None

        df_htf = df_htf.assign(
            ema_21=df_htf["close"].ewm(span=21, adjust=False).mean(),
            ema_50=df_htf["close"].ewm(span=50, adjust=False).mean(),
        )

        htf_idx = -1 if (is_htf_closing_min or len(df_htf) < 2) else -2
        htf_last = df_htf.iloc[htf_idx]
        htf_close = float(htf_last["close"])
        htf_ema21 = float(htf_last["ema_21"])
        htf_ema50 = float(htf_last["ema_50"])

        htf_bullish = (htf_close > htf_ema21 > htf_ema50)
        htf_bearish = (htf_close < htf_ema21 < htf_ema50)

        # 2b. LTF (5m) Execution Frame
        if self.ltf_minutes > 1:
            rule_ltf = f"{self.ltf_minutes}min"
            df_ltf = trailing_df.resample(rule_ltf, closed="left", label="left").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last"
            }).dropna()

            if len(df_ltf) < 20:
                return None

            df_ltf = df_ltf.assign(ema_21=df_ltf["close"].ewm(span=21, adjust=False).mean())

            tr1 = df_ltf["high"] - df_ltf["low"]
            tr2 = (df_ltf["high"] - df_ltf["close"].shift(1)).abs()
            tr3 = (df_ltf["low"] - df_ltf["close"].shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df_ltf = df_ltf.assign(atr=tr.rolling(14).mean())

            up_move = df_ltf["high"] - df_ltf["high"].shift(1)
            down_move = df_ltf["low"].shift(1) - df_ltf["low"]
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            tr_s = tr.ewm(alpha=1 / 14, adjust=False).mean()
            pdm_s = pd.Series(plus_dm, index=df_ltf.index).ewm(alpha=1 / 14, adjust=False).mean()
            mdm_s = pd.Series(minus_dm, index=df_ltf.index).ewm(alpha=1 / 14, adjust=False).mean()
            plus_di = 100 * (pdm_s / (tr_s + 1e-9))
            minus_di = 100 * (mdm_s / (tr_s + 1e-9))
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
            df_ltf = df_ltf.assign(adx=dx.ewm(alpha=1 / 14, adjust=False).mean())

            ltf_idx = -1 if (is_ltf_closing_min or len(df_ltf) < 2) else -2
            ltf_last = df_ltf.iloc[ltf_idx]
            ltf_bar_ts = df_ltf.index[ltf_idx]

            # Guard against evaluating the skipped 09:30-09:34 opening bar when clock reaches 09:35
            try:
                ltf_et = ltf_bar_ts.tz_convert("America/New_York")
                if self.skip_open_minutes > 0 and ltf_et.hour == 9 and 30 <= ltf_et.minute < (30 + self.skip_open_minutes):
                    self._last_evaluated_ltf_ts = ltf_bar_ts
                    return None
            except Exception:
                pass

            if self._last_signal_index == ltf_bar_ts:
                return None
            if self._last_evaluated_ltf_ts == ltf_bar_ts:
                return None
            self._last_evaluated_ltf_ts = ltf_bar_ts

            adx = float(ltf_last["adx"])
            atr = float(ltf_last["atr"])
            close_eval = float(ltf_last["close"])
            open_eval = float(ltf_last["open"])
            high_eval = float(ltf_last["high"])
            low_eval = float(ltf_last["low"])
            ema21_eval = float(ltf_last["ema_21"])
            vwap_eval = float(row.get("vwap", close_eval))
            signal_key = ltf_bar_ts
        else:
            if self._last_signal_index == index:
                return None
            adx = float(row.get("adx", 0.0))
            atr = float(row.get("atr_14", row.get("atr", 0.0)))
            close_eval = close
            open_eval = open_
            high_eval = high
            low_eval = low
            ema21_eval = float(row.get("ema_21", close))
            vwap_eval = float(row.get("vwap", close))
            signal_key = index

        if math.isnan(adx) or math.isnan(atr) or adx < self.min_adx or atr <= 0:
            return None

        # Calculate ATR-based Stop & Target in Ticks with float precision rounding
        stop_ticks = max(self.min_stop_ticks, min(self.max_stop_ticks, int(math.ceil(round((self.sl_atr_mult * atr) / self.tick_size, 4)))))
        rr = self.tp_atr_mult / max(self.sl_atr_mult, 1e-9)
        target_ticks = max(stop_ticks * 2, int(round(stop_ticks * rr)))

        raw_1m_atr = float(row.get("atr", 0.0) or 0.0)
        atr_1m = raw_1m_atr if (not math.isnan(raw_1m_atr) and raw_1m_atr > 0) else atr
        touch_tol = max(self.tick_size * 2.0, 0.20 * atr)

        # Long Trigger: HTF Bullish, LTF price pulled back to EMA21 or VWAP and reclaimed it, LTF bar is bullish
        if htf_bullish and "buy" in self.allowed_sides:
            touched_ema = (low_eval <= ema21_eval + touch_tol) and (close_eval >= ema21_eval)
            touched_vwap = (low_eval <= vwap_eval + touch_tol) and (close_eval >= vwap_eval)
            bullish_close = (close_eval > open_eval) and (close_eval >= ema21_eval)
            if (touched_ema or touched_vwap) and bullish_close:
                self._last_signal_index = signal_key
                return StrategySignal(
                    strategy_name=self.name,
                    side="Buy",
                    quantity=self.quantity,
                    sl_ticks=stop_ticks,
                    tp_ticks=target_ticks,
                    confidence=0.85,
                    reason=f"{self.ltf_minutes}m MTF Long: {self.htf_minutes}m Bullish, {self.ltf_minutes}m Pullback to EMA/VWAP, ATR SL={stop_ticks}t, TP={target_ticks}t",
                    metadata={"htf_bias": "bullish", "adx": adx, "atr": atr_1m, "ltf_atr": atr, "atr_scaled_sl": True}
                )

        # Short Trigger: HTF Bearish, LTF price rallied to EMA21 or VWAP and rejected it, LTF bar is bearish
        if htf_bearish and "sell" in self.allowed_sides:
            touched_ema = (high_eval >= ema21_eval - touch_tol) and (close_eval <= ema21_eval)
            touched_vwap = (high_eval >= vwap_eval - touch_tol) and (close_eval <= vwap_eval)
            bearish_close = (close_eval < open_eval) and (close_eval <= ema21_eval)
            if (touched_ema or touched_vwap) and bearish_close:
                self._last_signal_index = signal_key
                return StrategySignal(
                    strategy_name=self.name,
                    side="Sell",
                    quantity=self.quantity,
                    sl_ticks=stop_ticks,
                    tp_ticks=target_ticks,
                    confidence=0.85,
                    reason=f"{self.ltf_minutes}m MTF Short: {self.htf_minutes}m Bearish, {self.ltf_minutes}m Rally to EMA/VWAP, ATR SL={stop_ticks}t, TP={target_ticks}t",
                    metadata={"htf_bias": "bearish", "adx": adx, "atr": atr_1m, "ltf_atr": atr, "atr_scaled_sl": True}
                )

        return None
