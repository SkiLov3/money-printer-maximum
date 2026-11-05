from __future__ import annotations

from typing import Any

import pandas as pd

from src.strategies.base import BaseStrategy, StrategySignal


class AlwaysIn15mEMA200Strategy(BaseStrategy):
    """Always-in directional bias driven by the 15m EMA200.

    On every 1m bar this strategy emits a Buy or Sell signal based on the
    most-recently-closed 15m bar's close vs its EMA200. Downstream filters
    no-op same-side re-entries (the position is already in the bias
    direction) and treat opposite-side intents as flips. Stop-loss is sized
    by ATR; no take-profit is emitted (the position runs until the bias
    flips or the broker stop fires).

    After a broker-side stop-out, `notify_stop_exit` is called by the
    supervisor; the strategy then suppresses signals until the next 15m
    close has been observed (the "wait for fresh close" rule).
    """

    name = "always_in_15m_ema200"
    warmup_bars = 200 * 15  # ~3000 1m bars for a stable 15m EMA200

    def __init__(self, symbol: str, params: dict[str, Any] | None = None):
        super().__init__(symbol, params)
        p = self.params
        self.ema_period = int(p.get("ema_period", 200))
        self.timeframe_minutes = int(p.get("bias_timeframe_minutes", 15))
        self.atr_mult = float(p.get("atr_stop_mult", 2.0))
        self.min_stop_ticks = int(p.get("min_stop_ticks", 10))
        self.tick_size = float(p.get("tick_size", 0.25))
        self.quantity = int(p.get("quantity", 1))
        self.no_take_profit = bool(p.get("no_take_profit", True))
        self.wait_for_fresh_close = bool(p.get("wait_for_fresh_close_after_sl", True))

        caps = p.get("per_instrument_stop_caps") or {}
        sym_upper = (symbol or "").upper()
        self.stop_cap: int | None = None
        # Pick the longest matching prefix so MES doesn't accidentally match MGC etc.
        best_key_len = 0
        for key, value in caps.items():
            if value is None:
                continue
            key_upper = str(key).upper()
            if sym_upper.startswith(key_upper) and len(key_upper) > best_key_len:
                self.stop_cap = int(value)
                best_key_len = len(key_upper)

        self._last_15m_close_ts: pd.Timestamp | None = None
        self._last_bias: str | None = None  # "Buy" or "Sell"
        self._skip_until_ts: pd.Timestamp | None = None
        self._last_sl_at: pd.Timestamp | None = None

    def reset(self) -> None:
        self._last_15m_close_ts = None
        self._last_bias = None
        self._skip_until_ts = None
        self._last_sl_at = None

    def get_description(self) -> str:
        return (
            f"Always-in {self.timeframe_minutes}m EMA{self.ema_period} bias "
            f"(SL-only, no-TP, fresh-close re-entry={self.wait_for_fresh_close})"
        )

    def notify_stop_exit(self, fill_time) -> None:
        if not self.wait_for_fresh_close or fill_time is None:
            return
        ts = pd.to_datetime(fill_time, utc=True)
        self._last_sl_at = ts
        tf = pd.Timedelta(minutes=self.timeframe_minutes)
        current_bar_open = ts.floor(f"{self.timeframe_minutes}min")
        # Earliest re-entry is the close of the bar that contains the SL fill.
        self._skip_until_ts = current_bar_open + tf

    def _resample_15m(self, bars: pd.DataFrame) -> pd.DataFrame:
        if bars is None or bars.empty or "timestamp" not in bars.columns:
            return pd.DataFrame()
        n = self.warmup_bars + 50
        slim = bars.tail(n).copy()
        ts = pd.to_datetime(slim["timestamp"], utc=True, errors="coerce")
        slim = slim.assign(timestamp=ts).dropna(subset=["timestamp"]).set_index("timestamp")
        if slim.empty:
            return pd.DataFrame()
        if "close" not in slim.columns and "price" in slim.columns:
            slim = slim.assign(close=slim["price"])
        if "close" not in slim.columns:
            return pd.DataFrame()
        agg: dict[str, str] = {"close": "last"}
        for col in ("open", "high", "low"):
            if col in slim.columns:
                agg[col] = "first" if col == "open" else ("max" if col == "high" else "min")
        rule = f"{self.timeframe_minutes}min"
        out = slim.resample(rule, label="right", closed="right").agg(agg).dropna(subset=["close"])
        return out

    def on_bar(self, index, bars, indicators) -> StrategySignal | None:
        if indicators is None or indicators.empty:
            return None
        row = indicators.iloc[index]
        bar_ts_raw = row.get("timestamp")
        if bar_ts_raw is None:
            return None
        bar_ts = pd.to_datetime(bar_ts_raw, utc=True, errors="coerce")
        if pd.isna(bar_ts):
            return None

        resampled = self._resample_15m(bars)
        if resampled.empty:
            return None
        # Drop any 15m bar whose close is in the future relative to current 1m bar.
        resampled = resampled[resampled.index <= bar_ts]
        if len(resampled) < self.ema_period + 1:
            return None

        ema_series = resampled["close"].ewm(span=self.ema_period, adjust=False).mean()
        latest_close = float(resampled["close"].iloc[-1])
        latest_ema = float(ema_series.iloc[-1])
        latest_15m_ts = resampled.index[-1]

        # Update bias only on a freshly observed 15m close.
        if self._last_15m_close_ts is None or latest_15m_ts > self._last_15m_close_ts:
            self._last_15m_close_ts = latest_15m_ts
            if latest_close > latest_ema:
                self._last_bias = "Buy"
            elif latest_close < latest_ema:
                self._last_bias = "Sell"
            # equality => keep prior bias

            # A fresh close at or after the post-SL skip boundary clears the gate.
            if self._skip_until_ts is not None and latest_15m_ts >= self._skip_until_ts:
                self._skip_until_ts = None
                self._last_sl_at = None

        # Post-SL gate: suppress signals until the fresh-close requirement is met.
        if self._skip_until_ts is not None and bar_ts < self._skip_until_ts:
            return None

        if self._last_bias is None:
            return None

        atr_value = float(row.get("atr") or 0.0)
        if atr_value <= 0 or self.tick_size <= 0:
            return None
        raw_stop_ticks = int(round(self.atr_mult * (atr_value / self.tick_size)))
        sl_ticks = max(self.min_stop_ticks, raw_stop_ticks)
        if self.stop_cap is not None and self.stop_cap > 0:
            sl_ticks = min(sl_ticks, self.stop_cap)
        tp_ticks = 0 if self.no_take_profit else int(self.params.get("tp_ticks", 0))

        return StrategySignal(
            strategy_name=self.name,
            side=self._last_bias,
            quantity=int(self.quantity),
            sl_ticks=int(sl_ticks),
            tp_ticks=int(tp_ticks),
            confidence=1.0,
            reason=f"15m close {'>' if self._last_bias == 'Buy' else '<'} EMA{self.ema_period}",
            metadata={
                "bias_timeframe_minutes": self.timeframe_minutes,
                "ema_period": self.ema_period,
                "ema_value": latest_ema,
                "ref_close": latest_close,
                "atr_value": atr_value,
                "stop_cap": self.stop_cap,
                "raw_stop_ticks": raw_stop_ticks,
            },
        )

    def get_snapshot(self, index, bars, indicators) -> dict[str, Any]:
        snapshot = {"status": "active", "strategy": self.name, "bias": self._last_bias}
        if self._last_15m_close_ts is not None:
            snapshot["last_15m_close_ts"] = self._last_15m_close_ts.isoformat()
        if self._skip_until_ts is not None:
            snapshot["skip_until_ts"] = self._skip_until_ts.isoformat()
        return snapshot
