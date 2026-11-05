from __future__ import annotations

from typing import Any

from .base import BaseStrategy, StrategySignal


class SSSStrategy(BaseStrategy):
    name = "sss"
    warmup_bars = 40

    def __init__(self, symbol: str, params: dict[str, Any] | None = None):
        params = self._apply_symbol_overrides(symbol, params or {})
        super().__init__(symbol, params)
        self.tick_size = float(self.params.get("tick_size", 0.25))
        self.quantity = int(self.params.get("quantity", 1))
        
        # Dynamic risk params
        self.min_sl_ticks = int(self.params.get("min_sl_ticks", self.params.get("sl_ticks", 100)))
        self.sl_buffer_ticks = int(self.params.get("sl_buffer_ticks", 8))
        self.tp_sl_ratio = float(self.params.get("tp_sl_ratio", 2.0))
        # Helper tick multiplier calculated dynamically from tick size.
        self.tick_conv = 1.0 / self.tick_size if self.tick_size > 0 else 4.0
        
        self.lookback = int(self.params.get("swing_lookback", 3))
        self.min_swing_ticks = int(self.params.get("min_swing_ticks", max(4, self.min_sl_ticks // 6)))
        self.breakout_buffer_ticks = int(self.params.get("breakout_buffer_ticks", 2))
        self.min_confluence = int(self.params.get("min_confluence", 3))
        self.require_trend_alignment = bool(self.params.get("require_trend_alignment", True))
        self.adx_threshold = float(self.params.get("adx_threshold", 20))
        self.reset()

    @classmethod
    def _apply_symbol_overrides(cls, symbol: str, params: dict[str, Any]) -> dict[str, Any]:
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

    @staticmethod
    def _base_symbol(symbol_upper: str) -> str:
        known = ("MNQ", "MES", "MCL", "MGC", "MYM", "NQ", "ES", "CL", "GC", "YM")
        for prefix in known:
            if symbol_upper.startswith(prefix):
                return prefix
        return symbol_upper

    def reset(self) -> None:
        self.swing_highs = []
        self.swing_lows = []
        self._last_confirmed_index = -1
        self._last_signal_key = None

    def get_description(self) -> str:
        return (
            "Structural swing strategy: confirm pivot highs/lows, then trade breakouts with confluence "
            "from trend, VWAP, and ADX."
        )

    def _append_swing(self, collection, candidate):
        if collection and candidate["index"] <= collection[-1]["index"]:
            return
        if collection:
            delta_ticks = abs(candidate["price"] - collection[-1]["price"]) / self.tick_size
            if delta_ticks < self.min_swing_ticks:
                return
        collection.append(candidate)
        if len(collection) > 20:
            collection.pop(0)

    def _confirm_swings(self, index, bars):
        if index < (self.lookback * 2):
            return
        candidate_index = index - self.lookback
        if candidate_index <= self._last_confirmed_index:
            return

        start = max(0, candidate_index - self.lookback)
        end = min(len(bars), candidate_index + self.lookback + 1)
        window = bars.iloc[start:end]
        candidate = bars.iloc[candidate_index]

        if float(candidate["high"]) >= float(window["high"].max()):
            self._append_swing(
                self.swing_highs,
                {"index": candidate_index, "price": float(candidate["high"]), "timestamp": candidate["timestamp"]},
            )
        if float(candidate["low"]) <= float(window["low"].min()):
            self._append_swing(
                self.swing_lows,
                {"index": candidate_index, "price": float(candidate["low"]), "timestamp": candidate["timestamp"]},
            )
        self._last_confirmed_index = candidate_index

    def _score_confluence(self, side, row):
        score = 0
        reasons = []
        price = float(row["close"])
        ema200 = float(row.get("ema_200", price))
        vwap = float(row.get("vwap", price))
        adx = float(row.get("adx", 0) or 0)
        rsi = float(row.get("rsi", 50) or 50)

        momentum_override = adx > 35
        if side == "Buy":
            if price > ema200:
                score += 1
                reasons.append("above_ema200")
            if price > vwap:
                score += 1
                reasons.append("above_vwap")
            if rsi < 68 or momentum_override:
                score += 1
                reasons.append("rsi_not_extended" if rsi < 68 else "rsi_momentum")
        else:  # Sell side
            if price < ema200:
                score += 1
                reasons.append("below_ema200")
            if price < vwap:
                score += 1
                reasons.append("below_vwap")
            # Require RSI > 55 to confirm bearish momentum (neutral to overbought territory).
            # The old check (> 32) was nearly always true and added no discriminating value.
            if rsi > 55 or momentum_override:
                score += 1
                reasons.append("rsi_bearish" if rsi > 55 else "rsi_momentum")

        if adx >= self.adx_threshold:
            score += 1
            reasons.append("adx_trending")

        return score, reasons

    def get_snapshot(self, index, bars, indicators) -> dict[str, Any]:
        return {
            "status": "Active" if index + 1 >= self.warmup_bars else f"Warming up ({index + 1}/{self.warmup_bars})",
            "strategy": self.name,
            "reasoning": (
                f"highs={len(self.swing_highs)} lows={len(self.swing_lows)} "
                f"lookback={self.lookback} confluence>={self.min_confluence}"
            ),
            "swing_highs": len(self.swing_highs),
            "swing_lows": len(self.swing_lows),
            "last_swing_high": self.swing_highs[-1]["price"] if self.swing_highs else None,
            "last_swing_low": self.swing_lows[-1]["price"] if self.swing_lows else None,
        }

    def on_bar(self, index, bars, indicators):
        self._confirm_swings(index, bars)
        if index < self.warmup_bars or len(self.swing_highs) < 2 or len(self.swing_lows) < 2:
            return None

        row = indicators.iloc[index]
        bar = bars.iloc[index]
        last_high = self.swing_highs[-1]
        prev_high = self.swing_highs[-2]
        last_low = self.swing_lows[-1]
        prev_low = self.swing_lows[-2]
        close = float(bar["close"])
        buffer_px = self.breakout_buffer_ticks * self.tick_size

        long_structure = last_low["price"] > prev_low["price"] and close > last_high["price"] + buffer_px
        short_structure = last_high["price"] < prev_high["price"] and close < last_low["price"] - buffer_px

        atr_ticks = float(row.get("atr", 0.0) or 0.0) * self.tick_conv
        if atr_ticks < self.min_sl_ticks * 0.6:
            return None  # Skip low-volatility chop (high false breakout rate)

        if long_structure:
            score, reasons = self._score_confluence("Buy", row)
            if score >= self.min_confluence:
                signal_key = ("Buy", last_high["index"], last_low["index"])
                if signal_key != self._last_signal_key:
                    self._last_signal_key = signal_key
                    if not self.require_trend_alignment or close > float(row.get("ema_200", close)):
                        # Structural SL: directly below last swing low + buffer
                        sl_ticks = int((close - last_low["price"]) * self.tick_conv) + self.sl_buffer_ticks
                        sl_ticks = max(sl_ticks, self.min_sl_ticks)
                        return StrategySignal(
                            strategy_name=self.name,
                            side="Buy",
                            quantity=self.quantity,
                            sl_ticks=sl_ticks,
                            tp_ticks=int(sl_ticks * self.tp_sl_ratio),
                            confidence=min(0.9, 0.35 + (score * 0.12)),
                            reason=f"Bullish structure breakout (SL={sl_ticks})",
                            metadata={
                                "confluence_score": score,
                                "confluence_reasons": reasons,
                                "last_high": last_high["price"],
                                "last_low": last_low["price"],
                            },
                        )

        if short_structure:
            score, reasons = self._score_confluence("Sell", row)
            if score >= self.min_confluence:
                signal_key = ("Sell", last_high["index"], last_low["index"])
                if signal_key != self._last_signal_key:
                    self._last_signal_key = signal_key
                    if not self.require_trend_alignment or close < float(row.get("ema_200", close)):
                        # Structural SL: directly above last swing high + buffer
                        sl_ticks = int((last_high["price"] - close) * self.tick_conv) + self.sl_buffer_ticks
                        sl_ticks = max(sl_ticks, self.min_sl_ticks)
                        return StrategySignal(
                            strategy_name=self.name,
                            side="Sell",
                            quantity=self.quantity,
                            sl_ticks=sl_ticks,
                            tp_ticks=int(sl_ticks * self.tp_sl_ratio),
                            confidence=min(0.9, 0.35 + (score * 0.12)),
                            reason=f"Bearish structure breakdown (SL={sl_ticks})",
                            metadata={
                                "confluence_score": score,
                                "confluence_reasons": reasons,
                                "last_high": last_high["price"],
                                "last_low": last_low["price"],
                            },
                        )

        return None

