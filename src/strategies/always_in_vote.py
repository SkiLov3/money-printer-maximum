from __future__ import annotations

from typing import Any

import pandas as pd

from src.strategies.always_in_15m_ema200 import AlwaysIn15mEMA200Strategy
from src.strategies.base import BaseStrategy, StrategySignal


def _bias_golden_zone(row) -> str:
    fast = row.get("ema_21")
    slow = row.get("ema_50")
    adx = row.get("adx")
    if fast is None or slow is None or adx is None or pd.isna(fast) or pd.isna(slow) or pd.isna(adx):
        return "neutral"
    if adx < 10.0:
        return "neutral"
    if fast > slow:
        return "long"
    if fast < slow:
        return "short"
    return "neutral"


def _bias_ema_pullback(row) -> str:
    e14 = row.get("ema_14")
    e18 = row.get("ema_18")
    e24 = row.get("ema_24")
    if any(v is None or pd.isna(v) for v in (e14, e18, e24)):
        return "neutral"
    if e14 > e18 > e24:
        return "long"
    if e14 < e18 < e24:
        return "short"
    return "neutral"


def _bias_above_ema200_and_vwap(row) -> str:
    close = row.get("close") if row.get("close") is not None else row.get("price")
    ema200 = row.get("ema_200")
    vwap = row.get("vwap")
    if any(v is None or pd.isna(v) for v in (close, ema200, vwap)):
        return "neutral"
    if close > ema200 and close > vwap:
        return "long"
    if close < ema200 and close < vwap:
        return "short"
    return "neutral"


def _bias_ichimoku(row) -> str:
    close = row.get("close") if row.get("close") is not None else row.get("price")
    sa = row.get("senkou_a")
    sb = row.get("senkou_b")
    tk = row.get("tenkan")
    kj = row.get("kijun")
    if any(v is None or pd.isna(v) for v in (close, sa, sb, tk, kj)):
        return "neutral"
    if close > sa and close > sb and tk > kj:
        return "long"
    if close < sa and close < sb and tk < kj:
        return "short"
    return "neutral"


def _bias_fx_one_glance(row) -> str:
    close = row.get("close") if row.get("close") is not None else row.get("price")
    ema200 = row.get("ema_200")
    vwap = row.get("vwap")
    tk = row.get("tenkan")
    kj = row.get("kijun")
    if any(v is None or pd.isna(v) for v in (close, ema200, vwap, tk, kj)):
        return "neutral"
    if close > ema200 and close > vwap and tk > kj:
        return "long"
    if close < ema200 and close < vwap and tk < kj:
        return "short"
    return "neutral"


def _bias_trend_following(row) -> str:
    e9 = row.get("ema_9")
    e21 = row.get("ema_21")
    e200 = row.get("ema_200")
    close = row.get("close") if row.get("close") is not None else row.get("price")
    if any(v is None or pd.isna(v) for v in (e9, e21, e200, close)):
        return "neutral"
    if e9 > e21 > e200 and close > e200:
        return "long"
    if e9 < e21 < e200 and close < e200:
        return "short"
    return "neutral"


def _bias_orb_vwap(row) -> str:
    close = row.get("close") if row.get("close") is not None else row.get("price")
    vwap = row.get("vwap")
    rh = row.get("rolling_high_20")
    rl = row.get("rolling_low_20")
    if any(v is None or pd.isna(v) for v in (close, vwap, rh, rl)):
        return "neutral"
    if close > vwap and close >= rh:
        return "long"
    if close < vwap and close <= rl:
        return "short"
    return "neutral"


def _bias_resting_level(row) -> str:
    e9 = row.get("ema_9")
    e21 = row.get("ema_21")
    close = row.get("close") if row.get("close") is not None else row.get("price")
    rh = row.get("rolling_high_50")
    rl = row.get("rolling_low_50")
    if any(v is None or pd.isna(v) for v in (e9, e21, close, rh, rl)):
        return "neutral"
    if e9 > e21 and close >= rh:
        return "long"
    if e9 < e21 and close <= rl:
        return "short"
    return "neutral"


def _bias_squeeze_breakout(row) -> str:
    e21 = row.get("ema_21")
    close = row.get("close") if row.get("close") is not None else row.get("price")
    if any(v is None or pd.isna(v) for v in (e21, close)):
        return "neutral"
    if close > e21:
        return "long"
    if close < e21:
        return "short"
    return "neutral"


def _bias_htf_level(row) -> str:
    close = row.get("close") if row.get("close") is not None else row.get("price")
    rh = row.get("rolling_high_50")
    rl = row.get("rolling_low_50")
    if any(v is None or pd.isna(v) for v in (close, rh, rl)):
        return "neutral"
    if close >= rh:
        return "long"
    if close <= rl:
        return "short"
    return "neutral"


def _bias_smm(row) -> str:
    switch = row.get("smm_trend_switch")
    ha_close = row.get("smm_ha_close")
    ema21 = row.get("smm_ema21")
    if any(v is None or pd.isna(v) for v in (switch, ha_close, ema21)):
        return "neutral"
    if switch == 1 and ha_close > ema21:
        return "long"
    if switch == -1 and ha_close < ema21:
        return "short"
    return "neutral"


def _bias_otz(row) -> str:
    close = row.get("close") if row.get("close") is not None else row.get("price")
    ema50 = row.get("ema_50")
    vwap = row.get("vwap")
    if any(v is None or pd.isna(v) for v in (close, ema50, vwap)):
        return "neutral"
    if close > ema50 and close > vwap:
        return "long"
    if close < ema50 and close < vwap:
        return "short"
    return "neutral"


def _bias_above_ema21(row) -> str:
    close = row.get("close") if row.get("close") is not None else row.get("price")
    e21 = row.get("ema_21")
    if any(v is None or pd.isna(v) for v in (close, e21)):
        return "neutral"
    if close > e21:
        return "long"
    if close < e21:
        return "short"
    return "neutral"


def _bias_prior_extreme_reversion(row) -> str:
    # Mean-reversion fade: oversold + below VWAP -> long fade; overbought + above VWAP -> short fade
    close = row.get("close") if row.get("close") is not None else row.get("price")
    vwap = row.get("vwap")
    rsi = row.get("rsi")
    if any(v is None or pd.isna(v) for v in (close, vwap, rsi)):
        return "neutral"
    if rsi <= 30 and close < vwap:
        return "long"
    if rsi >= 70 and close > vwap:
        return "short"
    return "neutral"


# Dispatch table: strategy name -> bias function. Strategies whose state has
# no obvious continuous-bias notion (e.g. pure session-window strategies like
# asian_breakout, session_pivot_retest, previous_bar_breakout) default to a
# generic above-EMA21 cue so they participate without dominating.
BIAS_FUNCS = {
    "golden_zone_scalper": _bias_golden_zone,
    "ema_pullback": _bias_ema_pullback,
    "sss": _bias_above_ema200_and_vwap,
    "ichimoku": _bias_ichimoku,
    "fx_at_one_glance": _bias_fx_one_glance,
    "trend_following": _bias_trend_following,
    "asian_breakout": _bias_above_ema21,
    "orb_vwap": _bias_orb_vwap,
    "previous_bar_breakout": _bias_above_ema21,
    "prior_extreme_vwap_reversion": _bias_prior_extreme_reversion,
    "resting_level_breakout": _bias_resting_level,
    "volatility_squeeze_breakout": _bias_squeeze_breakout,
    "htf_level_retest": _bias_htf_level,
    "session_pivot_retest": _bias_above_ema200_and_vwap,
    "simple_market_metrics": _bias_smm,
    "otz_continuation": _bias_otz,
    "price_action_bs": _bias_above_ema21,
}


DEFAULT_VOTERS: list[str] = list(BIAS_FUNCS.keys())


class AlwaysInVoteStrategy(BaseStrategy):
    """Always-in directional bias driven by a vote of the existing strategy
    library plus a 15m EMA200 fallback.

    On every 1m bar:
    1. For each enabled voter, look up its current bias from `BIAS_FUNCS`
       (a deterministic function of the latest indicator row).
    2. Count Longs and Shorts. If only one side has votes, that's the bias.
    3. Otherwise (mixed or all neutral) fall back to the 15m EMA200 rule:
       15m close > EMA200 -> Buy, < -> Sell.
    4. Emit a Buy/Sell signal with ATR-based stop and no take-profit.
       Reuses the same fresh-close-after-SL gate as `AlwaysIn15mEMA200`.
    """

    name = "always_in_vote"
    warmup_bars = 200 * 15  # need enough data for 15m EMA200 fallback

    def __init__(self, symbol: str, params: dict[str, Any] | None = None):
        super().__init__(symbol, params)
        p = self.params
        voters_param = p.get("voters")
        if voters_param is None:
            self.voters: list[str] = list(DEFAULT_VOTERS)
        else:
            self.voters = [str(v) for v in voters_param if str(v) in BIAS_FUNCS]

        # Consensus and Regime parameters
        self.min_agreement_pct = float(p.get("min_agreement_pct", 0.6))  # e.g. 60% agreement required
        self.adx_chop_threshold = float(p.get("adx_chop_threshold", 20.0))
        self.use_fallback_on_mixed = bool(p.get("use_fallback_on_mixed", False))
        self.use_fallback_on_neutral = bool(p.get("use_fallback_on_neutral", False))

        # 15m EMA200 fallback delegates the heavy lifting (resampling, stop
        # sizing, fresh-close gate, notify_stop_exit) to the v1 strategy by
        # composition. We instantiate it with the same params so per-instrument
        # caps / no_take_profit / wait_for_fresh_close_after_sl all apply.
        self._fallback = AlwaysIn15mEMA200Strategy(symbol, params)
        # Override the fallback's name so any signals built via it still carry
        # this meta-strategy's name (matters for telemetry + cooldowns).
        self._fallback.name = self.name

    def reset(self) -> None:
        self._fallback.reset()

    def get_description(self) -> str:
        return (
            f"Always-in vote ({int(self.min_agreement_pct*100)}% consensus) of {len(self.voters)} strategies "
            f"+ 15m EMA{self._fallback.ema_period} fallback (SL-only, no-TP)"
        )

    def notify_stop_exit(self, fill_time) -> None:
        self._fallback.notify_stop_exit(fill_time)

    def _vote(self, row) -> tuple[int, int, dict[str, str]]:
        longs = 0
        shorts = 0
        per_voter: dict[str, str] = {}
        for name in self.voters:
            fn = BIAS_FUNCS.get(name)
            if fn is None:
                continue
            try:
                vote = fn(row)
            except Exception:
                vote = "neutral"
            per_voter[name] = vote
            if vote == "long":
                longs += 1
            elif vote == "short":
                shorts += 1
        return longs, shorts, per_voter

    def on_bar(self, index, bars, indicators) -> StrategySignal | None:
        if indicators is None or indicators.empty:
            return None
        row = indicators.iloc[index]
        
        # 1. Regime Filter: If ADX is low, we are in chop.
        adx = row.get("adx")
        if adx is not None and not pd.isna(adx) and adx < self.adx_chop_threshold:
            # Still run fallback to keep its internal state (fresh-close gate) updated,
            # but don't emit anything.
            self._fallback.on_bar(index, bars, indicators)
            return None

        # Always run the fallback first so its internal bias / fresh-close
        # state stays current. Its returned signal is what we'll emit if the
        # vote is mixed/neutral. We may override side based on the vote.
        fallback_signal = self._fallback.on_bar(index, bars, indicators)
        if fallback_signal is None:
            # Fallback declined to emit (warmup, gated by post-SL, missing
            # ATR, etc.). Honor that — the gate logic should win.
            return None

        longs, shorts, per_voter = self._vote(row)
        
        # Calculate agreement relative to total VOTERS (including neutrals)
        max_votes = max(longs, shorts)
        agreement_ratio = max_votes / len(self.voters) if self.voters else 0
        
        chosen_side: str | None = None
        source: str = ""

        if agreement_ratio >= self.min_agreement_pct:
            # Consensus reached
            if longs > shorts:
                chosen_side = "Buy"
                source = "vote_consensus_long"
            else:
                chosen_side = "Sell"
                source = "vote_consensus_short"
        else:
            # Mixed or low consensus
            if longs > 0 and shorts > 0:
                # Disagreement
                if self.use_fallback_on_mixed:
                    chosen_side = fallback_signal.side
                    source = "ema200_fallback_mixed"
            else:
                # All one side (but low count) or all neutral
                if self.use_fallback_on_neutral:
                    chosen_side = fallback_signal.side
                    source = "ema200_fallback_neutral"

        if chosen_side is None:
            return None

        # [OBSERVABILITY] Log the consensus details for transparency in trades.log
        self.logger.info(
            f"[VOTE] {self.symbol} consensus reached: {chosen_side} "
            f"({longs}L / {shorts}S of {len(self.voters)} voters, ratio={agreement_ratio:.2f})"
        )

        # Re-clothe the fallback's signal with the chosen side and richer
        # metadata. Stop sizing, qty, SL/TP fields all stay as the fallback
        # produced them (those depend on ATR + per-instrument caps, not side).
        fallback_signal.side = chosen_side
        fallback_signal.strategy_name = self.name
        fallback_signal.reason = (
            f"{source} ({longs}L / {shorts}S of {len(self.voters)} voters, ratio={agreement_ratio:.2f})"
        )
        meta = dict(fallback_signal.metadata or {})
        meta.update(
            {
                "vote_longs": longs,
                "vote_shorts": shorts,
                "vote_total": len(self.voters),
                "vote_ratio": agreement_ratio,
                "vote_source": source,
                "voter_biases": per_voter,
                "adx_value": adx,
            }
        )
        fallback_signal.metadata = meta
        return fallback_signal


    def get_snapshot(self, index, bars, indicators) -> dict[str, Any]:
        snap = self._fallback.get_snapshot(index, bars, indicators)
        
        # Add consensus metrics to the snapshot
        if indicators is not None and not indicators.empty and index < len(indicators):
            row = indicators.iloc[index]
            longs, shorts, _ = self._vote(row)
            max_votes = max(longs, shorts)
            agreement_ratio = max_votes / len(self.voters) if self.voters else 0
            adx = row.get("adx")
            
            snap.update({
                "vote_longs": longs,
                "vote_shorts": shorts,
                "vote_ratio": round(agreement_ratio, 2),
                "vote_consensus_threshold": self.min_agreement_pct,
                "adx_value": round(adx, 2) if adx is not None else None,
                "adx_threshold": self.adx_chop_threshold,
                "is_chop": adx < self.adx_chop_threshold if adx is not None else False
            })

        snap["strategy"] = self.name
        snap["voters"] = list(self.voters)
        return snap
