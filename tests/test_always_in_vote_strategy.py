"""Tests for the AlwaysInVoteStrategy meta-strategy.

We verify the vote -> bias mapping (long-only, short-only, mixed/neutral
fallback to 15m EMA200) and the per-voter bias dispatch table.
"""

from __future__ import annotations


import pandas as pd

from src.strategies import always_in_vote as aiv


def _build_1m_bars(prices, atr_value=2.0, start="2026-05-01 13:30:00"):
    n = len(prices)
    timestamps = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    bars = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices,
            "high": [p + 0.25 for p in prices],
            "low": [p - 0.25 for p in prices],
            "close": prices,
            "price": prices,
            "volume": [100] * n,
        }
    )
    indicators = bars.copy()
    indicators["atr"] = float(atr_value)
    return bars, indicators


def _make_vote_strategy(symbol="MNQM6", voters=None, **overrides):
    params = {
        "ema_period": 5,
        "bias_timeframe_minutes": 15,
        "atr_stop_mult": 2.0,
        "min_stop_ticks": 10,
        "tick_size": 0.25,
        "per_instrument_stop_caps": {"MNQ": 60, "MGC": 40, "MCL": None},
        "no_take_profit": True,
        "wait_for_fresh_close_after_sl": True,
        "quantity": 1,
    }
    if voters is not None:
        params["voters"] = voters
    params.update(overrides)
    return aiv.AlwaysInVoteStrategy(symbol=symbol, params=params)


def _last_signal(strat, bars, indicators):
    sig = None
    for idx in range(len(bars)):
        cur = strat.on_bar(idx, bars, indicators)
        if cur is not None:
            sig = cur
    return sig


# ----- Vote logic -----


def test_long_only_votes_emit_buy(monkeypatch):
    # Stub all voters to "long". Bars don't matter for vote logic; the
    # fallback still needs to emit a base signal so build a trending series.
    prices = [4000.0 + i for i in range(120)]
    bars, indicators = _build_1m_bars(prices)
    strat = _make_vote_strategy(voters=["golden_zone_scalper", "ema_pullback", "sss"])
    monkeypatch.setitem(aiv.BIAS_FUNCS, "golden_zone_scalper", lambda row: "long")
    monkeypatch.setitem(aiv.BIAS_FUNCS, "ema_pullback", lambda row: "long")
    monkeypatch.setitem(aiv.BIAS_FUNCS, "sss", lambda row: "long")
    sig = _last_signal(strat, bars, indicators)
    assert sig is not None
    assert sig.side == "Buy"
    assert sig.metadata["vote_source"] == "vote_consensus_long"
    assert sig.metadata["vote_longs"] == 3
    assert sig.metadata["vote_shorts"] == 0


def test_short_only_votes_emit_sell(monkeypatch):
    prices = [4500.0 - i for i in range(120)]  # falling so EMA200 fallback would say Sell anyway
    bars, indicators = _build_1m_bars(prices)
    strat = _make_vote_strategy(voters=["ichimoku", "trend_following"])
    monkeypatch.setitem(aiv.BIAS_FUNCS, "ichimoku", lambda row: "short")
    monkeypatch.setitem(aiv.BIAS_FUNCS, "trend_following", lambda row: "short")
    sig = _last_signal(strat, bars, indicators)
    assert sig is not None
    assert sig.side == "Sell"
    assert sig.metadata["vote_source"] == "vote_consensus_short"
    assert sig.metadata["vote_longs"] == 0
    assert sig.metadata["vote_shorts"] == 2


def test_mixed_votes_fall_back_to_ema200(monkeypatch):
    # Trending UP series so the 15m EMA200 fallback says Buy.
    prices = [4000.0 + i for i in range(120)]
    bars, indicators = _build_1m_bars(prices)
    strat = _make_vote_strategy(voters=["golden_zone_scalper", "ichimoku"], use_fallback_on_mixed=True)
    monkeypatch.setitem(aiv.BIAS_FUNCS, "golden_zone_scalper", lambda row: "long")
    monkeypatch.setitem(aiv.BIAS_FUNCS, "ichimoku", lambda row: "short")
    sig = _last_signal(strat, bars, indicators)
    assert sig is not None
    assert sig.metadata["vote_source"] == "ema200_fallback_mixed"
    # Trending up => fallback should choose Buy.
    assert sig.side == "Buy"


def test_all_neutral_falls_back_to_ema200(monkeypatch):
    prices = [4500.0 - i for i in range(120)]  # trending down
    bars, indicators = _build_1m_bars(prices)
    strat = _make_vote_strategy(voters=["golden_zone_scalper", "ichimoku"], use_fallback_on_neutral=True)
    monkeypatch.setitem(aiv.BIAS_FUNCS, "golden_zone_scalper", lambda row: "neutral")
    monkeypatch.setitem(aiv.BIAS_FUNCS, "ichimoku", lambda row: "neutral")
    sig = _last_signal(strat, bars, indicators)
    assert sig is not None
    assert sig.metadata["vote_source"] == "ema200_fallback_neutral"
    assert sig.side == "Sell"  # trending down


def test_signal_carries_meta_strategy_name(monkeypatch):
    prices = [4000.0 + i for i in range(120)]
    bars, indicators = _build_1m_bars(prices)
    strat = _make_vote_strategy(voters=["sss"])
    monkeypatch.setitem(aiv.BIAS_FUNCS, "sss", lambda row: "long")
    sig = _last_signal(strat, bars, indicators)
    assert sig is not None
    assert sig.strategy_name == "always_in_vote"
    assert sig.tp_ticks == 0


# ----- Per-voter bias function smoke tests -----


def _row(**fields):
    return pd.Series(fields)


def test_bias_funcs_handle_missing_indicators_as_neutral():
    # Empty row -> all bias funcs must return "neutral" (no NaN crashes).
    row = _row()
    for name, fn in aiv.BIAS_FUNCS.items():
        assert fn(row) == "neutral", f"{name} should return neutral on empty row"


def test_bias_func_golden_zone_long_when_fast_above_slow_with_adx():
    row = _row(ema_21=100.0, ema_50=99.0, adx=15.0)
    assert aiv._bias_golden_zone(row) == "long"


def test_bias_func_golden_zone_neutral_when_adx_below_threshold():
    row = _row(ema_21=100.0, ema_50=99.0, adx=5.0)
    assert aiv._bias_golden_zone(row) == "neutral"


def test_bias_func_ema_pullback_stack():
    long_row = _row(ema_14=10.5, ema_18=10.0, ema_24=9.5)
    assert aiv._bias_ema_pullback(long_row) == "long"
    short_row = _row(ema_14=9.5, ema_18=10.0, ema_24=10.5)
    assert aiv._bias_ema_pullback(short_row) == "short"


def test_bias_func_above_ema200_and_vwap():
    long_row = _row(close=101.0, ema_200=100.0, vwap=100.5)
    assert aiv._bias_above_ema200_and_vwap(long_row) == "long"
    short_row = _row(close=99.0, ema_200=100.0, vwap=99.5)
    assert aiv._bias_above_ema200_and_vwap(short_row) == "short"
    mixed_row = _row(close=100.5, ema_200=100.0, vwap=101.0)  # above ema200, below vwap
    assert aiv._bias_above_ema200_and_vwap(mixed_row) == "neutral"


def test_default_voters_cover_full_library():
    # Sanity check: the default voter list and BIAS_FUNCS dispatch table stay
    # in sync. If a strategy is added to the library it should also get a bias
    # function (or be intentionally omitted).
    assert set(aiv.DEFAULT_VOTERS) == set(aiv.BIAS_FUNCS.keys())
    assert len(aiv.DEFAULT_VOTERS) >= 15  # we expect ~17 voters
