
import pandas as pd

from src.strategies.always_in_15m_ema200 import AlwaysIn15mEMA200Strategy


def _build_1m_bars(prices, atr_value=2.0, start="2026-05-01 13:30:00"):
    """Build 1m bars and a matching indicators frame.

    The strategy needs `bars` (with timestamp/open/high/low/close/price) for
    its 15m resample and `indicators` (with `atr` and `timestamp`) for stop
    sizing. We construct both directly so tests don't depend on the full
    prepare_indicator_frame pipeline.
    """
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


def _make_strategy(symbol="MNQM6", **overrides):
    params = {
        "ema_period": 5,  # small so tests need fewer bars
        "bias_timeframe_minutes": 15,
        "atr_stop_mult": 2.0,
        "min_stop_ticks": 10,
        "tick_size": 0.25,
        "per_instrument_stop_caps": {"MNQ": 60, "MGC": 40, "MCL": None},
        "no_take_profit": True,
        "wait_for_fresh_close_after_sl": True,
        "quantity": 1,
    }
    params.update(overrides)
    return AlwaysIn15mEMA200Strategy(symbol=symbol, params=params)


def _last_signal(strategy, bars, indicators):
    signal = None
    for idx in range(len(bars)):
        current = strategy.on_bar(idx, bars, indicators)
        if current is not None:
            signal = current
    return signal


def test_emits_buy_when_15m_close_above_ema200():
    # 100 bars of strictly increasing prices => 15m closes above EMA200.
    prices = [4000.0 + i for i in range(120)]
    bars, indicators = _build_1m_bars(prices)
    strat = _make_strategy()
    sig = _last_signal(strat, bars, indicators)
    assert sig is not None, "expected a signal once warmup is satisfied"
    assert sig.side == "Buy"
    assert sig.tp_ticks == 0  # no-TP design


def test_emits_sell_when_15m_close_below_ema200():
    # Decreasing prices => 15m close < EMA200 => Sell bias.
    prices = [4500.0 - i for i in range(120)]
    bars, indicators = _build_1m_bars(prices)
    strat = _make_strategy()
    sig = _last_signal(strat, bars, indicators)
    assert sig is not None
    assert sig.side == "Sell"


def test_stop_ticks_capped_per_instrument():
    # ATR = 8.0 with tick_size=0.25 => raw stop = 2 * (8/0.25) = 64 ticks.
    # MNQ cap is 60, so the emitted SL must clamp to 60.
    prices = [4000.0 + i for i in range(120)]
    bars, indicators = _build_1m_bars(prices, atr_value=8.0)
    strat = _make_strategy(symbol="MNQM6")
    sig = _last_signal(strat, bars, indicators)
    assert sig is not None
    assert sig.sl_ticks == 60
    assert sig.metadata["raw_stop_ticks"] == 64


def test_stop_ticks_floor_at_min_stop_ticks():
    # ATR = 0.5 with tick_size=0.25 => raw stop = 2 * 2 = 4 ticks => floors to 10.
    prices = [4000.0 + i * 0.01 for i in range(120)]
    bars, indicators = _build_1m_bars(prices, atr_value=0.5)
    strat = _make_strategy(symbol="MGCM6")
    sig = _last_signal(strat, bars, indicators)
    assert sig is not None
    assert sig.sl_ticks == 10


def test_fresh_close_gate_suppresses_signal_within_same_15m_window():
    # Seed enough bars to establish bias, then end on a 15m boundary so we
    # have room within the next 15m window for extra bars to test suppression.
    # Start at 13:30, run 105 bars => last bar timestamp = 15:14.
    prices = [4000.0 + i for i in range(105)]
    bars, indicators = _build_1m_bars(prices)
    strat = _make_strategy()
    for idx in range(len(bars)):
        strat.on_bar(idx, bars, indicators)

    last_ts = pd.to_datetime(bars["timestamp"].iloc[-1], utc=True)
    assert last_ts.minute == 14  # sanity check on test setup

    # SL fires inside the 15:15-15:30 window (e.g. 15:17). Earliest re-entry is the 15:30 close.
    sl_at = last_ts.floor("15min") + pd.Timedelta(minutes=17)  # 15:17
    strat.notify_stop_exit(sl_at)

    # Add bars at 15:15 through 15:29 — all within the SL's 15m window.
    # Signals from any of these bars must be suppressed.
    extra_minutes = 15
    last_close = bars["close"].iloc[-1]
    extra_prices = [last_close + i for i in range(1, extra_minutes + 1)]
    extra_ts = pd.date_range(
        last_ts + pd.Timedelta(minutes=1), periods=extra_minutes, freq="1min", tz="UTC"
    )
    extra_bars = pd.DataFrame(
        {
            "timestamp": extra_ts,
            "open": extra_prices,
            "high": [p + 0.25 for p in extra_prices],
            "low": [p - 0.25 for p in extra_prices],
            "close": extra_prices,
            "price": extra_prices,
            "volume": [100] * extra_minutes,
        }
    )
    new_bars = pd.concat([bars, extra_bars], ignore_index=True)
    new_ind = new_bars.copy()
    new_ind["atr"] = 2.0

    suppressed = 0
    for offset in range(extra_minutes):
        idx = len(bars) + offset
        sig = strat.on_bar(idx, new_bars, new_ind)
        if sig is None:
            suppressed += 1
    assert suppressed == extra_minutes, "all signals within the SL's 15m window must be suppressed"


def test_fresh_close_after_sl_clears_gate():
    prices = [4000.0 + i for i in range(120)]
    bars, indicators = _build_1m_bars(prices)
    strat = _make_strategy()
    for idx in range(len(bars)):
        strat.on_bar(idx, bars, indicators)

    last_ts = pd.to_datetime(bars["timestamp"].iloc[-1], utc=True)
    sl_at = last_ts.floor("15min") + pd.Timedelta(minutes=2)
    strat.notify_stop_exit(sl_at)

    # Advance past the next 15m close — append enough 1m bars to cross the boundary.
    next_close_at = last_ts.floor("15min") + pd.Timedelta(minutes=15)
    minutes_needed = int((next_close_at - last_ts).total_seconds() // 60) + 2
    extra_prices = [bars["close"].iloc[-1] + 0.5 * i for i in range(1, minutes_needed + 1)]
    extra_ts = pd.date_range(last_ts + pd.Timedelta(minutes=1), periods=len(extra_prices), freq="1min", tz="UTC")
    extra_bars = pd.DataFrame(
        {
            "timestamp": extra_ts,
            "open": extra_prices,
            "high": [p + 0.25 for p in extra_prices],
            "low": [p - 0.25 for p in extra_prices],
            "close": extra_prices,
            "price": extra_prices,
            "volume": [100] * len(extra_prices),
        }
    )
    new_bars = pd.concat([bars, extra_bars], ignore_index=True)
    new_ind = new_bars.copy()
    new_ind["atr"] = 2.0

    sig = None
    for offset in range(len(extra_prices)):
        idx = len(bars) + offset
        sig = strat.on_bar(idx, new_bars, new_ind)
    assert sig is not None, "expected re-entry signal after the next 15m close"
    assert sig.side in ("Buy", "Sell")
