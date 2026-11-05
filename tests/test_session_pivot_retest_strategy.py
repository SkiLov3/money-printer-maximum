import pandas as pd

from src.strategies.engine import StrategyEngine
from src.strategies.indicators import prepare_indicator_frame
from src.strategies.session_pivot_retest import SessionPivotRetestStrategy


def _session_bars():
    prior = pd.date_range("2026-04-30 13:30:00", periods=390, freq="1min")
    current = pd.date_range("2026-05-01 13:30:00", periods=80, freq="1min")
    timestamps = prior.append(current)
    bars = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0] * len(timestamps),
            "high": [101.0] * len(timestamps),
            "low": [99.0] * len(timestamps),
            "close": [100.0] * len(timestamps),
            "price": [100.0] * len(timestamps),
            "volume": [100] * len(timestamps),
        }
    )
    bars.loc[390 + 70, ["open", "high", "low", "close", "price"]] = [101.5, 102.0, 100.8, 101.5, 101.5]
    bars.loc[390 + 71, ["open", "high", "low", "close", "price"]] = [101.5, 101.75, 100.9, 101.4, 101.4]
    return bars


def test_session_pivot_retest_emits_previous_rth_retest_signal():
    strategy = SessionPivotRetestStrategy(
        "MNQM6",
        {
            "tick_size": 0.25,
            "level_source": "previous_rth",
            "mode": "retest",
            "allowed_entry_hours_et": [10],
            "sl_ticks": 30,
            "tp_ticks": 45,
        },
    )
    bars = _session_bars()
    indicators = prepare_indicator_frame(bars, tick_size=0.25)

    signal = strategy.on_bar(461, bars, indicators)

    assert signal is not None
    assert signal.side == "Buy"
    assert signal.metadata["entry_order_type"] == "stop_market"
    assert signal.metadata["trigger_source"] == "previous_rth"


def test_session_pivot_retest_respects_side_filter():
    strategy = SessionPivotRetestStrategy(
        "MNQM6",
        {
            "tick_size": 0.25,
            "level_source": "previous_rth",
            "mode": "retest",
            "allowed_entry_hours_et": [10],
            "allowed_sides": ["sell"],
            "sl_ticks": 30,
            "tp_ticks": 45,
        },
    )
    bars = _session_bars()
    indicators = prepare_indicator_frame(bars, tick_size=0.25)

    signal = strategy.on_bar(461, bars, indicators)

    assert signal is None


def test_session_pivot_retest_respects_level_name_filter():
    strategy = SessionPivotRetestStrategy(
        "MNQM6",
        {
            "tick_size": 0.25,
            "level_source": "previous_rth",
            "mode": "retest",
            "allowed_entry_hours_et": [10],
            "enabled_level_names": ["low"],
            "sl_ticks": 30,
            "tp_ticks": 45,
        },
    )
    bars = _session_bars()
    indicators = prepare_indicator_frame(bars, tick_size=0.25)

    signal = strategy.on_bar(461, bars, indicators)

    assert signal is None


def test_strategy_engine_registry_builds_session_pivot_retest():
    engine = StrategyEngine(
        symbol="MNQM6",
        default_qty=1,
        bracket={"sl": 40, "tp": 80, "tick_size": 0.25},
        config={
            "strategies": [
                {
                    "name": "session_pivot_retest",
                    "enabled": True,
                    "priority": 50,
                    "params": {"level_source": "previous_rth"},
                }
            ]
        },
    )

    assert [strategy.name for strategy in engine.strategies] == ["session_pivot_retest"]
