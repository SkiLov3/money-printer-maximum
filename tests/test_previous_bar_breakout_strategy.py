import pandas as pd

from src.strategies.engine import StrategyEngine
from src.strategies.indicators import prepare_indicator_frame
from src.strategies.previous_bar_breakout import PreviousBarBreakoutStrategy


def _bars(close=104.0):
    periods = 36
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-05-01 13:30:00", periods=periods, freq="1min"),
            "open": [100.0] * periods,
            "high": [101.0] * periods,
            "low": [99.0] * periods,
            "close": [100.0] * periods,
            "price": [100.0] * periods,
            "volume": [100] * periods,
        }
    )
    bars.loc[35, ["open", "high", "low", "close", "price"]] = [101.0, close + 0.5, 100.75, close, close]
    return bars


def test_previous_bar_breakout_emits_completed_bar_signal():
    strategy = PreviousBarBreakoutStrategy(
        "MNQM6",
        {
            "tick_size": 0.25,
            "lookback": 4,
            "sl_ticks": 30,
            "tp_ticks": 60,
            "min_range_ticks": 4,
            "min_body_ratio": 0.4,
            "use_ema_filter": False,
        },
    )
    bars = _bars()
    indicators = prepare_indicator_frame(bars, tick_size=0.25)

    signal = strategy.on_bar(35, bars, indicators)

    assert signal is not None
    assert signal.strategy_name == "previous_bar_breakout"
    assert signal.side == "Buy"
    assert signal.sl_ticks == 30
    assert signal.tp_ticks == 60
    assert signal.metadata["execution_model"] == "next_bar_open_required"


def test_previous_bar_breakout_rejects_inside_range_close():
    strategy = PreviousBarBreakoutStrategy(
        "MNQM6",
        {
            "tick_size": 0.25,
            "lookback": 4,
            "min_range_ticks": 4,
            "min_body_ratio": 0.4,
            "use_ema_filter": False,
        },
    )
    bars = _bars(close=100.5)
    bars.loc[35, ["open", "high", "low", "close", "price"]] = [99.75, 101.25, 99.75, 100.5, 100.5]
    indicators = prepare_indicator_frame(bars, tick_size=0.25)

    assert strategy.on_bar(35, bars, indicators) is None
    assert strategy.get_snapshot(35, bars, indicators)["last_reject_reason"] == "no prior-range breakout"


def test_previous_bar_breakout_fade_reverses_breakout_side():
    strategy = PreviousBarBreakoutStrategy(
        "MNQM6",
        {
            "tick_size": 0.25,
            "lookback": 4,
            "direction_mode": "fade",
            "sl_ticks": 30,
            "tp_ticks": 45,
            "min_range_ticks": 4,
            "min_body_ratio": 0.4,
            "use_ema_filter": False,
        },
    )
    bars = _bars()
    indicators = prepare_indicator_frame(bars, tick_size=0.25)

    signal = strategy.on_bar(35, bars, indicators)

    assert signal is not None
    assert signal.side == "Sell"
    assert signal.metadata["direction_mode"] == "fade"


def test_strategy_engine_registry_builds_previous_bar_breakout():
    engine = StrategyEngine(
        symbol="MNQM6",
        default_qty=1,
        bracket={"sl": 40, "tp": 80, "tick_size": 0.25},
        config={
            "strategies": [
                {
                    "name": "previous_bar_breakout",
                    "enabled": True,
                    "priority": 50,
                    "params": {"lookback": 4, "use_ema_filter": False},
                }
            ]
        },
    )

    assert [strategy.name for strategy in engine.strategies] == ["previous_bar_breakout"]
