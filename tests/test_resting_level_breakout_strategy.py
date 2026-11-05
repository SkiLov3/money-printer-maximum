import pandas as pd

from src.strategies.engine import StrategyEngine
from src.strategies.indicators import prepare_indicator_frame
from src.strategies.resting_level_breakout import RestingLevelBreakoutStrategy


def test_resting_level_breakout_emits_stop_trigger_metadata():
    periods = 60
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-05-01 13:30:00", periods=periods, freq="1min"),
            "open": [100.0] * periods,
            "high": [101.0] * periods,
            "low": [99.0] * periods,
            "close": [100.5] * periods,
            "price": [100.5] * periods,
            "volume": [100] * periods,
        }
    )
    strategy = RestingLevelBreakoutStrategy(
        "MNQM6",
        {
            "tick_size": 0.25,
            "level_source": "rolling_donchian",
            "lookback": 12,
            "side_mode": "long_only",
            "min_adx": 0,
            "min_range_ticks": 4,
            "sl_ticks": 24,
            "tp_ticks": 36,
        },
    )
    indicators = prepare_indicator_frame(bars, tick_size=0.25)

    signal = strategy.on_bar(55, bars, indicators)

    assert signal is not None
    assert signal.side == "Buy"
    assert signal.metadata["entry_order_type"] == "stop_market"
    assert signal.metadata["entry_trigger_price"] == 101.25
    assert signal.metadata["active_from_index"] == 56


def test_strategy_engine_registry_builds_resting_level_breakout():
    engine = StrategyEngine(
        symbol="MNQM6",
        default_qty=1,
        bracket={"sl": 40, "tp": 80, "tick_size": 0.25},
        config={
            "strategies": [
                {
                    "name": "resting_level_breakout",
                    "enabled": True,
                    "priority": 50,
                    "params": {"lookback": 12, "side_mode": "long_only"},
                }
            ]
        },
    )

    assert [strategy.name for strategy in engine.strategies] == ["resting_level_breakout"]
