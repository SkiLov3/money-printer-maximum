import pandas as pd

from src.strategies.engine import StrategyEngine
from src.strategies.indicators import prepare_indicator_frame
from src.strategies.volatility_squeeze_breakout import VolatilitySqueezeBreakoutStrategy


def test_volatility_squeeze_breakout_emits_resting_stop_metadata():
    periods = 150
    closes = [100 + (0.03 if idx % 2 else -0.03) for idx in range(periods)]
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-05-01 13:30:00", periods=periods, freq="1min"),
            "open": closes,
            "high": [value + 0.08 for value in closes],
            "low": [value - 0.08 for value in closes],
            "close": closes,
            "price": closes,
            "volume": [100] * periods,
        }
    )
    strategy = VolatilitySqueezeBreakoutStrategy(
        "MNQM6",
        {
            "tick_size": 0.25,
            "bb_period": 20,
            "squeeze_lookback": 40,
            "squeeze_quantile": 1.0,
            "side_mode": "long_only",
            "sl_ticks": 30,
            "tp_ticks": 45,
        },
    )
    indicators = prepare_indicator_frame(bars, tick_size=0.25)

    signal = strategy.on_bar(140, bars, indicators)

    assert signal is not None
    assert signal.side == "Buy"
    assert signal.metadata["entry_order_type"] == "stop_market"
    assert signal.metadata["entry_trigger_price"] > float(bars.loc[140, "close"])


def test_strategy_engine_registry_builds_volatility_squeeze_breakout():
    engine = StrategyEngine(
        symbol="MNQM6",
        default_qty=1,
        bracket={"sl": 40, "tp": 80, "tick_size": 0.25},
        config={
            "strategies": [
                {
                    "name": "volatility_squeeze_breakout",
                    "enabled": True,
                    "priority": 50,
                    "params": {"side_mode": "long_only"},
                }
            ]
        },
    )

    assert [strategy.name for strategy in engine.strategies] == ["volatility_squeeze_breakout"]
