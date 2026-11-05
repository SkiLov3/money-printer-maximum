import pandas as pd

from src.strategies.engine import StrategyEngine
from src.strategies.indicators import prepare_indicator_frame
from src.strategies.prior_extreme_vwap_reversion import PriorExtremeVWAPReversionStrategy


def _stop_run_bars():
    periods = 60
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
    bars.loc[59, ["open", "high", "low", "close", "price", "volume"]] = [99.5, 100.0, 97.0, 99.4, 99.4, 200]
    return bars


def test_prior_extreme_vwap_reversion_emits_long_reclaim():
    strategy = PriorExtremeVWAPReversionStrategy(
        "MNQM6",
        {
            "tick_size": 0.25,
            "lookback": 8,
            "min_vwap_distance_atr": 0.1,
            "rsi_oversold": 55,
            "min_wick_ratio": 0.3,
            "sl_ticks": 30,
            "tp_ticks": 24,
        },
    )
    bars = _stop_run_bars()
    indicators = prepare_indicator_frame(bars, tick_size=0.25)

    signal = strategy.on_bar(59, bars, indicators)

    assert signal is not None
    assert signal.strategy_name == "prior_extreme_vwap_reversion"
    assert signal.side == "Buy"
    assert signal.sl_ticks == 30
    assert signal.tp_ticks == 24
    assert signal.metadata["execution_model"] == "next_bar_open_required"


def test_strategy_engine_registry_builds_prior_extreme_vwap_reversion():
    engine = StrategyEngine(
        symbol="MNQM6",
        default_qty=1,
        bracket={"sl": 40, "tp": 80, "tick_size": 0.25},
        config={
            "strategies": [
                {
                    "name": "prior_extreme_vwap_reversion",
                    "enabled": True,
                    "priority": 50,
                    "params": {"lookback": 8, "min_vwap_distance_atr": 0.1, "rsi_oversold": 55},
                }
            ]
        },
    )

    assert [strategy.name for strategy in engine.strategies] == ["prior_extreme_vwap_reversion"]
