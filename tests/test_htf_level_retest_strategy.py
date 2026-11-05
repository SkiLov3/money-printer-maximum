import pandas as pd

from src.strategies.engine import StrategyEngine
from src.strategies.htf_level_retest import HTFLevelRetestStrategy
from src.strategies.indicators import prepare_indicator_frame


def test_htf_level_retest_engine_registry_builds_strategy():
    engine = StrategyEngine(
        symbol="MNQM6",
        default_qty=1,
        bracket={"sl": 40, "tp": 80, "tick_size": 0.25},
        config={
            "strategies": [
                {
                    "name": "htf_level_retest",
                    "enabled": True,
                    "priority": 50,
                    "params": {"htf_minutes": 15, "level_lookback": 4},
                }
            ]
        },
    )

    assert [strategy.name for strategy in engine.strategies] == ["htf_level_retest"]


def test_htf_level_retest_uses_completed_htf_bars_only():
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-05-01 13:30:00", periods=150, freq="1min"),
            "open": [100.0] * 150,
            "high": [101.0] * 150,
            "low": [99.0] * 150,
            "close": [100.0] * 150,
            "price": [100.0] * 150,
            "volume": [100] * 150,
        }
    )
    strategy = HTFLevelRetestStrategy("MNQM6", {"htf_minutes": 15, "level_lookback": 4})
    indicators = prepare_indicator_frame(bars, tick_size=0.25)

    htf = strategy._completed_htf(bars.iloc[:121])

    assert not htf.empty
    assert htf["timestamp"].max() < pd.to_datetime(bars.loc[120, "timestamp"], utc=True)
    assert strategy.on_bar(120, bars, indicators) is None


def test_htf_level_retest_respects_symbol_filter():
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-05-01 13:30:00", periods=150, freq="1min"),
            "open": [100.0] * 150,
            "high": [101.0] * 150,
            "low": [99.0] * 150,
            "close": [100.0] * 150,
            "price": [100.0] * 150,
            "volume": [100] * 150,
        }
    )
    strategy = HTFLevelRetestStrategy("MNQM6", {"enabled_symbols": ["MCL"]})
    indicators = prepare_indicator_frame(bars, tick_size=0.25)

    assert strategy.on_bar(120, bars, indicators) is None
    assert strategy._last_reject_reason is None
