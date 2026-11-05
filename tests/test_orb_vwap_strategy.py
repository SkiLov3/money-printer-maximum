import pandas as pd

from src.strategies.engine import StrategyEngine
from src.strategies.indicators import prepare_indicator_frame
from src.strategies.orb_vwap import ORBVWAPStrategy


def _orb_bars(signal_close=104.0, signal_volume=300, periods=18):
    timestamps = pd.date_range("2026-05-01 13:30:00", periods=periods, freq="1min")
    bars = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0] * periods,
            "high": [101.0] * periods,
            "low": [99.0] * periods,
            "close": [100.0] * periods,
            "price": [100.0] * periods,
            "volume": [100] * periods,
        }
    )
    bars.loc[15, ["open", "high", "low", "close", "price", "volume"]] = [
        101.25,
        signal_close + 0.25,
        101.0,
        signal_close,
        signal_close,
        signal_volume,
    ]
    if periods > 16:
        bars.loc[16, ["open", "high", "low", "close", "price", "volume"]] = [
            104.0,
            105.0,
            103.75,
            104.75,
            104.75,
            350,
        ]
    return bars


def _run_strategy(strategy, bars):
    indicators = prepare_indicator_frame(bars, tick_size=strategy.tick_size)
    signal = None
    for idx in range(len(bars)):
        current = strategy.on_bar(idx, bars, indicators)
        if current is not None:
            signal = current
            break
    return signal


def test_orb_vwap_breakout_emits_base_strategy_signal():
    strategy = ORBVWAPStrategy(
        "MNQM6",
        {
            "tick_size": 0.25,
            "quantity": 2,
            "opening_range_start_et": "09:30",
            "opening_range_end_et": "09:45",
            "trade_window_end_et": "10:30",
            "direction_mode": "breakout",
            "use_vwap_filter": True,
            "min_or_ticks": 4,
            "max_or_ticks": 20,
            "min_volume_ratio": 1.2,
            "target_mode": "risk_multiple",
            "risk_reward": 2.0,
        },
    )

    signal = _run_strategy(strategy, _orb_bars())

    assert signal is not None
    assert signal.strategy_name == "orb_vwap"
    assert signal.side == "Buy"
    assert signal.quantity == 2
    assert signal.sl_ticks == 9
    assert signal.tp_ticks == 18
    assert signal.metadata["setup_type"] == "breakout"
    assert signal.metadata["range_ticks"] == 8
    assert signal.metadata["volume_ratio"] >= 1.2


def test_orb_vwap_respects_volume_ratio_filter():
    strategy = ORBVWAPStrategy(
        "MNQM6",
        {
            "tick_size": 0.25,
            "min_or_ticks": 4,
            "max_or_ticks": 20,
            "min_volume_ratio": 2.0,
        },
    )

    bars = _orb_bars(signal_volume=100, periods=16)
    signal = _run_strategy(strategy, bars)

    assert signal is None
    indicators = prepare_indicator_frame(bars, 0.25)
    snapshot = strategy.get_snapshot(15, bars, indicators)
    assert snapshot["last_reject_reason"] == "volume ratio below threshold"


def test_orb_vwap_uses_symbol_range_overrides():
    strategy = ORBVWAPStrategy(
        "MNQM6",
        {
            "tick_size": 0.25,
            "min_or_ticks": 4,
            "max_or_ticks": 20,
            "min_volume_ratio": 1.0,
            "symbol_params": {"MNQ": {"min_or_ticks": 12, "max_or_ticks": 30}},
        },
    )

    signal = _run_strategy(strategy, _orb_bars())

    assert signal is None
    assert strategy.min_or_ticks == 12
    assert strategy.max_or_ticks == 30


def test_orb_vwap_allows_one_trade_per_session():
    strategy = ORBVWAPStrategy(
        "MNQM6",
        {
            "tick_size": 0.25,
            "min_or_ticks": 4,
            "max_or_ticks": 20,
            "min_volume_ratio": 1.0,
            "max_one_trade_per_session": True,
        },
    )
    bars = _orb_bars(periods=18)
    indicators = prepare_indicator_frame(bars, tick_size=0.25)
    signals = []

    for idx in range(len(bars)):
        signal = strategy.on_bar(idx, bars, indicators)
        if signal is not None:
            signals.append(signal)

    assert len(signals) == 1


def test_strategy_engine_registry_builds_orb_vwap():
    engine = StrategyEngine(
        symbol="MNQM6",
        default_qty=1,
        bracket={"sl": 40, "tp": 80, "tick_size": 0.25},
        config={
            "strategies": [
                {
                    "name": "orb_vwap",
                    "enabled": True,
                    "priority": 50,
                    "params": {"min_or_ticks": 4, "max_or_ticks": 20},
                }
            ]
        },
    )

    assert [strategy.name for strategy in engine.strategies] == ["orb_vwap"]


def test_orb_vwap_opening_range_disabled_emits_no_ny_signal():
    strategy = ORBVWAPStrategy(
        "MNQM6",
        {
            "tick_size": 0.25,
            "opening_range_enabled": False,
            "min_or_ticks": 4,
            "max_or_ticks": 20,
            "min_volume_ratio": 1.0,
        },
    )
    bars = _orb_bars()
    signal = _run_strategy(strategy, bars)
    assert signal is None
    assert strategy.opening_range_enabled is False


def test_orb_vwap_asian_symbol_overrides():
    params = {
        "asian_enabled": True,
        "symbol_params": {
            "MES": {"asian_enabled": False},
            "MCL": {"asian_enabled": True},
        },
    }
    strat_mes = ORBVWAPStrategy("MESZ6", params)
    assert strat_mes.asian_enabled is False

    strat_mcl = ORBVWAPStrategy("MCLX6", params)
    assert strat_mcl.asian_enabled is True

