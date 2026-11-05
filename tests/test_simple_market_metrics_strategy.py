import pandas as pd

from src.strategies.engine import StrategyEngine
from src.strategies.simple_market_metrics import SimpleMarketMetricsStrategy


def _indicator_frame():
    rows = []
    for idx in range(230):
        rows.append(
            {
                "timestamp": pd.Timestamp("2026-04-01T14:00:00Z") + pd.Timedelta(minutes=idx),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 101.2,
                "volume": 100,
                "price": 101.2,
                "smm_ha_open": 100.0,
                "smm_ha_high": 101.0,
                "smm_ha_low": 100.0,
                "smm_ha_close": 101.0,
                "smm_ema8": 100.7,
                "smm_ema13": 100.4,
                "smm_ema21": 100.1,
                "smm_ema200": 99.0,
                "smm_trend_switch": 1,
                "smm_plus_di": 55.0,
                "smm_minus_di": 10.0,
                "smm_mfi": 60.0,
            }
        )
    return pd.DataFrame(rows)


def test_simple_market_metrics_emits_completed_bar_buy_signal():
    indicators = _indicator_frame()
    strategy = SimpleMarketMetricsStrategy(
        "MNQM6",
        {
            "tick_size": 0.25,
            "sl_ticks": 30,
            "tp_ticks": 20,
            "allowed_entry_hours_et": [10, 11, 12, 13, 14],
        },
    )

    signal = strategy.on_bar(220, indicators, indicators)

    assert signal is not None
    assert signal.strategy_name == "simple_market_metrics"
    assert signal.side == "Buy"
    assert signal.tp_ticks == 20
    assert signal.metadata["execution_model"] == "next_bar_open_required"
    assert signal.metadata["profit_wave_exit"] is True


def test_strict_simple_market_metrics_filters_require_fresh_flow_and_cross():
    indicators = _indicator_frame()
    for offset, idx in enumerate(range(187, 221)):
        indicators.loc[idx, "smm_ha_close"] = 98.0 + (offset * 0.1)
    indicators.loc[216, "smm_ema8"] = 100.0
    indicators.loc[216, "smm_ema13"] = 100.2
    indicators.loc[217:220, "smm_ema8"] = [100.4, 100.6, 100.7, 100.8]
    indicators.loc[217:220, "smm_ema13"] = [100.3, 100.4, 100.5, 100.6]
    indicators.loc[217:220, "smm_ema21"] = [100.1, 100.2, 100.3, 100.4]
    indicators.loc[217:220, "smm_mfi"] = [55.0, 56.0, 58.0, 60.0]
    strategy = SimpleMarketMetricsStrategy(
        "MNQM6",
        {
            "tick_size": 0.25,
            "sl_ticks": 30,
            "tp_ticks": 20,
            "allowed_entry_hours_et": [10, 11, 12, 13, 14],
            "require_mfi_slope": True,
            "mfi_slope_lookback": 3,
            "min_mfi_slope": 2.0,
            "require_recent_ema_cross": True,
            "ema_cross_lookback": 6,
            "require_linreg_slope": True,
            "linreg_length": 21,
            "min_linreg_slope_ticks": 0.1,
        },
    )

    signal = strategy.on_bar(220, indicators, indicators)

    assert signal is not None
    assert signal.metadata["smm_mfi_slope_filter"] is True
    assert signal.metadata["smm_recent_ema_cross_filter"] is True
    assert signal.metadata["smm_linreg_slope_filter"] is True


def test_strategy_engine_registry_builds_simple_market_metrics():
    engine = StrategyEngine(
        symbol="MNQM6",
        default_qty=1,
        bracket={"sl": 30, "tp": 20, "tick_size": 0.25},
        config={
            "strategies": [
                {
                    "name": "simple_market_metrics",
                    "enabled": True,
                    "priority": 100,
                    "params": {"tick_size": 0.25},
                }
            ]
        },
    )

    assert [strategy.name for strategy in engine.strategies] == ["simple_market_metrics"]


def test_simple_market_metrics_reset_mode_if_price_crosses_ema21():
    strategy = SimpleMarketMetricsStrategy("MNQM6", {"tick_size": 0.25})
    strategy._current_mode = "buy"
    row = {
        "smm_trend_switch": 1,
        "close": 99.0,
        "smm_ha_close": 98.5,
        "smm_ema21": 100.0,
    }
    strategy._reset_mode_if_needed(row)
    assert strategy._current_mode == "none"

    strategy._current_mode = "sell"
    row_sell = {
        "smm_trend_switch": 1,
        "close": 101.0,
        "smm_ha_close": 101.5,
        "smm_ema21": 100.0,
    }
    strategy._reset_mode_if_needed(row_sell)
    assert strategy._current_mode == "none"
