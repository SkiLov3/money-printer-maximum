import pandas as pd
import numpy as np
from src.strategies.mtf_trend_scalper import MTFTrendScalperStrategy


def create_sample_bars(num_bars=250):
    dates = pd.date_range("2026-08-17 09:35:00", periods=num_bars, freq="5min", tz="UTC")
    np.random.seed(42)
    close = 20000.0 + np.cumsum(np.random.randn(num_bars) * 5.0)
    high = close + np.random.uniform(2.0, 10.0, num_bars)
    low = close - np.random.uniform(2.0, 10.0, num_bars)
    open_ = close + np.random.uniform(-3.0, 3.0, num_bars)

    df = pd.DataFrame({
        "timestamp": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": 1000
    })
    return df


def test_mtf_trend_scalper_init():
    strat = MTFTrendScalperStrategy("MNQU6", {"min_adx": 25.0, "sl_atr_mult": 1.5, "tp_atr_mult": 3.0})
    assert strat.symbol == "MNQU6"
    assert strat.min_adx == 25.0
    assert strat.sl_atr_mult == 1.5
    assert strat.tp_atr_mult == 3.0


def test_mtf_trend_scalper_on_bar_warmup():
    strat = MTFTrendScalperStrategy("MNQU6")
    df = create_sample_bars(50)
    sig = strat.on_bar(10, df, df)
    assert sig is None


def test_mtf_trend_scalper_symbol_overrides():
    params = {
        "min_stop_ticks": 16,
        "max_stop_ticks": 80,
        "sl_atr_mult": 1.5,
        "symbol_params": {
            "MES": {
                "min_stop_ticks": 36,
                "max_stop_ticks": 60,
                "sl_atr_mult": 2.5,
            },
            "MCL": {
                "min_stop_ticks": 16,
                "max_stop_ticks": 50,
            }
        }
    }
    strat_mes = MTFTrendScalperStrategy("MESZ6", params)
    assert strat_mes.min_stop_ticks == 36
    assert strat_mes.max_stop_ticks == 60
    assert strat_mes.sl_atr_mult == 2.5

    strat_mcl = MTFTrendScalperStrategy("MCLX6", params)
    assert strat_mcl.min_stop_ticks == 16
    assert strat_mcl.max_stop_ticks == 50
    assert strat_mcl.sl_atr_mult == 1.5


def test_mtf_trend_scalper_5m_resampling_execution():
    strat = MTFTrendScalperStrategy("MCLX6", {
        "tick_size": 0.01,
        "htf_minutes": 15,
        "ltf_minutes": 5,
        "min_adx": 10.0,
        "sl_atr_mult": 1.5,
        "tp_atr_mult": 3.0,
        "min_stop_ticks": 10,
        "max_stop_ticks": 100,
        "allowed_entry_hours_et": [9, 10, 11, 12, 13, 14, 15],
    })
    strat.warmup_bars = 20

    dates = pd.date_range("2026-08-17 14:00:00", periods=250, freq="1min", tz="UTC")
    prices = 70.0 + np.linspace(0, 5.0, 250)
    bars = pd.DataFrame({
        "timestamp": dates,
        "open": prices - 0.02,
        "high": prices + 0.05,
        "low": prices - 0.05,
        "close": prices,
        "price": prices,
        "volume": 500,
    })
    from src.strategies.indicators import prepare_indicator_frame
    indicators = prepare_indicator_frame(bars, 0.01)

    signals = []
    for i in range(len(bars)):
        sig = strat.on_bar(i, bars, indicators)
        if sig is not None:
            signals.append((i, sig))

    # All signals must be spaced by at least 5 1-minute bars
    signal_indices = [s[0] for s in signals]
    for k in range(len(signal_indices) - 1):
        assert signal_indices[k+1] - signal_indices[k] >= 5

