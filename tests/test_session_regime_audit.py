import pandas as pd
from src.strategies.engine import StrategyEngine
from src.strategies.orb_vwap import ORBVWAPStrategy
from src.strategies.mtf_trend_scalper import MTFTrendScalperStrategy
from src.strategies.price_action_bs import PriceActionBSStrategy


def test_engine_session_regime_calculations():
    engine = StrategyEngine(
        symbol="MESZ6",
        default_qty=1,
        bracket={"sl": 40, "tp": 80, "tick_size": 0.25},
    )

    # 1. NY RTH: 10:00 AM ET = 14:00 UTC
    rth_ts = pd.to_datetime("2026-09-29 14:00:00", utc=True)
    assert engine._compute_session_regime(rth_ts) == "NY RTH"

    # 2. NY PRE: 08:30 AM ET = 12:30 UTC
    pre_ts = pd.to_datetime("2026-09-29 12:30:00", utc=True)
    assert engine._compute_session_regime(pre_ts) == "NY PRE"

    # 3. ASIAN: 20:00 ET (8 PM ET) = 00:00 UTC next day
    asian_ts = pd.to_datetime("2026-09-30 00:00:00", utc=True)
    assert engine._compute_session_regime(asian_ts) == "ASIAN"

    # 4. GLOBEX: 04:00 AM ET = 08:00 UTC
    globex_ts = pd.to_datetime("2026-09-29 08:00:00", utc=True)
    assert engine._compute_session_regime(globex_ts) == "GLOBEX"

    # 5. Synthetic RangeIndex or 0 should NOT evaluate as ASIAN
    assert engine._compute_session_regime(0) == "UNKNOWN"
    assert engine._compute_session_regime(100) == "UNKNOWN"

    # 6. Numeric epoch in seconds
    # 2026-09-29 14:00:00 UTC is ~1790690400 s
    epoch_rth_s = 1790690400
    assert engine._compute_session_regime(epoch_rth_s) == "NY RTH"

    # 7. Numeric epoch in milliseconds
    epoch_rth_ms = 1790690400000
    assert engine._compute_session_regime(epoch_rth_ms) == "NY RTH"


def test_orb_vwap_session_regime():
    strat = ORBVWAPStrategy("MESZ6", {"asian_enabled": True, "tick_size": 0.25})

    # NY: 10:00 AM ET
    row_ny = {"timestamp": "2026-09-29 14:00:00Z"}
    assert strat._get_session_regime(row_ny) == "NY"

    # Asian: 20:00 ET
    row_asian = {"timestamp": "2026-09-30 00:00:00Z"}
    assert strat._get_session_regime(row_asian) == "ASIAN"

    # Synthetic / 0
    assert strat._get_session_regime({"timestamp": 0}) == "OUT"


def test_mtf_trend_scalper_session_regime():
    strat = MTFTrendScalperStrategy("MNQU6")

    df = pd.DataFrame([{"timestamp": "2026-09-29 14:00:00Z", "close": 20000.0}])
    snap = strat.get_snapshot(0, df, df)
    assert snap["session_regime"] == "NY RTH"

    # Synthetic / 0
    df_zero = pd.DataFrame([{"timestamp": 0, "close": 20000.0}])
    snap_zero = strat.get_snapshot(0, df_zero, df_zero)
    assert snap_zero["session_regime"] == "UNKNOWN"


def test_price_action_bs_session_regime():
    strat = PriceActionBSStrategy("MESZ6")

    # NY: 10:00 AM ET = 14:00 UTC
    row_ny = {"timestamp": "2026-09-29 14:00:00Z"}
    assert strat._get_session_regime(row_ny) == "NY"

    # Asian: 20:00 ET = 00:00 UTC next day
    row_asian = {"timestamp": "2026-09-30 00:00:00Z"}
    assert strat._get_session_regime(row_asian) == "ASIAN"

    # Synthetic / 0
    assert strat._get_session_regime({"timestamp": 0}) == "OUT"
