import pandas as pd

from src.strategies.engine import StrategyEngine
from src.strategies.otz_continuation import OTZContinuationStrategy


def _bars():
    rows = []
    start = pd.Timestamp("2026-04-01T13:00:00Z")
    for idx in range(310):
        price = 100.0
        rows.append(
            {
                "timestamp": start + pd.Timedelta(minutes=idx),
                "open": price,
                "high": price + 0.5,
                "low": price - 0.5,
                "close": price,
                "volume": 100,
                "price": price,
                "ema_21": 106.0,
                "ema_200": 108.0,
            }
        )
    # Completed 15m bearish impulse creates a 50%-body supply zone around 105-110.
    for idx in range(240, 255):
        rows[idx].update({"open": 110.0, "high": 111.0, "low": 99.0, "close": 100.0, "price": 100.0})
    rows[279].update({"open": 106.0, "high": 108.0, "low": 105.0, "close": 107.0, "price": 107.0})
    rows[280].update({"open": 107.25, "high": 108.5, "low": 102.5, "close": 103.0, "price": 103.0})
    return pd.DataFrame(rows)


def test_otz_continuation_emits_next_bar_sell_after_completed_zone_touch():
    bars = _bars()
    strategy = OTZContinuationStrategy(
        "MGCM6",
        {
            "tick_size": 0.1,
            "allowed_entry_hours_et": [13],
            "htf_minutes": 15,
            "htf_lookback": 16,
            "min_htf_impulse_atr": 1.2,
            "min_htf_body_pct": 0.55,
            "zone_mode": "impulse_body_50",
            "confirmation": "engulf_or_123",
            "min_confirmation_body_ticks": 4,
        },
    )

    signal = strategy.on_bar(280, bars, bars)

    assert signal is not None
    assert signal.side == "Sell"
    assert signal.strategy_name == "otz_continuation"
    assert signal.metadata["execution_model"] == "next_bar_open_required"
    assert signal.metadata["zone_low"] < signal.metadata["zone_high"]


def test_otz_engulfed_wick_orderblock_uses_wick_area_to_orderblock_body():
    htf = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-04-01T13:00:00Z") + pd.Timedelta(minutes=15 * idx),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "range": 2.0,
                "body": 0.0,
                "atr": 2.0,
            }
            for idx in range(20)
        ]
    )
    htf.loc[18, ["open", "high", "low", "close", "range", "body", "atr"]] = [100.0, 108.0, 99.0, 106.0, 9.0, 6.0, 4.0]
    htf.loc[19, ["open", "high", "low", "close", "range", "body", "atr"]] = [107.0, 109.0, 96.0, 98.0, 13.0, 9.0, 4.0]
    strategy = OTZContinuationStrategy(
        "MGCM6",
        {
            "tick_size": 0.1,
            "zone_mode": "engulfed_wick_ob",
            "htf_lookback": 20,
            "min_htf_impulse_atr": 1.2,
            "min_htf_body_pct": 0.55,
            "ob_wick_area_bars": 3,
        },
    )

    zone = strategy._active_zone(htf)

    assert zone is not None
    assert zone["side"] == "Sell"
    assert zone["zone_low"] == 106.0
    assert zone["zone_high"] == 108.0
    assert zone["orderblock_open"] == 100.0
    assert zone["orderblock_close"] == 106.0


def test_body_engulf_mode_accepts_body_engulf_without_full_wick_engulf():
    htf = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-04-01T13:00:00Z") + pd.Timedelta(minutes=15 * idx),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "range": 2.0,
                "body": 0.0,
                "atr": 2.0,
            }
            for idx in range(20)
        ]
    )
    htf.loc[18, ["open", "high", "low", "close", "range", "body", "atr"]] = [100.0, 108.0, 99.0, 106.0, 9.0, 6.0, 4.0]
    htf.loc[19, ["open", "high", "low", "close", "range", "body", "atr"]] = [106.5, 107.0, 100.0, 99.5, 7.0, 7.0, 4.0]
    body_strategy = OTZContinuationStrategy("MGCM6", {"tick_size": 0.1, "zone_mode": "engulfed_wick_ob", "htf_lookback": 20, "engulf_mode": "body"})
    full_strategy = OTZContinuationStrategy("MGCM6", {"tick_size": 0.1, "zone_mode": "engulfed_wick_ob", "htf_lookback": 20, "engulf_mode": "full"})

    assert body_strategy._active_zone(htf) is not None
    assert full_strategy._active_zone(htf) is None


def test_123_modes_can_require_close_break_or_allow_wick_break():
    bars = _bars()
    bars.loc[278, ["open", "high", "low", "close"]] = [100.0, 104.0, 99.0, 101.0]
    bars.loc[279, ["open", "high", "low", "close"]] = [101.0, 105.0, 100.0, 102.0]
    bars.loc[280, ["open", "high", "low", "close"]] = [102.0, 106.0, 101.0, 104.0]
    close_break = OTZContinuationStrategy("MGCM6", {"tick_size": 0.1, "confirmation": "123", "structure_123_mode": "close_break"})
    wick_break = OTZContinuationStrategy("MGCM6", {"tick_size": 0.1, "confirmation": "123", "structure_123_mode": "wick_break"})

    assert close_break._confirmation_ok("Buy", 280, bars) is False
    assert wick_break._confirmation_ok("Buy", 280, bars) is True


def test_fib382_plus_one_requires_body_beyond_shallow_retracement():
    bars = _bars()
    bars.loc[279, ["open", "high", "low", "close"]] = [103.0, 106.0, 100.0, 105.0]
    bars.loc[280, ["open", "high", "low", "close"]] = [105.0, 107.0, 104.0, 106.0]
    strategy = OTZContinuationStrategy("MGCM6", {"tick_size": 0.1, "confirmation": "fib382_plus1"})

    assert strategy._confirmation_ok("Buy", 280, bars) is True


def test_trigger_zone_modes_can_require_close_inside_or_allow_wick_touch():
    strategy_any_close = OTZContinuationStrategy("MGCM6", {"tick_size": 0.1, "trigger_zone_mode": "wick_touch_any_close"})
    strategy_close_inside = OTZContinuationStrategy("MGCM6", {"tick_size": 0.1, "trigger_zone_mode": "close_inside"})
    row = pd.Series({"high": 108.2, "low": 105.8, "close": 109.0})
    zone = {"zone_low": 106.0, "zone_high": 108.0}

    assert strategy_any_close._trigger_location_ok(row, zone) is True
    assert strategy_close_inside._trigger_location_ok(row, zone) is False


def test_opposite_otz_target_uses_nearest_opposing_zone():
    htf = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-04-01T13:00:00Z") + pd.Timedelta(minutes=15 * idx),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "range": 2.0,
                "body": 0.0,
                "atr": 2.0,
            }
            for idx in range(20)
        ]
    )
    htf.loc[18, ["open", "high", "low", "close", "range", "body", "atr"]] = [100.0, 108.0, 99.0, 106.0, 9.0, 6.0, 4.0]
    htf.loc[19, ["open", "high", "low", "close", "range", "body", "atr"]] = [107.0, 109.0, 96.0, 98.0, 13.0, 9.0, 4.0]
    strategy = OTZContinuationStrategy(
        "MGCM6",
        {
            "tick_size": 0.1,
            "zone_mode": "engulfed_wick_ob",
            "target_mode": "opposite_otz",
            "htf_lookback": 20,
            "engulf_mode": "body",
        },
    )
    row = pd.Series({"close": 103.0})

    assert strategy._target_price("Buy", row, htf, {}) == 106.0


def test_overlapping_otz_keeps_older_zone_until_invalidated():
    strategy = OTZContinuationStrategy("MGCM6", {"tick_size": 0.1, "skip_overlapping_otz": True})
    older = {"side": "Sell", "zone_low": 106.0, "zone_high": 108.0}
    newer = {"side": "Sell", "zone_low": 107.0, "zone_high": 109.0}
    opposite = {"side": "Buy", "zone_low": 107.0, "zone_high": 109.0}

    assert strategy._zones_overlap(newer, older) is True
    assert strategy._zones_overlap(opposite, older) is False


def test_otz_invalidates_after_close_through_zone():
    bars = _bars()
    bars.loc[275, ["open", "high", "low", "close", "price"]] = [107.0, 110.0, 106.0, 109.5, 109.5]
    strategy = OTZContinuationStrategy(
        "MGCM6",
        {
            "tick_size": 0.1,
            "zone_mode": "engulfed_wick_ob",
        },
    )
    zone = {
        "side": "Sell",
        "zone_low": 106.0,
        "zone_high": 108.0,
        "engulfing_time": str(bars.loc[270, "timestamp"]),
    }

    assert strategy._zone_invalidated_since(bars, 280, zone) is True


def test_strategy_engine_registry_builds_otz_continuation():
    engine = StrategyEngine(
        symbol="MGCM6",
        default_qty=1,
        bracket={"sl": 40, "tp": 80, "tick_size": 0.1},
        config={
            "strategies": [
                {
                    "name": "otz_continuation",
                    "enabled": True,
                    "priority": 100,
                    "params": {"tick_size": 0.1},
                }
            ]
        },
    )

    assert [strategy.name for strategy in engine.strategies] == ["otz_continuation"]
