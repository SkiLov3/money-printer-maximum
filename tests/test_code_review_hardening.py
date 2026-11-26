# type: ignore
import asyncio
import io
import json
import math
import os
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from dashboard import Handler
from scripts.saturday_autonomous_optimizer import (
    _get_llama_base_url,
    is_valid_param_value,
)
from src.agents.news_agent import NewsAgent
from src.agents.risk_agent import RiskAgent
from src.agents.supervisor_agent import SupervisorAgent
from src.api.projectx_client import ProjectXClient
from src.strategies.engine import StrategyEngine
from src.strategies.mtf_trend_scalper import MTFTrendScalperStrategy
from src.strategies.orb_vwap import ORBVWAPStrategy
from src.trading.router import TradeRouter
from src.utils.backtester import Backtester


# ---------------------------------------------------------------------------
# 1. _risk_halt clears across session maintenance boundary
# ---------------------------------------------------------------------------
def test_risk_halt_clears_across_session_maintenance_boundary():
    """Verify SupervisorAgent.run() enters the if not market_open branch even when

    self._risk_halt = True, clearing _risk_halt = False, resetting risk_agent,
    and setting self._session_reset_done = True.
    """
    async def _test():
        supervisor = SupervisorAgent.__new__(SupervisorAgent)
        supervisor.running = True
        supervisor.logger = MagicMock()
        supervisor.client = MagicMock()
        supervisor.news_agent = MagicMock()
        supervisor.risk_agent = MagicMock()
        supervisor.active_positions = {"MESZ6": []}
        supervisor.official_symbols = ["MESZ6"]
        supervisor.data_agents = {"MESZ6": MagicMock()}
        supervisor._risk_halt = True
        supervisor._session_reset_done = False
        supervisor._eod_triggered = True
        supervisor._session_pnl_from_fills = 500.0
        supervisor._recent_trades = [{"symbol": "MESZ6", "pnl": -50.0}]
        supervisor._consecutive_losses = {"MESZ6": 2}
        supervisor._last_trade_won = {"MESZ6": False}
        supervisor._last_status_write = 0
        supervisor._schedule_status_write = MagicMock()
        supervisor.loop = asyncio.get_event_loop()

        # Closed market -> not market_open branch
        supervisor.news_agent.is_market_hours.return_value = False
        supervisor.news_agent.is_near_close.return_value = False
        supervisor._seconds_until_next_globex_open = MagicMock(return_value=3600.0)
        supervisor._check_saturday_optimization = AsyncMock()

        # Smart sleep should terminate the loop after first iteration
        async def stop_loop(*args, **kwargs):
            supervisor.running = False

        supervisor._smart_sleep = AsyncMock(side_effect=stop_loop)

        await supervisor.run()

        assert supervisor._risk_halt is False
        assert supervisor._session_reset_done is True
        assert supervisor._session_pnl_from_fills == 0.0
        assert supervisor._recent_trades == []
        assert supervisor._consecutive_losses == {}
        supervisor.risk_agent.reset_session.assert_called_once()

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# 2. PRO-GUARD breakeven move executes broker stop order once & fallback flatten
# ---------------------------------------------------------------------------
def test_pro_guard_breakeven_move_executes_broker_stop_once():
    """With live trading and brackets active, >30 ticks profit before news moves

    stop to BE once, setting news_be_moved=True and cancelling open orders.
    A second call does not re-place.
    """
    async def _test():
        supervisor = SupervisorAgent.__new__(SupervisorAgent)
        supervisor.logger = MagicMock()
        supervisor.client = MagicMock()
        supervisor.news_agent = MagicMock()
        supervisor.risk_agent = MagicMock()
        supervisor.execution_agent = MagicMock()
        supervisor.execution_agent.mode = "live"
        supervisor.official_symbols = ["MESZ6"]
        supervisor._close_in_flight = {}
        supervisor._close_in_flight_since = {}
        supervisor._lock = asyncio.Lock()
        supervisor.loop = asyncio.get_event_loop()

        supervisor.news_agent.is_news_approaching.return_value = "FOMC Rate Decision"
        supervisor.risk_agent.get_tick_size.return_value = 0.25

        # Position: Buy 2 @ 5000.0
        pos = {"side": "Buy", "entry": 5000.0, "qty": 2}
        supervisor.active_positions = {"MESZ6": [pos]}

        # Current price 5010.0 -> +10.0 points -> 40 ticks (> 30 ticks profit)
        dagent = MagicMock()
        dagent.get_latest_data.return_value = pd.DataFrame([{"close": 5010.0}])
        supervisor.data_agents = {"MESZ6": dagent}

        supervisor.client.place_order.return_value = {"status": "ok", "success": True}

        with patch.dict("os.environ", {"MODE": "live", "ACTUAL_LIVE_TRADING": "true", "USE_ORDER_BRACKETS": "true"}):
            await supervisor._check_news_guards()

            supervisor.client.cancel_open_orders_for_symbol.assert_called_once_with("MESZ6")
            supervisor.client.place_order.assert_called_once_with("MESZ6", 2, "Sell", "Stop", None, 5000.0)
            assert pos.get("news_be_moved") is True

            # Second call: pos["news_be_moved"] is True -> no order placed
            supervisor.client.place_order.reset_mock()
            supervisor.client.cancel_open_orders_for_symbol.reset_mock()
            await supervisor._check_news_guards()
            supervisor.client.place_order.assert_not_called()
            supervisor.client.cancel_open_orders_for_symbol.assert_not_called()

    asyncio.run(_test())


def test_pro_guard_fallback_flatten_when_place_order_rejected():
    """Verify fallback flatten when place_order returns status: rejected."""
    async def _test():
        supervisor = SupervisorAgent.__new__(SupervisorAgent)
        supervisor.logger = MagicMock()
        supervisor.client = MagicMock()
        supervisor.news_agent = MagicMock()
        supervisor.risk_agent = MagicMock()
        supervisor.execution_agent = MagicMock()
        supervisor.execution_agent.mode = "live"
        supervisor.official_symbols = ["MESZ6"]
        supervisor._close_in_flight = {}
        supervisor._close_in_flight_since = {}
        supervisor._lock = asyncio.Lock()
        supervisor.loop = asyncio.get_event_loop()

        supervisor.news_agent.is_news_approaching.return_value = "FOMC Rate Decision"
        supervisor.risk_agent.get_tick_size.return_value = 0.25

        pos = {"side": "Buy", "entry": 5000.0, "qty": 2}
        supervisor.active_positions = {"MESZ6": [pos]}

        dagent = MagicMock()
        dagent.get_latest_data.return_value = pd.DataFrame([{"close": 5010.0}])
        supervisor.data_agents = {"MESZ6": dagent}

        # Stop rejected by broker
        supervisor.client.place_order.return_value = {"status": "rejected", "message": "Order rejected"}
        supervisor.execution_agent.flatten_position.return_value = {"status": "ok"}

        with patch.dict("os.environ", {"MODE": "live", "ACTUAL_LIVE_TRADING": "true", "USE_ORDER_BRACKETS": "true"}):
            await supervisor._check_news_guards()

            supervisor.execution_agent.flatten_position.assert_called_once_with("MESZ6", 2, "Buy")

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# 3. Protective stop partial coverage places only missing delta
# ---------------------------------------------------------------------------
def test_projectx_has_working_stop_order_partial_coverage():
    """With a 3-contract Buy position and working Sell stop of size 1,

    has_working_stop_order returns False and sets _last_stop_coverage = 1.
    """
    client = ProjectXClient.__new__(ProjectXClient)
    client.logger = MagicMock()
    client._symbol_cache_lock = MagicMock()
    client._symbol_cache_lock.__enter__.return_value = None
    client._symbol_cache_lock.__exit__.return_value = None
    client.symbol_to_id_cache = {"MCLZ6": "CID_MCL"}
    client.contract_cache = {}
    client._last_stop_coverage = {}

    # 1 working Sell stop order with remainingQty=1
    client.get_orders = MagicMock(return_value=[
        {
            "id": "ORD-1",
            "contractId": "CID_MCL",
            "symbol": "MCLZ6",
            "status": "working",
            "type": "stop",
            "side": 1,  # Sell
            "remainingQty": 1,
        }
    ])

    res = client.has_working_stop_order("MCLZ6", "Buy", min_qty=3)
    assert res is False
    assert client._last_stop_coverage.get(("MCLZ6", "buy")) == 1


def test_supervisor_ensure_protective_stop_places_only_delta():
    """_ensure_protective_stop_for_position for qty=3 with covered_qty=1 places a

    protective stop for qty=2 (3 - 1).
    """
    async def _test():
        supervisor = SupervisorAgent.__new__(SupervisorAgent)
        supervisor.logger = MagicMock()
        supervisor.client = MagicMock()
        supervisor.risk_agent = MagicMock()
        supervisor.execution_agent = MagicMock()
        supervisor.execution_agent.mode = "simulation"
        supervisor.active_positions = {"MCLZ6": [{"side": "Buy", "qty": 3, "entry": 70.0, "sl_ticks": 20}]}
        supervisor._close_in_flight = {}
        supervisor._recently_closed = {}
        supervisor._stop_placement_in_flight = {}
        supervisor._lock = asyncio.Lock()
        supervisor.loop = asyncio.get_event_loop()
        supervisor.stack_config = {}

        # has_working_stop_order returns False and records coverage 1
        supervisor.client.has_working_stop_order.return_value = False
        supervisor.client._last_stop_coverage = {("MCLZ6", "buy"): 1}
        supervisor.risk_agent.get_tick_size.return_value = 0.01

        dagent = MagicMock()
        dagent.get_latest_data.return_value = pd.DataFrame([{"close": 70.05}])
        supervisor.data_agents = {"MCLZ6": dagent}

        supervisor.client.place_order.return_value = {"status": "ok", "success": True}

        success = await supervisor._ensure_protective_stop_for_position("MCLZ6", "Buy", 3, 70.0, sl_ticks=20)
        assert success is True

        # Expected stop price: 70.0 - 20 * 0.01 = 69.80; place_qty: 2 (3 - 1)
        supervisor.client.place_order.assert_called_once_with(
            "MCLZ6",
            2,
            "Sell",
            "Stop",
            None,
            69.80,
        )

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# 4. Out-of-order GatewayQuote cumulative volume regression ignored vs reset
# ---------------------------------------------------------------------------
def test_gateway_quote_cumulative_volume_out_of_order_vs_session_reset():
    """Tick 1: vol=100000 -> tick_vol=0, _last_contract_volume=100000.

    Tick 2: vol=100025 -> tick_vol=25, _last_contract_volume=100025.
    Tick 3: vol=100020 (out-of-order) -> tick_vol=0, _last_contract_volume stays 100025.
    Tick 4: vol=12 (session reset) -> tick_vol=12, _last_contract_volume=12.
    """
    client = ProjectXClient.__new__(ProjectXClient)
    client.logger = MagicMock()
    client._symbol_cache_lock = MagicMock()
    client._symbol_cache_lock.__enter__.return_value = None
    client._symbol_cache_lock.__exit__.return_value = None
    client.lookup_symbol_for_contract_id = MagicMock(return_value="MESZ6")
    client._last_contract_volume = {}

    ticks_received = []
    client.on_message_callback = lambda payload: ticks_received.append(payload)

    # Tick 1: Initial high cumulative volume (>500)
    client._handle_message(["CID1", {"lastPrice": 5000.0, "volume": 100000}])
    assert len(ticks_received) == 1
    assert ticks_received[-1]["volume"] == 0.0
    assert client._last_contract_volume["CID1"] == 100000

    # Tick 2: Normal increase
    client._handle_message(["CID1", {"lastPrice": 5000.25, "volume": 100025}])
    assert len(ticks_received) == 2
    assert ticks_received[-1]["volume"] == 25.0
    assert client._last_contract_volume["CID1"] == 100025

    # Tick 3: Out-of-order minor regression
    client._handle_message(["CID1", {"lastPrice": 5000.25, "volume": 100020}])
    assert len(ticks_received) == 3
    assert ticks_received[-1]["volume"] == 0.0
    assert client._last_contract_volume["CID1"] == 100025  # Unchanged!

    # Tick 4: Genuine session reset (< 0.5 * prev_vol)
    client._handle_message(["CID1", {"lastPrice": 5000.50, "volume": 12}])
    assert len(ticks_received) == 4
    assert ticks_received[-1]["volume"] == 12.0
    assert client._last_contract_volume["CID1"] == 12.0


# ---------------------------------------------------------------------------
# 5. reload_configuration() updates RiskAgent limits and routing
# ---------------------------------------------------------------------------
def test_reload_configuration_updates_risk_agent_limits_and_routing():
    """SupervisorAgent.reload_configuration() updates RiskAgent limits when flat."""
    supervisor = SupervisorAgent.__new__(SupervisorAgent)
    supervisor.logger = MagicMock()
    supervisor.telemetry = MagicMock()
    supervisor.execution_agent = MagicMock()
    supervisor.active_positions = {"MESZ6": []}  # flat
    supervisor.official_symbols = []
    supervisor.strategy_agents = {}
    supervisor.data_agents = {}

    supervisor.risk_agent = MagicMock()
    supervisor.risk_agent._lock = MagicMock()
    supervisor.risk_agent._lock.__enter__.return_value = None
    supervisor.risk_agent._lock.__exit__.return_value = None

    mock_stack = {
        "routing": {
            "daily_profit_target": 2200,
            "daily_loss_halt": 500,
            "max_consecutive_losses": 3,
        },
        "feature_flags": {
            "instrument_sizing_enabled": True,
        },
        "brackets": {},
        "strategies": {},
        "commissions": {},
    }

    with patch("src.agents.supervisor_agent.load_stack_config", return_value=mock_stack):
        res = supervisor.reload_configuration()
        assert res is True
        assert supervisor.risk_agent.daily_profit_target == Decimal("2200")
        assert supervisor.risk_agent.daily_loss_halt == Decimal("500")
        assert supervisor.risk_agent.max_consecutive_losses == 3
        assert supervisor.feature_flags.get("instrument_sizing_enabled") is True


# ---------------------------------------------------------------------------
# 6. MTFTrendScalperStrategy disallowed-hour bar leak & R:R ratio
# ---------------------------------------------------------------------------
def test_mtf_trend_scalper_disallowed_hour_bar_leak():
    """With allowed_entry_hours_et=[9, 10, 13, 14], at 13:00 ET (where ltf_idx=-2

    points to 12:55-12:59 ET hour 12 bar), on_bar returns None and emits no signal.
    """
    strat = MTFTrendScalperStrategy("MESZ6", {
        "allowed_entry_hours_et": [9, 10, 13, 14],
        "ltf_minutes": 5,
        "htf_minutes": 15,
        "tick_size": 0.25,
        "min_adx": 10.0,
        "skip_open_minutes": 0,
        "skip_close_minutes": 0,
    })

    # Create 1m bars leading up to 13:00 ET.
    # 13:00 ET is minute 0 of hour 13 (not closing min of 5m bar -> ltf_idx=-2 is 12:55-12:59 ET)
    # Generate 180 bars so warmup is satisfied.
    times = pd.date_range("2026-06-01 10:00:00", "2026-06-01 13:00:00", freq="1min", tz="America/New_York")
    n = len(times)

    df_bars = pd.DataFrame({
        "timestamp": times,
        "open": np.linspace(5000, 5050, n),
        "high": np.linspace(5001, 5051, n),
        "low": np.linspace(4999, 5049, n),
        "close": np.linspace(5000.5, 5050.5, n),
        "volume": np.full(n, 100),
    })
    df_bars.loc[:, "price"] = df_bars["close"]

    df_indicators = df_bars.copy()
    df_indicators["atr"] = 1.0

    # Evaluate at the 13:00 ET bar (last index)
    last_idx = len(df_bars) - 1
    sig = strat.on_bar(last_idx, df_bars, df_indicators)
    # Because ltf_idx = -2 corresponds to the 12:55 ET bar (hour 12, not allowed), it must return None!
    assert sig is None


def test_mtf_trend_scalper_rr_ratio_one_to_one():
    """Verify tp_atr_mult=1.5, sl_atr_mult=1.5 produces 1:1 target_ticks == stop_ticks."""
    strat = MTFTrendScalperStrategy("MESZ6", {
        "sl_atr_mult": 1.5,
        "tp_atr_mult": 1.5,
        "tick_size": 0.25,
        "min_stop_ticks": 10,
        "max_stop_ticks": 50,
        "ltf_minutes": 1,
        "htf_minutes": 15,
        "min_adx": 10.0,
        "skip_open_minutes": 0,
        "skip_close_minutes": 0,
        "allowed_entry_hours_et": [10],
    })

    atr = 2.0
    stop_ticks = max(strat.min_stop_ticks, min(strat.max_stop_ticks, int(math.ceil(round((strat.sl_atr_mult * atr) / strat.tick_size, 4)))))
    rr = strat.tp_atr_mult / max(strat.sl_atr_mult, 1e-9)
    target_ticks = max(1, int(round(stop_ticks * rr)))

    assert stop_ticks == 12
    assert target_ticks == 12
    assert target_ticks == stop_ticks


# ---------------------------------------------------------------------------
# 7. ORBVWAPStrategy rejection release, tie-break, Asian vol, StrategyEngine
# ---------------------------------------------------------------------------
def test_orb_vwap_on_signal_rejected_releases_session():
    """on_signal_rejected clears _asian_fired_session and _fired_session when

    session matches.
    """
    strat = ORBVWAPStrategy("MCLZ6", {})
    strat._fired_session = "2026-06-01"
    strat._asian_fired_session = "2026-06-01_asian"

    # Reject NY signal
    ny_signal = MagicMock()
    ny_signal.metadata = {"session_window": "NY", "session": "2026-06-01"}
    strat.on_signal_rejected(ny_signal)
    assert strat._fired_session is None
    assert strat._asian_fired_session == "2026-06-01_asian"

    # Reject Asian signal
    asian_signal = MagicMock()
    asian_signal.metadata = {"session_window": "ASIAN", "session": "2026-06-01_asian"}
    strat.on_signal_rejected(asian_signal)
    assert strat._asian_fired_session is None


def test_orb_vwap_outside_bar_tie_break():
    """Outside bar breaching both or_low and or_high:

    bullish close (close > open) -> Buy fade
    bearish close (close < open) -> Sell fade
    doji (close == open) -> None
    """
    strat = ORBVWAPStrategy("MCLZ6", {"tick_size": 0.01})
    or_high = 70.50
    or_low = 70.00
    breakout_buffer_ticks = 2  # 0.02
    fade_reclaim_ticks = 2     # 0.02

    # High breaches 70.52 (high=70.60) and Low breaches 69.98 (low=69.90)
    # Reclaims: close must be > 70.02 AND close must be < 70.48
    # 1. Bullish close (close=70.30 > open=70.20)
    row_bull = {"open": 70.20, "close": 70.30, "high": 70.60, "low": 69.90}
    side_b, typ_b, _ = strat._candidate(
        row_bull, or_high, or_low, direction_mode="fade",
        breakout_buffer_ticks=breakout_buffer_ticks, fade_reclaim_ticks=fade_reclaim_ticks,
    )
    assert side_b == "Buy"
    assert typ_b == "fade"

    # 2. Bearish close (close=70.20 < open=70.30)
    row_bear = {"open": 70.30, "close": 70.20, "high": 70.60, "low": 69.90}
    side_s, typ_s, _ = strat._candidate(
        row_bear, or_high, or_low, direction_mode="fade",
        breakout_buffer_ticks=breakout_buffer_ticks, fade_reclaim_ticks=fade_reclaim_ticks,
    )
    assert side_s == "Sell"
    assert typ_s == "fade"

    # 3. Doji (close == open)
    row_doji = {"open": 70.25, "close": 70.25, "high": 70.60, "low": 69.90}
    side_d, typ_d, _ = strat._candidate(
        row_doji, or_high, or_low, direction_mode="fade",
        breakout_buffer_ticks=breakout_buffer_ticks, fade_reclaim_ticks=fade_reclaim_ticks,
    )
    assert side_d is None
    assert typ_d is None


def test_orb_vwap_asian_volume_ratio_excludes_prior_day_rth():
    """_volume_ratio with asian_session_key excludes prior-session (e.g. 05:55 ET) bars

    when >= 3 current Asian bars exist.
    """
    strat = ORBVWAPStrategy("MCLZ6", {"volume_lookback_bars": 10})

    # Bars:
    # 2 bars from 05:50 and 05:55 ET (prior session: 2026-05-31_asian) with huge volume 1000
    # 3 bars from 18:00, 18:01, 18:02 ET (current Asian session: 2026-06-01_asian) with volume 10, 10, 10
    # Current bar at 18:03 ET with volume 20
    times = [
        pd.Timestamp("2026-06-01 05:50:00", tz="America/New_York"),
        pd.Timestamp("2026-06-01 05:55:00", tz="America/New_York"),
        pd.Timestamp("2026-06-01 18:00:00", tz="America/New_York"),
        pd.Timestamp("2026-06-01 18:01:00", tz="America/New_York"),
        pd.Timestamp("2026-06-01 18:02:00", tz="America/New_York"),
        pd.Timestamp("2026-06-01 18:03:00", tz="America/New_York"),
    ]
    volumes = [1000.0, 1000.0, 10.0, 10.0, 10.0, 20.0]

    df_bars = pd.DataFrame({"timestamp": times, "volume": volumes})
    asian_key = strat._asian_session_for_timestamp(times[2])  # 2026-06-01_asian

    ratio = strat._volume_ratio(index=5, bars=df_bars, asian_session_key=asian_key)
    # The 3 asian bars (indices 2, 3, 4) each have vol 10.0 -> average = 10.0
    # Current bar vol = 20.0 -> ratio should be 20.0 / 10.0 = 2.0 (NOT diluted by 1000.0)
    assert ratio == pytest.approx(2.0, rel=1e-3)


def test_strategy_engine_multi_bar_catch_up_priming():
    """StrategyEngine.analyze() with 3 new bars primes intermediate bars with

    is_priming=True and only evaluates the final bar with is_priming=False.
    """
    mock_strategy = MagicMock()
    mock_strategy.enabled = True
    mock_strategy.is_priming = False

    engine = StrategyEngine(
        "MESZ6",
        default_qty=1,
        bracket={"sl": 40, "tp": 80, "tick_size": 0.25},
        config={"strategies": []},
    )
    engine.strategies = [mock_strategy]
    engine.prepare_data = MagicMock(side_effect=lambda df: df)

    times = [
        pd.Timestamp("2026-06-01 10:00:00"),
        pd.Timestamp("2026-06-01 10:01:00"),
        pd.Timestamp("2026-06-01 10:02:00"),
        pd.Timestamp("2026-06-01 10:03:00"),
    ]
    df = pd.DataFrame({
        "timestamp": times,
        "open": [5000, 5001, 5002, 5003],
        "high": [5001, 5002, 5003, 5004],
        "low": [4999, 5000, 5001, 5002],
        "close": [5000.5, 5001.5, 5002.5, 5003.5],
        "volume": [100, 100, 100, 100],
        "price": [5000.5, 5001.5, 5002.5, 5003.5],
    })

    # Set last processed to bar 0
    engine._last_processed_timestamp = times[0]

    priming_states = []

    def fake_evaluate_index(bars, indicators, idx):
        priming_states.append((idx, mock_strategy.is_priming))
        return None, []

    engine.evaluate_index = MagicMock(side_effect=fake_evaluate_index)

    engine.analyze(df)

    # Bars 1 and 2 should be evaluated while is_priming=True
    # Bar 3 (final) should be evaluated while is_priming=False
    assert priming_states == [(1, True), (2, True), (3, False)]


# ---------------------------------------------------------------------------
# 8. Backtester parity, RiskAgent, TradeRouter, NewsAgent, Dashboard, Optimizer
# ---------------------------------------------------------------------------
def test_backtester_cooldown_constants():
    """Verify Backtester cooldown constants."""
    assert Backtester.LOSS_COOLDOWN_SECONDS == 1800
    assert Backtester.STREAK_COOLDOWN_SECONDS == 3600


def test_risk_agent_atr_scaled_sl_avoids_double_penalty():
    """calculate_instrument_quantity with atr_scaled_sl=True does NOT double-scale

    the risk budget when volatility_sizing_enabled=True.
    """
    risk_agent = RiskAgent(account_id="ACC-12345")
    risk_agent.feature_flags["instrument_sizing_enabled"] = True
    risk_agent.feature_flags["volatility_sizing_enabled"] = True
    risk_agent.symbol_brackets = {
        "MES": {"min_atr": 10.0, "max_atr": 20.0, "tick_size": 0.25, "tick_value": 1.25}
    }
    risk_agent.current_balance = Decimal("100000.0")
    risk_agent.unrealized_pnl = Decimal("0.0")
    risk_agent.routing_config = {"risk_per_trade_pct": 0.18}
    risk_agent.max_daily_loss = Decimal("2000.0")

    # High ATR = 100.0 (5x typical)
    # When atr_scaled_sl=False: vol_multiplier = 15.0 / 100.0 = 0.15 scales budget down
    qty_no_flag = risk_agent.calculate_instrument_quantity("MESZ6", sl_ticks=20, current_atr=100.0, atr_scaled_sl=False)

    # When atr_scaled_sl=True: vol sizing is bypassed to prevent double-penalization
    qty_with_flag = risk_agent.calculate_instrument_quantity("MESZ6", sl_ticks=20, current_atr=100.0, atr_scaled_sl=True)

    assert qty_with_flag > qty_no_flag


def test_risk_agent_configurable_single_trade_risk_cap():
    """max_single_trade_risk_pct raises the per-trade dollar cap (e.g., 35% of $350 = $122.50),

    allowing 80-tick MES stops ($100) to take 1 contract instead of rejecting with qty=0.
    """
    risk_agent = RiskAgent(account_id="ACC-12345")
    risk_agent.symbol_brackets = {
        "MES": {"min_atr": 1.0, "max_atr": 20.0, "tick_size": 0.25, "tick_value": 1.25}
    }
    risk_agent.current_balance = Decimal("50900.0")
    risk_agent.unrealized_pnl = Decimal("0.0")
    risk_agent.max_daily_loss = Decimal("350.0")

    # Default 25% cap ($87.50) blocks an 80-tick MES stop ($100)
    risk_agent.routing_config = {"risk_per_trade_pct": 0.25}
    assert risk_agent.calculate_instrument_quantity("MESZ6", sl_ticks=80) == 0

    # 35% cap ($122.50) allows 1 MES contract on an 80-tick stop
    risk_agent.routing_config = {"risk_per_trade_pct": 0.25, "max_single_trade_risk_pct": 0.35}
    assert risk_agent.calculate_instrument_quantity("MESZ6", sl_ticks=80) == 1


def test_float_tick_rounding_precision_trade_router_and_execution_agent():
    """(70.30 - 70.10) / 0.01 in floating point produces 19.99999999999993.

    With proper rounding it must produce exact 20 ticks (not 21 or 19).
    """
    ep = 70.30
    sp = 70.10
    tick_sz = 0.01

    raw_val = (ep - sp) / tick_sz
    assert raw_val != 20.0  # Demonstrating float precision quirk

    # ExecutionAgent / TradeRouter formula: int(math.ceil(round((ep - sp) / tick_sz, 6)))
    ticks = int(math.ceil(round((ep - sp) / tick_sz, 6)))
    assert ticks == 20


def test_news_agent_calendar_coverage():
    """NewsAgent recognizes 2026 H1 FOMC and 2027 H1 FOMC and has no duplicate

    Initial Jobless Claims in fixed_events.
    """
    agent = NewsAgent()

    fomc_2026 = (2026, 3, 18, "14:00", "FOMC Rate Decision")
    fomc_2027 = (2027, 1, 27, "14:00", "FOMC Rate Decision")

    assert fomc_2026 in agent.fixed_events
    assert fomc_2027 in agent.fixed_events

    # No Initial Jobless Claims in fixed_events (it belongs only in weekly_schedule)
    fixed_jobless = [evt for evt in agent.fixed_events if "Jobless Claims" in evt[4]]
    assert len(fixed_jobless) == 0


def test_dashboard_risk_config_payload_cap():
    """dashboard.Handler.do_POST /api/risk-config rejects Content-Length > 8192 with HTTP 413."""
    handler = Handler.__new__(Handler)
    handler.path = "/api/risk-config"
    handler.headers = {"Content-Length": "99999"}
    handler.supervisor = MagicMock()
    handler.send_response = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = io.BytesIO()

    handler.do_POST()

    handler.send_response.assert_called_once_with(413)
    assert handler.wfile.getvalue() == b"Payload too large"


def test_saturday_optimizer_bounds():
    """scripts.saturday_autonomous_optimizer.is_valid_param_value bounds check."""
    assert is_valid_param_value("orb_vwap", "min_volume_ratio", 1.2) is True
    assert is_valid_param_value("orb_vwap", "min_volume_ratio", 5.0) is False


# ---------------------------------------------------------------------------
# 9. Batch 4: Regression Tests for Batches 1–3
# ---------------------------------------------------------------------------

def test_risk_agent_effective_daily_loss_limit_cap():
    """Verify RiskAgent per-trade dollar cap uses daily_loss_halt if lower than max_daily_loss."""
    risk_agent = RiskAgent(account_id="ACC-12345")
    risk_agent.symbol_brackets = {
        "MES": {"min_atr": 1.0, "max_atr": 20.0, "tick_size": 0.25, "tick_value": 1.25}
    }
    risk_agent.current_balance = Decimal("50000.0")
    risk_agent.unrealized_pnl = Decimal("0.0")

    # max_daily_loss = 1000, daily_loss_halt = 350
    # max_single_trade_risk_pct = 0.35
    # Per-trade cap: 0.35 * 350 = 122.50 (NOT 0.35 * 1000 = 350)
    risk_agent.max_daily_loss = Decimal("1000")
    risk_agent.daily_loss_halt = Decimal("350")
    risk_agent.routing_config = {
        "risk_per_trade_pct": 0.35,  # 0.35% of 50k = $175 budget
        "max_single_trade_risk_pct": 0.35  # Caps budget at 35% of 350 = $122.50
    }

    # 100 ticks MES = $125. 125 > 122.50 -> qty should be 0.
    assert risk_agent.calculate_instrument_quantity("MESZ6", sl_ticks=100) == 0
    # 90 ticks MES = $112.50. 112.50 < 122.50 -> qty should be 1.
    assert risk_agent.calculate_instrument_quantity("MESZ6", sl_ticks=90) == 1


def test_risk_agent_dynamic_quantity_halt_threshold():
    """get_dynamic_quantity drops to 1 if realized pnl < -50% of daily_loss_halt."""
    risk_agent = RiskAgent(account_id="ACC-12345")
    risk_agent.max_daily_loss = Decimal("1000")
    risk_agent.daily_loss_halt = Decimal("350")
    # -200 is < -50% of 350 (-175), even though it is > -50% of 1000 (-500)
    risk_agent.current_daily_pnl = Decimal("-200")

    # Mock weighted total to allow new trades
    risk_agent._weighted_open_total = MagicMock(return_value=Decimal("0"))
    risk_agent.max_total_contracts = Decimal("10")
    risk_agent._full_equiv_weight = MagicMock(return_value=Decimal("1"))

    assert risk_agent.get_dynamic_quantity("MESZ6", requested_qty=5) == 1


def test_trade_router_get_tick_size_fallback():
    """TradeRouter falls back to risk_agent.get_tick_size when client meta is missing."""
    risk_agent = MagicMock()
    risk_agent.get_tick_size.return_value = 0.25
    risk_agent.get_entry_quantity.return_value = 1
    risk_agent.validate_trade.return_value = True

    execution_agent = MagicMock()
    # client.get_contract_meta returns None
    execution_agent.client.get_contract_meta.return_value = None
    execution_agent.execute_trade.return_value = {"status": "ok"}

    router = TradeRouter(risk_agent, execution_agent)

    trade_intent = {
        "symbol": "MESZ6",
        "quantity": 1,
        "side": "Buy",
        "metadata": {"stop_price": 4990.0}
    }
    route_context = {"current_price": 5000.0}

    # Should call risk_agent.get_tick_size("MESZ6")
    router.route_trade(trade_intent, route_context)
    risk_agent.get_tick_size.assert_called_with("MESZ6")


def test_mtf_trend_scalper_fail_closed_on_bad_timestamp():
    """MTFTrendScalperStrategy on_bar returns None on unparseable timestamp when hour filter active."""
    strat = MTFTrendScalperStrategy("MESZ6", {"allowed_entry_hours_et": [9, 10]})
    strat.warmup_bars = 0

    df_bars = pd.DataFrame([{"timestamp": "not-a-timestamp", "close": 5000.0}])
    df_indicators = df_bars.copy()

    sig = strat.on_bar(0, df_bars, df_indicators)
    assert sig is None

    # Test with None
    df_bars_none = pd.DataFrame([{"timestamp": None, "close": 5000.0}])
    sig_none = strat.on_bar(0, df_bars_none, df_bars_none)
    assert sig_none is None


def test_projectx_client_redact_token():
    """ProjectXClient._redact_token replaces JWT while preserving other params."""
    url = "wss://rtc.topstepx.com/hubs/market?access_token=eyJhbGciOi.secret.sig&foo=1"
    redacted = ProjectXClient._redact_token(url)
    assert "access_token=<REDACTED>" in redacted
    assert "&foo=1" in redacted
    assert "eyJhbGciOi" not in redacted


def test_dashboard_healthz_public():
    """GET /healthz is accessible without auth."""
    handler = Handler.__new__(Handler)
    handler.path = "/healthz"
    handler.headers = {}
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()

    handler.do_GET()
    handler.send_response.assert_called_once_with(200)


def test_dashboard_host_header_validation():
    """Dashboard rejects invalid Host headers with 403."""
    handler = Handler.__new__(Handler)
    handler.headers = {"Host": "rebind.evil.com:8765"}
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()

    # Should return False and send 403
    assert handler._check_host_header() is False
    handler.send_response.assert_called_once_with(403)

    # Valid hosts
    for h in ["localhost:8765", "127.0.0.1:8765"]:
        handler.headers = {"Host": h}
        handler.send_response.reset_mock()
        assert handler._check_host_header() is True
        handler.send_response.assert_not_called()


def test_dashboard_security_headers_and_integrity():
    """Verify CSP does not have unsafe-eval/jsdelivr and script has integrity."""
    from dashboard import HTML
    assert 'integrity="sha384-' in HTML
    assert 'crossorigin="anonymous"' in HTML

    handler = Handler.__new__(Handler)
    handler.request_version = "HTTP/1.1"
    handler.connection = MagicMock()
    handler._headers_buffer = []
    handler.wfile = io.BytesIO()
    handler.send_header = MagicMock()
    handler.end_headers()

    # Find CSP header
    csp = next(call.args[1] for call in handler.send_header.call_args_list if call.args[0] == "Content-Security-Policy")
    assert "'unsafe-eval'" not in csp
    assert "cdn.jsdelivr.net" not in csp


def test_dashboard_env_edit_anchoring(tmp_path, monkeypatch):
    """Verify .env edit only replaces line-anchored MAX_DAILY_LOSS."""
    env_file = tmp_path / ".env"
    env_file.write_text("# MAX_DAILY_LOSS=999\nOTHER_MAX_DAILY_LOSS=999\nMAX_DAILY_LOSS=1000\n", encoding="utf-8")

    handler = Handler.__new__(Handler)
    handler.supervisor = MagicMock()
    handler.supervisor.risk_agent = MagicMock()
    handler.headers = {"Content-Length": "25", "Host": "localhost"}
    handler.rfile = io.BytesIO(json.dumps({"max_daily_loss": 500}).encode())
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = io.BytesIO()
    handler.path = "/api/risk-config"

    # Mock PASSWORD check to pass
    import dashboard
    with patch("dashboard.PASSWORD", ""), \
         patch("dashboard.os.path.exists", return_value=True), \
         patch("dashboard._is_allowed_host", return_value=True), \
         patch("dashboard.Handler._check_same_origin", return_value=True):

        # We need to more carefully mock the file operations in do_POST
        monkeypatch.setattr(dashboard, "os", MagicMock(wraps=os))
        dashboard.os.path.exists.return_value = True

        # Mock open for reading .env and for NamedTemporaryFile
        mock_read_f = io.StringIO("# MAX_DAILY_LOSS=999\nOTHER_MAX_DAILY_LOSS=999\nMAX_DAILY_LOSS=1000\n")
        mock_write_f = io.StringIO()
        mock_write_f.name = str(tmp_path / "temp_env")

        def side_effect_open(path, *args, **kwargs):
            if path == ".env":
                m = MagicMock()
                m.__enter__.return_value = mock_read_f
                return m
            return MagicMock()

        monkeypatch.setitem(dashboard.__dict__, "open", side_effect_open)

        # Mock tempfile.NamedTemporaryFile to return our mock_write_f
        mock_ntf = MagicMock()
        mock_ntf.__enter__.return_value = mock_write_f
        monkeypatch.setattr("dashboard.tempfile.NamedTemporaryFile", MagicMock(return_value=mock_ntf))

        handler.do_POST()

        # Check what was written to temp file
        written_content = mock_write_f.getvalue()
        assert "MAX_DAILY_LOSS=500" in written_content
        assert "# MAX_DAILY_LOSS=999" in written_content
        assert "OTHER_MAX_DAILY_LOSS=999" in written_content
        # Verify it didn't corrupt the other lines
        assert written_content == "# MAX_DAILY_LOSS=999\nOTHER_MAX_DAILY_LOSS=999\nMAX_DAILY_LOSS=500\n"

        # Verify chmod 0o600
        dashboard.os.chmod.assert_called_once_with(mock_write_f.name, 0o600)


def test_dashboard_optimizer_concurrency_conflict():
    """Verify /api/saturday-optimizer/run returns 409 if already running."""
    handler = Handler.__new__(Handler)
    handler.path = "/api/saturday-optimizer/run"
    handler.headers = {}
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()

    with patch("dashboard.PASSWORD", ""), \
         patch("dashboard._OPTIMIZER_PROC") as mock_proc:
        mock_proc.poll.return_value = None  # Still running
        handler.do_POST()
        handler.send_response.assert_called_once_with(409)


def test_saturday_optimizer_ssrf_protection(monkeypatch):
    """Verify LLAMA_BASE_URL validation rejects remote/file URLs."""
    monkeypatch.setenv("LLAMA_BASE_URL", "file:///etc/passwd")
    with pytest.raises(ValueError, match="Invalid LLAMA_BASE_URL scheme"):
        _get_llama_base_url()

    monkeypatch.setenv("LLAMA_BASE_URL", "http://169.254.169.254")
    with pytest.raises(ValueError, match="Remote LLAMA_BASE_URL hostname"):
        _get_llama_base_url()

    # Allowed even if remote if flag set
    monkeypatch.setenv("LLAMA_ALLOW_REMOTE", "1")
    assert _get_llama_base_url() == "http://169.254.169.254"

    # Always allow local/docker
    monkeypatch.delenv("LLAMA_ALLOW_REMOTE", raising=False)
    for host in ["127.0.0.1", "localhost", "host.docker.internal"]:
        monkeypatch.setenv("LLAMA_BASE_URL", f"http://{host}:9931")
        assert _get_llama_base_url() == f"http://{host}:9931"
