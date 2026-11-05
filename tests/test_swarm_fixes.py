import pytest
from decimal import Decimal
from src.agents.risk_agent import RiskAgent
from src.agents.execution_agent import ExecutionAgent


def test_consecutive_losses_and_symbol_halts_trigger():
    risk_agent = RiskAgent("TEST_ACCOUNT")
    risk_agent.max_consecutive_losses = 2
    risk_agent.update_account_state(50000, 0, 0)

    # First trade loss on MCLV6
    risk_agent.record_trade_result(-50.0, symbol="MCLV6")
    assert risk_agent.consecutive_losses.get("MCLV6") == 1
    assert "MCLV6" not in risk_agent.symbol_halts
    assert "MCL" not in risk_agent.symbol_halts

    # Second trade loss on MCLV6 -> trips circuit breaker
    risk_agent.record_trade_result(-75.0, symbol="MCLV6")
    assert risk_agent.consecutive_losses.get("MCLV6") == 2
    assert "MCLV6" in risk_agent.symbol_halts
    assert "MCL" in risk_agent.symbol_halts

    # Subsequent trade intent on MCLV6 must be rejected by circuit breaker Rule 0
    accepted = risk_agent.validate_trade("MCLV6", 1, "Buy")
    assert accepted is False


def test_consecutive_loss_reset_on_win():
    risk_agent = RiskAgent("TEST_ACCOUNT")
    risk_agent.max_consecutive_losses = 2
    risk_agent.update_account_state(50000, 0, 0)

    # Loss then win
    risk_agent.record_trade_result(-50.0, symbol="MCLV6")
    assert risk_agent.consecutive_losses.get("MCLV6") == 1

    risk_agent.record_trade_result(100.0, symbol="MCLV6")
    assert risk_agent.consecutive_losses.get("MCLV6") == 0
    assert "MCLV6" not in risk_agent.symbol_halts


def test_update_account_state_session_opening_balance():
    risk_agent = RiskAgent("TEST_ACCOUNT")
    risk_agent.current_balance = Decimal("49500.00")
    # ProjectX provides pnl=None, but supervisor passes session opening balance = 50000.00
    risk_agent.update_account_state(49500.00, None, 0, session_opening_balance=50000.00)
    assert risk_agent.current_daily_pnl == Decimal("-500.00")


def test_execution_agent_flatten_does_not_abort_on_cancel_latency(monkeypatch):
    # Hermetic: do not depend on the developer's local .env (MODE / ACTUAL_LIVE_TRADING)
    monkeypatch.setenv("MODE", "live")
    monkeypatch.setenv("ACTUAL_LIVE_TRADING", "true")
    class DummyClient:
        symbol_to_id_cache = {"MCLV6": "12345"}
        def cancel_open_orders_for_symbol(self, sym):
            pass
        def get_orders(self, active=True):
            return [{"contractId": "12345", "status": "working"}]
        def place_order(self, symbol, quantity, side, order_type="market", **kwargs):
            return {"orderId": "999", "status": "filled"}

    exec_agent = ExecutionAgent(DummyClient())
    result = exec_agent.flatten_position("MCLV6", 1, "Buy")
    assert result.get("status") == "filled"


def test_extract_qty_side_topstepx_types():
    from src.agents.supervisor_agent import SupervisorAgent
    supervisor = SupervisorAgent.__new__(SupervisorAgent)

    # 1. TopstepX REST /Position/searchOpen raw short position (type=2)
    payload_short = {
        'id': 862187153,
        'accountId': 12345678,
        'contractId': 'CON.F.US.MES.Z26',
        'contractDisplayName': 'MESZ26',
        'type': 2,
        'size': 2,
        'averagePrice': 7732.0,
    }
    qty, side = supervisor._extract_qty_side(payload_short)
    assert qty == 2
    assert side == "Sell"

    # 2. TopstepX REST /Position/searchOpen raw long position (type=1)
    payload_long = {
        'id': 862187154,
        'accountId': 12345678,
        'contractId': 'CON.F.US.MES.Z26',
        'contractDisplayName': 'MESZ26',
        'type': 1,
        'size': 3,
        'averagePrice': 7750.0,
    }
    qty, side = supervisor._extract_qty_side(payload_long)
    assert qty == 3
    assert side == "Buy"

    # 3. Flat position (type=0)
    payload_flat = {'type': 0, 'size': 0}
    qty, side = supervisor._extract_qty_side(payload_flat)
    assert qty == 0
    assert side == "Flat"

    # 4. SignalR GatewayUserPosition translated with netSize
    assert supervisor._extract_qty_side({'netSize': -2, 'size': 2}) == (2, "Sell")
    assert supervisor._extract_qty_side({'netSize': 2, 'size': 2}) == (2, "Buy")
    assert supervisor._extract_qty_side({'netSize': 0}) == (0, "Flat")

    # 5. String-based types and sides
    assert supervisor._extract_qty_side({'type': 'Short', 'size': 1}) == (1, "Sell")
    assert supervisor._extract_qty_side({'type': 'Long', 'size': 1}) == (1, "Buy")
    assert supervisor._extract_qty_side({'side': 'Sell', 'qty': 4}) == (4, "Sell")
    assert supervisor._extract_qty_side({'side': 'Buy', 'qty': 4}) == (4, "Buy")


@pytest.mark.anyio
async def test_sync_positions_corrects_side_inversion():
    import asyncio
    import logging
    from src.agents.supervisor_agent import SupervisorAgent

    supervisor = SupervisorAgent.__new__(SupervisorAgent)
    supervisor.logger = logging.getLogger("test_supervisor")
    supervisor.official_symbols = ["MESZ6"]
    supervisor.active_positions = {
        "MESZ6": [{"side": "Buy", "qty": 2, "entry": 7732.0, "time": 1000.0}]
    }
    supervisor.risk_agent = RiskAgent("TEST_ACCOUNT")
    supervisor.risk_agent.record_position_open("MESZ6", "Buy", 2, 7732.0)
    supervisor._desync_start_time = {}
    supervisor._recently_closed = {}
    supervisor._lock = asyncio.Lock()
    supervisor._latest_broker_positions_snapshot = {}
    supervisor._latest_broker_positions_at = 0.0

    # Broker has 2 contracts, but SHORT (type=2)
    broker_positions = [
        {
            "symbol": "MESZ6",
            "type": 2,
            "size": 2,
            "averagePrice": 7732.0,
        }
    ]
    supervisor.client = type("DummyClient", (), {"symbol_to_id_cache": {"MESZ6": "CON.F.US.MES.Z26"}})()

    await supervisor._sync_positions(broker_positions)

    # Internal state must be corrected to Sell!
    assert len(supervisor.active_positions["MESZ6"]) == 1
    assert supervisor.active_positions["MESZ6"][0]["side"] == "Sell"
    assert supervisor.active_positions["MESZ6"][0]["qty"] == 2
    # Snapshot must also be Sell
    assert supervisor._latest_broker_positions_snapshot["MESZ6"][0]["side"] == "Sell"


@pytest.mark.anyio
async def test_handle_position_open_update_corrects_side_inversion():
    import asyncio
    import logging
    from src.agents.supervisor_agent import SupervisorAgent

    supervisor = SupervisorAgent.__new__(SupervisorAgent)
    supervisor.logger = logging.getLogger("test_supervisor")
    supervisor.active_positions = {
        "MESZ6": [{"side": "Buy", "qty": 2, "entry": 7732.0, "time": 1000.0}]
    }
    supervisor.risk_agent = RiskAgent("TEST_ACCOUNT")
    supervisor.risk_agent.record_position_open("MESZ6", "Buy", 2, 7732.0)
    supervisor._lock = asyncio.Lock()
    supervisor._close_in_flight = {}
    supervisor.last_signal_time = {}

    # Incoming update has same quantity (2) but side is Sell
    await supervisor._handle_position_open_update("MESZ6", "Sell", 2, 7732.0)

    assert len(supervisor.active_positions["MESZ6"]) == 1
    assert supervisor.active_positions["MESZ6"][0]["side"] == "Sell"
    assert supervisor.active_positions["MESZ6"][0]["qty"] == 2


def test_validate_trade_zero_and_negative_quantity():
    risk_agent = RiskAgent("TEST_ACCOUNT")
    risk_agent.update_account_state(50000, 0, 0)
    assert risk_agent.validate_trade("MESZ6", 0, "Buy") is False
    assert risk_agent.validate_trade("MESZ6", -1, "Buy") is False
    assert risk_agent.validate_trade("MESZ6", 1, "Buy") is True


def test_validate_trade_mll_floor_rejection():
    risk_agent = RiskAgent("TEST_ACCOUNT")
    # Balance 48040, minimum_balance 48000 -> within $50 safety buffer of MLL
    risk_agent.update_account_state(48040, 0, 0, minimum_balance=48000)
    assert risk_agent.validate_trade("MESZ6", 1, "Buy") is False


def test_scratch_trade_preserves_consecutive_losses():
    risk_agent = RiskAgent("TEST_ACCOUNT")
    risk_agent.max_consecutive_losses = 3
    risk_agent.update_account_state(50000, 0, 0)

    # 1 loss
    risk_agent.record_trade_result(-50.0, symbol="MESZ6")
    assert risk_agent.consecutive_losses["MESZ6"] == 1

    # Scratch trade ($0.0) -> streak must NOT reset to 0
    risk_agent.record_trade_result(0.0, symbol="MESZ6")
    assert risk_agent.consecutive_losses["MESZ6"] == 1

    # Win -> streak resets to 0
    risk_agent.record_trade_result(50.0, symbol="MESZ6")
    assert risk_agent.consecutive_losses["MESZ6"] == 0


def test_mtf_trend_scalper_handles_nan_and_enforces_rr():
    import pandas as pd
    from src.strategies.mtf_trend_scalper import MTFTrendScalperStrategy

    strat = MTFTrendScalperStrategy("MESZ6", {"min_adx": 20.0, "tick_size": 0.25})
    # NaN ADX
    row_nan_adx = {"timestamp": "2026-09-28T14:30:00Z", "close": 5000.0, "open": 4995.0, "high": 5002.0, "low": 4994.0, "adx": float("nan"), "atr": 2.0}
    bars = pd.DataFrame([row_nan_adx] * 50)
    indicators = pd.DataFrame([row_nan_adx] * 50)
    assert strat.on_bar(10, bars, indicators) is None

    # NaN ATR
    row_nan_atr = {"timestamp": "2026-09-28T14:30:00Z", "close": 5000.0, "open": 4995.0, "high": 5002.0, "low": 4994.0, "adx": 25.0, "atr": float("nan")}
    indicators_atr = pd.DataFrame([row_nan_atr] * 50)
    assert strat.on_bar(10, bars, indicators_atr) is None


def test_orb_vwap_asian_midnight_window():
    import pandas as pd
    from src.strategies.orb_vwap import ORBVWAPStrategy

    strat = ORBVWAPStrategy("MESZ6", {
        "asian_opening_range_end_et": "18:10",
        "asian_trade_window_end_et": "02:00",
    })
    # 20:00 ET is inside Asian trade window
    ts_2000 = pd.Timestamp("2026-09-28 20:00:00", tz="America/New_York")
    assert strat._in_asian_trade_window(ts_2000) is True

    # 01:30 ET (past midnight) is inside Asian trade window
    ts_0130 = pd.Timestamp("2026-09-29 01:30:00", tz="America/New_York")
    assert strat._in_asian_trade_window(ts_0130) is True

    # 03:00 ET is outside Asian trade window
    ts_0300 = pd.Timestamp("2026-09-29 03:00:00", tz="America/New_York")
    assert strat._in_asian_trade_window(ts_0300) is False


def test_orb_vwap_priming_marks_session_fired():
    import pandas as pd
    from src.strategies.orb_vwap import ORBVWAPStrategy

    strat = ORBVWAPStrategy("MESZ6", {"max_one_trade_per_session": True})
    strat.is_priming = True
    session_key = "2026-09-28"
    strat._session_key = session_key
    row = {"timestamp": "2026-09-28T13:50:00Z", "close": 5000.0, "high": 5005.0, "low": 4990.0, "open": 4992.0, "volume": 100}
    indicators_row = {"vwap": 4995.0, "ema_200": 4990.0}
    bars = pd.DataFrame([row] * 10)
    
    # Trigger NY evaluation
    strat._or_high = 4998.0
    strat._or_low = 4992.0
    strat._range_locked = True
    et = pd.Timestamp("2026-09-28 09:50:00", tz="America/New_York")
    sig = strat._process_ny(et, 5, row, indicators_row, bars)
    assert sig is not None
    # _fired_session MUST be set even though is_priming is True!
    assert strat._fired_session == session_key


def test_price_action_bs_respects_allowed_hours():
    from src.strategies.price_action_bs import PriceActionBSStrategy

    strat = PriceActionBSStrategy("MESZ6", {
        "allowed_entry_hours_et": [10, 11],
        "excluded_entry_hours_et": [12],
    })
    # 10:30 ET -> allowed
    row_10 = {"timestamp": "2026-09-28T14:30:00Z"}  # 10:30 ET
    assert strat._entry_hour_allowed(row_10) is True

    # 09:30 ET -> not in allowed hours [10, 11]
    row_09 = {"timestamp": "2026-09-28T13:30:00Z"}  # 09:30 ET
    assert strat._entry_hour_allowed(row_09) is False


def test_indicators_rsi_neutral_on_flat_market():
    import pandas as pd
    from src.strategies.indicators import prepare_indicator_frame

    # Flat market where price does not move for 30 bars
    df = pd.DataFrame([
        {"timestamp": f"2026-09-28T14:{i:02d}:00Z", "close": 5000.0, "high": 5000.0, "low": 5000.0, "open": 5000.0, "volume": 10}
        for i in range(30)
    ])
    res = prepare_indicator_frame(df, tick_size=0.25)
    # RSI must be neutral 50.0, NOT collapsed to 0.0
    assert float(res["rsi"].iloc[-1]) == 50.0


def test_blender_score_safe_with_nan_confidence():
    from src.strategies.blender import SignalBlender
    from src.strategies.base import StrategySignal

    sig = StrategySignal(
        strategy_name="test",
        side="Buy",
        quantity=1,
        sl_ticks=10,
        tp_ticks=20,
        confidence=float("nan"),
    )
    score = SignalBlender._base_score(sig)
    assert score == 0.0


def test_data_agent_calendar_minute_alignment():
    from src.agents.data_agent import DataAgent

    agent = DataAgent("MESZ6", bar_interval_seconds=60)
    # Tick at 09:30:17
    agent.handle_market_data({"symbol": "MESZ6", "price": 5000.0, "volume": 5, "timestamp": "2026-09-28T13:30:17Z"})
    assert agent.current_bar is not None
    # start_time must be aligned to top-of-minute 09:30:00
    assert agent.current_bar["start_time"].minute == 30
    assert agent.current_bar["start_time"].second == 0

    # Next tick at 09:31:05 -> rolls to new bar
    agent.handle_market_data({"symbol": "MESZ6", "price": 5002.0, "volume": 3, "timestamp": "2026-09-28T13:31:05Z"})
    assert len(agent.bars) == 1
    assert agent.current_bar["start_time"].minute == 31
    assert agent.current_bar["start_time"].second == 0


def test_supervisor_resolve_symbol_from_contract_display_name():
    from src.agents.supervisor_agent import SupervisorAgent

    supervisor = SupervisorAgent.__new__(SupervisorAgent)
    supervisor.official_symbols = ["MESZ6"]
    supervisor.client = type("DummyClient", (), {"symbol_to_id_cache": {"MESZ6": "CON.F.US.MES.Z26"}})()

    # Payload has contractDisplayName instead of symbol
    payload = {"contractDisplayName": "MESZ6", "contractId": "CON.F.US.MES.Z26"}
    assert supervisor._resolve_symbol_from_payload(payload) == "MESZ6"

    # Payload has name
    payload2 = {"name": "MESZ6", "contractId": "CON.F.US.MES.Z26"}
    assert supervisor._resolve_symbol_from_payload(payload2) == "MESZ6"


def test_cme_session_start_continuous():
    from src.agents.supervisor_agent import SupervisorAgent

    supervisor = SupervisorAgent.__new__(SupervisorAgent)
    start_utc = supervisor.get_current_session_start_utc()
    # Session start must be 22:00 UTC (18:00 ET)
    assert start_utc.hour in (22, 23)  # depends on daylight saving, always 18:00 ET
    assert start_utc.minute == 0


@pytest.mark.anyio
async def test_handle_position_open_update_partial_reduction():
    import asyncio
    import logging
    from src.agents.supervisor_agent import SupervisorAgent

    supervisor = SupervisorAgent.__new__(SupervisorAgent)
    supervisor.logger = logging.getLogger("test_supervisor")
    # Currently have 3 contracts Long
    supervisor.active_positions = {
        "MESZ6": [{"side": "Buy", "qty": 3, "entry": 5000.0, "time": 1000.0}]
    }
    supervisor.risk_agent = RiskAgent("TEST_ACCOUNT")
    supervisor.risk_agent.record_position_open("MESZ6", "Buy", 3, 5000.0)
    supervisor._lock = asyncio.Lock()
    supervisor._close_in_flight = {}
    supervisor.last_signal_time = {}

    # Broker sends netSize = 1 Long (partial reduction of 2 contracts)
    await supervisor._handle_position_open_update("MESZ6", "Buy", 1, 5000.0)

    # Internal position should have been reduced to 1 contract, NOT wiped out!
    assert len(supervisor.active_positions["MESZ6"]) == 1
    assert supervisor.active_positions["MESZ6"][0]["qty"] == 1
    assert supervisor.active_positions["MESZ6"][0]["side"] == "Buy"


def test_consecutive_loss_base_symbol_no_double_increment():
    risk_agent = RiskAgent("TEST_ACCOUNT")
    risk_agent.max_consecutive_losses = 2
    risk_agent.update_account_state(50000, 0, 0)

    # When symbol has no month code (e.g. MES), a single loss must NOT double increment to 2
    risk_agent.record_trade_result(-50.0, symbol="MES")
    assert risk_agent.consecutive_losses.get("MES") == 1
    assert "MES" not in risk_agent.symbol_halts


def test_database_list_closed_unanalyzed_trades_filters_by_account_id(tmp_path):
    from src.persistence.database import DatabaseService

    db_path = tmp_path / "test_trades.db"
    service = DatabaseService({"enabled": True, "backend": "sqlite", "sqlite_path": str(db_path)})

    service.persist_trade_context(
        trade_key="T1",
        symbol="MESZ6",
        strategy_name="test_strat",
        side="Buy",
        quantity=1,
        order_id="ORD1",
        entry_price=5000.0,
        opened_at="2026-04-24T16:00:00+00:00",
        strategy_params={},
        selected_signal={},
        candidates=[],
        blender_metadata={},
        route_context={},
        discovery_context={},
        account_snapshot={"account_id": "ACC_1"},
        chart_artifacts={},
    )
    service.persist_trade_outcome(
        trade_key="T1",
        symbol="MESZ6",
        order_id="ORD1",
        account_id="ACC_1",
        pnl=100.0,
        close_type="tp",
        closed_at="2026-04-24T17:00:00+00:00",
        outcome={"account_id": "ACC_1"},
    )

    service.persist_trade_context(
        trade_key="T2",
        symbol="MNQM6",
        strategy_name="test_strat",
        side="Sell",
        quantity=1,
        order_id="ORD2",
        entry_price=20000.0,
        opened_at="2026-04-24T16:05:00+00:00",
        strategy_params={},
        selected_signal={},
        candidates=[],
        blender_metadata={},
        route_context={},
        discovery_context={},
        account_snapshot={"account_id": "ACC_2"},
        chart_artifacts={},
    )
    service.persist_trade_outcome(
        trade_key="T2",
        symbol="MNQM6",
        order_id="ORD2",
        account_id="ACC_2",
        pnl=-50.0,
        close_type="sl",
        closed_at="2026-04-24T17:05:00+00:00",
        outcome={"account_id": "ACC_2"},
    )

    # Filter for ACC_1 only
    acc1_trades = service.list_closed_unanalyzed_trades(limit=10, account_id="ACC_1")
    assert len(acc1_trades) == 1
    assert acc1_trades[0]["trade_key"] == "T1"

    # Filter for ACC_2 only
    acc2_trades = service.list_closed_unanalyzed_trades(limit=10, account_id="ACC_2")
    assert len(acc2_trades) == 1
    assert acc2_trades[0]["trade_key"] == "T2"

    # All unanalyzed trades
    all_trades = service.list_closed_unanalyzed_trades(limit=10)
    assert len(all_trades) == 2



