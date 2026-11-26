from __future__ import annotations

import asyncio
import io
import json
import os
from email.message import Message
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import yaml

from dashboard import Handler
from scripts.saturday_autonomous_optimizer import (
    _QuotedTimeSafeDumper,
    evaluate_backtest_candidate,
    get_fallback_hypotheses,
    is_valid_param_value,
)
from src.agents.execution_agent import ExecutionAgent
from src.agents.supervisor_agent import SupervisorAgent
from src.api.projectx_client import ProjectXClient
from src.strategies.engine import StrategyEngine
from src.telemetry.store import TelemetryStore
from src.utils.backtester import Backtester


def test_param_bounds_validation():
    # Valid values
    assert is_valid_param_value("mtf_trend_scalper", "min_adx", 25.0)
    assert is_valid_param_value("mtf_trend_scalper", "sl_atr_mult", 1.5)
    assert is_valid_param_value("orb_vwap", "asian_fade_reclaim_ticks", 2)
    assert is_valid_param_value("mtf_trend_scalper", "allowed_entry_hours_et", [9, 10, 13, 14])

    # Out of bounds / invalid types
    assert not is_valid_param_value("mtf_trend_scalper", "min_adx", 5.0)   # below 15.0
    assert not is_valid_param_value("mtf_trend_scalper", "min_adx", 50.0)  # above 45.0
    assert not is_valid_param_value("mtf_trend_scalper", "sl_atr_mult", 0.5) # below 1.0
    assert not is_valid_param_value("orb_vwap", "asian_fade_reclaim_ticks", -1)
    assert not is_valid_param_value("orb_vwap", "asian_fade_reclaim_ticks", 15)
    # Boolean must be rejected even though isinstance(True, int) is True
    assert not is_valid_param_value("orb_vwap", "asian_fade_reclaim_ticks", True)
    assert not is_valid_param_value("mtf_trend_scalper", "min_adx", False)
    assert not is_valid_param_value("mtf_trend_scalper", "allowed_entry_hours_et", [9, 25])
    assert not is_valid_param_value("mtf_trend_scalper", "allowed_entry_hours_et", [])


def test_fallback_hypotheses_relative_generation():
    cfg = {
        "strategies": {
            "strategies": [
                {
                    "name": "mtf_trend_scalper",
                    "params": {
                        "min_adx": 35.0,
                        "sl_atr_mult": 1.5,
                    },
                }
            ]
        }
    }
    hypo = get_fallback_hypotheses(base_config=cfg)
    adx_test_vals = next(h["test_values"] for h in hypo["hypotheses"] if h["parameter"] == "min_adx")
    sl_test_vals = next(h["test_values"] for h in hypo["hypotheses"] if h["parameter"] == "sl_atr_mult")

    assert 35.0 in adx_test_vals
    assert 30.0 in adx_test_vals
    assert 1.5 in sl_test_vals
    assert 1.3 in sl_test_vals


def test_holdout_split_with_dataset_cutoff():
    trades = [
        {"time_open": "2026-10-01 10:00:00", "pnl": 100.0},
        {"time_open": "2026-10-05 10:00:00", "pnl": 50.0},
        {"time_open": "2026-10-25 10:00:00", "pnl": 75.0},
        {"time_open": "2026-10-26 10:00:00", "pnl": -25.0},
    ]
    with patch("scripts.saturday_autonomous_optimizer._run_single_symbol", return_value=trades):
        # CONTINUOUS_DATA has 2 symbols, so each symbol returns the 4 trades -> 8 total
        res = evaluate_backtest_candidate({}, holdout_days=10)
        assert res["total_trades"] == 8
        assert res["holdout_trades"] == 4
        assert res["holdout_pnl"] == 100.0
        assert res["in_sample_pnl"] == 300.0


def test_supervisor_init_corrupt_status_file():
    mock_file = io.StringIO("[]")
    with patch("builtins.open", return_value=mock_file):
        agent: Any = SupervisorAgent.__new__(SupervisorAgent)
        agent.logger = MagicMock()
        agent.account_id = "12345"
        agent.database = MagicMock()
        agent.get_current_session_start_utc = MagicMock()
        with open("logs/bot_status.json") as st_f:
            last_status = json.load(st_f)
            assert not isinstance(last_status, dict)


def test_retroactive_fill_preserves_excursions():
    agent: Any = SupervisorAgent.__new__(SupervisorAgent)
    agent.logger = MagicMock()
    agent.database = MagicMock()
    agent.database.connected = True
    agent.risk_agent = MagicMock()
    agent.risk_agent.get_tick_size.return_value = 0.25

    # Simulate closed position record with excursions
    closed_pos = {
        "trade_key": "TK-123",
        "symbol": "MES",
        "side": "Buy",
        "qty": 1,
        "entry": 5000.0,
        "sl_ticks": 20,
        "tp_ticks": 40,
        "mfe_price": 5010.0,
        "mae_price": 4995.0,
        "mfe_ticks": 40.0,
        "mae_ticks": 20.0,
        "mfe_r": 2.0,
        "mae_r": 1.0,
        "strategy_name": "mtf_trend_scalper",
    }

    # Initial close call with pnl=0.0
    agent._persist_closed_trade_outcome(
        symbol="MES",
        closed=closed_pos,
        pnl=0.0,
        order_type="stop_loss",
    )

    args = agent.database.persist_trade_outcome.call_args[1]
    outcome = args["outcome"]
    assert outcome["mfe_ticks"] == 40.0
    assert outcome["mae_ticks"] == 20.0
    assert outcome["mfe_r"] == 2.0

    # Now simulate retroactive fill update with pnl=-125.0 using the stripped recent_trades dict
    # where entry is 0 or missing, but excursions are preserved on the dict
    stripped_recent = {
        "trade_key": "TK-123",
        "symbol": "MES",
        "side": "Buy",
        "qty": 1,
        "entry": 5000.0,
        "mfe_ticks": 40.0,
        "mae_ticks": 20.0,
        "mfe_r": 2.0,
        "mae_r": 1.0,
    }
    agent._persist_closed_trade_outcome(
        symbol="MES",
        closed=stripped_recent,
        pnl=-125.0,
        order_type="stop_loss",
    )
    args2 = agent.database.persist_trade_outcome.call_args[1]
    outcome2 = args2["outcome"]
    assert outcome2["pnl"] == -125.0
    assert outcome2["mfe_ticks"] == 40.0
    assert outcome2["mae_ticks"] == 20.0
    assert outcome2["mfe_r"] == 2.0


def _build_headers(headers_dict: dict[str, str]) -> Message:
    msg = Message()
    for k, v in headers_dict.items():
        msg[k] = v
    return msg


def test_dashboard_csrf_same_origin():
    handler: Any = Handler.__new__(Handler)
    handler.headers = _build_headers({"Origin": "http://evil-site.com", "Host": "localhost:8765"})
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()

    # Cross origin must be rejected
    assert not handler._check_same_origin()
    handler.send_response.assert_called_with(403)

    # Same origin must be allowed
    handler.headers = _build_headers({"Origin": "http://localhost:8765", "Host": "localhost:8765"})
    assert handler._check_same_origin()

    # Missing Origin (curl / internal request) must be allowed
    handler.headers = _build_headers({"Host": "localhost:8765"})
    assert handler._check_same_origin()


def test_backtester_reversal_updates_excursions():
    bt: Any = Backtester.__new__(Backtester)
    bt.tick_size = 0.25
    bt.tick_value = 1.25
    bt.slippage_ticks = 0
    bt.commission_per_contract = 0.62
    bt.symbol = "MES"
    bt.risk_agent = MagicMock()
    bt.telemetry = MagicMock()
    bt.balance = 50000.0
    bt.active_trades = [
        {
            "side": "Buy",
            "entry": 5000.0,
            "qty": 1,
            "sl_ticks": 20,
            "tp_ticks": 40,
            "mfe_price": 5002.0,
            "mae_price": 4998.0,
            "time_open": "2026-11-01 10:00:00",
            "metadata": {},
        }
    ]

    recorded_trades = []
    bt._record_trade = lambda trade, exit_price, pnl, result, timestamp: recorded_trades.append((dict(trade), exit_price, result))

    # Reverse to Sell with entry/exit at 5010.0 (new MFE reached on exit bar)
    new_signal = {"side": "Sell", "quantity": 1, "sl_ticks": 20, "tp_ticks": 40}
    bt._open_trade(new_signal, "2026-11-01 10:05:00", 5010.0)

    assert len(recorded_trades) == 1
    closed_trade, exit_p, res_type = recorded_trades[0]
    assert exit_p == 5010.0
    assert res_type == "REVERSAL"
    # MFE price must be updated to 5010.0
    assert closed_trade["mfe_price"] == 5010.0


def test_projectx_client_hardening():
    client = ProjectXClient()
    client.account_id = 12345
    client.symbol_to_id_cache["MESZ6"] = "CON.F.US.MES.Z26"

    # 1. STOP_TYPES accepts stoploss string variants
    orders = [{"id": 77, "contractId": "CON.F.US.MES.Z26", "side": 1, "type": "stoploss", "status": 1}]
    with patch.object(client, "get_orders", return_value=orders):
        assert client.has_working_stop_order("MESZ6", "Buy") is True

    # 2. _build_refreshed_ws_url preserves &id=<connectionId>
    refreshed = client._build_refreshed_ws_url(
        "wss://rtc.topstepx.com/hubs/market",
        "NEW_JWT",
        "wss://rtc.topstepx.com/hubs/market?access_token=OLD&id=conn-abc-123",
    )
    assert "access_token=NEW_JWT" in refreshed
    assert "id=conn-abc-123" in refreshed

    # 3. _make_user_event_handler rejects cross-account envelopes even when inner dict lacks accountId
    events = []
    client._on_user_event = lambda et, p: events.append((et, p))
    handler = client._make_user_event_handler("GatewayUserOrder")
    handler([99999, 1, {"id": 501, "status": 2}])
    assert events == []
    handler([12345, 1, {"id": 502, "status": 2}])
    assert len(events) == 1
    assert events[0][1]["id"] == 502


def test_supervisor_protective_stop_respects_actual_live_switch():
    async def _run():
        with patch("src.agents.supervisor_agent.DatabaseService"):
            sup = SupervisorAgent(["MCL"], "12345")
        sup.loop = asyncio.get_running_loop()
        sup.execution_agent.mode = "live"
        with (
            patch.dict(os.environ, {"MODE": "live", "ACTUAL_LIVE_TRADING": "false"}),
            patch.object(sup.client, "has_working_stop_order", return_value=False),
            patch.object(sup.client, "place_order") as mock_place,
        ):
            res = await sup._ensure_protective_stop_for_position("MCLZ6", "Buy", 1, 88.5)
            assert res is False
            mock_place.assert_not_called()

    asyncio.run(_run())


def test_supervisor_close_in_flight_retained_until_drain_and_partial_fills():
    async def _run():
        with patch("src.agents.supervisor_agent.DatabaseService"):
            sup = SupervisorAgent(["MCL"], "12345")
        sup.loop = asyncio.get_running_loop()
        sup.client.cancel_open_orders_for_symbol = MagicMock(return_value=0)
        sup.active_positions["MCLZ6"] = [{"side": "Buy", "qty": 3, "entry": 88.5, "time": 1000.0}]

        # 1. netSize=0 sets _close_in_flight synchronously before drain coroutine executes
        sup.on_order_update({"contractId": "MCLZ6", "netSize": 0})
        assert sup._close_in_flight.get("MCLZ6") is True
        await asyncio.sleep(0.35)
        assert sup._close_in_flight.get("MCLZ6") is None
        assert sup.active_positions.get("MCLZ6") == []

        # 2. Partial fills (status 7) track cumulative fillVolume delta without double-counting
        sup.active_positions["MCLZ6"] = [{"side": "Buy", "qty": 3, "entry": 88.5, "time": 1000.0}]
        sup.on_order_update({"id": 9001, "contractId": "MCLZ6", "status": 7, "type": 1, "side": 1, "fillVolume": 1, "profitAndLoss": -10.0})
        await asyncio.sleep(0.05)
        assert sum(p["qty"] for p in sup.active_positions["MCLZ6"]) == 2
        sup.on_order_update({"id": 9001, "contractId": "MCLZ6", "status": 2, "type": 1, "side": 1, "fillVolume": 3, "profitAndLoss": -20.0})
        await asyncio.sleep(0.05)
        assert sup.active_positions.get("MCLZ6") == []

    asyncio.run(_run())


def test_strategy_engine_incremental_analyze_and_silent_priming():
    telemetry = TelemetryStore(enabled=True)
    eng = StrategyEngine("MESZ6", 1, {"sl": 40, "tp": 80, "tick_size": 0.25}, telemetry=telemetry, config={"strategies": []})
    call_indices = []

    class _MockStrat:
        name = "mock_strat"
        priority = 100
        enabled = True
        is_priming = False

        def on_bar(self, index, bars, indicators):
            call_indices.append((index, self.is_priming))
            return None

        def get_snapshot(self, index, bars, indicators):
            return {"status": "ok"}

        def reset(self):
            pass

    eng.strategies = [_MockStrat()]  # type: ignore[list-item]
    df = pd.DataFrame(
        [
            {"timestamp": f"2026-11-10T10:{i:02d}:00Z", "open": 5000.0, "high": 5001.0, "low": 4999.0, "close": 5000.5, "volume": 10}
            for i in range(5)
        ]
    )
    eng.analyze(df)
    # Bars 0..3 primed with is_priming=True, bar 4 evaluated live
    assert call_indices[:4] == [(0, True), (1, True), (2, True), (3, True)]
    assert call_indices[4] == (4, False)

    # Append 1 new bar -> only index 5 should be evaluated (no 0..4 loop!)
    call_indices.clear()
    df2 = pd.concat(
        [
            df,
            pd.DataFrame([{"timestamp": "2026-11-10T10:05:00Z", "open": 5000.5, "high": 5002.0, "low": 5000.0, "close": 5001.5, "volume": 12}]),
        ],
        ignore_index=True,
    )
    eng.analyze(df2)
    assert call_indices == [(5, False)]


def test_backtester_gap_open_stop_fill_and_execution_metadata_parity():
    # 1. Gap open past SL fills at open_p - slippage
    bt: Any = Backtester.__new__(Backtester)
    bt.tick_size = 0.25
    bt.tick_value = 1.25
    bt.slippage_ticks = 1
    bt.commission_per_contract = 0.0
    bt.symbol = "MES"
    bt.routing_mode = "optimizer_fast"
    bt.daily_profit_lock = 0.0
    bt.news_agent = MagicMock()
    bt.risk_agent = MagicMock()
    bt.telemetry = MagicMock()
    bt.balance = 50000.0
    bt.trades = []
    bt.active_trades = [
        {
            "side": "Buy",
            "entry": 5000.0,
            "sl_price": 4995.0,
            "tp_price": 5020.0,
            "qty": 1,
            "sl_ticks": 20,
            "tp_ticks": 80,
            "trail_mult": 0.0,
            "trail_trigger_ticks": 999,
            "be_triggered": False,
            "be_trigger_ticks": 999,
            "mfe_price": 5000.0,
            "mae_price": 5000.0,
            "time_open": "2026-11-01T10:00:00Z",
            "metadata": {},
        }
    ]
    bt._sync_risk_state = MagicMock()
    bt._update_active_trades(pd.Series({"open": 4990.0, "high": 4992.0, "low": 4989.0, "close": 4991.0, "timestamp": "2026-11-01T10:01:00Z"}))
    assert len(bt.trades) == 1
    assert bt.trades[0]["exit"] == 4989.75

    # 2. ExecutionAgent computes dynamic metadata brackets identically to Backtester
    mock_client = MagicMock()
    mock_client.get_contract_meta.return_value = {"tickSize": 0.25}
    mock_client.place_order.return_value = {"success": True, "orderId": 123}
    with patch.dict(os.environ, {"MODE": "live", "ACTUAL_LIVE_TRADING": "true", "USE_ORDER_BRACKETS": "true"}):
        ea = ExecutionAgent(mock_client)
        ea.execute_trade(
            {
                "symbol": "MESZ6",
                "side": "Buy",
                "quantity": 1,
                "entry_price": 5000.0,
                "sl_ticks": 40,
                "tp_ticks": 80,
                "metadata": {"stop_price": 4997.5, "target_price": 5005.0},
            }
        )
    assert mock_client.place_order.call_args.kwargs["sl_ticks"] == 10
    assert mock_client.place_order.call_args.kwargs["tp_ticks"] == 20


def test_optimizer_yaml_time_quoting_and_dashboard_referer_csrf():
    dumped = yaml.dump({"opening_range_start_et": "09:30", "trade_window_end_et": "14:00"}, Dumper=_QuotedTimeSafeDumper, sort_keys=False)
    reloaded = yaml.safe_load(dumped)
    assert reloaded["opening_range_start_et"] == "09:30"
    assert reloaded["trade_window_end_et"] == "14:00"
    assert isinstance(reloaded["trade_window_end_et"], str)

    handler: Any = Handler.__new__(Handler)
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()
    # Origin with missing Host must be rejected
    handler.headers = _build_headers({"Origin": "http://localhost:8765"})
    assert not handler._check_same_origin()
    # Cross-origin Referer without Origin must be rejected
    handler.headers = _build_headers({"Host": "localhost:8765", "Referer": "http://evil.com/attack.html"})
    assert not handler._check_same_origin()
    # Referer with missing Host must also be rejected
    handler.headers = _build_headers({"Referer": "http://localhost:8765/"})
    assert not handler._check_same_origin()


def test_backtester_optimizer_fast_honors_allow_flips_false():
    bt: Any = Backtester.__new__(Backtester)
    bt.routing_mode = "optimizer_fast"
    bt.routing_config = {"entry_filters": {"allow_flips": False}}
    bt.active_trades = [{"side": "Buy", "qty": 1}]
    bt.risk_agent = MagicMock()
    res = bt._route_signal(
        {"side": "Sell", "quantity": 1, "sl_ticks": 20, "metadata": {}},
        {"active_count": 1, "max_positions": 1, "current_price": 5000.0},
    )
    assert getattr(res, "stage", None) == "filter"


def test_supervisor_partial_fill_zero_volume_cancelled_and_partial_exit_releases_guard():
    async def _run():
        with patch("src.agents.supervisor_agent.DatabaseService"):
            sup = SupervisorAgent(["MCL"], "12345")
        sup.loop = asyncio.get_running_loop()
        sup.client.cancel_open_orders_for_symbol = MagicMock(return_value=0)
        sup.active_positions["MCLZ6"] = [{"side": "Buy", "qty": 3, "entry": 88.5, "time": 1000.0, "trade_key": "tk-1"}]

        # 1. status="3" (cancelled) must NOT trigger a fill
        sup.on_order_update({"id": 8001, "contractId": "MCLZ6", "status": 3, "type": 4, "side": 1, "size": 3})
        await asyncio.sleep(0.05)
        assert sum(p["qty"] for p in sup.active_positions["MCLZ6"]) == 3

        # 2. status=7 (partially_filled) with fillVolume=0 must NOT fall back to size=3
        sup.on_order_update({"id": 8002, "contractId": "MCLZ6", "status": 7, "type": 4, "side": 1, "fillVolume": 0, "size": 3})
        await asyncio.sleep(0.05)
        assert sum(p["qty"] for p in sup.active_positions["MCLZ6"]) == 3

        # 3. Partial exit (1 of 3) must release _close_in_flight once lock completes and not increment loss streak
        sup._consecutive_losses["MCLZ6"] = 0
        sup.on_order_update({"id": 8003, "contractId": "MCLZ6", "status": 7, "type": 4, "side": 1, "fillVolume": 1, "profitAndLoss": -15.0})
        await asyncio.sleep(0.05)
        assert sum(p["qty"] for p in sup.active_positions["MCLZ6"]) == 2
        assert sup._close_in_flight.get("MCLZ6") is None
        assert sup._consecutive_losses.get("MCLZ6", 0) == 0

    asyncio.run(_run())


def test_state_manager_sanitizes_infinity_and_db_bare_filename(tmp_path):
    from scripts.saturday_autonomous_optimizer import StateManager
    from src.persistence.database import DatabaseService

    state_file = tmp_path / "sat_state.json"
    mgr = StateManager(state_file)
    mgr.state["baseline_metrics"] = {"profit_factor": float("inf"), "bad_nan": float("nan")}
    mgr.save()
    loaded = json.loads(state_file.read_text(encoding="utf-8"))
    assert loaded["baseline_metrics"]["profit_factor"] == 999.0
    assert loaded["baseline_metrics"]["bad_nan"] == 0.0

    # Bare filename without directory component must not raise FileNotFoundError
    orig_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        db = DatabaseService({"enabled": True, "backend": "sqlite", "sqlite_path": "bare_swarm.db"})
        assert db.connected is True
        db.close()
    finally:
        os.chdir(orig_cwd)


def test_strategy_engine_reset_clears_candidates():
    eng = StrategyEngine("MESZ6", 1, {"sl": 40, "tp": 80, "tick_size": 0.25}, config={"strategies": []})
    eng._last_candidates = [{"side": "Buy"}]
    eng._last_selected = {"side": "Buy"}
    eng.reset()
    assert eng._last_candidates == []
    assert eng._last_selected is None

