# pyright: reportAttributeAccessIssue=false
import os
import pytest
from unittest.mock import MagicMock, patch

from src.api.projectx_client import (
    ProjectXClient,
    normalize_order_type,
    normalize_status,
    normalize_position_type,
)
from src.api.account_resolver import AccountResolver
from src.agents.execution_agent import ExecutionAgent


class TestBrokerProtocolEnums:
    def test_order_type_mappings_match_topstepx_spec(self):
        # TopstepX / ProjectX Gateway spec:
        # 1 = Limit, 2 = Market, 4 = Stop, 5 = TrailingStop
        assert normalize_order_type(1) == "limit"
        assert normalize_order_type(2) == "market"
        assert normalize_order_type(4) == "stop"
        assert normalize_order_type(5) == "trailing_stop"
        assert normalize_order_type("Limit") == "limit"
        assert normalize_order_type("Market") == "market"
        assert normalize_order_type("Stop") == "stop"
        assert normalize_order_type("trailing_stop") == "trailing_stop"

    def test_order_status_mappings_match_topstepx_spec(self):
        # 0=None, 1=Working, 2=Filled, 3=Cancelled, 4=Expired, 5=Rejected, 6=PendingNew, 7=PartiallyFilled, 8=PendingBracket
        assert normalize_status(0) == "none"
        assert normalize_status(1) == "working"
        assert normalize_status(2) == "filled"
        assert normalize_status(3) == "cancelled"
        assert normalize_status(4) == "expired"
        assert normalize_status(5) == "rejected"
        assert normalize_status(6) == "pending_new"
        assert normalize_status(7) == "partially_filled"
        assert normalize_status(8) == "pending_bracket"

    def test_position_type_mappings(self):
        # 0 = Flat, 1 = Long, 2 = Short
        assert normalize_position_type(0) == "flat"
        assert normalize_position_type(1) == "long"
        assert normalize_position_type(2) == "short"


class TestPlaceOrderProtocol:
    @patch.dict(os.environ, {"MODE": "live", "ACTUAL_LIVE_TRADING": "true"})
    def test_place_order_payload_structure(self):
        client = ProjectXClient()
        client.account_id = 12345
        client.symbol_to_id_cache = {"MNQ": "CON.F.US.MNQ.M26"}
        client._request_with_refresh = MagicMock()
        client._request_with_refresh.return_value.json.return_value = {"success": True, "orderId": 9999}
        client._request_with_refresh.return_value.raise_for_status = MagicMock()

        # 1. Test Buy Order with Brackets
        res = client.place_order(
            symbol="MNQ",
            quantity=2,
            side="Buy",
            order_type="Limit",
            limit_price=21050.25,
            sl_ticks=40,
            tp_ticks=80,
        )
        assert res.get("success") is True
        call_args = client._request_with_refresh.call_args
        payload = call_args[1]["json"]

        assert payload["accountId"] == 12345
        assert payload["contractId"] == "CON.F.US.MNQ.M26"
        assert payload["size"] == 2
        assert payload["side"] == 0  # 0 = Buy
        assert payload["type"] == 1  # 1 = Limit
        assert payload["limitPrice"] == 21050.25
        # Bracket checks for Buy: SL negative (-40), TP positive (+80)
        assert payload["stopLossBracket"] == {"ticks": -40, "type": 4}  # 4 = Stop
        assert payload["takeProfitBracket"] == {"ticks": 80, "type": 1}  # 1 = Limit

        # 2. Test Sell / Short Order with Brackets
        client.place_order(
            symbol="MNQ",
            quantity=1,
            side="Short",
            order_type="Market",
            sl_ticks=30,
            tp_ticks=60,
        )
        call_args2 = client._request_with_refresh.call_args
        payload2 = call_args2[1]["json"]

        assert payload2["side"] == 1  # 1 = Sell/Short
        assert payload2["type"] == 2  # 2 = Market
        # Bracket checks for Sell: SL positive (+30), TP negative (-60)
        assert payload2["stopLossBracket"] == {"ticks": 30, "type": 4}
        assert payload2["takeProfitBracket"] == {"ticks": -60, "type": 1}

    @patch.dict(os.environ, {"MODE": "live"})
    def test_place_order_side_handling_robustness(self):
        client = ProjectXClient()
        client.account_id = 12345
        client._request_with_refresh = MagicMock()
        client._request_with_refresh.return_value.json.return_value = {"success": True}
        client._request_with_refresh.return_value.raise_for_status = MagicMock()

        # Numeric side 1 -> Sell
        client.place_order("MNQ", 1, side=1)
        assert client._request_with_refresh.call_args[1]["json"]["side"] == 1

        # Numeric side 0 -> Buy
        client.place_order("MNQ", 1, side=0)
        assert client._request_with_refresh.call_args[1]["json"]["side"] == 0

        # "Ask" -> Sell
        client.place_order("MNQ", 1, side="Ask")
        assert client._request_with_refresh.call_args[1]["json"]["side"] == 1

    def test_cancel_open_orders_for_symbol_isolates_symbols(self):
        client = ProjectXClient()
        client.symbol_to_id_cache = {
            "MES": "CON.F.US.MES.Z26",
            "MCL": "CON.F.US.MCL.V26",
        }
        client.get_orders = MagicMock(return_value=[
            {
                "id": 101,
                "contractId": "CON.F.US.MES.Z26",
                "contractDisplayName": "MES",
                "status": 1,
                "side": 0,
            },
            {
                "id": 202,
                "contractId": "CON.F.US.MCL.V26",
                "contractDisplayName": "MCL",
                "status": 1,
                "side": 1,
            },
        ])
        client.cancel_order = MagicMock(return_value=True)

        cancelled = client.cancel_open_orders_for_symbol("MES")
        assert cancelled == 1
        client.cancel_order.assert_called_once_with(101)


class TestAccountResolver:
    def test_resolve_by_numeric_id(self):
        mock_client = MagicMock()
        mock_client.get_accounts.return_value = {
            "accounts": [
                {"id": 11111, "name": "Practice 123", "balance": 50000.0, "canTrade": True},
                {"id": 22222, "name": "Funded 456", "balance": 100000.0, "canTrade": True},
            ]
        }
        resolver = AccountResolver(mock_client)
        info = resolver.resolve_account(account_identifier="22222", mode="live")
        assert info is not None
        assert info.id == 22222
        assert info.name == "Funded 456"
        assert mock_client.account_id == 22222

    def test_resolve_by_name_substring(self):
        mock_client = MagicMock()
        mock_client.get_accounts.return_value = {
            "accounts": [
                {"id": 11111, "name": "Practice Account Alpha", "balance": 50000.0, "canTrade": True},
                {"id": 22222, "name": "Funded Account Beta", "balance": 100000.0, "canTrade": True},
            ]
        }
        resolver = AccountResolver(mock_client)
        info = resolver.resolve_account(account_identifier="alpha", mode="simulation")
        assert info is not None
        assert info.id == 11111

    def test_auto_discover_practice_in_simulation(self):
        mock_client = MagicMock()
        mock_client.get_accounts.return_value = {
            "accounts": [
                {"id": 22222, "name": "Funded Live", "balance": 100000.0, "canTrade": True},
                {"id": 11111, "name": "Practice Sim", "balance": 50000.0, "canTrade": True},
            ]
        }
        resolver = AccountResolver(mock_client)
        info = resolver.resolve_account(account_identifier=None, mode="simulation")
        assert info is not None
        assert info.id == 11111
        assert info.is_practice is True

    def test_aborts_in_live_mode_when_account_not_found(self):
        mock_client = MagicMock()
        mock_client.get_accounts.return_value = {
            "accounts": [
                {"id": 22222, "name": "Funded Live", "balance": 100000.0, "canTrade": True},
                {"id": 11111, "name": "Practice Sim", "balance": 50000.0, "canTrade": True},
            ]
        }
        resolver = AccountResolver(mock_client)
        info = resolver.resolve_account(account_identifier="99999", mode="live")
        assert info is None


class TestExecutionAgentHardening:
    @patch.dict(os.environ, {"MODE": "live", "ACTUAL_LIVE_TRADING": "true"})
    def test_flatten_proceeds_even_if_bracket_cancel_raises(self):
        mock_client = MagicMock()
        # Bracket cancel throws network exception
        mock_client.cancel_open_orders_for_symbol.side_effect = RuntimeError("Network timeout on cancel")
        mock_client.place_order.return_value = {"success": True, "orderId": 5555}

        agent = ExecutionAgent(mock_client)
        res = agent.flatten_position(symbol="MNQ", quantity=1, current_side="Buy")

        # Must not abort! Must proceed with market flatten order
        assert res.get("success") is True
        mock_client.place_order.assert_called_once_with(
            symbol="MNQ",
            quantity=1,
            side="Sell",
            order_type="Market",
        )

    @patch.dict(os.environ, {"MODE": "live", "ACTUAL_LIVE_TRADING": "true"})
    def test_execute_trade_propagates_order_types_and_prices(self):
        mock_client = MagicMock()
        mock_client.place_order.return_value = {"success": True, "orderId": 1234}
        agent = ExecutionAgent(mock_client)

        trade_intent = {
            "symbol": "MNQ",
            "side": "Buy",
            "quantity": 1,
            "order_type": "Limit",
            "limit_price": 21000.50,
            "sl_ticks": 40,
            "tp_ticks": 80,
        }
        res = agent.execute_trade(trade_intent)
        assert res.get("success") is True
        mock_client.place_order.assert_called_once_with(
            symbol="MNQ",
            quantity=1,
            side="Buy",
            order_type="Limit",
            limit_price=21000.50,
            stop_price=None,
            sl_ticks=40,
            tp_ticks=80,
        )

    @patch.dict(os.environ, {"MODE": "live", "ACTUAL_LIVE_TRADING": "true"})
    def test_execute_trade_handles_error_status_cleanly(self):
        mock_client = MagicMock()
        # Returns an error dictionary without success=False
        mock_client.place_order.return_value = {"status": "error", "message": "Risk limit exceeded"}
        agent = ExecutionAgent(mock_client)

        trade_intent = {"symbol": "MNQ", "side": "Buy", "quantity": 1}
        res = agent.execute_trade(trade_intent)
        assert res.get("status") == "rejected"
        assert res.get("message") == "Risk limit exceeded"


class TestSignalREnvelopeHandling:
    def test_multi_shape_envelope_extraction(self):
        client = ProjectXClient()
        received = []
        client._on_user_event = lambda event_type, payload: received.append((event_type, payload))

        handler = client._make_user_event_handler("GatewayUserOrder")

        # 1. 3-element list [accountId, actionInt, dataDict]
        handler([12345, 1, {"id": 101, "status": 2}])
        assert len(received) == 1
        assert received[-1][1]["id"] == 101

        # 2. 2-element list [accountId, dataDict]
        handler([12345, {"id": 102, "status": 1}])
        assert len(received) == 2
        assert received[-1][1]["id"] == 102

        # 3. 2-element list with batch list [accountId, [item1, item2]]
        handler([12345, [{"id": 103}, {"id": 104}]])
        assert len(received) == 4
        assert received[-2][1]["id"] == 103
        assert received[-1][1]["id"] == 104

        # 4. Bare dict
        handler({"id": 105, "status": 3})
        assert len(received) == 5
        assert received[-1][1]["id"] == 105


class TestSymbolVariantMatchingAndResolution:
    def test_variant_matching(self):
        from src.agents.supervisor_agent import SupervisorAgent
        sup = SupervisorAgent.__new__(SupervisorAgent)
        
        # 1-digit vs 2-digit year
        assert sup._match_symbol_variant("MESZ6", "MESZ26") is True
        assert sup._match_symbol_variant("MESZ26", "MESZ6") is True
        assert sup._match_symbol_variant("MCLX6", "MCLX26") is True
        assert sup._match_symbol_variant("MGCZ6", "MGCZ26") is True
        assert sup._match_symbol_variant("NQM6", "NQM26") is True
        
        # Slash prefixes
        assert sup._match_symbol_variant("/MES", "MESZ6") is True
        assert sup._match_symbol_variant("MES", "MESZ26") is True
        assert sup._match_symbol_variant("/MCL", "MCLX6") is True
        
        # Negative matches
        assert sup._match_symbol_variant("MESZ26", "MNQZ26") is False
        assert sup._match_symbol_variant("MESZ26", "MESU26") is False

    def test_resolve_symbol_from_payload(self):
        from src.agents.supervisor_agent import SupervisorAgent
        sup = SupervisorAgent.__new__(SupervisorAgent)
        sup.official_symbols = ["MCLX6", "MGCZ6", "MESZ6"]
        sup.symbol_map = {"MCL": "MCLX6", "MGC": "MGCZ6", "MES": "MESZ6"}
        sup.client = MagicMock()
        sup.client.symbol_to_id_cache = {
            "MESZ6": "CON.F.US.MES.Z26",
            "MESZ26": "CON.F.US.MES.Z26",
            "MES": "CON.F.US.MES.Z26",
            "MCLX6": "CON.F.US.MCLE.X26",
            "MGCZ6": "CON.F.US.MGC.Z26",
        }

        # Contract Display Name from TopstepX SignalR/REST
        p1 = {"contractDisplayName": "MESZ26", "contractId": "CON.F.US.MES.Z26"}
        assert sup._resolve_symbol_from_payload(p1) == "MESZ6"

        # Exchange slash symbol
        p2 = {"symbol": "/MES", "contractId": "CON.F.US.MES.Z26"}
        assert sup._resolve_symbol_from_payload(p2) == "MESZ6"

        # Display name only
        p3 = {"contractDisplayName": "MCLX26"}
        assert sup._resolve_symbol_from_payload(p3) == "MCLX6"

        # ContractId only
        p4 = {"contractId": "CON.F.US.MGC.Z26"}
        assert sup._resolve_symbol_from_payload(p4) == "MGCZ6"


class TestSignalRSelfHealingAndSessionTransitions:
    def test_signalrcore_monkeypatch_connection_alive_exists(self):
        from signalrcore.transport.base_transport import BaseTransport
        assert hasattr(BaseTransport, "connection_alive")
        assert BaseTransport.connection_alive is False

    def test_update_connection_urls_syncs_fresh_token(self):
        client = ProjectXClient()
        expected_value = "MOCK_UNIT_TEST_ONLY"  # gitleaks:allow
        client.access_token = expected_value

        mock_conn = MagicMock()
        mock_conn.transport = MagicMock()
        mock_conn.headers = {}
        mock_conn.transport.headers = {}
        client.connection = mock_conn

        mock_user_conn = MagicMock()
        mock_user_conn.transport = MagicMock()
        mock_user_conn.headers = {}
        mock_user_conn.transport.headers = {}
        client.user_connection = mock_user_conn

        client._update_connection_urls()

        assert expected_value in client.connection.url
        assert expected_value in client.connection.transport.url
        assert client.connection.headers["Authorization"] == f"Bearer {expected_value}"

        assert expected_value in client.user_connection.url
        assert expected_value in client.user_connection.transport.url
        assert client.user_connection.headers["Authorization"] == f"Bearer {expected_value}"

    def test_reconnect_all_streams_forces_reauth_and_restart(self):
        client = ProjectXClient()
        client.stop_streaming = MagicMock()
        client.stop_user_streaming = MagicMock()
        client.login = MagicMock(return_value=True)
        client.start_streaming = MagicMock()
        client.start_user_streaming = MagicMock()
        client._send_market_data_subscriptions = MagicMock()
        client.active_subscriptions = {"MESZ6"}

        with patch("time.sleep", return_value=None):
            client.reconnect_all_streams(force=True)

        client.stop_streaming.assert_called_once()
        client.stop_user_streaming.assert_called_once()
        client.login.assert_called_once()
        client.start_streaming.assert_called_once_with(force=True)
        client.start_user_streaming.assert_called_once_with(force=True)
        client._send_market_data_subscriptions.assert_called_once()

    def test_start_streaming_force_restarts_even_if_connected(self):
        client = ProjectXClient()
        client.is_connected = True
        client._ensure_token = MagicMock(return_value=True)
        client.setup_signalr = MagicMock()
        mock_conn = MagicMock()
        client.connection = mock_conn

        # Without force: should return early
        client.start_streaming(force=False)
        client.setup_signalr.assert_not_called()

        # With force=True: must restart
        client.start_streaming(force=True)
        client.setup_signalr.assert_called_once()
        mock_conn.start.assert_called_once()

    @pytest.mark.anyio
    async def test_supervisor_watchdog_triggers_self_healing_on_stall(self):
        import asyncio
        from unittest.mock import AsyncMock
        from src.agents.supervisor_agent import SupervisorAgent

        sup = SupervisorAgent.__new__(SupervisorAgent)
        sup.logger = MagicMock()
        sup.client = MagicMock()
        sup.client.last_tick_at = 1000.0  # 100 seconds ago
        sup.news_agent = MagicMock()
        sup.news_agent.is_market_hours.return_value = True
        sup.news_agent.is_near_close.return_value = False
        sup.risk_agent = MagicMock()
        sup.risk_agent.is_risk_limit_hit.return_value = (False, None)
        sup._risk_halt = False
        sup._lock = asyncio.Lock()
        sup.active_positions = {}
        sup.official_symbols = ["MESZ6"]
        sup.data_agents = {}
        sup._close_in_flight = {}
        sup._close_in_flight_since = {}
        sup._last_stall_reconnect_at = 0.0

        sup.loop = MagicMock()
        # Mock run_in_executor to execute the func directly
        async def fake_run_in_executor(executor, func, *args):
            return func(*args) if callable(func) else None
        sup.loop.run_in_executor = fake_run_in_executor
        sup._refresh_recent_bars = AsyncMock()

        # Execute check at t=1100.0 (stall of 100s > 90s)
        await sup._check_high_frequency_guards(1100.0)

        # Self-healing reconnect must have triggered
        sup.client.reconnect_all_streams.assert_called_once_with(force=True)
        sup._refresh_recent_bars.assert_called_once_with(bar_count=60)
        assert sup._last_stall_reconnect_at == 1100.0

    def test_strategy_engine_session_regime_fallback_to_now(self):
        import pandas as pd
        from src.strategies.engine import StrategyEngine

        engine = StrategyEngine(symbol="MESZ6", default_qty=1, bracket={"sl": 40, "tp": 80, "tick_size": 0.25})

        # Ancient timestamp from hours ago (e.g. 15:47 ET)
        ancient_ts = pd.to_datetime("2026-09-29 19:47:00", utc=True)
        
        # When fallback_to_now=False, it evaluates the exact ancient timestamp (15:47 ET = NY RTH)
        regime_historical = engine._compute_session_regime(ancient_ts, fallback_to_now=False)
        assert regime_historical == "NY RTH"

        # When fallback_to_now=True, it falls back to current clock
        regime_fallback = engine._compute_session_regime(ancient_ts, fallback_to_now=True)
        assert regime_fallback in ("NY RTH", "NY PRE", "ASIAN", "GLOBEX", "MAINTENANCE", "CLOSED")



