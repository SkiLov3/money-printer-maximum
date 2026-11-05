from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import requests

from src.agents.supervisor_agent import SupervisorAgent
from src.api.projectx_client import ProjectXClient


class TestProjectXClientPositionAndOrderFailures:
    def test_search_open_positions_failure_returns_none(self):
        client = ProjectXClient()
        client.account_id = 12345
        with patch.object(client, "_request_with_refresh", side_effect=requests.exceptions.ReadTimeout("Timeout")):
            assert client.search_open_positions() is None

    def test_search_open_positions_error_payload_returns_none(self):
        client = ProjectXClient()
        client.account_id = 12345
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"success": False, "errorMessage": "Service Unavailable"}
        mock_resp.raise_for_status = MagicMock()
        with patch.object(client, "_request_with_refresh", return_value=mock_resp):
            assert client.search_open_positions() is None

    def test_search_open_positions_success_returns_list(self):
        client = ProjectXClient()
        client.account_id = 12345
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"success": True, "positions": [{"contractId": "CON.MCL", "size": 1}]}
        mock_resp.raise_for_status = MagicMock()
        with patch.object(client, "_request_with_refresh", return_value=mock_resp):
            res = client.search_open_positions()
            assert res == [{"contractId": "CON.MCL", "size": 1}]

    def test_search_open_positions_empty_flat_returns_empty_list(self):
        client = ProjectXClient()
        client.account_id = 12345
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"success": True, "positions": []}
        mock_resp.raise_for_status = MagicMock()
        with patch.object(client, "_request_with_refresh", return_value=mock_resp):
            res = client.search_open_positions()
            assert res == []

    def test_get_positions_returns_none_on_failure(self):
        client = ProjectXClient()
        with patch.object(client, "search_open_positions", return_value=None):
            assert client.get_positions() is None

    def test_get_orders_returns_none_on_failure(self):
        client = ProjectXClient()
        client.account_id = 12345
        with patch.object(client, "_request_with_refresh", side_effect=requests.exceptions.ConnectTimeout("Connect timeout")):
            assert client.get_orders(active=True) is None

    def test_cancel_open_orders_for_symbol_handles_none_safely(self):
        client = ProjectXClient()
        client.symbol_to_id_cache["MCLZ6"] = "CON.MCL"
        with patch.object(client, "get_orders", return_value=None):
            # Must return 0 without throwing TypeError: 'NoneType' object is not iterable
            cancelled = client.cancel_open_orders_for_symbol("MCLZ6")
            assert cancelled == 0


class TestHasWorkingStopOrder:
    def test_has_working_stop_order_short_position(self):
        client = ProjectXClient()
        client.symbol_to_id_cache["MCLZ6"] = "CON.MCL"
        # Short position (Sell): expects a working Buy stop order (side 0, type 4, status 1)
        orders = [
            {"id": 101, "contractId": "CON.MCL", "side": 0, "type": 4, "status": 1, "stopPrice": 103.13},
        ]
        with patch.object(client, "get_orders", return_value=orders):
            assert client.has_working_stop_order("MCLZ6", "Sell") is True

    def test_has_working_stop_order_long_position(self):
        client = ProjectXClient()
        client.symbol_to_id_cache["MESZ6"] = "CON.MES"
        # Long position (Buy): expects a working Sell stop order (side 1, type 4, status 1)
        orders = [
            {"id": 102, "contractId": "CON.MES", "side": 1, "type": 4, "status": 1, "stopPrice": 5800.0},
        ]
        with patch.object(client, "get_orders", return_value=orders):
            assert client.has_working_stop_order("MESZ6", "Buy") is True

    def test_has_working_stop_order_missing_or_cancelled(self):
        client = ProjectXClient()
        client.symbol_to_id_cache["MCLZ6"] = "CON.MCL"
        # Only take profit limit order (type 1) exists, or stop is cancelled (status 3)
        orders = [
            {"id": 101, "contractId": "CON.MCL", "side": 0, "type": 1, "status": 1, "limitPrice": 100.0},
            {"id": 102, "contractId": "CON.MCL", "side": 0, "type": 4, "status": 3, "stopPrice": 103.13},
        ]
        with patch.object(client, "get_orders", return_value=orders):
            assert client.has_working_stop_order("MCLZ6", "Sell") is False

    def test_has_working_stop_order_returns_none_on_api_error(self):
        client = ProjectXClient()
        client.symbol_to_id_cache["MCLZ6"] = "CON.MCL"
        with patch.object(client, "get_orders", return_value=None):
            assert client.has_working_stop_order("MCLZ6", "Sell") is None


def _make_supervisor(symbols: list[str], account_id: str = "12345") -> SupervisorAgent:
    """Instantiate SupervisorAgent with DatabaseService mocked so unit tests never touch logs/swarm.db."""
    with patch("src.agents.supervisor_agent.DatabaseService"):
        return SupervisorAgent(symbols, account_id)


class TestSupervisorReconciliationAndProtectiveStop:
    def test_sync_positions_skips_when_broker_positions_is_none(self):
        async def _test():
            supervisor = _make_supervisor(["MCL"], "12345")
            supervisor.loop = asyncio.get_running_loop()
            supervisor.active_positions["MCLZ6"] = [{"side": "Sell", "qty": 1, "entry": 102.26, "time": 1000.0}]

            with patch.object(supervisor.client, "cancel_open_orders_for_symbol") as mock_cancel:
                await supervisor._sync_positions(None)
                # Must NOT clear active positions and must NOT cancel open orders
                assert len(supervisor.active_positions.get("MCLZ6", [])) == 1
                mock_cancel.assert_not_called()

        asyncio.run(_test())

    def test_ensure_protective_stop_places_order_when_missing(self):
        async def _test():
            supervisor = _make_supervisor(["MCL"], "12345")
            supervisor.loop = asyncio.get_running_loop()
            supervisor.stack_config = {
                "symbol_brackets": {
                    "MCL": {"sl": 90, "tp": 220, "tick_size": 0.01},
                }
            }
            supervisor.risk_agent.symbol_brackets = supervisor.stack_config["symbol_brackets"]

            with (
                patch.object(supervisor.client, "has_working_stop_order", return_value=False),
                patch.object(supervisor.client, "place_order", return_value={"success": True, "orderId": 9999}) as mock_place,
            ):
                # Sell 1x MCLZ6 @ 102.26 -> Stop must be Buy 1x @ 103.16 (102.26 + 90 * 0.01)
                success = await supervisor._ensure_protective_stop_for_position("MCLZ6", "Sell", 1, 102.26)
                assert success is True
                mock_place.assert_called_once_with("MCLZ6", 1, "Buy", "Stop", None, 103.16)

        asyncio.run(_test())

    def test_ensure_protective_stop_mes_quarter_point_rounding(self):
        async def _test():
            supervisor = _make_supervisor(["MES"], "12345")
            supervisor.loop = asyncio.get_running_loop()
            supervisor.stack_config = {
                "symbol_brackets": {
                    "MES": {"sl": 40, "tp": 100, "tick_size": 0.25},
                }
            }
            supervisor.risk_agent.symbol_brackets = supervisor.stack_config["symbol_brackets"]

            with (
                patch.object(supervisor.client, "has_working_stop_order", return_value=False),
                patch.object(supervisor.client, "place_order", return_value={"success": True, "orderId": 8888}) as mock_place,
            ):
                # Buy 1x MESZ6 @ 5800.25 -> Stop must be Sell 1x @ 5790.25 (5800.25 - 40 * 0.25)
                # Must be 5790.25, NOT 5790.2 or 5790.3!
                success = await supervisor._ensure_protective_stop_for_position("MESZ6", "Buy", 1, 5800.25)
                assert success is True
                mock_place.assert_called_once_with("MESZ6", 1, "Sell", "Stop", None, 5790.25)

        asyncio.run(_test())

    def test_ensure_protective_stop_in_flight_cooldown(self):
        async def _test():
            supervisor = _make_supervisor(["MCL"], "12345")
            supervisor.loop = asyncio.get_running_loop()
            supervisor.stack_config = {
                "symbol_brackets": {
                    "MCL": {"sl": 90, "tp": 220, "tick_size": 0.01},
                }
            }
            supervisor.risk_agent.symbol_brackets = supervisor.stack_config["symbol_brackets"]

            with (
                patch.object(supervisor.client, "has_working_stop_order", return_value=False),
                patch.object(supervisor.client, "place_order", return_value={"success": True, "orderId": 9999}) as mock_place,
            ):
                # First call places stop
                res1 = await supervisor._ensure_protective_stop_for_position("MCLZ6", "Sell", 1, 102.26)
                assert res1 is True
                assert mock_place.call_count == 1

                # Immediate second call within 60s cooldown must skip duplicate placement
                res2 = await supervisor._ensure_protective_stop_for_position("MCLZ6", "Sell", 1, 102.26)
                assert res2 is True
                assert mock_place.call_count == 1  # Still 1, NOT 2!

        asyncio.run(_test())

    def test_ensure_protective_stop_skips_when_already_active(self):
        async def _test():
            supervisor = _make_supervisor(["MCL"], "12345")
            supervisor.loop = asyncio.get_running_loop()

            with (
                patch.object(supervisor.client, "has_working_stop_order", return_value=True),
                patch.object(supervisor.client, "place_order") as mock_place,
            ):
                success = await supervisor._ensure_protective_stop_for_position("MCLZ6", "Sell", 1, 102.26)
                assert success is True
                mock_place.assert_not_called()

        asyncio.run(_test())

    def test_ensure_protective_stop_skips_when_api_check_fails(self):
        async def _test():
            supervisor = _make_supervisor(["MCL"], "12345")
            supervisor.loop = asyncio.get_running_loop()

            with (
                patch.object(supervisor.client, "has_working_stop_order", return_value=None),
                patch.object(supervisor.client, "place_order") as mock_place,
            ):
                success = await supervisor._ensure_protective_stop_for_position("MCLZ6", "Sell", 1, 102.26)
                assert success is False
                mock_place.assert_not_called()

        asyncio.run(_test())

    def test_ensure_protective_stop_honors_strategy_sl_ticks(self):
        async def _test():
            supervisor = _make_supervisor(["MCL"], "12345")
            supervisor.loop = asyncio.get_running_loop()
            supervisor.stack_config = {
                "symbol_brackets": {
                    "MCL": {"sl": 90, "tp": 220, "tick_size": 0.01},
                }
            }
            supervisor.risk_agent.symbol_brackets = supervisor.stack_config["symbol_brackets"]
            supervisor.active_positions["MCLZ6"] = [
                {"side": "Sell", "qty": 1, "entry": 102.26, "time": 1000.0, "sl_ticks": 25}
            ]

            with (
                patch.object(supervisor.client, "has_working_stop_order", return_value=False),
                patch.object(supervisor.client, "place_order", return_value={"success": True, "orderId": 7777}) as mock_place,
            ):
                # Must use strategy sl_ticks=25 (102.26 + 0.25 = 102.51), NOT static bracket 90 (103.16)!
                success = await supervisor._ensure_protective_stop_for_position("MCLZ6", "Sell", 1, 102.26)
                assert success is True
                mock_place.assert_called_once_with("MCLZ6", 1, "Buy", "Stop", None, 102.51)

        asyncio.run(_test())

    def test_ensure_protective_stop_aborts_on_toctou_position_close(self):
        async def _test():
            supervisor = _make_supervisor(["MCL"], "12345")
            supervisor.loop = asyncio.get_running_loop()
            supervisor.stack_config = {
                "symbol_brackets": {
                    "MCL": {"sl": 90, "tp": 220, "tick_size": 0.01},
                }
            }
            supervisor.risk_agent.symbol_brackets = supervisor.stack_config["symbol_brackets"]
            supervisor.active_positions["MCLZ6"] = [
                {"side": "Sell", "qty": 1, "entry": 102.26, "time": 1000.0}
            ]

            def side_effect_close_while_checking(sym, side):
                # Simulate position closing on SignalR while REST /Order/search is in flight
                supervisor.active_positions["MCLZ6"] = []
                return False

            with (
                patch.object(supervisor.client, "has_working_stop_order", side_effect=side_effect_close_while_checking),
                patch.object(supervisor.client, "place_order") as mock_place,
            ):
                success = await supervisor._ensure_protective_stop_for_position("MCLZ6", "Sell", 1, 102.26)
                assert success is True
                mock_place.assert_not_called()

        asyncio.run(_test())

    def test_close_in_flight_cleared_when_non_drain_future_wakes_after_drain(self):
        async def _test():
            supervisor = _make_supervisor(["MCL"], "12345")
            supervisor.loop = asyncio.get_running_loop()
            # Position was already cleared by an earlier drain=True future
            supervisor.active_positions["MCLZ6"] = []
            with patch.object(supervisor.client, "cancel_open_orders_for_symbol", return_value=0):
                await supervisor._handle_position_close_update("MCLZ6", "stop", -87.0, drain=False, qty=3)
                assert supervisor._close_in_flight.get("MCLZ6") is None
                assert supervisor._close_in_flight_since.get("MCLZ6") is None

        asyncio.run(_test())

    def test_restore_balance_persistence_preserves_peak_across_day_boundary(self, tmp_path):
        import json
        from decimal import Decimal

        supervisor = _make_supervisor(["MCL"], "12345678")
        bal_file = tmp_path / "session_opening_balance.json"
        bal_file.write_text(
            json.dumps({
                "day": "1999-01-01",
                "account_id": "12345678",
                "balance": 50431.25,
                "peak_balance": 51152.01,
            }),
            encoding="utf-8",
        )
        supervisor._balance_persistence_path = bal_file
        supervisor.risk_agent.peak_balance = Decimal("0.0")
        supervisor._restore_balance_persistence()
        assert supervisor.risk_agent.peak_balance == Decimal("51152.01")


