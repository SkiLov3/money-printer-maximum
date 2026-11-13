import time
from unittest.mock import MagicMock

from src.api.projectx_client import ProjectXClient
from src.agents.data_agent import DataAgent, parse_market_timestamp
from src.agents.supervisor_agent import SupervisorAgent
from src.strategies.engine import StrategyEngine


class TestMarketDataSymbolVariantRouting:
    def test_lookup_symbol_for_contract_id_integer_and_string_types(self):
        client = ProjectXClient()
        # Symbol cache populated with string contract ID
        client.symbol_to_id_cache = {"MESZ6": "12345", "MES": "12345"}
        
        # Inbound numeric int
        raw_id = 12345
        assert client.lookup_symbol_for_contract_id(raw_id) == "MESZ6"  # type: ignore[arg-type]
        # Inbound numeric string
        assert client.lookup_symbol_for_contract_id("12345") == "MESZ6"

    def test_lookup_symbol_for_contract_id_topstep_con_spec(self):
        client = ProjectXClient()
        client.symbol_to_id_cache = {}  # Empty cache
        
        # Raw Topstep contract IDs
        assert client.lookup_symbol_for_contract_id("CON.F.US.MES.Z26") == "MESZ6"
        assert client.lookup_symbol_for_contract_id("CON.F.US.MCLE.X26") == "MCLX6"
        assert client.lookup_symbol_for_contract_id("CON.F.US.MNQ.Z26") == "MNQZ6"

    def test_supervisor_match_symbol_variant_con_syntax(self):
        sup = SupervisorAgent.__new__(SupervisorAgent)
        
        # Matches CON.F.US syntax against 1-digit, 2-digit, and root symbols
        assert sup._match_symbol_variant("MESZ6", "CON.F.US.MES.Z26") is True
        assert sup._match_symbol_variant("MESZ26", "CON.F.US.MES.Z26") is True
        assert sup._match_symbol_variant("MES", "CON.F.US.MES.Z26") is True
        assert sup._match_symbol_variant("/MES", "CON.F.US.MES.Z26") is True
        assert sup._match_symbol_variant("MCLX6", "CON.F.US.MCLE.X26") is True
        assert sup._match_symbol_variant("MNQZ6", "CON.F.US.MNQ.Z26") is True
        
        # Negative matches
        assert sup._match_symbol_variant("MESZ6", "CON.F.US.MNQ.Z26") is False

    def test_data_agent_matches_symbol_con_syntax(self):
        da = DataAgent("MESZ6")
        assert da._matches_symbol("MESZ6") is True
        assert da._matches_symbol("MESZ26") is True
        assert da._matches_symbol("/MES") is True
        assert da._matches_symbol("MES") is True
        assert da._matches_symbol("CON.F.US.MES.Z26") is True
        assert da._matches_symbol("CON.F.US.MNQ.Z26") is False

        da_mcl = DataAgent("MCLX6")
        assert da_mcl._matches_symbol("CON.F.US.MCLE.X26") is True


class TestDataAgentTimestampAndBarLifecycle:
    def test_parse_market_timestamp_epoch_formats(self):
        # Current epoch seconds
        now_sec = 1790692563.0
        dt_sec = parse_market_timestamp(now_sec)
        assert dt_sec.year == 2026
        
        # Milliseconds epoch
        now_ms = 1790692563000
        dt_ms = parse_market_timestamp(now_ms)
        assert dt_ms.year == 2026

        # String numeric epoch
        dt_str = parse_market_timestamp("1790692563.0")
        assert dt_str.year == 2026

        # ISO string
        dt_iso = parse_market_timestamp("2026-09-29T10:30:00Z")
        assert dt_iso.year == 2026
        assert dt_iso.hour == 10
        assert dt_iso.minute == 30

    def test_data_agent_rejects_out_of_order_stale_ticks(self):
        agent = DataAgent("MESZ6", bar_interval_seconds=60)
        
        # Initial tick at 09:31:10
        agent.handle_market_data({"symbol": "MESZ6", "price": 5800.0, "volume": 1, "timestamp": "2026-09-29T09:31:10Z"})
        assert agent.current_bar is not None
        assert agent.current_bar["start_time"].minute == 31
        assert agent.current_bar["high"] == 5800.0
        
        # Late tick from 09:30:55 arrives late due to network jitter
        agent.handle_market_data({"symbol": "MESZ6", "price": 5999.0, "volume": 10, "timestamp": "2026-09-29T09:30:55Z"})
        # Should not corrupt 09:31 bar
        assert agent.current_bar is not None
        assert agent.current_bar["high"] == 5800.0
        assert agent.tick_count == 1

    def test_data_agent_close_bar_deduplication(self):
        agent = DataAgent("MESZ6", bar_interval_seconds=60)
        
        # Historical bar preloaded for 09:30:00
        agent.load_historical_bars([
            {"timestamp": "2026-09-29T09:30:00Z", "open": 5800.0, "high": 5805.0, "low": 5795.0, "close": 5802.0, "volume": 100}
        ])
        assert len(agent.bars) == 1
        
        # Live ticks for 09:30:00 arrive and form completed bar
        agent.handle_market_data({"symbol": "MESZ6", "price": 5800.0, "volume": 5, "timestamp": "2026-09-29T09:30:10Z"})
        agent.handle_market_data({"symbol": "MESZ6", "price": 5810.0, "volume": 5, "timestamp": "2026-09-29T09:30:50Z"})
        # Tick at 09:31 rolls and closes 09:30 bar
        agent.handle_market_data({"symbol": "MESZ6", "price": 5812.0, "volume": 2, "timestamp": "2026-09-29T09:31:05Z"})
        
        # Must replace preloaded partial 09:30 bar without duplicate timestamps
        assert len(agent.bars) == 1
        assert agent.bars.iloc[-1]["high"] == 5810.0


class TestSignalRStreamingAndReconnection:
    def test_subscribe_market_data_queues_when_not_connected(self):
        client = ProjectXClient()
        client.is_connected = False
        client.connection = MagicMock()
        
        # Subscribing before connection is open
        res = client.subscribe_market_data(["MESZ6", "MNQZ6"])
        assert res is False
        assert "MESZ6" in client.active_subscriptions
        assert "MNQZ6" in client.active_subscriptions
        # Does not attempt to send on pending connection
        client.connection.send.assert_not_called()

        # When on_open fires, queued subscriptions are immediately sent
        client.on_open()
        assert client.is_connected is True
        assert client.connection.send.call_count == 2

    def test_reconnection_reissues_active_subscriptions(self):
        client = ProjectXClient()
        client.is_connected = True
        client.connection = MagicMock()
        client.symbol_to_id_cache = {"MESZ6": "CON.F.US.MES.Z26"}
        client.active_subscriptions = {"MESZ6"}

        client.on_reconnect()
        client.connection.send.assert_called_with("SubscribeContractQuotes", ["CON.F.US.MES.Z26"])

    def test_user_event_handler_supports_batch_list_with_action_int(self):
        client = ProjectXClient()
        events = []
        client._on_user_event = lambda event_type, payload: events.append((event_type, payload))

        handler = client._make_user_event_handler("GatewayUserOrder")
        # 3-element list with batch list: [accountId, actionInt, [dict1, dict2]]
        handler([12345, 1, [{"id": 101, "status": 2}, {"id": 102, "status": 2}]])
        assert len(events) == 2
        assert events[0][1]["id"] == 101
        assert events[1][1]["id"] == 102

    def test_resolve_symbol_auto_registers_active_subscriptions(self):
        client = ProjectXClient()
        client.access_token = "fake_jwt"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "contracts": [
                {
                    "name": "MESZ6",
                    "contractDisplayName": "MESZ26",
                    "id": "CON.F.US.MES.Z26",
                    "activeContract": True,
                    "tickSize": 0.25,
                    "tickValue": 1.25,
                }
            ]
        }
        client._request_with_refresh = MagicMock(return_value=mock_resp)
        resolved = client.resolve_symbol("MES")
        assert resolved == "MESZ6"
        assert "MESZ6" in client.active_subscriptions

    def test_reconnect_all_streams_auto_subscribes_known_contracts(self):
        client = ProjectXClient()
        client.access_token = "fake_jwt"
        client.symbol_to_id_cache = {"MESZ6": "CON.F.US.MES.Z26"}
        client.contract_id_to_symbol = {"CON.F.US.MES.Z26": "MESZ6"}
        client.active_subscriptions = set()  # initially empty
        client.login = MagicMock(return_value=True)
        client.stop_streaming = MagicMock()
        client.stop_user_streaming = MagicMock()
        client.start_streaming = MagicMock()
        client.start_user_streaming = MagicMock()
        client.is_connected = True
        client.connection = MagicMock()

        client.reconnect_all_streams(force=True)
        assert "MESZ6" in client.active_subscriptions
        client.connection.send.assert_called_with("SubscribeContractQuotes", ["CON.F.US.MES.Z26"])


class TestStrategyEngineSessionRegime:
    def test_compute_session_regime_handles_numeric_epoch(self):
        eng = StrategyEngine("MES", 1, {})
        # Epoch timestamp representing current time should not parse as 1970
        current_epoch = time.time()
        regime = eng._compute_session_regime(current_epoch)
        assert regime in ("NY RTH", "NY PRE", "ASIAN", "GLOBEX", "MAINTENANCE", "CLOSED")

    def test_compute_session_regime_handles_none_fallback(self):
        eng = StrategyEngine("MES", 1, {})
        regime = eng._compute_session_regime(None)
        assert regime in ("NY RTH", "NY PRE", "ASIAN", "GLOBEX", "MAINTENANCE", "CLOSED")


class TestLiveDataAccuracyAndPreopenWarmup:
    def test_data_agent_ignores_non_trade_quote_midpoints(self):
        agent = DataAgent("MCLZ6", bar_interval_seconds=60)
        # Executed trade at 88.73
        agent.handle_market_data({"symbol": "MCLZ6", "price": 88.73, "volume": 10, "is_trade": True, "timestamp": "2026-11-12T23:46:15Z"})
        # Quote-only midpoint update (is_trade=False) must NOT mutate OHLC
        agent.handle_market_data({"symbol": "MCLZ6", "price": 88.71, "volume": 0, "is_trade": False, "timestamp": "2026-11-12T23:46:45Z"})
        assert agent.current_bar is not None
        assert agent.current_bar["close"] == 88.73
        assert agent.current_bar["low"] == 88.73

        # Quote-only tick in next minute still triggers bar expiry so the bar closes cleanly at 88.73
        agent.handle_market_data({"symbol": "MCLZ6", "price": 88.74, "volume": 0, "is_trade": False, "timestamp": "2026-11-12T23:47:01Z"})
        assert agent.current_bar is None
        assert len(agent.bars) == 1
        assert agent.bars.iloc[-1]["close"] == 88.73

    def test_data_agent_rejects_late_tick_for_live_closed_bar(self):
        from datetime import datetime, timezone

        agent = DataAgent("MESZ6", bar_interval_seconds=60)
        agent.handle_market_data({"symbol": "MESZ6", "price": 7544.0, "volume": 500, "is_trade": True, "timestamp": "2026-11-13T16:27:10Z"})
        # Grace period: 16:28:00.5 does NOT close yet when grace_seconds=2.0
        agent.check_bar_expiry(datetime(2026, 11, 13, 16, 28, 0, 500000, tzinfo=timezone.utc), grace_seconds=2.0)
        assert agent.current_bar is not None

        # Past grace period (16:28:02.5): closes cleanly
        agent.check_bar_expiry(datetime(2026, 11, 13, 16, 28, 2, 500000, tzinfo=timezone.utc), grace_seconds=2.0)
        assert agent.current_bar is None
        assert len(agent.bars) == 1
        assert agent.bars.iloc[-1]["volume"] == 500.0

        # Late 16:27:59 tick arriving after 16:27 bar already closed live must NOT reopen/overwrite it
        agent.handle_market_data({"symbol": "MESZ6", "price": 7546.75, "volume": 1, "is_trade": True, "timestamp": "2026-11-13T16:27:59Z"})
        assert agent.current_bar is None
        assert len(agent.bars) == 1
        assert agent.bars.iloc[-1]["volume"] == 500.0
        assert agent.bars.iloc[-1]["close"] == 7544.0

    def test_supervisor_seconds_until_next_globex_open(self):
        from datetime import datetime, timezone

        # Thursday 17:58:30 ET == 22:58:30 UTC (Nov is EST = UTC-5)
        dt_preopen = datetime(2026, 11, 12, 22, 58, 30, tzinfo=timezone.utc)
        secs = SupervisorAgent._seconds_until_next_globex_open(dt_preopen)
        assert secs == 90.0

        # Wednesday 17:30:00 ET == 22:30:00 UTC
        dt_maint = datetime(2026, 11, 11, 22, 30, 0, tzinfo=timezone.utc)
        assert SupervisorAgent._seconds_until_next_globex_open(dt_maint) == 1800.0

        # Friday 17:30:00 ET == 22:30:00 UTC (weekend closure, not opening at 18:00 ET)
        dt_fri = datetime(2026, 11, 13, 22, 30, 0, tzinfo=timezone.utc)
        assert SupervisorAgent._seconds_until_next_globex_open(dt_fri) is None

        # Sunday 17:59:00 ET == 22:59:00 UTC
        dt_sun = datetime(2026, 11, 15, 22, 59, 0, tzinfo=timezone.utc)
        assert SupervisorAgent._seconds_until_next_globex_open(dt_sun) == 60.0


