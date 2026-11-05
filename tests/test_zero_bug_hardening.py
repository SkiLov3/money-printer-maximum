import time
import pytest
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock
import requests

from src.api.projectx_client import ProjectXClient
from src.agents.risk_agent import RiskAgent
from src.agents.data_agent import DataAgent, parse_market_timestamp
from src.strategies.engine import StrategyEngine
from src.persistence.database import DatabaseService


# ---------------------------------------------------------------------------
# 1. ProjectXClient Tests
# ---------------------------------------------------------------------------

def test_request_with_refresh_timeout_non_idempotent():
    """Verify that allow_retry_on_timeout=False does not retry on network timeout."""
    client = ProjectXClient()
    client.access_token = "fake_jwt"
    client.token_acquired_at = time.time()

    mock_method = MagicMock(side_effect=requests.exceptions.Timeout("Connection timed out"))
    with pytest.raises(requests.exceptions.Timeout):
        client._request_with_refresh(mock_method, "https://api.topstepx.com/api/Order/place", allow_retry_on_timeout=False)
    
    assert mock_method.call_count == 1, "Non-idempotent request must NOT retry on timeout"


def test_request_with_refresh_double_checked_lock():
    """Verify thundering herd token refresh double-check avoids redundant logins."""
    client = ProjectXClient()
    client.access_token = "jwt_old"
    client.token_acquired_at = time.time() - 3600

    resp_401 = MagicMock()
    resp_401.status_code = 401
    resp_200 = MagicMock()
    resp_200.status_code = 200

    mock_method = MagicMock(side_effect=[resp_401, resp_200])
    
    # Simulate that another thread already refreshed token 1 second ago
    client.login = MagicMock(return_value=True)
    client.token_acquired_at = time.time() - 1.0  # Fresh!

    resp = client._request_with_refresh(mock_method, "https://api.topstepx.com/api/test")
    assert resp.status_code == 200
    assert client.login.call_count == 0, "Should use already-refreshed token within 5s window"


def test_stop_streaming_finally_resets_flags():
    """Verify that stop_streaming and stop_user_streaming reset connected flags even if stop() raises."""
    client = ProjectXClient()
    client.is_connected = True
    client.user_is_connected = True
    
    bad_conn = MagicMock()
    bad_conn.stop.side_effect = RuntimeError("WebSocket died")
    client.connection = bad_conn
    client.user_connection = bad_conn

    client.stop_streaming()
    assert client.is_connected is False

    client.stop_user_streaming()
    assert client.user_is_connected is False


def test_reconnect_all_streams_debounce():
    """Verify reconnect_all_streams debounces rapid calls within 10s."""
    client = ProjectXClient()
    client.stop_streaming = MagicMock()
    client.stop_user_streaming = MagicMock()
    client.start_streaming = MagicMock()
    client.start_user_streaming = MagicMock()
    client.login = MagicMock(return_value=True)

    client.reconnect_all_streams(force=True)
    assert client.start_streaming.call_count == 1

    # Immediate second call with force=False must be throttled
    client.reconnect_all_streams(force=False)
    assert client.start_streaming.call_count == 1


def test_retrieve_bars_auth_failure_return_shape():
    """Verify retrieve_bars returns dict shape on token failure."""
    client = ProjectXClient()
    client._ensure_token = MagicMock(return_value=False)
    res = client.retrieve_bars("CON.123", "2026-09-01T00:00:00Z", "2026-09-02T00:00:00Z")
    assert isinstance(res, dict)
    assert "bars" in res
    assert res["bars"] == []


# ---------------------------------------------------------------------------
# 2. RiskAgent Tests
# ---------------------------------------------------------------------------

def test_risk_agent_startup_zero_balance_no_false_mll():
    """Verify RiskAgent does NOT falsely report MLL hit when balance is not yet received from broker."""
    agent = RiskAgent("12345")
    agent.current_balance = Decimal("0.0")
    agent.minimum_balance = Decimal("48000.0")

    is_hit, reason = agent.is_risk_limit_hit()
    assert not is_hit, f"False MLL hit on startup: {reason}"


def test_risk_agent_trailing_floor_stops_at_starting_balance():
    """Verify trailing floor stops trailing at starting balance ($50,000) per Topstep rules."""
    agent = RiskAgent("12345")
    agent.starting_balance = Decimal("50000.0")
    agent.max_drawdown = Decimal("2000.0")
    agent.minimum_balance = Decimal("48000.0")

    # Account rises to $51,000 -> floor trails to $49,000
    agent.update_account_state(51000.0, 1000.0, 0)
    assert agent.minimum_balance == Decimal("49000.0")

    # Account rises to $52,000 -> floor trails to $50,000 (starting balance)
    agent.update_account_state(52000.0, 2000.0, 0)
    assert agent.minimum_balance == Decimal("50000.0")

    # Account rises to $53,000 -> floor MUST NOT exceed starting balance ($50,000)
    agent.update_account_state(53000.0, 3000.0, 0)
    assert agent.minimum_balance == Decimal("50000.0")


# ---------------------------------------------------------------------------
# 3. DataAgent & Timestamp Tests
# ---------------------------------------------------------------------------

def test_data_agent_symbol_matching_contract_month():
    """Verify DataAgent does not conflate different contract months (e.g. MESM6 vs MESZ6)."""
    agent = DataAgent("MESM6")
    # Same contract with 2-digit year format must match
    assert agent._matches_symbol("MESM26") is True
    # Different contract month MUST NOT match
    assert agent._matches_symbol("MESZ6") is False
    assert agent._matches_symbol("MESZ26") is False
    # Base ticker matches
    assert agent._matches_symbol("MES") is True


def test_data_agent_total_bars_generated_preservation():
    """Verify backfilling historical bars does not regress total_bars_generated."""
    agent = DataAgent("MESZ6", max_bars=100)
    agent.total_bars_generated = 150  # Simulating live bot that has run for hours

    # Simulate backfill of 50 bars
    dummy_bars = [
        {"startTime": f"2026-09-30T10:{i:02d}:00Z", "open": 5000.0, "high": 5001.0, "low": 4999.0, "close": 5000.5, "volume": 100}
        for i in range(50)
    ]
    agent.load_historical_bars(dummy_bars)
    assert agent.total_bars_generated >= 150, "total_bars_generated must not be regressed by backfill"


def test_parse_market_timestamp_hardening():
    """Verify parse_market_timestamp handles NaN, Inf, non-positive, and empty inputs gracefully."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    
    # Empty / whitespace
    assert abs((parse_market_timestamp("") - now).total_seconds()) < 5
    assert abs((parse_market_timestamp("   ") - now).total_seconds()) < 5
    assert abs((parse_market_timestamp(None) - now).total_seconds()) < 5

    # NaN / Inf / Negative
    assert abs((parse_market_timestamp(float("nan")) - now).total_seconds()) < 5
    assert abs((parse_market_timestamp(float("inf")) - now).total_seconds()) < 5
    assert abs((parse_market_timestamp(0) - now).total_seconds()) < 5
    assert abs((parse_market_timestamp(-1000) - now).total_seconds()) < 5

    # Valid seconds epoch
    parsed = parse_market_timestamp(1727712000)  # 2024-09-30 16:00:00 UTC
    assert parsed.year == 2024


# ---------------------------------------------------------------------------
# 4. StrategyEngine & CME Schedule Tests
# ---------------------------------------------------------------------------

def test_cme_weekend_and_maintenance_closure():
    """Verify _compute_session_regime returns CLOSED on weekends and MAINTENANCE Mon-Thu 17-18 ET."""
    engine = StrategyEngine("MESZ6", 1, {"sl_ticks": 40, "tp_ticks": 80})
    
    # Saturday 12:00 PM ET -> CLOSED
    sat_dt = datetime(2026, 10, 3, 16, 0, 0, tzinfo=timezone.utc) # Saturday 12:00 ET
    assert engine._compute_session_regime(sat_dt.isoformat()) == "CLOSED"

    # Sunday 2:00 PM ET -> CLOSED
    sun_dt = datetime(2026, 10, 4, 18, 0, 0, tzinfo=timezone.utc) # Sunday 14:00 ET
    assert engine._compute_session_regime(sun_dt.isoformat()) == "CLOSED"

    # Tuesday 17:30 ET -> MAINTENANCE
    tue_dt = datetime(2026, 10, 6, 21, 30, 0, tzinfo=timezone.utc) # Tuesday 17:30 ET
    assert engine._compute_session_regime(tue_dt.isoformat()) == "MAINTENANCE"

    # Tuesday 20:00 ET -> ASIAN
    tue_asian = datetime(2026, 10, 7, 0, 0, 0, tzinfo=timezone.utc) # Tuesday 20:00 ET
    assert engine._compute_session_regime(tue_asian.isoformat()) == "ASIAN"


# ---------------------------------------------------------------------------
# 5. DatabaseService Multi-Account Isolation Tests
# ---------------------------------------------------------------------------

def test_database_list_recent_trades_account_isolation(tmp_path):
    """Verify list_recent_trades only returns trades matching the exact account_id."""
    db_file = tmp_path / "test.db"
    db = DatabaseService({"backend": "sqlite", "sqlite_path": str(db_file), "enabled": True})

    # Record trade for account A
    now_iso = datetime.now(timezone.utc).isoformat()
    db.persist_trade_context(
        trade_key="trade_A",
        symbol="MESZ6",
        strategy_name="test_strat",
        side="Buy",
        quantity=1,
        order_id="ORD-1",
        entry_price=5000.0,
        opened_at=now_iso,
        strategy_params={},
        selected_signal={},
        candidates=[],
        blender_metadata={},
        route_context={"account_id": "11111"},
        discovery_context={},
        account_snapshot={"account_id": "11111"},
        chart_artifacts={},
    )
    db.persist_trade_outcome(
        trade_key="trade_A",
        symbol="MESZ6",
        order_id="ORD-1",
        account_id="11111",
        pnl=150.0,
        close_type="Target",
        closed_at=datetime.now(timezone.utc).isoformat(),
        outcome={"account_id": "11111"},
    )

    # Query account B (should return 0 trades, not fallback to account A)
    trades_b = db.list_recent_trades(account_id="22222")
    assert len(trades_b) == 0, "Account B must not see Account A trades"

    # Query account A
    trades_a = db.list_recent_trades(account_id="11111")
    assert len(trades_a) == 1
    assert trades_a[0]["symbol"] == "MESZ6"

    db.close()
    assert db.connected is False
