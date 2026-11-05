"""
Unit tests for session logging in trade persistence, database, and dashboard data feeds.
"""
import datetime
import tempfile
import pytest
import pandas as pd
from zoneinfo import ZoneInfo

from src.utils.session_utils import (
    parse_timestamp_to_utc,
    get_session_regime,
    get_trade_session,
)
from src.persistence.database import DatabaseService


def test_session_regime_classification():
    # 1. NY RTH: 10:00 AM ET = 14:00 UTC
    rth_ts = pd.to_datetime("2026-09-29 14:00:00", utc=True)
    assert get_session_regime(rth_ts) == "NY RTH"
    assert get_trade_session(rth_ts) == "NY RTH"

    # 2. NY PRE: 08:30 AM ET = 12:30 UTC
    pre_ts = pd.to_datetime("2026-09-29 12:30:00", utc=True)
    assert get_session_regime(pre_ts) == "NY PRE"
    assert get_trade_session(pre_ts) == "NY PRE"

    # 3. ASIAN: 20:00 ET (8 PM ET) = 00:00 UTC next day
    asian_ts = pd.to_datetime("2026-09-30 00:00:00", utc=True)
    assert get_session_regime(asian_ts) == "ASIAN"
    assert get_trade_session(asian_ts) == "ASIAN"

    # 4. Asian post-midnight: 01:30 AM ET = 05:30 UTC
    asian_late_ts = pd.to_datetime("2026-09-30 05:30:00", utc=True)
    assert get_session_regime(asian_late_ts) == "ASIAN"
    assert get_trade_session(asian_late_ts) == "ASIAN"

    # 5. LONDON: 04:00 AM ET = 08:00 UTC
    london_ts = pd.to_datetime("2026-09-29 08:00:00", utc=True)
    assert get_session_regime(london_ts, include_london=False) == "GLOBEX"
    assert get_session_regime(london_ts, include_london=True) == "LONDON"
    assert get_trade_session(london_ts) == "LONDON"

    # 6. Weekend closure: Saturday 12:00 PM ET
    sat_ts = pd.to_datetime("2026-10-03 16:00:00", utc=True)
    assert get_session_regime(sat_ts) == "CLOSED"
    assert get_trade_session(sat_ts) == "GLOBEX"

    # 7. CME Maintenance: Tuesday 17:30 ET = 21:30 UTC
    maint_ts = pd.to_datetime("2026-09-29 21:30:00", utc=True)
    assert get_session_regime(maint_ts) == "MAINTENANCE"
    assert get_trade_session(maint_ts) == "GLOBEX"

    # 8. Epoch inputs
    epoch_rth_s = 1790690400  # 2026-09-29 14:00:00 UTC
    assert get_session_regime(epoch_rth_s) == "NY RTH"
    assert get_session_regime(epoch_rth_s * 1000) == "NY RTH"
    assert get_session_regime(epoch_rth_s * 1000000) == "NY RTH"
    assert get_session_regime(epoch_rth_s * 1000000000) == "NY RTH"

    # 9. Synthetic index guards
    assert get_session_regime(0, fallback_to_now=False) == "UNKNOWN"
    assert get_session_regime(100, fallback_to_now=False) == "UNKNOWN"


def test_database_session_persistence_and_query():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        db = DatabaseService({"enabled": True, "sqlite_path": tmp.name})
        assert db.connected

        # Persist a trade context in NY RTH
        opened_at_rth = "2026-09-29T14:15:00+00:00"  # 10:15 AM ET
        db.persist_trade_context(
            trade_key="TEST_001",
            symbol="MESZ6",
            strategy_name="mtf_trend_scalper",
            side="Buy",
            quantity=1,
            order_id="ORD_101",
            entry_price=5750.25,
            opened_at=opened_at_rth,
            strategy_params={"lookback": 20},
            selected_signal={"side": "Buy"},
            candidates=[],
            blender_metadata={},
            route_context={},
            discovery_context={},
            account_snapshot={"account_id": "99999"},
            chart_artifacts={},
        )

        # Context row should have session = 'NY RTH'
        ctx = db.get_trade_context("TEST_001")
        assert ctx is not None
        assert ctx.get("session") == "NY RTH"

        # Persist trade outcome
        closed_at_rth = "2026-09-29T14:45:00+00:00"  # 10:45 AM ET
        db.persist_trade_outcome(
            trade_key="TEST_001",
            symbol="MESZ6",
            order_id="ORD_101",
            account_id="99999",
            pnl=125.0,
            close_type="tp",
            closed_at=closed_at_rth,
            outcome={"pnl": 125.0},
        )

        # Query recent trades
        recent = db.list_recent_trades(5, account_id="99999")
        assert len(recent) == 1
        trade = recent[0]
        assert trade["symbol"] == "MESZ6"
        assert trade["pnl"] == 125.0
        assert trade["session"] == "NY RTH"

        # Persist an Asian session trade
        opened_at_asian = "2026-09-30T00:30:00+00:00"  # 8:30 PM ET
        db.persist_trade_context(
            trade_key="TEST_002",
            symbol="MCLX6",
            strategy_name="asian_breakout",
            side="Sell",
            quantity=1,
            order_id="ORD_102",
            entry_price=72.50,
            opened_at=opened_at_asian,
            session="ASIAN",
            strategy_params={},
            selected_signal={"side": "Sell"},
            candidates=[],
            blender_metadata={},
            route_context={"session": "ASIAN"},
            discovery_context={},
            account_snapshot={"account_id": "99999"},
            chart_artifacts={},
        )
        closed_at_asian = "2026-09-30T01:00:00+00:00"  # 9:00 PM ET
        db.persist_trade_outcome(
            trade_key="TEST_002",
            symbol="MCLX6",
            order_id="ORD_102",
            account_id="99999",
            session="ASIAN",
            pnl=-50.0,
            close_type="sl",
            closed_at=closed_at_asian,
            outcome={"pnl": -50.0},
        )

        recent = db.list_recent_trades(5, account_id="99999")
        assert len(recent) == 2
        assert recent[0]["symbol"] == "MCLX6"
        assert recent[0]["session"] == "ASIAN"
        assert recent[1]["symbol"] == "MESZ6"
        assert recent[1]["session"] == "NY RTH"

        db.close()


def test_database_session_migration_backfill():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        # Create an older table schema without session column
        import sqlite3
        conn = sqlite3.connect(tmp.name)
        conn.execute("""
            CREATE TABLE trade_contexts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_key TEXT NOT NULL UNIQUE,
                symbol TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                order_id TEXT,
                entry_price REAL,
                opened_at TEXT NOT NULL,
                strategy_params_json TEXT NOT NULL,
                selected_signal_json TEXT NOT NULL,
                candidates_json TEXT NOT NULL,
                blender_metadata_json TEXT NOT NULL,
                route_context_json TEXT NOT NULL,
                discovery_context_json TEXT NOT NULL,
                account_snapshot_json TEXT NOT NULL,
                chart_artifacts_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
        """)
        conn.execute("""
            CREATE TABLE trade_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_key TEXT NOT NULL UNIQUE,
                symbol TEXT NOT NULL,
                order_id TEXT,
                account_id TEXT,
                pnl REAL NOT NULL,
                close_type TEXT NOT NULL,
                closed_at TEXT NOT NULL,
                outcome_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
        """)
        # Insert a historical trade outcome (closed at 14:00 UTC = 10:00 AM ET)
        conn.execute("""
            INSERT INTO trade_outcomes (trade_key, symbol, pnl, close_type, closed_at, outcome_json, created_at)
            VALUES ('OLD_001', 'MESZ6', 75.0, 'tp', '2026-09-29T14:00:00+00:00', '{}', '2026-09-29T14:00:00+00:00')
        """)
        conn.commit()
        conn.close()

        # Connect with DatabaseService to trigger migration and backfill
        db = DatabaseService({"enabled": True, "sqlite_path": tmp.name})
        trades = db.list_recent_trades(5)
        assert len(trades) == 1
        assert trades[0]["symbol"] == "MESZ6"
        assert trades[0]["session"] == "NY RTH"
        db.close()
