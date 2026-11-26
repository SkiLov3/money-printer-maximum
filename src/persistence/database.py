from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any


class DatabaseService:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", False))
        self.backend = str(self.config.get("backend", "sqlite")).lower()
        self.sqlite_path = self.config.get("sqlite_path", "logs/swarm.db")
        self.logger = logging.getLogger(__name__)
        self.connected = False
        self.reason = "disabled"
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock() # FIX: Ensure thread-safety for multi-threaded SQLite access
        self._connect()

    def _connect(self) -> None:
        if not self.enabled:
            self.connected = False
            self.reason = "disabled"
            return
        if self.backend == "sqlite":
            db_dir = os.path.dirname(self.sqlite_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
            # FIX H11: Use check_same_thread=False but PROTECT with self._lock
            self._conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            # FIX: Use DELETE journal mode + synchronous=FULL for Docker Desktop VirtioFS bind mounts.
            # SQLite WAL relies on shared-memory mmap (.db-shm) which does NOT synchronize across
            # the macOS host kernel and Linux container VM kernel, causing host readers to wipe
            # uncommitted container WAL frames. DELETE mode writes directly to the main DB file.
            self._conn.execute("PRAGMA journal_mode=DELETE")
            self._conn.execute("PRAGMA busy_timeout=10000")
            self._conn.execute("PRAGMA synchronous=FULL")
            self._init_sqlite()
            self.connected = True
            self.reason = "connected"
            self._log_connection_probe()
            return

        try:
            import psycopg  # type: ignore

            dsn = self.config.get("dsn", "")
            if not dsn:
                self.reason = "missing_dsn"
                return
            with psycopg.connect(dsn):
                self.connected = True
                self.reason = "connected"
        except Exception as exc:
            self.connected = False
            self.reason = f"degraded:{exc.__class__.__name__}"
            self.logger.warning("Database service degraded: %s", exc)

    def _log_connection_probe(self) -> None:
        """Log which file this connection really writes to and its newest rows.

        Persistence failures were previously invisible; this makes a stale or
        swapped DB file obvious in logs/trades.log at startup.
        """
        try:
            assert self._conn is not None
            outcomes, latest_out = self._conn.execute(
                "SELECT COUNT(*), MAX(closed_at) FROM trade_outcomes"
            ).fetchone()
            ctxs, latest_ctx = self._conn.execute(
                "SELECT COUNT(*), MAX(opened_at) FROM trade_contexts"
            ).fetchone()
            inode = os.stat(self.sqlite_path).st_ino
            self.logger.info(
                "[DB] connected path=%s inode=%s trade_outcomes=%s (latest %s) trade_contexts=%s (latest %s)",
                os.path.abspath(self.sqlite_path), inode, outcomes, latest_out, ctxs, latest_ctx,
            )
        except Exception as exc:  # observational only
            self.logger.warning("[DB] connection probe failed: %s", exc)

    def _init_sqlite(self) -> None:
        assert self._conn is not None
        cur = self._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS zones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                zone_type TEXT NOT NULL,
                price_low REAL NOT NULL,
                price_high REAL NOT NULL,
                strength REAL NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'active',
                touch_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS validation_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                verdict TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                recommendations_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS health_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                accepted INTEGER NOT NULL,
                reason TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                namespace TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(namespace, entity_id)
            );

            CREATE TABLE IF NOT EXISTS discovery_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS meta_optimizer_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                epoch INTEGER NOT NULL,
                score REAL NOT NULL,
                summary_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS llm_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                epoch INTEGER NOT NULL,
                reasoning TEXT NOT NULL,
                config_changes_json TEXT NOT NULL,
                code_patches_json TEXT NOT NULL,
                raw_output_path TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trade_contexts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_key TEXT NOT NULL UNIQUE,
                symbol TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                order_id TEXT,
                entry_price REAL,
                opened_at TEXT NOT NULL,
                session TEXT,
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

            CREATE TABLE IF NOT EXISTS trade_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_key TEXT NOT NULL UNIQUE,
                symbol TEXT NOT NULL,
                order_id TEXT,
                account_id TEXT,
                session TEXT,
                pnl REAL NOT NULL,
                close_type TEXT NOT NULL,
                closed_at TEXT NOT NULL,
                outcome_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS learning_batches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT NOT NULL UNIQUE,
                provider TEXT NOT NULL,
                status TEXT NOT NULL,
                item_count INTEGER NOT NULL DEFAULT 0,
                raw_output_path TEXT NOT NULL DEFAULT '',
                html_report_path TEXT NOT NULL DEFAULT '',
                json_report_path TEXT NOT NULL DEFAULT '',
                summary_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                completed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS learning_batch_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT NOT NULL,
                trade_key TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                UNIQUE(batch_id, trade_key)
            );

            CREATE TABLE IF NOT EXISTS strategy_learnings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                learning_id TEXT NOT NULL UNIQUE,
                batch_id TEXT NOT NULL,
                trade_key TEXT,
                symbol TEXT,
                category TEXT NOT NULL DEFAULT 'hypothesis',
                summary TEXT NOT NULL,
                evidence_json TEXT NOT NULL DEFAULT '{}',
                proposed_config_diff_json TEXT NOT NULL DEFAULT '{}',
                validation_status TEXT NOT NULL DEFAULT 'pending_optimizer_validation',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS candidate_promotions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL UNIQUE,
                batch_id TEXT NOT NULL,
                config_path TEXT NOT NULL DEFAULT '',
                config_diff_json TEXT NOT NULL DEFAULT '{}',
                gate_status_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'pending_validation',
                active_from_session TEXT,
                created_at TEXT NOT NULL,
                promoted_at TEXT
            );
            """
        )
        # Ensure account_id column and indices exist
        try:
            cur.execute("ALTER TABLE trade_outcomes ADD COLUMN account_id TEXT")
        except Exception:
            pass
        try:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_trade_outcomes_account ON trade_outcomes(account_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_trade_outcomes_closed_at ON trade_outcomes(closed_at DESC)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_trade_outcomes_account_closed ON trade_outcomes(account_id, closed_at DESC)")
        except Exception:
            pass
        try:
            cur.execute("""
                UPDATE trade_outcomes
                SET account_id = (
                    SELECT json_extract(c.account_snapshot_json, '$.account_id')
                    FROM trade_contexts c
                    WHERE c.trade_key = trade_outcomes.trade_key
                )
                WHERE account_id IS NULL AND EXISTS (
                    SELECT 1 FROM trade_contexts c WHERE c.trade_key = trade_outcomes.trade_key
                )
            """)
        except Exception:
            pass

        # Ensure session columns and indices exist
        try:
            cur.execute("ALTER TABLE trade_outcomes ADD COLUMN session TEXT")
        except Exception:
            pass
        try:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_trade_outcomes_session ON trade_outcomes(session)")
        except Exception:
            pass
        try:
            cur.execute("ALTER TABLE trade_contexts ADD COLUMN session TEXT")
        except Exception:
            pass
        try:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_trade_contexts_session ON trade_contexts(session)")
        except Exception:
            pass

        # Backfill session for trade_outcomes if NULL
        try:
            from src.utils.session_utils import get_trade_session
            outcome_rows = cur.execute("SELECT id, closed_at, outcome_json FROM trade_outcomes WHERE session IS NULL").fetchall()
            for r in outcome_rows:
                r_id = r[0]
                c_at = r[1]
                sess = None
                if r[2]:
                    try:
                        o_data = json.loads(r[2])
                        sess = o_data.get("session") or (o_data.get("closed_position") or {}).get("session")
                    except Exception:
                        pass
                if not sess and c_at:
                    sess = get_trade_session(c_at)
                if sess:
                    cur.execute("UPDATE trade_outcomes SET session = ? WHERE id = ?", (sess, r_id))
        except Exception:
            pass

        # Backfill session for trade_contexts if NULL
        try:
            from src.utils.session_utils import get_trade_session
            ctx_rows = cur.execute("SELECT id, opened_at, route_context_json FROM trade_contexts WHERE session IS NULL").fetchall()
            for r in ctx_rows:
                r_id = r[0]
                o_at = r[1]
                sess = None
                if r[2]:
                    try:
                        rt_data = json.loads(r[2])
                        sess = rt_data.get("session") or rt_data.get("session_regime")
                    except Exception:
                        pass
                if not sess and o_at:
                    sess = get_trade_session(o_at)
                if sess:
                    cur.execute("UPDATE trade_contexts SET session = ? WHERE id = ?", (sess, r_id))
        except Exception:
            pass
        self._conn.commit()

    def health_check(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "connected": self.connected,
            "reason": self.reason,
            "backend": self.backend,
            "sqlite_path": self.sqlite_path if self.backend == "sqlite" else "",
        }

    def persist_zone(self, symbol: str, zone: Any) -> None:
        if not (self.connected and self._conn is not None and self.backend == "sqlite"):
            return
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            # FIX: Robust type casting for SQLite REAL/INTEGER compatibility
            self._conn.execute(
                """
                INSERT INTO zones (symbol, zone_type, price_low, price_high, strength, status, touch_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    getattr(zone, "zone_type", "generic"),
                    float(getattr(zone, "price_low", 0.0)),
                    float(getattr(zone, "price_high", 0.0)),
                    float(getattr(zone, "strength", 0.0)),
                    str(getattr(zone, "status", "active")),
                    int(getattr(zone, "touch_count", 0)),
                    now,
                    now,
                ),
            )
            self._conn.commit()

    def list_active_zones(self, symbol: str) -> list[dict[str, Any]]:
        if not (self.connected and self._conn is not None and self.backend == "sqlite"):
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM zones WHERE symbol = ? AND status != 'invalidated' ORDER BY updated_at DESC",
                (symbol,),
            ).fetchall()
        return [dict(row) for row in rows]

    def persist_validation_result(self, symbol: str, verdict: str, metrics: dict[str, Any], recommendations: list[str]) -> None:
        if not (self.connected and self._conn is not None and self.backend == "sqlite"):
            return
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO validation_runs (symbol, verdict, metrics_json, recommendations_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    verdict,
                    json.dumps(metrics, default=str),
                    json.dumps(recommendations, default=str),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._conn.commit()

    def list_validation_runs(self, symbol: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        if not (self.connected and self._conn is not None and self.backend == "sqlite"):
            return []
        with self._lock:
            if symbol:
                rows = self._conn.execute(
                    "SELECT * FROM validation_runs WHERE symbol = ? ORDER BY created_at DESC LIMIT ?",
                    (symbol, int(limit)),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM validation_runs ORDER BY created_at DESC LIMIT ?",
                    (int(limit),),
                ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["metrics"] = json.loads(item.pop("metrics_json"))
            item["recommendations"] = json.loads(item.pop("recommendations_json"))
            result.append(item)
        return result

    def persist_health_event(self, symbol: str, accepted: bool, reason: str = "") -> None:
        if not (self.connected and self._conn is not None and self.backend == "sqlite"):
            return
        with self._lock:
            self._conn.execute(
                "INSERT INTO health_events (symbol, accepted, reason, created_at) VALUES (?, ?, ?, ?)",
                (symbol, 1 if accepted else 0, reason, datetime.now(timezone.utc).isoformat()),
            )
            self._conn.commit()

    def persist_embedding(self, namespace: str, entity_id: str, symbol: str, vector: list[float], metadata: dict[str, Any]) -> None:
        if not (self.connected and self._conn is not None and self.backend == "sqlite"):
            return
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO embeddings (namespace, entity_id, symbol, vector_json, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, entity_id) DO UPDATE SET
                    symbol = excluded.symbol,
                    vector_json = excluded.vector_json,
                    metadata_json = excluded.metadata_json,
                    created_at = excluded.created_at
                """,
                (
                    namespace,
                    entity_id,
                    symbol,
                    json.dumps(vector),
                    json.dumps(metadata, default=str),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._conn.commit()

    def query_similar_embeddings(self, namespace: str, symbol: str, vector: list[float], limit: int = 5) -> list[dict[str, Any]]:
        if not (self.connected and self._conn is not None and self.backend == "sqlite"):
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT entity_id, vector_json, metadata_json, created_at FROM embeddings WHERE namespace = ? AND symbol = ?",
                (namespace, symbol),
            ).fetchall()
        scored = []
        for row in rows:
            candidate = json.loads(row["vector_json"])
            similarity = self._cosine_similarity(vector, candidate)
            scored.append(
                {
                    "entity_id": row["entity_id"],
                    "similarity": similarity,
                    "metadata": json.loads(row["metadata_json"]),
                    "created_at": row["created_at"],
                }
            )
        scored.sort(key=lambda item: item["similarity"], reverse=True)
        return scored[:limit]

    def persist_discovery_run(self, symbol: str, summary: dict[str, Any]) -> None:
        if not (self.connected and self._conn is not None and self.backend == "sqlite"):
            return
        with self._lock:
            self._conn.execute(
                "INSERT INTO discovery_runs (symbol, summary_json, created_at) VALUES (?, ?, ?)",
                (symbol, json.dumps(summary, default=str), datetime.now(timezone.utc).isoformat()),
            )
            self._conn.commit()

    def persist_meta_optimizer_run(self, symbol: str, epoch: int, score: float, summary: dict[str, Any]) -> None:
        if not (self.connected and self._conn is not None and self.backend == "sqlite"):
            return
        with self._lock:
            # FIX: Robust type casting for SQLite compatibility
            self._conn.execute(
                "INSERT INTO meta_optimizer_runs (symbol, epoch, score, summary_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    symbol,
                    int(epoch),
                    float(score),
                    json.dumps(summary, default=str),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._conn.commit()

    def list_meta_optimizer_runs(self, symbol: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        if not (self.connected and self._conn is not None and self.backend == "sqlite"):
            return []
        with self._lock:
            if symbol:
                rows = self._conn.execute(
                    "SELECT * FROM meta_optimizer_runs WHERE symbol = ? ORDER BY created_at DESC LIMIT ?",
                    (symbol, int(limit)),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM meta_optimizer_runs ORDER BY created_at DESC LIMIT ?",
                    (int(limit),),
                ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["summary"] = json.loads(item.pop("summary_json"))
            result.append(item)
        return result

    def persist_llm_analysis(
        self,
        symbol: str,
        epoch: int,
        reasoning: str,
        config_changes: dict[str, Any],
        code_patches: list[dict[str, Any]],
        raw_output_path: str,
    ) -> None:
        if not (self.connected and self._conn is not None and self.backend == "sqlite"):
            return
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO llm_analysis (symbol, epoch, reasoning, config_changes_json, code_patches_json, raw_output_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    int(epoch),
                    reasoning,
                    json.dumps(config_changes, default=str),
                    json.dumps(code_patches, default=str),
                    raw_output_path,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._conn.commit()

    def list_llm_analysis(self, symbol: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        if not (self.connected and self._conn is not None and self.backend == "sqlite"):
            return []
        with self._lock:
            if symbol:
                rows = self._conn.execute(
                    "SELECT * FROM llm_analysis WHERE symbol = ? ORDER BY created_at DESC LIMIT ?",
                    (symbol, int(limit)),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM llm_analysis ORDER BY created_at DESC LIMIT ?",
                    (int(limit),),
                ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["config_changes"] = json.loads(item.pop("config_changes_json"))
            item["code_patches"] = json.loads(item.pop("code_patches_json"))
            result.append(item)
        return result

    def persist_trade_context(
        self,
        *,
        trade_key: str,
        symbol: str,
        strategy_name: str,
        side: str,
        quantity: int,
        order_id: str | None,
        entry_price: float | None,
        opened_at: str,
        strategy_params: dict[str, Any],
        selected_signal: dict[str, Any],
        candidates: list[dict[str, Any]],
        blender_metadata: dict[str, Any],
        route_context: dict[str, Any],
        discovery_context: dict[str, Any],
        account_snapshot: dict[str, Any],
        chart_artifacts: dict[str, Any],
        session: str | None = None,
    ) -> None:
        if not (self.connected and self._conn is not None and self.backend == "sqlite"):
            return
        now = datetime.now(timezone.utc).isoformat()
        if session is None:
            session = route_context.get("session") or route_context.get("session_regime")
        if not session and opened_at:
            from src.utils.session_utils import get_trade_session
            session = get_trade_session(opened_at)
        with self._lock:
            try:
                self._conn.execute(
                    """
                    INSERT INTO trade_contexts (
                        trade_key, symbol, strategy_name, side, quantity, order_id, entry_price, opened_at,
                        session, strategy_params_json, selected_signal_json, candidates_json, blender_metadata_json,
                        route_context_json, discovery_context_json, account_snapshot_json, chart_artifacts_json, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(trade_key) DO UPDATE SET
                        symbol = excluded.symbol,
                        strategy_name = excluded.strategy_name,
                        side = excluded.side,
                        quantity = excluded.quantity,
                        order_id = excluded.order_id,
                        entry_price = excluded.entry_price,
                        opened_at = excluded.opened_at,
                        session = COALESCE(excluded.session, trade_contexts.session),
                        strategy_params_json = excluded.strategy_params_json,
                        selected_signal_json = excluded.selected_signal_json,
                        candidates_json = excluded.candidates_json,
                        blender_metadata_json = excluded.blender_metadata_json,
                        route_context_json = excluded.route_context_json,
                        discovery_context_json = excluded.discovery_context_json,
                        account_snapshot_json = excluded.account_snapshot_json,
                        chart_artifacts_json = excluded.chart_artifacts_json
                    """,
                    (
                        trade_key,
                        symbol,
                        strategy_name,
                        side,
                        int(quantity),
                        order_id,
                        float(entry_price) if entry_price is not None else None,
                        opened_at,
                        str(session) if session is not None else None,
                        json.dumps(strategy_params, default=str),
                        json.dumps(selected_signal, default=str),
                        json.dumps(candidates, default=str),
                        json.dumps(blender_metadata, default=str),
                        json.dumps(route_context, default=str),
                        json.dumps(discovery_context, default=str),
                        json.dumps(account_snapshot, default=str),
                        json.dumps(chart_artifacts, default=str),
                        now,
                    ),
                )
                self._conn.commit()
                self.logger.info("[DB] persisted trade_context key=%s symbol=%s", trade_key, symbol)
            except sqlite3.Error as exc:
                self.logger.error("Failed to persist trade context (%s): %s", trade_key, exc)
                try:
                    self._conn.rollback()
                except Exception:
                    pass

    def persist_trade_outcome(
        self,
        *,
        trade_key: str,
        symbol: str,
        order_id: str | None,
        account_id: str | None = None,
        session: str | None = None,
        pnl: float,
        close_type: str,
        closed_at: str,
        outcome: dict[str, Any],
    ) -> None:
        if not (self.connected and self._conn is not None and self.backend == "sqlite"):
            return
        now = datetime.now(timezone.utc).isoformat()
        if session is None:
            session = outcome.get("session") or (outcome.get("closed_position") or {}).get("session")
        if not session and closed_at:
            from src.utils.session_utils import get_trade_session
            session = get_trade_session(closed_at)
        outcome_copy = dict(outcome)
        outcome_copy["session"] = session
        with self._lock:
            try:
                self._conn.execute(
                    """
                    INSERT INTO trade_outcomes (trade_key, symbol, order_id, account_id, session, pnl, close_type, closed_at, outcome_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(trade_key) DO UPDATE SET
                        symbol = excluded.symbol,
                        order_id = excluded.order_id,
                        account_id = COALESCE(excluded.account_id, trade_outcomes.account_id),
                        session = COALESCE(excluded.session, trade_outcomes.session),
                        pnl = excluded.pnl,
                        close_type = excluded.close_type,
                        closed_at = excluded.closed_at,
                        outcome_json = excluded.outcome_json
                    """,
                    (
                        trade_key,
                        symbol,
                        order_id,
                        str(account_id) if account_id is not None else None,
                        str(session) if session is not None else None,
                        float(pnl),
                        str(close_type or ""),
                        closed_at,
                        json.dumps(outcome_copy, default=str),
                        now,
                    ),
                )
                self._conn.commit()
                self.logger.info("[DB] persisted trade_outcome key=%s symbol=%s pnl=%.2f", trade_key, symbol, float(pnl))
            except sqlite3.Error as exc:
                self.logger.error("Failed to persist trade outcome (%s): %s", trade_key, exc)
                try:
                    self._conn.rollback()
                except Exception:
                    pass

    def get_trade_context(self, trade_key: str) -> dict[str, Any] | None:
        if not (self.connected and self._conn is not None and self.backend == "sqlite"):
            return None
        with self._lock:
            row = self._conn.execute("SELECT * FROM trade_contexts WHERE trade_key = ?", (trade_key,)).fetchone()
        return self._decode_trade_context_row(row) if row else None

    def list_closed_unanalyzed_trades(self, limit: int = 25, account_id: str | None = None) -> list[dict[str, Any]]:
        if not (self.connected and self._conn is not None and self.backend == "sqlite"):
            return []
        with self._lock:
            if account_id:
                acc_str = str(account_id)
                rows = self._conn.execute(
                    """
                    SELECT 
                        c.*,
                        o.pnl AS outcome_pnl,
                        o.close_type AS outcome_close_type,
                        o.closed_at AS outcome_closed_at,
                        o.outcome_json AS outcome_json,
                        COALESCE(o.session, c.session) AS outcome_session
                    FROM trade_contexts c
                    INNER JOIN trade_outcomes o ON o.trade_key = c.trade_key
                    WHERE NOT EXISTS (
                        SELECT 1 FROM learning_batch_items i
                        WHERE i.trade_key = c.trade_key AND i.status = 'analyzed'
                    )
                    AND (
                        o.account_id = ?
                        OR CAST(json_extract(c.account_snapshot_json, '$.account_id') AS TEXT) = ?
                        OR CAST(json_extract(o.outcome_json, '$.account_id') AS TEXT) = ?
                        OR CAST(json_extract(o.outcome_json, '$.closed_position.account_id') AS TEXT) = ?
                    )
                    ORDER BY o.closed_at ASC
                    LIMIT ?
                    """,
                    (acc_str, acc_str, acc_str, acc_str, int(limit)),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT 
                        c.*,
                        o.pnl AS outcome_pnl,
                        o.close_type AS outcome_close_type,
                        o.closed_at AS outcome_closed_at,
                        o.outcome_json AS outcome_json,
                        COALESCE(o.session, c.session) AS outcome_session
                    FROM trade_contexts c
                    INNER JOIN trade_outcomes o ON o.trade_key = c.trade_key
                    WHERE NOT EXISTS (
                        SELECT 1 FROM learning_batch_items i
                        WHERE i.trade_key = c.trade_key AND i.status = 'analyzed'
                    )
                    ORDER BY o.closed_at ASC
                    LIMIT ?
                    """,
                    (int(limit),),
                ).fetchall()
        return [self._decode_closed_trade_row(row) for row in rows]

    def list_recent_trades(self, limit: int = 10, account_id: str | None = None) -> list[dict[str, Any]]:
        if not (self.connected and self._conn is not None and self.backend == "sqlite"):
            return []
        with self._lock:
            if account_id:
                acc_str = str(account_id)
                rows = self._conn.execute(
                    """
                    SELECT 
                        o.closed_at,
                        o.symbol,
                        COALESCE(c.side, json_extract(o.outcome_json, '$.closed_position.side'), 'Flat') AS side,
                        o.close_type,
                        o.pnl,
                        COALESCE(
                            o.session,
                            c.session,
                            json_extract(o.outcome_json, '$.session'),
                            json_extract(o.outcome_json, '$.closed_position.session'),
                            json_extract(c.route_context_json, '$.session')
                        ) AS session,
                        COALESCE(c.opened_at, json_extract(o.outcome_json, '$.closed_position.opened_at')) AS opened_at
                    FROM trade_outcomes o
                    LEFT JOIN trade_contexts c ON o.trade_key = c.trade_key
                    WHERE (
                        o.account_id = ?
                        OR CAST(json_extract(c.account_snapshot_json, '$.account_id') AS TEXT) = ?
                        OR CAST(json_extract(o.outcome_json, '$.account_id') AS TEXT) = ?
                        OR CAST(json_extract(o.outcome_json, '$.closed_position.account_id') AS TEXT) = ?
                    )
                    ORDER BY o.closed_at DESC
                    LIMIT ?
                    """,
                    (
                        acc_str,
                        acc_str,
                        acc_str,
                        acc_str,
                        int(limit),
                    ),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT 
                        o.closed_at,
                        o.symbol,
                        COALESCE(c.side, json_extract(o.outcome_json, '$.closed_position.side'), 'Flat') AS side,
                        o.close_type,
                        o.pnl,
                        COALESCE(
                            o.session,
                            c.session,
                            json_extract(o.outcome_json, '$.session'),
                            json_extract(o.outcome_json, '$.closed_position.session'),
                            json_extract(c.route_context_json, '$.session')
                        ) AS session,
                        COALESCE(c.opened_at, json_extract(o.outcome_json, '$.closed_position.opened_at')) AS opened_at
                    FROM trade_outcomes o
                    LEFT JOIN trade_contexts c ON o.trade_key = c.trade_key
                    ORDER BY o.closed_at DESC
                    LIMIT ?
                    """,
                    (int(limit),),
                ).fetchall()
        
        from src.utils.session_utils import get_trade_session
        res = []
        for row in rows:
            closed_at_str = row[0]
            time_part = "--"
            if closed_at_str:
                try:
                    dt_obj = datetime.fromisoformat(str(closed_at_str).replace("Z", "+00:00"))
                    time_part = dt_obj.strftime("%m/%d %H:%M")
                except Exception:
                    time_part = str(closed_at_str)[:16].replace("T", " ")
            
            trade_session = row[5]
            if not trade_session or trade_session == "UNKNOWN":
                ref_ts = row[6] or row[0]
                trade_session = get_trade_session(ref_ts) if ref_ts else "GLOBEX"

            res.append({
                "time": time_part,
                "timestamp": str(closed_at_str) if closed_at_str else "",
                "symbol": row[1],
                "side": row[2] or "Flat",
                "type": row[3],
                "pnl": round(float(row[4] or 0.0), 2),
                "session": trade_session,
            })
        return res

    def create_learning_batch(self, batch_id: str, provider: str, trade_keys: list[str], status: str = "created") -> None:
        if not (self.connected and self._conn is not None and self.backend == "sqlite"):
            return
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO learning_batches (batch_id, provider, status, item_count, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(batch_id) DO NOTHING
                """,
                (batch_id, provider, status, len(trade_keys), now),
            )
            for trade_key in trade_keys:
                self._conn.execute(
                    """
                    INSERT INTO learning_batch_items (batch_id, trade_key, status, created_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(batch_id, trade_key) DO NOTHING
                    """,
                    (batch_id, trade_key, status, now),
                )
            self._conn.commit()

    def complete_learning_batch(
        self,
        *,
        batch_id: str,
        status: str,
        raw_output_path: str,
        html_report_path: str,
        json_report_path: str,
        summary: dict[str, Any],
        analyzed_trade_keys: list[str],
    ) -> None:
        if not (self.connected and self._conn is not None and self.backend == "sqlite"):
            return
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """
                UPDATE learning_batches
                SET status = ?, raw_output_path = ?, html_report_path = ?, json_report_path = ?,
                    summary_json = ?, completed_at = ?
                WHERE batch_id = ?
                """,
                (
                    status,
                    raw_output_path,
                    html_report_path,
                    json_report_path,
                    json.dumps(summary, default=str),
                    now,
                    batch_id,
                ),
            )
            for trade_key in analyzed_trade_keys:
                self._conn.execute(
                    """
                    UPDATE learning_batch_items
                    SET status = 'analyzed', completed_at = ?
                    WHERE batch_id = ? AND trade_key = ?
                    """,
                    (now, batch_id, trade_key),
                )
            self._conn.commit()

    def persist_strategy_learning(
        self,
        *,
        learning_id: str,
        batch_id: str,
        trade_key: str | None,
        symbol: str | None,
        category: str,
        summary: str,
        evidence: dict[str, Any],
        proposed_config_diff: dict[str, Any],
        validation_status: str = "pending_optimizer_validation",
    ) -> None:
        if not (self.connected and self._conn is not None and self.backend == "sqlite"):
            return
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO strategy_learnings (
                    learning_id, batch_id, trade_key, symbol, category, summary,
                    evidence_json, proposed_config_diff_json, validation_status, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(learning_id) DO UPDATE SET
                    summary = excluded.summary,
                    evidence_json = excluded.evidence_json,
                    proposed_config_diff_json = excluded.proposed_config_diff_json,
                    validation_status = excluded.validation_status
                """,
                (
                    learning_id,
                    batch_id,
                    trade_key,
                    symbol,
                    category,
                    summary,
                    json.dumps(evidence, default=str),
                    json.dumps(proposed_config_diff, default=str),
                    validation_status,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._conn.commit()

    def persist_candidate_promotion(
        self,
        *,
        candidate_id: str,
        batch_id: str,
        config_path: str,
        config_diff: dict[str, Any],
        gate_status: dict[str, Any],
        status: str = "pending_validation",
        active_from_session: str | None = None,
    ) -> None:
        if not (self.connected and self._conn is not None and self.backend == "sqlite"):
            return
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO candidate_promotions (
                    candidate_id, batch_id, config_path, config_diff_json, gate_status_json,
                    status, active_from_session, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_id) DO UPDATE SET
                    config_path = excluded.config_path,
                    config_diff_json = excluded.config_diff_json,
                    gate_status_json = excluded.gate_status_json,
                    status = excluded.status,
                    active_from_session = excluded.active_from_session
                """,
                (
                    candidate_id,
                    batch_id,
                    config_path,
                    json.dumps(config_diff, default=str),
                    json.dumps(gate_status, default=str),
                    status,
                    active_from_session,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._conn.commit()

    def list_candidate_promotions(self, status: str | None = None, limit: int = 25) -> list[dict[str, Any]]:
        if not (self.connected and self._conn is not None and self.backend == "sqlite"):
            return []
        with self._lock:
            if status:
                rows = self._conn.execute(
                    "SELECT * FROM candidate_promotions WHERE status = ? ORDER BY created_at ASC LIMIT ?",
                    (status, int(limit)),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM candidate_promotions ORDER BY created_at ASC LIMIT ?",
                    (int(limit),),
                ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["config_diff"] = json.loads(item.pop("config_diff_json") or "{}")
            item["gate_status"] = json.loads(item.pop("gate_status_json") or "{}")
            out.append(item)
        return out

    def _decode_trade_context_row(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        for key in (
            "strategy_params_json",
            "selected_signal_json",
            "candidates_json",
            "blender_metadata_json",
            "route_context_json",
            "discovery_context_json",
            "account_snapshot_json",
            "chart_artifacts_json",
        ):
            out_key = key.removesuffix("_json")
            item[out_key] = json.loads(item.pop(key) or "{}")
        return item

    def _decode_closed_trade_row(self, row: sqlite3.Row) -> dict[str, Any]:
        item = self._decode_trade_context_row(row)
        outcome_json = item.pop("outcome_json", "{}")
        item["outcome"] = json.loads(outcome_json or "{}")
        item["outcome"]["pnl"] = item.pop("outcome_pnl")
        item["outcome"]["close_type"] = item.pop("outcome_close_type")
        item["outcome"]["closed_at"] = item.pop("outcome_closed_at")
        outcome_session = item.pop("outcome_session", None)
        if outcome_session:
            item["outcome"]["session"] = outcome_session
        elif not item["outcome"].get("session") and item.get("session"):
            item["outcome"]["session"] = item.get("session")
        return item

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(y * y for y in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def close(self) -> None:
        """Close SQLite database connection safely."""
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.commit()
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
            self.connected = False
            self.reason = "closed"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
