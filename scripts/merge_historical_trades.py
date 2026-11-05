#!/usr/bin/env python3
"""
merge_historical_trades.py

Safely merges historical trade records from data/swarm.db into logs/swarm.db
with zero loss of trade history, integrity checks, and dry-run validation.
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.session_utils import get_trade_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("merge_historical_trades")


def verify_db_integrity(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA integrity_check")
    status = cur.fetchone()[0]
    conn.close()
    if status != "ok":
        raise RuntimeError(f"Database integrity check failed for {db_path}: {status}")
    logger.info("Integrity check passed for %s: %s", db_path, status)


def merge_databases(target_path: str, source_path: str, dry_run: bool = False) -> dict[str, int]:
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Target DB not found: {target_path}")
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source DB not found: {source_path}")

    logger.info("Starting verification before merge...")
    verify_db_integrity(target_path)
    verify_db_integrity(source_path)

    conn_target = sqlite3.connect(target_path)
    conn_target.execute("PRAGMA busy_timeout=5000")
    conn_target.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    conn_source = sqlite3.connect(source_path)
    conn_source.execute("PRAGMA busy_timeout=5000")

    cur_target = conn_target.cursor()
    cur_source = conn_source.cursor()

    # Pre-merge counts
    cur_target.execute("SELECT count(*) FROM trade_outcomes")
    pre_outcomes = cur_target.fetchone()[0]
    cur_target.execute("SELECT count(*) FROM trade_contexts")
    pre_contexts = cur_target.fetchone()[0]
    logger.info("Pre-merge target counts: trade_outcomes=%d, trade_contexts=%d", pre_outcomes, pre_contexts)

    stats = {
        "outcomes_before": pre_outcomes,
        "contexts_before": pre_contexts,
        "outcomes_inserted": 0,
        "contexts_inserted": 0,
        "embeddings_inserted": 0,
        "health_events_inserted": 0,
        "discovery_runs_inserted": 0,
    }

    try:
        # 1. Merge trade_contexts
        cur_target.execute("SELECT trade_key FROM trade_contexts")
        existing_context_keys = set(r[0] for r in cur_target.fetchall())

        cur_source.execute("SELECT * FROM trade_contexts")
        cols_ctx = [d[0] for d in cur_source.description]
        source_contexts = cur_source.fetchall()

        for row in source_contexts:
            row_dict = dict(zip(cols_ctx, row))
            tk = row_dict["trade_key"]
            if tk not in existing_context_keys:
                sess = row_dict.get("session") or get_trade_session(row_dict.get("opened_at") or "")
                cur_target.execute(
                    """
                    INSERT INTO trade_contexts (
                        trade_key, symbol, strategy_name, side, quantity, order_id,
                        entry_price, opened_at, session, strategy_params_json,
                        selected_signal_json, candidates_json, blender_metadata_json,
                        route_context_json, discovery_context_json, account_snapshot_json,
                        chart_artifacts_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row_dict["trade_key"],
                        row_dict["symbol"],
                        row_dict["strategy_name"],
                        row_dict["side"],
                        row_dict["quantity"],
                        row_dict.get("order_id"),
                        row_dict.get("entry_price"),
                        row_dict["opened_at"],
                        sess,
                        row_dict["strategy_params_json"],
                        row_dict["selected_signal_json"],
                        row_dict["candidates_json"],
                        row_dict["blender_metadata_json"],
                        row_dict["route_context_json"],
                        row_dict["discovery_context_json"],
                        row_dict["account_snapshot_json"],
                        row_dict["chart_artifacts_json"],
                        row_dict["created_at"],
                    ),
                )
                existing_context_keys.add(tk)
                stats["contexts_inserted"] += 1

        # 2. Merge trade_outcomes
        cur_target.execute("SELECT trade_key FROM trade_outcomes")
        existing_outcome_keys = set(r[0] for r in cur_target.fetchall())

        cur_source.execute("SELECT * FROM trade_outcomes")
        cols_out = [d[0] for d in cur_source.description]
        source_outcomes = cur_source.fetchall()

        for row in source_outcomes:
            row_dict = dict(zip(cols_out, row))
            tk = row_dict["trade_key"]
            if tk not in existing_outcome_keys:
                sess = row_dict.get("session") or get_trade_session(row_dict.get("closed_at") or "")
                cur_target.execute(
                    """
                    INSERT INTO trade_outcomes (
                        trade_key, symbol, order_id, pnl, close_type,
                        closed_at, outcome_json, created_at, account_id, session
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row_dict["trade_key"],
                        row_dict["symbol"],
                        row_dict.get("order_id"),
                        row_dict["pnl"],
                        row_dict["close_type"],
                        row_dict["closed_at"],
                        row_dict["outcome_json"],
                        row_dict["created_at"],
                        row_dict.get("account_id"),
                        sess,
                    ),
                )
                existing_outcome_keys.add(tk)
                stats["outcomes_inserted"] += 1

        # 3. Merge embeddings
        cur_source.execute("SELECT namespace, entity_id, symbol, vector_json, metadata_json, created_at FROM embeddings")
        emb_rows = cur_source.fetchall()
        cur_target.executemany(
            """
            INSERT OR IGNORE INTO embeddings (namespace, entity_id, symbol, vector_json, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            emb_rows,
        )
        stats["embeddings_inserted"] = cur_target.rowcount

        # 4. Merge discovery_runs (idempotent by symbol + created_at)
        cur_target.execute("SELECT symbol, created_at FROM discovery_runs")
        existing_disc = set((r[0], r[1]) for r in cur_target.fetchall())

        cur_source.execute("SELECT symbol, summary_json, created_at FROM discovery_runs")
        disc_rows = cur_source.fetchall()
        inserted_disc = 0
        for r in disc_rows:
            if (r[0], r[2]) not in existing_disc:
                cur_target.execute(
                    "INSERT INTO discovery_runs (symbol, summary_json, created_at) VALUES (?, ?, ?)",
                    (r[0], r[1], r[2]),
                )
                existing_disc.add((r[0], r[2]))
                inserted_disc += 1
        stats["discovery_runs_inserted"] = inserted_disc

        # 5. Merge health_events (idempotent by symbol + created_at)
        cur_target.execute("SELECT symbol, created_at FROM health_events")
        existing_health = set((r[0], r[1]) for r in cur_target.fetchall())

        cur_source.execute("SELECT symbol, accepted, reason, created_at FROM health_events")
        health_rows = cur_source.fetchall()
        inserted_health = 0
        for r in health_rows:
            if (r[0], r[3]) not in existing_health:
                cur_target.execute(
                    "INSERT INTO health_events (symbol, accepted, reason, created_at) VALUES (?, ?, ?, ?)",
                    (r[0], r[1], r[2], r[3]),
                )
                existing_health.add((r[0], r[3]))
                inserted_health += 1
        stats["health_events_inserted"] = inserted_health

        # Verification
        cur_target.execute("SELECT count(*) FROM trade_outcomes")
        post_outcomes = cur_target.fetchone()[0]
        cur_target.execute("SELECT count(*) FROM trade_contexts")
        post_contexts = cur_target.fetchone()[0]

        stats["outcomes_after"] = post_outcomes
        stats["contexts_after"] = post_contexts

        logger.info(
            "Merge stats: trade_outcomes: %d -> %d (+%d), trade_contexts: %d -> %d (+%d)",
            pre_outcomes,
            post_outcomes,
            stats["outcomes_inserted"],
            pre_contexts,
            post_contexts,
            stats["contexts_inserted"],
        )

        cur_target.execute("PRAGMA integrity_check")
        post_status = cur_target.fetchone()[0]
        if post_status != "ok":
            raise RuntimeError(f"Post-merge integrity check failed: {post_status}")
        logger.info("Post-merge integrity check: %s", post_status)

        if dry_run:
            logger.info("DRY RUN mode active: rolling back transaction.")
            conn_target.rollback()
        else:
            logger.info("Committing merged records to %s...", target_path)
            conn_target.commit()
            conn_target.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            logger.info("Commit and WAL truncation complete.")

    except Exception:
        conn_target.rollback()
        logger.exception("Merge failed; transaction rolled back.")
        raise
    finally:
        conn_target.close()
        conn_source.close()

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge historical trades into production database.")
    parser.add_argument("--target-db", default="logs/swarm.db", help="Target SQLite database (production)")
    parser.add_argument("--source-db", default="data/swarm.db", help="Source SQLite database (historical archive)")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry run without committing changes")
    args = parser.parse_args()

    logger.info("Target DB: %s", args.target_db)
    logger.info("Source DB: %s", args.source_db)
    logger.info("Dry run: %s", args.dry_run)

    stats = merge_databases(args.target_db, args.source_db, dry_run=args.dry_run)
    logger.info("Merge execution completed successfully: %s", stats)


if __name__ == "__main__":
    main()
