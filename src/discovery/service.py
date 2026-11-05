from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.optimization.context_embedder import embed_market
from src.optimization.experience_store import ExperienceStore


class DiscoveryService:
    def __init__(self, config: dict[str, Any] | None = None, database=None):
        self.config = config or {}
        self.database = database
        self.enabled = bool(self.config.get("enabled", False))
        self.experience = ExperienceStore(self.config.get("experience_path", "logs/telemetry/experience_store.jsonl"))

    def health_check(self) -> dict[str, Any]:
        db_health = self.database.health_check() if self.database else {"connected": False, "reason": "no_database"}
        return {
            "enabled": self.enabled,
            "ready": bool(self.enabled),
            "database": db_health,
            "llm_enabled": bool(self.config.get("llm_enabled", False)),
            "vector_enabled": bool(self.config.get("vector_enabled", False)),
        }

    def evaluate_trade(self, symbol: str, latest_data: pd.DataFrame, trade_intent: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled or latest_data.empty:
            return {"allowed": True, "quantity_multiplier": 1.0, "enforced": False, "reason": "discovery_disabled"}

        intent_side = trade_intent.get("side")
        intent_strategy = trade_intent.get("strategy_name")
        failure_advisory_threshold = float(self.config.get("advisory_failure_similarity_threshold", 0.7))
        success_advisory_threshold = float(self.config.get("advisory_success_similarity_threshold", 0.75))

        market_embedding = embed_market(latest_data.tail(int(self.config.get("embedding_window", 120))))
        successes = self.experience.find_similar_successes(
            market_embedding,
            symbol,
            limit=5,
            side=intent_side,
            strategy_name=intent_strategy,
        )
        failures = self.experience.find_similar_failures(
            market_embedding,
            symbol,
            limit=5,
            min_similarity=failure_advisory_threshold,
            side=intent_side,
            strategy_name=intent_strategy,
        )
        db_matches = []
        if self.database and self.database.connected:
            db_matches = self.database.query_similar_embeddings("market_context", symbol, market_embedding, limit=5)
            self.database.persist_embedding(
                "market_context",
                f"{symbol}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
                symbol,
                market_embedding,
                {
                    "strategy_name": intent_strategy or "unknown",
                    "side": intent_side or "unknown",
                },
            )

        best_success = successes[0]["similarity"] if successes else 0.0
        best_failure = failures[0]["similarity"] if failures else 0.0
        db_similarity = db_matches[0]["similarity"] if db_matches else 0.0

        # Discovery is advisory-only in the live path. It records similar
        # success/failure context for review, but it must never block, veto,
        # resize, or override trades selected by configured parameters.
        reason = "neutral"
        if best_failure > best_success and best_failure >= failure_advisory_threshold:
            reason = f"discovery advisory: similar failure context ({best_failure:.2f})"
        elif best_success > best_failure and best_success >= success_advisory_threshold:
            reason = f"discovery advisory: similar success context ({best_success:.2f})"

        matched_failure: dict[str, Any] = {}
        if failures:
            top = failures[0]
            matched_failure = {
                "run_id": top.get("run_id"),
                "side": top.get("side"),
                "strategy_name": top.get("strategy_name"),
                "source": top.get("source"),
                "similarity": float(top.get("similarity", 0.0)),
            }

        summary = {
            "symbol": symbol,
            "strategy_name": intent_strategy or "unknown",
            "side": intent_side,
            "allowed": True,
            "quantity_multiplier": 1.0,
            "enforced": False,
            "reason": reason,
            "best_success_similarity": best_success,
            "best_failure_similarity": best_failure,
            "db_similarity": db_similarity,
            "matched_failure": matched_failure,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        if self.database and self.database.connected:
            self.database.persist_discovery_run(symbol, summary)
        return summary
