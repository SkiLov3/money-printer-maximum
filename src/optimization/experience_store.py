from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from src.optimization.context_embedder import cosine_similarity


class ExperienceStore:
    def __init__(self, path: str = "logs/telemetry/experience_store.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8"):
                pass

    def _read_all(self) -> list[dict[str, Any]]:
        rows = []
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def _append(self, record: dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    def persist_trial(
        self,
        *,
        instrument: str,
        market_embedding: list[float],
        params_embedding: list[float],
        outcome_embedding: list[float],
        full_params: dict[str, Any],
        outcome: dict[str, Any],
        passed_combine: bool,
        passed_permutation: bool = False,
        epoch: int = 0,
        side: str | None = None,
        strategy_name: str | None = None,
        source: str = "optimizer",
    ) -> str:
        run_id = f"{instrument}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        record = {
            "type": "trial",
            "run_id": run_id,
            "instrument": instrument,
            "epoch": epoch,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "market_embedding": market_embedding,
            "params_embedding": params_embedding,
            "outcome_embedding": outcome_embedding,
            "full_params": full_params,
            "outcome": outcome,
            "passed_combine": bool(passed_combine),
            "passed_permutation": bool(passed_permutation),
            "side": side,
            "strategy_name": strategy_name,
            "source": source,
        }
        self._append(record)
        return run_id

    def persist_live_outcome(
        self,
        *,
        instrument: str,
        market_embedding: list[float],
        side: str | None,
        strategy_name: str | None,
        pnl: float,
        order_type: str = "",
        params_embedding: list[float] | None = None,
        outcome_embedding: list[float] | None = None,
    ) -> str:
        """Record a closed live trade so discovery can learn from actual live outcomes.

        Winning trades become "successes" (passed_combine=True); losers become
        "failures". Side and strategy_name are filled so downstream
        find_similar_* can filter direction-aware.
        """
        return self.persist_trial(
            instrument=instrument,
            market_embedding=market_embedding,
            params_embedding=params_embedding or [],
            outcome_embedding=outcome_embedding or [float(pnl)],
            full_params={},
            outcome={"pnl": float(pnl), "order_type": order_type},
            passed_combine=bool(pnl > 0),
            passed_permutation=False,
            epoch=0,
            side=side,
            strategy_name=strategy_name,
            source="live",
        )

    def save_proven_config(
        self,
        *,
        instrument: str,
        run_id: str,
        params: dict[str, Any],
        p_value: float,
        combine_passes: int,
    ) -> None:
        record = {
            "type": "proven",
            "instrument": instrument,
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "params": params,
            "p_value": float(p_value),
            "combine_passes": int(combine_passes),
        }
        self._append(record)

    def revoke_proven_configs(self, *, instruments: list[str], reason: str) -> None:
        for instrument in instruments:
            record = {
                "type": "proven_revoked",
                "instrument": instrument,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "reason": reason,
            }
            self._append(record)

    def get_proven_config(self, instrument: str) -> dict[str, Any] | None:
        rows = self._read_all()
        latest_revoke = max(
            (
                row.get("timestamp_utc", "")
                for row in rows
                if row.get("type") == "proven_revoked" and row.get("instrument") == instrument
            ),
            default="",
        )
        items = [
            row for row in rows
            if row.get("type") == "proven"
            and row.get("instrument") == instrument
            and row.get("timestamp_utc", "") > latest_revoke
        ]
        if not items:
            return None
        items.sort(key=lambda row: row.get("timestamp_utc", ""))
        return items[-1]

    def count_trials(self, instrument: str | None = None) -> int:
        rows = self._read_all()
        rows = [row for row in rows if row.get("type") == "trial"]
        if instrument:
            rows = [row for row in rows if row.get("instrument") == instrument]
        return len(rows)

    @staticmethod
    def _attribute_matches(stored: Any, query: str | None) -> bool:
        """Match specific live attributes while preserving legacy aggregate trials.

        Optimizer and pre-side-filter records often have side/strategy as None.
        Those records are aggregate regime evidence, so they should still match
        specific live intents. Concrete live attributes must match exactly.
        """
        if query is None or stored is None:
            return True
        return str(stored) == str(query)

    def _find_similar(
        self,
        *,
        market_embedding: list[float],
        instrument: str,
        passed: bool,
        limit: int,
        min_similarity: float = 0.0,
        side: str | None = None,
        strategy_name: str | None = None,
    ) -> list[dict[str, Any]]:
        trials = [
            row for row in self._read_all()
            if row.get("type") == "trial"
            and row.get("instrument") == instrument
            and bool(row.get("passed_combine")) == passed
            and self._attribute_matches(row.get("side"), side)
            and self._attribute_matches(row.get("strategy_name"), strategy_name)
        ]
        scored = []
        for row in trials:
            sim = cosine_similarity(market_embedding, row.get("market_embedding", []))
            if sim >= min_similarity:
                row = dict(row)
                row["similarity"] = sim
                scored.append(row)
        scored.sort(key=lambda row: row.get("similarity", 0.0), reverse=True)
        return scored[:limit]

    def find_similar_successes(
        self,
        market_embedding: list[float],
        instrument: str,
        limit: int = 5,
        side: str | None = None,
        strategy_name: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._find_similar(
            market_embedding=market_embedding,
            instrument=instrument,
            passed=True,
            limit=limit,
            min_similarity=0.0,
            side=side,
            strategy_name=strategy_name,
        )

    def find_similar_failures(
        self,
        market_embedding: list[float],
        instrument: str,
        limit: int = 10,
        min_similarity: float = 0.7,
        side: str | None = None,
        strategy_name: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._find_similar(
            market_embedding=market_embedding,
            instrument=instrument,
            passed=False,
            limit=limit,
            min_similarity=min_similarity,
            side=side,
            strategy_name=strategy_name,
        )
