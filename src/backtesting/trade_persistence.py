from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.optimization.context_embedder import cosine_similarity, embed_params

logger = logging.getLogger(__name__)


class TradePersistence:
    def __init__(self, path: str = "reports/optimization_loop/optimizer_memory.jsonl") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.write_text("", encoding="utf-8")

    def persist_run(
        self,
        *,
        run_id: str,
        symbol: str,
        trades: list[dict[str, Any]],
        config_snapshot: dict[str, Any],
        metrics: dict[str, Any],
        report_path: str = "",
    ) -> None:
        record = {
            "run_id": run_id,
            "symbol": symbol,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "trade_count": len(trades),
            "metrics": metrics,
            "config_snapshot": config_snapshot,
            "config_embedding": embed_params(config_snapshot.get("strategies", config_snapshot)),
            "report_path": report_path,
        }
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    def get_similar_past_configs(self, config_embedding: list[float], *, symbol: str, top_k: int = 5) -> list[dict[str, Any]]:
        rows = []
        for item in self._read_all():
            if item.get("symbol") != symbol:
                continue
            similarity = cosine_similarity(config_embedding, item.get("config_embedding", []))
            rows.append(
                {
                    "run_id": item.get("run_id"),
                    "similarity": similarity,
                    "trade_count": item.get("trade_count", 0),
                    "win_rate": item.get("metrics", {}).get("win_rate", 0.0),
                    "avg_r": item.get("metrics", {}).get("avg_pnl_per_trade", 0.0),
                    "config_snapshot": item.get("config_snapshot", {}),
                }
            )
        rows.sort(key=lambda entry: entry["similarity"], reverse=True)
        return rows[:top_k]

    def get_run_history(self, *, symbol: str | None = None, last_n: int = 10) -> list[dict[str, Any]]:
        rows = self._read_all()
        if symbol:
            rows = [row for row in rows if row.get("symbol") == symbol]
        rows.sort(key=lambda row: row.get("timestamp_utc", ""), reverse=True)
        return rows[:last_n]

    def _read_all(self) -> list[dict[str, Any]]:
        result = []
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed optimizer memory line")
        return result
