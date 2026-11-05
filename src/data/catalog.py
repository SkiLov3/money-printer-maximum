from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


class DatasetCatalog:
    def __init__(self, path: str = "data/dataset_catalog.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return loaded if isinstance(loaded, list) else []

    def _save(self, rows: List[Dict[str, Any]]) -> None:
        self.path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")

    def register_dataset(
        self,
        symbol: str,
        path: str,
        *,
        source: str = "unknown",
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        dataset_path = Path(path)
        row = {
            "symbol": str(symbol).upper(),
            "path": str(path),
            "source": source,
            "metadata": dict(metadata or {}),
            "registered_at": now,
        }
        if dataset_path.exists():
            stat = dataset_path.stat()
            row["size_bytes"] = stat.st_size
            row["modified_at"] = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()

        rows = self._load()
        rows = [
            existing
            for existing in rows
            if not (
                str(existing.get("symbol", "")).upper() == row["symbol"]
                and str(existing.get("path", "")) == row["path"]
            )
        ]
        rows.append(row)
        self._save(rows)
        return row

    def list_datasets(self, symbol: str | None = None) -> List[Dict[str, Any]]:
        rows = self._load()
        if symbol is None:
            return rows
        symbol_upper = str(symbol).upper()
        return [row for row in rows if str(row.get("symbol", "")).upper() == symbol_upper]

    def latest_for_symbol(self, symbol: str) -> Dict[str, Any] | None:
        rows = self.list_datasets(symbol)
        if not rows:
            return None
        rows.sort(key=lambda row: row.get("registered_at", ""))
        return rows[-1]
