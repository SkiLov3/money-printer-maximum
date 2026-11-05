from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any


class TelemetryStore:
    def __init__(self, root: str = "logs/telemetry", enabled: bool = True):
        self.root = root
        self.enabled = enabled
        self._lock = threading.RLock()
        if self.enabled:
            os.makedirs(self.root, exist_ok=True)

    def _write_jsonl(self, filename: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        record = dict(payload)
        record.setdefault("recorded_at", datetime.now(timezone.utc).isoformat())
        path = os.path.join(self.root, filename)
        with self._lock, open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    def record_signal(
        self,
        symbol: str,
        strategy_name: str,
        status: str,
        side: str | None,
        timestamp,
        price: float | None,
        reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._write_jsonl(
            "signals.jsonl",
            {
                "symbol": symbol,
                "strategy_name": strategy_name,
                "status": status,
                "side": side,
                "timestamp": str(timestamp),
                "price": price,
                "reason": reason,
                "metadata": metadata or {},
            },
        )

    def record_trade_open(self, symbol: str, strategy_name: str, side: str, qty: int, price: float, metadata=None) -> None:
        self._write_jsonl(
            "trades.jsonl",
            {
                "event": "open",
                "symbol": symbol,
                "strategy_name": strategy_name,
                "side": side,
                "qty": qty,
                "price": price,
                "metadata": metadata or {},
            },
        )

    def record_trade_close(
        self,
        symbol: str,
        strategy_name: str,
        side: str,
        qty: int,
        pnl: float,
        exit_reason: str,
        metadata=None,
    ) -> None:
        self._write_jsonl(
            "trades.jsonl",
            {
                "event": "close",
                "symbol": symbol,
                "strategy_name": strategy_name,
                "side": side,
                "qty": qty,
                "pnl": pnl,
                "exit_reason": exit_reason,
                "metadata": metadata or {},
            },
        )

    def record_backtest(self, symbol: str, dataset: str, config: dict[str, Any], results: dict[str, Any]) -> None:
        self._write_jsonl(
            "backtests.jsonl",
            {
                "symbol": symbol,
                "dataset": dataset,
                "config": config,
                "results": results,
            },
        )

    def record_optimization(self, symbol: str, dataset: str, best_config: dict[str, Any], validation: dict[str, Any]) -> None:
        self._write_jsonl(
            "optimizations.jsonl",
            {
                "symbol": symbol,
                "dataset": dataset,
                "best_config": best_config,
                "validation": validation,
            },
        )
