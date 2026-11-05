from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MAX_REPORT_BYTES = 50 * 1024
_EXTREME_TRADE_COUNT = 5
_TRADE_FIELDS: tuple[str, ...] = (
    "time_open",
    "time_close",
    "entry",
    "exit",
    "side",
    "qty",
    "pnl",
    "pnl_ticks",
    "strategy_name",
    "metadata",
)
_SUMMARY_METRICS: tuple[str, ...] = (
    "total_pnl",
    "win_rate",
    "total_trades",
    "max_drawdown",
    "profit_factor",
)


class ResultsExporter:
    def __init__(self, reports_dir: str = "reports/optimization_loop") -> None:
        self._reports_dir = Path(reports_dir)
        self._reports_dir.mkdir(parents=True, exist_ok=True)

    def export_run_report(
        self,
        *,
        result: dict[str, Any],
        config: dict[str, Any],
        run_id: str,
        previous_runs: list[dict[str, Any]] | None = None,
    ) -> Path:
        metrics = dict(result.get("summary", {}))
        trades = list(result.get("trades", []))
        best_trades, worst_trades = self._select_extreme_trades(trades)
        comparison = self._compute_comparison(metrics, self._get_latest_metrics(previous_runs))
        report = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "best_trades": best_trades,
            "worst_trades": worst_trades,
            "config_snapshot": config,
            "comparison": comparison,
            "trade_count": len(trades),
        }
        payload = json.dumps(report, indent=2, default=str)
        if len(payload.encode("utf-8")) > _MAX_REPORT_BYTES:
            report["config_snapshot"] = {"trimmed": True}
            report["best_trades"] = report["best_trades"][:3]
            report["worst_trades"] = report["worst_trades"][:3]
            payload = json.dumps(report, indent=2, default=str)
        path = self._reports_dir / f"{run_id}.json"
        path.write_text(payload, encoding="utf-8")
        return path

    def load_run_report(self, run_id: str) -> dict[str, Any]:
        return json.loads((self._reports_dir / f"{run_id}.json").read_text(encoding="utf-8"))

    def list_runs(self) -> list[dict[str, Any]]:
        result = []
        for item in sorted(self._reports_dir.glob("*.json")):
            report = json.loads(item.read_text(encoding="utf-8"))
            summary = {"run_id": report.get("run_id", item.stem), "timestamp": report.get("timestamp", "")}
            metrics = report.get("metrics", {})
            for key in _SUMMARY_METRICS:
                summary[key] = metrics.get(key)
            result.append(summary)
        result.sort(key=lambda row: row.get("timestamp") or "")
        return result

    def _select_extreme_trades(self, trades: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        def score(trade: dict[str, Any]) -> float:
            return float(trade.get("pnl", 0.0))

        trimmed = [{key: trade.get(key) for key in _TRADE_FIELDS if key in trade} for trade in trades]
        ordered = sorted(trimmed, key=score)
        worst = ordered[:_EXTREME_TRADE_COUNT]
        best = list(reversed(ordered[-_EXTREME_TRADE_COUNT:]))
        return best, worst

    def _get_latest_metrics(self, previous_runs: list[dict[str, Any]] | None) -> dict[str, Any] | None:
        if not previous_runs:
            return None
        return previous_runs[-1].get("metrics")

    def _compute_comparison(self, metrics: dict[str, Any], previous_metrics: dict[str, Any] | None) -> dict[str, Any]:
        if not previous_metrics:
            return {}
        comparison = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and isinstance(previous_metrics.get(key), (int, float)):
                comparison[key] = float(value) - float(previous_metrics[key])
        return comparison
