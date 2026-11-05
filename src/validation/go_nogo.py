from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.optimization.validator import StrategyValidator


@dataclass
class GoNoGoResult:
    verdict: str
    metrics: dict[str, Any]
    recommendations: list[str]


class GoNoGoValidator:
    def __init__(self, symbol: str, initial_balance: float = 50000.0, config: dict[str, Any] | None = None, database=None):
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.config = config or {}
        self.database = database
        self.validator = StrategyValidator(symbol, initial_balance=initial_balance)

    def validate(self, df: pd.DataFrame, strategy_config: dict[str, Any]) -> GoNoGoResult:
        summary = self.validator.validate(df, strategy_config)
        validation = summary.get("validation", {})
        win_rate = float(validation.get("win_rate", 0.0))
        profit_factor = float(validation.get("profit_factor", 0.0))
        trades = int(validation.get("total_trades", 0))
        mc_positive = float(summary.get("mc_positive_rate", 0.0))

        thresholds = {
            "win_rate": float(self.config.get("min_win_rate", 40.0)),
            "profit_factor": float(self.config.get("min_profit_factor", 1.1)),
            "trades": int(self.config.get("min_trades", 5)),
            "mc_positive": float(self.config.get("min_mc_positive_rate", 0.5)),
        }
        passed = (
            win_rate >= thresholds["win_rate"]
            and profit_factor >= thresholds["profit_factor"]
            and trades >= thresholds["trades"]
            and mc_positive >= thresholds["mc_positive"]
        )
        verdict = "GO" if passed else "NO-GO"
        recommendations = []
        if not passed:
            recommendations.append("Tune strategy configuration before live deployment.")
        if trades < thresholds["trades"]:
            recommendations.append("Collect more trades before trusting validation results.")
        result = GoNoGoResult(
            verdict=verdict,
            metrics={
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "total_trades": trades,
                "mc_positive_rate": mc_positive,
                "score": float(summary.get("score", 0.0)),
            },
            recommendations=recommendations,
        )
        if self.database and self.database.connected:
            self.database.persist_validation_result(self.symbol, result.verdict, result.metrics, result.recommendations)
        return result
