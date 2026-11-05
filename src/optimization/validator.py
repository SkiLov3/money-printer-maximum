from __future__ import annotations

import random
from typing import Any

import pandas as pd


class StrategyValidator:
    def __init__(self, symbol: str, initial_balance: float = 50000):
        self.symbol = symbol
        self.initial_balance = initial_balance

    def split_dataframe(self, df: pd.DataFrame, validation_fraction: float = 0.3, warmup_bars: int = 250):
        split_index = max(warmup_bars, int(len(df) * (1 - validation_fraction)))
        train_df = df.iloc[:split_index].reset_index(drop=True)
        validation_start = max(0, split_index - warmup_bars)
        validation_df = df.iloc[validation_start:].reset_index(drop=True)
        return train_df, validation_df

    def _run(self, df: pd.DataFrame, strategy_config: dict[str, Any]):
        from src.utils.backtester import Backtester
        backtester = Backtester(self.symbol, initial_balance=int(self.initial_balance), strategy_config=strategy_config, telemetry_enabled=False)
        return backtester.run_dataframe(df, dataset_label="validation")

    def bootstrap_positive_rate(self, trades: list[dict[str, Any]], iterations: int = 250) -> float:
        if not trades:
            return 0.0
        pnls = [float(trade["pnl"]) for trade in trades]
        positive = 0
        rng = random.Random(42)
        for _ in range(iterations):
            sample = [rng.choice(pnls) for _ in pnls]
            if sum(sample) > 0:
                positive += 1
        return positive / iterations

    def validate(self, df: pd.DataFrame, strategy_config: dict[str, Any], validation_fraction: float = 0.3) -> dict[str, Any]:
        train_df, validation_df = self.split_dataframe(df, validation_fraction=validation_fraction)
        train_results = self._run(train_df, strategy_config)
        validation_results = self._run(validation_df, strategy_config)

        mc_positive_rate = self.bootstrap_positive_rate(validation_results["trades"])
        train_summary = train_results["summary"]
        val_summary = validation_results["summary"]
        score = (
            float(val_summary["total_pnl"])
            - (abs(float(val_summary["max_drawdown"])) * 0.5)
            + (float(val_summary["win_rate"]) * 2.0)
            + (mc_positive_rate * 100.0)
        )
        if val_summary["total_trades"] < 3:
            score -= 100

        passed = (
            val_summary["total_pnl"] > 0
            and train_summary["total_pnl"] >= 0
            and mc_positive_rate >= 0.55
            and val_summary["total_trades"] >= 3
        )

        return {
            "score": round(score, 2),
            "passed": passed,
            "mc_positive_rate": round(mc_positive_rate, 3),
            "train": train_summary,
            "validation": val_summary,
        }
