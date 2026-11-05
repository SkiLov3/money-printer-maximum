from __future__ import annotations

import copy
import itertools
import json
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.optimization.validator import StrategyValidator
from src.telemetry.store import TelemetryStore


class StrategyOptimizer:
    def __init__(self, symbol: str, telemetry: TelemetryStore | None = None, optimizer_config: dict[str, Any] | None = None):
        self.symbol = symbol
        self.telemetry = telemetry or TelemetryStore()
        self.validator = StrategyValidator(symbol)
        self.optimizer_config = optimizer_config or {}

    def _direction_candidates(self) -> list[dict[str, list[str]]]:
        configured = self.optimizer_config.get("direction_candidates", {})
        upper = self.symbol.upper()
        if upper.startswith("MGC") and configured.get("mgc"):
            return configured["mgc"]
        if upper.startswith("MCL") and configured.get("mcl"):
            return configured["mcl"]
        if upper.startswith("MNQ") and configured.get("mnq"):
            return configured["mnq"]
        if configured.get("default"):
            return configured["default"]

        if upper.startswith("MCL"):
            return [
                {"trend_following": ["Buy", "Sell"], "asian_breakout": ["Buy"], "sss": ["Buy"]},
                {"trend_following": ["Buy", "Sell"], "asian_breakout": ["Buy", "Sell"], "sss": ["Buy"]},
            ]
        if upper.startswith("MNQ"):
            return [
                {"trend_following": ["Buy", "Sell"], "asian_breakout": ["Buy", "Sell"], "sss": ["Sell"]},
                {"trend_following": ["Buy", "Sell"], "asian_breakout": ["Buy", "Sell"], "sss": ["Buy", "Sell"]},
            ]
        return [
            {"trend_following": ["Buy", "Sell"], "asian_breakout": ["Buy", "Sell"], "sss": ["Buy", "Sell"]},
        ]

    def generate_candidates(self) -> list[dict[str, Any]]:
        configured_candidates = self.optimizer_config.get("strategy_candidates")
        if isinstance(configured_candidates, list) and configured_candidates:
            return copy.deepcopy(configured_candidates)

        candidates = []
        enabled_sets = self.optimizer_config.get("enabled_sets", [
            {"ichimoku": True, "fx_at_one_glance": True, "ema_pullback": True, "asian_breakout": True, "sss": True, "trend_following": False},
        ])
        adx_values = self.optimizer_config.get("adx_threshold_delta_values", [-5, 0, 5])
        asian_rr_values = self.optimizer_config.get("asian_rr_values", [1.25, 1.5, 2.0])
        sss_confluence_values = self.optimizer_config.get("sss_min_confluence_values", [2, 3])
        ichimoku_confluence_values = self.optimizer_config.get("ichimoku_min_confluence_values", [4, 5])
        fx_confluence_values = self.optimizer_config.get("fx_min_confluence_values", [6, 7])
        fx_signal_gap_values = self.optimizer_config.get("fx_signal_gap_bars_values", [12, 24, 36])
        fx_score_edge_values = self.optimizer_config.get("fx_min_score_edge_values", [2])
        fx_breakout_values = self.optimizer_config.get("fx_require_breakout_values", [True])

        for (
            enabled,
            adx_delta,
            asian_rr,
            sss_confluence,
            ichimoku_confluence,
            fx_confluence,
            fx_signal_gap,
            fx_score_edge,
            fx_require_breakout,
            directions,
        ) in itertools.product(
            enabled_sets,
            adx_values,
            asian_rr_values,
            sss_confluence_values,
            ichimoku_confluence_values,
            fx_confluence_values,
            fx_signal_gap_values,
            fx_score_edge_values,
            fx_breakout_values,
            self._direction_candidates(),
        ):
            strategies = [
                {
                    "name": "ichimoku",
                    "enabled": enabled.get("ichimoku", True),
                    "priority": 100,
                    "params": {"min_confluence": ichimoku_confluence},
                },
                {
                    "name": "fx_at_one_glance",
                    "enabled": enabled.get("fx_at_one_glance", True),
                    "priority": 95,
                    "params": {
                        "min_confluence": fx_confluence,
                        "min_signal_gap_bars": fx_signal_gap,
                        "min_score_edge": fx_score_edge,
                        "require_breakout": fx_require_breakout,
                    },
                },
                {
                    "name": "ema_pullback",
                    "enabled": enabled.get("ema_pullback", True),
                    "priority": 90,
                    "params": {},
                },
                {
                    "name": "trend_following",
                    "enabled": enabled.get("trend_following", False),
                    "priority": 70,
                    "params": {"adx_threshold_delta": adx_delta},
                },
                {
                    "name": "asian_breakout",
                    "enabled": enabled.get("asian_breakout", True),
                    "priority": 85,
                    "params": {"rr_ratio": asian_rr},
                },
                {
                    "name": "sss",
                    "enabled": enabled.get("sss", True),
                    "priority": 80,
                    "params": {"min_confluence": sss_confluence},
                },
            ]
            candidates.append(
                {
                    "direction_filters": directions,
                    "strategies": strategies,
                }
            )
        return candidates

    def optimize_dataframe(self, df: pd.DataFrame, output_dir: str = "reports/optimization") -> dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        best = None
        trials = []
        for idx, candidate in enumerate(self.generate_candidates(), start=1):
            validation = self.validator.validate(df, candidate)
            trial = {
                "trial": idx,
                "config": candidate,
                "validation": validation,
            }
            trials.append(trial)
            if best is None or validation["score"] > best["validation"]["score"]:
                best = trial

        result = {
            "symbol": self.symbol,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "best": best,
            "trials": trials,
        }

        out_path = os.path.join(output_dir, f"{self.symbol.lower()}_optimizer_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json")
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, default=str)

        if best is not None:
            self.telemetry.record_optimization(self.symbol, output_dir, best["config"], best["validation"])

        return result
