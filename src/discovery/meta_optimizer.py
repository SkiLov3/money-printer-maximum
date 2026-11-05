from __future__ import annotations

from typing import Any

from src.config.settings import (
    get_adaptive_optimizer_config,
    get_combine_rules,
    get_database_config,
    get_strategy_config,
    load_stack_config,
)
from src.discovery.service import DiscoveryService
from src.optimization.adaptive_runner import AdaptiveOptimizerRunner
from src.persistence.database import DatabaseService
from src.validation.go_nogo import GoNoGoValidator


class MetaOptimizerService:
    def __init__(self, symbol: str, data_file: str, config_path: str | None = None, database: DatabaseService | None = None):
        self.symbol = symbol
        self.data_file = data_file
        self.stack_config = load_stack_config(config_path)
        self.database = database or DatabaseService({"enabled": False})
        self.discovery = DiscoveryService(self.stack_config.get("discovery", {}), database=self.database)
        self.validator = GoNoGoValidator(symbol, config=self.stack_config.get("validation", {}), database=self.database)
        self.runner = AdaptiveOptimizerRunner(
            symbol,
            data_file,
            strategy_template=get_strategy_config(self.stack_config),
            optimizer_config=get_adaptive_optimizer_config(self.stack_config),
            combine_rules=get_combine_rules(self.stack_config),
            experience_path=self.stack_config.get("adaptive_optimizer", {}).get("experience_path", "logs/telemetry/experience_store.jsonl"),
            checkpoint_path=self.stack_config.get("adaptive_optimizer", {}).get("checkpoint_path", "logs/telemetry/adaptive_checkpoint.json"),
            reports_dir=self.stack_config.get("adaptive_optimizer", {}).get("reports_dir", "reports/optimization"),
            database_config=get_database_config(self.stack_config),
        )

    def run_epoch(self, epoch: int | None = None, trials_per_epoch: int | None = None) -> dict[str, Any]:
        epoch = int(epoch or self.runner.resume_epoch())
        trials_per_epoch = int(trials_per_epoch or self.stack_config.get("adaptive_optimizer", {}).get("trials_per_epoch", 50))
        result = self.runner.run_once(epoch=epoch, trials_per_epoch=trials_per_epoch)
        best_trial = result.get("status", {}).get("best_trial")
        recommendation = self._build_recommendation(result)
        payload = {
            "symbol": self.symbol,
            "epoch": epoch,
            "best_score": result.get("status", {}).get("best_score"),
            "best_trial": best_trial,
            "recommendation": recommendation,
            "status": result.get("status", {}),
        }
        if self.database.connected:
            self.database.persist_meta_optimizer_run(self.symbol, epoch, float(result.get("status", {}).get("best_score") or 0.0), payload)
        return payload

    def _build_recommendation(self, optimizer_result: dict[str, Any]) -> dict[str, Any]:
        status = optimizer_result.get("status", {})
        best_trial = status.get("best_trial") or {}
        best_config = best_trial.get("config") or get_strategy_config(self.stack_config)
        import pandas as pd

        df = pd.read_csv(self.data_file)
        validation_result = self.validator.validate(df, best_config)
        recommendation = {
            "verdict": validation_result.verdict,
            "metrics": validation_result.metrics,
            "recommendations": validation_result.recommendations,
            "optimizer_status": {
                "consecutive_passes": bool(status.get("consecutive_passes", False)),
                "permutation_significant": bool(status.get("permutation_significant", False)),
                "proven_config_saved": bool(status.get("proven_config_saved", False)),
            },
        }
        return recommendation
