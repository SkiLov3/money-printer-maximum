#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import (
    get_adaptive_optimizer_config,
    get_combine_rule_keys,
    get_combine_rules,
    get_database_config,
    get_optimizer_config,
    get_strategy_config,
    load_stack_config,
)
from src.optimization.adaptive_runner import AdaptiveOptimizerRunner


def _symbol_checkpoint_path(configured: str | None, symbol: str) -> str:
    default_path = f"logs/telemetry/adaptive_{symbol.lower()}_checkpoint.json"
    if not configured:
        return default_path
    if "{symbol}" in configured:
        return configured.format(symbol=symbol.lower(), SYMBOL=symbol.upper())
    path = Path(configured)
    if path.name == "adaptive_checkpoint.json":
        return str(path.with_name(f"adaptive_{symbol.lower()}_checkpoint.json"))
    return str(path)


def _compact_trial(trial: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not isinstance(trial, dict):
        return None
    return {
        "trial": trial.get("trial"),
        "run_id": trial.get("run_id"),
        "score": trial.get("score"),
        "passed_combine": trial.get("passed_combine"),
        "summary": trial.get("summary") or {},
        "combine_evaluation": trial.get("combine_evaluation") or {},
        "source_trial_score": trial.get("source_trial_score"),
    }


def _compact_status(status: Dict[str, Any]) -> Dict[str, Any]:
    best_trial = _compact_trial(status.get("best_trial"))
    validation_trials = [
        compact for compact in (_compact_trial(item) for item in status.get("validation_trials", [])) if compact
    ]
    return {
        "symbol": status.get("symbol"),
        "epoch": status.get("epoch"),
        "trials_attempted": status.get("trials_attempted"),
        "trials_recorded": status.get("trials_recorded"),
        "best_score": status.get("best_score"),
        "best_trial": best_trial,
        "validation_trials": validation_trials,
        "full_live_validation": status.get("full_live_validation"),
        "promotion_quality": status.get("promotion_quality"),
        "consecutive_passes": status.get("consecutive_passes"),
        "consecutive_details": status.get("consecutive_details"),
        "permutation_significant": status.get("permutation_significant"),
        "permutation_details": status.get("permutation_details"),
        "final_full_validation": status.get("final_full_validation"),
        "final_promotion_quality": status.get("final_promotion_quality"),
        "proven_config_saved": status.get("proven_config_saved"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Adaptive optimization runner (guardrails + permutation + checkpoint).")
    parser.add_argument("--symbol", required=True, help="Contract symbol (e.g. MNQM6, MCLK6)")
    parser.add_argument("--file", required=True, help="CSV dataset path")
    parser.add_argument("--config", default=None, help="YAML config path (default: config/trading_stack.yaml)")
    parser.add_argument("--profile", default=None, help="Optional stack config profile (e.g. full_learning)")
    parser.add_argument("--combine", default="topstep_50k", help="Combine rules key from YAML")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs to run in this invocation")
    parser.add_argument("--trials", type=int, default=None, help="Candidates per epoch override")
    parser.add_argument("--max-search-bars", type=int, default=None, help="Override adaptive_optimizer.max_search_bars")
    parser.add_argument("--max-validation-bars", type=int, default=None, help="Override adaptive_optimizer.max_validation_bars")
    parser.add_argument("--validation-candidate-limit", type=int, default=None, help="Override adaptive_optimizer.validation_candidate_limit")
    parser.add_argument("--resume", action="store_true", help="Resume epoch index from checkpoint")
    parser.add_argument("--forever", action="store_true", help="Run continuously")
    parser.add_argument("--sleep", type=int, default=60, help="Sleep seconds between epochs when --forever")
    parser.add_argument("--list-combine", action="store_true", help="List available combine rule set keys in config")
    args = parser.parse_args()

    try:
        stack_config = load_stack_config(args.config, profile=args.profile)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1
    combine_keys = get_combine_rule_keys(stack_config)
    if args.list_combine:
        print("\nAvailable combine rule sets:")
        for key in combine_keys:
            print(f"  - {key}")
        return 0

    if args.combine not in combine_keys:
        print(f"ERROR: Unknown combine ruleset '{args.combine}'.")
        if combine_keys:
            print("Available keys:")
            for key in combine_keys:
                print(f"  - {key}")
        return 1

    strategy_config = get_strategy_config(stack_config)
    optimizer_config = get_optimizer_config(stack_config)
    adaptive_config = get_adaptive_optimizer_config(stack_config)
    database_config = get_database_config(stack_config)
    combine_rules = get_combine_rules(stack_config, args.combine)

    merged_optimizer_config = dict(optimizer_config)
    merged_optimizer_config.update(adaptive_config)
    if args.max_search_bars is not None:
        merged_optimizer_config["max_search_bars"] = args.max_search_bars
    if args.max_validation_bars is not None:
        merged_optimizer_config["max_validation_bars"] = args.max_validation_bars
    if args.validation_candidate_limit is not None:
        merged_optimizer_config["validation_candidate_limit"] = args.validation_candidate_limit
    trials_per_epoch = int(args.trials if args.trials is not None else adaptive_config.get("trials_per_epoch", 50))
    checkpoint_path = _symbol_checkpoint_path(adaptive_config.get("checkpoint_path"), args.symbol)
    experience_path = adaptive_config.get("experience_path", "logs/telemetry/experience_store.jsonl")
    reports_dir = adaptive_config.get("reports_dir", "reports/optimization")

    runner = AdaptiveOptimizerRunner(
        args.symbol,
        args.file,
        strategy_template=strategy_config,
        optimizer_config=merged_optimizer_config,
        combine_rules=combine_rules,
        experience_path=experience_path,
        checkpoint_path=checkpoint_path,
        reports_dir=reports_dir,
        initial_balance=float(combine_rules.get("starting_balance", 50000)),
        database_config=database_config,
    )

    if args.forever:
        start_epoch = runner.resume_epoch() if args.resume else 1
        runner.run_forever(start_epoch=start_epoch, trials_per_epoch=trials_per_epoch, sleep_seconds=args.sleep)
        return 0

    epoch = runner.resume_epoch() if args.resume else 1
    final = None
    for _ in range(max(1, args.epochs)):
        final = runner.run_once(epoch=epoch, trials_per_epoch=trials_per_epoch)
        epoch += 1

    print(json.dumps(_compact_status(final["status"] if final else {}), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
