from __future__ import annotations

import copy
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.optimization.combine_rules import evaluate_combine_rules
from src.optimization.context_embedder import embed_market, embed_outcome, embed_params
from src.optimization.experience_store import ExperienceStore
from src.optimization.guardrails import (
    check_consecutive_passes,
    check_permutation_significance,
)
from src.optimization.llm_analysis_builder import AnalysisBuilder
from src.optimization.llm_analyst import LLMAnalyst
from src.optimization.optimizer import StrategyOptimizer
from src.persistence.database import DatabaseService

logger = logging.getLogger(__name__)


class AdaptiveOptimizerRunner:
    def __init__(
        self,
        symbol: str,
        data_file: str,
        *,
        strategy_template: dict[str, Any] | None = None,
        optimizer_config: dict[str, Any] | None = None,
        combine_rules: dict[str, Any] | None = None,
        experience_path: str = "logs/telemetry/experience_store.jsonl",
        checkpoint_path: str = "logs/telemetry/adaptive_checkpoint.json",
        reports_dir: str = "reports/optimization",
        initial_balance: float = 50000,
        database_config: dict[str, Any] | None = None,
    ):
        self.symbol = symbol
        self.data_file = data_file
        self.strategy_template = copy.deepcopy(strategy_template or {})
        self.optimizer_config = optimizer_config or {}
        self.combine_rules = combine_rules or {}
        self.experience = ExperienceStore(experience_path)
        self.checkpoint_path = checkpoint_path
        self.reports_dir = reports_dir
        self.initial_balance = float(initial_balance)
        self.database = DatabaseService(database_config or {"enabled": False})
        self.llm_config = copy.deepcopy(self.optimizer_config.get("llm", {}))
        self.screenshot_config = copy.deepcopy(self.optimizer_config.get("screenshots", {}))
        self.search_backtest_mode = str(self.optimizer_config.get("search_backtest_mode", "optimizer_search"))
        self.validation_backtest_mode = str(self.optimizer_config.get("validation_backtest_mode", "full_live"))
        self.progress_every_trials = int(self.optimizer_config.get("progress_every_trials", 10) or 0)
        self.max_search_bars = int(self.optimizer_config.get("max_search_bars", 20000) or 0)
        self.max_validation_bars = int(self.optimizer_config.get("max_validation_bars", 5000) or 0)
        self.validation_candidate_limit = int(self.optimizer_config.get("validation_candidate_limit", 1) or 1)
        self.promotion_quality_config = copy.deepcopy(self.optimizer_config.get("promotion_quality", {}))
        self.analysis_builder = AnalysisBuilder(self.llm_config.get("reports_dir", "reports/llm_analysis"))
        self.llm_analyst = LLMAnalyst(database=self.database, config=self.llm_config)
        self._analysis_seed_candidate: dict[str, Any] | None = None
        self._prepared_cache: dict[str, pd.DataFrame] = {}

        try:
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
            os.makedirs(self.reports_dir, exist_ok=True)
        except OSError:
            pass

        self.optimizer = StrategyOptimizer(symbol, optimizer_config=self.optimizer_config)

    def run_once(self, *, epoch: int, trials_per_epoch: int = 50) -> dict[str, Any]:
        df = pd.read_csv(self.data_file)
        df.columns = [c.lower() for c in df.columns]
        if self.max_search_bars > 0 and len(df) > self.max_search_bars:
            search_df = df.tail(self.max_search_bars).reset_index(drop=True)
        else:
            search_df = df
        market_embedding = embed_market(df)

        candidates = self._ordered_candidates(market_embedding)
        candidates = self._inject_analysis_candidate(candidates)
        if trials_per_epoch > 0:
            candidates = self._epoch_candidate_batch(candidates, epoch, trials_per_epoch)

        best_trial = None
        best_score = float("-inf")
        trials = []

        for idx, candidate in enumerate(candidates, start=1):
            result = self._run_backtest(search_df, candidate)
            if result is None:
                continue

            summary = result.get("summary", {})
            combine_eval = result.get("combine_evaluation", {})
            passed_combine = bool(combine_eval.get("passed", False))
            score = self._score_trial(summary, passed_combine)
            passed_permutation = False

            run_id = self.experience.persist_trial(
                instrument=self.symbol,
                market_embedding=market_embedding,
                params_embedding=embed_params(candidate),
                outcome_embedding=embed_outcome(summary),
                full_params=candidate,
                outcome=result,
                passed_combine=passed_combine,
                passed_permutation=passed_permutation,
                epoch=epoch,
                # Optimizer trials aggregate trades across both directions, so side
                # stays None. Discovery's None-wildcard rule preserves current
                # regime-only matching for these legacy-style records.
                side=None,
                strategy_name=None,
                source="optimizer",
            )

            trial = {
                "trial": idx,
                "run_id": run_id,
                "score": score,
                "passed_combine": passed_combine,
                "config": candidate,
                "summary": summary,
                "combine_evaluation": combine_eval,
                "result": result,
            }
            trials.append(trial)

            if score > best_score:
                best_score = score
                best_trial = trial
            if self.progress_every_trials > 0 and (idx % self.progress_every_trials == 0 or idx == len(candidates)):
                self._write_progress(epoch, idx, len(candidates), best_trial)

        search_best_trial = best_trial
        validation_df = self._tail_frame(df, self.max_validation_bars)
        validation_trials = self._validate_top_trials(validation_df, trials)
        passing_validation_trials = [trial for trial in validation_trials if trial.get("passed_combine")]
        if passing_validation_trials:
            best_trial = max(passing_validation_trials, key=lambda item: float(item.get("score", float("-inf"))))
        elif validation_trials:
            best_trial = max(validation_trials, key=lambda item: float(item.get("score", float("-inf"))))

        status = {
            "symbol": self.symbol,
            "epoch": epoch,
            "trials_attempted": len(candidates),
            "trials_recorded": len(trials),
            "best_score": best_trial.get("score") if best_trial else None,
            "best_trial": best_trial,
            "search_best_trial": search_best_trial,
            "validation_trials": validation_trials,
            "consecutive_passes": False,
            "permutation_significant": False,
            "proven_config_saved": False,
        }

        if best_trial and best_trial.get("passed_combine"):
            best_config = best_trial["config"]
            validation_result = best_trial.get("result") or self._run_validation_backtest(df, best_config)
            status["full_live_validation"] = {
                "passed_combine": bool((validation_result or {}).get("combine_evaluation", {}).get("passed", False)),
                "summary": (validation_result or {}).get("summary", {}),
                "combine_evaluation": (validation_result or {}).get("combine_evaluation", {}),
            }
            promotion_quality = self._evaluate_promotion_quality(validation_result or {})
            status["promotion_quality"] = promotion_quality
            if not status["full_live_validation"]["passed_combine"]:
                out = {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "status": status,
                    "trials": trials,
                }
                out["llm_analysis"] = self._run_llm_analysis(epoch=epoch, optimizer_result=out, market_bars=df)
                self._write_epoch_report(epoch, out)
                self._write_checkpoint(epoch, status)
                return out
            if not promotion_quality.get("passed", True):
                out = {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "status": status,
                    "trials": trials,
                }
                out["llm_analysis"] = self._run_llm_analysis(epoch=epoch, optimizer_result=out, market_bars=df)
                self._write_epoch_report(epoch, out)
                self._write_checkpoint(epoch, status)
                return out

            consecutive_cfg = self.optimizer_config.get("consecutive_pass", {})
            permutation_cfg = self.optimizer_config.get("permutation", {})
            consecutive = check_consecutive_passes(
                data=validation_df,
                config=best_config,
                backtest_fn=self._run_validation_backtest,
                n_required=int(consecutive_cfg.get("n_required", 3)),
                offset_days=consecutive_cfg.get("offset_days", [0, 3, 7]),
                bars_per_day=int(consecutive_cfg.get("bars_per_day", 60 * 23)),
                min_bars=int(consecutive_cfg.get("min_bars", 5000)),
            )
            status["consecutive_passes"] = bool(consecutive.get("all_passed", False))
            status["consecutive_details"] = consecutive

            if status["consecutive_passes"]:
                real_return = float((validation_result or {}).get("summary", {}).get("total_pnl", 0.0))
                permutation = check_permutation_significance(
                    real_return=real_return,
                    data=validation_df,
                    config=best_config,
                    backtest_fn=self._run_validation_backtest,
                    n_permutations=int(permutation_cfg.get("n_permutations", 20)),
                    p_threshold=float(permutation_cfg.get("p_threshold", 0.05)),
                )
                status["permutation_significant"] = bool(permutation.get("significant", False))
                status["permutation_details"] = permutation

                if status["permutation_significant"]:
                    final_full = self._run_validation_backtest(df, best_config)
                    status["final_full_validation"] = {
                        "passed_combine": bool((final_full or {}).get("combine_evaluation", {}).get("passed", False)),
                        "summary": (final_full or {}).get("summary", {}),
                        "combine_evaluation": (final_full or {}).get("combine_evaluation", {}),
                    }
                    final_quality = self._evaluate_promotion_quality(final_full or {})
                    status["final_promotion_quality"] = final_quality
                    if status["final_full_validation"]["passed_combine"] and final_quality.get("passed", True):
                        self.experience.save_proven_config(
                            instrument=self.symbol,
                            run_id=best_trial["run_id"],
                            params=best_config,
                            p_value=float(permutation["p_value"]),
                            combine_passes=int(consecutive_cfg.get("n_required", 3)),
                        )
                        status["proven_config_saved"] = True

        out = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "trials": trials,
        }
        out["llm_analysis"] = self._run_llm_analysis(epoch=epoch, optimizer_result=out, market_bars=df)
        self._write_epoch_report(epoch, out)
        self._write_checkpoint(epoch, status)
        return out

    def run_forever(self, *, start_epoch: int = 1, trials_per_epoch: int = 50, sleep_seconds: int = 60) -> None:
        import time

        epoch = start_epoch
        while True:
            self.run_once(epoch=epoch, trials_per_epoch=trials_per_epoch)
            epoch += 1
            time.sleep(max(1, int(sleep_seconds)))

    def resume_epoch(self) -> int:
        if not os.path.exists(self.checkpoint_path):
            return 1
        try:
            with open(self.checkpoint_path, encoding="utf-8") as handle:
                checkpoint = json.load(handle)
        except Exception:
            return 1
        if checkpoint.get("symbol") != self.symbol:
            return 1
        return int(checkpoint.get("last_epoch", 0)) + 1

    def _ordered_candidates(self, market_embedding: list[float]) -> list[dict[str, Any]]:
        generated = self.optimizer.generate_candidates()
        successes = self.experience.find_similar_successes(market_embedding, self.symbol, limit=5)
        promotions = self._pending_promotion_candidates()

        def _normalize(cfg: dict[str, Any]) -> str:
            return json.dumps(cfg, sort_keys=True)

        ordered = []
        seen = set()
        for cfg in promotions:
            key = _normalize(cfg)
            if key not in seen:
                ordered.append(cfg)
                seen.add(key)
        for row in successes:
            cfg = row.get("full_params")
            if isinstance(cfg, dict):
                key = _normalize(cfg)
                if key not in seen:
                    ordered.append(cfg)
                    seen.add(key)
        for cfg in generated:
            key = _normalize(cfg)
            if key not in seen:
                ordered.append(cfg)
                seen.add(key)
        return ordered

    def _inject_analysis_candidate(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        suggested = self._analysis_seed_candidate
        if not isinstance(suggested, dict):
            return candidates
        normalized = json.dumps(suggested, sort_keys=True)
        deduped = [cfg for cfg in candidates if json.dumps(cfg, sort_keys=True) != normalized]
        return [suggested] + deduped

    def _epoch_candidate_batch(self, candidates: list[dict[str, Any]], epoch: int, limit: int) -> list[dict[str, Any]]:
        if limit <= 0 or len(candidates) <= limit:
            return candidates
        start = ((max(1, int(epoch)) - 1) * int(limit)) % len(candidates)
        rotated = candidates[start:] + candidates[:start]
        return rotated[:limit]

    def _tail_frame(self, df: pd.DataFrame, limit: int) -> pd.DataFrame:
        if limit > 0 and len(df) > limit:
            return df.tail(limit).reset_index(drop=True)
        return df

    def _score_trial(self, summary: dict[str, Any], passed_combine: bool) -> float:
        pnl = float(summary.get("total_pnl", 0.0))
        drawdown = abs(float(summary.get("max_drawdown", 0.0)))
        win_rate = float(summary.get("win_rate", 0.0))
        trades = float(summary.get("total_trades", 0))
        combine_bonus = 10_000.0 if passed_combine else 0.0
        return combine_bonus + pnl - (drawdown * 0.5) + (win_rate * 10.0) + min(trades, 200.0)

    def _compose_strategy_config(self, candidate: dict[str, Any]) -> dict[str, Any]:
        if self.strategy_template:
            merged = copy.deepcopy(self.strategy_template)
            merged["direction_filters"] = candidate.get(
                "direction_filters",
                merged.get("direction_filters", {}),
            )
            merged["strategies"] = candidate.get("strategies", merged.get("strategies", []))
            return merged
        return candidate

    def _run_backtest(self, df: pd.DataFrame, candidate: dict[str, Any]) -> dict[str, Any] | None:
        return self._run_backtest_with_mode(df, candidate, routing_mode=self.search_backtest_mode)

    def _run_validation_backtest(self, df: pd.DataFrame, candidate: dict[str, Any]) -> dict[str, Any] | None:
        return self._run_backtest_with_mode(df, candidate, routing_mode=self.validation_backtest_mode)

    def _run_backtest_with_mode(self, df: pd.DataFrame, candidate: dict[str, Any], *, routing_mode: str) -> dict[str, Any] | None:
        try:
            from src.utils.backtester import Backtester
            config = self._compose_strategy_config(candidate)
            prepared_indicators = self._prepared_indicators(df, routing_mode, config)
            bt = Backtester(
                self.symbol,
                initial_balance=int(self.initial_balance),
                strategy_config=config,
                telemetry_enabled=False,
                screenshot_config=self.screenshot_config,
                run_id=f"{self.symbol.lower()}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}",
                routing_mode=routing_mode,
            )
            results = bt.run_dataframe(df, dataset_label=self.data_file, prepared_indicators=prepared_indicators)
            results["combine_evaluation"] = evaluate_combine_rules(results, self.combine_rules)
            results["backtest_mode"] = routing_mode
            return results
        except Exception:
            return None

    def _prepared_indicators(self, df: pd.DataFrame, routing_mode: str, config: dict[str, Any]) -> pd.DataFrame:
        first_ts = ""
        last_ts = ""
        if "timestamp" in df.columns and not df.empty:
            first_ts = str(df["timestamp"].iloc[0])
            last_ts = str(df["timestamp"].iloc[-1])
        key = f"{self.symbol}:{routing_mode}:{len(df)}:{first_ts}:{last_ts}:{id(df)}"
        cached = self._prepared_cache.get(key)
        if cached is not None:
            return cached
        from src.utils.backtester import Backtester

        bt = Backtester(
            self.symbol,
            initial_balance=int(self.initial_balance),
            strategy_config=config,
            telemetry_enabled=False,
            screenshot_config={"enabled": False},
            routing_mode=routing_mode,
        )
        prepared = bt.strategy._calculate_indicators(bt._normalize_dataframe(df))
        self._prepared_cache[key] = prepared
        return prepared

    def _pending_promotion_candidates(self) -> list[dict[str, Any]]:
        if not self.database.connected or not hasattr(self.database, "list_candidate_promotions"):
            return []
        rows = self.database.list_candidate_promotions(status="pending_optimizer_validation", limit=20)
        candidates = []
        for row in rows:
            diff = row.get("config_diff") or {}
            if isinstance(diff, dict) and diff:
                candidates.append(self._candidate_from_config_changes(diff))
        return candidates

    def _validate_top_trials(self, df: pd.DataFrame, trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.validation_candidate_limit <= 0:
            return []
        top_trials = sorted(trials, key=lambda item: float(item.get("score", float("-inf"))), reverse=True)[: self.validation_candidate_limit]
        validated = []
        for trial in top_trials:
            result = self._run_validation_backtest(df, trial["config"])
            if result is None:
                continue
            summary = result.get("summary", {})
            combine_eval = result.get("combine_evaluation", {})
            passed_combine = bool(combine_eval.get("passed", False))
            validated.append(
                {
                    "trial": trial.get("trial"),
                    "run_id": trial.get("run_id"),
                    "score": self._score_trial(summary, passed_combine),
                    "passed_combine": passed_combine,
                    "config": trial["config"],
                    "summary": summary,
                    "combine_evaluation": combine_eval,
                    "result": result,
                    "source_trial_score": trial.get("score"),
                }
            )
        return validated

    def _evaluate_promotion_quality(self, result: dict[str, Any]) -> dict[str, Any]:
        cfg = self.promotion_quality_config or {}
        if not bool(cfg.get("enabled", False)):
            return {"enabled": False, "passed": True, "checks": {}}

        summary = result.get("summary", {}) or {}
        trades = result.get("trades", []) or []
        daily_pnl = ((result.get("combine_evaluation") or {}).get("daily_pnl") or {})
        trading_days = max(1, len(daily_pnl))
        total_trades = int(summary.get("total_trades", len(trades)) or 0)
        trades_per_day = total_trades / trading_days
        profit_factor_raw = summary.get("profit_factor", 0.0)
        profit_factor = 999.0 if profit_factor_raw == "Inf" else float(profit_factor_raw or 0.0)
        avg_trade = float(summary.get("avg_trade", 0.0) or 0.0)

        by_strategy = result.get("by_strategy", {}) or {}
        fx_trades = int((by_strategy.get("fx_at_one_glance") or {}).get("trades", 0) or 0)
        fx_share_pct = (fx_trades / total_trades * 100.0) if total_trades > 0 else 0.0

        checks = {
            "max_trades_per_day": {
                "ok": trades_per_day <= float(cfg.get("max_trades_per_day", 999999)),
                "actual": round(trades_per_day, 2),
                "limit": float(cfg.get("max_trades_per_day", 999999)),
            },
            "max_total_trades": {
                "ok": total_trades <= int(cfg.get("max_total_trades", 999999)),
                "actual": total_trades,
                "limit": int(cfg.get("max_total_trades", 999999)),
            },
            "min_profit_factor": {
                "ok": profit_factor >= float(cfg.get("min_profit_factor", 0.0)),
                "actual": round(profit_factor, 2),
                "minimum": float(cfg.get("min_profit_factor", 0.0)),
            },
            "min_avg_trade": {
                "ok": avg_trade >= float(cfg.get("min_avg_trade", 0.0)),
                "actual": round(avg_trade, 2),
                "minimum": float(cfg.get("min_avg_trade", 0.0)),
            },
            "max_fx_at_one_glance_trade_share_pct": {
                "ok": fx_share_pct <= float(cfg.get("max_fx_at_one_glance_trade_share_pct", 100.0)),
                "actual": round(fx_share_pct, 2),
                "limit": float(cfg.get("max_fx_at_one_glance_trade_share_pct", 100.0)),
            },
        }
        return {
            "enabled": True,
            "passed": all(item["ok"] for item in checks.values()),
            "checks": checks,
        }

    def _write_progress(self, epoch: int, completed: int, total: int, best_trial: dict[str, Any] | None) -> None:
        summary = {
            "epoch": epoch,
            "completed": completed,
            "total": total,
            "best_score": (best_trial or {}).get("score"),
            "best_pnl": (((best_trial or {}).get("summary") or {}).get("total_pnl")),
            "best_passed_combine": (best_trial or {}).get("passed_combine"),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        print(json.dumps({"optimizer_progress": summary}, default=str), flush=True)
        path = os.path.join(self.reports_dir, f"{self.symbol.lower()}_adaptive_epoch{epoch}_progress.json")
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, default=str)
        except Exception as exc:
            logger.warning(f"Failed to write progress report {path}: {exc}")

    def _run_llm_analysis(self, *, epoch: int, optimizer_result: dict[str, Any], market_bars: pd.DataFrame | None = None) -> dict[str, Any]:
        if not self.llm_config.get("enabled", False):
            return {"enabled": False}
        prior_analysis = self.database.list_llm_analysis(self.symbol, limit=int(self.llm_config.get("history_limit", 5)))
        screenshot_paths = self._extract_screenshot_paths(optimizer_result)
        prompt, prompt_path = self.analysis_builder.build_prompt(
            instrument=self.symbol,
            epoch=epoch,
            optimizer_result=optimizer_result,
            screenshot_paths=screenshot_paths,
            prior_analysis=prior_analysis,
            market_bars=market_bars,
            strategy_config=self.strategy_template,
            routing_config=self.optimizer_config.get("routing", {}),
            evidence_config=self.optimizer_config.get("evidence", {}),
            max_prompt_chars=int(self.llm_config.get("max_prompt_chars", 12000) or 12000),
        )
        result = self.llm_analyst.analyze(self.symbol, epoch, prompt)
        if self.llm_config.get("auto_apply_code_patches", False):
            self._apply_code_patches(result.code_patches)
        if result.config_changes:
            self._analysis_seed_candidate = self._candidate_from_config_changes(result.config_changes)
            if self.database.connected:
                self.database.persist_candidate_promotion(
                    candidate_id=f"{self.symbol}_epoch_{epoch}_llm_candidate",
                    batch_id=f"optimizer_epoch_{epoch}",
                    config_path="",
                    config_diff=result.config_changes,
                    gate_status={
                        "captured_optimizer_evidence": True,
                        "three_combine_passes": False,
                        "permutation_significance": False,
                        "exact_50k_profile": False,
                    },
                    status="pending_optimizer_validation",
                    active_from_session=None,
                )
        return {
            "enabled": True,
            "prompt_path": str(prompt_path),
            "output_path": result.output_path,
            "reasoning": result.reasoning,
            "config_changes": result.config_changes,
            "code_patches": result.code_patches,
            "screenshots": screenshot_paths,
        }

    def _extract_screenshot_paths(self, optimizer_result: dict[str, Any]) -> list[str]:
        best_trial = (optimizer_result.get("status", {}) or {}).get("best_trial") or {}
        trades = ((best_trial.get("result") or {}).get("trades") or [])
        paths: list[str] = []
        seen = set()
        for trade in trades:
            path = trade.get("entry_screenshot")
            if isinstance(path, str) and path and path not in seen:
                paths.append(path)
                seen.add(path)
        return paths

    def _candidate_from_config_changes(self, config_changes: dict[str, Any]) -> dict[str, Any]:
        candidate = {
            "direction_filters": copy.deepcopy(self.strategy_template.get("direction_filters", {})),
            "strategies": copy.deepcopy(self.strategy_template.get("strategies", [])),
        }
        for dotted_path, value in config_changes.items():
            self._set_dotted_value(candidate, str(dotted_path), value)
        return candidate

    def _set_dotted_value(self, obj: Any, dotted_path: str, value: Any) -> None:
        parts = [part for part in dotted_path.split(".") if part]
        if not parts:
            return
        current = obj
        for idx, part in enumerate(parts[:-1]):
            if isinstance(current, list):
                index = int(part)
                while len(current) <= index:
                    current.append({})
                current = current[index]
                continue
            if part not in current or not isinstance(current[part], (dict, list)):
                next_part = parts[idx + 1]
                current[part] = [] if next_part.isdigit() else {}
            current = current[part]
        last = parts[-1]
        if isinstance(current, list):
            index = int(last)
            while len(current) <= index:
                current.append(None)
            current[index] = value
        else:
            current[last] = value

    def _apply_code_patches(self, patches: list[dict[str, Any]]) -> None:
        # Per AGENTS.md: offline/learning/optimizer features must never mutate live code paths directly.
        return

    def _write_epoch_report(self, epoch: int, payload: dict[str, Any]) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.reports_dir, f"{self.symbol.lower()}_adaptive_epoch{epoch}_{stamp}.json")
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, default=str)
        except Exception as exc:
            logger.warning(f"Failed to write epoch report {path}: {exc}")
        return path

    def _write_checkpoint(self, epoch: int, status: dict[str, Any]) -> None:
        payload = {
            "symbol": self.symbol,
            "last_epoch": epoch,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "status": self._compact_checkpoint_status(status),
        }
        try:
            with open(self.checkpoint_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, default=str)
        except Exception as exc:
            logger.warning(f"Failed to write checkpoint {self.checkpoint_path}: {exc}")

    def _compact_checkpoint_status(self, status: dict[str, Any]) -> dict[str, Any]:
        best = status.get("best_trial") or {}
        return {
            "symbol": status.get("symbol"),
            "epoch": status.get("epoch"),
            "trials_attempted": status.get("trials_attempted"),
            "trials_recorded": status.get("trials_recorded"),
            "best_score": status.get("best_score"),
            "best_trial": {
                "trial": best.get("trial"),
                "run_id": best.get("run_id"),
                "score": best.get("score"),
                "passed_combine": best.get("passed_combine"),
                "summary": best.get("summary") or {},
                "combine_evaluation": best.get("combine_evaluation") or {},
            } if best else None,
            "full_live_validation": status.get("full_live_validation"),
            "promotion_quality": status.get("promotion_quality"),
            "consecutive_passes": status.get("consecutive_passes"),
            "permutation_significant": status.get("permutation_significant"),
            "final_promotion_quality": status.get("final_promotion_quality"),
            "proven_config_saved": status.get("proven_config_saved"),
        }
