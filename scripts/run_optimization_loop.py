#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtesting.results_exporter import ResultsExporter
from src.backtesting.trade_persistence import TradePersistence
from src.learning.claude_md_writer import ClaudeMdWriter
from src.optimization.context_embedder import embed_params
from src.utils.backtester import Backtester


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("optimization_loop")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "optimization_loop.yaml"
STACK_CONFIG_PATH = REPO_ROOT / "config" / "trading_stack.yaml"
README_PATH = REPO_ROOT / "README.md"
DEBUG_PATH = REPO_ROOT / "debug.md"


def resolve_command(command: List[str]) -> List[str]:
    if not command:
        return []
    binary = command[0]
    resolved_path = shutil.which(binary)
    if resolved_path is not None:
        resolved = list(command)
        resolved[0] = resolved_path
        return resolved
    if sys.platform.startswith("win") and "." not in Path(binary).name:
        for suffix in (".exe", ".cmd", ".bat"):
            candidate = f"{binary}{suffix}"
            resolved_path = shutil.which(candidate)
            if resolved_path is not None:
                resolved = list(command)
                resolved[0] = resolved_path
                return resolved
    return command


class OptimizationLoop:
    def __init__(self, config: Dict[str, Any], *, symbol: str, data_file: str) -> None:
        self._config = config
        self._symbol = symbol
        self._data_file = Path(data_file)
        opt_cfg = config.get("optimization", {})
        persistence_cfg = config.get("persistence", {})
        self._max_iterations = int(opt_cfg.get("max_iterations", 10))
        self._target_score = float(opt_cfg.get("target_score", 500.0))
        self._optimization_days = int(opt_cfg.get("optimization_days", 60))
        self._plateau_threshold = float(opt_cfg.get("plateau_threshold", 0.02))
        self._plateau_iterations = int(opt_cfg.get("plateau_iterations", 3))
        self._max_changes = int(opt_cfg.get("max_changes_per_iteration", 3))
        self._reports = ResultsExporter(str(persistence_cfg.get("reports_dir", "reports/optimization_loop")))
        self._memory = TradePersistence(str(persistence_cfg.get("memory_path", "reports/optimization_loop/optimizer_memory.jsonl")))
        self._claude_writer = ClaudeMdWriter(str(REPO_ROOT / "CLAUDE.md"))

    def run(self) -> Dict[str, Any]:
        if not self._data_file.exists():
            raise FileNotFoundError(self._data_file)

        full_df = pd.read_csv(self._data_file)
        opt_df = self._slice_data(full_df, days=self._optimization_days)
        history: List[Dict[str, Any]] = []
        best_metrics: Dict[str, Any] = {}
        best_config: Dict[str, Any] = {}
        stop_reason = "max_iterations"

        for iteration in range(1, self._max_iterations + 1):
            run_id = f"opt_iter_{iteration:03d}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            config_snapshot = self._load_stack_config()
            result = self._run_backtest(opt_df, config_snapshot)
            metrics = dict(result.get("summary", {}))
            score = self._score_metrics(metrics)
            report_path = self._reports.export_run_report(
                result=result,
                config=config_snapshot,
                run_id=run_id,
                previous_runs=[self._reports.load_run_report(item["run_id"]) for item in self._reports.list_runs()[-3:]],
            )
            self._memory.persist_run(
                run_id=run_id,
                symbol=self._symbol,
                trades=list(result.get("trades", [])),
                config_snapshot=config_snapshot,
                metrics=metrics,
                report_path=str(report_path),
            )

            if not best_metrics or score > float(best_metrics.get("optimization_score", float("-inf"))):
                best_metrics = dict(metrics)
                best_metrics["optimization_score"] = score
                best_config = config_snapshot

            history.append({"iteration": iteration, "run_id": run_id, "metrics": metrics, "score": score})
            logger.info("Iteration %d score=%.2f pnl=%.2f trades=%s", iteration, score, float(metrics.get("total_pnl", 0.0)), metrics.get("total_trades", 0))

            if score >= self._target_score:
                stop_reason = "target_reached"
                break
            if self._check_plateau(history):
                stop_reason = "plateau"
                break
            if iteration == self._max_iterations:
                break

            previous_report = self._reports.load_run_report(history[-2]["run_id"]) if len(history) >= 2 else None
            prompt = self._build_optimizer_prompt(
                report=self._reports.load_run_report(run_id),
                previous_report=previous_report,
                similar_configs=self._memory.get_similar_past_configs(
                    embed_params(config_snapshot.get("strategies", config_snapshot)),
                    symbol=self._symbol,
                    top_k=5,
                ),
                learnings=self._claude_writer.get_learnings_summary(),
            )
            model_output = self._spawn_external_model(prompt)
            if not model_output.strip():
                stop_reason = "empty_model_output"
                break

            diff = self._detect_changes()
            if not diff.strip():
                self._claude_writer.append_learning(
                    run_id=run_id,
                    changes_made="(no changes)",
                    metrics_before={"pass_rate": score / max(self._target_score, 1.0), "sharpe": None, "win_rate": metrics.get("win_rate"), "max_dd": metrics.get("max_drawdown")},
                    metrics_after={"pass_rate": score / max(self._target_score, 1.0), "sharpe": None, "win_rate": metrics.get("win_rate"), "max_dd": metrics.get("max_drawdown")},
                    verdict="kept",
                )
                continue

            new_config_snapshot = self._load_stack_config()
            new_result = self._run_backtest(opt_df, new_config_snapshot)
            new_metrics = dict(new_result.get("summary", {}))
            new_score = self._score_metrics(new_metrics)

            if new_score >= score:
                verdict = "kept"
                history[-1]["metrics"] = new_metrics
                history[-1]["score"] = new_score
            else:
                verdict = "reverted"
                self._revert_changes()

            self._claude_writer.append_learning(
                run_id=run_id,
                changes_made=self._summarise_diff(diff),
                metrics_before={"pass_rate": score / max(self._target_score, 1.0), "sharpe": None, "win_rate": metrics.get("win_rate"), "max_dd": metrics.get("max_drawdown")},
                metrics_after={"pass_rate": new_score / max(self._target_score, 1.0), "sharpe": None, "win_rate": new_metrics.get("win_rate"), "max_dd": new_metrics.get("max_drawdown")},
                verdict=verdict,
            )

        return {
            "best_config": best_config,
            "best_metrics": best_metrics,
            "iteration_history": history,
            "stop_reason": stop_reason,
        }

    def _run_backtest(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        backtester = Backtester(self._symbol, strategy_config=config.get("strategies", {}), telemetry_enabled=False)
        return backtester.run_dataframe(df, dataset_label=str(self._data_file))

    def _load_stack_config(self) -> Dict[str, Any]:
        return yaml.safe_load(STACK_CONFIG_PATH.read_text(encoding="utf-8")) or {}

    def _build_optimizer_prompt(
        self,
        *,
        report: Dict[str, Any],
        previous_report: Optional[Dict[str, Any]],
        similar_configs: List[Dict[str, Any]],
        learnings: str,
    ) -> str:
        metrics = report.get("metrics", {})
        lines = [
            "You are a quantitative futures strategy optimizer for TopstepX Trading Swarm.",
            f"Current symbol: {self._symbol}",
            "",
            "Current run metrics:",
            f"- Total PnL: {metrics.get('total_pnl', 0.0)}",
            f"- Win Rate: {metrics.get('win_rate', 0.0)}",
            f"- Max Drawdown: {metrics.get('max_drawdown', 0.0)}",
            f"- Profit Factor: {metrics.get('profit_factor', 0.0)}",
            f"- Total Trades: {metrics.get('total_trades', 0)}",
            "",
            "Worst trades:",
        ]
        for trade in report.get("worst_trades", [])[:5]:
            lines.append(
                f"- {trade.get('strategy_name')} {trade.get('side')} entry={trade.get('entry')} exit={trade.get('exit')} pnl={trade.get('pnl')}"
            )
        if previous_report:
            lines.extend(["", "Previous report comparison:", json.dumps(previous_report.get("comparison", {}), indent=2, default=str)])
        if similar_configs:
            lines.append("")
            lines.append("Similar past configurations:")
            for config in similar_configs:
                lines.append(
                    f"- Run {config.get('run_id')} similarity={config.get('similarity', 0.0):.3f} "
                    f"win_rate={config.get('win_rate', 0.0):.3f} trades={config.get('trade_count', 0)}"
                )
        lines.extend(
            [
                "",
                "Accumulated strategy learnings:",
                learnings,
                "",
                "Editable files:",
            ]
        )
        for item in self._config.get("editable_files", []):
            lines.append(f"- {item}")
        lines.extend(
            [
                "",
                f"Make at most {self._max_changes} meaningful changes total.",
                "Edit files directly in the repository.",
                "Prefer fixing root-cause logic bugs over random parameter churn.",
                "After edits, explain what you changed and why.",
            ]
        )
        return "\n".join(lines)

    def _spawn_external_model(self, prompt: str) -> str:
        agent_cfg = self._config.get("external_agents", {})
        max_calls = max(1, int(agent_cfg.get("max_provider_calls_per_iteration", 1)))
        prompt = prompt[: int(agent_cfg.get("max_prompt_chars", 8000))]
        calls = 0
        for key in ("claude", "codex", "gemini"):
            if calls >= max_calls:
                break
            cfg = self._config.get(key, {})
            if not cfg.get("enabled", False):
                continue
            command = list(cfg.get("command", []))
            timeout = int(cfg.get("timeout_seconds", 600))
            if not command:
                continue
            command = resolve_command(command)
            calls += 1
            output = self._run_command(command, prompt, timeout=timeout, key=key)
            if output.strip():
                return output
        return ""

    def _run_command(self, command: List[str], prompt: str, *, timeout: int, key: str) -> str:
        try:
            binary_name = Path(command[0]).stem.lower() if command else ""
            if key == "codex" and binary_name == "codex" and len(command) >= 2 and command[1] == "exec":
                command = list(command) + [prompt[:8000]]
                prompt = ""
            completed = subprocess.run(
                command,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(REPO_ROOT),
                encoding="utf-8",
            )
            output = completed.stdout or ""
            if key == "claude" and self._is_rate_limited(output):
                retries = int(self._config.get("claude", {}).get("rate_limit_retries", 1))
                if retries > 0:
                    time.sleep(30)
                    self._config["claude"]["rate_limit_retries"] = retries - 1
                    return self._run_command(command, prompt, timeout=timeout, key=key)
            return output
        except Exception as exc:
            logger.warning("%s runner failed: %s", key, exc)
            return ""

    def _is_rate_limited(self, output: str) -> bool:
        return bool(output and any(token in output.lower() for token in ("rate limit", "you've hit your limit", "resets in")))

    def _detect_changes(self) -> str:
        files = list(self._config.get("editable_files", [])) + ["README.md", "debug.md"]
        completed = subprocess.run(["git", "diff", "--"] + files, capture_output=True, text=True, cwd=str(REPO_ROOT))
        return completed.stdout or ""

    def _revert_changes(self) -> None:
        files = list(self._config.get("editable_files", []))
        subprocess.run(["git", "checkout", "--"] + files, capture_output=True, text=True, cwd=str(REPO_ROOT))

    def _score_metrics(self, metrics: Dict[str, Any]) -> float:
        pnl = float(metrics.get("total_pnl", 0.0))
        drawdown = abs(float(metrics.get("max_drawdown", 0.0)))
        win_rate = float(metrics.get("win_rate", 0.0))
        trades = float(metrics.get("total_trades", 0.0))
        profit_factor = float(metrics.get("profit_factor", 0.0)) if metrics.get("profit_factor") != "Inf" else 5.0
        return pnl - (drawdown * 0.5) + (win_rate * 100.0) + min(trades, 200.0) + (profit_factor * 25.0)

    def _slice_data(self, df: pd.DataFrame, *, days: int) -> pd.DataFrame:
        frame = df.copy()
        frame.columns = [column.lower() for column in frame.columns]
        if "timestamp" not in frame.columns:
            return frame
        frame.loc[:, "timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        cutoff = frame["timestamp"].max() - timedelta(days=days)
        return frame.loc[frame["timestamp"] >= cutoff].reset_index(drop=True)

    def _check_plateau(self, history: List[Dict[str, Any]]) -> bool:
        n = self._plateau_iterations
        if len(history) < n:
            return False
        recent = history[-n:]
        for index in range(1, len(recent)):
            previous = float(recent[index - 1].get("score", 0.0))
            current = float(recent[index].get("score", 0.0))
            if previous == 0:
                return False
            improvement = (current - previous) / abs(previous)
            if improvement >= self._plateau_threshold:
                return False
        return True

    def _summarise_diff(self, diff: str) -> str:
        lines = [line for line in diff.splitlines() if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))]
        return "; ".join(lines[:6]) if lines else "(no changes)"


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Agentic optimization loop for TopstepX Trading Swarm.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    config = _load_yaml(Path(args.config))
    loop = OptimizationLoop(config, symbol=args.symbol, data_file=args.data_file)
    summary = loop.run()
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
