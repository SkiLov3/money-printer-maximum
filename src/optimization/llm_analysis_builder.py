from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from src.learning.evidence_builder import build_evidence_pack, prompt_with_evidence

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AnalysisBuilder:
    def __init__(self, reports_dir: str = "reports/llm_analysis") -> None:
        self.reports_dir = PROJECT_ROOT / reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def build_prompt(
        self,
        *,
        instrument: str,
        epoch: int,
        optimizer_result: dict[str, Any],
        screenshot_paths: Sequence[str] | None = None,
        prior_analysis: Sequence[dict[str, Any]] | None = None,
        market_bars: pd.DataFrame | None = None,
        strategy_config: dict[str, Any] | None = None,
        routing_config: dict[str, Any] | None = None,
        evidence_config: dict[str, Any] | None = None,
        max_prompt_chars: int = 12000,
    ) -> tuple[str, Path]:
        evidence_pack = build_evidence_pack(
            role="optimizer_llm_analysis",
            symbol=instrument,
            epoch=epoch,
            optimizer_result=optimizer_result,
            market_bars=market_bars,
            strategy_config=strategy_config,
            routing_config=routing_config,
            evidence_config=evidence_config,
        )
        strategy_code = self._load_strategy_source()
        prompt = prompt_with_evidence(
            heading=f"You are a quantitative futures strategy analyst reviewing {instrument} epoch {epoch}.",
            evidence_pack=evidence_pack,
            prior_analysis=prior_analysis or [],
            strategy_code=strategy_code,
            max_prompt_chars=max_prompt_chars,
            instructions=[
                "Review completed optimizer/backtest evidence only.",
                "Return strategy or parameter hypotheses, not live execution overrides.",
                "Proposed changes must be deterministic config diffs that can enter the next optimizer epoch.",
                "Do not suggest gates that block, resize, veto, or override live trades outside validated parameters.",
            ],
        )
        if screenshot_paths:
            prompt += "\n\nReferenced screenshots:\n" + "\n".join(f"- {path}" for path in screenshot_paths[:20])
        prompt_path = self.reports_dir / f"{instrument}_epoch_{epoch}_prompt.md"
        prompt_path.write_text(prompt, encoding="utf-8")
        return prompt, prompt_path

    def build_legacy_prompt(
        self,
        *,
        instrument: str,
        epoch: int,
        optimizer_result: dict[str, Any],
        screenshot_paths: Sequence[str] | None = None,
        prior_analysis: Sequence[dict[str, Any]] | None = None,
    ) -> tuple[str, Path]:
        summary = self._build_trial_summary(optimizer_result)
        patterns = self._build_strategy_patterns(optimizer_result)
        charts = self._build_chart_section(screenshot_paths or [])
        prior = self._build_prior_analysis(prior_analysis or [])
        strategy_code = self._load_strategy_source()
        prompt = self._assemble(instrument, epoch, summary, patterns, charts, prior, strategy_code)
        prompt_path = self.reports_dir / f"{instrument}_epoch_{epoch}_prompt.md"
        prompt_path.write_text(prompt, encoding="utf-8")
        return prompt, prompt_path

    def _build_trial_summary(self, optimizer_result: dict[str, Any]) -> str:
        status = optimizer_result.get("status", {})
        trials = optimizer_result.get("trials", [])
        lines = [
            f"Epoch: {status.get('epoch', '--')}",
            f"Trials attempted: {status.get('trials_attempted', 0)}",
            f"Trials recorded: {status.get('trials_recorded', 0)}",
            f"Best score: {status.get('best_score', '--')}",
            f"Consecutive passes: {status.get('consecutive_passes', False)}",
            f"Permutation significant: {status.get('permutation_significant', False)}",
            "",
            "Top trials:",
        ]
        for trial in sorted(trials, key=lambda item: float(item.get("score", float("-inf"))), reverse=True)[:5]:
            summary = trial.get("summary", {})
            lines.append(
                f"- trial={trial.get('trial')} score={trial.get('score')} pnl={summary.get('total_pnl')} "
                f"win_rate={summary.get('win_rate')} trades={summary.get('total_trades')} "
                f"pf={summary.get('profit_factor')} dd={summary.get('max_drawdown')}"
            )
        return "\n".join(lines)

    def _build_strategy_patterns(self, optimizer_result: dict[str, Any]) -> str:
        best = (optimizer_result.get("status", {}) or {}).get("best_trial") or {}
        by_strategy = (best.get("summary") or {}).get("by_strategy") or {}
        if not by_strategy:
            return "No strategy pattern data available."
        lines = ["Strategy | Trades | Win Rate | PnL", "-" * 48]
        for name, bucket in sorted(by_strategy.items(), key=lambda item: float(item[1].get("pnl", 0.0)), reverse=True)[:5]:
            lines.append(
                f"{name:18s} | {int(bucket.get('trades', 0)):6d} | {float(bucket.get('win_rate', 0.0)):7.2f}% | {float(bucket.get('pnl', 0.0)):+.2f}"
            )
        return "\n".join(lines)

    def _build_chart_section(self, screenshot_paths: Sequence[str]) -> str:
        if not screenshot_paths:
            return "## Charts\nNo chart screenshots available this epoch."
        lines = ["## Charts"]
        for path in screenshot_paths[:20]:
            lines.append(f"- {path}")
        return "\n".join(lines)

    def _build_prior_analysis(self, prior_analysis: Sequence[dict[str, Any]]) -> str:
        if not prior_analysis:
            return "No prior analysis history."
        lines = []
        for item in prior_analysis[:5]:
            reasoning = str(item.get("reasoning", "")).replace("\n", " ")[:300]
            lines.append(
                f"Epoch {item.get('epoch')}: {reasoning} | config_changes={len(item.get('config_changes', {}))} "
                f"| code_patches={len(item.get('code_patches', []))}"
            )
        return "\n".join(lines)

    def _load_strategy_source(self) -> str:
        files = ["src/strategies/ichimoku.py", "src/strategies/fx_at_one_glance.py", "src/strategies/sss.py"]
        parts: list[str] = []
        for rel in files:
            path = PROJECT_ROOT / rel
            if not path.exists():
                continue
            content = path.read_text(encoding="utf-8")
            parts.append(f"### {rel}\n```python\n" + "\n".join(content.splitlines()[:25]) + "\n```")
        return "\n\n".join(parts)

    def _assemble(
        self,
        instrument: str,
        epoch: int,
        summary: str,
        patterns: str,
        charts: str,
        prior: str,
        strategy_code: str,
    ) -> str:
        return f"""You are a quantitative futures strategy analyst. You are reviewing optimization results for {instrument} epoch {epoch}.

Analyze the results, screenshots, strategy breakdowns, prior analysis, and current code.
Keep the response compact and decisive:
- REASONING under 220 words
- at most 5 config changes
- at most 3 code patches

## Trial Summary
{summary}

## Strategy Pattern Analysis
{patterns}

{charts}

## Prior Analysis
{prior}

## Current Strategy Code
{strategy_code}

Respond with EXACTLY three sections:

### REASONING
Explain what is working, what is failing, and why.

### CONFIG_CHANGES
```json
{{"config_changes": {{"strategies.strategies.0.params.some_key": 1}}}}
```

### CODE_PATCHES
```json
{{"code_patches": [{{"file": "src/path.py", "description": "why", "search": "old", "replace": "new"}}]}}
```
"""
