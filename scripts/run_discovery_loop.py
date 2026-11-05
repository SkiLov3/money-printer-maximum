#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.discovery.service import DiscoveryService
from src.utils.backtester import Backtester


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("discovery_loop")

REPO_ROOT = Path(__file__).resolve().parents[1]


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


class DiscoveryLoop:
    def __init__(self, config: Dict[str, Any], *, symbol: str, data_file: str) -> None:
        self._config = config
        self._symbol = symbol
        self._data_file = Path(data_file)
        self._report_dir = Path(config.get("reporting", {}).get("reports_dir", "reports/discovery_loop"))
        self._report_dir.mkdir(parents=True, exist_ok=True)
        self._stack_config = yaml.safe_load((REPO_ROOT / "config" / "trading_stack.yaml").read_text(encoding="utf-8")) or {}
        self._discovery = DiscoveryService(self._stack_config.get("discovery", {}))

    def run(self, *, enable_external_agents: bool = False) -> Dict[str, Any]:
        frame = pd.read_csv(self._data_file)
        frame.columns = [column.lower() for column in frame.columns]
        frame.loc[:, "timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        windows = self._slice_windows(frame)
        reports: List[Dict[str, Any]] = []
        for index, window in enumerate(windows):
            backtester = Backtester(self._symbol, strategy_config=self._stack_config.get("strategies", {}), telemetry_enabled=False)
            result = backtester.run_dataframe(window, dataset_label=f"window_{index:03d}")
            summary = dict(result.get("summary", {}))
            trades = list(result.get("trades", []))
            decision = self._discovery.evaluate_trade(
                self._symbol,
                window.tail(150),
                {"symbol": self._symbol, "strategy_name": "window_probe", "quantity": 1, "side": "Buy"},
            )
            hypothesis = ""
            if enable_external_agents and self._config.get("discovery", {}).get("hypothesis_enabled", True):
                hypothesis = self._spawn_hypothesis_agent(summary, decision)
            report = {
                "window_id": f"w_{index:03d}",
                "start": window["timestamp"].min().isoformat(),
                "end": window["timestamp"].max().isoformat(),
                "metrics": summary,
                "trade_count": len(trades),
                "discovery_decision": decision,
                "hypothesis": hypothesis,
            }
            (self._report_dir / f"window_{index:03d}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            reports.append(report)
        summary = {
            "windows_processed": len(reports),
            "reports_dir": str(self._report_dir),
            "summary_report": {
                "average_total_pnl": sum(float(item["metrics"].get("total_pnl", 0.0)) for item in reports) / max(len(reports), 1),
                "average_win_rate": sum(float(item["metrics"].get("win_rate", 0.0)) for item in reports) / max(len(reports), 1),
            },
        }
        (self._report_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        return summary

    def _slice_windows(self, frame: pd.DataFrame) -> List[pd.DataFrame]:
        window_days = int(self._config.get("orchestrator", {}).get("window_size_trading_days", 22))
        max_windows = int(self._config.get("orchestrator", {}).get("max_windows", 12))
        dates = sorted(frame["timestamp"].dt.normalize().dropna().unique())
        windows: List[pd.DataFrame] = []
        start_index = 0
        while start_index < len(dates) and len(windows) < max_windows:
            end_index = min(start_index + window_days, len(dates))
            window_dates = dates[start_index:end_index]
            if len(window_dates) < 3:
                break
            start = window_dates[0]
            end = window_dates[-1] + pd.Timedelta(days=1)
            window = frame.loc[(frame["timestamp"] >= start) & (frame["timestamp"] < end)].reset_index(drop=True)
            if len(window) >= 100:
                windows.append(window)
            start_index = end_index
        return windows

    def _spawn_hypothesis_agent(self, summary: Dict[str, Any], decision: Dict[str, Any]) -> str:
        prompt = (
            f"Analyze this TopstepX futures discovery window.\n"
            f"Summary: {json.dumps(summary, default=str)}\n"
            f"Discovery decision: {json.dumps(decision, default=str)}\n"
            "Return 3 concise hypotheses for why this window behaved this way and one suggested config change."
        )
        agent_cfg = self._config.get("external_agents", {})
        prompt = prompt[: int(agent_cfg.get("max_prompt_chars", 8000))]
        max_calls = max(1, int(agent_cfg.get("max_provider_calls_per_run", 1)))
        calls = 0
        for key in ("claude", "codex", "gemini"):
            if calls >= max_calls:
                break
            cfg = self._config.get(key, {})
            if not cfg.get("enabled", False):
                continue
            command = list(cfg.get("command", []))
            if not command:
                continue
            command = resolve_command(command)
            calls += 1
            try:
                binary_name = Path(command[0]).stem.lower() if command else ""
                if key == "codex" and binary_name == "codex" and len(command) >= 2 and command[1] == "exec":
                    command.append(prompt[:8000])
                    prompt = ""
                completed = subprocess.run(
                    command,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=int(cfg.get("timeout_seconds", 600)),
                    cwd=str(REPO_ROOT),
                    encoding="utf-8",
                )
                output = completed.stdout or ""
                if output.strip():
                    return output.strip()
            except Exception as exc:
                logger.warning("Discovery %s runner failed: %s", key, exc)
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Rolling discovery loop for TopstepX Trading Swarm.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--config", default="config/discovery_loop.yaml")
    parser.add_argument("--enable-external-agents", action="store_true")
    parser.add_argument("--disable-external-agents", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    loop = DiscoveryLoop(config, symbol=args.symbol, data_file=args.data_file)
    enable_external = bool(args.enable_external_agents and not args.disable_external_agents)
    result = loop.run(enable_external_agents=enable_external)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
