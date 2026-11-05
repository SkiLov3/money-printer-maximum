#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.strategy_agent import StrategyAgent
from src.config.settings import get_mega_vision_config, get_strategy_config, load_stack_config
from src.mega_vision.agent import MegaStrategyAgent, MegaVisionConfig


def _resolve_agent_pick(decision: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    if decision.get("no_pick"):
        return None
    strategy_name = decision.get("strategy_name")
    side = decision.get("side")
    for candidate in candidates:
        if candidate.get("strategy_name") == strategy_name and (not side or candidate.get("side") == side):
            return candidate
    return None


async def run_shadow(symbol: str, data_file: str, config_file: str, max_decisions: int = 5, *, disable_agent_calls: bool = True, timeout_seconds: int | None = None) -> Dict[str, Any]:
    stack_config = load_stack_config(config_file)
    strategy_config = get_strategy_config(stack_config)
    mega_cfg_dict = get_mega_vision_config(stack_config)
    if timeout_seconds is not None:
        mega_cfg_dict["fallback_on_timeout_seconds"] = int(timeout_seconds)
    mega_cfg_dict["external_calls_enabled"] = not disable_agent_calls
    if disable_agent_calls:
        mega_cfg_dict["claude"] = {"command": ["missing-claude-binary", "-p"]}
        mega_cfg_dict["codex"] = {"command": ["missing-codex-binary", "exec", "--full-auto"]}
        mega_cfg_dict["gemini"] = {"command": ["missing-gemini-binary", "-p"]}
    mega_cfg = MegaVisionConfig.from_dict(mega_cfg_dict)
    agent = StrategyAgent(symbol, strategy_config=strategy_config)
    mega = MegaStrategyAgent(mega_cfg)

    df = pd.read_csv(data_file)
    indicators = agent.engine.prepare_data(df)
    bars = pd.DataFrame(indicators[["timestamp", "open", "high", "low", "close", "volume", "price"]]).copy()

    rows: List[Dict[str, Any]] = []
    for index in range(len(indicators)):
        native_pick, candidates = agent.engine.evaluate_index(bars, indicators, index)
        if not candidates:
            continue
        row = indicators.iloc[index]
        ts = row.get("timestamp")
        if isinstance(ts, pd.Timestamp):
            event_ts = ts.to_pydatetime()
        elif isinstance(ts, datetime):
            event_ts = ts
        else:
            event_ts = datetime.now(timezone.utc)
        context = {
            "symbol": symbol,
            "market_open": True,
            "news_standdown": False,
            "near_close": False,
            "active_count": 0,
            "max_positions": 1,
            "current_price": float(row.get("price", row.get("close", 0.0)) or 0.0),
        }
        decision = await mega.decide(event_ts if event_ts.tzinfo else event_ts.replace(tzinfo=timezone.utc), candidates, context)
        agent_pick = _resolve_agent_pick(decision, candidates)
        agreement = (
            native_pick is not None
            and agent_pick is not None
            and native_pick.get("strategy_name") == agent_pick.get("strategy_name")
            and native_pick.get("side") == agent_pick.get("side")
        )
        rows.append(
            {
                "timestamp": event_ts.isoformat(),
                "native_pick": native_pick,
                "agent_decision": decision,
                "agent_pick": agent_pick,
                "agreement": agreement,
                "candidate_count": len(candidates),
            }
        )
        if len(rows) >= max_decisions:
            break

    agreement_count = sum(1 for row in rows if row["agreement"])
    no_picks = sum(1 for row in rows if row["agent_decision"].get("no_pick"))
    summary = {
        "symbol": symbol,
        "data_file": data_file,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_decisions": len(rows),
        "agreement_count": agreement_count,
        "agreement_rate": (agreement_count / len(rows)) if rows else 0.0,
        "no_pick_count": no_picks,
        "events": rows,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate an offline Mega Vision shadow-vs-native comparison report.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--config", default="config/trading_stack.yaml")
    parser.add_argument("--max-decisions", type=int, default=5)
    parser.add_argument("--output-dir", default="reports/mega_vision_shadow")
    parser.add_argument("--enable-agent-calls", action="store_true", help="Opt in to external model calls for advisory shadow analysis")
    parser.add_argument("--disable-agent-calls", action="store_true", help="Deprecated; external model calls are disabled by default")
    parser.add_argument("--timeout-seconds", type=int, default=None)
    args = parser.parse_args()

    summary = asyncio.run(
        run_shadow(
            args.symbol,
            args.data_file,
            args.config,
            max_decisions=args.max_decisions,
            disable_agent_calls=(not args.enable_agent_calls) or args.disable_agent_calls,
            timeout_seconds=args.timeout_seconds,
        )
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{args.symbol.lower()}_shadow_{stamp}.json"
    output_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"report": str(output_path), "summary": {k: v for k, v in summary.items() if k != "events"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
