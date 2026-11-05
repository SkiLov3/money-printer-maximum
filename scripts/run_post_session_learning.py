#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import get_database_config, get_learning_review_config, load_stack_config
from src.persistence.database import DatabaseService


REQUIRED_GATES = {
    "multi_split_validation": False,
    "captured_live_evidence": False,
    "three_combine_passes": False,
    "permutation_significance": False,
    "exact_50k_profile": False,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one post-session learning review over closed unanalyzed live trades.")
    parser.add_argument("--config", default="config/trading_stack.yaml")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    stack = load_stack_config(args.config)
    learning_cfg = get_learning_review_config(stack)
    if not bool(learning_cfg.get("enabled", False)):
        print("learning_review disabled; no provider calls made.")
        return 0

    database = DatabaseService(get_database_config(stack))
    if not database.connected:
        print(f"database unavailable: {database.reason}", file=sys.stderr)
        return 2

    limit = int(args.limit or learning_cfg.get("batch_size", 25) or 25)
    trades = database.list_closed_unanalyzed_trades(limit=limit)
    if not trades:
        print("no closed unanalyzed trades found.")
        return 0

    provider = str(learning_cfg.get("provider", "local"))
    batch_id = f"lr_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    trade_keys = [str(trade["trade_key"]) for trade in trades]
    database.create_learning_batch(batch_id, provider, trade_keys, status="processing")

    reports_dir = Path(str(learning_cfg.get("reports_dir") or "reports/learning_review"))
    raw_dir = Path(str(learning_cfg.get("raw_output_dir") or reports_dir / "raw"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    prompt_payload = _build_prompt_payload(batch_id=batch_id, trades=trades, stack=stack)
    raw_output = _call_provider(provider=provider, learning_cfg=learning_cfg, prompt_payload=prompt_payload)
    raw_output_path = raw_dir / f"{batch_id}.txt"
    raw_output_path.write_text(raw_output, encoding="utf-8")

    parsed = _parse_provider_output(raw_output)
    learnings = parsed.get("learnings", [])
    proposed_diffs = parsed.get("proposed_config_diffs", [])
    for idx, learning in enumerate(learnings):
        trade_key = learning.get("trade_key")
        database.persist_strategy_learning(
            learning_id=str(learning.get("learning_id") or f"{batch_id}_learning_{idx}"),
            batch_id=batch_id,
            trade_key=trade_key,
            symbol=learning.get("symbol"),
            category=str(learning.get("category") or "hypothesis"),
            summary=str(learning.get("summary") or ""),
            evidence=dict(learning.get("evidence") or {}),
            proposed_config_diff=dict(learning.get("proposed_config_diff") or {}),
        )

    promotions = []
    for idx, diff in enumerate(proposed_diffs):
        candidate_id = str(diff.get("candidate_id") or f"{batch_id}_candidate_{idx}")
        config_diff = dict(diff.get("config_diff") or diff.get("diff") or {})
        gate_status = dict(REQUIRED_GATES)
        if trades:
            gate_status["captured_live_evidence"] = True
        database.persist_candidate_promotion(
            candidate_id=candidate_id,
            batch_id=batch_id,
            config_path="",
            config_diff=config_diff,
            gate_status=gate_status,
            status="pending_optimizer_validation",
            active_from_session=None,
        )
        promotions.append(
            {
                "candidate_id": candidate_id,
                "status": "pending_optimizer_validation",
                "config_diff": config_diff,
                "gate_status": gate_status,
            }
        )

    report = {
        "batch_id": batch_id,
        "provider": provider,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trades": trades,
        "raw_output_path": str(raw_output_path),
        "learnings": learnings,
        "proposed_config_diffs": proposed_diffs,
        "candidate_promotions": promotions,
        "promotion_policy": "No proposed config can deploy until optimizer gates pass; auto-switch is next-session only.",
    }
    json_report_path = reports_dir / f"{batch_id}.json"
    html_report_path = reports_dir / f"{batch_id}.html"
    json_report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    html_report_path.write_text(_render_html_report(report), encoding="utf-8")

    database.complete_learning_batch(
        batch_id=batch_id,
        status="completed",
        raw_output_path=str(raw_output_path),
        html_report_path=str(html_report_path),
        json_report_path=str(json_report_path),
        summary={
            "trade_count": len(trades),
            "learning_count": len(learnings),
            "candidate_count": len(promotions),
        },
        analyzed_trade_keys=trade_keys,
    )
    print(json.dumps({"batch_id": batch_id, "html_report": str(html_report_path), "json_report": str(json_report_path)}, indent=2))
    return 0


def _build_prompt_payload(*, batch_id: str, trades: List[Dict[str, Any]], stack: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "role": "post_session_learning_analyst",
        "batch_id": batch_id,
        "instructions": [
            "Review only completed trades.",
            "Do not recommend intraday overrides or live signal overrides.",
            "Return JSON with learnings and proposed_config_diffs.",
            "Proposed config diffs are hypotheses and require optimizer validation before deployment.",
        ],
        "promotion_gates": REQUIRED_GATES,
        "strategy_config": stack.get("strategies", {}),
        "routing_config": stack.get("routing", {}),
        "trades": trades,
    }


def _call_provider(*, provider: str, learning_cfg: Dict[str, Any], prompt_payload: Dict[str, Any]) -> str:
    provider_cfg = (learning_cfg.get("providers") or {}).get(provider, {})
    command = provider_cfg.get("command") or learning_cfg.get(f"{provider}_command") or []
    if not command:
        return json.dumps(_local_review(prompt_payload), indent=2)
    if isinstance(command, str):
        command = [command]
    timeout = int(learning_cfg.get("timeout_seconds", 300) or 300)
    completed = subprocess.run(
        list(command),
        input=json.dumps(prompt_payload, default=str),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    output = completed.stdout.strip()
    if completed.returncode != 0:
        raise RuntimeError(f"provider {provider} failed with code {completed.returncode}: {completed.stderr.strip()}")
    return output


def _local_review(prompt_payload: Dict[str, Any]) -> Dict[str, Any]:
    trades = prompt_payload.get("trades") or []
    learnings = []
    losses = [trade for trade in trades if float((trade.get("outcome") or {}).get("pnl") or 0.0) < 0]
    wins = [trade for trade in trades if float((trade.get("outcome") or {}).get("pnl") or 0.0) > 0]
    if losses:
        learnings.append(
            {
                "category": "loss_review",
                "summary": f"{len(losses)} losing trade(s) need optimizer validation before changing stops, targets, filters, or sizing.",
                "evidence": {"trade_keys": [trade.get("trade_key") for trade in losses]},
                "proposed_config_diff": {},
            }
        )
    if wins:
        learnings.append(
            {
                "category": "win_review",
                "summary": f"{len(wins)} winning trade(s) captured as live evidence for future validation.",
                "evidence": {"trade_keys": [trade.get("trade_key") for trade in wins]},
                "proposed_config_diff": {},
            }
        )
    if not learnings:
        learnings.append(
            {
                "category": "session_review",
                "summary": "Closed trades were captured, but there is not enough outcome contrast for a useful config hypothesis.",
                "evidence": {"trade_keys": [trade.get("trade_key") for trade in trades]},
                "proposed_config_diff": {},
            }
        )
    return {"learnings": learnings, "proposed_config_diffs": []}


def _parse_provider_output(raw_output: str) -> Dict[str, Any]:
    text = raw_output.strip()
    if not text:
        return {"learnings": [], "proposed_config_diffs": []}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return {
                "learnings": [{"category": "raw_review", "summary": text[:2000], "evidence": {}, "proposed_config_diff": {}}],
                "proposed_config_diffs": [],
            }
        parsed = json.loads(text[start : end + 1])
    if isinstance(parsed, list):
        parsed = {"learnings": parsed, "proposed_config_diffs": []}
    parsed.setdefault("learnings", [])
    parsed.setdefault("proposed_config_diffs", [])
    return parsed


def _render_html_report(report: Dict[str, Any]) -> str:
    trade_sections = []
    for trade in report.get("trades", []):
        artifacts = ((trade.get("chart_artifacts") or {}).get("timeframes") or {})
        images = []
        for timeframe, artifact in artifacts.items():
            if artifact.get("status") == "captured" and artifact.get("svg_path"):
                image_src = Path(str(artifact["svg_path"])).resolve().as_uri()
                images.append(
                    f'<figure><figcaption>{html.escape(str(timeframe))}</figcaption>'
                    f'<img src="{html.escape(image_src)}" alt="{html.escape(str(timeframe))} chart"></figure>'
                )
            else:
                images.append(f'<p>{html.escape(str(timeframe))}: {html.escape(str(artifact.get("status", "unknown")))}</p>')
        outcome = trade.get("outcome") or {}
        trade_sections.append(
            "<section>"
            f"<h2>{html.escape(str(trade.get('symbol')))} {html.escape(str(trade.get('side')))} "
            f"{html.escape(str(trade.get('quantity')))}x</h2>"
            f"<p>Strategy: {html.escape(str(trade.get('strategy_name')))} | "
            f"PnL: {html.escape(str(outcome.get('pnl')))} | Close: {html.escape(str(outcome.get('close_type')))}</p>"
            f"<p>Trade key: {html.escape(str(trade.get('trade_key')))}</p>"
            + "".join(images)
            + "</section>"
        )
    learnings = "".join(f"<li>{html.escape(str(item.get('summary', item)))}</li>" for item in report.get("learnings", []))
    promotions = "".join(
        f"<li>{html.escape(str(item.get('candidate_id')))}: {html.escape(str(item.get('status')))}</li>"
        for item in report.get("candidate_promotions", [])
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Post-Session Learning Review</title>"
        "<style>body{font-family:Arial,sans-serif;margin:32px;background:#f8fafc;color:#111827}"
        "section{border-top:1px solid #cbd5e1;padding:20px 0}figure{margin:16px 0}"
        "img{max-width:100%;border:1px solid #cbd5e1;background:#0b1220}</style></head><body>"
        f"<h1>Learning Batch {html.escape(str(report.get('batch_id')))}</h1>"
        f"<p>Provider: {html.escape(str(report.get('provider')))}</p>"
        f"<p>{html.escape(str(report.get('promotion_policy')))}</p>"
        "<h2>Learnings</h2><ul>"
        + learnings
        + "</ul><h2>Promotion Candidates</h2><ul>"
        + promotions
        + "</ul>"
        + "".join(trade_sections)
        + "</body></html>"
    )


if __name__ == "__main__":
    raise SystemExit(main())
