#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.no_lookahead_validate_candidate import (
    evaluate_next_bar_open,
    evaluate_resting_stop_orders,
    scenario_definitions,
)
from scripts.optimize_portfolio_mix import (
    Candidate,
    build_candidates,
    candidate_to_dict,
    flatten_evaluation,
    load_symbol_data,
    sorted_rows,
)
from src.charting import resample_timeframe
from src.config.settings import get_combine_rules, load_stack_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strict no-lookahead portfolio optimizer. Signals are generated from completed bars "
            "and scored only after next-bar-open fills."
        )
    )
    parser.add_argument("--config", default="config/trading_stack.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="reports/no_lookahead_optimizer")
    parser.add_argument("--tail-bars", type=int, default=0, help="Use the most recent N bars per symbol; 0 means full CSV.")
    parser.add_argument("--timeframe", default="1m", help="Validation bar timeframe, e.g. 1m or 2m.")
    parser.add_argument("--risk", type=float, default=0.12)
    parser.add_argument(
        "--execution-model",
        choices=("next_bar_open", "resting_stop"),
        default="next_bar_open",
        help="Strict fill model used for candidate evaluation.",
    )
    parser.add_argument("--max-pending-bars", type=int, default=5, help="Bars before an unfilled resting stop expires.")
    parser.add_argument("--max-candidates", type=int, default=0, help="Limit candidates after filters; 0 means all.")
    parser.add_argument("--candidate-id", action="append", default=[], help="Evaluate only this candidate id.")
    parser.add_argument(
        "--candidate-id-contains",
        action="append",
        default=[],
        help="Evaluate only candidates whose id contains this text. Repeating this option is OR-matched.",
    )
    parser.add_argument("--family", action="append", default=[], help="Evaluate only this candidate family.")
    parser.add_argument(
        "--search-scenario",
        default="next_bar_open_topstep_fee_slip1",
        help="Scenario used for ranking candidates. Must be one of the strict validator scenarios.",
    )
    parser.add_argument(
        "--validate-top-n",
        type=int,
        default=0,
        help="After search, run the top N candidates through the full core strict scenario set.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    run_dir = Path(args.output_dir) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)

    config = load_stack_config(args.config)
    config.setdefault("routing", {})["risk_per_trade_pct"] = float(args.risk)
    combine_rules = {
        "topstep_50k": get_combine_rules(config, "topstep_50k"),
        "topstep_50k_daily_enforced": get_combine_rules(config, "topstep_50k_daily_enforced"),
    }
    data = resample_loaded_data(load_symbol_data(Path(args.data_dir), int(args.tail_bars)), str(args.timeframe))
    candidates = filter_candidates(build_candidates(config), args)

    scenarios = {scenario["name"]: scenario for scenario in scenario_definitions("core")}
    if args.search_scenario not in scenarios:
        raise ValueError(f"Unknown --search-scenario {args.search_scenario!r}; choices: {', '.join(sorted(scenarios))}")
    search_scenario = scenarios[args.search_scenario]

    print(
        f"Strict next-bar-open search: {len(candidates)} candidates, scenario={search_scenario['name']} -> {run_dir}",
        flush=True,
    )
    write_json(run_dir / "candidates.json", [candidate_to_dict(candidate) for candidate in candidates])

    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        try:
            evaluation = evaluate_strict_candidate(candidate, config, data, combine_rules, search_scenario, args)
            evaluation["candidate_family"] = candidate.family
            row = flatten_evaluation(evaluation)
        except Exception as exc:  # pragma: no cover - keeps long searches checkpointed
            evaluation = {
                "candidate_id": candidate.candidate_id,
                "candidate_family": candidate.family,
                "routing_mode": "optimizer_fast_no_lookahead_next_bar_open",
                "strategy_config": candidate.strategy_config,
                "scenario": search_scenario,
                "error": f"{type(exc).__name__}: {exc}",
            }
            row = {
                "candidate_id": candidate.candidate_id,
                "candidate_family": candidate.family,
                "routing_mode": "optimizer_fast_no_lookahead_next_bar_open",
                "scenario": search_scenario["name"],
                "error": evaluation["error"],
                "total_pnl": 0.0,
                "total_trades": 0,
                "score": -1_000_000.0,
            }
        row["scenario"] = search_scenario["name"]
        rows.append(row)
        details.append(evaluation)
        ranked = sorted_rows(rows)
        write_csv(run_dir / "candidate_scores.csv", ranked)
        print(
            f"{index}/{len(candidates)} {candidate.candidate_id} "
            f"trades={row.get('total_trades', 0)} pnl={row.get('total_pnl', 0.0)} "
            f"pf={row.get('profit_factor', 0.0)} pass50k={row.get('topstep_50k_daily_enforced_passed', False)}",
            flush=True,
        )

    ranked_rows = sorted_rows(rows)
    write_csv(run_dir / "candidate_scores.csv", ranked_rows)
    write_json(run_dir / "candidate_details.json", details)

    if ranked_rows:
        best_id = str(ranked_rows[0]["candidate_id"])
        best = next(item for item in details if item.get("candidate_id") == best_id)
        write_json(run_dir / "best_config.json", {"candidate_id": best_id, "strategy_config": best["strategy_config"]})
        write_csv(run_dir / "best_trades.csv", best.get("portfolio_trades", []))

    core_rows, core_details = validate_top_candidates(
        args=args,
        candidates=candidates,
        ranked_rows=ranked_rows,
        config=config,
        data=data,
        combine_rules=combine_rules,
    )
    if core_rows:
        write_csv(run_dir / "core_validation_scores.csv", core_rows)
        write_json(run_dir / "core_validation_details.json", core_details)

    write_report(run_dir, ranked_rows, core_rows, args, elapsed=time.perf_counter() - started)
    print(f"Report: {run_dir / 'analysis.md'}", flush=True)
    return 0


def filter_candidates(candidates: list[Candidate], args: argparse.Namespace) -> list[Candidate]:
    requested_ids = {str(item) for item in args.candidate_id}
    requested_families = {str(item) for item in args.family}
    if requested_ids:
        candidates = [candidate for candidate in candidates if candidate.candidate_id in requested_ids]
        missing = sorted(requested_ids - {candidate.candidate_id for candidate in candidates})
        if missing:
            raise ValueError(f"Unknown candidate id(s): {', '.join(missing)}")
    if requested_families:
        candidates = [candidate for candidate in candidates if candidate.family in requested_families]
        missing = sorted(requested_families - {candidate.family for candidate in candidates})
        if missing:
            raise ValueError(f"Unknown candidate family/families: {', '.join(missing)}")
    requested_substrings = [str(item) for item in args.candidate_id_contains]
    if requested_substrings:
        candidates = [
            candidate
            for candidate in candidates
            if any(fragment in candidate.candidate_id for fragment in requested_substrings)
        ]
    if int(args.max_candidates) > 0:
        candidates = candidates[: int(args.max_candidates)]
    return candidates


def resample_loaded_data(data: dict[str, dict[str, Any]], timeframe: str) -> dict[str, dict[str, Any]]:
    normalized = str(timeframe or "1m").strip().lower()
    if normalized in ("", "1m", "1min"):
        return data
    out: dict[str, dict[str, Any]] = {}
    for symbol, payload in data.items():
        frame = resample_timeframe(payload["frame"], normalized)
        out[symbol] = {**payload, "frame": frame, "path": f"{payload['path']}::{normalized}"}
    return out


def validate_top_candidates(
    *,
    args: argparse.Namespace,
    candidates: list[Candidate],
    ranked_rows: list[dict[str, Any]],
    config: dict[str, Any],
    data: dict[str, dict[str, Any]],
    combine_rules: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    top_n = int(args.validate_top_n)
    if top_n <= 0 or not ranked_rows:
        return [], []

    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    scenarios = scenario_definitions("core")
    for rank, ranked in enumerate(ranked_rows[:top_n], start=1):
        candidate = candidate_by_id[str(ranked["candidate_id"])]
        for scenario in scenarios:
            evaluation = evaluate_strict_candidate(candidate, config, data, combine_rules, scenario, args)
            evaluation["candidate_family"] = candidate.family
            row = flatten_evaluation(evaluation)
            row["search_rank"] = rank
            row["scenario"] = scenario["name"]
            rows.append(row)
            details.append(evaluation)
            print(
                f"core {rank}/{top_n} {candidate.candidate_id} {scenario['name']} "
                f"trades={row.get('total_trades', 0)} pnl={row.get('total_pnl', 0.0)} "
                f"pf={row.get('profit_factor', 0.0)} pass50k={row.get('topstep_50k_daily_enforced_passed', False)}",
                flush=True,
            )
    return rows, details


def evaluate_strict_candidate(
    candidate: Candidate,
    config: dict[str, Any],
    data: dict[str, dict[str, Any]],
    combine_rules: dict[str, dict[str, Any]],
    scenario: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.execution_model == "resting_stop":
        return evaluate_resting_stop_orders(
            candidate_id=candidate.candidate_id,
            strategy_config=candidate.strategy_config,
            config=config,
            data=data,
            combine_rules=combine_rules,
            scenario=scenario,
            max_pending_bars=int(args.max_pending_bars),
        )
    return evaluate_next_bar_open(
        candidate_id=candidate.candidate_id,
        strategy_config=candidate.strategy_config,
        config=config,
        data=data,
        combine_rules=combine_rules,
        scenario=scenario,
    )


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    run_dir: Path,
    ranked_rows: list[dict[str, Any]],
    core_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    elapsed: float,
) -> None:
    lines = [
        "# Strict No-Lookahead Portfolio Optimization",
        "",
        f"- Execution model: `{args.execution_model}`",
        f"- Search scenario: `{args.search_scenario}`",
        f"- Risk per trade: `{args.risk}`",
        f"- Tail bars: `{args.tail_bars}`",
        f"- Timeframe: `{args.timeframe}`",
        f"- Max pending bars: `{args.max_pending_bars}`",
        f"- Candidates evaluated: `{len(ranked_rows)}`",
        f"- Elapsed seconds: `{elapsed:.1f}`",
        "",
        "## Search Ranking",
        "",
        markdown_table(ranked_rows[:20]),
    ]
    if core_rows:
        lines.extend(["", "## Core Validation", "", markdown_table(core_rows)])
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Candidate scores: `{run_dir / 'candidate_scores.csv'}`",
            f"- Candidate details: `{run_dir / 'candidate_details.json'}`",
            f"- Best config: `{run_dir / 'best_config.json'}`",
            f"- Best trades: `{run_dir / 'best_trades.csv'}`",
        ]
    )
    if core_rows:
        lines.extend(
            [
                f"- Core validation scores: `{run_dir / 'core_validation_scores.csv'}`",
                f"- Core validation details: `{run_dir / 'core_validation_details.json'}`",
            ]
        )
    (run_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    columns = [
        "candidate_id",
        "candidate_family",
        "scenario",
        "total_pnl",
        "avg_daily_pnl_trade_dates",
        "total_trades",
        "profit_factor",
        "best_day_consistency_pct",
        "worst_daily_loss",
        "max_drawdown",
        "topstep_50k_daily_enforced_passed",
        "score",
    ]
    out = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(format_value(row.get(column, "")) for column in columns) + " |")
    return "\n".join(out)


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())
