#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import load_stack_config


def _extract_candidate(payload: dict) -> dict:
    if isinstance(payload.get("best_trial"), dict):
        return payload["best_trial"].get("config", {})
    if isinstance(payload.get("status"), dict) and isinstance(payload["status"].get("best_trial"), dict):
        return payload["status"]["best_trial"].get("config", {})
    if isinstance(payload.get("best"), dict):
        return payload["best"].get("config", {})
    raise ValueError("Could not find best candidate config in report.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote the best optimizer/report config into a runnable stack YAML.")
    parser.add_argument("--report", required=True, help="Path to optimization/meta-optimizer JSON report")
    parser.add_argument("--base-config", default="config/trading_stack.yaml")
    parser.add_argument("--output", required=True, help="Output YAML path for promoted stack config")
    args = parser.parse_args()

    base = load_stack_config(args.base_config)
    report_path = Path(args.report)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    candidate = _extract_candidate(payload)
    if not isinstance(candidate, dict) or "strategies" not in candidate:
        raise ValueError("Report candidate config does not contain a strategies block.")

    promoted = dict(base)
    promoted["strategies"] = candidate

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(promoted, sort_keys=False), encoding="utf-8")
    print(str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
