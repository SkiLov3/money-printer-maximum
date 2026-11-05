#!/usr/bin/env python3
"""Fetch raw broker fills (read-only) for a date range and dump them to JSON.

Usage (inside the bot container so credentials are present):
    python scripts/backfill_broker_trades.py --start 2026-10-15 --end 2026-10-25 --out logs/broker_fills.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from src.api.projectx_client import ProjectXClient  # noqa: E402


def fetch_fills(client: ProjectXClient, start: str, end: str) -> list[dict]:
    url = f"{client.base_url}/Trade/search"
    payload = {"accountId": client.account_id, "startTimestamp": start, "endTimestamp": end}
    resp = client._request_with_refresh(requests.post, url, json=payload)
    resp.raise_for_status()
    body = resp.json()
    if not body.get("success", True):
        raise RuntimeError(f"Trade/search failed: {body}")
    return body.get("trades", [])


CONTRACT_BASE_MAP = {"MCLE": "MCL", "CLE": "MCL", "MGC": "MGC", "MES": "MES", "MNQ": "MNQ"}
CONTRACT_SYMBOL = {"MCLE": "MCLZ6", "CLE": "MCLZ6", "MGC": "MGCZ6", "MES": "MESZ6"}


def _symbol_for(contract_id: str) -> str:
    parts = contract_id.split(".")
    if len(parts) >= 2:
        root = parts[-2]
        month_year = parts[-1]
        base = CONTRACT_BASE_MAP.get(root)
        if base and len(month_year) >= 2 and month_year[0].isalpha() and month_year[1:].isdigit():
            return f"{base}{month_year[0]}{month_year[-1]}"
        return CONTRACT_SYMBOL.get(root, root)
    return contract_id


def _infer_strategy(symbol: str, opened_at: datetime) -> str:
    """Best-effort attribution (broker fills carry no strategy tag).

    MCL entries in the Asian opening window (18:00-20:30 ET) belong to
    orb_vwap's Asian fade; everything else live that week was mtf_trend_scalper.
    """
    et = opened_at.astimezone(ZoneInfo("America/New_York"))
    minutes = et.hour * 60 + et.minute
    if symbol.startswith("MCL") and (18 * 60 <= minutes <= 20 * 60 + 30):
        return "orb_vwap"
    return "mtf_trend_scalper"


def pair_round_trips(fills: list[dict]) -> list[dict]:
    """FIFO-pair entry fills (pnl is None) with exit fills (pnl present) per contract, supporting partial fills."""
    open_entries: dict[str, list[dict]] = {}
    trips: list[dict] = []
    for f in sorted((x for x in fills if not x.get("voided")), key=lambda x: x["creationTimestamp"]):
        queue = open_entries.setdefault(f["contractId"], [])
        if f.get("profitAndLoss") is None:
            entry_copy = dict(f)
            entry_copy["_remaining_size"] = int(entry_copy.get("size", 1) or 1)
            queue.append(entry_copy)
            continue
        if not queue:
            continue
        exit_qty = int(f.get("size", 1) or 1)
        entry = queue[0]
        entry_orig_qty = int(entry.get("size", 1) or 1)
        entry_rem_qty = int(entry.get("_remaining_size", entry_orig_qty))
        matched_qty = min(exit_qty, entry_rem_qty)
        if entry_rem_qty <= exit_qty:
            queue.pop(0)
        else:
            entry["_remaining_size"] = entry_rem_qty - matched_qty

        entry_fee_share = float(entry.get("fees", 0)) * (matched_qty / max(entry_orig_qty, 1))
        symbol = _symbol_for(f["contractId"])
        opened = datetime.fromisoformat(entry["creationTimestamp"])
        trips.append({
            "trade_key": str(entry["orderId"]) if matched_qty == entry_orig_qty else f"{entry['orderId']}_{f['orderId']}",
            "symbol": symbol,
            "side": "Buy" if entry["side"] == 0 else "Sell",
            "quantity": matched_qty,
            "entry_price": float(entry["price"]),
            "exit_price": float(f["price"]),
            "opened_at": entry["creationTimestamp"],
            "closed_at": f["creationTimestamp"],
            "entry_order_id": str(entry["orderId"]),
            "exit_order_id": str(f["orderId"]),
            "pnl": float(f["profitAndLoss"]),
            "fees": round(entry_fee_share + float(f.get("fees", 0)), 4),
            "strategy_name": _infer_strategy(symbol, opened),
        })
    return trips


def apply_to_db(trips: list[dict], db_path: str, account_id: str, since: str) -> int:
    from src.persistence.database import DatabaseService
    from src.utils.session_utils import get_trade_session

    db = DatabaseService({"enabled": True, "backend": "sqlite", "sqlite_path": db_path})
    written = 0
    for t in trips:
        if t["closed_at"] < since:
            continue
        opened = datetime.fromisoformat(t["opened_at"])
        session = get_trade_session(opened)
        marker = {"backfilled_from_broker": True, "strategy_inferred": True}
        db.persist_trade_context(
            trade_key=t["trade_key"], symbol=t["symbol"], strategy_name=t["strategy_name"],
            side=t["side"], quantity=t["quantity"], order_id=t["entry_order_id"],
            entry_price=t["entry_price"], opened_at=t["opened_at"], session=session,
            strategy_params={}, selected_signal={"strategy_name": t["strategy_name"], "side": t["side"], "metadata": dict(marker)},
            candidates=[], blender_metadata={}, route_context={}, discovery_context={},
            account_snapshot={"account_id": account_id}, chart_artifacts={},
        )
        db.persist_trade_outcome(
            trade_key=t["trade_key"], symbol=t["symbol"], order_id=t["exit_order_id"],
            account_id=account_id, session=session, pnl=t["pnl"],
            close_type="tp" if t["pnl"] > 0 else "sl", closed_at=t["closed_at"],
            outcome={**marker, "close_type_inferred": True, "fees": t["fees"], "exit_price": t["exit_price"]},
        )
        written += 1
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", default="logs/broker_fills.json")
    ap.add_argument("--apply", action="store_true", help="write paired round trips into the swarm DB")
    ap.add_argument("--db", default="logs/swarm.db")
    ap.add_argument("--since", default="2026-10-15T20:11:00", help="only insert trips closed at/after this UTC time")
    args = ap.parse_args()
    load_dotenv()
    client = ProjectXClient()
    if not client.login():
        print("login failed")
        return 1
    fills = fetch_fills(client, f"{args.start}T00:00:00Z", f"{args.end}T00:00:00Z")
    try:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(fills, fh, indent=1)
    except OSError as exc:
        print(f"could not write {args.out}: {exc}")
        return 1
    print(f"account={client.account_id} fills={len(fills)} -> {args.out}")
    trips = pair_round_trips(fills)
    print(f"round trips paired: {len(trips)}")
    if args.apply:
        n = apply_to_db(trips, args.db, str(client.account_id), args.since)
        print(f"inserted/updated {n} round trips into {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
