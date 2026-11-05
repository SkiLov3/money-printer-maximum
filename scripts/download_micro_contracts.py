#!/usr/bin/env python3
# pyright: reportArgumentType=false
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.projectx_client import ProjectXClient


DEFAULT_MICROS = ["MNQ", "MES", "MYM", "MCL", "MGC"]
MAJOR_TO_MICRO = {
    "NQ": "MNQ",
    "ES": "MES",
    "YM": "MYM",
    "CL": "MCL",
    "GC": "MGC",
}
BAR_UNIT_MINUTE = 2
BAR_UNIT_NUMBER = 1
MAX_BARS_PER_REQUEST = 20_000


def parse_utc(ts: str) -> datetime:
    value = ts.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def fetch_range(
    client: ProjectXClient,
    contract_id: str,
    start: datetime,
    end: datetime,
    limit: int,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    cursor = start
    step = timedelta(minutes=limit)

    while cursor < end:
        chunk_end = min(end, cursor + step)
        payload = client.retrieve_bars(
            contract_id=contract_id,
            start_time=cursor,
            end_time=chunk_end,
            unit=BAR_UNIT_MINUTE,
            unit_number=BAR_UNIT_NUMBER,
            limit=limit,
        )
        bars = payload.get("bars", []) if isinstance(payload, dict) else []

        if bars:
            frame = pd.DataFrame(bars).rename(
                columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
            )
            frame.loc[:, "timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            frame = frame[["timestamp", "open", "high", "low", "close", "volume"]]
            frames.append(frame)
            last_ts = frame["timestamp"].max().to_pydatetime()
            cursor = last_ts + timedelta(minutes=1)
        else:
            cursor = chunk_end + timedelta(minutes=1)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return merged.reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download ProjectX active micro-contract data for backtesting.")
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_MICROS),
        help="Comma-separated base products to resolve and download (default: MNQ,MES,MYM,MCL,MGC)",
    )
    parser.add_argument(
        "--microize",
        action="store_true",
        help="Map major symbols to micro equivalents (NQ->MNQ, ES->MES, YM->MYM, CL->MCL, GC->MGC)",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="UTC start time, e.g. 2026-03-01T00:00:00Z",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="UTC end time, e.g. 2026-04-14T00:00:00Z (default: now)",
    )
    parser.add_argument("--outdir", default="data", help="Output directory (default: data)")
    parser.add_argument("--limit", type=int, default=MAX_BARS_PER_REQUEST, help="Bars per API request (default: 20000)")
    parser.add_argument(
        "--env-file",
        default=None,
        help="Optional path to a .env file containing ProjectX credentials",
    )
    args = parser.parse_args()

    if args.env_file:
        load_dotenv(args.env_file, override=False)

    start = parse_utc(args.start)
    end = parse_utc(args.end) if args.end else datetime.now(timezone.utc).replace(second=0, microsecond=0)
    if end <= start:
        raise SystemExit("end must be after start")

    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    if args.microize:
        symbols = [MAJOR_TO_MICRO.get(sym, sym) for sym in symbols]
    os.makedirs(args.outdir, exist_ok=True)

    client = ProjectXClient()
    if not client.login():
        raise SystemExit("ProjectX login failed. Check PROJECTX_API_KEY/TOPSTEPX_USERNAME in .env")

    print(f"Downloading active contracts from {start.isoformat()} to {end.isoformat()}")
    for base_symbol in symbols:
        resolved = client.resolve_symbol(base_symbol)
        if not resolved:
            print(f"[{base_symbol}] could not resolve active contract; skipped")
            continue
        contract_id = client.symbol_to_id_cache.get(resolved)
        if not contract_id:
            print(f"[{base_symbol}] no contract id resolved for {resolved}; skipped")
            continue

        print(f"[{base_symbol}] {resolved} ({contract_id})")
        frame = fetch_range(client, contract_id, start, end, limit=max(1000, int(args.limit)))
        if frame.empty:
            print(f"[{base_symbol}] no bars returned")
            continue

        out_name = f"{resolved.lower()}_1m_{start:%Y%m%d}_{end:%Y%m%d}.csv"
        out_path = os.path.join(args.outdir, out_name)
        frame.to_csv(out_path, index=False)
        print(f"[{base_symbol}] saved {len(frame)} bars -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
