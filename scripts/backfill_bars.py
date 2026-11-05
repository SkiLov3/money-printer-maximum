#!/usr/bin/env python3
# pyright: reportArgumentType=false
"""Read-only 1m bar backfill for explicit ProjectX contract IDs (including expired ones).

Places no orders. Output: data/backfill/<contract_id>.csv (timestamp,open,high,low,close,volume).

Usage:
  python scripts/backfill_bars.py --start 2026-05-01 CON.F.US.MCLE.V26 CON.F.US.MGC.Z26
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.projectx_client import ProjectXClient  # noqa: E402

CHUNK_MINUTES = 15_000


def fetch(client, contract_id, start, end):
    frames, cursor = [], start
    while cursor < end:
        chunk_end = min(end, cursor + timedelta(minutes=CHUNK_MINUTES))
        payload = client.retrieve_bars(contract_id=contract_id, start_time=cursor, end_time=chunk_end,
                                       unit=2, unit_number=1, limit=20000)
        bars = payload.get("bars", []) if isinstance(payload, dict) else []
        if bars:
            f = pd.DataFrame(bars).rename(columns={"t": "timestamp", "o": "open", "h": "high",
                                                   "l": "low", "c": "close", "v": "volume"})
            f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)
            frames.append(f[["timestamp", "open", "high", "low", "close", "volume"]])
            last = f["timestamp"].max().to_pydatetime()
            # advance past last bar; if the API capped the chunk, resume from there
            cursor = max(last + timedelta(minutes=1), cursor + timedelta(minutes=1)) if last < chunk_end - timedelta(minutes=5) else chunk_end
        else:
            cursor = chunk_end
        time.sleep(0.6)  # gentle rate limit
    if not frames:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    return pd.concat(frames).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("contracts", nargs="+")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", default=None)
    ap.add_argument("--outdir", default="data/backfill")
    a = ap.parse_args()
    load_dotenv()
    start = datetime.fromisoformat(a.start).replace(tzinfo=timezone.utc)
    end = (datetime.fromisoformat(a.end).replace(tzinfo=timezone.utc) if a.end
           else datetime.now(timezone.utc).replace(second=0, microsecond=0))
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    client = ProjectXClient()
    if not client.login():
        raise SystemExit("login failed")
    for cid in a.contracts:
        df = fetch(client, cid, start, end)
        if df.empty:
            print(f"{cid}: no bars", flush=True)
            continue
        out = Path(a.outdir) / f"{cid}.csv"
        df.to_csv(out, index=False)
        print(f"{cid}: {len(df)} bars {df.timestamp.min():%Y-%m-%d} -> {df.timestamp.max():%Y-%m-%d} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
