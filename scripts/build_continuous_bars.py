#!/usr/bin/env python3
"""Build continuous, deduplicated 1m bar series per root symbol from per-contract CSVs.

Offline data prep only. Roll rule: each UTC trading day uses the contract with the highest
volume that day, with hysteresis (never rolls back to an earlier contract). Prices are
additively back-adjusted at each roll using the close difference at the roll timestamp, which
preserves point P&L. The unadjusted close is kept in `close_raw`.

Outputs: data/continuous/<ROOT>_1m.csv and data/continuous/gap_report.json
"""
from __future__ import annotations

import glob
import json
import re
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "continuous"
# root -> (old-file prefixes, backfill product code)
ROOTS = {"MCL": ("mcl", "MCLE"), "MGC": ("mgc", "MGC"), "MES": ("mes", "MES")}
MONTH_ORDER = "FGHJKMNQUVXZ"


def contract_rank(code: str) -> int:
    m = re.search(r"([FGHJKMNQUVXZ])(\d{1,2})$", code)
    return (int(m.group(2)) % 10) * 12 + MONTH_ORDER.index(m.group(1)) if m else 0


def load_frames(root: str) -> dict[str, pd.DataFrame]:
    prefix, prod = ROOTS[root]
    frames: dict[str, list[pd.DataFrame]] = {}
    for f in glob.glob(str(ROOT / "data" / f"{prefix}[a-z][0-9]_1m_*.csv")):
        code = Path(f).name.split("_")[0].upper()
        frames.setdefault(code, []).append(pd.read_csv(f))
    for f in glob.glob(str(ROOT / "data" / "backfill" / f"CON.F.US.{prod}.*.csv")):
        code = prefix.upper() + Path(f).stem.split(".")[-1][0] + Path(f).stem.split(".")[-1][-1]
        frames.setdefault(code, []).append(pd.read_csv(f))
    out = {}
    for code, parts in frames.items():
        d = pd.concat(parts)
        d.columns = [c.lower() for c in d.columns]
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
        d = d.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        d["volume"] = pd.to_numeric(d.get("volume", 0), errors="coerce").fillna(0)
        out[code] = d[["timestamp", "open", "high", "low", "close", "volume"]]
    return out


def build(root: str):
    frames = load_frames(root)
    if not frames:
        return None, {}
    allb = pd.concat([d.assign(contract=c) for c, d in frames.items()])
    allb["day"] = allb["timestamp"].dt.date
    dayvol = allb.groupby(["day", "contract"])["volume"].sum().unstack(fill_value=0)
    chosen, current = {}, None
    for day, row in dayvol.iterrows():
        best = row.idxmax()
        if current is None or (contract_rank(best) > contract_rank(current) and row[best] > row.get(current, 0)):
            current = best
        elif row.get(current, 0) == 0 and best != current:
            current = best  # current contract has no data this day
        chosen[day] = current
    sel = allb[allb["day"].map(chosen) == allb["contract"]].sort_values("timestamp").copy()
    sel = sel.drop_duplicates("timestamp").reset_index(drop=True)
    sel["close_raw"] = sel["close"]
    # additive back-adjustment at each roll
    rolls = []
    sel["adj"] = 0.0
    idx_change = sel.index[sel["contract"] != sel["contract"].shift()].tolist()[1:]
    for i in reversed(idx_change):
        prev_c, new_c = sel.at[i - 1, "contract"], sel.at[i, "contract"]
        t = sel.at[i, "timestamp"]
        pf, nf = frames[prev_c], frames[new_c]
        p = pf[pf["timestamp"] <= t].tail(1)
        n = nf[nf["timestamp"] <= t].tail(1)
        diff = float(n["close"].iloc[0] - p["close"].iloc[0]) if len(p) and len(n) and (t - p["timestamp"].iloc[0]).total_seconds() < 3 * 86400 else 0.0
        sel.loc[: i - 1, "adj"] += diff
        rolls.append({"at": str(t), "from": prev_c, "to": new_c, "basis": round(diff, 4)})
    for c in ["open", "high", "low", "close"]:
        sel[c] = sel[c] + sel["adj"]
    # gap report
    days = pd.to_datetime(pd.Series(sorted(sel["timestamp"].dt.date.unique())))
    g = days.diff().dt.days
    gaps = [(str(days[i - 1].date()), str(days[i].date())) for i in g[g > 4].index]
    per_day = sel.groupby(sel["timestamp"].dt.date).size()
    report = {
        "bars": int(len(sel)), "start": str(sel.timestamp.min()), "end": str(sel.timestamp.max()),
        "trading_days": int(len(per_day)), "full_days_ge_300": int((per_day >= 300).sum()),
        "gaps_gt_4d": gaps, "rolls": sorted(rolls, key=lambda r: r["at"]),
        "contracts": sorted(frames, key=contract_rank),
    }
    return sel[["timestamp", "open", "high", "low", "close", "volume", "contract", "close_raw"]], report


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {}
    for root in ROOTS:
        df, r = build(root)
        if df is None:
            print(f"{root}: no data")
            continue
        df.to_csv(OUT / f"{root}_1m.csv", index=False)
        rep[root] = r
        print(f"{root}: {r['bars']} bars {r['start'][:10]} -> {r['end'][:10]} days={r['trading_days']} full={r['full_days_ge_300']}")
        for ro in r["rolls"]:
            print(f"   roll {ro['at'][:10]} {ro['from']}->{ro['to']} basis={ro['basis']}")
        print(f"   gaps>4d: {r['gaps_gt_4d']}")
    (OUT / "gap_report.json").write_text(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
