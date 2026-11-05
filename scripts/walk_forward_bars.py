#!/usr/bin/env python3
"""Offline walk-forward / holdout evaluation of the live strategy config on continuous bars.

No tuning happens here: the configured strategy is run unchanged over each contiguous data
segment (split at gaps > 4 days so indicators never span missing weeks), and trades are bucketed
into time blocks. The last --holdout-days are reported separately as an untouched holdout.
Results are observational only and never feed live trading.

Usage:
  python scripts/walk_forward_bars.py [--config config/trading_stack.yaml] [--block-days 14]
         [--holdout-days 21] [--roots MCL,MGC,MES] [--only mtf_trend_scalper,orb_vwap]
"""
from __future__ import annotations

import argparse
import copy
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.settings import get_strategy_config, load_stack_config  # noqa: E402
from src.utils.backtester import Backtester  # noqa: E402

GAP = pd.Timedelta(days=4)


def segments(df: pd.DataFrame) -> list[pd.DataFrame]:
    breaks = df["timestamp"].diff() > GAP
    ids = breaks.cumsum()
    return [g.reset_index(drop=True) for _, g in df.groupby(ids) if len(g) > 600]


def restrict(strategy_config: dict, only: set[str] | None) -> dict:
    cfg = copy.deepcopy(strategy_config)
    if only:
        for s in cfg.get("strategies", []):
            if s.get("name") not in only:
                s["enabled"] = False
    return cfg


def run_root(root: str, cfg: dict, data_dir: Path) -> list[dict]:
    df = pd.read_csv(data_dir / f"{root}_1m.csv", parse_dates=["timestamp"])
    out = []
    for si, seg in enumerate(segments(df)):
        bt = Backtester(root, initial_balance=50000, strategy_config=copy.deepcopy(cfg),
                        telemetry_enabled=False, screenshot_config={"enabled": False})
        res = bt.run_dataframe(pd.DataFrame(seg[["timestamp", "open", "high", "low", "close", "volume"]]),
                               dataset_label=f"{root}_seg{si}")
        n = 0
        for t in res.get("trades", []):
            n += 1
            out.append({"root": root, "segment": si, "side": t["side"], "pnl": float(t["pnl"]),
                        "strategy": t.get("strategy_name", "unknown"), "result": t.get("result"),
                        "time_open": pd.to_datetime(t["time_open"], utc=True),
                        "time_close": pd.to_datetime(t["time_close"], utc=True), "qty": t.get("qty", 1)})
        print(f"  {root} seg{si}: {seg.timestamp.min():%m-%d}->{seg.timestamp.max():%m-%d} "
              f"bars={len(seg)} trades={n}", flush=True)
    return out


def stats(g: pd.DataFrame) -> pd.Series:
    w, l = g.loc[g.pnl > 0, "pnl"].sum(), -g.loc[g.pnl < 0, "pnl"].sum()
    return pd.Series({"n": len(g), "pnl": round(g.pnl.sum(), 1), "win%": round(100 * (g.pnl > 0).mean(), 1),
                      "avg": round(g.pnl.mean(), 2), "pf": round(w / l, 2) if l else float("inf")})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-dir", default="data/continuous")
    ap.add_argument("--roots", default="MCL,MGC,MES")
    ap.add_argument("--only", default="")
    ap.add_argument("--block-days", type=int, default=14)
    ap.add_argument("--holdout-days", type=int, default=21)
    ap.add_argument("--out", default="reports/walk_forward_bars")
    a = ap.parse_args()

    stack = load_stack_config(a.config) if a.config else load_stack_config()
    cfg = restrict(get_strategy_config(stack), set(filter(None, a.only.split(","))) or None)
    rows: list[dict] = []
    for root in [r.strip() for r in a.roots.split(",") if r.strip()]:
        print(f"== {root}", flush=True)
        rows += run_root(root, cfg, ROOT / a.data_dir)
    if not rows:
        print("no trades")
        return 0
    t = pd.DataFrame(rows).sort_values("time_close")
    end = t.time_close.max()
    t["block"] = (t.time_close - t.time_close.min()).dt.days // a.block_days
    t["block_start"] = (t.time_close.min() + pd.to_timedelta(t.block * a.block_days, unit="D")).dt.strftime("%m-%d")
    t["holdout"] = t.time_close > end - pd.Timedelta(days=a.holdout_days)
    outdir = ROOT / a.out
    outdir.mkdir(parents=True, exist_ok=True)
    t.to_csv(outdir / "trades.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nOVERALL"); print(stats(t).to_string())
    print("\nBY ROOT x STRATEGY"); print(t.groupby(["root", "strategy"]).apply(stats))
    print("\nBY SIDE"); print(t.groupby(["root", "side"]).apply(stats))
    print(f"\nBY {a.block_days}-DAY BLOCK (all)"); print(t.groupby("block_start").apply(stats))
    print("\nBLOCK pnl x ROOT x STRATEGY")
    print(t.pivot_table(index="block_start", columns=["root", "strategy"], values="pnl", aggfunc="sum").round(0).fillna(0))
    print(f"\nHOLDOUT (last {a.holdout_days}d) vs TRAIN"); print(t.groupby(["holdout", "root", "strategy"]).apply(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
