#!/usr/bin/env python3
"""Offline parameter hypothesis tester against the continuous bar dataset.

Tests specific quantitative hypotheses proposed by Qwen 3.6 against the baseline:
- Baseline: Live config (mtf_trend_scalper on MCL, MGC, MES; orb_vwap on MCL)
- H1: Noise suppression (widen sl_atr_mult from 1.5 -> 2.0, tp_atr_mult 3.0 -> 4.0)
- H2: Asset/Hour pruning (disable MGC; disable MES 15:00 ET; disable MCL 14:00 ET)
- H1+H2 Combined: Both H1 and H2

Evaluates each on both the training set (< last 21d) and the untouched 21-day holdout.
Observational/offline only per AGENTS.md.
"""
from __future__ import annotations

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


def run_experiment(name: str, strat_cfg: dict, roots: list[str], data_dir: Path, holdout_days: int = 21):
    trades = []
    for root in roots:
        df = pd.read_csv(data_dir / f"{root}_1m.csv", parse_dates=["timestamp"])
        for si, seg in enumerate(segments(df)):
            bt = Backtester(root, initial_balance=50000, strategy_config=copy.deepcopy(strat_cfg),
                            telemetry_enabled=False, screenshot_config={"enabled": False})
            res = bt.run_dataframe(seg[["timestamp", "open", "high", "low", "close", "volume"]],
                                   dataset_label=f"{root}_seg{si}")
            for t in res.get("trades", []):
                trades.append({
                    "root": root, "side": t["side"], "pnl": float(t["pnl"]),
                    "strategy": t.get("strategy_name", "unknown"),
                    "time_open": pd.to_datetime(t["time_open"], utc=True),
                    "time_close": pd.to_datetime(t["time_close"], utc=True),
                })
    if not trades:
        return {"name": name, "n": 0, "pnl": 0.0, "pf": 0.0, "holdout_pnl": 0.0, "holdout_pf": 0.0}
    tdf = pd.DataFrame(trades).sort_values("time_close")
    end = tdf["time_close"].max()
    tdf["holdout"] = tdf["time_close"] > (end - pd.Timedelta(days=holdout_days))

    def stats(g):
        w = g.loc[g.pnl > 0, "pnl"].sum()
        l = -g.loc[g.pnl < 0, "pnl"].sum()
        return len(g), round(g.pnl.sum(), 1), round(w / l, 2) if l else (999.0 if w > 0 else 0.0)

    n_all, pnl_all, pf_all = stats(tdf)
    train_df = tdf[~tdf["holdout"]]
    n_train, pnl_train, pf_train = stats(train_df) if len(train_df) else (0, 0.0, 0.0)
    hold_df = tdf[tdf["holdout"]]
    n_hold, pnl_hold, pf_hold = stats(hold_df) if len(hold_df) else (0, 0.0, 0.0)

    return {
        "experiment": name,
        "n_total": n_all, "pnl_total": pnl_all, "pf_total": pf_all,
        "n_train": n_train, "pnl_train": pnl_train, "pf_train": pf_train,
        "n_holdout": n_hold, "pnl_holdout": pnl_hold, "pf_holdout": pf_hold,
    }


def main():
    stack = load_stack_config()
    base_cfg = get_strategy_config(stack)
    data_dir = ROOT / "data" / "continuous"

    experiments = []

    # 1. Baseline
    experiments.append(("0_Baseline", copy.deepcopy(base_cfg), ["MCL", "MGC", "MES"]))

    # 2. H1: Noise suppression (sl_atr_mult 2.0, tp_atr_mult 4.0)
    h1_cfg = copy.deepcopy(base_cfg)
    for s in h1_cfg.get("strategies", []):
        if s.get("name") == "mtf_trend_scalper":
            s["params"]["sl_atr_mult"] = 2.0
            s["params"]["tp_atr_mult"] = 4.0
    experiments.append(("H1_Wider_ATR_Stop", h1_cfg, ["MCL", "MGC", "MES"]))

    # 3. H2: Asset/Hour Pruning (Disable MGC entirely, prune MES 15h, prune MCL 14h)
    h2_cfg = copy.deepcopy(base_cfg)
    for s in h2_cfg.get("strategies", []):
        if s.get("name") == "mtf_trend_scalper":
            s["params"]["enabled_symbols"] = ["MCL", "MES"]
            # Prune MES 15:00 ET
            s["params"].setdefault("symbol_params", {})
            s["params"]["symbol_params"]["MES"] = {"allowed_entry_hours_et": [9, 10, 11, 13, 14]}
            # Prune MCL 14:00 ET (was 11, 12, 13, 14, 15)
            s["params"]["symbol_params"]["MCL"] = {"allowed_entry_hours_et": [11, 12, 13, 15]}
    experiments.append(("H2_Hour_Asset_Pruning", h2_cfg, ["MCL", "MES"]))

    # 4. H1 + H2 Combined
    h12_cfg = copy.deepcopy(h2_cfg)
    for s in h12_cfg.get("strategies", []):
        if s.get("name") == "mtf_trend_scalper":
            s["params"]["sl_atr_mult"] = 2.0
            s["params"]["tp_atr_mult"] = 4.0
    experiments.append(("H1_plus_H2_Combined", h12_cfg, ["MCL", "MES"]))

    results = []
    for name, cfg, roots in experiments:
        print(f"Running {name} on {roots}...", flush=True)
        res = run_experiment(name, cfg, roots, data_dir)
        results.append(res)
        print(f"  Total: n={res['n_total']} PnL=${res['pnl_total']} PF={res['pf_total']} | "
              f"Holdout: n={res['n_holdout']} PnL=${res['pnl_holdout']} PF={res['pf_holdout']}", flush=True)

    summary_df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("HYPOTHESIS TESTING SUMMARY (Train vs 21-Day Untouched Holdout)")
    print("=" * 80)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
