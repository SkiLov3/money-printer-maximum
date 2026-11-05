#!/usr/bin/env python3
"""
TopstepX Trading Swarm — Unified Backtest Runner

Usage:
  python run_backtest.py <symbol> <csv_path>                  # Single symbol
  python run_backtest.py <symbol> <csv_path> --days 5         # Last N trading days only
  python run_backtest.py --multi                               # Run predefined multi-symbol suite
  python run_backtest.py --multi --days 5                      # Multi-symbol, last 5 days
"""
import argparse
import json
import logging
import os
import sys
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from src.utils.backtester import Backtester
from src.api.projectx_client import ProjectXClient
from src.config.settings import get_combine_rule_keys, get_combine_rules, get_strategy_config, load_stack_config
from src.data.catalog import DatasetCatalog
from src.optimization.combine_rules import evaluate_combine_rules

load_dotenv()

logging.basicConfig(level=logging.WARNING)
DATASET_CATALOG = DatasetCatalog()
STACK_CONFIG = load_stack_config()
STRATEGY_CONFIG = get_strategy_config(STACK_CONFIG)
DEFAULT_COMBINE_RULES = get_combine_rules(STACK_CONFIG, "topstep_50k")

# Default multi-symbol test suite
MULTI_SUITE = [
    ("MNQM6", "data/mnq_7w_rth_full.csv"),
    ("MCLK6", "data/mcl_7w_rth_full.csv"),
]


def slice_last_n_days(csv_path, n_days):
    """Loads a CSV and returns only the last N trading days of data."""
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    unique_dates = sorted(df['date'].unique())
    if len(unique_dates) <= n_days:
        return csv_path  # Not enough data to slice, use all

    cutoff_dates = unique_dates[-n_days:]
    sliced = df[df['date'].isin(cutoff_dates)].drop(columns=['date']).reset_index(drop=True)

    # We need 200 bars of warmup from BEFORE the cutoff for EMA-200
    warmup_df = df[~df['date'].isin(cutoff_dates)].drop(columns=['date']).tail(200)
    full = pd.concat([warmup_df, sliced], ignore_index=True)

    # Save to temp file
    tmp_path = csv_path.replace('.csv', f'_last{n_days}d.csv')
    full.to_csv(tmp_path, index=False)
    return tmp_path


def get_previous_business_week_range():
    """Calculates the start and end of the most recent full business week (Mon-Fri)."""
    today = datetime.now(timezone.utc)
    # weekday(): 0=Monday, 6=Sunday
    days_to_monday = today.weekday()
    
    # If today is Sat(5) or Sun(6), previous business week is the one that just ended.
    # Otherwise, it's the full week prior to the current partially-complete week.
    offset = 7
    if today.weekday() >= 5: # Weekend
        offset = 7
    
    monday = (today - timedelta(days=days_to_monday + offset)).replace(hour=0, minute=0, second=0, microsecond=0)
    friday = (monday + timedelta(days=4)).replace(hour=23, minute=59, second=59, microsecond=0)
    
    return monday, friday


def fetch_live_data(symbol):
    """Fetches previous business week's data from API and returns the path to the saved CSV."""
    client = ProjectXClient()
    if not client.login():
        print("  ERROR: Failed to login to TopstepX API. Check your .env credentials.")
        sys.exit(1)

    # 1. Resolve Symbol (e.g. NQ -> NQM6)
    resolved = client.resolve_symbol(symbol)
    if not resolved:
        print(f"  ERROR: Could not resolve active contract for {symbol}")
        sys.exit(1)
    
    contract_id = client.symbol_to_id_cache.get(resolved)
    start_dt, end_dt = get_previous_business_week_range()
    
    print(f"  FETCHING: {resolved} ({contract_id})")
    print(f"  RANGE:    {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")

    # 2. Retrieve Bars
    bars_data = client.retrieve_bars(
        contract_id=contract_id,
        start_time=start_dt,
        end_time=end_dt,
        unit=2,  # Minute
        unit_number=1,
        limit=20000
    )

    bars = bars_data.get('bars', [])
    if not bars:
        print(f"  ERROR: No data returned for {resolved} in that date range.")
        sys.exit(1)

    # 3. Process and Sort
    df = pd.DataFrame(bars)
    # Mapping for TopstepX abbreviated keys: t=timestamp, o=open, h=high, l=low, c=close, v=volume
    mapping = {'t': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}
    df = df.rename(columns=mapping)
    df.columns = [c.lower() for c in df.columns]
    
    # Ensure chronological order
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 4. Save to CSV
    os.makedirs('data', exist_ok=True)
    filename = f"{symbol}_auto_fetch.csv"
    output_path = os.path.join('data', filename)
    df.to_csv(output_path, index=False)
    DATASET_CATALOG.register_dataset(symbol, output_path, source="projectx_fetch")
    
    print(f"  SAVED:    {output_path} ({len(df)} bars)")
    return output_path


def run_single(symbol, csv_path, label=None, strategy_config=None, combine_rules=None):
    """Run a backtest for a single symbol and print results."""
    display = label or symbol
    print(f"\n{'='*60}")
    print(f"  BACKTEST: {display}")
    print(f"  Data:     {csv_path}")
    print(f"{'='*60}")

    if not os.path.exists(csv_path):
        print(f"  ERROR: File not found — {csv_path}")
        return None

    bt = Backtester(symbol, strategy_config=strategy_config)
    results = bt.run(csv_path)

    if not results:
        print("  ERROR: Backtest returned no results!")
        return None

    s = results['summary']
    trades = results['trades']
    equity = results['equity_history']
    by_strategy = results.get('by_strategy', {})
    combine_eval = evaluate_combine_rules(results, combine_rules or DEFAULT_COMBINE_RULES)

    # Additional metrics
    wins_pnl = [t['pnl'] for t in trades if t['pnl'] > 0]
    losses_pnl = [t['pnl'] for t in trades if t['pnl'] < 0]
    avg_win = sum(wins_pnl) / len(wins_pnl) if wins_pnl else 0
    avg_loss = sum(losses_pnl) / len(losses_pnl) if losses_pnl else 0
    profit_factor = abs(sum(wins_pnl) / sum(losses_pnl)) if losses_pnl and sum(losses_pnl) != 0 else float('inf')

    peak = equity[0]
    max_dd = 0
    for eq in equity:
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd

    # Exit type breakdown
    exit_types = {}
    for t in trades:
        r = t['result']
        if r not in exit_types:
            exit_types[r] = {'count': 0, 'pnl': 0}
        exit_types[r]['count'] += 1
        exit_types[r]['pnl'] += t['pnl']

    print(f"\n  Total Trades:    {s['total_trades']}")
    print(f"  Winners:         {s['wins']}  ({s['win_rate']:.1f}%)")
    print(f"  Losers:          {s['losses']}")
    print(f"  Scratches (BE):  {s['scratches']}")
    print(f"  Total PnL:       ${s['total_pnl']:,.2f}")
    print(f"  Final Balance:   ${s['final_balance']:,.2f}")
    print(f"  Avg Winner:      ${avg_win:,.2f}")
    print(f"  Avg Loser:       ${avg_loss:,.2f}")
    print(f"  Profit Factor:   {profit_factor:.2f}")
    print(f"  Max Drawdown:    ${max_dd:,.2f}")
    print(f"  Combine 50K:     {'PASS' if combine_eval['passed'] else 'FAIL'}")
    print(
        f"    Profit Target  {'OK' if combine_eval['checks']['profit_target']['ok'] else 'NO'}  "
        f"({combine_eval['checks']['profit_target']['actual']:.2f} / {combine_eval['checks']['profit_target']['target']:.2f})"
    )
    limit_info = combine_eval['checks']['max_loss_limit']
    floor = limit_info.get('balance_floor', limit_info.get('final_mll', limit_info.get('initial_floor', 0.0)))
    print(
        f"    Max Loss Rule  {'OK' if limit_info['ok'] else 'NO'}  "
        f"(min balance {limit_info['min_balance_seen']:.2f} > "
        f"{floor:.2f})"
    )
    print(
        f"    Consistency    {'OK' if combine_eval['checks']['consistency']['ok'] else 'NO'}  "
        f"(best day {combine_eval['checks']['consistency']['actual_best_day_pct']:.2f}% < "
        f"{combine_eval['checks']['consistency']['max_best_day_pct']:.2f}%)"
    )

    if by_strategy:
        print("\n  By Strategy:")
        for strategy_name, summary in sorted(by_strategy.items()):
            print(
                f"    {strategy_name:16s} Trades: {summary['trades']:3d}  "
                f"Win Rate: {summary['win_rate']:5.1f}%  PnL: ${summary['pnl']:>9,.2f}"
            )

    print("\n  Exit Breakdown:")
    for etype, data in sorted(exit_types.items()):
        print(f"    {etype:10s}  Count: {data['count']:3d}  PnL: ${data['pnl']:>10,.2f}")

    if trades:
        print("\n  Last 5 Trades:")
        for t in trades[-5:]:
            print(f"    {t['side']:4s} | Entry: {t['entry']:>10.2f} | Exit: {t['exit']:>10.2f} | "
                  f"PnL: ${t['pnl']:>8.2f} | {t['result']:8s} | Qty: {t['qty']}")

    return {
        'symbol': symbol,
        'summary': s,
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 'inf',
        'max_drawdown': round(max_dd, 2),
        'exit_breakdown': {k: {'count': v['count'], 'pnl': round(v['pnl'], 2)} for k, v in exit_types.items()},
        'by_strategy': by_strategy,
        'combine_evaluation': combine_eval,
        'trades': trades
    }


def print_combined(all_results):
    """Print combined summary across all symbols."""
    print(f"\n{'='*60}")
    print("  COMBINED RESULTS")
    print(f"{'='*60}")
    total_pnl = sum(r['summary']['total_pnl'] for r in all_results.values())
    total_trades = sum(r['summary']['total_trades'] for r in all_results.values())
    total_wins = sum(r['summary']['wins'] for r in all_results.values())
    combined_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    combined_max_dd = max((r['max_drawdown'] for r in all_results.values()), default=0)
    passed_symbols = [sym for sym, data in all_results.items() if data.get("combine_evaluation", {}).get("passed")]

    print(f"  Total Trades:      {total_trades}")
    print(f"  Combined Win Rate: {combined_wr:.1f}%")
    print(f"  Combined PnL:      ${total_pnl:,.2f}")
    print(f"  Worst Drawdown:    ${combined_max_dd:,.2f}")
    print(f"  Combine Passes:    {len(passed_symbols)}/{len(all_results)} ({', '.join(passed_symbols) if passed_symbols else 'None'})")


def main():
    parser = argparse.ArgumentParser(description="TopstepX Backtest Runner")
    parser.add_argument("symbol", nargs="?", help="Symbol (e.g., MNQM6)")
    parser.add_argument("csv_path", nargs="?", help="Path to CSV data file")
    parser.add_argument("--multi", action="store_true", help="Run all symbols in the test suite")
    parser.add_argument("--fetch", action="store_true", help="Auto-fetch previous business week data from API")
    parser.add_argument("--days", type=int, default=0, help="Limit to last N trading days")
    parser.add_argument("--save", type=str, default=None, help="Save results JSON to path")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML stack config")
    parser.add_argument("--combine", type=str, default="topstep_50k", help="Combine rule set key from YAML")
    parser.add_argument("--list-combine", action="store_true", help="List available combine rule set keys in config")
    args = parser.parse_args()

    stack_config = load_stack_config(args.config)
    combine_keys = get_combine_rule_keys(stack_config)
    if args.list_combine:
        print("\nAvailable combine rule sets:")
        for key in combine_keys:
            print(f"  - {key}")
        return

    if args.combine not in combine_keys:
        print(f"ERROR: Unknown combine ruleset '{args.combine}'.")
        if combine_keys:
            print("Available keys:")
            for key in combine_keys:
                print(f"  - {key}")
        sys.exit(1)

    strategy_config = get_strategy_config(stack_config)
    combine_rules = get_combine_rules(stack_config, args.combine) or DEFAULT_COMBINE_RULES

    if not args.multi and not args.fetch and not (args.symbol and args.csv_path):
        parser.print_help()
        sys.exit(1)

    # Handle fetch logic
    if args.fetch:
        if not args.symbol:
            print("ERROR: --fetch requires a symbol (e.g. NQ)")
            sys.exit(1)
        csv_path = fetch_live_data(args.symbol)
        tests = [(args.symbol, csv_path)]
    else:
        tests = MULTI_SUITE if args.multi else [(args.symbol, args.csv_path)]
    day_label = f" (Last {args.days} trading days)" if args.days else " (Full dataset)"
    print(f"\n  Mode:{day_label}")

    all_results = {}
    for symbol, csv_path in tests:
        effective_path = csv_path
        if args.days > 0:
            effective_path = slice_last_n_days(csv_path, args.days)

        result = run_single(
            symbol,
            effective_path,
            label=f"{symbol}{day_label}",
            strategy_config=strategy_config,
            combine_rules=combine_rules,
        )
        if result:
            all_results[symbol] = result

    if len(all_results) > 1:
        print_combined(all_results)

    if args.save:
        with open(args.save, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Results saved to: {args.save}")


if __name__ == "__main__":
    main()
