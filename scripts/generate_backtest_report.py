import os
import logging
import argparse
from datetime import datetime
from src.utils.backtester import Backtester

def generate_markdown_report(results, symbol):
    summary = results['summary']
    trades = results['trades']
    
    report = f"""# Trading Swarm Performance Report: {symbol}
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
| Metric | Value |
| :--- | :--- |
| **Total Net P&L** | **${summary['total_pnl']:,}** |
| **Profit Factor** | {summary['profit_factor']} |
| **Win Rate** | {summary['win_rate']}% |
| **Max Drawdown** | ${summary['max_drawdown']:,} |
| **Sharpe Ratio** | {summary['sharpe_ratio']} |
| Final Balance | ${summary['final_balance']:,} |
| Total Trades | {summary['total_trades']} |
| Avg Trade | ${summary['avg_trade']} |

## Trade Distribution
- **Wins:** {summary['wins']}
- **Losses:** {summary['losses']}
- **Scratches (BE/Exit):** {summary['scratches']}

## Trade Log (Last 10)
| Time Close | Side | Qty | Entry | Exit | P&L | Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for t in trades[-10:]:
        report += f"| {t['time_close']} | {t['side']} | {t['qty']} | {t['entry']:.2f} | {t['exit']:.2f} | ${t['pnl']:.2f} | {t['result']} |\n"
        
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run professional backtest and generate report.")
    parser.add_argument("--symbol", type=str, required=True, help="Symbol (e.g., MNQ, NQ)")
    parser.add_argument("--file", type=str, required=True, help="Path to historical CSV")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    backtester = Backtester(args.symbol)
    results = backtester.run(args.file)
    
    if results:
        report = generate_markdown_report(results, args.symbol)
        
        # Save report
        report_file = f"reports/backtest_{args.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        os.makedirs("reports", exist_ok=True)
        with open(report_file, "w") as f:
            f.write(report)
            
        print("\n[BACKTEST COMPLETE]")
        print(f"Report generated: {report_file}")
        print("\n--- Summary ---")
        for k, v in results['summary'].items():
            print(f"{k.replace('_', ' ').title()}: {v}")
