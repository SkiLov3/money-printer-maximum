#!/usr/bin/env python3
"""
DevOps Validation: Tests every API endpoint the bot depends on.
Run: python scripts/devops_check.py
"""
import os
import sys
import time
import logging
from datetime import datetime, timedelta, timezone

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from src.api.projectx_client import ProjectXClient

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("DevOps")
logger.setLevel(logging.INFO)

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
WARN = "\033[93m⚠ WARN\033[0m"

def timed(func):
    """Decorator to time a function call."""
    start = time.time()
    result = func()
    elapsed = (time.time() - start) * 1000
    return result, elapsed


def main():
    load_dotenv()
    client = ProjectXClient()
    results = []

    print("\n" + "="*65)
    print("  TopstepX DevOps Validation")
    print("="*65)

    # ── 1. Environment Variables ──
    print("\n  [1] Environment Variables")
    required = {
        "PROJECTX_API_KEY": os.getenv("PROJECTX_API_KEY"),
        "TOPSTEPX_USERNAME": os.getenv("TOPSTEPX_USERNAME"),
        "PROJECTX_ACCOUNT_ID": os.getenv("PROJECTX_ACCOUNT_ID"),
        "SYMBOL": os.getenv("SYMBOL"),
        "MODE": os.getenv("MODE"),
        "MAX_DAILY_LOSS": os.getenv("MAX_DAILY_LOSS"),
        "MAX_DRAWDOWN": os.getenv("MAX_DRAWDOWN"),
    }
    for k, v in required.items():
        status = PASS if v else FAIL
        display = "***" if "KEY" in k or "USER" in k else (v or "MISSING")
        print(f"      {status}  {k} = {display}")
        results.append(("env", k, bool(v)))

    mode = os.getenv("MODE", "simulation").strip()
    if mode == "live":
        print(f"      {WARN}  MODE=live — bot WILL place real orders!")
    else:
        print(f"      {PASS}  MODE={mode} (safe for testing)")

    # ── 2. Authentication ──
    print("\n  [2] POST /Auth/loginKey")
    login_ok, login_ms = timed(lambda: client.login())
    status = PASS if login_ok else FAIL
    print(f"      {status}  Login ({login_ms:.0f}ms)")
    results.append(("api", "login", login_ok))

    if not login_ok:
        print("\n  ✗ Cannot continue without authentication. Aborting.")
        return False

    # ── 3. Account Search ──
    print("\n  [3] POST /Account/search")
    accounts, acct_ms = timed(lambda: client.get_accounts())
    acct_list = accounts.get('accounts', []) if isinstance(accounts, dict) else []
    status = PASS if acct_list else FAIL
    print(f"      {status}  Found {len(acct_list)} account(s) ({acct_ms:.0f}ms)")
    results.append(("api", "accounts", bool(acct_list)))

    # ── 4. Account Status ──
    print("\n  [4] Account Status (ID: {})".format(client.account_id))
    acct_status, status_ms = timed(lambda: client.get_account_status())
    if acct_status:
        balance = acct_status.get('balance', 0)
        daily_pnl = acct_status.get('dailyPnL', 0)
        trailing_dd = acct_status.get('trailingDrawdown', 'N/A')
        print(f"      {PASS}  Balance: ${balance:,.2f} | Daily PnL: ${daily_pnl:,.2f} ({status_ms:.0f}ms)")
        print(f"             Trailing DD: {trailing_dd}")
        
        # Cross-check with .env limits
        max_loss = float(os.getenv("MAX_DAILY_LOSS", 1000))
        max_dd = float(os.getenv("MAX_DRAWDOWN", 2000))
        print(f"             Bot Limits: Max Daily Loss=${max_loss:,.0f} | Max DD=${max_dd:,.0f}")
        
        if daily_pnl <= -max_loss:
            print(f"      {WARN}  Daily loss limit already hit! Bot will NOT trade.")
    else:
        print(f"      {FAIL}  Could not fetch account status ({status_ms:.0f}ms)")
    results.append(("api", "account_status", bool(acct_status)))

    # ── 5. Contract Resolution ──
    symbols = os.getenv("SYMBOL", "MNQ").split(",")
    print(f"\n  [5] Contract Resolution ({', '.join(s.strip() for s in symbols)})")
    for sym in symbols:
        sym = sym.strip()
        resolved, res_ms = timed(lambda s=sym: client.resolve_symbol(s))
        meta = client.get_contract_meta(sym)
        if resolved and meta:
            print(f"      {PASS}  {sym} → {resolved} "
                  f"(tick={meta['tickSize']}, $val={meta['tickValue']}) ({res_ms:.0f}ms)")
        else:
            print(f"      {FAIL}  {sym} → resolution failed ({res_ms:.0f}ms)")
        results.append(("api", f"resolve_{sym}", bool(resolved)))

    # ── 6. Historical Data (retrieve_bars) ──
    print("\n  [6] POST /History/retrieveBars")
    for sym in symbols:
        sym = sym.strip()
        meta = client.get_contract_meta(sym)
        if not meta or not meta.get('id'):
            print(f"      {WARN}  Skipping {sym} — no contract ID cached")
            continue

        contract_id = meta['id']
        end_time = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        start_time = (datetime.now(tz=timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ")

        bars, bars_ms = timed(lambda cid=contract_id: client.retrieve_bars(
            cid, start_time, end_time, unit=2, unit_number=1, limit=100
        ))

        bar_list = bars.get('bars', []) if isinstance(bars, dict) else bars if isinstance(bars, list) else []
        if bar_list:
            print(f"      {PASS}  {sym}: Got {len(bar_list)} bars ({bars_ms:.0f}ms)")
        else:
            print(f"      {WARN}  {sym}: No bars returned (market may be closed) ({bars_ms:.0f}ms)")
        results.append(("api", f"bars_{sym}", bool(bar_list)))

    # ── 7. Order Placement (DRY RUN — simulation check only) ──
    print("\n  [7] Order Endpoint (Dry Check)")
    if mode.lower() != "live":
        print(f"      {PASS}  MODE={mode} — order endpoint NOT tested (safe)")
    else:
        print(f"      {WARN}  MODE=live — skipping order test to avoid real fills")

    # ── 8. SignalR WebSocket ──
    print("\n  [8] SignalR Hub Connectivity")
    try:
        import requests
        # SignalR negotiate requires POST, not GET
        hub_check = requests.post("https://rtc.topstepx.com/hubs/market/negotiate?negotiateVersion=1",
                                headers={"Authorization": f"Bearer {client.access_token}"},
                                timeout=5)
        if hub_check.status_code == 200:
            data = hub_check.json() if hub_check.text else {}
            conn_id = data.get('connectionId', data.get('connectionToken', 'unknown'))
            print(f"      {PASS}  SignalR negotiate OK (connId: {conn_id[:12]}...)")
        else:
            print(f"      {WARN}  SignalR negotiate returned {hub_check.status_code}")
    except Exception as e:
        print(f"      {WARN}  SignalR negotiate probe failed: {e}")

    # ── Summary ──
    total = len(results)
    passed = sum(1 for _, _, ok in results if ok)
    failed = total - passed

    print(f"\n{'='*65}")
    print(f"  Results: {passed}/{total} checks passed", end="")
    if failed > 0:
        print(f" ({failed} failed)")
    else:
        print(" — ALL CLEAR ✓")
    print(f"{'='*65}\n")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
