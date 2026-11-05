#!/usr/bin/env python3
"""
List all Topstep / ProjectX accounts associated with the logged-in credentials.
"""
import os
import sys
from dotenv import load_dotenv

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.projectx_client import ProjectXClient

def main():
    load_dotenv()
    client = ProjectXClient()
    
    print("\nAuthenticating with TopstepX...")
    if not client.login():
        print("✗ Authentication failed. Please check your credentials in .env.")
        sys.exit(1)
        
    print("\nFetching accounts...")
    res = client.get_accounts()
    accounts = res.get("accounts", []) if isinstance(res, dict) else []
    
    if not accounts:
        print("✗ No active accounts found.")
        return
        
    active_id = str(os.getenv("PROJECTX_ACCOUNT_ID", ""))
    
    print("\n" + "=" * 65)
    print("  Topstep Accounts")
    print("=" * 65)
    
    for acc in accounts:
        acc_id = str(acc.get("id"))
        name = acc.get("name", "Unknown")
        balance = acc.get("balance", 0.0)
        can_trade = acc.get("canTrade", True)
        is_selected = " [ACTIVE in .env]" if acc_id == active_id else ""
        
        print(f"  ID: {acc_id:<10} | Name: {name:<25} | Balance: ${balance:>10,.2f} | Can Trade: {can_trade}{is_selected}")
        
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
