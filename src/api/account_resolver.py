from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from src.api.projectx_client import ProjectXClient


@dataclass
class AccountInfo:
    id: int
    name: str
    can_trade: bool
    is_practice: bool
    balance: float
    raw: Dict[str, Any]

    def __str__(self) -> str:
        practice_tag = " [Practice]" if self.is_practice else " [Funded/Live]"
        trade_tag = "CanTrade" if self.can_trade else "LOCKED"
        return f"Account {self.name} (ID: {self.id}){practice_tag} | Balance: ${self.balance:,.2f} | {trade_tag}"


class AccountResolver:
    """Authoritative Account Resolver for TopstepX / ProjectX API.

    Resolves, validates, and assigns account credentials:
    1. Supports resolution by integer ID, string ID, or account name.
    2. Enforces environment mode safety (simulation vs live).
    3. Verifies `canTrade` permission before allowing execution.
    4. Auto-discovers eligible accounts when no ID is explicitly configured.
    """

    def __init__(self, client: ProjectXClient):
        self.client = client
        self.logger = logging.getLogger(__name__)

    def fetch_all_accounts(self) -> List[AccountInfo]:
        """Queries POST /api/Account/search via client and parses into AccountInfo objects."""
        raw_res = self.client.get_accounts()
        accounts_raw = raw_res.get("accounts", []) if isinstance(raw_res, dict) else []
        
        parsed: List[AccountInfo] = []
        for acc in accounts_raw:
            try:
                acc_id = int(acc.get("id", 0))
                if not acc_id:
                    continue
                name = str(acc.get("name") or "Unknown")
                raw_can_trade = acc.get("canTrade", True)
                if isinstance(raw_can_trade, str):
                    can_trade = raw_can_trade.strip().lower() not in ("false", "0", "no", "off", "")
                else:
                    can_trade = bool(raw_can_trade)
                is_practice = "practice" in name.lower()
                balance = float(acc.get("balance", 0.0) or 0.0)
                parsed.append(
                    AccountInfo(
                        id=acc_id,
                        name=name,
                        can_trade=can_trade,
                        is_practice=is_practice,
                        balance=balance,
                        raw=acc,
                    )
                )
            except Exception as e:
                self.logger.warning(f"Error parsing account entry {acc}: {e}")
        return parsed

    def resolve_account(
        self,
        account_identifier: Optional[Union[int, str]] = None,
        mode: Optional[str] = None,
    ) -> Optional[AccountInfo]:
        """Resolves the active trading account against TopstepX broker API.

        Args:
            account_identifier: Optional numeric ID or account name string.
            mode: Optional trading mode ('simulation' or 'live'). Defaults to env MODE.

        Returns:
            Resolved AccountInfo or None if no valid account could be resolved.
        """
        if mode is None:
            mode = os.getenv("MODE", "simulation").strip().lower()
        else:
            mode = mode.strip().lower()

        if account_identifier is None:
            account_identifier = os.getenv("PROJECTX_ACCOUNT_ID")

        # 1. Fetch available accounts from broker
        accounts = self.fetch_all_accounts()
        if not accounts:
            self.logger.error("No active accounts found from TopstepX /Account/search.")
            return None

        resolved: Optional[AccountInfo] = None

        # 2. If an identifier was provided, match by ID or Name
        if account_identifier is not None and str(account_identifier).strip():
            ident_str = str(account_identifier).strip()
            
            # Numeric ID match
            if ident_str.isdigit():
                target_id = int(ident_str)
                for acc in accounts:
                    if acc.id == target_id:
                        resolved = acc
                        break
            
            # Name match (exact or substring), preferring tradeable accounts first
            if not resolved:
                ident_lower = ident_str.lower()
                ordered_accounts = sorted(accounts, key=lambda a: (not a.can_trade, a.id))
                for acc in ordered_accounts:
                    if acc.name.lower() == ident_lower:
                        resolved = acc
                        break
                if not resolved:
                    for acc in ordered_accounts:
                        if ident_lower in acc.name.lower():
                            resolved = acc
                            break

        # 3. If an explicit account_identifier was provided but not found:
        # In live mode, abort immediately to prevent trading on an unintended account.
        if account_identifier and not resolved:
            if mode == "live":
                self.logger.critical(
                    f"CRITICAL: Specified account '{account_identifier}' was not found among active broker accounts. "
                    "Halting resolution to prevent trading on an unintended account in live mode."
                )
                return None
            self.logger.warning(
                f"Specified account '{account_identifier}' not found among broker accounts. "
                f"Falling back to auto-discovery for mode='{mode}'..."
            )

        # 4. If no identifier was configured (or simulation fallback), auto-discover best candidate matching mode
        if not resolved:
            self.logger.info(
                f"No account specified. Auto-discovering candidate for mode='{mode}'..."
            )
            # Filter tradeable accounts first
            tradeable = [a for a in accounts if a.can_trade]
            candidates = tradeable if tradeable else accounts

            if mode == "live":
                # In live mode, prioritize non-practice accounts
                live_candidates = [a for a in candidates if not a.is_practice]
                resolved = live_candidates[0] if live_candidates else candidates[0]
            else:
                # In simulation mode, prioritize practice accounts
                sim_candidates = [a for a in candidates if a.is_practice]
                resolved = sim_candidates[0] if sim_candidates else candidates[0]

        if not resolved:
            self.logger.error(f"Failed to resolve any usable account for mode '{mode}'.")
            return None

        # 4. Mode safety validation
        if mode == "live" and resolved.is_practice:
            self.logger.warning(
                f"⚠️ CAUTION: Using PRACTICE account '{resolved.name}' ({resolved.id}) while in LIVE mode."
            )
        elif mode != "live" and not resolved.is_practice:
            self.logger.warning(
                f"⚠️ POTENTIAL MISCONFIGURATION: Using LIVE/FUNDED account '{resolved.name}' ({resolved.id}) "
                f"while in SIMULATION mode."
            )

        if not resolved.can_trade:
            self.logger.critical(
                f"CRITICAL: Account '{resolved.name}' ({resolved.id}) has canTrade=False. "
                "Orders placed on this account will be rejected by the broker."
            )
            if mode == "live":
                return None

        # 5. Synchronize with client
        self.client.account_id = resolved.id
        self.logger.info(f"Resolved Account: {resolved}")
        return resolved
