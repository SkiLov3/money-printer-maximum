from __future__ import annotations

import contextlib
import logging
import os
import time

import requests
from dotenv import load_dotenv
from signalrcore.hub_connection_builder import HubConnectionBuilder

# Monkey-patch signalrcore bugs:
# 1. BaseTransport.connection_alive attribute missing causing crash on deferred_reconnect
# 2. WebsocketTransport.on_socket_close unhandled exception when max reconnect attempts reached
try:
    from signalrcore.transport.base_transport import BaseTransport, TransportState
    BaseTransport.connection_alive = False  # type: ignore[attr-defined]

    from signalrcore.transport.websockets.websocket_transport import WebsocketTransport

    def _safe_on_socket_close(self):
        if not self.manually_closing and self.reconnection_handler is not None and not self.is_reconnecting():
            try:
                self.handle_reconnect()
                if getattr(self, "state", None) == TransportState.connected:
                    return
            except Exception as ex:
                self.logger.warning(f"SignalR reconnect attempt failed/exhausted: {ex}")
        try:
            self._set_state(TransportState.disconnected)
        except Exception:
            self.state = TransportState.disconnected

    WebsocketTransport.on_socket_close = _safe_on_socket_close
except Exception as _patch_err:
    logging.getLogger(__name__).warning(f"Could not patch signalrcore: {_patch_err}")

load_dotenv()


# ProjectX / TopstepX broker enum codes observed on live user-hub events.
# Sources: supervisor_agent._sync_orders_ledger comment block, and
# ProjectXClient.place_order order_type mapping comment.
PROJECTX_STATUS_CODES = {
    0: "none",
    1: "working",
    2: "filled",
    3: "cancelled",
    4: "expired",
    5: "rejected",
    6: "pending_new",
    7: "partially_filled",
    8: "pending_bracket",
}

PROJECTX_ORDER_TYPE_CODES = {
    1: "limit",
    2: "market",
    3: "stoplimit",
    4: "stop",
    5: "trailing_stop",
    6: "join_bid",
    7: "join_ask",
}

PROJECTX_POSITION_TYPE_CODES = {
    0: "flat",
    1: "long",
    2: "short",
}


def _normalize_enum(raw, mapping):
    """Coerce a broker enum value (int | float | str | None) to a lowercase
    string. Unknown ints are returned as ``code_<n>`` so downstream equality
    checks can be made defensively without crashing on `.lower()`.
    """
    if raw is None:
        return ""
    if isinstance(raw, bool):
        # bool is a subclass of int — guard first to avoid 1/0 being mapped.
        return ""
    if isinstance(raw, (int, float)):
        code = int(raw)
        return mapping.get(code, f"code_{code}")
    return str(raw).strip().lower()


def normalize_status(raw) -> str:
    return _normalize_enum(raw, PROJECTX_STATUS_CODES)


def normalize_order_type(raw) -> str:
    return _normalize_enum(raw, PROJECTX_ORDER_TYPE_CODES)


def normalize_position_type(raw) -> str:
    return _normalize_enum(raw, PROJECTX_POSITION_TYPE_CODES)


class ProjectXClient:
    def __init__(self):
        self.api_key = os.getenv("PROJECTX_API_KEY")
        self.username = os.getenv("TOPSTEPX_USERNAME") or os.getenv("PROJECTX_USERNAME")
        acc_id = os.getenv("PROJECTX_ACCOUNT_ID")
        self.account_id = int(acc_id) if acc_id and acc_id.isdigit() else None
        
        # Base URLs for ProjectX
        self.base_url = "https://api.topstepx.com/api"
        self.hub_url = "wss://rtc.topstepx.com/hubs/market"
        # User hub broadcasts GatewayUserAccount/Order/Position/Trade events.
        self.user_hub_url = "wss://rtc.topstepx.com/hubs/user"
        self.logger = logging.getLogger(__name__)
        
        # Mute SignalR spam
        logging.getLogger("signalrcore").setLevel(logging.ERROR)
        
        self.access_token = None
        self.token_acquired_at = 0  # epoch timestamp
        self.token_ttl = 3000       # re-auth after 50 mins (safe margin)
        self.is_connected = False
        self.on_message_callback = None
        self._on_order_update = None  # FIX #3: Always initialize before setup_signalr is called
        self._on_reconnect_ext = None
        self.connection = None
        self.active_subscriptions = set() # Track for re-subscription on reconnect
        self.last_tick_at = 0             # Heartbeat monitor

        # User-hub state (real-time fills/positions/account events)
        self.user_connection = None
        self.user_is_connected = False
        self._on_user_event = None
        self.last_user_event_at = 0
        
        # Contract metadata cache: base_symbol -> {name, tickSize, tickValue, id}
        self.contract_cache = {}
        self.symbol_to_id_cache = {}  # Resolves 'NQM6' -> 'CON.F.US.ENQ.M26'
        self.contract_id_to_symbol = {}  # Reverse: 'CON.F.US.MES.Z26' -> 'MESZ6'
        self._last_contract_volume = {}  # Tracks cumulative session volume per contractId
        
        # DEVSEC FIX: Atomic Lock for thread-safe authentication
        import threading
        self.auth_lock = threading.Lock()
        self._reconnect_lock = threading.Lock()
        self._symbol_cache_lock = threading.RLock()
        self._last_reconnect_attempt = 0.0

    @staticmethod
    def _build_refreshed_ws_url(base_url: str, access_token: str, existing_url: str | None) -> str:
        from urllib.parse import parse_qs, quote, urlparse
        url = f"{base_url}?access_token={quote(str(access_token), safe='')}"
        if existing_url:
            try:
                qs = parse_qs(urlparse(str(existing_url)).query)
                conn_ids = qs.get("id")
                if conn_ids and conn_ids[0]:
                    url = f"{url}&id={quote(str(conn_ids[0]), safe='')}"
            except Exception as err:
                logging.getLogger(__name__).debug(f"Failed to extract WS connection id: {err}")
        return url

    def _update_connection_urls(self):
        """Synchronize the fresh JWT into active SignalR hub connection URLs and headers."""
        if not self.access_token:
            return
        auth_header = f"Bearer {self.access_token}"

        if self.connection:
            existing_mkt = getattr(getattr(self.connection, "transport", None), "url", None) or getattr(self.connection, "url", None)
            market_url = self._build_refreshed_ws_url(self.hub_url, self.access_token, existing_mkt)
            self.connection.url = market_url
            if hasattr(self.connection, "headers") and isinstance(self.connection.headers, dict):
                self.connection.headers["Authorization"] = auth_header
            if hasattr(self.connection, "transport") and self.connection.transport:
                self.connection.transport.url = market_url
                if hasattr(self.connection.transport, "headers") and isinstance(self.connection.transport.headers, dict):
                    self.connection.transport.headers["Authorization"] = auth_header

        if self.user_connection:
            existing_usr = getattr(getattr(self.user_connection, "transport", None), "url", None) or getattr(self.user_connection, "url", None)
            user_url = self._build_refreshed_ws_url(self.user_hub_url, self.access_token, existing_usr)
            self.user_connection.url = user_url
            if hasattr(self.user_connection, "headers") and isinstance(self.user_connection.headers, dict):
                self.user_connection.headers["Authorization"] = auth_header
            if hasattr(self.user_connection, "transport") and self.user_connection.transport:
                self.user_connection.transport.url = user_url
                if hasattr(self.user_connection.transport, "headers") and isinstance(self.user_connection.transport.headers, dict):
                    self.user_connection.transport.headers["Authorization"] = auth_header

    def login(self):
        """Exchanges API Key and Username for a JWT Bearer Token."""
        if not self.api_key or not self.username:
            self.logger.error("Missing API Key or Username in .env")
            return False
            
        url = f"{self.base_url}/Auth/loginKey"
        payload = {
            "userName": self.username,
            "apiKey": self.api_key
        }
        headers = {
            "Content-Type": "application/json",
            "accept": "text/plain"
        }
        
        try:
            self.logger.info("Attempting login to ProjectX...")
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success') and data.get('token'):
                    self.access_token = data.get('token')
                    self.token_acquired_at = time.time()
                    self.logger.info("Successfully authenticated with ProjectX")
                    self._update_connection_urls()
                    return True
                else:
                    error_msg = data.get('errorMessage') or "No error message provided"
                    self.logger.error(f"Auth failed: {error_msg}")
            else:
                self.logger.error(f"Login failed: status {response.status_code}")
        except Exception as e:
            self.logger.error(f"Login request failed: {e}")
        
        return False

    def _ensure_token(self):
        """FIX H4: Refresh JWT if expired or about to expire. Thread-safe."""
        with self.auth_lock:
            if not self.access_token:
                return self.login()
            
            elapsed = time.time() - self.token_acquired_at
            if elapsed >= self.token_ttl:
                self.logger.info("JWT token approaching expiry, re-authenticating...")
                return self.login()
            return True

    def _token_factory(self):
        """Used by SignalR to get a valid token, even during reconnection."""
        self._ensure_token()
        return self.access_token

    def _get_headers(self):
        headers = {
            "Content-Type": "application/json",
            "accept": "application/json"
        }
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        elif self.api_key:
            headers["X-API-KEY"] = self.api_key
        return headers

    def _request_with_refresh(self, method, url, allow_retry_on_timeout: bool = True, **kwargs):
        """Makes an API request, retrying once on 401 with a fresh token,
        handling HTTP 429 rate limits with exponential backoff, and retrying
        transient HTTP 5xx server errors. Thread-safe.
        """
        # FIX H6: Ensure all blocking HTTP requests have a timeout so they don't freeze the asyncio loop forever
        kwargs.setdefault('timeout', 15)
        self._ensure_token()
        
        # Use headers snapshot
        with self.auth_lock:
            kwargs['headers'] = self._get_headers()
        
        max_retries = 3
        response = None
        for attempt in range(max_retries):
            try:
                try:
                    response = method(url, **kwargs)
                except requests.exceptions.RequestException as req_err:
                    if not allow_retry_on_timeout:
                        self.logger.error(
                            f"Non-idempotent request to {url} failed with {req_err}; "
                            "retry suppressed to prevent duplicate order executions."
                        )
                        raise
                    if attempt < max_retries - 1:
                        self.logger.warning(f"Request to {url} failed ({req_err}), retrying (attempt {attempt + 1})...")
                        time.sleep(1.0 * (attempt + 1))
                        continue
                    raise
                
                # Handle 401 Unauthorized (atomic JWT refresh with double-checked lock)
                if response.status_code == 401 and attempt < max_retries - 1:
                    self.logger.warning("Got 401 Unauthorized, triggering atomic refresh...")
                    token_before = self.access_token
                    with self.auth_lock:
                        if self.access_token != token_before or (time.time() - self.token_acquired_at) < 5.0:
                            kwargs['headers'] = self._get_headers()
                            continue
                        if self.login():
                            kwargs['headers'] = self._get_headers()
                            continue
                
                # Handle 429 Too Many Requests (Rate limit backoff)
                if response.status_code == 429 and attempt < max_retries - 1:
                    retry_after = response.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after and retry_after.isdigit() else (2.0 * (attempt + 1))
                    self.logger.warning(f"Got 429 Rate Limit from {url}, backing off for {delay:.1f}s...")
                    time.sleep(delay)
                    continue
                
                # Handle 5xx Transient Server Errors
                if response.status_code in (500, 502, 503, 504) and attempt < max_retries - 1:
                    if not allow_retry_on_timeout:
                        self.logger.error(
                            f"Non-idempotent request to {url} returned HTTP {response.status_code}; "
                            "retry suppressed to prevent duplicate order executions."
                        )
                        return response
                    delay = 1.0 * (attempt + 1)
                    self.logger.warning(f"Got HTTP {response.status_code} from {url}, retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    continue
                
                return response
            except Exception as e:
                if not allow_retry_on_timeout or attempt >= max_retries - 1:
                    self.logger.error(f"Request to {url} failed: {e}")
                    raise
        if response is not None:
            return response
        raise RuntimeError(f"Request to {url} exhausted {max_retries} retries without receiving a response.")

    def get_accounts(self):
        """Uses the POST /Account/search endpoint with spec-compliant parameters."""
        url = f"{self.base_url}/Account/search"
        try:
            payload = {"onlyActiveAccounts": True, "active": True}
            response = self._request_with_refresh(requests.post, url, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to fetch accounts: {e}")
            return {}

    def get_account_status(self):
        """Fetches status by searching accounts and filtering for the active account_id."""
        if not self.account_id:
            return None
        try:
            data = self.get_accounts()
            accounts = data.get('accounts', [])
            for acc in accounts:
                if acc.get('id') == self.account_id:
                    acc_name = acc.get('name', 'Unknown')
                    # Log a prominent warning if the account name doesn't contain 'Practice' but we are in simulation
                    mode = os.getenv("MODE", "simulation").strip().lower()
                    if mode != 'live' and 'practice' not in acc_name.lower():
                        self.logger.warning(f"⚠️ POTENTIAL MISCONFIGURATION: Using account '{acc_name}' ({self.account_id}) while in SIMULATION mode.")
                    else:
                        self.logger.info(f"Verified Account: {acc_name} ({self.account_id})")
                    return acc
            self.logger.warning(f"Account ID '{self.account_id}' not found in search results.")
            return None
        except Exception as e:
            self.logger.error(f"Failed to fetch account status: {e}")
            return None

    def search_open_positions(self):
        """Hits POST /api/Position/searchOpen — the authoritative ProjectX endpoint for open positions.

        Returns:
            list: Active positions (can be empty list [] if account is flat).
            None: If the request fails, times out, or error response is received.
        """
        if not self.account_id:
            return None

        url = f"{self.base_url}/Position/searchOpen"
        payload = {"accountId": self.account_id}

        try:
            response = self._request_with_refresh(requests.post, url, json=payload)
            response.raise_for_status()
            data = response.json()
            if not data.get("success", True):
                self.logger.error(f"/Position/searchOpen returned success=False: {data}")
                return None
            return data.get("positions", [])
        except Exception as e:
            self.logger.error(f"Failed to fetch open positions from /Position/searchOpen: {e}")
            return None

    def get_positions(self):
        """Fetches all active positions for the current account.

        Account status (POST /Account/search) does NOT include positions — it returns
        balance/PnL/MLL only. Positions live behind POST /Position/searchOpen.

        Returns:
            list: Active positions on success ([] if flat).
            None: If the broker request failed or timed out.
        """
        positions = self.search_open_positions()
        if positions is not None and positions:
            self.logger.info(f"Retrieved {len(positions)} open positions for account {self.account_id}")
        return positions

    def get_orders(self, active=True):
        """Fetches active or historical orders for the current account.

        Returns:
            list: Orders on success ([] if none found).
            None: If the broker request failed or timed out.
        """
        if not self.account_id:
            return None

        # FIX #H15: /Order/search requires a startTimestamp even for active orders.
        # Without it, the gateway returns HTTP 400.
        from datetime import datetime as _dt
        from datetime import timedelta
        from datetime import timezone as _tz
        start_ts = (_dt.now(_tz.utc) - timedelta(hours=24)).isoformat()

        url = f"{self.base_url}/Order/search"
        payload = {
            "accountId": self.account_id,
            "active": active,
            "startTimestamp": start_ts
        }

        try:
            response = self._request_with_refresh(requests.post, url, json=payload)
            response.raise_for_status()
            data = response.json()
            if not data.get("success", True):
                self.logger.error(f"/Order/search returned success=False: {data}")
                return None
            return data.get('orders', [])
        except Exception as e:
            self.logger.error(f"Failed to fetch orders: {e}")
            return None

    def has_working_stop_order(self, symbol: str, position_side: str, min_qty: int = 1) -> bool | None:
        """Checks if there is an active working stop-loss order protecting an open position.

        Args:
            symbol: Contract symbol (e.g. 'MCLZ6')
            position_side: 'Buy' (long position) or 'Sell' (short position)
            min_qty: Minimum open contract quantity that working stop orders must cover.

        Returns:
            True: If active opposite-side stop order(s) exist covering the position.
            False: If active orders were fetched successfully and coverage is missing/insufficient.
            None: If active orders could not be fetched due to an API error/timeout.
        """
        with self._symbol_cache_lock:
            contract_id = self.symbol_to_id_cache.get(symbol)
            if not contract_id:
                meta = self.contract_cache.get(symbol)
                if meta and meta.get("id"):
                    contract_id = meta.get("id")
                    self.symbol_to_id_cache[symbol] = contract_id
            sym_aliases = dict(self.symbol_to_id_cache)

        active_orders = self.get_orders(active=True)
        if active_orders is None:
            self.logger.warning(f"Could not verify working stop orders for {symbol}: get_orders returned None.")
            return None

        target_cid = str(contract_id) if contract_id else ""
        # Expected stop order side: opposite of position side.
        # Long position (Buy) -> Stop order to exit is Sell (side 1).
        # Short position (Sell) -> Stop order to exit is Buy (side 0).
        pos_side_norm = str(position_side).strip().lower()
        expected_stop_side_int = 1 if pos_side_norm in ("buy", "long", "0") else 0

        ACTIVE_STATUSES = {
            "working", "open", "pending_new", "pending_bracket",
            "partially_filled", "code_1", "code_6", "code_7", "code_8",
        }
        STOP_TYPES = {
            "stop",
            "trailing_stop",
            "stoploss",
            "stop_loss",
            "stop_limit",
            "stoplimit",
            "code_3",
            "code_4",
            "code_5",
        }

        covered_qty = 0
        has_unsized_stop = False
        for o in active_orders:
            oid = o.get("id")
            ocid = str(o.get("contractId") or "")
            osym = str(o.get("symbol") or o.get("contractDisplayName") or "")
            ostatus = normalize_status(o.get("status"))
            otype = normalize_order_type(o.get("type"))
            raw_side = o.get("side")
            try:
                oside_int = int(raw_side) if raw_side is not None else -1
            except (ValueError, TypeError):
                side_s = str(raw_side).strip().lower()
                if side_s in ("sell", "short", "1"):
                    oside_int = 1
                elif side_s in ("buy", "long", "0"):
                    oside_int = 0
                else:
                    oside_int = -1

            if not oid:
                continue

            matches_contract = bool(target_cid and ocid == target_cid)
            matches_symbol = bool(osym and (osym == symbol or (target_cid and sym_aliases.get(osym) == target_cid)))

            if (
                (matches_contract or matches_symbol)
                and ostatus in ACTIVE_STATUSES
                and otype in STOP_TYPES
                and oside_int == expected_stop_side_int
            ):
                rem_raw = o.get("remainingQty")
                if rem_raw is None:
                    rem_raw = o.get("leavesQty")
                if rem_raw is None:
                    rem_raw = o.get("size")
                if rem_raw is None:
                    has_unsized_stop = True
                else:
                    try:
                        covered_qty += max(0, int(abs(float(rem_raw))))
                    except (ValueError, TypeError):
                        has_unsized_stop = True

        if covered_qty >= max(1, int(min_qty)):
            return True
        return bool(has_unsized_stop and covered_qty == 0)

    def cancel_order(self, order_id: int) -> bool:
        """Cancels a single open order by ID via POST /Order/cancel.
        Returns True on success, False on failure.
        """
        if not self.account_id:
            return False
        url = f"{self.base_url}/Order/cancel"
        payload = {"accountId": self.account_id, "orderId": int(order_id)}
        try:
            resp = self._request_with_refresh(requests.post, url, json=payload)
            resp.raise_for_status()
            result = resp.json()
            if result.get("success"):
                self.logger.info(f"Cancelled order {order_id}")
                return True
            self.logger.warning(f"Cancel order {order_id} returned success=False: {result.get('errorMessage')}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def cancel_open_orders_for_symbol(self, symbol: str) -> int:
        """Fetches all active orders for the account and cancels any that match
        the given symbol (by contractId lookup). Returns number of orders cancelled.
        Called before flatten_position() to sweep orphaned SL/TP bracket legs.
        """
        contract_id = self.symbol_to_id_cache.get(symbol)
        if not contract_id:
            # Fallback to contract_cache lookup if base_symbol was passed
            meta = self.contract_cache.get(symbol)
            if meta and meta.get("id"):
                contract_id = meta.get("id")
                self.symbol_to_id_cache[symbol] = contract_id

        active_orders = self.get_orders(active=True)
        if not active_orders:
            if active_orders is None:
                self.logger.warning(f"Could not sweep open orders for {symbol}: failed to fetch active orders (API error).")
            return 0
        cancelled = 0
        target_cid = str(contract_id) if contract_id else ""

        for o in active_orders:
            oid = o.get("id")
            ocid = str(o.get("contractId") or "")
            osym = str(o.get("symbol") or o.get("contractDisplayName") or "")
            ostatus = normalize_status(o.get("status"))
            oside = "Sell" if o.get("side") == 1 else "Buy"

            if not oid:
                continue

            # Must match target contractId or symbol
            matches_contract = bool(target_cid and ocid == target_cid)
            matches_symbol = bool(osym and (osym == symbol or (target_cid and self.symbol_to_id_cache.get(osym) == target_cid)))

            if matches_contract or matches_symbol:
                if ostatus not in ("working", "open", "pending_new", "pending_bracket", "partially_filled", "code_1", "code_6", "code_7", "code_8"):
                    continue
                self.logger.info(f"Attempting to sweep orphaned {oside} order {oid} (status: {ostatus}) for {symbol}...")
                if self.cancel_order(oid):
                    cancelled += 1
        if cancelled:
            self.logger.info(f"Swept {cancelled} orphaned bracket order(s) for {symbol}")
        return cancelled

    def close_contract(self, symbol: str) -> dict:
        """Server-side position close via POST /api/Position/closeContract.
        Atomically closes the entire position on the broker side.
        """
        if not self.account_id:
            return {"status": "error", "message": "No account_id configured."}

        contract_id = self.symbol_to_id_cache.get(symbol)
        if not contract_id:
            meta = self.contract_cache.get(symbol)
            if meta and meta.get("id"):
                contract_id = meta.get("id")

        if not contract_id:
            contract_id = symbol

        url = f"{self.base_url}/Position/closeContract"
        payload = {
            "accountId": self.account_id,
            "contractId": contract_id,
        }
        self.logger.info(f"[CLOSE CONTRACT] account={self.account_id} contractId={contract_id}")
        try:
            response = self._request_with_refresh(requests.post, url, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Position closeContract failed for {symbol}: {e}")
            return {"status": "error", "message": str(e)}

    def get_order_history(self, limit=50, lookback_hours=24):
        """Fetches recent order history to check for filled bracket targets.

        ProjectX `/Order/search` requires a `startTimestamp` (ISO 8601). The previous
        payload sent `active: False` and `limit`, which the gateway rejects with HTTP 400.
        We default to a 24h lookback window which is more than enough for the supervisor's
        startup ledger reconstruction (today's fills only).
        """
        if not self.account_id:
            return []

        from datetime import datetime as _dt
        from datetime import timedelta
        from datetime import timezone as _tz
        start_ts = (_dt.now(_tz.utc) - timedelta(hours=int(lookback_hours))).isoformat()

        url = f"{self.base_url}/Order/search"
        payload = {
            "accountId": self.account_id,
            "startTimestamp": start_ts,
        }

        try:
            response = self._request_with_refresh(requests.post, url, json=payload)
            response.raise_for_status()
            orders = response.json().get('orders', [])
            # Caller may want a soft cap; honor `limit` client-side since the API ignores it.
            return orders[:limit] if limit and limit > 0 else orders
        except Exception as e:
            self.logger.error(f"Failed to fetch order history: {e}")
            return []

    def place_order(
        self,
        symbol,
        quantity,
        side,
        order_type="Market",
        limit_price=None,
        stop_price=None,
        sl_ticks=None,
        tp_ticks=None,
    ):
        # DOUBLE-LOCK SAFETY: Hard-stop if not explicitly in live mode
        mode = os.getenv("MODE", "simulation").strip().lower()
        if mode != "live":
            self.logger.critical(f"HARD STOP: Attempted to place LIVE order for {symbol} while in '{mode}' mode.")
            return {"status": "error", "message": f"Order blocked: Bot is in {mode} mode, not LIVE."}

        if not self.account_id:
            self.logger.critical(f"HARD STOP: No account_id configured for placing order on {symbol}.")
            return {"status": "error", "message": "No account_id configured."}

        url = f"{self.base_url}/Order/place"
        
        # Translate internal side to Topstep API Enums: 0=Buy, 1=Sell
        if isinstance(side, int):
            ts_side = 1 if side == 1 else 0
        else:
            side_norm = str(side).strip().lower()
            ts_side = 1 if side_norm in ("sell", "short", "ask") else 0
        
        # FIX: Mapping TopstepX Order Types: 1=Limit, 2=Market, 4=Stop, 5=TrailingStop
        if isinstance(order_type, int):
            ts_type = order_type
        else:
            ot_lower = str(order_type).strip().lower()
            if ot_lower in ("market", "2"):
                ts_type = 2
            elif ot_lower in ("limit", "1"):
                ts_type = 1
            elif ot_lower in ("stop", "4"):
                ts_type = 4
            elif ot_lower in ("trailing_stop", "trailingstop", "5"):
                ts_type = 5
            else:
                ts_type = 2  # Fallback to Market if unknown
        
        contract_id = self.symbol_to_id_cache.get(symbol)
        if not contract_id:
            meta = self.contract_cache.get(symbol)
            if meta and meta.get("id"):
                contract_id = meta.get("id")
        if not contract_id:
            contract_id = symbol
        
        payload = {
            "accountId": self.account_id,
            "contractId": contract_id,
            "size": int(quantity),
            "side": ts_side,
            "type": ts_type,
        }

        if limit_price is not None and ts_type == 1:
            payload["limitPrice"] = float(limit_price)
        if stop_price is not None and ts_type == 4:
            payload["stopPrice"] = float(stop_price)

        # Topstep bracket logic: ticks are signed offsets from entry price.
        # SL: negative for Buy (below entry), positive for Sell (above entry)
        # TP: positive for Buy (above entry), negative for Sell (below entry)
        if sl_ticks:
            sl_val = int(round(float(sl_ticks)))
            sl_val = -abs(sl_val) if ts_side == 0 else abs(sl_val)
            payload["stopLossBracket"] = {"ticks": sl_val, "type": 4}  # 4 = Stop
        if tp_ticks:
            tp_val = int(round(float(tp_ticks)))
            tp_val = abs(tp_val) if ts_side == 0 else -abs(tp_val)
            payload["takeProfitBracket"] = {"ticks": tp_val, "type": 1}  # 1 = Limit

        self.logger.info(f"[ORDER PAYLOAD] {symbol} {side} {quantity}x -> {payload}")
        try:
            response = self._request_with_refresh(requests.post, url, allow_retry_on_timeout=False, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Order placement failed: {e}")
            return {"status": "error", "message": str(e)}

    def setup_signalr(self, on_market_data=None, on_order_update=None, on_reconnect=None):
        if not self._ensure_token():
            self.logger.error("SignalR setup aborted: Authentication failed.")
            return

        if on_market_data is not None:
            self.on_message_callback = on_market_data
        if on_order_update is not None:
            self._on_order_update = on_order_update
        if on_reconnect is not None:
            self._on_reconnect_ext = on_reconnect
        try:
            # TopstepX requires the token explicitly in the WSS query string
            ws_url = f"{self.hub_url}?access_token={self.access_token}"
            self.connection = HubConnectionBuilder()\
                .with_url(ws_url, {
                    "access_token_factory": self._token_factory,
                })\
                .with_automatic_reconnect({
                    "type": "raw",
                    "keep_alive_interval": 10,
                    "reconnect_interval": 5,
                    "max_attempts": 3
                })\
                .build()

            self.connection.on_open(self.on_open)
            self.connection.on_reconnect(self.on_reconnect)
            self.connection.on_close(self.on_close)
            def _on_market_error(error):
                self.logger.error(f"SignalR Error: {error}")
                if "401" in str(error) or "Unauthorized" in str(error):
                    self.logger.warning("SignalR 401 detected in error handler, triggering token refresh")
                    self.is_connected = False
                    with self.auth_lock:
                        self.login()
            self.connection.on_error(_on_market_error)
            self.connection.on("GatewayQuote", self._handle_message)
            
            def on_logout(msg):
                self.logger.error("ALARM: Received GatewayLogout from server. This usually means you logged in from another device (Browser or Mobile App). TopstepX only allows one connection per account!")
                self.is_connected = False
            self.connection.on("GatewayLogout", on_logout)
            
            # NOTE: Order/Position updates do NOT come through the market hub. They live on
            # the user hub (/hubs/user) — see setup_user_hub_signalr(). The supervisor wires
            # both setups in run().
        except Exception as e:
            self.logger.error(f"Failed to build SignalR connection: {e}")
            self.connection = None

    def lookup_symbol_for_contract_id(self, contract_id: str) -> str:
        """Reverse resolves contract_id to the most specific matching futures symbol (e.g. 'MESZ6' or 'MESZ26'), avoiding bare product roots like 'MES'."""
        if not contract_id:
            return ""
        cid_str = str(contract_id)
        cache_lock = getattr(self, "_symbol_cache_lock", None)
        with cache_lock if cache_lock is not None else contextlib.nullcontext():
            if hasattr(self, "contract_id_to_symbol"):
                if contract_id in self.contract_id_to_symbol:
                    return self.contract_id_to_symbol[contract_id]
                if cid_str in self.contract_id_to_symbol:
                    return self.contract_id_to_symbol[cid_str]

            candidates = [k for k, v in list(self.symbol_to_id_cache.items()) if str(v) == cid_str and not k.startswith("/")]
            if not candidates:
                # Check if contract_id itself is a Topstep contract specifier: e.g. CON.F.US.MES.Z26
                import re
                m = re.search(r'\b([A-Z0-9]+)\.([FGHJKMNQUVXZ]\d{1,2})\b', cid_str)
                if m:
                    root, my = m.group(1), m.group(2)
                    if root == "ENQ":
                        root = "NQ"
                    elif root == "EP":
                        root = "ES"
                    elif root.endswith("E") and len(root) > 3:
                        root = root[:-1]
                    if len(my) == 3:  # e.g. Z26 -> Z6
                        resolved = f"{root}{my[0]}{my[-1]}"
                    else:
                        resolved = f"{root}{my}"
                    if hasattr(self, "contract_id_to_symbol"):
                        self.contract_id_to_symbol[contract_id] = resolved
                        self.contract_id_to_symbol[cid_str] = resolved
                    return resolved
                return cid_str

            import re
            # Look for contract codes with month code and 1 or 2 digit year (e.g. MESZ6, MESZ26)
            specific = [c for c in candidates if re.search(r'[FGHJKMNQUVXZ]\d{1,2}$', c)]
            if specific:
                # Prefer 1-digit year shorthand if present (standard convention used across the bot)
                one_digit = [c for c in specific if re.search(r'[FGHJKMNQUVXZ]\d$', c)]
                best = one_digit[0] if one_digit else specific[0]
                if hasattr(self, "contract_id_to_symbol"):
                    self.contract_id_to_symbol[contract_id] = best
                    self.contract_id_to_symbol[cid_str] = best
                return best
            best = candidates[0]
            if hasattr(self, "contract_id_to_symbol"):
                self.contract_id_to_symbol[contract_id] = best
                self.contract_id_to_symbol[cid_str] = best
            return best

    def _handle_message(self, message):
        if self.on_message_callback:
            try:
                if not message or len(message) < 2:
                    return
                # TopstepX GatewayQuote array: [contract_id, { dictionary }]
                contract_id = message[0]
                raw_data = message[1]
                if isinstance(raw_data, list):
                    raw_data = raw_data[-1] if raw_data else {}
                if not isinstance(raw_data, dict):
                    return
                
                # Reverse lookup contract_id to the configured active symbol name (e.g. 'MESZ6')
                active_symbol = self.lookup_symbol_for_contract_id(contract_id)
                
                # FALLBACK PRICE LOGIC: trade price -> mid(bid, ask) -> ask -> bid -> 0
                trade_price = raw_data.get('lastPrice')
                ask_price = raw_data.get('bestAsk')
                bid_price = raw_data.get('bestBid')
                
                price = trade_price
                if price is None:
                    try:
                        if ask_price is not None and bid_price is not None and float(ask_price) > 0 and float(bid_price) > 0:
                            price = (float(ask_price) + float(bid_price)) / 2.0
                        else:
                            price = ask_price if ask_price is not None else (bid_price or 0)
                    except (ValueError, TypeError):
                        price = ask_price or bid_price or 0
                try:
                    price = float(price or 0)
                except (ValueError, TypeError):
                    price = 0.0

                tick_ts = (
                    raw_data.get('timestamp')
                    or raw_data.get('time')
                    or raw_data.get('t')
                    or raw_data.get('timestampUtc')
                )

                # GatewayQuote 'volume' is cumulative session volume on TopstepX.
                # Convert cumulative session volume to per-tick volume delta.
                try:
                    raw_vol = float(raw_data.get('volume', 0) or 0)
                except (ValueError, TypeError):
                    raw_vol = 0.0

                cid_key = str(contract_id)
                is_trade = trade_price is not None
                if not hasattr(self, "_last_contract_volume"):
                    self._last_contract_volume = {}
                prev_vol = self._last_contract_volume.get(cid_key)
                if raw_vol > 0:
                    if prev_vol is not None:
                        tick_vol = (raw_vol - prev_vol) if raw_vol >= prev_vol else raw_vol
                    else:
                        # First tick seen for this contract: only use raw_vol directly if it's small (e.g. unit test / session start)
                        tick_vol = raw_vol if raw_vol <= 500 else 0.0
                    if is_trade:
                        self._last_contract_volume[cid_key] = raw_vol
                else:
                    tick_vol = 0.0
                
                # Normalize payload for DataAgent
                normalized = {
                    'symbol': active_symbol,
                    'contractId': cid_key,
                    'price': price,
                    'volume': tick_vol,
                    'is_trade': is_trade,
                    'timestamp': tick_ts,
                    'raw': raw_data
                }
                
                # Only pass it if we have any price to work with
                if price > 0:
                    self.last_tick_at = time.time() # Update heartbeat
                    self.on_message_callback(normalized)
                    
            except Exception as e:
                self.logger.error(f"Error parsing GatewayQuote: {e}")

    def _handle_order_update(self, message):
        if self._on_order_update:
            self._on_order_update(message)

    def on_open(self):
        self.logger.info("SignalR Connection Opened")
        self.is_connected = True
        if self.active_subscriptions:
            self._send_market_data_subscriptions()

    def on_close(self):
        self.logger.info("SignalR Connection Closed")
        self.is_connected = False

    def on_reconnect(self):
        self.logger.warning("SignalR Reconnected! Re-issuing active subscriptions...")
        self.is_connected = True
        if self.active_subscriptions:
            self._send_market_data_subscriptions()
        if hasattr(self, '_on_reconnect_ext') and self._on_reconnect_ext:
            self._on_reconnect_ext()

    def start_streaming(self, force=False):
        if not force and self.is_connected:
            self.logger.info("SignalR Market Data already streaming.")
            return

        if not self._ensure_token():
            self.logger.error("SignalR start aborted: Authentication failed.")
            return

        self.is_connected = False
        if self.connection:
            self.stop_streaming()

        # Re-build connection object to pick up the fresh access_token in the WSS query string
        self.setup_signalr(
            on_market_data=self.on_message_callback,
            on_order_update=self._on_order_update,
            on_reconnect=self._on_reconnect_ext
        )

        if self.connection:
            try:
                self.logger.info("Starting SignalR Market Data streaming...")
                self.connection.start()
            except Exception as e:
                self.logger.error(f"SignalR start failed: {e}")
        else:
            self.logger.warning("Streaming not started: No active SignalR connection. Check your credentials.")

    def stop_streaming(self):
        if self.connection:
            try:
                self.logger.info("Stopping SignalR Market Data streaming...")
                self.connection.stop()
            except Exception as e:
                self.logger.error(f"SignalR stop failed: {e}")
            finally:
                try:
                    transport = getattr(self.connection, "transport", None)
                    checker = getattr(transport, "connection_checker", None)
                    if checker and getattr(checker, "running", False):
                        checker.stop()
                except Exception:
                    pass
                self.is_connected = False

    # ------------------------------------------------------------------ User Hub
    # ProjectX/TopstepX broadcasts account/order/position/trade events on a SECOND
    # SignalR hub at /hubs/user. Subscriptions are per-account RPCs sent by the client
    # after connection: SubscribeAccounts (no args), SubscribeOrders/Positions/Trades
    # with the accountId as a positional arg. Events arrive as [accountId, payloadDict].

    USER_HUB_SUBSCRIPTIONS = (
        ("SubscribeAccounts", False),  # no args
        ("SubscribeOrders", True),     # accountId arg
        ("SubscribePositions", True),  # accountId arg
        ("SubscribeTrades", True),     # accountId arg
    )
    # Primary spec event names. The catch-all set below also probes alternate names that
    # different ProjectX gateway revisions have used, so we log whatever actually arrives.
    USER_HUB_EVENTS = ("GatewayUserAccount", "GatewayUserOrder", "GatewayUserPosition", "GatewayUserTrade")
    USER_HUB_EVENT_PROBES = (
        # Alternate / legacy event names seen across ProjectX gateway versions. Registering
        # them all means any inbound payload surfaces in the log so we can wire it up.
        "AccountUpdate", "OrderUpdate", "PositionUpdate", "TradeUpdate",
        "OnAccountUpdate", "OnOrderUpdate", "OnPositionUpdate", "OnTradeUpdate",
        "GatewayUserOrderUpdate", "GatewayUserPositionUpdate", "GatewayUserTradeUpdate",
        "Account", "Order", "Position", "Trade",
    )

    def setup_user_hub_signalr(self, on_user_event=None):
        """Builds the user-hub SignalR connection. on_user_event(event_type, payload) is called
        for each GatewayUser* message after we extract the payload dict from the [accountId, dict] envelope.
        """
        if not self._ensure_token():
            self.logger.error("User-hub SignalR setup aborted: Authentication failed.")
            return
        if not self.account_id:
            self.logger.error("User-hub SignalR setup aborted: No account_id configured.")
            return

        if on_user_event is not None:
            self._on_user_event = on_user_event
        try:
            ws_url = f"{self.user_hub_url}?access_token={self.access_token}"
            self.user_connection = HubConnectionBuilder()\
                .with_url(ws_url, {"access_token_factory": self._token_factory})\
                .with_automatic_reconnect({
                    "type": "raw",
                    "keep_alive_interval": 10,
                    "reconnect_interval": 5,
                    "max_attempts": 3
                })\
                .build()

            self.user_connection.on_open(self._on_user_hub_open)
            self.user_connection.on_close(self._on_user_hub_close)
            self.user_connection.on_reconnect(self._on_user_hub_reconnect)
            def _on_user_error(error):
                self.logger.error(f"User-hub SignalR Error: {error}")
                if "401" in str(error) or "Unauthorized" in str(error):
                    self.logger.warning("User-hub 401 detected in error handler, triggering token refresh")
                    self.user_is_connected = False
                    with self.auth_lock:
                        self.login()
            self.user_connection.on_error(_on_user_error)

            for event_name in self.USER_HUB_EVENTS:
                # Closure captures event_name for each event type.
                self.user_connection.on(event_name, self._make_user_event_handler(event_name))
            # Probe handlers: log raw payload at INFO so we can SEE what TopstepX actually
            # broadcasts, then wire the real names into _on_user_event later.
            for probe_name in self.USER_HUB_EVENT_PROBES:
                self.user_connection.on(probe_name, self._make_user_event_probe(probe_name))

            def on_logout(_msg):
                self.logger.error("User-hub: Received GatewayLogout (concurrent session detected).")
                self.user_is_connected = False
            self.user_connection.on("GatewayLogout", on_logout)

            self.logger.info("User-hub SignalR connection built; will subscribe on open.")
        except Exception as e:
            self.logger.error(f"Failed to build user-hub SignalR connection: {e}")
            self.user_connection = None

    def _make_user_event_handler(self, event_name):
        def handler(message):
            try:
                self.last_user_event_at = time.time()
                # Envelope shapes observed on TopstepX / ProjectX SignalR:
                # 1. [accountId, actionInt, payloadDict]
                # 2. [accountId, payloadDict]
                # 3. [accountId, [payloadDict1, payloadDict2]]
                # 4. [[payloadDict1, payloadDict2]]
                # 5. {"action": ..., "data": {...}}
                # 6. Unwrapped dict
                envelope_account_id = None
                payloads = []
                if isinstance(message, list):
                    if (
                        len(message) >= 2
                        and isinstance(message[0], (int, str))
                        and str(message[0]).isdigit()
                        and (len(message) >= 3 or int(message[0]) > 100)
                    ):
                        envelope_account_id = int(message[0])
                    if len(message) >= 3 and isinstance(message[2], dict):
                        payloads = [message[2]]
                    elif len(message) >= 3 and isinstance(message[2], list):
                        payloads = [item for item in message[2] if isinstance(item, dict)]
                    elif len(message) >= 2 and isinstance(message[1], dict):
                        payloads = [message[1]]
                    elif len(message) >= 2 and isinstance(message[1], list):
                        payloads = [item for item in message[1] if isinstance(item, dict)]
                    elif len(message) >= 1 and isinstance(message[0], dict):
                        payloads = [message[0]]
                    elif len(message) == 1 and isinstance(message[0], list):
                        payloads = [item for item in message[0] if isinstance(item, dict)]
                elif isinstance(message, dict):
                    payloads = [message]

                if not payloads:
                    self.logger.warning(f"User-hub {event_name}: unparseable payload {message!r}")
                    return

                if (
                    envelope_account_id is not None
                    and self.account_id is not None
                    and int(envelope_account_id) != int(self.account_id)
                ):
                    return

                for p in payloads:
                    if isinstance(p.get("data"), dict) and "id" not in p and "accountId" not in p:
                        p = p["data"]
                    p_acc_id = p.get("accountId") if event_name != "GatewayUserAccount" else p.get("id")
                    if p_acc_id is None:
                        p_acc_id = envelope_account_id
                    if p_acc_id is not None and self.account_id is not None:
                        try:
                            if int(p_acc_id) != int(self.account_id):
                                continue
                        except (ValueError, TypeError):
                            if str(p_acc_id).strip() != str(self.account_id).strip():
                                continue
                    self.logger.info(f"[USER-HUB EVENT] {event_name} payload={p!r}")
                    if self._on_user_event:
                        self._on_user_event(event_name, p)
            except Exception as e:
                self.logger.error(f"User-hub {event_name} handler error: {e}")
        return handler

    def _make_user_event_probe(self, event_name):
        """Probe handler — logs raw payload so we can identify any event names TopstepX
        actually broadcasts that aren't in USER_HUB_EVENTS yet."""
        def probe(message):
            try:
                self.last_user_event_at = time.time()
                self.logger.info(f"[USER-HUB PROBE] {event_name} raw={message!r}")
            except Exception as e:
                self.logger.error(f"User-hub probe {event_name} error: {e}")
        return probe

    def _on_user_hub_open(self):
        self.logger.info("User-hub SignalR Connection Opened")
        self.user_is_connected = True
        self._send_user_hub_subscriptions()

    def _on_user_hub_close(self):
        self.logger.info("User-hub SignalR Connection Closed")
        self.user_is_connected = False

    def _on_user_hub_reconnect(self):
        self.logger.warning("User-hub SignalR Reconnected. Re-issuing subscriptions...")
        self.user_is_connected = True
        self._send_user_hub_subscriptions()

    def _send_user_hub_subscriptions(self):
        if not self.user_connection:
            return
        for rpc_name, needs_account in self.USER_HUB_SUBSCRIPTIONS:
            try:
                args = [self.account_id] if needs_account else []
                self.user_connection.send(rpc_name, args)
                self.logger.info(f"User-hub: sent {rpc_name}({args})")
            except Exception as e:
                self.logger.error(f"User-hub: failed to send {rpc_name}: {e}")

    def start_user_streaming(self, force=False):
        if not force and self.user_is_connected:
            self.logger.info("User-hub SignalR already streaming.")
            return

        if not self._ensure_token():
            self.logger.error("User-hub SignalR start aborted: Authentication failed.")
            return

        self.user_is_connected = False
        if self.user_connection:
            self.stop_user_streaming()

        self.setup_user_hub_signalr(on_user_event=self._on_user_event)

        if self.user_connection:
            try:
                self.logger.info("Starting SignalR User Hub streaming...")
                self.user_connection.start()
            except Exception as e:
                self.logger.error(f"User-hub start failed: {e}")
        else:
            self.logger.warning("User-hub stream not started: connection not built.")

    def stop_user_streaming(self):
        if self.user_connection:
            try:
                self.logger.info("Stopping SignalR User Hub streaming...")
                self.user_connection.stop()
            except Exception as e:
                self.logger.error(f"User-hub stop failed: {e}")
            finally:
                try:
                    transport = getattr(self.user_connection, "transport", None)
                    checker = getattr(transport, "connection_checker", None)
                    if checker and getattr(checker, "running", False):
                        checker.stop()
                except Exception as exc:
                    self.logger.debug("Ignoring connection_checker stop failure: %s", exc)
                self.user_is_connected = False

    def reconnect_all_streams(self, force=True):
        """Forcefully refreshes authentication and restarts both market and user streams. Thread-safe & debounced."""
        now = time.time()
        with self._reconnect_lock:
            if not force and (now - self._last_reconnect_attempt < 10.0):
                self.logger.warning("reconnect_all_streams throttled (called within 10s).")
                return
            self._last_reconnect_attempt = now
            self.logger.warning("Initiating full SignalR stream reconnect (market + user)...")
            self.stop_streaming()
            self.stop_user_streaming()
            time.sleep(1.0)
            with self.auth_lock:
                self.access_token = None
                self.login()
            self.start_streaming(force=True)
            self.start_user_streaming(force=True)

            # Ensure active_subscriptions is populated with all known trading contracts
            cache_lock = getattr(self, "_symbol_cache_lock", None)
            with cache_lock if cache_lock is not None else contextlib.nullcontext():
                for sym in list(self.contract_id_to_symbol.values()):
                    self.active_subscriptions.add(sym)
                for sym in list(self.symbol_to_id_cache.keys()):
                    if not sym.startswith("/") and len(sym) <= 6:
                        self.active_subscriptions.add(sym)

            # Wait up to 3 seconds for connection handshake to complete
            for _ in range(12):
                if self.is_connected:
                    break
                time.sleep(0.25)

            if self.active_subscriptions:
                self._send_market_data_subscriptions()
            self.logger.info("Full SignalR stream reconnect sequence completed.")

    def resolve_symbol(self, base_symbol):
        """
        Resolves a base ticker (e.g., 'MNQ') to its active contract name (e.g., 'MNQM6').
        Also caches contract metadata (tickSize, tickValue) for use by other agents.
        """
        # FIX #12: Ensure we have a valid token before resolving
        self._ensure_token()
        
        mode = os.getenv("MODE", "simulation").strip().lower()
        is_live = (mode == "live")
        
        url = f"{self.base_url}/Contract/search"
        payload = {
            "searchText": base_symbol,
            # TopstepX evaluations internally exist on the simulation feed.
            # Only TRUE funded accounts use live=True in the TopstepX backend.
            "live": False 
        }
        
        try:
            self.logger.info(f"Resolving active contract for '{base_symbol}' (Live: {is_live})...")
            response = self._request_with_refresh(requests.post, url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            contracts = data.get('contracts', [])
            
            import datetime
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            valid_contracts = []
            for contract in contracts:
                if not contract.get('activeContract'):
                    continue
                exp_str = contract.get('expirationDate') or contract.get('lastTradingDay')
                exp_dt = None
                if exp_str:
                    try:
                        exp_dt = datetime.datetime.fromisoformat(exp_str.replace("Z", "+00:00"))
                        if exp_dt.tzinfo is None:
                            exp_dt = exp_dt.replace(tzinfo=datetime.timezone.utc)
                    except Exception:
                        exp_dt = None
                if exp_dt is None or exp_dt > now_utc:
                    cname_upper = str(contract.get('name') or "").upper()
                    prefix_rank = 0 if cname_upper.startswith(str(base_symbol).upper()) else 1
                    valid_contracts.append((prefix_rank, exp_dt or datetime.datetime.max.replace(tzinfo=datetime.timezone.utc), contract))

            if valid_contracts:
                valid_contracts.sort(key=lambda x: (x[0], x[1]))
                contract = valid_contracts[0][2]
                resolved_name = contract.get('name') or ""
                contract_id = contract.get('id')
                display_name = contract.get('contractDisplayName') or contract.get('contractName') or ""
                
                meta = {
                    "name": resolved_name,
                    "contractDisplayName": display_name,
                    "id": contract_id,
                    "tickSize": contract.get('tickSize', 0.25),
                    "tickValue": contract.get('tickValue', 0.5),
                    "description": contract.get('description', ''),
                    "symbolId": contract.get('symbolId', '')
                }
                
                # Cache tick metadata under base symbol, resolved name, and display name
                with self._symbol_cache_lock:
                    self.contract_cache[base_symbol] = meta
                    self.symbol_to_id_cache[base_symbol] = contract_id
                    self.symbol_to_id_cache[f"/{base_symbol}"] = contract_id
                    if resolved_name:
                        self.contract_cache[resolved_name] = meta
                        self.symbol_to_id_cache[resolved_name] = contract_id
                        self.symbol_to_id_cache[f"/{resolved_name}"] = contract_id
                    if display_name:
                        self.contract_cache[display_name] = meta
                        self.symbol_to_id_cache[display_name] = contract_id
                        self.symbol_to_id_cache[f"/{display_name}"] = contract_id

                    # Generate 1-digit vs 2-digit year aliases (e.g. MESZ6 <-> MESZ26, MCLX6 <-> MCLX26)
                    import re
                    aliases = []
                    for s in (resolved_name, display_name):
                        if not s:
                            continue
                        m = re.match(r'^([A-Z0-9]+[FGHJKMNQUVXZ])(\d{1,2})$', s)
                        if m:
                            root, yr = m.group(1), m.group(2)
                            if len(yr) == 1:
                                alias = f"{root}2{yr}"
                            elif len(yr) == 2 and yr.startswith("2"):
                                alias = f"{root}{yr[-1]}"
                            else:
                                alias = None
                            if alias:
                                aliases.append(alias)
                                self.symbol_to_id_cache[alias] = contract_id
                                self.symbol_to_id_cache[f"/{alias}"] = contract_id
                                self.contract_cache[alias] = meta

                    candidates = [c for c in (resolved_name, display_name, *aliases, base_symbol) if c]
                    one_digit = [c for c in candidates if re.search(r'^[A-Z0-9]+[FGHJKMNQUVXZ]\d$', c)]
                    primary_symbol = one_digit[0] if one_digit else (resolved_name or display_name or base_symbol)
                    self.contract_id_to_symbol[contract_id] = primary_symbol
                    self.contract_id_to_symbol[str(contract_id)] = primary_symbol
                    self.active_subscriptions.add(primary_symbol)
                    if resolved_name:
                        self.active_subscriptions.add(resolved_name)

                self.logger.info(
                    f"Resolved '{base_symbol}' -> '{resolved_name}' (id={contract_id}, display={display_name}) "
                    f"(tickSize={contract.get('tickSize')}, tickValue=${contract.get('tickValue')})"
                )
                return resolved_name
            
            self.logger.error(f"No active contract found for '{base_symbol}'")
            return None
            
        except Exception as e:
            self.logger.error(f"Symbol resolution failed for '{base_symbol}': {e}")
            return None

    def get_contract_meta(self, base_symbol):
        """Returns cached contract metadata for a base symbol."""
        return self.contract_cache.get(base_symbol)

    def retrieve_bars(self, contract_id, start_time, end_time, unit=2, unit_number=1, limit=20000):
        """
        Fetches historical bars from TopstepX API.
        unit: 2 = Minute, 3 = Hour, 4 = Day
        start_time/end_time: ISO 8601 strings (e.g., '2026-03-01T00:00:00Z')
        """
        # FIX #7: Use _ensure_token() consistently (handles expiry + refresh)
        if not self._ensure_token():
            return {"bars": []}
        
        url = f"{self.base_url}/History/retrieveBars"
        
        import datetime
        st_str = start_time.isoformat() if isinstance(start_time, datetime.datetime) else start_time
        et_str = end_time.isoformat() if isinstance(end_time, datetime.datetime) else end_time

        payload = {
            "contractId": contract_id,
            # TopstepX evaluations internally exist on the simulation feed.
            "live": False, 
            "startTime": st_str,
            "endTime": et_str,
            "unit": unit,
            "unitNumber": unit_number,
            "limit": limit,
            "includePartialBar": False
        }
        
        try:
            self.logger.info(f"Fetching {limit} bars for '{contract_id}' from {start_time}...")
            response = self._request_with_refresh(requests.post, url, json=payload)
            res = response.json()
            if isinstance(res, list):
                return {"bars": res}
            return res if isinstance(res, dict) else {"bars": []}
        except Exception as e:
            self.logger.error(f"Failed to retrieve bars for '{contract_id}': {e}")
            return {"bars": []}

    def _send_market_data_subscriptions(self):
        """Sends SubscribeContractQuotes for all active subscriptions (deduplicated by contract_id)."""
        if not self.is_connected or not self.connection:
            return False
        try:
            sent_contracts = set()
            for sym in list(self.active_subscriptions):
                contract_id = (
                    self.symbol_to_id_cache.get(sym)
                    or (self.contract_cache.get(sym, {}).get("id"))
                    or self.symbol_to_id_cache.get(sym.rstrip("0123456789"))
                    or sym
                )
                cid_str = str(contract_id)
                if cid_str in sent_contracts:
                    continue
                sent_contracts.add(cid_str)
                self.connection.send("SubscribeContractQuotes", [cid_str])
                self.logger.info(f"Sent subscription request for {cid_str} ({sym})")
            return True
        except Exception as e:
            self.logger.error(f"Failed to send market data subscriptions: {e}")
            return False

    def subscribe_market_data(self, symbols):
        """Subscribe with error handling and connection check. Queues subscriptions if connection is pending."""
        try:
            payload = symbols if isinstance(symbols, list) else [symbols]
            for sym in payload:
                self.active_subscriptions.add(sym)
                
            if not self.is_connected or not self.connection:
                self.logger.warning("SignalR not connected yet; market data subscriptions queued.")
                return False
            
            return self._send_market_data_subscriptions()
        except Exception as e:
            self.logger.error(f"Failed to subscribe to market data: {e}")
            return False
