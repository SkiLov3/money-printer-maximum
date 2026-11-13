# pyright: reportArgumentType=false, reportOptionalMemberAccess=false, reportCallIssue=false, reportOperatorIssue=false
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from datetime import time as dt_time
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pytz

from src.agents.data_agent import DataAgent
from src.agents.execution_agent import ExecutionAgent
from src.agents.news_agent import NewsAgent
from src.agents.risk_agent import RiskAgent
from src.agents.strategy_agent import StrategyAgent
from src.api.account_resolver import AccountResolver
from src.api.projectx_client import (
    PROJECTX_STATUS_CODES,
    ProjectXClient,
    normalize_order_type,
    normalize_status,
)
from src.charting import bars_to_chart_records, chart_time, normalize_frame
from src.config.settings import (
    get_data_config,
    get_database_config,
    get_discovery_config,
    get_feature_flags,
    get_health_config,
    get_learning_review_config,
    get_mega_vision_config,
    get_meta_optimizer_config,
    get_routing_config,
    get_strategy_config,
    get_symbol_brackets,
    get_zone_config,
    load_stack_config,
)
from src.discovery.meta_optimizer import MetaOptimizerService
from src.discovery.service import DiscoveryService
from src.learning.session_capture import (
    build_trade_key,
    capture_trade_context_artifacts,
)
from src.monitoring.health import StrategyHealthMonitor
from src.optimization.context_embedder import embed_market
from src.persistence.database import DatabaseService
from src.telemetry.store import TelemetryStore
from src.trading.filters import FilterManager
from src.trading.router import RouterRejection, TradeRouter
from src.zones.engine import ZoneSignalEngine


class SupervisorAgent:
    SIGNAL_COOLDOWN_SECONDS = 300   # Default cooldown between signals
    WIN_COOLDOWN_SECONDS    = 120   # Shorter cooldown after a winning trade
    LOSS_COOLDOWN_SECONDS   = 1800  # 30-min stand-down after a stop-loss (was 600)
    MAX_CONSECUTIVE_LOSSES  = 3     # After N consecutive losses, escalate cooldown
    STREAK_COOLDOWN_SECONDS = 3600  # 60-min halt after consecutive loss streak
    POST_FILL_LOCKOUT_SECONDS = 30  # Block new orders for N seconds after a position close (anti-stacking)

    def __init__(self, symbols: list, account_id: str):
        self.symbols = symbols
        self.account_id = account_id
        self.logger = logging.getLogger(__name__)

        self.client = ProjectXClient()
        if self.account_id and str(self.account_id).isdigit():
            self.client.account_id = int(self.account_id)
        self.risk_agent = RiskAgent(account_id)
        self.execution_agent = ExecutionAgent(self.client)
        self.news_agent = NewsAgent()
        self.telemetry = TelemetryStore()
        self.stack_config = load_stack_config()
        self.strategy_config = get_strategy_config(self.stack_config)
        self.feature_flags = get_feature_flags(self.stack_config)
        self.routing_config = get_routing_config(self.stack_config)
        self.data_config = get_data_config(self.stack_config)
        self.zone_config = get_zone_config(self.stack_config)
        self.health_config = get_health_config(self.stack_config)
        self.discovery_config = get_discovery_config(self.stack_config)
        self.mega_vision_config = get_mega_vision_config(self.stack_config)
        self.meta_optimizer_config = get_meta_optimizer_config(self.stack_config)
        self.learning_review_config = get_learning_review_config(self.stack_config)
        self.database_config = get_database_config(self.stack_config)

        self.official_symbols = []
        self.symbol_map = {}
        self.data_agents = {}
        self.strategy_agents = {}
        self.zone_engines = {}

        db_config = self.database_config if self.feature_flags.get("database_enabled", False) else {"enabled": False}
        self.database = DatabaseService(db_config)
        self.discovery = DiscoveryService(self.discovery_config, database=self.database)
        self.health_monitor = StrategyHealthMonitor(self.health_config, database=self.database)
        self.filter_manager = FilterManager(self.routing_config)
        self.trade_router = TradeRouter(self.risk_agent, self.execution_agent, telemetry=self.telemetry, filter_manager=self.filter_manager)

        self.last_signal_time = {}
        self.active_positions = {}
        # Per-symbol persistent-desync timestamps used by _sync_positions for orphan adoption.
        self._desync_start_time = {}
        self._last_full_rest_sync = 0.0
        self._recently_closed = {}   # symbol -> timestamp (60s grace period)
        self._eod_triggered = False  # Flag to prevent multiple EOD flattens
        # FIX: Tracks symbols where a close/flatten order is in-flight.
        # While True, _handle_position_open_update will NOT re-record a new open position
        # from broker netSize events (which fire mid-close and carry intermediate values).
        # Cleared when broker confirms netSize=0 (flat) for the symbol.
        # _close_in_flight_since timestamps each entry so we can auto-expire stale flags
        # if the broker never sends a netSize=0 confirmation (e.g. dropped WebSocket message).
        self._close_in_flight: dict = {}
        self._close_in_flight_since: dict = {}  # symbol -> float (time.time())
        # Tracks when a netSize=0 drain future is already queued for a symbol.
        # Prevents Case B (order fill) from scheduling a second _handle_position_close_update
        # that would race to pop _pending_fill_pnl and record pnl=0.0.
        self._drain_queued: dict = {}  # symbol -> bool
        # FIX: One-shot risk halt — once a daily loss/drawdown limit is breached,
        # this flag is set and the supervisor stops all trading immediately.
        # The bot continues running (streaming/logging) but dispatches NO new trades
        # and does NOT re-trigger the emergency flatten on every subsequent tick.
        self._risk_halt: bool = False
        self._session_reset_done: bool = False  # True once we've reset for the current maintenance window
        self._preopen_streams_ready: bool = False
        self._session_opening_balance: Decimal | None = None
        self._balance_persistence_path = Path("logs/session_opening_balance.json")
        self._restore_balance_persistence()

        self.running = False
        self._lock = asyncio.Lock()
        self._last_account_poll = 0.0
        self._last_status_write = 0.0
        self.loop: Any = None
        self._recent_trades = []
        self._chart_trade_events = []
        # FIX: Stash PnL from GatewayUserTrade events so the GatewayUserPosition
        # close handler (which fires with netSize=0 but no PnL field) can pick it up.
        self._pending_fill_pnl: dict = {}  # symbol -> accumulated float PnL
        self._session_pnl_from_fills = 0.0  # Running total from GatewayUserTrade events
        self._pnl_history = []
        self._last_saturday_optimizer_run: str | None = None
        self._optimizer_task: asyncio.Task | None = None
        self._optimizer_failures: int = 0
        self._optimizer_next_retry_at: float = 0.0
        try:
            with open("logs/bot_status.json") as f:
                last_status = json.load(f)
                if isinstance(last_status, dict):
                    if str(last_status.get("account_id")) == str(self.account_id):
                        self._pnl_history = last_status.get("pnl_history", [])
                        recent = last_status.get("session_trades", last_status.get("recent_trades", []))
                        if not isinstance(recent, list):
                            recent = []
                        session_start_utc = self.get_current_session_start_utc()

                        valid_session_trades = []
                        for t in reversed(recent):
                            if not isinstance(t, dict):
                                continue
                            ts_str = t.get("timestamp")
                            if ts_str:
                                try:
                                    trade_dt = datetime.fromisoformat(ts_str)
                                    if trade_dt >= session_start_utc:
                                        valid_session_trades.append(t)
                                except (ValueError, TypeError):
                                    continue
                        self._recent_trades = valid_session_trades
                    else:
                        self.logger.info(f"Account ID changed from '{last_status.get('account_id')}' to '{self.account_id}'. Clearing old session history.")
                        self._pnl_history = []
                        self._recent_trades = []
        except (OSError, ValueError, TypeError, AttributeError, KeyError) as e:
            self.logger.debug(f"Could not load initial bot_status.json: {e}")

        # Restore recent/session trades from persistent database
        if self.database and self.database.connected:
            try:
                db_trades = self.database.list_recent_trades(20, account_id=self.account_id)
                session_start_utc = self.get_current_session_start_utc()
                valid_session = []
                for t in reversed(db_trades):
                    ts_str = t.get("timestamp")
                    if ts_str:
                        try:
                            trade_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            if trade_dt >= session_start_utc:
                                valid_session.append(t)
                        except (ValueError, TypeError):
                            continue
                if valid_session:
                    self._recent_trades = valid_session
                elif not self._recent_trades and db_trades:
                    self._recent_trades = list(reversed(db_trades[-20:]))
            except Exception as e:
                self.logger.debug(f"Failed to restore recent trades from database on init: {e}")

        self._ticks_received = 0
        self._last_trade_won = {}
        self._consecutive_losses: dict = {}  # symbol -> int count of consecutive losses
        self._last_discovery_decisions = {}
        self._last_mega_vision_decisions = {}
        self._last_meta_optimizer_runs = {}
        self._meta_optimizer_last_run_at = {}
        self._meta_optimizer_services = {}
        self._meta_optimizer_catalog = self._load_dataset_catalog()
        # Latest broker poll snapshot — written to status JSON so the dashboard always
        # reflects broker truth (not the bot's internal cache, which can briefly diverge).
        self._latest_broker_positions_snapshot = {}
        self._latest_broker_positions_at = 0.0
        self._last_stall_reconnect_at = 0.0
        self._reconnect_backoff_secs = 10.0
        self._next_reconnect_at = 0.0
        self._order_filled_qty: dict[str, int] = {}
        self._db_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="swarm-db")
        self._last_bracket_guard_at = 0.0
        self._stop_placement_in_flight: dict[str, float] = {}
        self._started_at = time.time()

    async def start(self):
        self.logger.info(f"Supervisor initializing for symbols: {self.symbols}...")
        self.loop = asyncio.get_running_loop()
        # FIX: Track the main task so we can forcefully cancel it on shutdown
        # if it's stuck in a long-running await (e.g. hanging REST call).
        self._main_task = asyncio.current_task()

        if not self.client.login():
            self.logger.error("Critical: Authentication failed. Aborting startup.")
            return

        try:
            resolver = AccountResolver(self.client)
            resolved_acc = resolver.resolve_account(self.account_id)
            if resolved_acc:
                self.account_id = str(resolved_acc.id)
                self.client.account_id = resolved_acc.id
                self.risk_agent.account_id = str(resolved_acc.id)
                self.logger.info(f"TopstepX Active Account: {resolved_acc}")
                self._restore_balance_persistence()
                if self.database and self.database.connected:
                    try:
                        db_trades = self.database.list_recent_trades(20, account_id=self.account_id)
                        if db_trades:
                            session_start_utc = self.get_current_session_start_utc()
                            valid_session = []
                            for t in reversed(db_trades):
                                ts_str = t.get("timestamp")
                                if ts_str:
                                    try:
                                        trade_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                                        if trade_dt >= session_start_utc:
                                            valid_session.append(t)
                                    except (ValueError, TypeError):
                                        continue
                            if valid_session:
                                self._recent_trades = valid_session
                            elif not self._recent_trades:
                                self._recent_trades = list(reversed(db_trades[-20:]))
                    except Exception as sync_exc:
                        self.logger.debug(f"Failed to sync recent trades for resolved account {self.account_id}: {sync_exc}")
        except Exception as e:
            self.logger.warning(f"Account resolution encountered error: {e}. Continuing with account {self.account_id}")

        resolved_map = {}
        for sym in self.symbols:
            resolved = self.client.resolve_symbol(sym)
            if not resolved:
                self.logger.error(f"Critical: Could not resolve symbol '{sym}' to an active contract. Aborting for safety.")
                return
            resolved_map[sym] = resolved

        self.official_symbols = list(resolved_map.values())
        self.symbol_map = resolved_map
        # Pre-register subscriptions so ProjectXClient knows which contracts to stream
        self.client.subscribe_market_data(self.official_symbols)

        # Initialize default telemetry/states so the dashboard widgets display instantly on start
        self.health_monitor.initialize_symbols(self.official_symbols)
        self._last_discovery_decisions = {
            sym: {"allowed": True, "reason": "neutral (startup)", "quantity_multiplier": 1.0, "best_success_similarity": 0.0, "best_failure_similarity": 0.0}
            for sym in self.official_symbols
        }
        max_bars = int(self.data_config.get("max_bars", 12000) or 12000)
        self.data_agents = {official: DataAgent(official, max_bars=max_bars) for official in self.official_symbols}
        self.strategy_agents = {
            official: StrategyAgent(official, strategy_config=self.strategy_config, telemetry=self.telemetry)
            for official in self.official_symbols
        }
        self.zone_engines = {
            official: ZoneSignalEngine(official, config=self.zone_config, database=self.database)
            for official in self.official_symbols
        }

        from datetime import timedelta

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=int(self.data_config.get("history_warmup_days", 10) or 10))

        for base_sym, official_sym in resolved_map.items():
            meta = self.client.get_contract_meta(base_sym)
            if meta:
                self.risk_agent.register_contract(official_sym, meta)
                self.logger.info(f"Warming up {official_sym} with historical data...")
                hist_data = self.client.retrieve_bars(
                    meta["id"],
                    start_time,
                    end_time,
                    unit=2,
                    unit_number=1,
                    limit=int(self.data_config.get("history_limit", max_bars) or max_bars),
                )
                bars = hist_data.get("bars", []) if isinstance(hist_data, dict) else hist_data
                self.data_agents[official_sym].load_historical_bars(bars)

        self.logger.info("Initializing Execution Ledger from today's fills...")
        broker_positions_at_start: list | None = None
        try:
            # Sync initial account status (MLL, Balance) before trading starts
            status = self.client.get_account_status()
            if status:
                balance = status.get("balance", 0)
                pnl = status.get("dailyPnL")  # None when absent — preserves fill-accumulated P&L
                mll = status.get("minimumBalance")
                self.risk_agent.update_account_state(
                    balance, pnl, 0, minimum_balance=mll, session_opening_balance=self._session_opening_balance
                )
                self.logger.info(f"Initial Account Sync: Balance=${balance}, MLL=${mll}")

            # FIX #H13: Reconstruct net position from execution history (Ledger)
            # Cross-validated against /Position/searchOpen so broker is always truth.
            orders = await self.loop.run_in_executor(None, self.client.get_order_history, 300)
            if orders:
                self.logger.info(f"Scanning {len(orders)} recent orders for fills...")
                ledger = {}  # sym -> net_qty

                FILLED_STATUS_CODES = {
                    code for code, name in PROJECTX_STATUS_CODES.items() if name == "filled"
                }
                FILLED_STATUS_STRINGS = {"filled", "completed", "executed"}

                today = datetime.now(timezone.utc).date()
                filled_orders = []
                for o in orders:
                    raw_status = o.get("status")
                    is_filled = (
                        (isinstance(raw_status, (int, float)) and int(raw_status) in FILLED_STATUS_CODES)
                        or (isinstance(raw_status, str) and raw_status.lower() in FILLED_STATUS_STRINGS)
                    )
                    if is_filled:
                        otime_str = o.get("executedAt") or o.get("updatedAt") or o.get("createdAt") or o.get("updateTimestamp") or o.get("creationTimestamp")
                        if otime_str:
                            otime = pd.to_datetime(otime_str).to_pydatetime()
                            if otime.date() >= today:
                                filled_orders.append(o)

                filled_orders.sort(key=lambda x: x.get("executedAt") or x.get("updatedAt") or x.get("updateTimestamp") or "")

                for o in filled_orders:
                    sym = self._resolve_symbol_from_payload(o)
                    # FIX Bug 4: Use fillVolume (actual filled qty) not size (order qty).
                    qty_raw = o.get("fillVolume") or o.get("filledQty") or o.get("size") or 0
                    qty = int(abs(self._safe_float(qty_raw)))
                    raw_side = o.get("side")
                    if isinstance(raw_side, (int, float)):
                        side = "Buy" if int(raw_side) == 0 else "Sell"
                    else:
                        side = "Buy" if str(raw_side).lower() in ("0", "buy", "long") else "Sell"

                    if sym not in ledger:
                        ledger[sym] = 0
                    ledger[sym] += qty if side == "Buy" else -qty

                # FIX Bug 4: Cross-validate ledger net against live broker REST before adopting.
                # The broker's /Position/searchOpen is authoritative — if it disagrees with the
                # order-history calc, trust the broker. This handles partial fills, cancelled
                # brackets and any other edge cases the order history doesn't capture.
                broker_positions_at_start = await self.loop.run_in_executor(None, self.client.get_positions)
                broker_net = {}
                broker_avg_price = {}
                if broker_positions_at_start is not None:
                    for p in broker_positions_at_start:
                        sym = self._resolve_symbol_from_payload(p)
                        pos_type = int(p.get("type", 0))
                        qty = int(abs(self._safe_float(p.get("size", 0))))
                        avg_p = float(p.get("averagePrice") or p.get("avgPrice") or p.get("entryPrice") or 0.0)
                        if pos_type == 1:
                            broker_net[sym] = broker_net.get(sym, 0) + qty       # long
                        elif pos_type == 2:
                            broker_net[sym] = broker_net.get(sym, 0) - qty      # short
                        if avg_p > 0:
                            broker_avg_price[sym] = avg_p
                else:
                    self.logger.warning(
                        "LEDGER STARTUP: Broker positions fetch returned None (API error/timeout). "
                        "Trusting order-history ledger without REST cross-validation."
                    )

                for sym, net_qty in ledger.items():
                    if broker_positions_at_start is not None:
                        broker_qty = broker_net.get(sym, 0)
                        if net_qty != broker_qty:
                            self.logger.warning(
                                f"LEDGER STARTUP mismatch for {sym}: order-history net={net_qty}, "
                                f"broker REST={broker_qty}. Trusting broker REST as source of truth."
                            )
                            net_qty = broker_qty

                    if net_qty != 0:
                        side = "Buy" if net_qty > 0 else "Sell"
                        qty = abs(net_qty)
                        avg_price = broker_avg_price.get(sym, 0.0)
                        self.logger.info(f"LEDGER STARTUP: Resolved {sym} net position: {net_qty} @ {avg_price}")
                        await self._handle_position_open_update(sym, side, qty, avg_price)
                        await self._ensure_protective_stop_for_position(sym, side, qty, avg_price)

                # Also adopt any broker positions not present in the order history
                if broker_positions_at_start is not None:
                    for sym, broker_qty in broker_net.items():
                        if sym not in ledger and broker_qty != 0:
                            side = "Buy" if broker_qty > 0 else "Sell"
                            qty = abs(broker_qty)
                            avg_price = broker_avg_price.get(sym, 0.0)
                            self.logger.warning(f"LEDGER STARTUP: Broker has {sym} {side} {qty}x @ {avg_price} not in order history — adopting.")
                            await self._handle_position_open_update(sym, side, qty, avg_price)
                            await self._ensure_protective_stop_for_position(sym, side, qty, avg_price)
            else:
                self.logger.info("No recent orders found. Checking broker for open positions...")
                broker_positions_at_start = await self.loop.run_in_executor(None, self.client.get_positions)
                if broker_positions_at_start is None:
                    self.logger.warning("LEDGER STARTUP: Broker positions fetch returned None (API error/timeout).")
                else:
                    for p in broker_positions_at_start:
                        sym = self._resolve_symbol_from_payload(p)
                        pos_type = int(p.get("type", 0))
                        qty = int(abs(self._safe_float(p.get("size", 0))))
                        avg_p = float(p.get("averagePrice") or p.get("avgPrice") or p.get("entryPrice") or 0.0)
                        if pos_type in (1, 2) and qty > 0:
                            side = "Buy" if pos_type == 1 else "Sell"
                            self.logger.warning(f"LEDGER STARTUP: No order history but broker has {sym} {side} {qty}x @ {avg_p} — adopting.")
                            await self._handle_position_open_update(sym, side, qty, avg_p)
                            await self._ensure_protective_stop_for_position(sym, side, qty, avg_p)

        except Exception as e:
            self.logger.error(f"Ledger initialization failed: {e}")

        # STARTUP BRACKET SWEEP — Cancel any open orders left from previous sessions.
        # For each symbol we're trading: if our ledger says we're FLAT (no active position),
        # any open bracket orders at the broker are orphans from a prior run and must be
        # cancelled immediately. If we DO have an active position, leave orders untouched
        # since they are the live SL/TP brackets protecting that position.
        try:
            if broker_positions_at_start is None:
                self.logger.warning(
                    "STARTUP SWEEP: Skipping startup bracket sweep because broker positions fetch returned None (API error/timeout). "
                    "Skipping order cancellation for safety."
                )
            else:
                self.logger.info("Running startup bracket sweep for orphaned bracket orders...")
                for sym in self.official_symbols:
                    has_position = bool(self.active_positions.get(sym))
                    if not has_position:
                        cancelled = await self.loop.run_in_executor(
                            None, self.client.cancel_open_orders_for_symbol, sym
                        )
                        if cancelled:
                            self.logger.warning(
                                f"STARTUP SWEEP: Cancelled {cancelled} orphaned bracket order(s) for {sym} — "
                                f"no active position exists for this symbol."
                            )
                        else:
                            self.logger.info(f"STARTUP SWEEP: No orphaned orders found for {sym}.")
        except Exception as e:
            self.logger.warning(f"Startup bracket sweep failed (non-fatal): {e}")


        # Prime all strategy engines with historical data so indicators are
        # fully seeded.  analyze() will then only evaluate genuinely new bars.
        for official_sym in self.official_symbols:
            latest = self.data_agents[official_sym].get_latest_data()
            if not latest.empty:
                self.logger.info(f"Priming {official_sym} strategy engine ({len(latest)} bars) — no trades dispatched")
                engine = self.strategy_agents[official_sym].engine
                indicators = engine.prepare_data(latest)
                bars = indicators[["timestamp", "open", "high", "low", "close", "volume", "price"]].copy()
                engine.prime(bars, indicators, len(indicators) - 1)

        self.logger.info(f"Supervisor starting Trading Swarm for: {self.official_symbols}")
        self.running = True
        self._warmup_complete = True
        await self.run()

    @staticmethod
    def _seconds_until_next_globex_open(now_dt: datetime) -> float | None:
        """Return seconds until 18:00 ET Globex open when currently inside a pre-open maintenance/Sunday window."""
        try:
            import zoneinfo
            et_tz = zoneinfo.ZoneInfo("America/New_York")
            dt_utc = now_dt if now_dt.tzinfo is not None else now_dt.replace(tzinfo=timezone.utc)
            dt_et = dt_utc.astimezone(et_tz)
            wd = dt_et.weekday()
            # Mon-Thu 17:00-18:00 ET or Sunday before 18:00 ET
            if (wd in (0, 1, 2, 3) and dt_et.hour == 17) or (wd == 6 and dt_et.hour < 18):
                open_et = dt_et.replace(hour=18, minute=0, second=0, microsecond=0)
                secs = (open_et - dt_et).total_seconds()
                return max(0.0, secs)
        except Exception:
            return None
        return None

    async def _prepare_session_open_streams(self, last_processed_bar_count: dict[str, int]) -> None:
        """Resolve contract rollovers, reconnect SignalR streams, and backfill bars ahead of 18:00 ET open."""
        self.client.last_tick_at = time.time()
        for base_sym, old_off in list(getattr(self, "symbol_map", {}).items()):
            if not self.active_positions.get(old_off):
                try:
                    new_off = await self.loop.run_in_executor(None, self.client.resolve_symbol, base_sym)
                    if new_off and new_off != old_off:
                        self.logger.warning(f"[CONTRACT ROLLOVER] {base_sym} rolled from {old_off} -> {new_off}")
                        self.symbol_map[base_sym] = new_off
                        if old_off in self.official_symbols:
                            self.official_symbols = [new_off if s == old_off else s for s in self.official_symbols]
                        elif new_off not in self.official_symbols:
                            self.official_symbols.append(new_off)
                        if old_off in self.data_agents:
                            dagent = self.data_agents.pop(old_off)
                            dagent.symbol = new_off
                            self.data_agents[new_off] = dagent
                        if old_off in self.strategy_agents:
                            sagent = self.strategy_agents.pop(old_off)
                            sagent.symbol = new_off
                            if hasattr(sagent, "engine") and sagent.engine:
                                sagent.engine.symbol = new_off
                            self.strategy_agents[new_off] = sagent
                        if old_off in self.zone_engines:
                            zeng = self.zone_engines.pop(old_off)
                            zeng.symbol = new_off
                            self.zone_engines[new_off] = zeng
                        if old_off in last_processed_bar_count:
                            last_processed_bar_count[new_off] = last_processed_bar_count.pop(old_off)
                        meta = self.client.get_contract_meta(base_sym)
                        if meta:
                            self.risk_agent.register_contract(new_off, meta)
                except Exception as roll_exc:
                    self.logger.debug(f"Rollover check skipped for {base_sym}: {roll_exc}")
        await self.loop.run_in_executor(None, lambda: self.client.reconnect_all_streams(force=True))
        self.client.subscribe_market_data(self.official_symbols)
        await self._smart_sleep(3)
        await self._refresh_recent_bars(bar_count=120)
        for sym in self.official_symbols:
            if sym in self.data_agents:
                last_processed_bar_count[sym] = self.data_agents[sym].get_bar_count()

    async def run(self):
        self.logger.info("Supervisor swarm initiated...")

        self.client.setup_signalr(
            on_market_data=self._on_market_data_threadsafe,
            on_order_update=self._on_order_update_threadsafe,
            on_reconnect=self._on_reconnect_threadsafe
        )
        self.client.setup_user_hub_signalr(self._on_user_event_threadsafe)

        # Subscribe to market data quotes for all trading symbols
        self.client.subscribe_market_data(self.official_symbols)

        last_processed_bar_count = {sym: self.data_agents[sym].get_bar_count() for sym in self.official_symbols}


        while self.running:
            try:
                now_ts = time.time()
                now_dt = datetime.now(timezone.utc)
                market_open = self.news_agent.is_market_hours(now_dt)

                # If we just entered a new session after maintenance / weekend closure,
                # execute the 18:00 ET transition BEFORE _write_status() locks the new
                # session_opening_balance.json anchor.
                if market_open and self._session_reset_done:
                    self.logger.info("Market session opening (18:00 ET Globex). Executing session transition...")
                    self._session_reset_done = False
                    self._session_opening_balance = None
                    # Reset tick timestamp so the 90s/300s data stall watchdog doesn't fire immediately
                    self.client.last_tick_at = time.time()
                    if getattr(self, "_preopen_streams_ready", False) and self.client.is_connected and self.client.user_is_connected:
                        self.logger.info("Pre-open SignalR streams already connected — skipping reconnect to preserve 18:00 ET opening bar.")
                        self._preopen_streams_ready = False
                    else:
                        self._preopen_streams_ready = False
                        await self._prepare_session_open_streams(last_processed_bar_count)
                    try:
                        acc_status = await self.loop.run_in_executor(None, self.client.get_account_status)
                        if acc_status:
                            cur_bal = acc_status.get("balance", 0)
                            pnl_val = acc_status.get("dailyPnL")
                            mll_val = acc_status.get("minimumBalance")
                            if cur_bal and float(cur_bal) > 0:
                                self._session_opening_balance = Decimal(str(cur_bal))
                            self.risk_agent.update_account_state(
                                cur_bal, pnl_val, 0, minimum_balance=mll_val, session_opening_balance=self._session_opening_balance
                            )
                    except Exception as exc:
                        self.logger.warning(f"Session transition account sync failed: {exc}")
                    self._schedule_status_write()
                    self._last_status_write = time.time()
                    self._eod_triggered = False

                # Always keep dashboard updated — even when halted or market closed
                has_open_position = any(
                    self.active_positions.get(sym)
                    for sym in set(self.official_symbols) | set(self.active_positions)
                )
                status_interval = 1 if has_open_position else 5
                if now_ts - self._last_status_write >= status_interval:
                    self._schedule_status_write()
                    self._last_status_write = now_ts

                # RISK HALT: If daily loss/drawdown limit was previously hit this session,
                # skip all analysis and trade dispatch — bot is halted for the day.
                if self._risk_halt:
                    await self._smart_sleep(30)
                    continue

                # EOD HARD STOP CHECK (4:50 PM ET) — runs BEFORE the market-closed gate
                # so that the flatten fires even though is_market_hours returns False at 16:50.
                if market_open and self.news_agent.is_near_close(now_dt, lead_minutes=0):
                    if not self._eod_triggered:
                        self.logger.critical("EOD HARD STOP (4:50 PM ET) REACHED: Flattening all positions and suspending trading.")
                        await self._flatten_all_emergency()
                        self._eod_triggered = True
                    await self._smart_sleep(60)
                    continue

                # If market is closed, shut down streaming to avoid noise (except during 2m pre-open warm-up)
                if not market_open:
                    secs_to_open = self._seconds_until_next_globex_open(now_dt)
                    in_preopen_window = secs_to_open is not None and 0.0 < secs_to_open <= 120.0

                    # Flush any open 16:59 ET bars before shutting down streams
                    for sym in self.official_symbols:
                        if sym in self.data_agents:
                            self.data_agents[sym].check_bar_expiry(now_dt)
                            last_processed_bar_count[sym] = self.data_agents[sym].get_bar_count()
                    if not in_preopen_window and (self.client.is_connected or self.client.user_is_connected):
                        self.logger.info("Market is closed: Shutting down SignalR streaming.")
                        self.client.stop_streaming()
                        self.client.stop_user_streaming()
                        self._preopen_streams_ready = False
                    self._eod_triggered = False

                    # SESSION BOUNDARY RESET (fires once per maintenance window)
                    # When a daily halt fired this session, clear it now so the bot
                    # resumes trading when the next Globex session opens at 18:00 ET.
                    # We also zero the RiskAgent's accumulated daily P&L so the
                    # circuit-breaker won't immediately re-trigger on the new session.
                    if not self._session_reset_done:
                        if self._risk_halt:
                            self.logger.info(
                                "SESSION BOUNDARY: Clearing _risk_halt for new Globex session. "
                                "RiskAgent daily state will be reset."
                            )
                        self._risk_halt = False
                        self._session_pnl_from_fills = 0.0
                        self._recent_trades = []
                        self._consecutive_losses = {}
                        self._last_trade_won = {}
                        self.risk_agent.reset_session()
                        self._session_reset_done = True

                    # Pre-connect SignalR 2 minutes before 18:00 ET Globex open so the
                    # opening minute bar (18:00-18:01 ET) is captured without reconnect gaps.
                    if in_preopen_window and not getattr(self, "_preopen_streams_ready", False):
                        self.logger.info("Pre-open warm-up (2m before 18:00 ET Globex): connecting SignalR streams...")
                        await self._prepare_session_open_streams(last_processed_bar_count)
                        self._preopen_streams_ready = True

                    # Autonomous Saturday Optimization Routine (runs on Saturday morning when flat)
                    await self._check_saturday_optimization()

                    if secs_to_open is not None:
                        if in_preopen_window:
                            sleep_secs = min(5.0, max(0.5, secs_to_open))
                        else:
                            sleep_secs = min(60.0, max(1.0, secs_to_open - 120.0))
                    else:
                        sleep_secs = 60.0
                    await self._smart_sleep(sleep_secs)
                    continue

                # Ensure streams are active during market hours (exponential backoff prevents auth/429 storms)
                if not self.client.is_connected or not self.client.user_is_connected:
                    if now_ts >= getattr(self, "_next_reconnect_at", 0.0):
                        backoff = float(getattr(self, "_reconnect_backoff_secs", 10.0) or 10.0)
                        self.logger.warning(
                            f"Market session active but SignalR stream(s) disconnected. Re-initializing (next backoff={backoff:.0f}s)..."
                        )
                        await self.loop.run_in_executor(None, lambda: self.client.reconnect_all_streams(force=False))
                        self.client.subscribe_market_data(self.official_symbols)
                        await self._smart_sleep(3)
                        await self._refresh_recent_bars(bar_count=60)
                        self._eod_triggered = False
                        self._next_reconnect_at = time.time() + backoff
                        self._reconnect_backoff_secs = min(60.0, backoff * 2.0)
                else:
                    self._reconnect_backoff_secs = 10.0
                    self._next_reconnect_at = 0.0

                if now_ts - self._last_account_poll >= 30:
                    # Heartbeat Sync: Every 30s we poll account status.
                    # Every 300s (5m) we specifically log it as a heartbeat sync.
                    is_heartbeat = (now_ts - self._last_full_rest_sync >= 300)
                    if is_heartbeat:
                        self.logger.info(f"[HEARTBEAT] Performing full REST synchronization for {self.official_symbols}...")
                        self._last_full_rest_sync = now_ts

                    broker_positions = await self.loop.run_in_executor(None, self.client.get_positions)
                    if broker_positions is None:
                        self.logger.warning(
                            "Failed to fetch open positions from broker REST (get_positions returned None) — "
                            "skipping position reconciliation pass to preserve internal state."
                        )
                    else:
                        await self._sync_positions(broker_positions)

                    status = await self.loop.run_in_executor(None, self.client.get_account_status)
                    if status:
                        balance = status.get("balance", 0)
                        pnl = status.get("dailyPnL")  # None when absent — preserves fill-accumulated P&L
                        mll = status.get("minimumBalance")
                        async with self._lock:
                            open_pos = sum(len(v) for v in self.active_positions.values())
                        self.risk_agent.update_account_state(balance, pnl, open_pos, minimum_balance=mll, session_opening_balance=self._session_opening_balance)
                    self._last_account_poll = now_ts

                await self._run_meta_optimizer_cycle(now_ts)

                await self._check_high_frequency_guards(now_ts)
                await self._check_news_guards()

                now_dt = datetime.now(timezone.utc)
                for sym in self.official_symbols:
                    self.data_agents[sym].check_bar_expiry(now_dt, grace_seconds=2.0)
                    current_bar_count = self.data_agents[sym].get_bar_count()
                    if current_bar_count <= last_processed_bar_count[sym]:
                        continue

                    self.logger.info(f"Analyzing {sym} (New bar closed: {current_bar_count})")
                    latest_data = self.data_agents[sym].get_latest_data()
                    strategy_intent = self.strategy_agents[sym].analyze(latest_data)
                    zone_intent = None
                    if self.feature_flags.get("zone_engine_enabled", False):
                        zone_intent = self.zone_engines[sym].analyze(latest_data)
                    last_processed_bar_count[sym] = current_bar_count
                    async with self._lock:
                        active_count = len(self.active_positions.get(sym, []))
                        positions_for_sym = self.active_positions.get(sym, [])
                        existing_side = positions_for_sym[0]["side"] if positions_for_sym else ""
                        base_sym = StrategyAgent._extract_base_static(sym)
                        max_pos = self.risk_agent.position_limits.get(base_sym, {}).get("max_positions", 1)

                    # Graduated cooldown: short after wins, long after losses,
                    # longest after a streak of consecutive losses.
                    if self._last_trade_won.get(sym) is True:
                        cooldown_seconds = self.WIN_COOLDOWN_SECONDS
                    elif self._last_trade_won.get(sym) is False:
                        streak = self._consecutive_losses.get(sym, 0)
                        if streak >= self.MAX_CONSECUTIVE_LOSSES:
                            cooldown_seconds = self.STREAK_COOLDOWN_SECONDS
                        else:
                            cooldown_seconds = self.LOSS_COOLDOWN_SECONDS
                    else:
                        cooldown_seconds = self.SIGNAL_COOLDOWN_SECONDS
                    # Per-strategy cooldown override. Lets always-in strategies set
                    # base_seconds=0 so the filter no-ops while still applying to
                    # other strategies.
                    intent_strategy_name = (
                        strategy_intent.get("strategy_name") if isinstance(strategy_intent, dict) else None
                    )
                    if intent_strategy_name:
                        per_strategy_cd = (
                            (self.strategy_config.get("cooldowns") or {}).get(intent_strategy_name) or {}
                        )
                        if per_strategy_cd:
                            if self._last_trade_won.get(sym) is True:
                                cooldown_seconds = int(
                                    per_strategy_cd.get("win_seconds", per_strategy_cd.get("base_seconds", cooldown_seconds))
                                )
                            else:
                                cooldown_seconds = int(per_strategy_cd.get("base_seconds", cooldown_seconds))

                    route_context = {
                        "now_dt": now_dt,
                        "market_open": self.news_agent.is_market_hours(now_dt),
                        "news_standdown": self.news_agent.is_stand_down(now_dt),
                        "near_close": self.news_agent.is_near_close(now_dt, lead_minutes=15),
                        "active_count": active_count,
                        "max_positions": max_pos,
                        "existing_side": existing_side,
                        "pending_order": sym in self.risk_agent.pending_orders,
                        "last_signal_time": self.last_signal_time.get(sym, 0),
                        "cooldown_seconds": cooldown_seconds,
                        "current_price": latest_data.iloc[-1]["close"] if not latest_data.empty else 0,
                    }
                    trade_intent = await self._select_trade_intent(sym, strategy_intent, zone_intent, route_context=route_context)
                    if trade_intent:
                        # ANTI-STACKING GUARD: Block new orders if a position for this
                        # symbol was recently closed or is currently being closed. This
                        # prevents the always-in strategy from placing duplicate orders
                        # during the window between a broker SL fill and the internal
                        # position tracker catching up via reconciliation.
                        if self._close_in_flight.get(sym):
                            self.logger.info(f"[ANTI-STACK] {sym}: Blocked — close still in-flight.")
                            continue
                        last_closed_at = self._recently_closed.get(sym, 0)
                        seconds_since_close = time.time() - last_closed_at
                        if last_closed_at > 0 and seconds_since_close < self.POST_FILL_LOCKOUT_SECONDS:
                            self.logger.info(
                                f"[ANTI-STACK] {sym}: Blocked — position closed {int(seconds_since_close)}s ago "
                                f"(lockout {self.POST_FILL_LOCKOUT_SECONDS}s)."
                            )
                            continue
                        await self._dispatch_trade(sym, trade_intent, route_context)
                        await asyncio.sleep(2)


                await asyncio.sleep(1)

            except Exception as e:
                self.logger.exception(f"Supervisor Loop Critical Error: {e}")
                await self._smart_sleep(10)

    async def _refresh_recent_bars(self, bar_count=120):
        """Backfill recent bars via REST API to ensure data agents and strategies stay current."""
        from datetime import datetime, timedelta, timezone
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=bar_count + 15)
        symbol_map = getattr(self, "symbol_map", {})
        if not symbol_map:
            official = getattr(self, "official_symbols", [])
            symbol_map = {s: s for s in official}

        for base_sym, official_sym in symbol_map.items():
            meta = self.client.get_contract_meta(base_sym)
            contract_id = (meta.get("id") if meta else None) or self.client.symbol_to_id_cache.get(official_sym) or self.client.symbol_to_id_cache.get(base_sym)
            if not contract_id:
                continue
            try:
                hist_data = await self.loop.run_in_executor(
                    None,
                    lambda cid=contract_id: self.client.retrieve_bars(
                        cid,
                        start_time,
                        end_time,
                        unit=2,
                        unit_number=1,
                        limit=bar_count,
                    )
                )
                bars = hist_data.get("bars", []) if isinstance(hist_data, dict) else hist_data
                if bars and official_sym in self.data_agents:
                    self.data_agents[official_sym].load_historical_bars(bars)
                    self.logger.info(f"Backfilled {len(bars)} recent bars for {official_sym} via REST API.")
            except Exception as e:
                self.logger.warning(f"Failed to refresh recent bars for {official_sym}: {e}")

    async def _check_high_frequency_guards(self, now_ts):
        last_tick = self.client.last_tick_at
        now_dt = datetime.now(timezone.utc)
        market_open = self.news_agent.is_market_hours(now_dt)

        # 1. Self-healing data stall watchdog:
        # If market is open and ticks have stalled for > 90s (or never arrived within 60s of startup),
        # trigger reconnect, re-subscribe to market data, and backfill recent bars.
        is_stalled = False
        stall_duration = 0
        if market_open:
            if last_tick > 0 and (now_ts - last_tick) > 90:
                is_stalled = True
                stall_duration = int(now_ts - last_tick)
            elif last_tick == 0 and (now_ts - getattr(self, "_started_at", now_ts)) > 60:
                is_stalled = True
                stall_duration = int(now_ts - getattr(self, "_started_at", now_ts))

        if is_stalled:
            if (now_ts - self._last_stall_reconnect_at) > 60:
                self._last_stall_reconnect_at = now_ts
                self.logger.warning(
                    f"[DATA WATCHDOG] Data stall detected: No ticks for {stall_duration}s while market is open. "
                    f"Triggering self-healing reconnect, re-subscription, and bar backfill..."
                )
                try:
                    await self.loop.run_in_executor(None, lambda: self.client.reconnect_all_streams(force=True))
                    self.client.subscribe_market_data(self.official_symbols)
                    await self._refresh_recent_bars(bar_count=60)
                    self.logger.info("[DATA WATCHDOG] Self-healing reconnect and re-subscription completed.")
                except Exception as exc:
                    self.logger.error(f"[DATA WATCHDOG] Reconnect error: {exc}")

        # Mandatory emergency flatten if stall > 300s AND there are open positions not already closing
        if last_tick > 0 and (now_ts - last_tick) > 300:
            should_flatten = False
            async with self._lock:
                should_flatten = any(
                    plist and not self._close_in_flight.get(sym)
                    for sym, plist in self.active_positions.items()
                )
            if should_flatten:
                self.logger.critical("DATA STALL DETECTED: No ticks for 300s with open positions. MANDATORY FLATTEN ALL triggered for security.")
                await self._flatten_all_emergency()

        current_prices = {}
        bar_extremes = {}
        for sym, dagent in self.data_agents.items():
            lb = dagent.get_latest_data()
            if not lb.empty:
                last_row = lb.iloc[-1]
                c_price = float(last_row["close"])
                current_prices[sym] = c_price
                bar_extremes[sym] = (
                    float(last_row.get("high", c_price)),
                    float(last_row.get("low", c_price)),
                    c_price,
                )
        self._update_active_positions_excursions(bar_extremes)
        self.risk_agent.update_unrealized_pnl(current_prices)

        for sym in self.official_symbols:
            self.risk_agent.check_pending_timeout(sym)

        # Bracket Guard Watchdog: Every 30s, verify all active positions have a working protective stop order
        # Only run when live trading + API order brackets are enabled (simulation / server-side Position Brackets have no API stops).
        exec_mode = getattr(getattr(self, "execution_agent", None), "mode", None) or os.getenv("MODE", "simulation").strip().lower()
        use_brackets = os.getenv("USE_ORDER_BRACKETS", "true").strip().lower() == "true"
        live_switch = os.getenv("ACTUAL_LIVE_TRADING", "false").strip().lower() == "true"
        if (
            exec_mode == "live"
            and live_switch
            and use_brackets
            and now_ts - getattr(self, "_last_bracket_guard_at", 0) >= 30
        ):
            self._last_bracket_guard_at = now_ts
            async with self._lock:
                open_items = [
                    (sym, list(pos_list))
                    for sym, pos_list in self.active_positions.items()
                    if pos_list
                ]
            for sym, pos_list in open_items:
                if self._close_in_flight.get(sym):
                    continue
                total_qty = sum(p.get("qty", 0) for p in pos_list)
                if total_qty > 0 and pos_list:
                    # Give newest entry at least 30s before flagging missing bracket (order placement & index latency)
                    newest_time = max((p.get("time", 0) for p in pos_list), default=0)
                    if now_ts - newest_time > 30:
                        side = pos_list[0].get("side", "Buy")
                        entry = pos_list[0].get("entry", 0.0)
                        pos_sl_ticks = pos_list[0].get("sl_ticks")
                        await self._ensure_protective_stop_for_position(
                            sym, side, total_qty, entry, sl_ticks=pos_sl_ticks
                        )

        # FIX (Critical): Auto-expire stale _close_in_flight flags.
        # If a broker netSize=0 confirmation is dropped during a WebSocket reconnect,
        # the flag would otherwise stay True forever, silently blocking all new trades
        # on that symbol for the rest of the session.
        _CLOSE_IN_FLIGHT_TIMEOUT = 30.0
        stale_syms = [
            sym for sym, set_at in list(self._close_in_flight_since.items())
            if self._close_in_flight.get(sym) and (now_ts - set_at) > _CLOSE_IN_FLIGHT_TIMEOUT
        ]
        for sym in stale_syms:
            self.logger.warning(
                f"[ANTI-STACK] {sym}: _close_in_flight stuck for >{_CLOSE_IN_FLIGHT_TIMEOUT:.0f}s — "
                f"auto-clearing (broker netSize=0 likely dropped during reconnect)."
            )
            self._close_in_flight.pop(sym, None)
            self._close_in_flight_since.pop(sym, None)

        now_dt = datetime.now(timezone.utc)
        if self.news_agent.is_near_close(now_dt, lead_minutes=15):
            should_flatten = False
            async with self._lock:
                should_flatten = any(
                    plist and not self._close_in_flight.get(sym)
                    for sym, plist in self.active_positions.items()
                )
            if should_flatten:
                self.logger.critical("TOPSTEP SAFETY: Proactive EOD Flatten (15m before close) triggered.")
                await self._flatten_all_emergency()

        hit, reason = self.risk_agent.is_risk_limit_hit()
        if hit and not self._risk_halt:
            self.logger.critical(
                f"RISK LIMIT BREACH: {reason}. EMERGENCY FLATTEN triggered. "
                f"Bot halting all trading for the remainder of this session."
            )
            self._risk_halt = True
            await self._flatten_all_emergency()

    async def _flatten_all_emergency(self):
        self.logger.warning("!!! EMERGENCY FLATTEN ALL TRIGGERED !!!")

        # 1. Fetch AUTHORITATIVE state from broker REST (never rely on internal state here)
        broker_active_syms = set()
        broker_flattened_syms = set()
        fetch_failed = False
        try:
            broker_positions = await self.loop.run_in_executor(None, self.client.get_positions)
            if broker_positions is not None and broker_positions:
                self.logger.warning(f"Emergency Flatten: Found {len(broker_positions)} positions at broker to close.")
                for p in broker_positions:
                    sym = self._resolve_symbol_from_payload(p)

                    # FIX Bug 2 & 3: Read size and side DIRECTLY from the REST API fields.
                    # The /Position/searchOpen endpoint returns:
                    #   type: 0=flat, 1=long, 2=short
                    #   size: true open contract count (NOT a rolling average)
                    # Never use _extract_qty_side here — it reads ambiguous SignalR fields.
                    pos_type = int(p.get("type", 0))
                    qty = int(abs(self._safe_float(p.get("size", 0))))
                    if pos_type == 0 or qty == 0:
                        self.logger.info(f"Emergency Flatten: {sym} appears flat at broker (type={pos_type}, size={qty}) — skipping.")
                        continue

                    if self._close_in_flight.get(sym):
                        self.logger.info(f"Emergency Flatten: {sym} flatten already in flight — skipping duplicate close order.")
                        broker_flattened_syms.add(sym)
                        continue

                    broker_active_syms.add(sym)
                    side = "Buy" if pos_type == 1 else "Sell"

                    self.logger.critical(
                        f"Emergency Flatten: Closing {sym} {side} {qty}x "
                        f"(raw broker payload: type={pos_type} size={qty} avg={p.get('averagePrice')})"
                    )

                    # Mark close in-flight BEFORE sending the order so the SignalR handler
                    # ignores intermediate netSize updates from the flatten fill sequence.
                    self._close_in_flight[sym] = True
                    self._close_in_flight_since[sym] = time.time()

                    res = await self.loop.run_in_executor(
                        None, self.execution_agent.flatten_position, sym, qty, side
                    )

                    if res and res.get("status") in ("error", "rejected"):
                        self.logger.error(f"Emergency Flatten: Execution FAILED for {sym}. Keeping internal tracking active.")
                        # Remove from close_in_flight so sync can attempt adoption/recovery if needed
                        self._close_in_flight.pop(sym, None)
                        self._close_in_flight_since.pop(sym, None)
                        continue

                    # Flatten order accepted — keep internal position under _close_in_flight
                    # so GatewayUserPosition (netSize=0) + GatewayUserTrade attribute realized PnL cleanly.
                    broker_active_syms.discard(sym)
                    broker_flattened_syms.add(sym)

                    # Give the broker time to process before checking next position
                    await asyncio.sleep(2)
            elif broker_positions is not None:
                self.logger.info("Emergency Flatten: No open positions found on broker — already flat. Cancelling any lingering orders.")
                for sym in self.official_symbols:
                    try:
                        await self.loop.run_in_executor(None, self.client.cancel_open_orders_for_symbol, sym)
                    except Exception as ce:
                        self.logger.warning(f"Failed to cancel open orders for {sym} during emergency flatten: {ce}")
            else:
                fetch_failed = True
                self.logger.critical(
                    "Emergency Flatten: Failed to fetch positions from broker REST (get_positions returned None) — "
                    "falling back to emergency market order flatten on internal active positions!"
                )
                async with self._lock:
                    internal_to_flatten = [
                        (sym, list(plist))
                        for sym, plist in self.active_positions.items()
                        if plist and not self._close_in_flight.get(sym)
                    ]
                for sym, plist in internal_to_flatten:
                    tot_qty = sum(p["qty"] for p in plist)
                    side = plist[0]["side"]
                    self.logger.critical(f"Emergency Flatten (Internal Fallback): Flattening {sym} {side} {tot_qty}x")
                    self._close_in_flight[sym] = True
                    self._close_in_flight_since[sym] = time.time()
                    await self.loop.run_in_executor(None, self.execution_agent.flatten_position, sym, tot_qty, side)
        except Exception as e:
            fetch_failed = True
            self.logger.error(f"Emergency Flatten: Failed to fetch/flatten broker positions: {e}")

        # 2. Safety: Clear all internal tracking ONLY for those that actually closed
        # or were not found on the broker.
        if fetch_failed:
            self.logger.critical("Emergency Flatten: Aborting internal state wipe because broker positions fetch failed!")
            return

        async with self._lock:
            # Clear internal tracking ONLY for symbols already flat on the broker.
            # Symbols with a flatten order in flight remain guarded until GatewayUserPosition
            # (netSize=0) + GatewayUserTrade attribute their realized PnL (with 30s auto-expiry fallback).
            for sym in self.official_symbols:
                if sym not in broker_active_syms and sym not in broker_flattened_syms:
                    self.active_positions[sym] = []
                    with self.risk_agent._lock:
                        self.risk_agent.open_positions[sym] = []
                    self._latest_broker_positions_snapshot[sym] = []
                    self._close_in_flight.pop(sym, None)
                    self._close_in_flight_since.pop(sym, None)
                    self._recently_closed[sym] = time.time()
            self._latest_broker_positions_at = time.time()
            self.risk_agent.current_pos_count = sum(len(v) for v in self.active_positions.values())
            self.logger.info("Emergency Flatten: Internal position tracking reconciliation complete.")
        self._schedule_status_write()

    def _on_market_data_threadsafe(self, message):
        self.loop.call_soon_threadsafe(self.on_global_market_data, message)

    def _on_order_update_threadsafe(self, message):
        self.loop.call_soon_threadsafe(self.on_order_update, message)

    def _on_reconnect_threadsafe(self):
        self.loop.call_soon_threadsafe(self.on_reconnect)

    def _on_user_event_threadsafe(self, event_type, payload):
        # Hop from the SignalR background thread back into the asyncio loop.
        self.loop.call_soon_threadsafe(self.on_user_event, event_type, payload)

    def on_user_event(self, event_type, payload):
        """Routes GatewayUser* payloads from the user hub to the right handlers.

        - GatewayUserPosition: same shape used by on_order_update (netSize/averagePrice).
          Reuse it so we get position open/close handling for free.
        - GatewayUserTrade: a fill event with realizedPnl on closes — drives session_pnl.
        - GatewayUserOrder: order lifecycle (new/working/filled/cancelled). Filled stop/TP
          types still flow through on_order_update for close attribution.
        - GatewayUserAccount: balance/MLL/dailyPnL refresh in real time.
        """
        try:
            if not isinstance(payload, dict):
                return
            self.logger.debug(f"[USER-HUB] {event_type}: {payload!r}")

            # ProjectX user-hub wraps every payload in {action: int, data: {...}}. Unwrap.
            data = payload.get("data") if isinstance(payload.get("data"), dict) else payload

            if event_type == "GatewayUserPosition":
                # Translate ProjectX position shape to the {netSize, averagePrice, contractId}
                # shape that on_order_update expects. ProjectX uses type: 0=flat, 1=long, 2=short.
                pos_type = data.get("type")
                size = self._safe_float(data.get("size"), 0)
                if pos_type == 1:       # Long
                    net_size = abs(size)
                elif pos_type == 2:     # Short
                    net_size = -abs(size)
                else:                   # 0 = flat / closed
                    net_size = 0
                translated = dict(data)
                translated["netSize"] = net_size

                # Immediate broker snapshot sync: update _latest_broker_positions_snapshot
                # so the dashboard telemetry never serves a stale 30s-old REST snapshot
                # when real-time SignalR user-hub position events arrive.
                sym = self._resolve_symbol_from_payload(data)
                if sym:
                    if net_size == 0:
                        self._latest_broker_positions_snapshot[sym] = []
                    else:
                        side = "Buy" if net_size > 0 else "Sell"
                        avg_price = self._safe_float(data.get("averagePrice") or data.get("entryPrice") or 0.0)
                        self._latest_broker_positions_snapshot[sym] = [{"side": side, "qty": abs(int(net_size)), "entry": avg_price}]
                    self._latest_broker_positions_at = time.time()

                self.on_order_update(translated)
                return

            if event_type == "GatewayUserOrder":
                self.on_order_update(data)
                return

            if event_type == "GatewayUserTrade":
                # ProjectX uses `profitAndLoss` (not realizedPnl). Accumulate only for closing
                # legs — opens have no P&L field. Fees/commissions are booked into account
                # balance separately by the broker.
                pnl = data.get("profitAndLoss")
                if pnl is None:
                    pnl = data.get("realizedPnl") or data.get("pnl")
                if pnl is not None:
                    pnl_f = float(pnl)
                    # Stash the PnL per symbol so the position-close handler
                    # (triggered by netSize=0) can attribute it to the trade log.
                    sym = self._resolve_symbol_from_payload(data)
                    if sym:
                        self._pending_fill_pnl[sym] = self._pending_fill_pnl.get(sym, 0.0) + pnl_f
                        # Retroactive race check: If sym is flat internally and a recent trade
                        # closed within the last 5 seconds with 0.0 PnL, update that trade now so
                        # pnl is not stranded in _pending_fill_pnl.
                        if not self.active_positions.get(sym) and self._recent_trades:
                            fill_oid = str(data.get("orderId") or "")
                            matched_trade = None
                            for cand in reversed(self._recent_trades[-5:]):
                                if cand.get("symbol") != sym or float(cand.get("pnl", 0.0)) != 0.0:
                                    continue
                                if fill_oid and cand.get("order_id") and str(cand.get("order_id")) == fill_oid:
                                    matched_trade = cand
                                    break
                                if matched_trade is None:
                                    matched_trade = cand
                            if matched_trade is not None and time.time() - self._recently_closed.get(sym, 0) < 5.0:
                                self.logger.info(
                                    f"[USER-HUB FILL] Retroactively attributing ${pnl_f:.2f} to recently closed {sym} trade."
                                )
                                matched_trade["pnl"] = round(pnl_f, 2)
                                self._pending_fill_pnl.pop(sym, None)
                                self._last_trade_won[sym] = pnl_f > 0
                                if pnl_f > 0:
                                    self._consecutive_losses[sym] = 0
                                elif pnl_f < 0:
                                    self._consecutive_losses[sym] = self._consecutive_losses.get(sym, 0) + 1
                                    self._notify_strategy_stop_exit(sym)
                                try:
                                    self.risk_agent.record_trade_result(pnl_f, symbol=sym)
                                except Exception as e:
                                    self.logger.debug(f"record_trade_result skipped for {sym}: {e}")
                                if matched_trade.get("trade_key"):
                                    self._persist_closed_trade_outcome(
                                        symbol=sym,
                                        closed=matched_trade,
                                        pnl=pnl_f,
                                        order_type=matched_trade.get("type", "retroactive_fill"),
                                        session=matched_trade.get("session"),
                                    )

                    prev_fills_pnl = self._session_pnl_from_fills
                    self._session_pnl_from_fills += pnl_f
                    # Avoid double-counting if GatewayUserAccount already updated current_balance
                    # relative to _session_opening_balance.
                    if self._session_opening_balance and self._session_opening_balance > 0 and self.risk_agent.current_balance > 0:
                        bal_delta = float(self.risk_agent.current_balance - self._session_opening_balance)
                        if abs(bal_delta - prev_fills_pnl) <= abs(bal_delta - self._session_pnl_from_fills):
                            self.risk_agent.add_realized_pnl(pnl_f)
                    else:
                        self.risk_agent.add_realized_pnl(pnl_f)
                    self.logger.info(
                        f"[USER-HUB FILL] symbol={data.get('contractId')} realized={pnl} "
                        f"(running session_pnl_from_fills={self._session_pnl_from_fills:.2f})"
                    )
                return

            if event_type == "GatewayUserAccount":
                balance = data.get("balance")
                # Only use dailyPnL/dayPnL if the broker actually sends it.
                # ProjectX does NOT include this field — if absent we must NOT
                # zero-out current_daily_pnl (which is accumulated from fills above).
                pnl = data.get("dailyPnL") if "dailyPnL" in data else data.get("dayPnL")
                mll = data.get("minimumBalance")
                if balance is not None:
                    try:
                        open_pos_count = sum(len(v) for v in self.active_positions.values())
                    except Exception:
                        open_pos_count = 0
                    self.risk_agent.update_account_state(
                        float(balance), pnl, open_pos_count, minimum_balance=mll, session_opening_balance=self._session_opening_balance
                    )
                    self._schedule_status_write()
                return
        except Exception as e:
            self.logger.error(f"on_user_event({event_type}) error: {e}")

    def on_reconnect(self):
        self.logger.warning("SignalR reconnected! Execution ledger remains lead clock.")
        # FIX #H14: We no longer force a sync from laggy positions API on reconnect.
        # SignalR will stream catch-up updates if any orders were filled during downtime.

    def on_global_market_data(self, message):
        try:
            if not message:
                return
            data = message[0] if isinstance(message, list) else message
            if not isinstance(data, dict):
                return
            self._ticks_received += 1
            if self._ticks_received % 1000 == 0:
                self.logger.info(f"Market Data Flowing: {self._ticks_received} ticks processed for {self.official_symbols}")
            raw_sym = data.get("symbol")
            resolved_official = self._resolve_symbol_from_payload(data)
            for sym in self.official_symbols:
                matched = False
                if resolved_official:
                    matched = (resolved_official == sym)
                elif raw_sym:
                    matched = (raw_sym == sym or self._match_symbol_variant(sym, raw_sym))
                elif len(self.official_symbols) == 1:
                    matched = True

                if matched:
                    target_data = data
                    if raw_sym != sym:
                        target_data = dict(data)
                        target_data["symbol"] = sym
                    self.data_agents[sym].handle_market_data(target_data)
                    t_price = self._safe_float(target_data.get("price"))
                    if t_price > 0 and self.active_positions.get(sym):
                        self._update_active_positions_excursions({sym: (t_price, t_price, t_price)})
        except Exception as e:
            self.logger.error(f"Broadcasting error: {e}")

    def on_order_update(self, message):
        try:
            if not message:
                return
            data = message[0] if isinstance(message, list) else message
            if not isinstance(data, dict):
                return

            symbol = self._resolve_symbol_from_payload(data)
            if not symbol:
                return

            # Broker sends status/type as ints on the user-hub; normalize defensively
            # so downstream string comparisons work and we never crash on .lower().
            order_status = normalize_status(data.get("status"))
            order_type = normalize_order_type(data.get("type"))

            # Use the most accurate execution data available
            fill_price = self._safe_float(
                data.get("fillPrice") or data.get("averagePrice") or data.get("entryPrice") or data.get("price") or 0.0
            )

            # Case A: Explicit NetSize update (Fastest broker-side ledger)
            if "netSize" in data:
                net_size = self._safe_float(data.get("netSize"))
                if abs(net_size) < 1e-9:
                    self.logger.info(f"[LEDGER CLOSE] {symbol} flattened (netSize=0)")
                    # Peek (do NOT pop) the stash so Future B below can read it.
                    # The order-based Future A (Case B) may also try to pop it.
                    # Whichever future runs first wins; the other sees an empty position
                    # ledger and exits cleanly without double-recording.
                    pnl_peek = self._pending_fill_pnl.get(symbol)
                    if pnl_peek is None:
                        realized_pnl = data.get("realizedPnl") or data.get("pnl")
                        pnl_peek = float(realized_pnl) if realized_pnl is not None else None
                    # Determine close type from stashed PnL context
                    close_type = "ledger_sync"
                    if pnl_peek is not None:
                        close_type = "tp" if pnl_peek > 0 else ("sl" if pnl_peek < 0 else "scratch")
                    # Notify the active strategy of stop-loss exits so always-in
                    # strategies can apply their fresh-close re-entry gate.
                    if close_type == "sl":
                        self._notify_strategy_stop_exit(symbol)
                    # Keep close-in-flight active until _handle_position_close_update acquires
                    # self._lock and drains active_positions (prevents 250ms TOCTOU stop race).
                    self._close_in_flight[symbol] = True
                    self._close_in_flight_since[symbol] = time.time()
                    # Signal to Case B that a drain is already queued for this symbol
                    # so it does not schedule a second _handle_position_close_update.
                    self._drain_queued[symbol] = True
                    asyncio.ensure_future(self._handle_position_close_update(symbol, close_type, None, drain=True))
                    return

                side = "Buy" if net_size > 0 else "Sell"
                qty = int(abs(net_size))
                self.logger.info(f"[LEDGER UPDATE] {symbol} netSize: {net_size} @ {fill_price}")
                asyncio.ensure_future(self._handle_position_open_update(symbol, side, qty, fill_price))
                return

            # Case B: Standard Order Fill (including partial fills status=7; never status=3 cancelled)
            if order_status in ("filled", "completed", "executed", "partially_filled", "2", "7"):
                is_partial_status = order_status in ("partially_filled", "7")
                cum_raw = data.get("fillVolume")
                if cum_raw is None:
                    cum_raw = data.get("filledQty")
                if cum_raw is None and not is_partial_status:
                    cum_raw = data.get("size")
                cum_qty = int(abs(self._safe_float(cum_raw or 0)))
                oid_raw = data.get("id") or data.get("orderId")
                if oid_raw is not None:
                    if not hasattr(self, "_order_filled_qty"):
                        self._order_filled_qty = {}
                    oid_key = str(oid_raw)
                    prev_filled = int(self._order_filled_qty.get(oid_key, 0))
                    qty = max(0, cum_qty - prev_filled)
                    self._order_filled_qty[oid_key] = max(prev_filled, cum_qty)
                    if len(self._order_filled_qty) > 1000:
                        for old_k in list(self._order_filled_qty.keys())[:500]:
                            self._order_filled_qty.pop(old_k, None)
                else:
                    qty = cum_qty

                if qty > 0:
                    self.logger.info(f"[LEDGER FILL] {symbol} {order_status.upper()} {qty}x @ {fill_price}")
                    # 4=stoploss, 1=takeprofit/limit
                    if order_type in ("stoploss", "takeprofit", "stop", "limit", "4", "1"):
                        # Verify that if we know the order side and have an active position,
                        # a limit/stop fill is actually on the opposite (closing) side.
                        active_pos = self.active_positions.get(symbol, [])
                        raw_side = data.get("side")
                        if active_pos and raw_side is not None:
                            order_side = "Buy" if str(raw_side).strip().lower() in ("0", "buy", "long") else "Sell"
                            if order_side == active_pos[0].get("side"):
                                self.logger.debug(
                                    f"[LEDGER] {symbol} same-side {order_type} fill ({order_side}) — treating as entry, not close."
                                )
                                return
                        # Skip if a drain (netSize=0) is already queued for this symbol.
                        # The drain future will handle attribution — scheduling a second
                        # future here would race to pop _pending_fill_pnl and record pnl=0.
                        if self._drain_queued.pop(symbol, False):
                            self.logger.debug(
                                f"[LEDGER] {symbol} drain already queued by netSize=0 — "
                                "skipping redundant order-fill close future."
                            )
                        else:
                            realized_pnl = data.get("realizedPnl") or data.get("pnl") or data.get("profitAndLoss")
                            pnl = float(realized_pnl) if realized_pnl is not None else None
                            if order_type in ("stoploss", "stop", "4"):
                                self._notify_strategy_stop_exit(symbol)
                            asyncio.ensure_future(self._handle_position_close_update(symbol, order_type, pnl, drain=False, qty=qty))
                    else:
                        # For entries, we rely on 'netSize' for authoritative sync.
                        self.logger.debug(f"[LEDGER] fill recorded for {symbol}. Awaiting netSize update.")
        except Exception as e:
            self.logger.error(f"Order update handling error: {e}")

    async def _select_trade_intent(self, sym, strategy_intent, zone_intent, route_context=None):
        def selection_snapshot(candidate):
            snapshot = dict(candidate)
            metadata = dict(snapshot.get("metadata") or {})
            metadata.pop("selection", None)
            snapshot["metadata"] = metadata
            return snapshot

        candidates = []
        if strategy_intent:
            strategy_candidates = []
            try:
                strategy_agent = self.strategy_agents.get(sym)
                engine = getattr(strategy_agent, "engine", None)
                strategy_candidates = [dict(item) for item in getattr(engine, "_last_candidates", []) or []]
            except Exception:
                strategy_candidates = []
            candidates.extend(strategy_candidates or [dict(strategy_intent)])
        if self.feature_flags.get("zone_engine_enabled", False) and zone_intent:
            candidates.append(dict(zone_intent))
        if not candidates:
            return None
        selected_candidate = (
            dict(zone_intent)
            if (self.feature_flags.get("zone_engine_enabled", False) and zone_intent)
            else (dict(strategy_intent) if strategy_intent else candidates[0])
        )
        selected = selection_snapshot(selected_candidate)
        selected.setdefault("metadata", {})
        selected["metadata"]["selection"] = {
            "authority": "deterministic",
            "candidates": [selection_snapshot(candidate) for candidate in candidates],
            "selected_strategy": selected.get("strategy_name", "unknown"),
            "zone_engine_enabled": bool(self.feature_flags.get("zone_engine_enabled", False)),
        }
        return selected

    async def _dispatch_trade(self, sym, trade_intent, route_context):
        try:
            discovery_decision = {}
            try:
                discovery_decision = self.discovery.evaluate_trade(sym, self.data_agents[sym].get_latest_data(), trade_intent) or {}
                self._last_discovery_decisions[sym] = discovery_decision
                if discovery_decision.get("reason") not in (None, "", "neutral", "discovery_disabled"):
                    match = discovery_decision.get("matched_failure") or {}
                    similarity = float(match.get("similarity", 0.0)) if match else 0.0
                    self.logger.info(
                        f"[DISCOVERY ADVISORY] {sym}: {discovery_decision.get('reason')} "
                        f"(side={match.get('side')} strat={match.get('strategy_name')} sim={similarity:.3f}) - ADVISORY ONLY"
                    )
            except Exception as disc_exc:
                self.logger.warning(f"[DISCOVERY ADVISORY] skipped for {sym}: {disc_exc}", exc_info=True)

            discovery_decision = dict(discovery_decision or {})
            discovery_decision["allowed"] = True
            discovery_decision["quantity_multiplier"] = 1.0
            discovery_decision["enforced"] = True
            trade_intent = dict(trade_intent)
            trade_intent.setdefault("metadata", {})
            trade_intent["metadata"]["discovery"] = discovery_decision
            result = await self.loop.run_in_executor(None, self.trade_router.route_trade, trade_intent, route_context)
            if isinstance(result, RouterRejection):
                self.health_monitor.record_signal(sym, False, result.reason)
                self.logger.info(f"[ROUTER REJECT] {sym}: {result.stage} {result.reason}")
                return

            self.health_monitor.record_signal(sym, True, result.reason)
            self.logger.info(f"[EXECUTION] {sym}: {result.result}")
            if result.accepted:
                opened_at_dt = datetime.now(timezone.utc)
                execution_price = (
                    result.result.get("price")
                    or result.result.get("fillPrice")
                    or result.result.get("avgPrice")
                    or route_context.get("current_price", 0)
                )
                order_id = result.result.get("orderId") or result.result.get("id")
                trade_key = build_trade_key(sym, order_id, opened_at_dt.isoformat())
                async with self._lock:
                    self._record_position_open(
                        sym,
                        trade_intent,
                        result.quantity,
                        execution_price,
                        order_id=order_id,
                        trade_key=trade_key,
                    )
                self._capture_accepted_trade_context(
                    sym=sym,
                    trade_intent=trade_intent,
                    quantity=result.quantity,
                    execution_price=execution_price,
                    order_id=order_id,
                    trade_key=trade_key,
                    opened_at_dt=opened_at_dt,
                    route_context=route_context,
                    discovery_decision=discovery_decision,
                )
        except Exception as exec_err:
            self.logger.error(f"Order failed to send for {sym}: {exec_err}")
            self.risk_agent.set_pending(sym, False)

    def _notify_strategy_stop_exit(self, symbol):
        """Forward a broker-side stop-loss fill to the active strategies for
        this symbol so they can apply post-SL gating (e.g. always-in strategies
        wait for a fresh higher-timeframe close before re-entering)."""
        try:
            strategy_agent = self.strategy_agents.get(symbol)
            engine = getattr(strategy_agent, "engine", None) if strategy_agent else None
            strategies = getattr(engine, "strategies", None) if engine else None
            if not strategies:
                return
            now = datetime.now(timezone.utc)
            for strategy in strategies:
                hook = getattr(strategy, "notify_stop_exit", None)
                if callable(hook):
                    try:
                        hook(now)
                    except Exception as inner:
                        self.logger.debug(f"notify_stop_exit failed for {strategy.name}: {inner}")
        except Exception as exc:
            self.logger.debug(f"_notify_strategy_stop_exit error for {symbol}: {exc}")

    def _record_position_open(self, sym, trade_intent, quantity, execution_price, order_id=None, trade_key=None):
        self.last_signal_time[sym] = time.time()
        self._drain_queued.pop(sym, None)

        # FIX: Net-Position Awareness & Desync Prevention
        # If we have existing positions for this symbol, check if they are the opposite side.
        if self.active_positions.get(sym):
            existing_side = self.active_positions[sym][0]["side"]
            new_side = trade_intent["side"]

            if existing_side != new_side:
                # Calculate total existing quantity
                existing_qty = sum(p["qty"] for p in self.active_positions[sym])
                self.logger.info(f"[NET POSITION] {sym}: {new_side} {quantity}x against existing {existing_side} {existing_qty}x.")

                if quantity >= existing_qty:
                    # Closing the entire existing position and possibly flipping
                    self.logger.info(f"[NET POSITION] {sym}: Entire {existing_side} position closed.")
                    while self.active_positions.get(sym):
                        closed = self.clear_position(sym)
                        if closed:
                            # Calculate realized PnL for the flipped close
                            try:
                                tick_size = float(self.risk_agent.get_tick_size(sym) or 0.25)
                                tick_value = float(self.risk_agent.get_tick_value(sym) or 0.5)
                                entry_price = float(closed.get("entry", 0.0))
                                cside = closed.get("side")
                                cqty = int(closed.get("qty", 0))

                                price_diff = (execution_price - entry_price) if cside == "Buy" else (entry_price - execution_price)
                                ticks = price_diff / tick_size if tick_size > 0 else 0.0
                                calc_pnl = ticks * tick_value * cqty

                                self.logger.info(f"[NET POSITION CLOSE] {sym} closed {cside} {cqty}x @ {execution_price} (entry={entry_price}, calc_pnl=${calc_pnl:.2f})")
                                self._persist_closed_trade_outcome(symbol=sym, closed=closed, pnl=calc_pnl, order_type="flip")
                                self.risk_agent.record_trade_result(calc_pnl, symbol=sym)
                                self.telemetry.record_trade_close(
                                    sym,
                                    closed.get("strategy_name", "unknown"),
                                    cside,
                                    cqty,
                                    calc_pnl,
                                    "flip",
                                )
                                self._record_chart_trade_event("exit", sym, closed, pnl=calc_pnl, order_type="flip")
                                now_utc = datetime.now(timezone.utc)
                                self._recent_trades.append({
                                    "symbol": sym,
                                    "strategy": closed.get("strategy_name", "unknown"),
                                    "side": cside,
                                    "qty": cqty,
                                    "entry": entry_price,
                                    "exit": execution_price,
                                    "pnl": calc_pnl,
                                    "type": "flip",
                                    "time": now_utc.strftime("%m/%d %H:%M"),
                                    "timestamp": now_utc.isoformat(),
                                    "order_type": "flip",
                                })
                                self._last_trade_won[sym] = calc_pnl > 0
                                self._recently_closed[sym] = time.time()
                                self._pending_fill_pnl.pop(sym, None)
                            except Exception as e:
                                self.logger.error(f"Failed to calculate and persist flip close outcome for {sym}: {e}")

                    remaining_qty = quantity - existing_qty
                    if remaining_qty > 0:
                        self.logger.info(f"[NET POSITION] {sym}: Flipping to {new_side} {remaining_qty}x.")
                        pos_entry = {
                            "side": new_side,
                            "time": time.time(),
                            "qty": remaining_qty,
                            "entry": execution_price,
                            "strategy_name": trade_intent.get("strategy_name", "unknown"),
                            "order_id": order_id,
                            "trade_key": trade_key,
                            "account_id": str(getattr(self, "account_id", None)) if getattr(self, "account_id", None) else None,
                            "sl_ticks": trade_intent.get("sl_ticks"),
                            "tp_ticks": trade_intent.get("tp_ticks"),
                            "reason": trade_intent.get("reason", ""),
                            "metadata": dict(trade_intent.get("metadata") or {}),
                        }
                        self.active_positions[sym].append(pos_entry)
                        self.risk_agent.record_position_open(sym, new_side, remaining_qty, execution_price, order_id=order_id)
                        self._record_chart_trade_event("entry", sym, pos_entry)
                else:
                    # Partial close of existing position
                    self.logger.info(f"[NET POSITION] {sym}: Partial close of {existing_side} position by {quantity}x.")
                    closed = self.clear_position(sym, qty=quantity)
                    if closed:
                        try:
                            tick_size = float(self.risk_agent.get_tick_size(sym) or 0.25)
                            tick_value = float(self.risk_agent.get_tick_value(sym) or 0.5)
                            entry_price = float(closed.get("entry", 0.0))
                            cside = closed.get("side")
                            cqty = int(closed.get("qty", 0))

                            price_diff = (execution_price - entry_price) if cside == "Buy" else (entry_price - execution_price)
                            ticks = price_diff / tick_size if tick_size > 0 else 0.0
                            calc_pnl = ticks * tick_value * cqty

                            self.logger.info(f"[NET POSITION CLOSE PARTIAL] {sym} closed {cside} {cqty}x @ {execution_price} (entry={entry_price}, calc_pnl=${calc_pnl:.2f})")
                            self._persist_closed_trade_outcome(symbol=sym, closed=closed, pnl=calc_pnl, order_type="flip_partial")
                            self.risk_agent.record_trade_result(calc_pnl, symbol=sym)
                            self.telemetry.record_trade_close(
                                sym,
                                closed.get("strategy_name", "unknown"),
                                cside,
                                cqty,
                                calc_pnl,
                                "flip_partial",
                            )
                            self._record_chart_trade_event("exit", sym, closed, pnl=calc_pnl, order_type="flip_partial")
                            now_utc = datetime.now(timezone.utc)
                            self._recent_trades.append({
                                "symbol": sym,
                                "strategy": closed.get("strategy_name", "unknown"),
                                "side": cside,
                                "qty": cqty,
                                "entry": entry_price,
                                "exit": execution_price,
                                "pnl": calc_pnl,
                                "type": "flip_partial",
                                "time": now_utc.strftime("%m/%d %H:%M"),
                                "timestamp": now_utc.isoformat(),
                                "order_type": "flip_partial",
                            })
                            self._last_trade_won[sym] = calc_pnl > 0
                            self._recently_closed[sym] = time.time()
                            self._pending_fill_pnl.pop(sym, None)
                        except Exception as e:
                            self.logger.error(f"Failed to calculate and persist flip partial close outcome for {sym}: {e}")

                self.risk_agent.set_pending(sym, False)
                return

        # Standard open (no existing position or same side)
        if sym not in self.active_positions:
            self.active_positions[sym] = []
        from src.utils.session_utils import get_trade_session
        trade_session = get_trade_session(time.time())
        pos_entry = {
            "side": trade_intent["side"],
            "time": time.time(),
            "qty": quantity,
            "entry": execution_price,
            "strategy_name": trade_intent.get("strategy_name", "unknown"),
            "order_id": order_id,
            "trade_key": trade_key,
            "account_id": str(getattr(self, "account_id", None)) if getattr(self, "account_id", None) else None,
            "session": trade_session,
            "sl_ticks": trade_intent.get("sl_ticks"),
            "tp_ticks": trade_intent.get("tp_ticks"),
            "reason": trade_intent.get("reason", ""),
            "metadata": dict(trade_intent.get("metadata") or {}),
        }
        self.active_positions[sym].append(pos_entry)
        self.risk_agent.record_position_open(sym, trade_intent["side"], quantity, execution_price, order_id=order_id)
        self._record_chart_trade_event("entry", sym, pos_entry)
        self.telemetry.record_trade_open(
            sym,
            trade_intent.get("strategy_name", "unknown"),
            trade_intent["side"],
            quantity,
            execution_price,
            metadata=trade_intent.get("metadata", {}),
        )

    def _capture_accepted_trade_context(
        self,
        *,
        sym,
        trade_intent,
        quantity,
        execution_price,
        order_id,
        trade_key,
        opened_at_dt,
        route_context,
        discovery_decision,
    ):
        try:
            if not (self.database and self.database.connected):
                return
            selection = (trade_intent.get("metadata") or {}).get("selection") or {}
            candidates = selection.get("candidates") or [dict(trade_intent)]
            blender_metadata = (trade_intent.get("metadata") or {}).get("blender") or {}
            capture_cfg = (self.learning_review_config or {}).get("capture", {})
            latest_bars = self.data_agents[sym].get_latest_data()
            chart_artifacts = capture_trade_context_artifacts(
                bars=latest_bars,
                trade_key=trade_key,
                symbol=sym,
                opened_at=opened_at_dt,
                config=capture_cfg,
            )
            from src.utils.session_utils import get_trade_session
            session_name = get_trade_session(opened_at_dt)
            ctx_kwargs = {
                "trade_key": trade_key,
                "symbol": sym,
                "strategy_name": trade_intent.get("strategy_name", "unknown"),
                "side": trade_intent.get("side", ""),
                "quantity": int(quantity),
                "order_id": order_id,
                "entry_price": float(execution_price or 0.0),
                "opened_at": opened_at_dt.isoformat(),
                "session": session_name,
                "strategy_params": self._strategy_params_snapshot(sym),
                "selected_signal": dict(trade_intent),
                "candidates": [dict(c) for c in candidates],
                "blender_metadata": dict(blender_metadata),
                "route_context": dict(route_context or {}),
                "discovery_context": dict(discovery_decision or {}),
                "account_snapshot": self._account_risk_snapshot(),
                "chart_artifacts": chart_artifacts,
            }
            loop = getattr(self, "loop", None)
            db_exec = getattr(self, "_db_executor", None)
            if loop is not None and hasattr(loop, "is_running") and loop.is_running() and db_exec is not None:
                loop.run_in_executor(db_exec, lambda kw=ctx_kwargs: self.database.persist_trade_context(**kw))
            else:
                self.database.persist_trade_context(**ctx_kwargs)
        except Exception as exc:
            self.logger.warning("[LEARNING CAPTURE] trade context NOT persisted for %s: %s", sym, exc, exc_info=True)

    def _record_chart_trade_event(self, kind: str, symbol: str, position: dict, *, pnl: float | None = None, order_type: str | None = None) -> None:
        ts = position.get("time", time.time())
        if isinstance(ts, (int, float)):
            event_time = int(ts)
        else:
            event_time = chart_time(ts)
        price = position.get("entry")
        if kind == "exit":
            latest = self.data_agents.get(symbol).get_latest_data() if symbol in self.data_agents else None
            if latest is not None and not latest.empty:
                price = float(latest.iloc[-1]["close"])
            event_time = int(time.time())
        event = {
            "time": event_time,
            "kind": kind,
            "symbol": symbol,
            "side": position.get("side"),
            "qty": int(position.get("qty", 0) or 0),
            "price": float(price or 0.0),
            "strategy_name": position.get("strategy_name", "unknown"),
            "order_id": position.get("order_id"),
            "trade_key": position.get("trade_key"),
            "sl_ticks": position.get("sl_ticks"),
            "tp_ticks": position.get("tp_ticks"),
            "reason": position.get("reason", ""),
            "setup_id": ((position.get("metadata") or {}).get("setup_id")),
            "pnl": pnl,
            "order_type": order_type,
        }
        self._chart_trade_events.append({k: v for k, v in event.items() if v is not None})
        self._chart_trade_events = self._chart_trade_events[-120:]

    def _strategy_params_snapshot(self, sym):
        try:
            agent = self.strategy_agents.get(sym)
            engine = getattr(agent, "engine", None)
            if engine is not None:
                return {
                    "config": getattr(engine, "config", {}),
                    "bracket": getattr(engine, "bracket", {}),
                    "blender": getattr(engine, "blender_config", {}),
                }
        except Exception as e:
            self.logger.debug(f"_strategy_params_snapshot fallback for {sym}: {e}")
        return dict(self.strategy_config or {})

    def _account_risk_snapshot(self):
        return {
            "account_id": str(getattr(self, "account_id", "")) if getattr(self, "account_id", None) is not None else None,
            "balance": float(self.risk_agent.current_balance),
            "daily_pnl": float(self.risk_agent.current_daily_pnl),
            "unrealized_pnl": float(self.risk_agent.unrealized_pnl),
            "peak_balance": float(self.risk_agent.peak_balance),
            "max_daily_loss": float(self.risk_agent.max_daily_loss),
            "max_drawdown": float(self.risk_agent.max_drawdown),
            "open_positions": {
                symbol: [dict(pos) for pos in positions]
                for symbol, positions in self.risk_agent.open_positions.items()
            },
            "risk_per_trade_pct": self.routing_config.get("risk_per_trade_pct"),
        }

    def get_chart_payload(self, symbol: str, frame: str = "1m", limit: int = 500) -> dict[str, Any]:
        symbol = str(symbol or "").strip().upper()
        if not symbol:
            symbol = self.official_symbols[0] if self.official_symbols else ""
        if symbol not in self.data_agents:
            raise ValueError(f"Unknown chart symbol '{symbol}'")

        normalized_frame = self._validate_chart_frame(frame)

        limit = max(50, min(int(limit or 500), 2000))
        data_agent = self.data_agents[symbol]
        chart_frame = data_agent.get_chart_frame(normalized_frame, limit=limit)
        strategy_data = data_agent.get_latest_data()
        overlays = self.strategy_agents[symbol].get_chart_overlays(strategy_data) if symbol in self.strategy_agents and not strategy_data.empty else {
            "lines": [],
            "levels": [],
            "markers": [],
            "events": [],
            "state": {},
        }
        position_overlays = self._chart_position_overlays(symbol, chart_frame)
        return {
            "symbol": symbol,
            "frame": normalized_frame,
            "frames": ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
            "symbols": list(self.official_symbols),
            "bars": bars_to_chart_records(chart_frame, limit=limit),
            "overlays": {
                "lines": list(overlays.get("lines") or []) + position_overlays["lines"],
                "levels": list(overlays.get("levels") or []) + position_overlays["levels"],
                "markers": list(overlays.get("markers") or []) + position_overlays["markers"],
                "events": list(overlays.get("events") or []) + self._chart_events_for_symbol(symbol),
                "state": overlays.get("state") or {},
            },
            "active_positions": list(self.active_positions.get(symbol, [])),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    def get_chart_live_payload(self, symbol: str, frame: str = "1m") -> dict[str, Any]:
        symbol = str(symbol or "").strip().upper()
        if not symbol:
            symbol = self.official_symbols[0] if self.official_symbols else ""
        if symbol not in self.data_agents:
            raise ValueError(f"Unknown chart symbol '{symbol}'")

        normalized_frame = self._validate_chart_frame(frame)
        chart_frame = self.data_agents[symbol].get_chart_frame(normalized_frame, limit=3)
        records = bars_to_chart_records(chart_frame, limit=3)
        latest = records[-1] if records else None
        previous = records[-2] if len(records) > 1 else None
        return {
            "symbol": symbol,
            "frame": normalized_frame,
            "bar": latest,
            "previous_bar": previous,
            "bar_count": len(chart_frame) if chart_frame is not None else 0,
            "last_price": latest.get("close") if latest else None,
            "last_tick": self.client.last_tick_at,
            "signalr_connected": self.client.is_connected,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    def _validate_chart_frame(self, frame: str) -> str:
        normalized_frame = normalize_frame(frame)
        allowed_frames = {"1m", "5m", "15m", "30m", "1h", "4h", "1d"}
        if normalized_frame not in allowed_frames:
            raise ValueError(f"Unsupported chart frame '{frame}'")
        return normalized_frame

    def _chart_events_for_symbol(self, symbol: str) -> list[dict[str, Any]]:
        return [dict(event) for event in self._chart_trade_events if event.get("symbol") == symbol][-80:]

    def _chart_position_overlays(self, symbol: str, chart_frame: pd.DataFrame) -> dict[str, list]:
        positions = list(self.active_positions.get(symbol, []))
        if chart_frame is None or chart_frame.empty:
            start_time = end_time = int(time.time())
        else:
            start_time = chart_time(chart_frame.iloc[0].get("timestamp"))
            end_time = chart_time(chart_frame.iloc[-1].get("timestamp"))
        levels = []
        markers = []
        tick_size = float(self.risk_agent.get_tick_size(symbol) or 0.25)
        for idx, pos in enumerate(positions):
            side = str(pos.get("side") or "")
            entry = float(pos.get("entry") or 0.0)
            color = "#10b981" if side == "Buy" else "#ef4444"
            levels.append(
                {
                    "id": f"active_entry_{idx}",
                    "label": f"{side} {int(pos.get('qty', 0) or 0)} Entry",
                    "price": entry,
                    "start_time": start_time,
                    "end_time": end_time,
                    "color": color,
                    "style": "solid",
                }
            )
            sl_ticks = int(pos.get("sl_ticks") or 0)
            tp_ticks = int(pos.get("tp_ticks") or 0)
            if sl_ticks > 0:
                stop = entry - sl_ticks * tick_size if side == "Buy" else entry + sl_ticks * tick_size
                levels.append(
                    {
                        "id": f"active_stop_{idx}",
                        "label": "Stop",
                        "price": stop,
                        "start_time": start_time,
                        "end_time": end_time,
                        "color": "#ef4444",
                        "style": "dotted",
                    }
                )
            if tp_ticks > 0:
                target = entry + tp_ticks * tick_size if side == "Buy" else entry - tp_ticks * tick_size
                levels.append(
                    {
                        "id": f"active_target_{idx}",
                        "label": "Target",
                        "price": target,
                        "start_time": start_time,
                        "end_time": end_time,
                        "color": "#10b981",
                        "style": "dotted",
                    }
                )
            event_time = int(pos.get("time", time.time())) if isinstance(pos.get("time"), (int, float)) else chart_time(pos.get("time"))
            markers.append(
                {
                    "time": event_time,
                    "price": entry,
                    "kind": "active_position",
                    "label": f"{side} {int(pos.get('qty', 0) or 0)}",
                    "color": color,
                }
            )
        return {"lines": [], "levels": levels, "markers": markers}

    async def _handle_position_open_update(self, symbol, side, qty, avg_price):
        async with self._lock:
            internal_positions = self.active_positions.get(symbol, [])
            internal_qty = sum(p.get("qty", 0) for p in internal_positions)
            internal_side = internal_positions[0].get("side") if internal_positions else None
            if internal_qty == qty and (not internal_positions or internal_side == side):
                if internal_positions and avg_price > 0 and abs(float(internal_positions[0].get("entry", 0.0)) - avg_price) > 1e-6:
                    internal_positions[0]["entry"] = avg_price
                    side_norm = str(internal_positions[0].get("side") or "Buy").strip().lower()
                    if side_norm in ("buy", "long", "0"):
                        internal_positions[0]["mfe_price"] = max(self._safe_float(internal_positions[0].get("mfe_price"), avg_price), avg_price)
                        internal_positions[0]["mae_price"] = min(self._safe_float(internal_positions[0].get("mae_price"), avg_price), avg_price)
                    else:
                        internal_positions[0]["mfe_price"] = min(self._safe_float(internal_positions[0].get("mfe_price"), avg_price), avg_price)
                        internal_positions[0]["mae_price"] = max(self._safe_float(internal_positions[0].get("mae_price"), avg_price), avg_price)
                    with self.risk_agent._lock:
                        if self.risk_agent.open_positions.get(symbol):
                            self.risk_agent.open_positions[symbol][0]["entry"] = Decimal(str(avg_price))
                return

            # FIX Bug 1: Close-in-flight guard — broker streams intermediate GatewayUserPosition
            # events during SL/TP bracket fills that carry non-zero netSize values (e.g. size=34
            # when averaging two fill legs). If we're in the middle of closing this symbol, these
            # events must be ignored — they are NOT new open positions. The flag is set in
            # _handle_position_close_update and cleared when broker confirms netSize=0.
            if self._close_in_flight.get(symbol):
                self.logger.debug(
                    f"[POSITION SYNC SKIP] {symbol}: broker pushed {side} {qty}x but "
                    f"a close is in-flight — ignoring intermediate netSize update."
                )
                return

            # FIX Bug 1 (entry race): GatewayUserPosition can arrive BEFORE the HTTP response
            # for place_order() returns. If a pending order is in-flight the broker netSize
            # update is about the order we just sent — _record_position_open will handle it
            # authoritatively once the HTTP response comes back with the real order ID.
            if symbol in self.risk_agent.pending_orders:
                self.logger.debug(
                    f"[POSITION SYNC SKIP] {symbol}: broker pushed {side} {qty}x but "
                    f"an entry order is in-flight — deferring to _record_position_open."
                )
                return

            if internal_positions and internal_side == side and qty < internal_qty:
                delta = internal_qty - qty
                self.logger.info(f"[POSITION SYNC PARTIAL REDUCE] {symbol} reduced by {delta}x (from {internal_qty}x to {qty}x)")
                self.clear_position(symbol, qty=delta)
                if hasattr(self, "_latest_broker_positions_snapshot"):
                    self._latest_broker_positions_snapshot[symbol] = [{"side": side, "qty": qty, "entry": avg_price}]
                    self._latest_broker_positions_at = time.time()
                self._schedule_status_write()
                return

            preserved_meta = dict(internal_positions[0]) if (internal_positions and internal_side == side) else {}
            while self.active_positions.get(symbol):
                self.clear_position(symbol)
            if symbol not in self.active_positions:
                self.active_positions[symbol] = []
            from src.utils.session_utils import get_trade_session
            entry_time = preserved_meta.get("time", time.time())
            entry_oid = preserved_meta.get("order_id")
            pos_entry = {
                "side": side,
                "time": entry_time,
                "qty": qty,
                "entry": avg_price if avg_price > 0 else preserved_meta.get("entry", 0.0),
                "strategy_name": preserved_meta.get("strategy_name", "position_update_sync"),
                "order_id": entry_oid,
                "trade_key": preserved_meta.get("trade_key") or build_trade_key(symbol, entry_oid, entry_time),
                "session": preserved_meta.get("session") or get_trade_session(time.time()),
                "sl_ticks": preserved_meta.get("sl_ticks"),
                "tp_ticks": preserved_meta.get("tp_ticks"),
                "reason": preserved_meta.get("reason", ""),
                "metadata": preserved_meta.get("metadata", {}),
                "account_id": str(getattr(self, "account_id", None)) if getattr(self, "account_id", None) else None,
            }
            self.active_positions[symbol].append(pos_entry)
            self.risk_agent.record_position_open(symbol, side, qty, pos_entry["entry"], order_id=pos_entry.get("order_id"))
            self.last_signal_time[symbol] = time.time()
            if hasattr(self, "_latest_broker_positions_snapshot"):
                self._latest_broker_positions_snapshot[symbol] = [{"side": side, "qty": qty, "entry": pos_entry["entry"]}]
                self._latest_broker_positions_at = time.time()
            self.logger.info(f"[POSITION SYNC OPEN] {symbol} {side} {qty}x @ {pos_entry['entry']}")
        self._schedule_status_write()

    async def _handle_position_close_update(self, symbol, order_type, pnl, drain=False, qty: int | None = None):
        """Serializes position clear + attribution updates behind the supervisor lock."""
        # Set _close_in_flight BEFORE sleeping so intermediate GatewayUserPosition events
        # arriving during the 250ms fill-PnL wait cannot re-open a phantom position.
        if not drain:
            self._close_in_flight[symbol] = True
            self._close_in_flight_since[symbol] = time.time()

        if pnl is None and symbol not in self._pending_fill_pnl:
            for _ in range(5):
                await asyncio.sleep(0.05)
                if symbol in self._pending_fill_pnl:
                    break

        async with self._lock:
            if hasattr(self, "_drain_queued"):
                self._drain_queued.pop(symbol, None)
            if pnl is None:
                pnl = self._pending_fill_pnl.pop(symbol, None)
            else:
                self._pending_fill_pnl.pop(symbol, None)

            if not drain:
                self._close_in_flight[symbol] = True
                self._close_in_flight_since[symbol] = time.time()
            else:
                # If draining, we are confirmed flat — ensure flag is cleared.
                self._close_in_flight.pop(symbol, None)
                self._close_in_flight_since.pop(symbol, None)
            # Record the closure time to prevent sync-re-adoption "ghosts" and enforce post-close cooldown
            now_close_ts = time.time()
            self._recently_closed[symbol] = now_close_ts
            if hasattr(self, "last_signal_time"):
                self.last_signal_time[symbol] = now_close_ts

            # FIX #H4: DRAIN ALL tracked positions ONLY if broker says we are flat (netSize=0)
            # If it's just an individual order fill, we only clear the specific quantity.
            closed = None
            if drain:
                if hasattr(self, "_stop_placement_in_flight"):
                    self._stop_placement_in_flight.pop(symbol, None)
                if hasattr(self, "_latest_broker_positions_snapshot"):
                    self._latest_broker_positions_snapshot[symbol] = []
                    self._latest_broker_positions_at = time.time()
                while self.active_positions.get(symbol):
                    item = self.clear_position(symbol)
                    if not item:
                        break
                    if closed is None:
                        closed = item # Attribute PnL to the primary (oldest) position
                # Sweep any orphaned bracket orders on confirmed flat
                try:
                    asyncio.ensure_future(
                        self.loop.run_in_executor(None, self.client.cancel_open_orders_for_symbol, symbol)
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to schedule orphaned order cancellation for {symbol}: {e}")
            else:
                closed = self.clear_position(symbol, qty=qty)
                if not self.active_positions.get(symbol):
                    self._close_in_flight.pop(symbol, None)
                    self._close_in_flight_since.pop(symbol, None)
                    if hasattr(self, "_stop_placement_in_flight"):
                        self._stop_placement_in_flight.pop(symbol, None)
                    if hasattr(self, "_latest_broker_positions_snapshot"):
                        self._latest_broker_positions_snapshot[symbol] = []
                        self._latest_broker_positions_at = time.time()
                    try:
                        asyncio.ensure_future(
                            self.loop.run_in_executor(None, self.client.cancel_open_orders_for_symbol, symbol)
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to schedule orphaned order cancellation for {symbol}: {e}")
                else:
                    # Partial exit left remaining open contracts: release _close_in_flight
                    # immediately so bracket/news guards remain active for the open remainder.
                    self._close_in_flight.pop(symbol, None)
                    self._close_in_flight_since.pop(symbol, None)

            still_open = bool(self.active_positions.get(symbol))
            if closed:
                _pnl = float(pnl) if pnl is not None else 0.0
                try:
                    self.risk_agent.record_trade_result(_pnl, symbol=symbol)
                except TypeError:
                    self.risk_agent.record_trade_result(_pnl)
                if not still_open:
                    self._last_trade_won[symbol] = _pnl > 0
                    # Track consecutive losses for streak cooldown escalation.
                    # Breakeven trades (_pnl == 0) are neutral — they don't reset
                    # the streak but also don't increment it.
                    if _pnl > 0:
                        self._consecutive_losses[symbol] = 0
                    elif _pnl < 0:
                        self._consecutive_losses[symbol] = self._consecutive_losses.get(symbol, 0) + 1
                        streak = self._consecutive_losses[symbol]
                        if streak >= self.MAX_CONSECUTIVE_LOSSES:
                            self.logger.warning(
                                f"LOSS STREAK: {symbol} hit {streak} consecutive losses — "
                                "tracking for post-session review; no intraday rollback applied."
                            )
                self.telemetry.record_trade_close(
                    symbol,
                    closed.get("strategy_name", "unknown"),
                    closed.get("side", ""),
                    closed.get("qty", 0),
                    _pnl,
                    order_type,
                )
                self._record_chart_trade_event("exit", symbol, closed, pnl=_pnl, order_type=order_type)
                from src.utils.session_utils import get_trade_session
                now_utc = datetime.now(timezone.utc)
                trade_session = (closed or {}).get("session") or get_trade_session((closed or {}).get("time") or now_utc)
                persist_closed = dict(closed)
                if still_open:
                    base_tk = persist_closed.get("trade_key") or build_trade_key(symbol, persist_closed.get("order_id"), persist_closed.get("time"))
                    persist_closed["trade_key"] = f"{base_tk}:part:{int(now_close_ts * 1000)}"
                self._persist_closed_trade_outcome(symbol=symbol, closed=persist_closed, pnl=_pnl, order_type=order_type, session=trade_session)
                excursions = self._compute_position_excursions(symbol, closed)
                self._recent_trades.append({
                    "symbol": symbol,
                    "trade_key": closed.get("trade_key"),
                    "order_id": closed.get("order_id"),
                    "pnl": round(_pnl, 2),
                    "type": order_type,
                    "side": closed.get("side", ""),
                    "entry": closed.get("entry"),
                    "qty": closed.get("qty"),
                    "strategy_name": closed.get("strategy_name", "unknown"),
                    "sl_ticks": closed.get("sl_ticks"),
                    "tp_ticks": closed.get("tp_ticks"),
                    "mfe_price": closed.get("mfe_price"),
                    "mae_price": closed.get("mae_price"),
                    "reason": closed.get("reason", ""),
                    "metadata": closed.get("metadata", {}),
                    "time": now_utc.strftime("%m/%d %H:%M"),
                    "timestamp": now_utc.isoformat(),
                    "session": trade_session,
                    **excursions,
                })
                self._recent_trades = self._recent_trades[-20:]
                self._feed_live_outcome_to_discovery(
                    symbol=symbol,
                    closed=closed,
                    pnl=_pnl,
                    order_type=order_type,
                )
                self._score_mega_vision_decision(
                    symbol=symbol,
                    pnl=_pnl,
                    closed=closed,
                )
            else:
                self.logger.debug(f"Position for {symbol} already cleared; skipping redundant attribution.")

        self._schedule_status_write()

    def _update_active_positions_excursions(self, bar_extremes: dict[str, tuple[float, float, float]]) -> None:
        if not bar_extremes:
            return
        for sym, pos_list in list(self.active_positions.items()):
            extremes = bar_extremes.get(sym)
            if not extremes or not pos_list:
                continue
            high_p, low_p, close_p = extremes
            for pos in pos_list:
                entry = self._safe_float(pos.get("entry"))
                if entry <= 0:
                    continue
                mfe_prev = self._safe_float(pos.get("mfe_price"), default=entry)
                mae_prev = self._safe_float(pos.get("mae_price"), default=entry)
                side_norm = str(pos.get("side") or "Buy").strip().lower()
                if side_norm in ("buy", "long", "0"):
                    pos["mfe_price"] = max(mfe_prev, high_p, close_p)
                    pos["mae_price"] = min(mae_prev, low_p, close_p)
                else:
                    pos["mfe_price"] = min(mfe_prev, low_p, close_p)
                    pos["mae_price"] = max(mae_prev, high_p, close_p)

    def _compute_position_excursions(self, symbol: str, pos: dict) -> dict[str, float]:
        if "mfe_ticks" in pos and "mae_ticks" in pos and pos.get("mfe_ticks") is not None:
            result = {
                "mfe_price": self._safe_float(pos.get("mfe_price")),
                "mae_price": self._safe_float(pos.get("mae_price")),
                "mfe_ticks": self._safe_float(pos.get("mfe_ticks")),
                "mae_ticks": self._safe_float(pos.get("mae_ticks")),
            }
            if "mfe_r" in pos and pos.get("mfe_r") is not None:
                result["mfe_r"] = self._safe_float(pos.get("mfe_r"))
            if "mae_r" in pos and pos.get("mae_r") is not None:
                result["mae_r"] = self._safe_float(pos.get("mae_r"))
            return result
        entry = self._safe_float(pos.get("entry"))
        if entry <= 0:
            return {}
        tick_fn = getattr(getattr(self, "risk_agent", None), "get_tick_size", None)
        raw_tick = tick_fn(symbol) if callable(tick_fn) else 0.25
        tick_size = self._safe_float(raw_tick, default=0.25)
        if tick_size <= 0:
            tick_size = 0.25
        mfe_price = self._safe_float(pos.get("mfe_price"), default=entry)
        mae_price = self._safe_float(pos.get("mae_price"), default=entry)
        side_norm = str(pos.get("side") or "Buy").strip().lower()
        if side_norm in ("buy", "long", "0"):
            mfe_ticks = max(0.0, (mfe_price - entry) / tick_size)
            mae_ticks = max(0.0, (entry - mae_price) / tick_size)
        else:
            mfe_ticks = max(0.0, (entry - mfe_price) / tick_size)
            mae_ticks = max(0.0, (mae_price - entry) / tick_size)
        result = {
            "mfe_price": round(mfe_price, 6),
            "mae_price": round(mae_price, 6),
            "mfe_ticks": round(mfe_ticks, 2),
            "mae_ticks": round(mae_ticks, 2),
        }
        sl_ticks = self._safe_float(pos.get("sl_ticks"))
        if sl_ticks > 0:
            result["mfe_r"] = round(mfe_ticks / sl_ticks, 2)
            result["mae_r"] = round(mae_ticks / sl_ticks, 2)
        return result

    def _persist_closed_trade_outcome(self, *, symbol, closed, pnl, order_type, session: str | None = None):
        try:
            if not (self.database and self.database.connected):
                return
            from src.utils.session_utils import get_trade_session
            trade_key = closed.get("trade_key") or build_trade_key(symbol, closed.get("order_id"), closed.get("time"))
            closed_at = datetime.now(timezone.utc).isoformat()
            account_id = closed.get("account_id") or (str(getattr(self, "account_id", None)) if getattr(self, "account_id", None) else None)
            trade_session = session or closed.get("session") or get_trade_session(closed.get("time") or closed_at)
            closed_copy = dict(closed)
            excursions = self._compute_position_excursions(symbol, closed_copy)
            if not excursions:
                for k in ("mfe_ticks", "mae_ticks", "mfe_r", "mae_r", "mfe_price", "mae_price"):
                    if k in closed_copy and closed_copy[k] is not None:
                        excursions[k] = closed_copy[k]
            else:
                for k in ("mfe_ticks", "mae_ticks", "mfe_r", "mae_r", "mfe_price", "mae_price"):
                    if k not in excursions and k in closed_copy and closed_copy[k] is not None:
                        excursions[k] = closed_copy[k]
            closed_copy.update(excursions)
            outcome_kwargs = {
                "trade_key": trade_key,
                "symbol": symbol,
                "order_id": closed.get("order_id"),
                "account_id": account_id,
                "session": trade_session,
                "pnl": float(pnl),
                "close_type": str(order_type or ""),
                "closed_at": closed_at,
                "outcome": {
                    "account_id": account_id,
                    "closed_position": closed_copy,
                    "session": trade_session,
                    "pnl": float(pnl),
                    "close_type": str(order_type or ""),
                    "closed_at": closed_at,
                    **excursions,
                },
            }
            loop = getattr(self, "loop", None)
            db_exec = getattr(self, "_db_executor", None)
            if loop is not None and hasattr(loop, "is_running") and loop.is_running() and db_exec is not None:
                loop.run_in_executor(db_exec, lambda kw=outcome_kwargs: self.database.persist_trade_outcome(**kw))
            else:
                self.database.persist_trade_outcome(**outcome_kwargs)
        except Exception as exc:
            self.logger.warning("[LEARNING OUTCOME] trade outcome NOT persisted for %s: %s", symbol, exc, exc_info=True)

    def _feed_live_outcome_to_discovery(self, *, symbol, closed, pnl, order_type):
        """Persist a closed live trade into the discovery experience store so
        future discovery decisions learn from actual live outcomes (not just
        optimizer trials). Best-effort — failures here never break the close
        attribution path.
        """
        try:
            experience = getattr(self.discovery, "experience", None)
            data_agent = self.data_agents.get(symbol)
            if experience is None or data_agent is None:
                return
            latest = data_agent.get_latest_data()
            if latest is None or latest.empty:
                return
            window = int((self.discovery_config or {}).get("embedding_window", 120))
            market_embedding = embed_market(latest.tail(window))
            experience.persist_live_outcome(
                instrument=symbol,
                market_embedding=market_embedding,
                side=closed.get("side"),
                strategy_name=closed.get("strategy_name"),
                pnl=float(pnl),
                order_type=str(order_type or ""),
            )
        except Exception as e:
            self.logger.debug(f"[DISCOVERY FEEDBACK] skipped for {symbol}: {e}")

    def _score_mega_vision_decision(self, *, symbol, pnl, closed):
        """Score mega vision's shadow decision against the actual trade outcome
        and append to a JSONL log so the system can learn which LLM calls were
        right. Best-effort — never breaks the close attribution path."""
        try:
            decision = self._last_mega_vision_decisions.get(symbol)
            if not decision:
                return
            won = pnl > 0
            agreed = decision.get("agreed", False)
            no_pick = decision.get("no_pick", False)
            # Advisory-only scorecard: no_pick is not a trade veto.
            if no_pick:
                correct = not won
            elif agreed:
                correct = won      # agree was right if trade won
            else:
                correct = not won  # disagree was right if trade lost
            verdict = "CORRECT" if correct else "WRONG"
            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "pnl": round(pnl, 2),
                "won": won,
                "llm_agreed": agreed,
                "llm_no_pick": no_pick,
                "llm_pick": decision.get("strategy_name"),
                "native_pick": decision.get("native_pick"),
                "confidence": decision.get("confidence", 0),
                "reasoning": decision.get("reasoning", ""),
                "correct": correct,
            }
            log_path = Path("logs/telemetry/mega_vision_scorecard.jsonl")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            self.logger.info(
                "[MEGA VISION SCORE] %s %s — LLM %s (pnl=%.2f, confidence=%.2f)",
                symbol, verdict,
                "no_pick" if no_pick else ("agreed" if agreed else "disagreed"),
                pnl, decision.get("confidence", 0),
            )
        except Exception as e:
            self.logger.debug(f"[MEGA VISION SCORE] skipped for {symbol}: {e}")

    async def _ensure_protective_stop_for_position(
        self,
        symbol: str,
        side: str,
        qty: int,
        entry_price: float,
        sl_ticks: int | None = None,
    ) -> bool:
        """Verifies an active stop-loss order exists for the given position; places an emergency stop if missing."""
        try:
            now_ts = time.time()
            async with self._lock:
                if self._close_in_flight.get(symbol) or (now_ts - self._recently_closed.get(symbol, 0.0)) < 5.0:
                    self.logger.debug(
                        f"[BRACKET GUARD] {symbol}: Skipping protective stop check — position is closing or just closed."
                    )
                    return True
                in_flight_dict = getattr(self, "_stop_placement_in_flight", {})
                if (now_ts - in_flight_dict.get(symbol, 0.0)) < 60.0:
                    self.logger.info(
                        f"[BRACKET GUARD] {symbol}: Emergency stop was placed {int(now_ts - in_flight_dict[symbol])}s ago — "
                        "skipping duplicate placement while order is in flight/propagation."
                    )
                    return True
                had_active_pos = bool(self.active_positions.get(symbol))
                resolved_sl_ticks = sl_ticks
                if not resolved_sl_ticks and had_active_pos:
                    resolved_sl_ticks = self.active_positions[symbol][0].get("sl_ticks")
                if hasattr(self, "_stop_placement_in_flight"):
                    self._stop_placement_in_flight[symbol] = now_ts

            try:
                has_stop = await self.loop.run_in_executor(None, self.client.has_working_stop_order, symbol, side, qty)
            except TypeError:
                has_stop = await self.loop.run_in_executor(None, self.client.has_working_stop_order, symbol, side)
            if has_stop is True:
                async with self._lock:
                    if hasattr(self, "_stop_placement_in_flight"):
                        self._stop_placement_in_flight.pop(symbol, None)
                self.logger.debug(f"[BRACKET GUARD] Active stop order confirmed for {symbol} {side} {qty}x.")
                return True
            if has_stop is None:
                async with self._lock:
                    if hasattr(self, "_stop_placement_in_flight"):
                        self._stop_placement_in_flight.pop(symbol, None)
                self.logger.warning(f"[BRACKET GUARD] Could not verify stop orders for {symbol} (API error/timeout). Skipping placement.")
                return False

            # TOCTOU re-check under lock after REST call returns: ensure position did not close while awaiting
            async with self._lock:
                if (
                    self._close_in_flight.get(symbol)
                    or (time.time() - self._recently_closed.get(symbol, 0.0)) < 5.0
                    or (had_active_pos and not self.active_positions.get(symbol))
                ):
                    if hasattr(self, "_stop_placement_in_flight"):
                        self._stop_placement_in_flight.pop(symbol, None)
                    self.logger.info(
                        f"[BRACKET GUARD] {symbol}: Position closed while stop check was in flight — aborting emergency stop."
                    )
                    return True

            base_sym = StrategyAgent._extract_base_static(symbol)
            bracket = get_symbol_brackets(self.stack_config).get(base_sym, {})
            effective_sl_ticks = int(round(float(resolved_sl_ticks))) if (resolved_sl_ticks and float(resolved_sl_ticks) > 0) else int(bracket.get("sl", 40))
            try:
                tick_size = float(self.risk_agent.get_tick_size(symbol) or bracket.get("tick_size", 0.25) or 0.25)
            except Exception:
                tick_size = float(bracket.get("tick_size", 0.25) or 0.25)

            pos_side_norm = str(side).strip().lower()
            is_long = pos_side_norm in ("buy", "long", "0")
            stop_side = "Sell" if is_long else "Buy"

            entry_p = float(entry_price or 0.0)
            if entry_p <= 0:
                dagent = self.data_agents.get(symbol)
                latest = dagent.get_latest_data() if dagent else None
                if latest is not None and not latest.empty:
                    entry_p = float(latest.iloc[-1]["close"])

            if entry_p <= 0:
                async with self._lock:
                    if hasattr(self, "_stop_placement_in_flight"):
                        self._stop_placement_in_flight.pop(symbol, None)
                self.logger.critical(f"[BRACKET GUARD] Cannot place emergency stop for {symbol}: entry_price is 0 or unavailable.")
                return False

            stop_distance = float(effective_sl_ticks) * tick_size
            stop_price = (entry_p - stop_distance) if is_long else (entry_p + stop_distance)

            ts_str = f"{tick_size:.6f}".rstrip("0")
            prec = len(ts_str.split(".")[1]) if "." in ts_str else 0
            stop_price = round(round(stop_price / tick_size) * tick_size, prec)

            # Check if live price has already breached stop_price (a Stop order would be rejected by broker).
            cur_market_price = 0.0
            dagent = self.data_agents.get(symbol)
            latest_df = dagent.get_latest_data() if dagent else None
            if latest_df is not None and not latest_df.empty:
                cur_market_price = float(latest_df.iloc[-1]["close"] or 0.0)
            stop_already_breached = cur_market_price > 0 and (
                (is_long and cur_market_price <= stop_price)
                or (not is_long and cur_market_price >= stop_price)
            )
            if stop_already_breached:
                self.logger.critical(
                    f"[BRACKET GUARD] Market price ({cur_market_price}) already breached stop ({stop_price}) for {symbol} {side} {qty}x — "
                    "flattening at market instead of placing rejected Stop order."
                )
                self._close_in_flight[symbol] = True
                self._close_in_flight_since[symbol] = time.time()
                await self.loop.run_in_executor(None, self.execution_agent.flatten_position, symbol, qty, side)
                return True

            self.logger.critical(
                f"[BRACKET GUARD] ⚠️ UNPROTECTED POSITION DETECTED! {symbol} {side} {qty}x @ {entry_p} has NO active stop-loss! "
                f"Placing emergency protective stop: {stop_side} {qty}x Stop @ {stop_price} ({effective_sl_ticks} ticks)..."
            )

            exec_agent = getattr(self, "execution_agent", None)
            exec_mode = getattr(exec_agent, "mode", None) or os.getenv("MODE", "simulation").strip().lower()
            if exec_mode == "live" and os.getenv("ACTUAL_LIVE_TRADING", "false").strip().lower() != "true":
                async with self._lock:
                    if hasattr(self, "_stop_placement_in_flight"):
                        self._stop_placement_in_flight.pop(symbol, None)
                self.logger.critical(
                    f"[BRACKET GUARD] LIVE MODE BLOCK: ACTUAL_LIVE_TRADING is not 'true' — blocking emergency stop for {symbol}."
                )
                return False

            res = await self.loop.run_in_executor(
                None,
                self.client.place_order,
                symbol,
                qty,
                stop_side,
                "Stop",
                None,  # limit_price
                stop_price,  # stop_price
            )

            if isinstance(res, dict) and (res.get("status") in ("error", "rejected") or not res.get("success", True)):
                async with self._lock:
                    if hasattr(self, "_stop_placement_in_flight"):
                        self._stop_placement_in_flight.pop(symbol, None)
                self.logger.critical(f"[BRACKET GUARD] ❌ FAILED to place emergency protective stop for {symbol}: {res}")
                return False

            async with self._lock:
                if hasattr(self, "_stop_placement_in_flight"):
                    self._stop_placement_in_flight[symbol] = time.time()

            self.logger.warning(
                f"[BRACKET GUARD] ✅ Emergency protective stop placed for {symbol}: {stop_side} {qty}x Stop @ {stop_price} (res={res})"
            )
            return True
        except Exception as e:
            async with self._lock:
                if hasattr(self, "_stop_placement_in_flight"):
                    self._stop_placement_in_flight.pop(symbol, None)
            self.logger.exception(f"[BRACKET GUARD] Error while checking/placing protective stop for {symbol}: {e}")
            return False

    async def _sync_positions(self, broker_positions):
        """
        Reconciles internal active_positions with the broker's truth to recover from missed SignalR updates.
        Trusts /Position/searchOpen as the source of truth after a 60s grace for newly-opened positions.
        """
        if broker_positions is None:
            self.logger.warning("Position sync skipped: broker_positions is None (API error/timeout).")
            return
        positions_to_protect: list[tuple[str, str, int, float]] = []
        try:
            # Group broker positions by resolved symbol
            broker_state = {}
            for p in broker_positions:
                sym = self._resolve_symbol_from_payload(p)
                if not sym:
                    continue
                qty, side = self._extract_qty_side(p)
                avg_price = self._safe_float(p.get("averagePrice") or p.get("entryPrice") or p.get("avgPrice") or 0.0)
                if qty > 0:
                    broker_state.setdefault(sym, []).append({"side": side, "qty": qty, "entry": avg_price})

            # Cache broker truth for the dashboard (status JSON pulls from this snapshot
            # so the UI never shows stale internal cache).
            self._latest_broker_positions_snapshot = {
                sym: list(positions) for sym, positions in broker_state.items()
            }
            self._latest_broker_positions_at = time.time()

            async with self._lock:
                symbols_to_check = set(self.official_symbols) | set(self.active_positions.keys()) | set(broker_state.keys())
                for sym in symbols_to_check:
                    internal_pos_list = self.active_positions.get(sym, [])
                    if not internal_pos_list and sym not in broker_state:
                        continue

                    internal_total_qty = sum(p["qty"] for p in internal_pos_list)
                    broker_total_qty = sum(p["qty"] for p in broker_state.get(sym, []))

                    internal_side = internal_pos_list[0].get("side") if internal_pos_list else None
                    broker_side = broker_state.get(sym, [{}])[0].get("side") if broker_state.get(sym) else None

                    if internal_total_qty == broker_total_qty:
                        if broker_total_qty == 0 or internal_side == broker_side:
                            # Clear desync timer if they match completely
                            self._desync_start_time.pop(sym, None)
                            continue
                        else:
                            # Quantities match but side is inverted! Broker is authoritative.
                            self.logger.critical(
                                f"POSITION SIDE INVERSION for {sym}: Internal={internal_side} {internal_total_qty}x, "
                                f"Broker={broker_side} {broker_total_qty}x. Forcing reconciliation to broker side!"
                            )
                            while self.active_positions.get(sym):
                                self.clear_position(sym)
                            self._recently_closed[sym] = time.time()
                            self._desync_start_time.pop(sym, None)
                            for bp in broker_state.get(sym, []):
                                if sym not in self.active_positions:
                                    self.active_positions[sym] = []
                                pos_entry = {
                                    "side": bp["side"],
                                    "time": time.time(),
                                    "qty": bp["qty"],
                                    "entry": bp["entry"],
                                    "strategy_name": "broker_reconcile_side_flip",
                                    "account_id": str(getattr(self, "account_id", None)) if getattr(self, "account_id", None) else None,
                                }
                                self.active_positions[sym].append(pos_entry)
                                self.risk_agent.record_position_open(sym, bp["side"], bp["qty"], bp["entry"])
                                positions_to_protect.append((sym, bp["side"], bp["qty"], bp["entry"]))
                            continue

                    now = time.time()

                    # CASE 1: Internal > Broker (Bot thinks trade is open, Broker doesn't see it)
                    # /Position/searchOpen has no measurable search lag — old 15-min zombie
                    # preservation was over-conservative and caused chronic dashboard staleness.
                    # New rule: 60s grace ONLY for positions opened in the last 60s (broker may
                    # not have indexed them yet); past that, broker is authoritative — rebuild
                    # internal state from broker truth.
                    if internal_total_qty > broker_total_qty:
                        most_recent_open = max((p.get("time", 0) for p in internal_pos_list), default=0)
                        age_of_newest = now - most_recent_open
                        grace_threshold = 30 if broker_total_qty == 0 else 60
                        if age_of_newest < grace_threshold:
                            self.logger.info(
                                f"POSITION DISCREPANCY for {sym} (Internal={internal_total_qty}, Broker={broker_total_qty}), "
                                f"newest internal entry is {int(age_of_newest)}s old — granting {grace_threshold}s grace before reconciling."
                            )
                            continue

                        self.logger.warning(
                            f"BROKER RECONCILE for {sym}: Internal={internal_total_qty}, Broker={broker_total_qty}. "
                            f"Trusting broker truth and rebuilding internal state."
                        )
                        preserved_meta = dict(internal_pos_list[0]) if internal_pos_list else {}
                        # Wipe internal state for this symbol, then rebuild from broker (if any).
                        while self.active_positions.get(sym):
                            self.clear_position(sym)
                        self._recently_closed[sym] = time.time()
                        self._desync_start_time.pop(sym, None)
                        has_matching_broker_pos = any(
                            self._match_symbol_variant(sym, b_sym)
                            for b_sym in broker_state
                        )
                        if not broker_state.get(sym) and not has_matching_broker_pos:
                            try:
                                asyncio.ensure_future(
                                    self.loop.run_in_executor(None, self.client.cancel_open_orders_for_symbol, sym)
                                )
                            except Exception as e:
                                self.logger.warning(f"Failed to cancel orphaned orders for {sym} during reconcile: {e}")
                        for bp in broker_state.get(sym, []):
                            if sym not in self.active_positions:
                                self.active_positions[sym] = []
                            same_side = preserved_meta.get("side") == bp["side"]
                            pos_entry = {
                                "side": bp["side"],
                                "time": preserved_meta.get("time", time.time()) if same_side else time.time(),
                                "qty": bp["qty"],
                                "entry": bp["entry"],
                                "strategy_name": preserved_meta.get("strategy_name", "broker_reconcile") if same_side else "broker_reconcile",
                                "order_id": preserved_meta.get("order_id") if same_side else None,
                                "trade_key": preserved_meta.get("trade_key") if same_side else None,
                                "session": preserved_meta.get("session") if same_side else None,
                                "account_id": str(getattr(self, "account_id", None)) if getattr(self, "account_id", None) else None,
                                "sl_ticks": preserved_meta.get("sl_ticks") if same_side else None,
                                "tp_ticks": preserved_meta.get("tp_ticks") if same_side else None,
                                "mfe_price": preserved_meta.get("mfe_price", bp["entry"]) if same_side else bp["entry"],
                                "mae_price": preserved_meta.get("mae_price", bp["entry"]) if same_side else bp["entry"],
                                "reason": preserved_meta.get("reason", "") if same_side else "",
                                "metadata": preserved_meta.get("metadata", {}) if same_side else {},
                            }
                            self.active_positions[sym].append(pos_entry)
                            self.risk_agent.record_position_open(sym, bp["side"], bp["qty"], bp["entry"])
                        continue

                    # CASE 2: Broker > Internal (Orphan found or missed fill)
                    # We adopt these after a 60-second persistence check to ensure it's not a lagging report of a closed trade.

                    # FIX: RECENTLY CLOSED GRACE — If we just closed this symbol internally,
                    # do NOT re-adopt it even if the broker still reports it (API index lag).
                    last_closed = self._recently_closed.get(sym, 0)
                    if now - last_closed < 60:
                        self.logger.info(f"ORPHAN for {sym} ignored: Recently closed internally ({int(now - last_closed)}s ago).")
                        continue

                    desync_since = self._desync_start_time.get(sym)
                    if desync_since is None:
                        self._desync_start_time[sym] = now
                        self.logger.warning(
                            f"ORPHAN DETECTED for {sym}: Broker={broker_total_qty}, Internal={internal_total_qty}. "
                            f"Waiting 60s for persistence check before adoption."
                        )
                        continue

                    if now - desync_since < 60:
                        self.logger.info(f"ORPHAN for {sym} still showing. Persistence: {int(now - desync_since)}s/60s.")
                        continue

                    self.logger.warning(
                        f"ORPHAN PERSISTED for {sym}: Broker={broker_total_qty}, Internal={internal_total_qty}. "
                        f"Forcing sync to acquire unmanaged broker position."
                    )

                    preserved_meta = dict(internal_pos_list[0]) if internal_pos_list else {}
                    # Reset internal state to match broker's truth
                    while self.active_positions.get(sym):
                        self.clear_position(sym)

                    self._desync_start_time.pop(sym, None)

                    for bp in broker_state.get(sym, []):
                        if sym not in self.active_positions:
                            self.active_positions[sym] = []
                        same_side = preserved_meta.get("side") == bp["side"]
                        pos_entry = {
                            "side": bp["side"],
                            "time": preserved_meta.get("time", time.time()) if same_side else time.time(),
                            "qty": bp["qty"],
                            "entry": bp["entry"],
                            "strategy_name": preserved_meta.get("strategy_name", "sync_recovery") if same_side else "sync_recovery",
                            "order_id": preserved_meta.get("order_id") if same_side else None,
                            "trade_key": preserved_meta.get("trade_key") if same_side else None,
                            "session": preserved_meta.get("session") if same_side else None,
                            "account_id": str(getattr(self, "account_id", None)) if getattr(self, "account_id", None) else None,
                            "sl_ticks": preserved_meta.get("sl_ticks") if same_side else None,
                            "tp_ticks": preserved_meta.get("tp_ticks") if same_side else None,
                            "mfe_price": preserved_meta.get("mfe_price", bp["entry"]) if same_side else bp["entry"],
                            "mae_price": preserved_meta.get("mae_price", bp["entry"]) if same_side else bp["entry"],
                            "reason": preserved_meta.get("reason", "") if same_side else "",
                            "metadata": preserved_meta.get("metadata", {}) if same_side else {},
                        }
                        self.active_positions[sym].append(pos_entry)
                        self.risk_agent.record_position_open(sym, bp["side"], bp["qty"], bp["entry"])
                        positions_to_protect.append((sym, bp["side"], bp["qty"], bp["entry"]))

        except Exception as e:
            self.logger.exception(f"Position sync failed: {e}")

        for p_sym, p_side, p_qty, p_entry in positions_to_protect:
            await self._ensure_protective_stop_for_position(p_sym, p_side, p_qty, p_entry)

    def _match_symbol_variant(self, sym_a: str, sym_b: str) -> bool:
        """Matches symbol variants like MESZ6 with MESZ26, /MES with MES, MCLX6 with MCLX26, CON.F.US.MES.Z26, etc."""
        if not sym_a or not sym_b:
            return False
        import re
        def _norm(s: str) -> str:
            clean = str(s).lstrip("/").upper().strip()
            m_con = re.search(r'\b([A-Z0-9]+)\.([FGHJKMNQUVXZ]\d{1,2})\b', clean)
            if m_con:
                root, my = m_con.group(1), m_con.group(2)
                if root.endswith("E") and len(root) > 3:
                    root = root[:-1]
                return f"{root}{my}"
            return clean

        a = _norm(sym_a)
        b = _norm(sym_b)
        if a == b:
            return True
        # Check base symbol match if one is root: e.g. "MES" matching "MESZ6" or "MESZ26"
        known_roots = ("MES", "MCL", "MGC", "MNQ", "ES", "NQ", "CL", "GC", "YM", "MYM", "M6E", "6E", "M2K", "RTY")
        if a in known_roots and b.startswith(a):
            return True
        if b in known_roots and a.startswith(b):
            return True
        # Futures month code + year variant: MESZ26 (2-digit year) vs MESZ6 (1-digit year), including M6EZ6/6EZ6
        m_a = re.match(r"^([A-Z0-9]+[FGHJKMNQUVXZ])(\d{1,2})$", a)
        m_b = re.match(r"^([A-Z0-9]+[FGHJKMNQUVXZ])(\d{1,2})$", b)
        if m_a and m_b and m_a.group(1) == m_b.group(1):
            return m_a.group(2)[-1] == m_b.group(2)[-1]
        return False

    def _resolve_symbol_from_payload(self, payload):
        """Authoritatively resolves any broker payload (order, trade, position)
        to the supervisor's official symbol (e.g. 'MESZ6').
        Handles contractId lookups, TopstepX contractDisplayName ('MESZ26'),
        exchange display format ('/MES'), and symbol cache variations.
        """
        if not isinstance(payload, dict):
            return ""

        # 1. Authoritative contractId resolution via client cache
        cid = payload.get("contractId")
        if cid is not None:
            cid_str = str(cid)
            # Find in client cache matching an official symbol first
            for off in self.official_symbols:
                if str(self.client.symbol_to_id_cache.get(off, "")) == cid_str:
                    return off
            # Match any key in symbol_to_id_cache that corresponds to an official symbol
            for k, v in self.client.symbol_to_id_cache.items():
                if str(v) == cid_str:
                    for off in self.official_symbols:
                        if self._match_symbol_variant(k, off):
                            return off
                    return k

            # Check if official symbol or base symbol is in contractId string (e.g. CON.F.US.MES.Z26)
            for off in self.official_symbols:
                if off in cid_str or self._match_symbol_variant(off, cid_str):
                    return off
            # Special check for e.g. CON.F.US.MCLE.X26 matching MCLX6, or CON.F.US.ENQ.M26 matching NQM6
            import re
            for off in self.official_symbols:
                m_root = re.match(r"^([A-Z0-9]+?)(?:[FGHJKMNQUVXZ]\d{1,2})?$", off)
                base = m_root.group(1) if m_root else off[:3]
                if f".{base}." in cid_str or f".{base}E." in cid_str or f".E{base}." in cid_str:
                    return off

        # 2. String-based symbol resolution from payload fields
        raw_symbol = str(
            payload.get("symbol")
            or payload.get("contractDisplayName")
            or payload.get("name")
            or payload.get("contractName")
            or ""
        ).strip()

        if raw_symbol:
            clean = raw_symbol.lstrip("/")
            # Direct match
            if clean in self.official_symbols:
                return clean
            # Match against official symbols using variant matcher
            for off in self.official_symbols:
                if self._match_symbol_variant(clean, off):
                    return off
            # Match via symbol_map
            if clean in self.symbol_map:
                return self.symbol_map[clean]
            # Match via client symbol_to_id_cache
            if clean in self.client.symbol_to_id_cache:
                cached_cid = str(self.client.symbol_to_id_cache[clean])
                for off in self.official_symbols:
                    if str(self.client.symbol_to_id_cache.get(off, "")) == cached_cid:
                        return off
            for off in self.official_symbols:
                if clean == off or clean.startswith(off):
                    return off
            return clean

        return str(cid or "")

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _extract_qty_side(self, payload):
        """
        Extracts (qty, side) from broker REST /Position/searchOpen, SignalR GatewayUserPosition,
        or internal position payloads.
        TopstepX / ProjectX positions use:
          type: 0 = Flat, 1 = Long (Buy), 2 = Short (Sell)
          size: contract quantity (positive integer)
        """
        if not isinstance(payload, dict):
            return 0, "Flat"

        # 1. netSize: standard signed size (+ for Buy/Long, - for Sell/Short, 0 for flat)
        net_size_raw = payload.get("netSize")
        if net_size_raw is not None:
            net_size = self._safe_float(net_size_raw, 0.0)
            qty = int(abs(net_size))
            if qty == 0:
                return 0, "Flat"
            return qty, ("Buy" if net_size > 0 else "Sell")

        qty_raw = payload.get("size") or payload.get("quantity") or payload.get("qty") or 0
        qty = int(abs(self._safe_float(qty_raw, 0.0)))
        if qty == 0:
            return 0, "Flat"

        # 2. Check position type from broker REST /Position/searchOpen or GatewayUserPosition
        # TopstepX position types: 1 = Long (Buy), 2 = Short (Sell), 0 = Flat
        pos_type = payload.get("type")
        if pos_type is not None:
            pos_type_str = str(pos_type).strip().lower()
            if pos_type in (2, "2") or pos_type_str in ("short", "sell"):
                return qty, "Sell"
            elif pos_type in (1, "1") or pos_type_str in ("long", "buy"):
                return qty, "Buy"
            elif pos_type in (0, "0") or pos_type_str in ("flat", "closed"):
                return 0, "Flat"

        # 3. Explicit side or positionSide strings
        side_raw = str(payload.get("side") or payload.get("positionSide") or "").strip().lower()
        if side_raw in ("sell", "short", "1"):
            return qty, "Sell"
        elif side_raw in ("buy", "long", "0"):
            return qty, "Buy"

        return qty, "Buy"

    def clear_position(self, symbol, qty: int | None = None):
        if self.active_positions.get(symbol):
            # If qty is provided, we perform a partial reduction on the oldest entries.
            # This logic mirrors RiskAgent to ensure they stay in sync.
            if qty is not None:
                remaining_to_clear = int(qty)
                self.risk_agent.record_position_close(symbol, qty=remaining_to_clear)

                total_qty = 0
                weighted_entry_sum = 0.0
                sample_pos = dict(self.active_positions[symbol][0])
                while remaining_to_clear > 0 and self.active_positions[symbol]:
                    oldest = self.active_positions[symbol][0]
                    if oldest['qty'] <= remaining_to_clear:
                        popped = self.active_positions[symbol].pop(0)
                        total_qty += popped['qty']
                        weighted_entry_sum += float(popped.get('entry', 0.0)) * popped['qty']
                        remaining_to_clear -= popped['qty']
                    else:
                        oldest['qty'] -= remaining_to_clear
                        total_qty += remaining_to_clear
                        weighted_entry_sum += float(oldest.get('entry', 0.0)) * remaining_to_clear
                        remaining_to_clear = 0
                sample_pos['qty'] = total_qty
                sample_pos['entry'] = (weighted_entry_sum / total_qty) if total_qty > 0 else sample_pos.get('entry', 0.0)
                return sample_pos

            # Legacy behavior: pop the oldest entry
            closed = self.active_positions[symbol].pop(0)
            self.risk_agent.record_position_close(symbol)
            self.logger.info(f"Position cleared for {symbol}")
            return closed
        return None

    async def _check_news_guards(self):
        now = datetime.now(timezone.utc)
        news_event = self.news_agent.is_news_approaching(now, lead_minutes=2)
        if not news_event:
            return
        for sym in self.official_symbols:
            if self._close_in_flight.get(sym):
                continue
            async with self._lock:
                positions = list(self.active_positions.get(sym, []))
            if not positions:
                continue
            latest_bar = self.data_agents[sym].get_latest_data()
            if latest_bar.empty:
                continue
            curr_price = latest_bar.iloc[-1]["close"]
            tick_size = self.risk_agent.get_tick_size(sym)
            for pos in positions:
                side, entry, qty = pos["side"], pos["entry"], pos["qty"]
                points_profit = (curr_price - entry) if side == "Buy" else (entry - curr_price)
                ticks_profit = round(points_profit / tick_size)
                if ticks_profit > 30:
                    self.logger.info(f"[PRO-GUARD] Moving {sym} {side} to BE for {news_event}")
                else:
                    self.logger.warning(f"[PRO-GUARD] Flattening {sym} {side} {qty}x before {news_event}")
                    self._close_in_flight[sym] = True
                    self._close_in_flight_since[sym] = time.time()
                    res = await self.loop.run_in_executor(None, self.execution_agent.flatten_position, sym, qty, side)
                    if res and res.get("status") in ("error", "rejected"):
                        self._close_in_flight.pop(sym, None)
                        self._close_in_flight_since.pop(sym, None)

    @staticmethod
    def _json_safe(value, seen=None):
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            import math
            if math.isnan(value) or math.isinf(value):
                return None
            return value
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, (datetime, pd.Timestamp)):
            return value.isoformat()

        if seen is None:
            seen = set()

        if isinstance(value, dict):
            obj_id = id(value)
            if obj_id in seen:
                return "<circular>"
            seen.add(obj_id)
            safe = {str(key): SupervisorAgent._json_safe(item, seen) for key, item in value.items()}
            seen.remove(obj_id)
            return safe

        if isinstance(value, (list, tuple, set)):
            obj_id = id(value)
            if obj_id in seen:
                return ["<circular>"]
            seen.add(obj_id)
            safe = [SupervisorAgent._json_safe(item, seen) for item in value]
            seen.remove(obj_id)
            return safe

        if hasattr(value, "item"):
            try:
                return value.item()
            except (ValueError, TypeError, AttributeError):
                return str(value)

        return str(value)

    def _latest_prices_snapshot(self) -> dict[str, float]:
        prices = {}
        for sym, data_agent in getattr(self, "data_agents", {}).items():
            try:
                latest = data_agent.get_latest_data()
                if latest is not None and not latest.empty:
                    prices[sym] = float(latest.iloc[-1]["close"])
            except Exception:
                continue
        return prices

    def _position_status_entry(
        self,
        symbol: str,
        *,
        side: str,
        qty: int,
        entry: float,
        time_text: str,
        latest_prices: dict[str, float],
    ) -> dict:
        item = {
            "side": side,
            "qty": qty,
            "entry": entry,
            "time": time_text,
        }
        last_price = latest_prices.get(symbol)
        if last_price is None:
            for k, p in latest_prices.items():
                if self._match_symbol_variant(symbol, k):
                    last_price = p
                    break
        if last_price is None or not entry:
            return item

        meta = getattr(self.risk_agent, "contract_meta", {}).get(symbol, {})
        if not meta:
            for k, m in getattr(self.risk_agent, "contract_meta", {}).items():
                if self._match_symbol_variant(symbol, k):
                    meta = m
                    break
        tick_size = float(meta.get("tickSize") or self.risk_agent.get_tick_size(symbol) or 0.25)
        tick_value = float(meta.get("tickValue") or self.risk_agent.get_tick_value(symbol) or 0.5)
        if tick_size <= 0:
            return item

        side_mult = 1 if str(side).lower() in ("buy", "long") else -1
        open_ticks = round((float(last_price) - float(entry)) / tick_size) * side_mult
        open_pnl = open_ticks * tick_value * int(qty)
        item.update(
            {
                "last_price": round(float(last_price), 6),
                "open_ticks": int(open_ticks),
                "open_pnl": round(float(open_pnl), 2),
            }
        )
        return item

    def get_current_session_start_utc(self) -> datetime:
        from datetime import datetime, timedelta, timezone
        from zoneinfo import ZoneInfo

        tz_ny = ZoneInfo('America/New_York')
        now_et = datetime.now(timezone.utc).astimezone(tz_ny)
        current_time_et = now_et.time()
        wd = now_et.weekday()
        asian_start_time = dt_time(18, 0)

        # During weekend closure (Fri >= 17:00, all Saturday, Sunday < 18:00),
        # return Thursday 18:00 ET (start of Friday's trading session).
        if (wd == 4 and current_time_et >= dt_time(17, 0)) or wd == 5 or (wd == 6 and current_time_et < asian_start_time):
            days_back_to_thu = (wd - 3) % 7
            session_start_et = (now_et - timedelta(days=days_back_to_thu)).replace(hour=18, minute=0, second=0, microsecond=0)
        elif current_time_et >= asian_start_time:
            session_start_et = now_et.replace(hour=18, minute=0, second=0, microsecond=0)
        else:
            session_start_et = (now_et - timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)

        return session_start_et.astimezone(timezone.utc)

    def get_current_session_trades(self) -> list:
        from datetime import datetime

        from src.utils.session_utils import get_trade_session
        session_start = self.get_current_session_start_utc()
        valid_trades = []
        for t in self._recent_trades:
            ts_str = t.get("timestamp")
            if ts_str:
                try:
                    trade_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    if trade_dt >= session_start:
                        item = dict(t)
                        if not item.get("session") or item.get("session") == "UNKNOWN":
                            item["session"] = get_trade_session(trade_dt)
                        valid_trades.append(item)
                except (ValueError, TypeError):
                    continue
        return valid_trades


    def _get_cached_status_indicators(self, sym: str, agent) -> dict:
        if not hasattr(self, "_status_indicators_cache"):
            self._status_indicators_cache = {}
        latest = self.data_agents[sym].get_latest_data() if sym in self.data_agents else None
        if latest is None or latest.empty:
            return {}
        key = (len(latest), float(latest.iloc[-1].get("close", 0.0)), str(latest.iloc[-1].get("timestamp", "")))
        cached = self._status_indicators_cache.get(sym)
        if cached and cached[0] == key:
            return cached[1]
        ind = agent.get_latest_indicators(latest)
        self._status_indicators_cache[sym] = (key, ind)
        return ind

    def _schedule_status_write(self) -> None:
        """Schedule _write_status off the asyncio event loop onto _db_executor without piling up."""
        loop = getattr(self, "loop", None)
        db_exec = getattr(self, "_db_executor", None)
        if loop is None or not hasattr(loop, "is_running") or not loop.is_running() or db_exec is None:
            self._write_status()
            return
        if getattr(self, "_status_write_in_flight", False):
            return
        self._status_write_in_flight = True

        def _run_status():
            try:
                self._write_status()
            finally:
                self._status_write_in_flight = False

        try:
            loop.run_in_executor(db_exec, _run_status)
        except Exception:
            self._status_write_in_flight = False
            self._write_status()

    def _write_status(self):
        try:
            now_dt = datetime.now(timezone.utc).astimezone()

            # [EXPERT FIX] Persistent Daily P&L Tracking
            # This ensures the dashboard matches Topstep's 'RP&L Today' even after restarts.
            current_bal = self.risk_agent.current_balance
            if current_bal > 0:
                # Reset persistence at 18:00 ET (CME Day Start)
                et_now = datetime.now(timezone.utc).astimezone(pytz.timezone("US/Eastern"))
                day_key = et_now.strftime("%Y-%m-%d")
                if et_now.time() >= dt_time(18, 0):
                    # After 6pm ET, we are technically on the 'Next' trading day
                    day_key = (et_now + timedelta(days=1)).strftime("%Y-%m-%d")

                opening_data = {}
                if self._balance_persistence_path.exists():
                    try:
                        opening_data = json.loads(self._balance_persistence_path.read_text())
                        if not isinstance(opening_data, dict):
                            opening_data = {}
                    except (OSError, ValueError, TypeError):
                        opening_data = {}

                stored_account = str(opening_data.get("account_id") or "")
                same_account = (not stored_account) or (stored_account == str(self.account_id))
                persisted_peak = float(opening_data.get("peak_balance", 0.0)) if same_account else 0.0
                if persisted_peak > float(self.risk_agent.peak_balance):
                    self.risk_agent.peak_balance = Decimal(str(persisted_peak))
                if opening_data.get("day") != day_key or (stored_account and stored_account != str(self.account_id)):
                    # New day or new account detected: set current balance as the opening anchor
                    self._session_opening_balance = current_bal
                    self._pnl_history = []
                    payload = {
                        "day": day_key,
                        "account_id": str(self.account_id),
                        "balance": float(current_bal),
                        "peak_balance": max(float(self.risk_agent.peak_balance), float(current_bal)),
                    }
                    try:
                        tmp_bal = self._balance_persistence_path.with_suffix(".tmp")
                        tmp_bal.write_text(json.dumps(payload, indent=2))
                        os.replace(tmp_bal, self._balance_persistence_path)
                    except Exception as e:
                        self.logger.warning(f"Failed to persist balance anchor: {e}")
                    self.logger.info(f"[PERSISTENCE] New session anchor set: ${current_bal} for {day_key} (Account: {self.account_id})")
                else:
                    self._session_opening_balance = Decimal(str(opening_data["balance"]))
                    if float(self.risk_agent.peak_balance) > persisted_peak:
                        opening_data["peak_balance"] = float(self.risk_agent.peak_balance)
                        try:
                            tmp_bal = self._balance_persistence_path.with_suffix(".tmp")
                            tmp_bal.write_text(json.dumps(opening_data, indent=2))
                            os.replace(tmp_bal, self._balance_persistence_path)
                        except OSError as e:
                            self.logger.debug(f"Peak balance persistence update skipped: {e}")

            latest_prices = self._latest_prices_snapshot()
            open_pnl_values = []
            position_rows = 0

            # Prefer broker truth for the dashboard. Fall back to the bot's internal cache
            # only if we haven't polled the broker in the last 90s (e.g., startup or outage).
            broker_snapshot_age = time.time() - self._latest_broker_positions_at
            use_broker_snapshot = self._latest_broker_positions_at > 0 and broker_snapshot_age < 90
            positions_snapshot = {}
            if use_broker_snapshot:
                for sym, broker_positions in self._latest_broker_positions_snapshot.items():
                    positions_snapshot[sym] = []
                    # Guard: If internal active_positions for sym is empty and was closed recently (<60s),
                    # do not let a stale REST snapshot resurrect a ghost position on the dashboard.
                    if not self.active_positions.get(sym) and (time.time() - self._recently_closed.get(sym, 0) < 60):
                        continue
                    for bp in broker_positions:
                        position_rows += 1
                        item = self._position_status_entry(
                            sym,
                            side=bp["side"],
                            qty=int(bp["qty"]),
                            entry=float(bp["entry"]),
                            time_text="(broker)",
                            latest_prices=latest_prices,
                        )
                        if "open_pnl" in item:
                            open_pnl_values.append(float(item["open_pnl"]))
                        positions_snapshot[sym].append(item)
            else:
                for sym, pos_list in self.active_positions.items():
                    positions_snapshot[sym] = []
                    for p in pos_list:
                        position_rows += 1
                        item = self._position_status_entry(
                            sym,
                            side=p["side"],
                            qty=int(p["qty"]),
                            entry=float(p["entry"]),
                            time_text=datetime.fromtimestamp(p["time"], tz=timezone.utc).strftime("%H:%M:%S") if isinstance(p["time"], (int, float)) else str(p["time"]),
                            latest_prices=latest_prices,
                        )
                        if "open_pnl" in item:
                            open_pnl_values.append(float(item["open_pnl"]))
                        positions_snapshot[sym].append(item)

            zone_snapshot = {}
            for sym, engine in self.zone_engines.items():
                active_zones = engine.manager.active_zones[:10]
                db_zones = self.database.list_active_zones(sym)[:10] if self.database.connected else []
                zone_snapshot[sym] = {
                    "memory": [
                        {
                            "zone_type": getattr(zone, "zone_type", "generic"),
                            "price_low": float(getattr(zone, "price_low", 0.0)),
                            "price_high": float(getattr(zone, "price_high", 0.0)),
                            "strength": float(getattr(zone, "strength", 0.0)),
                            "status": str(getattr(zone, "status", "active")),
                            "touch_count": int(getattr(zone, "touch_count", 0)),
                        }
                        for zone in active_zones
                    ],
                    "database": db_zones,
                }

            validation_snapshot = {}
            for sym in self.official_symbols:
                validation_snapshot[sym] = self.database.list_validation_runs(sym, limit=5) if self.database.connected else []

            meta_optimizer_snapshot = {}
            for sym in self.official_symbols:
                recent_runs = self.database.list_meta_optimizer_runs(sym, limit=int(self.meta_optimizer_config.get("max_recent_runs", 5))) if self.database.connected else []
                latest = self._last_meta_optimizer_runs.get(sym)
                meta_optimizer_snapshot[sym] = {
                    "latest": latest or (recent_runs[0]["summary"] if recent_runs else None),
                    "recent_runs": recent_runs,
                    "dataset": self._resolve_meta_optimizer_data_file(sym),
                    "last_run_at": self._meta_optimizer_last_run_at.get(sym),
                }

            market_open = self.news_agent.is_market_hours(now_dt)

            # Persistent PnL History tracking
            # [EXPERT FIX] Use authoritative balance delta for session P&L
            if self._session_opening_balance and self.risk_agent.current_balance > 0:
                current_pnl = round(float(self.risk_agent.current_balance - self._session_opening_balance), 2)
            else:
                current_pnl = round(float(self.risk_agent.current_daily_pnl), 2)

            if not self._pnl_history or self._pnl_history[-1].get("pnl") != current_pnl:
                now_str = now_dt.strftime("%I:%M:%S %p")
                self._pnl_history.append({"pnl": current_pnl, "label": now_str})
                if len(self._pnl_history) > 120:
                    self._pnl_history.pop(0)

            unrealized_pnl = (
                round(sum(open_pnl_values), 2)
                if position_rows > 0 and len(open_pnl_values) == position_rows
                else round(self.risk_agent.unrealized_pnl, 2)
            )
            net_equity = round(float(self.risk_agent.current_balance) + float(unrealized_pnl), 2)
            status = {
                "last_updated": now_dt.strftime("%Y-%m-%d %H:%M:%S %Z"),
                "running": self.running,
                "mode": os.getenv("MODE", "simulation").strip().upper(),
                "market_open": market_open,
                "account_id": getattr(self, "account_id", None),
                "news_standdown": self.news_agent.is_stand_down(now_dt),
                "symbols": self.official_symbols,
                "balance": round(self.risk_agent.current_balance, 2),
                "net_equity": net_equity,
                "peak_balance": round(self.risk_agent.peak_balance, 2),
                "minimum_balance": round(getattr(self.risk_agent, "minimum_balance", 0), 2),
                "session_pnl": current_pnl,
                "pnl_history": self._pnl_history,
                "unrealized_pnl": unrealized_pnl,
                "drawdown": round(self.risk_agent.peak_balance - self.risk_agent.current_balance, 2),
                "max_drawdown": round(self.risk_agent.max_drawdown, 2),
                "max_daily_loss": round(self.risk_agent.max_daily_loss, 2),
                "active_positions": positions_snapshot,
                "recent_trades": self.database.list_recent_trades(10, account_id=getattr(self, "account_id", None)),
                "session_trades": list(reversed(self.get_current_session_trades())),
                "health": self.health_monitor.get_status(),
                "discovery": dict(self._last_discovery_decisions),
                "mega_vision": dict(self._last_mega_vision_decisions),
                "zones": zone_snapshot,
                "validation_history": validation_snapshot,
                "meta_optimizer": meta_optimizer_snapshot,
                "meta_optimizer_runtime": self._meta_optimizer_runtime_snapshot(),
                "subsystems": {
                    "database": self.database.health_check(),
                    "discovery": self.discovery.health_check(),
                    "mega_vision": {
                        "connected": self.feature_flags.get("mega_vision_enabled", False),
                        "reason": self.mega_vision_config.get("mode", "disabled"),
                    },
                    "meta_optimizer": self._meta_optimizer_health_check(),
                    "routing_enabled": self.feature_flags.get("routing_enabled", False),
                    "zone_engine_enabled": self.feature_flags.get("zone_engine_enabled", False),
                    "health_monitor_enabled": self.feature_flags.get("health_monitor_enabled", False),
                },
                "signalr_connected": self.client.is_connected,
                "last_tick": datetime.fromtimestamp(self.client.last_tick_at, tz=timezone.utc).strftime("%H:%M:%S") if self.client.last_tick_at > 0 else "None",
                "strategies": {
                    sym: {
                        "description": agent.get_strategy_description(),
                        "indicators": self._get_cached_status_indicators(sym, agent),
                    }
                    for sym, agent in self.strategy_agents.items()
                },
            }

            os.makedirs("logs", exist_ok=True)
            tmp_path = "logs/bot_status.tmp"
            final_path = "logs/bot_status.json"

            with open(tmp_path, "w") as f:
                json.dump(self._json_safe(status), f, indent=2)
            os.replace(tmp_path, final_path)
        except Exception as e:
            self.logger.warning(f"Dashboard status write failed: {e}")

    async def _smart_sleep(self, seconds):
        """Sleep for N seconds but check self.running frequently for fast shutdown."""
        import time
        start_time = time.time()
        while time.time() - start_time < seconds:
            if not self.running:
                break
            remaining = seconds - (time.time() - start_time)
            await asyncio.sleep(min(0.5, max(0, remaining)))

    def _restore_balance_persistence(self) -> None:
        """Restores account-lifetime peak_balance and current-day session_opening_balance from disk."""
        if not self._balance_persistence_path.exists():
            return
        try:
            opening_data = json.loads(self._balance_persistence_path.read_text())
            if str(opening_data.get("account_id") or "") == str(self.account_id):
                # Peak balance is an account-lifetime high-water mark (never resets at 18:00 ET)
                if "peak_balance" in opening_data:
                    persisted_peak = self._safe_float(opening_data.get("peak_balance"), 0.0)
                    if persisted_peak > float(self.risk_agent.peak_balance):
                        self.risk_agent.peak_balance = Decimal(str(persisted_peak))
                et_now = datetime.now(timezone.utc).astimezone(pytz.timezone("US/Eastern"))
                day_key = et_now.strftime("%Y-%m-%d")
                if et_now.time() >= dt_time(18, 0):
                    day_key = (et_now + timedelta(days=1)).strftime("%Y-%m-%d")
                if opening_data.get("day") == day_key:
                    self._session_opening_balance = Decimal(str(opening_data.get("balance", 0.0)))
                    self.logger.info(
                        f"[PERSISTENCE] Restored session anchor ${self._session_opening_balance} "
                        f"(peak=${self.risk_agent.peak_balance}) for {day_key}"
                    )
                else:
                    self.logger.info(
                        f"[PERSISTENCE] Restored lifetime peak_balance=${self.risk_agent.peak_balance} "
                        f"(prior day={opening_data.get('day')}, current day={day_key})"
                    )
        except Exception as e:
            self.logger.warning(f"Failed to load session persistence: {e}")

    def stop(self):
        self.logger.info("Stopping Supervisor swarm...")
        self.running = False
        try:
            if self.client:
                self.client.stop_streaming()
                self.client.stop_user_streaming()
            if getattr(self, "_db_executor", None) is not None:
                self._db_executor.shutdown(wait=True)
            if getattr(self, "database", None) and hasattr(self.database, "close"):
                self.database.close()

            # FIX: Forcefully cancel the main loop task if it's stuck in an await.
            # This ensures we don't hang on SIGINT if a REST call is hanging.
            if hasattr(self, '_main_task') and self._main_task and not self._main_task.done():
                self.logger.info("Interrupting main loop task...")
                self._main_task.cancel()

            self.logger.info("Supervisor stopped successfully.")
        except Exception as e:
            self.logger.error(f"Error during supervisor shutdown: {e}")

    def _load_dataset_catalog(self):
        catalog_path = Path(self.meta_optimizer_config.get("dataset_catalog", "data/dataset_catalog.json"))
        if not catalog_path.exists():
            return []
        try:
            with catalog_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            return loaded if isinstance(loaded, list) else []
        except Exception as exc:
            self.logger.warning(f"Failed to load dataset catalog from {catalog_path}: {exc}")
            return []

    def _resolve_meta_optimizer_data_file(self, symbol: str):
        symbol_upper = str(symbol).upper()
        for item in self._meta_optimizer_catalog:
            if str(item.get("symbol", "")).upper() == symbol_upper:
                path = item.get("path")
                if path and Path(path).exists():
                    return path
        base_symbol = StrategyAgent._extract_base_static(symbol_upper).lower()
        data_dir = Path("data")
        if not data_dir.exists():
            return None
        matches = sorted(
            [candidate for candidate in data_dir.glob("*.csv") if candidate.name.lower().startswith(base_symbol)],
            key=lambda candidate: candidate.stat().st_mtime,
            reverse=True,
        )
        return str(matches[0]) if matches else None

    def _meta_optimizer_health_check(self):
        enabled = bool(
            self.feature_flags.get("meta_optimizer_enabled", False)
            and self.meta_optimizer_config.get("enabled", False)
        )
        datasets_ready = sum(1 for sym in self.official_symbols if self._resolve_meta_optimizer_data_file(sym))
        return {
            "connected": enabled,
            "reason": "enabled" if enabled else "disabled",
            "datasets_ready": datasets_ready,
            "symbols_tracked": len(self.official_symbols),
            "interval_seconds": int(self.meta_optimizer_config.get("run_interval_seconds", 1800)),
            "trials_per_epoch": int(self.meta_optimizer_config.get("trials_per_epoch", 2)),
            "market_closed_only": bool(self.meta_optimizer_config.get("market_closed_only", True)),
        }

    def _meta_optimizer_runtime_snapshot(self):
        return {
            "enabled": bool(self.meta_optimizer_config.get("enabled", False)),
            "run_interval_seconds": int(self.meta_optimizer_config.get("run_interval_seconds", 1800)),
            "trials_per_epoch": int(self.meta_optimizer_config.get("trials_per_epoch", 2)),
            "market_closed_only": bool(self.meta_optimizer_config.get("market_closed_only", True)),
            "max_recent_runs": int(self.meta_optimizer_config.get("max_recent_runs", 5)),
        }

    def update_meta_optimizer_runtime_config(self, updates):
        if not isinstance(updates, dict):
            raise ValueError("updates must be a dictionary")
        if "enabled" in updates:
            self.meta_optimizer_config["enabled"] = bool(updates["enabled"])
        if "run_interval_seconds" in updates:
            self.meta_optimizer_config["run_interval_seconds"] = max(int(updates["run_interval_seconds"]), 60)
        if "trials_per_epoch" in updates:
            self.meta_optimizer_config["trials_per_epoch"] = max(int(updates["trials_per_epoch"]), 1)
        if "market_closed_only" in updates:
            self.meta_optimizer_config["market_closed_only"] = bool(updates["market_closed_only"])
        self._write_status()
        return self._meta_optimizer_runtime_snapshot()

    async def _run_meta_optimizer_cycle(self, now_ts: float):
        if not self.feature_flags.get("meta_optimizer_enabled", False):
            return
        if not self.meta_optimizer_config.get("enabled", False):
            return
        if self.meta_optimizer_config.get("market_closed_only", True):
            now_dt = datetime.now(timezone.utc)
            if self.news_agent.is_market_hours(now_dt):
                return
        interval = max(int(self.meta_optimizer_config.get("run_interval_seconds", 1800)), 60)
        trials_per_epoch = int(self.meta_optimizer_config.get("trials_per_epoch", 2))
        for sym in self.official_symbols:
            last_run = float(self._meta_optimizer_last_run_at.get(sym, 0.0))
            if now_ts - last_run < interval:
                continue
            data_file = self._resolve_meta_optimizer_data_file(sym)
            if not data_file:
                self._last_meta_optimizer_runs[sym] = {
                    "symbol": sym,
                    "status": "skipped",
                    "reason": "no_dataset",
                    "recommendation": None,
                }
                self._meta_optimizer_last_run_at[sym] = now_ts
                continue
            self._meta_optimizer_last_run_at[sym] = now_ts
            try:
                service = self._meta_optimizer_services.get(sym)
                if service is None or getattr(service, "data_file", None) != data_file:
                    service = MetaOptimizerService(sym, data_file, database=self.database)
                    self._meta_optimizer_services[sym] = service
                next_epoch = 1
                if self.database.connected:
                    recent = self.database.list_meta_optimizer_runs(sym, limit=1)
                    if recent:
                        next_epoch = int(recent[0].get("epoch", 0)) + 1
                result = await self.loop.run_in_executor(
                    None,
                    lambda svc=service, epoch=next_epoch, trials=trials_per_epoch: svc.run_epoch(epoch=epoch, trials_per_epoch=trials),
                )
                result["status"] = "completed"
                result["data_file"] = data_file
                self._last_meta_optimizer_runs[sym] = result
                self.logger.info(
                    f"[META OPTIMIZER] {sym} epoch={result.get('epoch')} best_score={result.get('best_score')} "
                    f"verdict={(result.get('recommendation') or {}).get('verdict')}"
                )
            except Exception as exc:
                self.logger.warning(f"Meta optimizer run failed for {sym}: {exc}", exc_info=True)
                self._last_meta_optimizer_runs[sym] = {
                    "symbol": sym,
                    "status": "failed",
                    "reason": str(exc),
                    "data_file": data_file,
                }

    def reload_configuration(self):
        """Dynamically reloads stack configuration and updates strategy agents while flat."""
        try:
            if any(self.active_positions.values()):
                self.logger.warning("Hot-reload aborted: open positions detected.")
                return False
            self.stack_config = load_stack_config()
            self.strategy_config = get_strategy_config(self.stack_config)
            self.routing_config = get_routing_config(self.stack_config)
            self.filter_manager = FilterManager(self.routing_config)
            self.trade_router = TradeRouter(
                self.risk_agent,
                self.execution_agent,
                telemetry=self.telemetry,
                filter_manager=self.filter_manager,
            )
            if hasattr(self, "risk_agent") and self.risk_agent is not None and "symbol_brackets" in self.stack_config:
                self.risk_agent.symbol_brackets = self.stack_config.get("symbol_brackets", {})
            for official in self.official_symbols:
                if official in self.strategy_agents:
                    try:
                        self.strategy_agents[official].reset()
                    except Exception as e:
                        self.logger.debug(f"Strategy reset before reload skipped for {official}: {e}")
                    self.strategy_agents[official] = StrategyAgent(
                        official, strategy_config=self.strategy_config, telemetry=self.telemetry
                    )
                    data_agent = getattr(self, "data_agents", {}).get(official)
                    if data_agent is not None:
                        latest = data_agent.get_latest_data()
                        if latest is not None and not latest.empty:
                            engine = self.strategy_agents[official].engine
                            indicators = engine.prepare_data(latest)
                            bars = indicators[["timestamp", "open", "high", "low", "close", "volume", "price"]].copy()
                            engine.prime(bars, indicators, len(indicators) - 1)
            self.logger.info("Successfully reloaded configuration and primed active strategy agents.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to reload configuration: {e}", exc_info=True)
            return False

    async def _check_saturday_optimization(self):
        """Autonomously triggers the Saturday LLM optimizer on Saturday mornings when flat."""
        try:
            from zoneinfo import ZoneInfo
            et_now = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
            today_str = et_now.strftime("%Y-%m-%d")

            # Must be Saturday (weekday == 5) and at/after 08:00 ET
            if et_now.weekday() != 5 or et_now.hour < 8:
                return

            # Must not have run today already in memory
            if self._last_saturday_optimizer_run == today_str:
                return

            # Retry backoff / attempt cap after failures
            if self._optimizer_failures >= 3 or time.time() < self._optimizer_next_retry_at:
                return

            # Must be completely flat
            if any(self.active_positions.values()):
                self.logger.info("Saturday optimization pending: waiting for all positions to be flat.")
                return

            # Check state file on disk (in case bot was restarted on Saturday)
            state_file = Path("logs/saturday_optimizer_state.json")
            if state_file.exists():
                try:
                    state_data = json.loads(state_file.read_text(encoding="utf-8"))
                    if state_data.get("run_id") == today_str and state_data.get("phase") == "COMPLETED":
                        self._last_saturday_optimizer_run = today_str
                        self.logger.info(f"Saturday optimizer already completed for {today_str} (from state file).")
                        return
                except Exception as e:
                    self.logger.warning(f"Error checking optimizer state file: {e}")

            # Check if an optimizer task is already running
            if self._optimizer_task and not self._optimizer_task.done():
                return

            self.logger.info(f"==> [AUTONOMOUS SATURDAY OPTIMIZER] Triggering weekly LLM optimization for {today_str}...")
            self._last_saturday_optimizer_run = today_str
            self._optimizer_task = asyncio.create_task(self._run_saturday_optimizer_async(today_str))
        except Exception as e:
            self.logger.error(f"Error checking Saturday optimization: {e}", exc_info=True)

    async def _run_saturday_optimizer_async(self, run_date: str):
        """Executes the optimizer script in a decoupled subprocess and reloads config if promoted."""
        try:
            script_path = Path(__file__).resolve().parent.parent.parent / "scripts" / "saturday_autonomous_optimizer.py"
            cmd = [sys.executable, str(script_path), "--days", "7"]
            self.logger.info(f"Launching Saturday optimizer subprocess: {' '.join(cmd)}")

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await proc.communicate()
            finally:
                if proc.returncode is None:
                    proc.kill()
                    await proc.wait()
            try:
                log_dir = Path("logs")
                log_dir.mkdir(exist_ok=True)
                (log_dir / "saturday_optimizer_last_run.log").write_text(
                    (stdout or b"").decode(errors="replace") + "\n--- STDERR ---\n" + (stderr or b"").decode(errors="replace"),
                    encoding="utf-8",
                )
            except OSError as e:
                self.logger.debug(f"Could not write saturday_optimizer_last_run.log: {e}")

            if proc.returncode == 0:
                self._optimizer_failures = 0
                self.logger.info("Saturday Autonomous Optimizer completed successfully.")
                state_file = Path("logs/saturday_optimizer_state.json")
                if state_file.exists():
                    state_data = json.loads(state_file.read_text(encoding="utf-8"))
                    winner = state_data.get("promoted_candidate")
                    if winner and state_data.get("dry_run"):
                        self.logger.info("Saturday optimizer ran in dry-run mode; skipping config reload.")
                    elif winner:
                        self.logger.info(
                            f"[OPTIMIZATION PROMOTION] Promoted: {winner.get('candidate_id')} "
                            f"(PF={winner.get('profit_factor')}, Holdout=${winner.get('holdout_pnl')}). Reloading configuration..."
                        )
                        self.reload_configuration()
                        if self.health_monitor:
                            self.health_monitor.record_event(
                                "SATURDAY_OPTIMIZATION_PROMOTED",
                                f"Promoted: {winner.get('candidate_id')}, PF={winner.get('profit_factor')}"
                            )
                    else:
                        self.logger.info("Saturday optimization complete: Baseline held, no candidate outperformed holdout gates.")
            else:
                self.logger.error(f"Saturday optimizer exited with returncode {proc.returncode}: {stderr.decode()[:500]}")
                self._last_saturday_optimizer_run = None
                self._optimizer_failures += 1
                self._optimizer_next_retry_at = time.time() + 1800
        except Exception as e:
            self.logger.error(f"Error in Saturday optimizer subprocess execution: {e}", exc_info=True)
            self._last_saturday_optimizer_run = None
            self._optimizer_failures += 1
            self._optimizer_next_retry_at = time.time() + 1800

