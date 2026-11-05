from __future__ import annotations

import logging
import math
import os
import threading
from decimal import Decimal

from dotenv import load_dotenv

from src.config.settings import (
    get_combine_rules,
    get_commissions_config,
    get_feature_flags,
    get_routing_config,
    get_symbol_brackets,
    load_stack_config,
)

load_dotenv()

class RiskAgent:
    # Known micro-futures root symbols — used for position limits and commission tiers.
    _MICRO_PREFIXES = {"MNQ", "MES", "MCL", "MGC", "MYM", "M2K", "MHG", "M6E", "M6A", "M6B", "M6J"}

    # Exchange contract specifications for fallback tick value ($/tick) and tick size
    _DEFAULT_TICK_VALUES = {
        "MNQ": 0.50, "NQ": 5.0, "MES": 1.25, "ES": 12.50,
        "MCL": 1.00, "CL": 10.00, "MGC": 1.00, "GC": 10.00,
        "MYM": 0.50, "YM": 5.00, "M2K": 0.50, "RTY": 5.00,
        "6E": 6.25, "M6E": 0.625
    }
    _DEFAULT_TICK_SIZES = {
        "MNQ": 0.25, "NQ": 0.25, "MES": 0.25, "ES": 0.25,
        "MCL": 0.01, "CL": 0.01, "MGC": 0.10, "GC": 0.10,
        "MYM": 1.00, "YM": 1.00, "M2K": 0.10, "RTY": 0.10,
        "6E": 0.00005, "M6E": 0.00005
    }

    def __init__(self, account_id: str):
        self.account_id = account_id
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()
        
        # Use Decimal for financial precision
        self.stack_config = load_stack_config()
        default_rules = get_combine_rules(self.stack_config, "topstep_50k_daily_enforced")
        self.max_daily_loss = Decimal(os.getenv("MAX_DAILY_LOSS", str(default_rules.get("daily_loss_limit", 1000))))
        self.max_drawdown = Decimal(os.getenv("MAX_DRAWDOWN", str(default_rules.get("max_loss_limit", 2000))))
        self.max_total_contracts = Decimal(os.getenv("MAX_TOTAL_CONTRACTS", str(default_rules.get("max_contracts", 5))))
        starting_bal = Decimal(str(default_rules.get("starting_balance", 50000)))
        self.starting_balance = starting_bal
        self.minimum_balance = max(Decimal("0.0"), starting_bal - self.max_drawdown)
        
        self.current_daily_pnl = Decimal("0.0")
        self.current_pos_count = 0     # Number of open positions (not contracts)
        self.current_balance = Decimal("0.0")
        self.peak_balance = Decimal("0.0")
        
        # FIX H3: Per-contract-type position limits
        # Full futures: 1 position, 1 contract max
        # Micro futures: 2 positions, 3 contracts each max
        self.position_limits = {}  # symbol -> {"max_positions": int, "max_qty": int}
        self.open_positions = {}   # symbol -> list of open position records
        
        # Contract metadata from API
        self.contract_meta = {}  # symbol -> {tickSize, tickValue, ...}
        
        # FIX H3: Track recent trade results for alpha scaling (symbol -> list)
        self.recent_trade_results = {} # symbol -> List of floats (PnL)
        self.unrealized_pnl = Decimal("0.0")      # Real-time unrealized P&L
        self.pending_orders = set()    # Track symbols with working orders
        self.pending_since = {}        # symbol -> timestamp
        self.total_contracts_traded_today = 0 # For commission estimation
        
        self.symbol_brackets = get_symbol_brackets(self.stack_config)
        self.feature_flags = get_feature_flags(self.stack_config)
        self.routing_config = get_routing_config(self.stack_config)
        self.commissions_config = get_commissions_config(self.stack_config)

        # Bot-side daily halts. These ride alongside the existing MLL/daily-loss
        # protections in is_risk_limit_hit. Both are optional: when unset, the
        # legacy max_daily_loss / MLL behavior is unchanged.
        # daily_profit_target: realized P&L >= this amount halts trading for the day.
        # daily_loss_halt: realized P&L <= -this amount halts trading for the day.
        target_raw = self.routing_config.get("daily_profit_target")
        self.daily_profit_target = Decimal(str(target_raw)) if target_raw not in (None, "") else None
        loss_raw = self.routing_config.get("daily_loss_halt")
        self.daily_loss_halt = Decimal(str(loss_raw)) if loss_raw not in (None, "") else None

        # Hard Consecutive Loss Circuit Breakers
        self.consecutive_losses = {}  # symbol -> int count
        self.symbol_halts = set()     # set of halted symbols for current session
        self.max_consecutive_losses = int(self.routing_config.get("max_consecutive_losses", 2))

    def _extract_base(self, symbol: str) -> str:
        import re

        symbol_upper = str(symbol).upper()
        if symbol_upper in self.symbol_brackets:
            return symbol_upper
        best_match = None
        for known in self.symbol_brackets:
            if symbol_upper.startswith(known):
                if best_match is None or len(known) > len(best_match):
                    best_match = known
        if best_match:
            return best_match
        match = re.match(r"^([A-Z0-9]+?)[FGHJKMNQUVXZ]\d{1,2}$", symbol_upper)
        if match:
            return match.group(1)
        return symbol_upper

    def _get_commission_per_side(self, symbol):
        """Returns commission cost per side for a symbol based on config."""
        base_sym = self._extract_base(symbol)
        
        # 1. Direct symbol match
        if base_sym in self.commissions_config:
            return Decimal(str(self.commissions_config[base_sym]))
            
        # 2. Category match
        if base_sym in self._MICRO_PREFIXES:
            return Decimal(str(self.commissions_config.get("micro", 0.62)))
            
        return Decimal(str(self.commissions_config.get("full", 2.10)))

    def add_realized_pnl(self, pnl: float):
        """Incrementally adds realized PnL from a single fill event.
        Used when the broker account event does not carry a dailyPnL field
        (as is the case with ProjectX's GatewayUserAccount stream)."""
        with self._lock:
            self.current_daily_pnl += Decimal(str(pnl))

    def register_contract(self, symbol, meta):
        """
        Register a resolved contract with its metadata.
        Determines if it's micro or full-size for position limits.
        """
        with self._lock:
            base_sym = self._extract_base(symbol)
            # Prefer known-prefix list over fragile string matching on API description/symbolId.
            is_micro = base_sym in self._MICRO_PREFIXES
            if not is_micro:
                # Fallback: check description in case a new symbol hasn't been added to the list yet
                description = meta.get('description', '').lower()
                is_micro = 'micro' in description

            self.contract_meta[symbol] = meta

            if is_micro:
                self.position_limits[base_sym] = {"max_positions": 1}
                self.logger.info(f"Registered {symbol} as MICRO {base_sym} (max 1 position, qty from instrument sizer)")
            else:
                self.position_limits[base_sym] = {"max_positions": 1}
                self.logger.info(f"Registered {symbol} as FULL {base_sym} (max 1 position, qty from instrument sizer)")

    @staticmethod
    def _parse_positive_float(val) -> float | None:
        """Safely parse a finite positive float value."""
        if val is None:
            return None
        try:
            f_val = float(val)
            if math.isfinite(f_val) and f_val > 0.0:
                return f_val
        except (ValueError, TypeError, OverflowError):
            pass
        return None

    def get_tick_value(self, symbol) -> float:
        """Returns the tick value for a symbol from contract metadata, brackets, or validated exchange specs."""
        meta = self.contract_meta.get(symbol) if isinstance(self.contract_meta, dict) else None
        meta_dict = meta if isinstance(meta, dict) else {}
        val = self._parse_positive_float(meta_dict.get('tickValue'))
        if val is not None:
            return val
        base = self._extract_base(symbol)
        bracket = self.symbol_brackets.get(base) if isinstance(self.symbol_brackets, dict) else None
        bracket_dict = bracket if isinstance(bracket, dict) else {}
        bracket_val = self._parse_positive_float(bracket_dict.get("tick_value"))
        if bracket_val is not None:
            return bracket_val
        spec_val = self._DEFAULT_TICK_VALUES.get(base.upper()) if base else None
        if spec_val is not None:
            return spec_val
        raise ValueError(
            f"Unvalidated instrument {symbol} (base: {base}): positive tick_value cannot be resolved "
            f"from contract metadata, symbol brackets, or exchange specifications."
        )

    # NQ ($5/tick) is the reference for "1 full-equivalent contract" against the global ceiling.
    _FULL_EQUIV_REFERENCE_TICK_VALUE = 5.0

    def _full_equiv_weight(self, symbol):
        """Weight a single contract against the global ceiling using its tick value."""
        tv = float(self.get_tick_value(symbol) or 0.0)
        if tv <= 0:
            return Decimal("1.0")
        return Decimal(str(tv)) / Decimal(str(self._FULL_EQUIV_REFERENCE_TICK_VALUE))

    def _weighted_open_total(self):
        """Sum of open contract qty across all symbols, weighted by each symbol's full-equiv."""
        with self._lock:
            total = Decimal("0.0")
            for sym, pos_list in list(self.open_positions.items()):
                qty = sum(p.get("qty", 0) for p in pos_list)
                if qty <= 0:
                    continue
                total += Decimal(str(qty)) * self._full_equiv_weight(sym)
            return total

    def get_tick_size(self, symbol) -> float:
        """Returns the tick size for a symbol from contract metadata, brackets, or validated exchange specs."""
        meta = self.contract_meta.get(symbol) if isinstance(self.contract_meta, dict) else None
        meta_dict = meta if isinstance(meta, dict) else {}
        val = self._parse_positive_float(meta_dict.get('tickSize'))
        if val is not None:
            return val
        base = self._extract_base(symbol)
        bracket = self.symbol_brackets.get(base) if isinstance(self.symbol_brackets, dict) else None
        bracket_dict = bracket if isinstance(bracket, dict) else {}
        bracket_val = self._parse_positive_float(bracket_dict.get("tick_size"))
        if bracket_val is not None:
            return bracket_val
        spec_val = self._DEFAULT_TICK_SIZES.get(base.upper()) if base else None
        if spec_val is not None:
            return spec_val
        raise ValueError(
            f"Unvalidated instrument {symbol} (base: {base}): positive tick_size cannot be resolved "
            f"from contract metadata, symbol brackets, or exchange specifications."
        )

    def get_entry_quantity(self, symbol, requested_qty, sl_ticks=None, current_price=None, current_atr=None):
        """Compute quantity using instrument-aware sizing when enabled, else fallback to legacy heuristics."""
        if self.feature_flags.get("instrument_sizing_enabled", False):
            sized_qty = self.calculate_instrument_quantity(symbol, sl_ticks=sl_ticks, current_atr=current_atr)
            if sized_qty <= 0:
                return 0
            return min(self.get_dynamic_quantity(symbol, sized_qty), sized_qty)
        return self.get_dynamic_quantity(symbol, requested_qty)

    def calculate_instrument_quantity(self, symbol, sl_ticks=None, risk_pct=None, current_atr=None):
        """Size by risk budget / stop cost per contract. Returns 0 when a single contract would over-risk."""
        with self._lock:
            sl_ticks = int(sl_ticks or 0)
            if sl_ticks <= 0:
                return 0

            tick_value = float(self.get_tick_value(symbol) or 0.0)
            if tick_value <= 0:
                return 0

            equity = self.current_balance + self.unrealized_pnl
            if equity <= 0:
                equity = self.current_balance
            if equity <= 0:
                return 0

            risk_pct = risk_pct if risk_pct is not None else self.routing_config.get("risk_per_trade_pct")
            if risk_pct is None:
                risk_pct = 0.18  # Hardened default — matches the configured 0.18% in trading_stack.yaml
                self.logger.warning(
                    "risk_per_trade_pct not found in routing config — falling back to 0.18%. "
                    "Check trading_stack.yaml routing section."
                )
            risk_pct = float(risk_pct)
            base_risk_budget = float(equity) * (risk_pct / 100.0)
            risk_budget = base_risk_budget

            # Apply dynamic volatility adjustment if enabled and current_atr is provided
            if self.feature_flags.get("volatility_sizing_enabled", False) and current_atr is not None and current_atr > 0:
                base_sym = self._extract_base(symbol)
                bracket = self.symbol_brackets.get(base_sym)
                if bracket and "min_atr" in bracket and "max_atr" in bracket:
                    min_atr = float(bracket["min_atr"])
                    max_atr = float(bracket["max_atr"])
                    typical_atr = (min_atr + max_atr) / 2.0
                    
                    vol_multiplier = min(1.0, typical_atr / float(current_atr))
                    risk_budget *= vol_multiplier
                    self.logger.info(
                        f"[VOL SIZING] {symbol}: atr={float(current_atr):.3f}, typical={typical_atr:.3f}, "
                        f"mult={vol_multiplier:.3f}, risk_budget scaled from ${base_risk_budget:.2f} to ${risk_budget:.2f}"
                    )

            # Protective Cap: A single trade should never risk more than 25% of the daily loss limit
            unscaled_capped_budget = base_risk_budget
            if self.max_daily_loss > 0:
                max_single_trade_risk = float(self.max_daily_loss) * 0.25
                unscaled_capped_budget = min(unscaled_capped_budget, max_single_trade_risk)
                if risk_budget > max_single_trade_risk:
                    self.logger.info(
                        f"Capping single-trade risk budget from ${risk_budget:.2f} to ${max_single_trade_risk:.2f} "
                        f"(25% of daily loss limit ${float(self.max_daily_loss):.2f})"
                    )
                    risk_budget = max_single_trade_risk

            contract_risk = float(sl_ticks) * tick_value
            if contract_risk <= 0:
                return 0
            contracts = int(risk_budget / contract_risk)
            return max(0, contracts)

    def get_dynamic_quantity(self, symbol, requested_qty):
        """Adjusts quantity based on daily performance (Defense-first, but reward streaks)."""
        with self._lock:
            if requested_qty <= 0:
                return 0

            # Global ceiling: weighted by per-contract notional so 1 MNQ != 1 NQ.
            # MAX_TOTAL_CONTRACTS is interpreted as "full-equivalent contracts" (NQ-units).
            weighted_total = self._weighted_open_total()
            raw_total = sum(sum(p.get('qty', 0) for p in pos_list) for pos_list in list(self.open_positions.values()))
            if weighted_total >= self.max_total_contracts:
                self.logger.warning(
                    f"Global cap reached: {float(weighted_total):.2f} NQ-equivalent / {float(self.max_total_contracts):.2f} "
                    f"(raw={raw_total} contracts open)."
                )
                return 0  # No new contracts allowed

            # Headroom for THIS symbol = remaining weighted units / its per-contract weight.
            symbol_weight = self._full_equiv_weight(symbol)
            headroom_for_symbol = max(0, int((self.max_total_contracts - weighted_total) / symbol_weight)) if symbol_weight > 0 else 0

            # 1. DEFENSE: Approaching daily profit target — protect gains with reduced size.
            if self.daily_profit_target is not None:
                if self.current_daily_pnl >= (Decimal(str(self.daily_profit_target)) * Decimal("0.8")):
                    self.logger.info("Approaching daily profit target. Reducing size to defend gains.")
                    requested_qty = max(1, int(float(requested_qty) * 0.5))

            # 2. DEFENSE: If we are down significantly, reduce to slow the bleed.
            max_dll_dec = Decimal(str(self.max_daily_loss))
            if self.current_daily_pnl < -(max_dll_dec * Decimal("0.5")):
                self.logger.info("Daily loss safeguard. Reducing size to minimum.")
                requested_qty = 1

            # 3. ALPHA SCALING: Reward winning streaks with modest size increase.
            # Capped at 1.5x (not 2x) to avoid doubling the risk budget from instrument sizing.
            final_qty = requested_qty
            sym_results = self.recent_trade_results.get(symbol) or self.recent_trade_results.get("GLOBAL", [])
            if len(sym_results) >= 3:
                if sum(sym_results[-3:]) > 0:
                    if self.current_daily_pnl > 0:
                        scaled = int(requested_qty * 1.5)
                        if scaled > requested_qty:
                            self.logger.info(f"Winning streak detected for {symbol}. Scaling {requested_qty} -> {scaled} (1.5x cap).")
                            final_qty = scaled

            return max(0, min(final_qty, headroom_for_symbol))

    def update_unrealized_pnl(self, current_prices: dict):
        """
        Calculates unrealized P&L based on current market prices.
        symbol_prices: { symbol: price }
        """
        with self._lock:
            total_unrealized = Decimal("0.0")
            for symbol, positions in self.open_positions.items():
                price = current_prices.get(symbol)
                if not price:
                    continue
                
                tick_size = Decimal(str(self.get_tick_size(symbol)))
                tick_value = Decimal(str(self.get_tick_value(symbol)))
                
                for pos in positions:
                    entry = Decimal(str(pos.get('entry', "0")))
                    if entry == 0: continue
                    
                    # Unrealized P&L = (Current - Entry) / TickSize * TickValue * Qty
                    # For SHORT positions: (Entry - Current)
                    side_mult = Decimal(1) if pos['side'].lower() in ('buy', 'long') else Decimal(-1)
                    price_decimal = Decimal(str(price))
                    ticks = Decimal(str(round(float(price_decimal - entry) / float(tick_size)))) * side_mult
                    total_unrealized += (ticks * tick_value * Decimal(str(pos['qty'])))
            
            self.unrealized_pnl = total_unrealized

    def is_risk_limit_hit(self):
        """Final circuit breaker: Realized + Unrealized loss check against MLL with commission buffer."""
        with self._lock:
            # Bot-side daily halts. These fire on REALIZED P&L only so an open winner
            # can keep running until it closes. The supervisor responds by flattening
            # all positions and setting _risk_halt for the rest of the session.
            if self.daily_profit_target is not None and self.current_daily_pnl >= self.daily_profit_target:
                return True, (
                    f"Daily profit target reached: +${float(self.current_daily_pnl):.2f} "
                    f">= +${float(self.daily_profit_target):.2f}"
                )
            if self.daily_loss_halt is not None and self.current_daily_pnl <= -self.daily_loss_halt:
                return True, (
                    f"Daily loss halt: -${abs(float(self.current_daily_pnl)):.2f} "
                    f">= -${float(self.daily_loss_halt):.2f}"
                )

            # Estimate total commissions: Symbol-specific from config
            total_commissions = Decimal("0.0")
            
            # 1. Past trades: TopstepX 'balance' and 'dailyPnL' already include
            # all realized commissions. We ONLY subtract the estimated cost to
            # close the currently open positions.
            
            # 2. Current open positions: Precise calculation
            for symbol, positions in self.open_positions.items():
                comm_per_side = self._get_commission_per_side(symbol)
                qty = sum(p['qty'] for p in positions)
                total_commissions += Decimal(str(qty)) * comm_per_side

            # 1. Daily Loss Limit Check
            total_pnl = self.current_daily_pnl + self.unrealized_pnl - total_commissions
            if total_pnl <= -self.max_daily_loss:
                return True, f"Daily Loss Limit hit (incl. commissions): ${float(total_pnl):.2f}"

            # 2. Maximum Loss Limit (MLL) Check
            current_equity = self.current_balance + self.unrealized_pnl - total_commissions
            safety_buffer = Decimal("50.0") # $50 buffer to ensure we flatten before blowing the account
            
            mll_floor = getattr(self, 'minimum_balance', Decimal("0.0"))
            if self.current_balance > Decimal("0.0") and mll_floor > 0 and current_equity <= (Decimal(str(mll_floor)) + safety_buffer):
                return True, f"Maximum Loss Limit (MLL) approached. Equity: ${float(current_equity):.2f}, Floor: ${float(mll_floor):.2f}"
            
            return False, ""

    def validate_trade(self, symbol, quantity, side):
        """Validates a Trade Intent against Topstep rules and position limits."""
        with self._lock:
            if quantity <= 0:
                self.logger.warning(f"Trade REJECTED: {symbol} non-positive quantity {quantity}")
                return False

            is_hit, reason = self.is_risk_limit_hit()
            if is_hit:
                self.logger.warning(f"Trade REJECTED: Risk limit hit: {reason}")
                return False

            base_sym = self._extract_base(symbol)

            # Rule 0: Consecutive Loss Halt per Symbol
            if symbol in self.symbol_halts or base_sym in self.symbol_halts:
                self.logger.warning(f"Trade REJECTED: Symbol {symbol} is halted for session due to {self.max_consecutive_losses} consecutive losses.")
                return False

            # Rule 1: Max Daily Loss — HARD STOP
            if self.current_daily_pnl <= -self.max_daily_loss:
                self.logger.warning(f"Trade REJECTED: Daily loss limit reached (${self.current_daily_pnl:.2f})")
                return False
                
            # Rule 2: Max Drawdown / MLL Floor — HARD STOP
            # Under Topstep combine rules, trailing drawdown stops trailing once starting balance is reached
            if self.minimum_balance > 0 and self.current_balance <= self.minimum_balance:
                self.logger.warning(f"Trade REJECTED: Current balance (${self.current_balance:.2f}) at or below MLL floor (${self.minimum_balance:.2f}).")
                return False
            elif self.minimum_balance <= 0 and (self.peak_balance - self.current_balance) >= self.max_drawdown:
                self.logger.warning("Trade REJECTED: Maximum drawdown reached.")
                return False

            # Rule 3: Per-Symbol Position Limits
            limits = self.position_limits.get(base_sym, {"max_positions": 1, "max_qty_per_pos": 1})
            current_positions = self.open_positions.get(symbol, [])

            # Flip exemption: an opposite-side intent against an existing position is a
            # net-position flip (broker handles close + reopen in one fill). The
            # max-positions cap does not apply because the existing position will be
            # consumed by the same order.
            is_flip = bool(
                current_positions
                and side
                and str(current_positions[0].get("side", "")).lower() != str(side).lower()
            )

            if not is_flip and len(current_positions) >= limits.get("max_positions", 1):
                self.logger.warning(
                    f"Trade REJECTED: {symbol} already at max positions "
                    f"({len(current_positions)}/{limits.get('max_positions', 1)})"
                )
                return False

            if quantity > limits.get("max_qty_per_pos", 9999):
                self.logger.warning(
                    f"Trade REJECTED: {symbol} quantity {quantity} exceeds per-position max "
                    f"({limits.get('max_qty_per_pos', 9999)})"
                )
                return False
                
            # Rule 5: Pending Order Guard
            if symbol in self.pending_orders:
                self.logger.warning(f"Trade REJECTED: Already have a pending order for {symbol}")
                return False
                
            self.logger.info(f"Trade APPROVED: {side} {quantity}x {symbol}")
            return True

    def set_pending(self, symbol, is_pending=True):
        import time
        with self._lock:
            if is_pending:
                self.pending_orders.add(symbol)
                self.pending_since[symbol] = time.time()
            else:
                self.pending_orders.discard(symbol)
                self.pending_since.pop(symbol, None)

    def check_pending_timeout(self, symbol, timeout_seconds=60):
        """Clears a pending order if it hasn't filled within the timeout window."""
        import time
        with self._lock:
            if symbol in self.pending_since:
                start_time = self.pending_since[symbol]
                if time.time() - start_time > timeout_seconds:
                    self.logger.warning(f"PENDING TIMEOUT: Force-clearing working order flag for {symbol} after {timeout_seconds}s")
                    self.set_pending(symbol, False)
                    return True
            return False
    def record_position_open(self, symbol, side, qty, entry_price=0.0, order_id=None):
        """Record that a new position was opened."""
        with self._lock:
            self.set_pending(symbol, False) # Clear pending once filled
            self.total_contracts_traded_today += qty # Track for commission
            if symbol not in self.open_positions:
                self.open_positions[symbol] = []
            self.open_positions[symbol].append({
                "side": side, 
                "qty": int(qty), 
                "entry": Decimal(str(entry_price)),
                "order_id": order_id
            })
            self.logger.info(f"Position recorded: {side} {qty}x {symbol} @ {entry_price} (ID: {order_id})")

    def record_position_close(self, symbol, qty: int | None = None):
        """Record that a position was closed. Handles partial reductions if qty is provided."""
        with self._lock:
            if symbol not in self.open_positions or not self.open_positions[symbol]:
                self.logger.warning(f"Attempted to close position for {symbol} but none found in RiskAgent.")
                return

            if qty is None:
                # Legacy/Default behavior: pop the oldest entry
                closed = self.open_positions[symbol].pop(0)
                self.logger.info(f"Position closed (oldest popped): {closed['side']} {closed['qty']}x {symbol}")
                return

            # Granular reduction
            remaining_to_close = int(qty)
            self.logger.info(f"Recording partial/full close for {symbol}: {remaining_to_close} units.")
            
            while remaining_to_close > 0 and self.open_positions[symbol]:
                oldest = self.open_positions[symbol][0]
                if oldest['qty'] <= remaining_to_close:
                    # This entry is fully consumed
                    consumed = self.open_positions[symbol].pop(0)
                    remaining_to_close -= consumed['qty']
                    self.logger.info(f"Entry fully closed: {consumed['side']} {consumed['qty']}x {symbol}")
                else:
                    # This entry is partially consumed
                    oldest['qty'] -= remaining_to_close
                    self.logger.info(f"Entry partially closed: {remaining_to_close} units deducted from {symbol} (remaining in entry: {oldest['qty']})")
                    remaining_to_close = 0
            
            if remaining_to_close > 0:
                self.logger.warning(f"Closed {qty} for {symbol} but RiskAgent only had {qty - remaining_to_close} tracked.")

    def sync_positions(self, positions_list):
        """
        Overwrites current open positions with a list from the broker.
        positions_list: list of dicts with {symbol, side, qty, entryPrice}
        """
        with self._lock:
            new_open = {}
            for p in positions_list:
                sym = p['symbol']
                if sym not in new_open:
                    new_open[sym] = []
                
                # Map various API field names for entry price
                entry = p.get('entryPrice') or p.get('entry') or p.get('averagePrice', 0)
                
                new_open[sym].append({
                    "side": p['side'], 
                    "qty": int(p['qty']),
                    "entry": Decimal(str(entry))
                })
            
            self.open_positions = new_open
            self.logger.info(f"Risk Agent synchronized {len(positions_list)} positions from broker.")

    def update_account_state(self, balance, pnl, pos_count, minimum_balance=None, session_opening_balance=None):
        with self._lock:
            self.current_balance = Decimal(str(balance))
            self.peak_balance = max(self.peak_balance, self.current_balance)
            # Only update daily PnL if the broker explicitly provided it.
            # When pnl is None the caller does not know the current P&L
            # (e.g. ProjectX does not include dailyPnL in GatewayUserAccount)
            # and we must NOT overwrite the running total that was accumulated
            # incrementally from GatewayUserTrade fill events.
            if pnl is not None:
                self.current_daily_pnl = Decimal(str(pnl))
            elif session_opening_balance is not None and float(session_opening_balance) > 0:
                # Synchronize running daily PnL from persisted session opening balance on restart
                self.current_daily_pnl = self.current_balance - Decimal(str(session_opening_balance))
            self.current_pos_count = int(pos_count)
            if minimum_balance is not None and float(minimum_balance) > 0:
                self.minimum_balance = Decimal(str(minimum_balance))
            elif self.peak_balance > 0 and self.max_drawdown > 0:
                raw_floor = self.peak_balance - self.max_drawdown
                trailing_floor = min(getattr(self, "starting_balance", raw_floor), raw_floor)
                self.minimum_balance = max(getattr(self, "minimum_balance", Decimal("0.0")), trailing_floor)

    def record_trade_result(self, pnl: float, symbol: str | None = None):
        """Track completed trade result, update consecutive losses, and enforce symbol halts."""
        with self._lock:
            pnl_val = float(pnl)
            key = symbol or "GLOBAL"
            if key not in self.recent_trade_results:
                self.recent_trade_results[key] = []
            self.recent_trade_results[key].append(pnl_val)
            if len(self.recent_trade_results[key]) > 20:
                self.recent_trade_results[key].pop(0)

            if symbol:
                base_sym = self._extract_base(symbol)
                if pnl_val < 0:
                    self.consecutive_losses[symbol] = self.consecutive_losses.get(symbol, 0) + 1
                    if symbol != base_sym:
                        self.consecutive_losses[base_sym] = self.consecutive_losses.get(base_sym, 0) + 1
                    self.logger.warning(f"Trade LOSS on {symbol}: ${pnl_val:.2f}. Consecutive losses = {self.consecutive_losses[symbol]}")
                    if self.consecutive_losses[symbol] >= self.max_consecutive_losses or self.consecutive_losses[base_sym] >= self.max_consecutive_losses:
                        self.symbol_halts.add(symbol)
                        self.symbol_halts.add(base_sym)
                        self.logger.warning(f"CIRCUIT BREAKER: {symbol} hit {self.max_consecutive_losses} consecutive losses. Halting symbol for remainder of session.")
                elif pnl_val > 0:
                    self.consecutive_losses[symbol] = 0
                    self.consecutive_losses[base_sym] = 0

    def reset_session(self):
        """Resets all session-scoped state for a new Globex trading day.

        Called by SupervisorAgent when the market re-opens after the daily
        17:00–18:00 ET maintenance window.  Clears realized P&L, unrealized
        P&L, pending-order tracking, and the recent-trade streak so that
        daily-loss/profit-target circuit breakers start fresh and do NOT
        re-trigger immediately from the previous session's numbers.

        NOTE: balance, peak_balance, and open_positions are NOT cleared here
        because the broker is the authoritative source for those; they will
        be refreshed on the next account-status heartbeat / position sync.
        """
        with self._lock:
            self.current_daily_pnl = Decimal("0.0")
            self.unrealized_pnl = Decimal("0.0")
            self.total_contracts_traded_today = 0
            self.recent_trade_results = {}
            self.consecutive_losses = {}
            self.symbol_halts = set()
            self.pending_orders = set()
            self.pending_since = {}
            self.logger.info(
                "RiskAgent session reset: daily P&L, unrealized P&L, "
                "pending orders, and trade history cleared for new session."
            )
