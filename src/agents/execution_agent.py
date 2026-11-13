import logging
import os

from src.api.projectx_client import ProjectXClient


class ExecutionAgent:
    def __init__(self, client: ProjectXClient):
        self.client = client
        self.logger = logging.getLogger(__name__)
        self.mode = os.getenv("MODE", "simulation").strip().lower()
        self.simulated_session_pnl = 0.0

    def execute_trade(self, trade_intent):
        """Places an order with ProjectX, including SL/TP brackets."""
        symbol = trade_intent['symbol']
        side = trade_intent['side']
        quantity = trade_intent['quantity']
        sl_ticks = trade_intent.get('sl_ticks')
        tp_ticks = trade_intent.get('tp_ticks')

        # FIX: Support for disabling order brackets if the account uses "Position Brackets"
        # which are incompatible with API-sent SL/TP brackets.
        use_brackets = os.getenv("USE_ORDER_BRACKETS", "true").strip().lower() == "true"
        if not use_brackets:
            self.logger.info(f"USE_ORDER_BRACKETS is false. Stripping brackets from {symbol} order.")
            sl_ticks = None
            tp_ticks = None
        else:
            metadata = trade_intent.get("metadata") or {}
            if isinstance(metadata, dict) and (
                metadata.get("stop_price") is not None
                or metadata.get("target_price") is not None
                or metadata.get("target_r_multiple") is not None
            ):
                import math
                entry_ref = trade_intent.get("entry_price")
                meta_fn = getattr(self.client, "get_contract_meta", None)
                c_meta = meta_fn(symbol) if callable(meta_fn) else None
                tick_sz = float(c_meta.get("tickSize") or 0.0) if isinstance(c_meta, dict) else 0.0
                if entry_ref is not None and tick_sz > 0:
                    try:
                        ep = float(entry_ref)
                        if math.isfinite(ep) and ep > 0:
                            sp = metadata.get("stop_price")
                            if sp is not None:
                                sp_f = float(sp)
                                if math.isfinite(sp_f):
                                    if side == "Buy" and sp_f < ep:
                                        sl_ticks = max(1, int(math.ceil((ep - sp_f) / tick_sz)))
                                    elif side == "Sell" and sp_f > ep:
                                        sl_ticks = max(1, int(math.ceil((sp_f - ep) / tick_sz)))
                            tp = metadata.get("target_price")
                            tr_mult = metadata.get("target_r_multiple")
                            if tp is not None:
                                tp_f = float(tp)
                                if math.isfinite(tp_f):
                                    if side == "Buy" and tp_f > ep:
                                        tp_ticks = max(1, int(math.floor((tp_f - ep) / tick_sz)))
                                    elif side == "Sell" and tp_f < ep:
                                        tp_ticks = max(1, int(math.floor((ep - tp_f) / tick_sz)))
                            elif tr_mult is not None and sl_ticks:
                                tr_f = float(tr_mult)
                                if math.isfinite(tr_f) and tr_f > 0:
                                    tp_ticks = max(1, round(int(sl_ticks) * tr_f))
                    except (ValueError, TypeError) as meta_err:
                        self.logger.debug(f"Metadata bracket override skipped for {symbol}: {meta_err}")

        # Log that we are sending a simulated order and INTERCEPT it
        if self.mode != "live":
            self.logger.info(f"SIMULATION [{symbol}]: Dispatching paper trade to broker API -> {side} {quantity} (SL:{sl_ticks}, TP:{tp_ticks})")
            # HARD INTERCEPT: Return a fake success payload instantly
            import uuid
            return {
                "success": True, 
                "orderId": f"MOCK-{uuid.uuid4().hex[:8]}", 
                "status": "simulation_fill"
            }

        # EXTRA SAFETY SWITCH for LIVE mode
        live_switch = os.getenv("ACTUAL_LIVE_TRADING", "false").strip().lower()
        if live_switch != "true":
            self.logger.critical(f"LIVE MODE BLOCK: Bot is in live mode but ACTUAL_LIVE_TRADING is not 'true'. Intercepting order for {symbol}.")
            return {"status": "error", "message": "Live trading switch is OFF."}

        try:
            # Call ProjectX API via client with brackets and order type parameters
            response = self.client.place_order(
                symbol=symbol,
                quantity=quantity,
                side=side,
                order_type=trade_intent.get('order_type', 'Market'),
                limit_price=trade_intent.get('limit_price'),
                stop_price=trade_intent.get('stop_price'),
                sl_ticks=sl_ticks,
                tp_ticks=tp_ticks,
            )
            
            # FIX #15: Verify order success before reporting
            if isinstance(response, dict):
                if response.get('status') == 'error' or response.get('success') is False:
                    error_msg = response.get('errorMessage') or response.get('message') or 'Unknown order error'
                    
                    # PROACTIVE HELP: Detect TopstepX bracket conflict
                    if "Position Brackets" in str(error_msg):
                        self.logger.critical("TOPSTEPX BRACKET CONFLICT DETECTED: Your account has 'Position Brackets' enabled. "
                                           "Either disable 'Position Brackets' in TopstepX dashboard, or set USE_ORDER_BRACKETS=false in your .env.")
                    
                    self.logger.error(f"Order REJECTED by broker: {error_msg}")
                    return {"status": "rejected", "message": error_msg}
                    
                order_id = response.get('orderId') or response.get('id')
                self.logger.info(f"LIVE Order CONFIRMED: {side} {quantity}x {symbol} (ID: {order_id})")
            else:
                self.logger.warning(f"Unexpected response type from broker: {type(response)}")
            
            return response
        except Exception as e:
            self.logger.error(f"Execution failed: {e}")
            return {"status": "error", "message": str(e)}

    def flatten_position(self, symbol, quantity, current_side):
        """FIX #4: Closes a specific position for a symbol (market order opposite side)."""
        if self.mode != "live":
            self.logger.info(f"SIMULATION: Dispatching paper flatten to broker API -> {symbol} {quantity}x {current_side}")
            import uuid
            return {
                "success": True, 
                "orderId": f"MOCK-FLAT-{uuid.uuid4().hex[:8]}", 
                "status": "simulation_fill"
            }
        
        # EXTRA SAFETY SWITCH for LIVE mode
        live_switch = os.getenv("ACTUAL_LIVE_TRADING", "false").strip().lower()
        if live_switch != "true":
            self.logger.critical(f"LIVE MODE BLOCK (Flatten): Intercepting order for {symbol}.")
            return {"status": "error", "message": "Live trading switch is OFF."}

        if quantity <= 0:
            self.logger.info(f"Position for {symbol} already flat (quantity={quantity}).")
            return {"status": "noop", "message": f"Position already flat (quantity={quantity})"}
        
        try:
            # [STRICT GATE] Cancel any open SL/TP bracket orders for this symbol BEFORE sending
            # the closing market order. Even if sweeping fails or times out, we MUST PROCEED
            # with closing the position to guarantee emergency liquidation and avoid catastrophic runaway risk.
            try:
                cancelled = self.client.cancel_open_orders_for_symbol(symbol)
                self.logger.info(f"Pre-flatten: cancelled {cancelled} open bracket order(s) for {symbol}")
            except Exception as ce:
                self.logger.error(
                    f"Pre-flatten bracket cancel warning for {symbol}: {ce}. "
                    "Proceeding with market flatten order to guarantee position liquidation."
                )

            # Determine opposite side (case-insensitive and type-safe)
            if isinstance(current_side, int):
                close_side = "Sell" if current_side == 0 else "Buy"
            else:
                side_norm = str(current_side).strip().lower()
                close_side = "Sell" if side_norm in ("0", "buy", "long", "bid") else "Buy"
            
            # Place a market order to close
            response = self.client.place_order(
                symbol=symbol,
                quantity=quantity,
                side=close_side,
                order_type="Market"
            )
            
            if isinstance(response, dict) and (response.get('success') is False or response.get('status') in ('error', 'rejected')):
                error_msg = response.get('errorMessage') or response.get('message') or 'Unknown flatten error'
                self.logger.error(f"Flatten FAILED for {symbol}: {error_msg}")
                return {"status": "rejected", "success": False, "message": error_msg}

            self.logger.info(f"Flatten order ({close_side} {quantity}) sent for {symbol}")

            # Post-flatten sweep: ensure no lingering orphaned bracket orders remain
            try:
                self.client.cancel_open_orders_for_symbol(symbol)
            except Exception:
                pass
            
            return response
        except Exception as e:
            self.logger.error(f"Flatten execution failed for {symbol}: {e}")
            return {"status": "error", "message": str(e)}
