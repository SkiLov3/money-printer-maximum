import logging

import numpy as np
import pandas as pd

from src.agents.news_agent import NewsAgent
from src.agents.risk_agent import RiskAgent
from src.agents.strategy_agent import StrategyAgent
from src.backtesting.screenshot_capture import ScreenshotCapture, ScreenshotConfig
from src.config.settings import get_routing_config, load_stack_config
from src.data.catalog import DatasetCatalog
from src.telemetry.store import TelemetryStore
from src.trading.filters import FilterManager
from src.trading.router import RouterRejection, TradeRouter


class _BacktestExecutionAgent:
    def execute_trade(self, trade_intent):
        return {
            "status": "simulation_fill",
            "avgPrice": trade_intent.get("entry_price", 0),
            "orderId": f"BT-{trade_intent.get('symbol', 'UNKNOWN')}",
        }


class Backtester:
    SIGNAL_COOLDOWN_SECONDS = 300
    WIN_COOLDOWN_SECONDS = 120
    LOSS_COOLDOWN_SECONDS = 600
    STREAK_COOLDOWN_SECONDS = 1800

    def __init__(
        self,
        symbol: str,
        initial_balance=50000,
        strategy_config: dict | None = None,
        telemetry_enabled: bool = True,
        screenshot_config: dict | None = None,
        run_id: str = "backtest",
        routing_mode: str = "full_live",
        execution_model: str | None = None,
    ):
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.routing_mode = str(routing_mode or "full_live")
        self.stack_config = load_stack_config()
        backtest_config = dict(self.stack_config.get("backtesting", {}) or {})
        self.strategy_config = dict(strategy_config or {})
        strategy_risk_config = dict(self.strategy_config.get("risk", {}) or {})
        self.daily_profit_lock = float(strategy_risk_config.get("daily_profit_lock", backtest_config.get("daily_profit_lock", 0.0)) or 0.0)
        self.execution_model = str(execution_model or backtest_config.get("execution_model", "next_bar_open"))
        if self.execution_model not in ("next_bar_open", "same_bar_open"):
            raise ValueError("execution_model must be 'next_bar_open' or 'same_bar_open'")
        self.logger = logging.getLogger(f"Backtester-{symbol}")
        self.telemetry = TelemetryStore(enabled=telemetry_enabled)
        self.dataset_catalog = DatasetCatalog()
        self.strategy = StrategyAgent(symbol, strategy_config=strategy_config, telemetry=self.telemetry)
        self.news_agent = NewsAgent()
        self.risk_agent = RiskAgent("BT-101")
        self.routing_config = self._build_routing_config(get_routing_config(self.stack_config))
        self.trade_router = TradeRouter(
            self.risk_agent,
            _BacktestExecutionAgent(),
            telemetry=self.telemetry,
            filter_manager=FilterManager(self.routing_config),
        )
        self.base_symbol = self.strategy.base_symbol
        self.tick_size = self.strategy.tick_size

        tick_values = {
            "NQ": 5.0,
            "ES": 12.5,
            "MNQ": 0.5,
            "MES": 1.25,
            "CL": 10.0,
            "MCL": 1.0,
            "GC": 10.0,
            "MGC": 1.0,
            "YM": 5.0,
            "MYM": 0.5,
            "NKD": 25.0,
            "MNK": 0.5,
        }
        self.tick_value = tick_values.get(self.base_symbol, 5.0)
        self.max_positions = 2 if self.strategy.is_micro else 1
        self.slippage_ticks = 1
        commissions_cfg = dict(self.stack_config.get("commissions", {}) or {})
        default_comm = 0.62 if self.strategy.is_micro else 2.44
        self.commission_per_contract = float(commissions_cfg.get(self.base_symbol, default_comm))
        self.screenshot_capture = ScreenshotCapture(ScreenshotConfig.from_dict(screenshot_config), run_id=run_id)
        self._run_id = run_id
        self.last_signal_time = 0.0
        self._last_trade_won = None
        self.risk_agent.register_contract(
            self.symbol,
            {
                "tickSize": self.tick_size,
                "tickValue": self.tick_value,
                "description": "Micro Synthetic" if self.strategy.is_micro else "Full Synthetic",
                "symbolId": f"F.US.{self.base_symbol}" if not self.strategy.is_micro else f"F.US.M{self.base_symbol}",
            },
        )

        self._reset_state()

    def _build_routing_config(self, routing_config: dict) -> dict:
        config = dict(routing_config or {})
        if self.routing_mode not in ("optimizer_search", "optimizer_fast"):
            return config

        entry_filters = dict(config.get("entry_filters", {}))
        # Broad optimizer search should preserve sizing/account constraints while
        # skipping operational timing filters that are re-applied to finalists.
        entry_filters.update(
            {
                "market_hours": False,
                "news_standdown": False,
                "near_close": False,
                "cooldown": False,
                "pending_order": False,
                "max_positions": True,
            }
        )
        config["entry_filters"] = entry_filters
        return config

    @staticmethod
    def _session_key(ts) -> str | None:
        if ts is None or pd.isna(ts):
            return None
        dt = pd.to_datetime(ts, errors="coerce")
        if pd.isna(dt):
            return None
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        et = dt.tz_convert("America/New_York")
        return (et + pd.Timedelta(hours=6)).date().isoformat()

    def _reset_state(self):
        self.balance = self.initial_balance
        self.trades = []
        self.active_trades = []
        self.equity_history = []
        self.equity_curve = []
        self.last_signal_time = 0.0
        self._last_trade_won = None
        self._current_session = None
        self.strategy.reset()
        self.risk_agent = RiskAgent("BT-101")
        self.trade_router = TradeRouter(
            self.risk_agent,
            _BacktestExecutionAgent(),
            telemetry=self.telemetry,
            filter_manager=FilterManager(self.routing_config),
        )
        self.risk_agent.register_contract(
            self.symbol,
            {
                "tickSize": self.tick_size,
                "tickValue": self.tick_value,
                "description": "Micro Synthetic" if self.strategy.is_micro else "Full Synthetic",
                "symbolId": f"F.US.{self.base_symbol}" if not self.strategy.is_micro else f"F.US.M{self.base_symbol}",
            },
        )
        self._sync_risk_state()

    def _sync_risk_state(self):
        daily = 0.0
        if self._current_session is not None:
            for t in self.trades:
                closed = t.get("time_close") or t.get("timestamp")
                if closed is not None and self._session_key(closed) == self._current_session:
                    daily += float(t.get("pnl", 0.0) or 0.0)
        if hasattr(self, "_current_bar_price") and self._current_bar_price > 0:
            self.risk_agent.update_unrealized_pnl({self.symbol: self._current_bar_price})
        self.risk_agent.update_account_state(
            self.balance,
            daily,
            len(self.active_trades),
            minimum_balance=None,
        )

    def _normalize_dataframe(self, df: pd.DataFrame):
        frame = df.copy()
        frame.columns = [c.lower() for c in frame.columns]
        required = ["open", "high", "low", "close"]
        if not all(col in frame.columns for col in required):
            raise ValueError(f"CSV must contain OHLC columns: {required}")
        if "price" not in frame.columns:
            frame.loc[:, "price"] = frame["close"]
        if "timestamp" in frame.columns:
            frame.loc[:, "timestamp"] = pd.to_datetime(frame["timestamp"])
        return frame

    def run(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self.dataset_catalog.register_dataset(self.symbol, csv_path, source="backtest_csv")
        return self.run_dataframe(df, dataset_label=csv_path)

    def run_dataframe(self, df: pd.DataFrame, dataset_label: str = "dataframe", prepared_indicators: pd.DataFrame | None = None):
        self._reset_state()
        frame = self._normalize_dataframe(df)
        indicators = prepared_indicators.copy() if prepared_indicators is not None else self.strategy._calculate_indicators(frame)
        bars = pd.DataFrame(indicators[["timestamp", "open", "high", "low", "close", "volume", "price"]].copy())

        self.equity_history = [self.balance]
        self.equity_curve = []
        pending_signal = None

        for index in range(len(indicators)):
            row = indicators.iloc[index]
            current_time = pd.to_datetime(row.get("timestamp", pd.Timestamp.now("UTC")))
            current_time_str = current_time.isoformat()

            session_key = self._session_key(current_time)
            if session_key != self._current_session:
                if self._current_session is not None:
                    self.risk_agent.reset_session()
                self._current_session = session_key

            if pending_signal is not None:
                self._execute_signal_at_bar(pending_signal, row, current_time)
                pending_signal = None

            self._current_bar_price = float(row["close"])
            self._update_active_trades(row)
            self._sync_risk_state()

            if self._daily_profit_lock_reached(current_time, float(row["close"]), include_unrealized=True):
                for trade in self.active_trades:
                    trade["pending_daily_profit_lock_exit"] = True
                floating_pnl = sum(self._calculate_pnl(trade, float(row["close"])) for trade in self.active_trades)
                unrealized_commission = sum(2.0 * trade["qty"] * self.commission_per_contract for trade in self.active_trades)
                equity_value = self.balance + floating_pnl - unrealized_commission
                self.equity_history.append(equity_value)
                self.equity_curve.append({"timestamp": current_time_str, "equity": equity_value})
                self.screenshot_capture.on_bar_end()
                continue

            signal, _ = self.strategy.engine.evaluate_index(bars, indicators, index)
            if signal:
                self._attach_signal_screenshot(signal, bars, index, current_time)
                if self.execution_model == "same_bar_open":
                    self._execute_signal_at_bar(signal, row, current_time)
                else:
                    pending_signal = dict(signal)

            floating_pnl = sum(self._calculate_pnl(trade, float(row["close"])) for trade in self.active_trades)
            unrealized_commission = sum(2.0 * trade["qty"] * self.commission_per_contract for trade in self.active_trades)
            equity_value = self.balance + floating_pnl - unrealized_commission
            self.equity_history.append(equity_value)
            self.equity_curve.append({"timestamp": current_time_str, "equity": equity_value})
            self.screenshot_capture.on_bar_end()
            if self.routing_mode == "optimizer_fast" and not self.active_trades and pending_signal is None:
                risk_hit, _ = self.risk_agent.is_risk_limit_hit()
                if risk_hit:
                    break

        results = self._get_results()
        self.telemetry.record_backtest(self.symbol, dataset_label, self.strategy.engine.config, results["summary"])
        return results

    def _attach_signal_screenshot(self, signal: dict, bars: pd.DataFrame, index: int, current_time: pd.Timestamp) -> None:
        screenshot_path = self.screenshot_capture.capture(
            candles=bars.iloc[max(0, index - 100): index + 1].set_index("timestamp"),
            timestamp=current_time.to_pydatetime() if hasattr(current_time, "to_pydatetime") else current_time,
            strategy_name=signal.get("strategy_name", "unknown"),
            event_type="signal_generated",
            metadata={"symbol": self.symbol, "side": signal.get("side"), "reason": signal.get("reason", "")},
        )
        if screenshot_path is not None:
            signal["entry_screenshot"] = str(screenshot_path)

    def _get_cooldown_seconds(self) -> int:
        if self._last_trade_won is True:
            return self.WIN_COOLDOWN_SECONDS
        sym_losses = self.risk_agent.consecutive_losses.get(self.base_symbol, 0)
        if sym_losses >= 2:
            return self.STREAK_COOLDOWN_SECONDS
        if sym_losses == 1:
            return self.LOSS_COOLDOWN_SECONDS
        return self.SIGNAL_COOLDOWN_SECONDS

    def _execute_signal_at_bar(self, signal: dict, row: pd.Series, current_time: pd.Timestamp) -> bool:
        current_time_str = current_time.isoformat()
        entry_price_base = float(row["open"])
        entry_price = (
            entry_price_base + (self.slippage_ticks * self.tick_size)
            if signal["side"] == "Buy"
            else entry_price_base - (self.slippage_ticks * self.tick_size)
        )
        signal["entry_price"] = entry_price
        max_pos = self.risk_agent.position_limits.get(self.base_symbol, {}).get("max_positions", 1)
        operational_filters_enabled = self.routing_mode not in ("optimizer_search", "optimizer_fast")
        existing_side = self.active_trades[0]["side"] if self.active_trades else ""
        route_context = {
            "now_dt": current_time.to_pydatetime() if hasattr(current_time, "to_pydatetime") else current_time,
            "market_open": True if not operational_filters_enabled else self.news_agent.is_market_hours(current_time),
            "news_standdown": False if not operational_filters_enabled else self.news_agent.is_stand_down(current_time),
            "near_close": False if not operational_filters_enabled else self.news_agent.is_near_close(current_time, lead_minutes=15),
            "active_count": len(self.active_trades),
            "existing_side": existing_side,
            "max_positions": max_pos,
            "pending_order": self.symbol in self.risk_agent.pending_orders,
            "last_signal_time": self.last_signal_time,
            "cooldown_seconds": self._get_cooldown_seconds(),
            "current_price": entry_price,
        }
        routed = self._route_signal(signal, route_context)
        if isinstance(routed, RouterRejection):
            self.telemetry.record_signal(
                self.symbol,
                signal["strategy_name"],
                f"blocked_{routed.stage}",
                signal["side"],
                current_time,
                float(row.get("close", row.get("price", 0)) or 0.0),
                routed.reason,
                signal.get("metadata", {}),
            )
            return False
        if routed.accepted:
            signal["quantity"] = routed.quantity
            self._open_trade(signal, current_time_str, entry_price)
            self._sync_risk_state()
            self.last_signal_time = current_time.timestamp()
            return True
        return False

    def _daily_profit_locked(self, current_time: pd.Timestamp) -> bool:
        return self._daily_profit_lock_reached(current_time, current_price=0.0, include_unrealized=False)

    def _realized_pnl_for_date(self, current_time: pd.Timestamp) -> float:
        session_key = self._session_key(current_time)
        realized = 0.0
        for trade in self.trades:
            closed_at = trade.get("time_close") or trade.get("timestamp")
            if closed_at is None:
                continue
            if self._session_key(closed_at) != session_key:
                continue
            realized += float(trade.get("pnl", 0.0) or 0.0)
        return realized

    def _daily_profit_lock_reached(self, current_time: pd.Timestamp, current_price: float, *, include_unrealized: bool) -> bool:
        if self.daily_profit_lock <= 0:
            return False
        realized = self._realized_pnl_for_date(current_time)
        if include_unrealized and self.active_trades:
            realized += sum(self._calculate_pnl(trade, current_price) for trade in self.active_trades)
            realized -= sum(2.0 * trade["qty"] * self.commission_per_contract for trade in self.active_trades)
        return realized >= self.daily_profit_lock

    def _route_signal(self, signal: dict, route_context: dict):
        if self.routing_mode != "optimizer_fast":
            return self.trade_router.route_trade(signal, route_context)

        is_flip = any(existing["side"] != signal["side"] for existing in self.active_trades)
        if not is_flip and int(route_context.get("active_count", 0)) >= int(route_context.get("max_positions", 1)):
            return RouterRejection("already at max positions", "filter", "max_positions")
        risk_hit, risk_reason = self.risk_agent.is_risk_limit_hit()
        if risk_hit:
            return RouterRejection(risk_reason or "risk limit hit", "risk")

        metadata = signal.get("metadata", {}) or {}
        quantity = self.risk_agent.get_entry_quantity(
            self.symbol,
            int(signal.get("quantity", self.strategy.default_qty)),
            sl_ticks=signal.get("sl_ticks"),
            current_price=route_context.get("current_price"),
            current_atr=metadata.get("atr"),
        )
        if quantity <= 0:
            return RouterRejection("quantity sizing returned 0", "size")
        if not self.risk_agent.validate_trade(self.symbol, quantity, signal["side"]):
            return RouterRejection("risk validation rejected trade", "risk")

        class _Accepted:
            accepted = True

            def __init__(self, qty: int):
                self.quantity = qty
                self.reason = "accepted"
                self.result = {"status": "simulation_fill", "avgPrice": signal.get("entry_price", 0)}

        return _Accepted(quantity)

    def _open_trade(self, signal, timestamp, entry_price):
        # FIX #H6: Net-Position Awareness for Backtester
        # If we have an existing trade on the opposite side, treat this as a CLOSE.
        for existing in list(self.active_trades):
            if existing["side"] != signal["side"]:
                self.active_trades.remove(existing)
                exit_price = (
                    entry_price - (self.slippage_ticks * self.tick_size)
                    if existing["side"] == "Buy"
                    else entry_price + (self.slippage_ticks * self.tick_size)
                )
                if existing["side"] == "Buy":
                    existing["mfe_price"] = max(float(existing.get("mfe_price", existing["entry"])), exit_price)
                    existing["mae_price"] = min(float(existing.get("mae_price", existing["entry"])), exit_price)
                else:
                    existing["mfe_price"] = min(float(existing.get("mfe_price", existing["entry"])), exit_price)
                    existing["mae_price"] = max(float(existing.get("mae_price", existing["entry"])), exit_price)
                pnl = self._calculate_final_pnl(existing, exit_price)
                timestamp_str = timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp)
                self._record_trade(existing, exit_price, pnl, "REVERSAL", timestamp_str)
                self.risk_agent.record_position_close(self.symbol)
                self.risk_agent.record_trade_result(pnl, symbol=self.symbol)
                self._last_trade_won = pnl > 0
                return

        metadata = dict(signal.get("metadata", {}) or {})
        sl_ticks = int(signal["sl_ticks"])
        tp_ticks = int(signal["tp_ticks"])
        stop_price = metadata.get("stop_price")
        if stop_price is not None and not pd.isna(stop_price):
            stop_price = float(stop_price)
            if signal["side"] == "Buy" and stop_price < entry_price:
                sl_ticks = max(1, int(np.ceil((entry_price - stop_price) / self.tick_size)))
            elif signal["side"] == "Sell" and stop_price > entry_price:
                sl_ticks = max(1, int(np.ceil((stop_price - entry_price) / self.tick_size)))
        target_price = metadata.get("target_price")
        target_r_multiple = metadata.get("target_r_multiple")
        if target_price is not None and not pd.isna(target_price):
            target_price = float(target_price)
            if signal["side"] == "Buy" and target_price > entry_price:
                tp_ticks = max(1, int(np.floor((target_price - entry_price) / self.tick_size)))
            elif signal["side"] == "Sell" and target_price < entry_price:
                tp_ticks = max(1, int(np.floor((entry_price - target_price) / self.tick_size)))
        elif target_r_multiple is not None and not pd.isna(target_r_multiple):
            tp_ticks = max(1, round(sl_ticks * float(target_r_multiple)))

        trade = {
            "side": signal["side"],
            "entry": entry_price,
            "qty": signal.get("quantity", 1),
            "sl_ticks": sl_ticks,
            "tp_ticks": tp_ticks,
            "trail_mult": signal.get("trail_stop_atr_mult", 0.0),
            "trail_trigger_ticks": signal.get("trail_trigger_ticks", 40),
            "time_open": timestamp,
            "be_triggered": False,
            "be_trigger_ticks": signal.get("be_trigger_ticks", 999),
            "mfe_price": entry_price,
            "mae_price": entry_price,
            "strategy_name": signal.get("strategy_name", "unknown"),
            "metadata": metadata,
            "entry_screenshot": signal.get("entry_screenshot"),
        }

        if trade["side"] == "Buy":
            trade["sl_price"] = entry_price - (trade["sl_ticks"] * self.tick_size)
            trade["tp_price"] = entry_price + (trade["tp_ticks"] * self.tick_size)
        else:
            trade["sl_price"] = entry_price + (trade["sl_ticks"] * self.tick_size)
            trade["tp_price"] = entry_price - (trade["tp_ticks"] * self.tick_size)

        self.active_trades.append(trade)
        self.risk_agent.record_position_open(self.symbol, trade["side"], trade["qty"], entry_price)
        self.telemetry.record_trade_open(
            self.symbol,
            trade["strategy_name"],
            trade["side"],
            trade["qty"],
            entry_price,
            metadata=trade["metadata"],
        )

    def _update_active_trades(self, row):
        open_p = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        atr = float(row.get("atr", 0) or 0)
        timestamp = pd.to_datetime(row.get("timestamp", pd.Timestamp.now("UTC")))
        timestamp_str = timestamp.isoformat()
        to_remove = []
        operational_filters_enabled = self.routing_mode not in ("optimizer_search", "optimizer_fast")
        news_event = operational_filters_enabled and self.news_agent.is_news_approaching(timestamp, lead_minutes=2)
        eod_flatten = operational_filters_enabled and self.news_agent.is_near_close(timestamp, lead_minutes=0)

        for trade in self.active_trades:
            if trade.pop("pending_daily_profit_lock_exit", False):
                exit_price = (
                    open_p - (self.slippage_ticks * self.tick_size)
                    if trade["side"] == "Buy"
                    else open_p + (self.slippage_ticks * self.tick_size)
                )
                pnl = self._calculate_final_pnl(trade, exit_price)
                self._record_trade(trade, exit_price, pnl, "DAILY-PROFIT-LOCK", timestamp_str)
                self.risk_agent.record_trade_result(pnl, symbol=self.symbol)
                self._last_trade_won = pnl > 0
                to_remove.append(trade)
                continue

            if trade.pop("pending_profit_wave_exit", False):
                exit_price = (
                    open_p - (self.slippage_ticks * self.tick_size)
                    if trade["side"] == "Buy"
                    else open_p + (self.slippage_ticks * self.tick_size)
                )
                pnl = self._calculate_final_pnl(trade, exit_price)
                self._record_trade(trade, exit_price, pnl, "PROFIT-WAVE-EXIT", timestamp_str)
                self.risk_agent.record_trade_result(pnl, symbol=self.symbol)
                self._last_trade_won = pnl > 0
                to_remove.append(trade)
                continue

            trade["mfe_price"] = max(trade["mfe_price"], high) if trade["side"] == "Buy" else min(trade["mfe_price"], low)
            trade["mae_price"] = min(trade["mae_price"], low) if trade["side"] == "Buy" else max(trade["mae_price"], high)

            # Evaluate intrabar SL and TP against active brackets BEFORE end-of-bar close updates
            is_sl_hit = (trade["side"] == "Buy" and low <= trade["sl_price"]) or (trade["side"] == "Sell" and high >= trade["sl_price"])
            is_tp_hit = (trade["side"] == "Buy" and high >= trade["tp_price"]) or (trade["side"] == "Sell" and low <= trade["tp_price"])

            if is_sl_hit:
                exit_price = (
                    trade["sl_price"] - (self.slippage_ticks * self.tick_size)
                    if trade["side"] == "Buy"
                    else trade["sl_price"] + (self.slippage_ticks * self.tick_size)
                )
                pnl = self._calculate_final_pnl(trade, exit_price)
                result = "BE" if trade["be_triggered"] else "SL"
                is_profitable_trail = (trade["side"] == "Buy" and exit_price > trade["entry"]) or (trade["side"] == "Sell" and exit_price < trade["entry"])
                if trade["be_triggered"] and trade["trail_mult"] > 0 and is_profitable_trail:
                    result = "TRAIL"
                self._record_trade(trade, exit_price, pnl, result, timestamp_str)
                self.risk_agent.record_trade_result(pnl, symbol=self.symbol)
                self._last_trade_won = pnl > 0
                to_remove.append(trade)
                continue

            if is_tp_hit:
                exit_price = (
                    trade["tp_price"] - (self.slippage_ticks * self.tick_size)
                    if trade["side"] == "Buy"
                    else trade["tp_price"] + (self.slippage_ticks * self.tick_size)
                )
                pnl = self._calculate_final_pnl(trade, exit_price)
                self._record_trade(trade, exit_price, pnl, "TP", timestamp_str)
                self.risk_agent.record_trade_result(pnl, symbol=self.symbol)
                self._last_trade_won = pnl > 0
                to_remove.append(trade)
                continue

            if eod_flatten:
                exit_price = (
                    close - (self.slippage_ticks * self.tick_size)
                    if trade["side"] == "Buy"
                    else close + (self.slippage_ticks * self.tick_size)
                )
                pnl = self._calculate_final_pnl(trade, exit_price)
                self._record_trade(trade, exit_price, pnl, "EOD-FLATTEN", timestamp_str)
                self.risk_agent.record_trade_result(pnl, symbol=self.symbol)
                self._last_trade_won = pnl > 0
                to_remove.append(trade)
                continue

            profit_ticks = (close - trade["entry"]) / self.tick_size if trade["side"] == "Buy" else (trade["entry"] - close) / self.tick_size

            if news_event:
                if profit_ticks > 30:
                    if not trade["be_triggered"]:
                        trade["be_triggered"] = True
                        trade["sl_price"] = trade["entry"]
                else:
                    exit_price = open_p
                    pnl = self._calculate_final_pnl(trade, exit_price)
                    self._record_trade(trade, exit_price, pnl, "PRO-EXIT", timestamp_str)
                    self.risk_agent.record_trade_result(pnl, symbol=self.symbol)
                    self._last_trade_won = pnl > 0
                    to_remove.append(trade)
                    continue

            if trade["trail_mult"] > 0 and profit_ticks >= trade["trail_trigger_ticks"] and atr > 0:
                if trade["side"] == "Buy":
                    new_trail = close - (atr * trade["trail_mult"])
                    trade["sl_price"] = max(trade["sl_price"], new_trail)
                else:
                    new_trail = close + (atr * trade["trail_mult"])
                    trade["sl_price"] = min(trade["sl_price"], new_trail)

            if not trade["be_triggered"] and profit_ticks >= trade["be_trigger_ticks"]:
                trade["be_triggered"] = True
                trade["sl_price"] = trade["entry"]

            if self._profit_wave_exit_triggered(trade, row):
                trade["pending_profit_wave_exit"] = True

            if self._daily_profit_lock_reached(timestamp, close, include_unrealized=True):
                trade["pending_daily_profit_lock_exit"] = True

        for trade in to_remove:
            self.active_trades.remove(trade)
            self.risk_agent.record_position_close(self.symbol)
        if to_remove:
            self._sync_risk_state()

    def _profit_wave_exit_triggered(self, trade: dict, row: pd.Series) -> bool:
        metadata = dict(trade.get("metadata", {}) or {})
        if not metadata.get("profit_wave_exit"):
            return False
        column = str(metadata.get("profit_wave_slow_column", "smm_ema21"))
        wave = row.get(column)
        close = float(row.get("close", row.get("price", 0.0)) or 0.0)
        if metadata.get("smm_exit_on_dashboard_reversal"):
            trend_switch = int(row.get("smm_trend_switch", 0) or 0)
            mfi = float(row.get("smm_mfi", 50.0) or 50.0)
            if trade["side"] == "Buy" and (trend_switch < 0 or mfi < 50.0):
                return True
            if trade["side"] == "Sell" and (trend_switch > 0 or mfi > 50.0):
                return True
        if wave is None or pd.isna(wave):
            return False
        if trade["side"] == "Buy":
            return close < float(wave)
        return close > float(wave)

    def _calculate_pnl(self, trade, current_price):
        points = (current_price - trade["entry"]) if trade["side"] == "Buy" else (trade["entry"] - current_price)
        ticks = points / self.tick_size
        return ticks * self.tick_value * trade["qty"]

    def _calculate_final_pnl(self, trade, exit_price):
        gross_pnl = self._calculate_pnl(trade, exit_price)
        round_trip_commission = 2.0 * trade["qty"] * self.commission_per_contract
        return gross_pnl - round_trip_commission

    def _record_trade(self, trade, exit_price, pnl, result, timestamp):
        entry_price = float(trade["entry"])
        mfe_price = float(trade.get("mfe_price", entry_price))
        mae_price = float(trade.get("mae_price", entry_price))
        if trade["side"] == "Buy":
            mfe_ticks = max(0.0, (mfe_price - entry_price) / self.tick_size) if self.tick_size > 0 else 0.0
            mae_ticks = max(0.0, (entry_price - mae_price) / self.tick_size) if self.tick_size > 0 else 0.0
        else:
            mfe_ticks = max(0.0, (entry_price - mfe_price) / self.tick_size) if self.tick_size > 0 else 0.0
            mae_ticks = max(0.0, (mae_price - entry_price) / self.tick_size) if self.tick_size > 0 else 0.0
        sl_ticks = max(1, int(trade.get("sl_ticks", 1) or 1))
        record = {
            "side": trade["side"],
            "entry": trade["entry"],
            "exit": exit_price,
            "pnl": pnl,
            "result": result,
            "time_open": trade["time_open"],
            "time_close": timestamp,
            "qty": trade["qty"],
            "sl_ticks": int(trade.get("sl_ticks", 0) or 0),
            "tp_ticks": int(trade.get("tp_ticks", 0) or 0),
            "mfe_ticks": round(mfe_ticks, 2),
            "mae_ticks": round(mae_ticks, 2),
            "mfe_r": round(mfe_ticks / sl_ticks, 2),
            "mae_r": round(mae_ticks / sl_ticks, 2),
            "strategy_name": trade.get("strategy_name", "unknown"),
            "metadata": trade.get("metadata", {}),
            "entry_screenshot": trade.get("entry_screenshot"),
        }
        self.trades.append(record)
        self.balance += pnl
        try:
            self.last_signal_time = pd.to_datetime(timestamp).timestamp()
        except (ValueError, TypeError):
            self.last_signal_time = None
        self._sync_risk_state()
        self.telemetry.record_trade_close(
            self.symbol,
            record["strategy_name"],
            record["side"],
            record["qty"],
            pnl,
            result,
            metadata=trade.get("metadata", {}),
        )

    def _build_strategy_breakdown(self):
        by_strategy = {}
        for trade in self.trades:
            name = trade.get("strategy_name", "unknown")
            bucket = by_strategy.setdefault(name, {"trades": 0, "wins": 0, "pnl": 0.0})
            bucket["trades"] += 1
            if trade["pnl"] > 0:
                bucket["wins"] += 1
            bucket["pnl"] += float(trade["pnl"])

        for bucket in by_strategy.values():
            trades = bucket["trades"]
            bucket["win_rate"] = round((bucket["wins"] / trades * 100) if trades else 0.0, 2)
            bucket["pnl"] = round(bucket["pnl"], 2)
        return by_strategy

    def _get_results(self):
        total_trades = len(self.trades)
        if total_trades == 0:
            return {
                "summary": {
                    "total_trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "scratches": 0,
                    "win_rate": 0.0,
                    "total_pnl": 0.0,
                    "final_balance": round(self.balance, 2),
                    "profit_factor": 0.0,
                    "max_drawdown": 0.0,
                    "avg_trade": 0.0,
                    "sharpe_ratio": 0.0,
                },
                "by_strategy": {},
                "trades": [],
                "equity_history": self.equity_history,
                "equity_curve": self.equity_curve,
            }

        wins = len([trade for trade in self.trades if trade["pnl"] > 0])
        losses = len([trade for trade in self.trades if trade["pnl"] < 0])
        scratches = len([trade for trade in self.trades if abs(trade["pnl"]) < 0.01])

        gross_profit = sum(trade["pnl"] for trade in self.trades if trade["pnl"] > 0)
        gross_loss = abs(sum(trade["pnl"] for trade in self.trades if trade["pnl"] < 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf

        total_pnl = sum(trade["pnl"] for trade in self.trades)
        avg_trade = total_pnl / total_trades

        equity_series = pd.Series(self.equity_history)
        rolling_max = equity_series.cummax()
        drawdowns = equity_series - rolling_max
        max_drawdown = drawdowns.min()

        returns = equity_series.pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252 * 400)) if returns.std() > 0 else 0

        return {
            "summary": {
                "total_trades": total_trades,
                "wins": wins,
                "losses": losses,
                "scratches": scratches,
                "win_rate": round((wins / total_trades) * 100, 2),
                "total_pnl": round(total_pnl, 2),
                "final_balance": round(self.balance, 2),
                "profit_factor": round(profit_factor, 2) if profit_factor != np.inf else "Inf",
                "max_drawdown": round(max_drawdown, 2),
                "avg_trade": round(avg_trade, 2),
                "sharpe_ratio": round(sharpe, 2),
            },
            "by_strategy": self._build_strategy_breakdown(),
            "trades": self.trades,
            "equity_history": self.equity_history,
            "equity_curve": self.equity_curve,
        }
