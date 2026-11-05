from decimal import Decimal


from src.agents.risk_agent import RiskAgent


def _make_agent(daily_profit_target=None, daily_loss_halt=None):
    agent = RiskAgent(account_id="TEST")
    # Override config-derived halts for deterministic tests.
    agent.daily_profit_target = (
        Decimal(str(daily_profit_target)) if daily_profit_target is not None else None
    )
    agent.daily_loss_halt = (
        Decimal(str(daily_loss_halt)) if daily_loss_halt is not None else None
    )
    # Ensure neither MLL nor legacy daily-loss false-positives during halt tests.
    agent.max_daily_loss = Decimal("999999")
    agent.max_drawdown = Decimal("999999")
    agent.minimum_balance = Decimal("0")
    agent.current_balance = Decimal("50000")
    agent.peak_balance = Decimal("50000")
    return agent


def test_validate_trade_allows_opposite_side_flip_at_max_positions():
    agent = _make_agent()
    sym = "MNQM6"
    agent.position_limits["MNQ"] = {"max_positions": 1}
    agent.open_positions[sym] = [{"side": "Buy", "qty": 1, "entry": Decimal("20000")}]
    # An opposite-side intent against an existing position is a flip — must pass.
    assert agent.validate_trade(sym, 1, "Sell") is True


def test_validate_trade_rejects_same_side_at_max_positions():
    agent = _make_agent()
    sym = "MNQM6"
    agent.position_limits["MNQ"] = {"max_positions": 1}
    agent.open_positions[sym] = [{"side": "Buy", "qty": 1, "entry": Decimal("20000")}]
    # Same-side intent should still be blocked by max-positions.
    assert agent.validate_trade(sym, 1, "Buy") is False


def test_validate_trade_allows_first_entry_when_flat():
    agent = _make_agent()
    sym = "MNQM6"
    agent.position_limits["MNQ"] = {"max_positions": 1}
    agent.open_positions[sym] = []
    assert agent.validate_trade(sym, 1, "Buy") is True


def test_is_risk_limit_hit_fires_on_daily_profit_target():
    agent = _make_agent(daily_profit_target=1400)
    agent.current_daily_pnl = Decimal("1401")
    hit, reason = agent.is_risk_limit_hit()
    assert hit is True
    assert "Daily profit target reached" in reason


def test_is_risk_limit_hit_fires_on_daily_loss_halt():
    agent = _make_agent(daily_loss_halt=900)
    agent.current_daily_pnl = Decimal("-901")
    hit, reason = agent.is_risk_limit_hit()
    assert hit is True
    assert "Daily loss halt" in reason


def test_is_risk_limit_hit_not_fired_within_thresholds():
    agent = _make_agent(daily_profit_target=1400, daily_loss_halt=900)
    agent.current_daily_pnl = Decimal("1399")
    hit, _ = agent.is_risk_limit_hit()
    assert hit is False
    agent.current_daily_pnl = Decimal("-899")
    hit, _ = agent.is_risk_limit_hit()
    assert hit is False


def test_volatility_sizing_scales_quantity():
    agent = _make_agent()
    agent.feature_flags["instrument_sizing_enabled"] = True
    agent.feature_flags["volatility_sizing_enabled"] = True
    agent.routing_config["risk_per_trade_pct"] = 1.0  # 1% risk per trade
    agent.current_balance = Decimal("10000")
    # Equity = 10000, 1% = 100 risk budget
    
    agent.contract_meta["MESU6"] = {"tickSize": 0.25, "tickValue": 1.25}
    agent.symbol_brackets["MES"] = {"min_atr": 1.0, "max_atr": 3.0} # typical = 2.0
    
    # Cost per contract with 40 tick stop: 40 * 1.25 = 50.
    # Without volatility scaling: contracts = int(100 / 50) = 2.
    
    # 1. Volatility is equal to typical ATR (2.0) -> Multiplier = 1.0
    qty = agent.get_entry_quantity("MESU6", 10, sl_ticks=40, current_atr=2.0)
    assert qty == 2
    
    # 2. Volatility is higher (4.0) -> typical/atr = 2.0/4.0 = 0.5 -> Scaled budget = 50
    # contracts = int(50 / 50) = 1.
    qty_high_vol = agent.get_entry_quantity("MESU6", 10, sl_ticks=40, current_atr=4.0)
    assert qty_high_vol == 1
    
    # 3. Volatility is lower (1.0) -> typical/atr = 2.0/1.0 = 2.0 -> capped at 1.0 -> Scaled budget = 100
    # contracts = int(100 / 50) = 2.
    qty_low_vol = agent.get_entry_quantity("MESU6", 10, sl_ticks=40, current_atr=1.0)
    assert qty_low_vol == 2
