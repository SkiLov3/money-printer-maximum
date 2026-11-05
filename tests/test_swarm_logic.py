import pandas as pd
from src.agents.risk_agent import RiskAgent
from src.agents.strategy_agent import StrategyAgent

def test_risk_validation_approve():
    """Test that a clean trade is approved."""
    risk_agent = RiskAgent("TEST-ACCOUNT")
    risk_agent.update_account_state(50000, 0, 0)
    
    # Should approve: no positions, within limits
    assert risk_agent.validate_trade("NQ", 1, "Buy") == True

def test_risk_validation_reject_max_positions():
    """Test that exceeding max positions is rejected (uses open_positions tracking)."""
    risk_agent = RiskAgent("TEST-ACCOUNT")
    risk_agent.update_account_state(50000, 0, 0)
    
    # Register NQ as a full-size contract (max 1 position)
    risk_agent.register_contract("NQ", {
        "tickSize": 0.25, "tickValue": 5.0,
        "description": "E-mini NASDAQ", "symbolId": "F.US.NQ"
    })
    
    # Record one open position
    risk_agent.record_position_open("NQ", "Buy", 1)
    
    # Should reject: already at max positions (1/1)
    assert risk_agent.validate_trade("NQ", 1, "Buy") == False

def test_risk_validation_reject_daily_loss():
    """Test that exceeding daily loss limit is rejected."""
    risk_agent = RiskAgent("TEST-ACCOUNT")
    risk_agent.update_account_state(50000, -1500, 0)  # Loss > $1000 default limit
    
    assert risk_agent.validate_trade("NQ", 1, "Buy") == False

def test_risk_entry_quantity_rejects_when_one_contract_exceeds_budget():
    """Test that sizing returns zero when one contract cannot fit the risk budget."""
    risk_agent = RiskAgent("TEST-ACCOUNT")
    risk_agent.update_account_state(50000, 0, 0)
    
    # Register as full-size (max 1 qty per position)
    risk_agent.register_contract("NQ", {
        "tickSize": 0.25, "tickValue": 5.0,
        "description": "E-mini NASDAQ", "symbolId": "F.US.NQ"
    })
    
    # 100 NQ ticks risks $500/contract; budget at 0.75% of $50k is $375.
    # One contract at $500 exceeds the budget, so it returns 0.
    assert risk_agent.get_entry_quantity("NQ", requested_qty=3, sl_ticks=100) == 0

def test_risk_drawdown_reject():
    """Test that max drawdown triggers rejection."""
    risk_agent = RiskAgent("TEST-ACCOUNT")
    risk_agent.update_account_state(52000, 0, 0)  # Set peak to 52000
    risk_agent.update_account_state(49500, -500, 0)  # Drawdown = 52000 - 49500 = 2500 > 2000
    
    assert risk_agent.validate_trade("NQ", 1, "Buy") == False

def test_risk_dynamic_quantity_protect_gains():
    """Test that approaching daily profit target (80%) reduces size."""
    risk_agent = RiskAgent("TEST-ACCOUNT")
    assert risk_agent.daily_profit_target is not None, "daily_profit_target must be set in trading_stack.yaml routing config"
    target_pnl = float(risk_agent.daily_profit_target) * 0.85
    risk_agent.update_account_state(50000, target_pnl, 0)
    
    result = risk_agent.get_dynamic_quantity("NQ", 3)
    assert result == 1  # max(1, int(3 * 0.5)) = 1

def test_risk_dynamic_quantity_stop_bleed():
    """Test loss safeguard reduces to minimum."""
    risk_agent = RiskAgent("TEST-ACCOUNT")
    risk_agent.update_account_state(50000, -600, 0)  # -600 < -500 (50% of $1000)
    
    result = risk_agent.get_dynamic_quantity("NQ", 3)
    assert result == 1

def test_risk_alpha_scaling():
    """Test streak detection applies modest dynamic sizing before entry-budget caps."""
    risk_agent = RiskAgent("TEST-ACCOUNT")
    risk_agent.update_account_state(50000, 200, 0)  # Positive daily PnL
    
    # Register micro contract limits
    risk_agent.register_contract("MNQ", {
        "tickSize": 0.25, "tickValue": 0.5,
        "description": "Micro NASDAQ", "symbolId": "F.US.MNQ"
    })
    
    # Record 3 winning trades
    risk_agent.record_trade_result(50.0)
    risk_agent.record_trade_result(75.0)
    risk_agent.record_trade_result(30.0)
    
    result = risk_agent.get_dynamic_quantity("MNQ", 3)
    assert result == 4

def test_strategy_symbol_extraction():
    """Test base symbol extraction from resolved contract names."""
    assert StrategyAgent._extract_base_static(
"MNQM6") == "MNQ"
    assert StrategyAgent._extract_base_static(
"ESM6") == "ES"
    assert StrategyAgent._extract_base_static(
"MCLK6") == "MCL"
    assert StrategyAgent._extract_base_static(
"NQM6") == "NQ"
    assert StrategyAgent._extract_base_static(
"GCM6") == "GC"
    assert StrategyAgent._extract_base_static(
"6EU6") == "6E"

def test_strategy_analysis_insufficient_data():
    """Test that analysis returns None with insufficient data."""
    strategy = StrategyAgent("NQM6")
    data = pd.DataFrame({'price': [100] * 50})  # Only 50 rows, need 200
    
    assert strategy.analyze(data) is None

def test_strategy_analysis_with_data():
    """Test that strategy can process 200+ bars without crashing."""
    strategy = StrategyAgent("NQM6")
    
    # Create 250 bars of synthetic data
    data = pd.DataFrame({
        'price': [100 + i * 0.1 for i in range(250)],
        'high': [100.5 + i * 0.1 for i in range(250)],
        'low': [99.5 + i * 0.1 for i in range(250)],
        'close': [100 + i * 0.1 for i in range(250)],
        'volume': [1000] * 250,
        'timestamp': pd.date_range('2026-03-01', periods=250, freq='1min')
    })
    
    # Should not crash — signal may or may not trigger
    signal = strategy.analyze(data)
    if signal:
        assert signal['symbol'] == "NQM6"
        assert signal['side'] in ["Buy", "Sell"]
        assert 'sl_ticks' in signal
        assert 'tp_ticks' in signal

def test_risk_position_tracking():
    """Test position open/close tracking."""
    risk_agent = RiskAgent("TEST")
    risk_agent.register_contract("MNQ", {
        "tickSize": 0.25, "tickValue": 0.5,
        "description": "Micro NASDAQ", "symbolId": "F.US.MNQ"
    })
    risk_agent.update_account_state(50000, 0, 0)
    
    # Open position
    risk_agent.record_position_open("MNQ", "Buy", 3)
    assert len(risk_agent.open_positions["MNQ"]) == 1
    
    # Close position (FIFO)
    risk_agent.record_position_close("MNQ")
    assert len(risk_agent.open_positions["MNQ"]) == 0
    
    # Should approve again
    assert risk_agent.validate_trade("MNQ", 3, "Buy") == True
