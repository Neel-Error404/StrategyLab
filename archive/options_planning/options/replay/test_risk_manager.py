from datetime import datetime

import pandas as pd

from src.core.options.replay.config import RiskConfig, RiskKillSwitchConfig
from src.core.options.replay.risk import RiskManager


def _risk_config() -> RiskConfig:
    return RiskConfig(
        initial_portfolio_value=1_000_000,
        max_portfolio_allocation=0.2,
        max_concurrent_positions=2,
        max_position_size_per_trade=0.05,
        max_drawdown_pct=0.2,
        stop_trading_on_drawdown=False,
        kill_switch=RiskKillSwitchConfig(
            enabled=True,
            max_intraday_loss_pct=0.1,
            max_single_trade_loss_pct=0.05,
            reason_codes=True,
        ),
    )


def test_risk_manager_rejects_oversized_trade():
    rm = RiskManager(_risk_config())
    ts = pd.Timestamp(datetime(2024, 1, 1, 9, 30), tz="Asia/Kolkata")
    allowed, reasons = rm.evaluate_entry("T1", entry_cost=80_000, timestamp=ts, ticker="TEST")
    assert allowed is False
    assert "max_position_size_per_trade" in reasons


def test_risk_manager_kill_switch_triggered_on_loss():
    rm = RiskManager(_risk_config())
    ts = pd.Timestamp(datetime(2024, 1, 1, 9, 30), tz="Asia/Kolkata")
    rm.register_entry("T1", entry_cost=40_000, quantity=100, timestamp=ts, ticker="TEST")
    events = rm.register_exit("T1", realized_pnl=-120_000, timestamp=ts, ticker="TEST")
    assert any(event.event_type == "kill_switch" for event in events)
    assert rm.kill_switch_triggered is True
