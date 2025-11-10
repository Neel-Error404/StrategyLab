import pytest
from pathlib import Path

from src.core.options.replay.config import OptionsReplayConfig


def _minimal_config_dict():
    return {
        "inputs": {
            "equity_trades_path": "outputs/20251006_024924/mse/2022-01-01_to_2025-08-31/data/all_trades_merged.csv",
            "equity_data_root": "data/pools",
            "options_data_root": "data/pools/options",
            "underlying_timeframe": "1minute",
            "options_timeframe": ["1minute", "5minute", "1day"],
        },
        "pricing": {
            "mode": "hybrid",
            "synthetic": {},
            "actual": {},
        },
        "strike_selection": {"method": "atm"},
        "expiry_selection": {"method": "nearest_weekly"},
        "option_type": {
            "long_signal": "CE",
            "short_signal": "PE",
            "strategy": "directional",
        },
        "lot_sizing": {"method": "fixed", "fixed": {"lots_per_trade": 1}},
        "position_management": {
            "entry": {"min_dte_to_enter": 3, "max_dte_to_enter": 45, "skip_if_illiquid": True},
            "exit": {"follow_equity_signal": True},
        },
        "liquidity": {
            "min_open_interest": 10,
            "max_spread_pct": 0.1,
            "min_volume": 1,
            "on_filter_fail": "skip_trade",
        },
        "greeks": {"calculate": True, "metrics": ["delta"], "frequency": "every_bar"},
        "risk": {
            "initial_portfolio_value": 1000000,
            "max_portfolio_allocation": 0.3,
            "max_concurrent_positions": 5,
            "max_position_size_per_trade": 0.05,
            "max_drawdown_pct": 0.2,
            "stop_trading_on_drawdown": False,
        },
        "data_quality": {},
        "output": {"output_dir": "outputs"},
        "visualization": {},
        "logging": {},
        "performance": {},
        "validation": {},
    }


def test_options_replay_config_from_dict(tmp_path: Path):
    config_dict = _minimal_config_dict()
    repo_root = Path(__file__).resolve().parents[3]  # project root
    config = OptionsReplayConfig.from_dict(config_dict, base_dir=repo_root)
    assert config.inputs.equity_trades_path.is_absolute()
    assert config.pricing.mode == "hybrid"
    assert config.strike_selection.method == "atm"
    assert config.risk.max_portfolio_allocation == pytest.approx(0.3)
    assert config.inputs.options_timeframes == ("1minute", "5minute", "1day")
    assert config.inputs.options_timeframe == "1minute"
    assert config.config_hash
