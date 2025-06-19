# config/strategy_config.py
"""
Centralized configuration management for backtesting framework.
All strategy, risk, and system parameters in one place.
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging

class BacktestConfig:
    """
    Centralized configuration manager for all backtesting parameters.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to JSON configuration file (optional)
        """
        self.logger = logging.getLogger("BacktestConfig")
        self.config = self._load_default_config()
        
        if config_file:
            self._load_config_file(config_file)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration settings."""
        return {            # System Configuration
            "system": {
                "data_pool_dir": "data/pools",
                "output_dir": "outputs",
                "num_processes": 4,
                "log_level": "INFO",
                "enable_profiling": False
            },
            
            # Data Configuration
            "data": {
                "providers": {
                    "default": "upstox",
                    "available": ["upstox", "zerodha"]
                },
                "timeframes": {
                    "default": ["1m"],
                    "available": ["1m", "5m", "15m", "30m", "1h", "day"]
                },
                "data_quality": {
                    "remove_outliers": True,
                    "outlier_threshold": 5,  # Standard deviations
                    "fill_gaps": True,
                    "gap_fill_method": "forward"
                }
            },
            
            # Strategy Configuration
            "strategies": {
                "default": ["mse"],
                "parameters": {
                    "mse": {
                        "macd_fast": 12,
                        "macd_slow": 26,
                        "macd_signal": 9,
                        "ema_short": 9,
                        "ema_long": 20,
                        "warmup_periods": 390  # minutes
                    }
                },
                "optimization": {
                    "enabled": False,
                    "method": "grid_search",  # or "bayesian", "genetic"
                    "metric": "sharpe_ratio",
                    "min_trades": 30,
                    "walk_forward": {
                        "enabled": False,
                        "in_sample_ratio": 0.7,
                        "step_size": 0.1
                    }
                }
            },
            
            # Risk Management Configuration
            "risk": {
                "enabled": True,
                "position_sizing": {
                    "method": "volatility_targeting",  # "kelly", "fixed_fractional", "equal_weight"
                    "max_position_size": 0.1,  # 10% of portfolio
                    "min_position_size": 0.01,  # 1% minimum
                    "volatility_target": 0.15,  # 15% annualized
                    "kelly_fraction": 0.25  # Kelly criterion cap
                },
                "portfolio_limits": {
                    "max_positions": 20,
                    "max_sector_exposure": 0.3,
                    "max_correlation": 0.7,
                    "max_leverage": 1.0,
                    "max_drawdown": 0.2  # 20% stop
                },
                "trade_controls": {
                    "stop_loss": 0.02,  # 2% stop loss
                    "take_profit": 0.05,  # 5% take profit
                    "trailing_stop": {
                        "enabled": False,
                        "trigger": 0.03,  # Activate after 3% profit
                        "distance": 0.01  # Trail by 1%
                    },
                    "time_stop": {
                        "enabled": False,
                        "max_holding_days": 30
                    }
                },
                "checks": {
                    "position_size": True,
                    "drawdown": True,
                    "concentration": True,
                    "liquidity": True,
                    "volatility": True,
                    "correlation": True
                }
            },
            
            # Transaction Cost Configuration
            "costs": {
                "enabled": True,
                "commission": {
                    "rate": 0.0003,  # 3 basis points
                    "minimum": 20,  # Minimum commission
                    "model": "percentage"  # or "per_share", "fixed"
                },
                "slippage": {
                    "model": "market_impact",  # or "fixed", "volatility_based"
                    "base_spread": 0.0001,  # 1 basis point
                    "impact_coefficient": 0.1,
                    "urgency_factor": 1.0
                },
                "market_impact": {
                    "temporary": {
                        "coefficient": 0.1,
                        "exponent": 0.5  # Square root model
                    },
                    "permanent": {
                        "coefficient": 0.05,
                        "exponent": 0.5
                    }
                },
                "borrow_cost": {
                    "enabled": False,
                    "rate": 0.02  # 2% annual for shorts
                }
            },
            
            # Bias Detection Configuration
            "bias_detection": {
                "enabled": True,
                "checks": {
                    "look_ahead": True,
                    "survivorship": True,
                    "data_mining": True,
                    "selection": True
                },
                "look_ahead": {
                    "strict_mode": True,
                    "check_indicators": True,
                    "check_signals": True
                },
                "survivorship": {
                    "use_point_in_time_data": False,
                    "delisted_data_path": None
                }
            },
            
            # Options Configuration
            "options": {
                "enabled": False,
                "use_synthetic": True,
                "pricing_model": "black_scholes",
                "volatility": {
                    "model": "historical",  # or "implied", "garch"
                    "window": 20,
                    "min_volatility": 0.1,
                    "max_volatility": 2.0
                },
                "risk_free_rate": 0.05,
                "dividend_yield": 0.02,
                "american_exercise": False,
                "greeks_calculation": True
            },
            
            # Reporting Configuration
            "reporting": {
                "metrics": {
                    "basic": ["total_return", "win_rate", "profit_factor"],
                    "risk_adjusted": ["sharpe_ratio", "sortino_ratio", "calmar_ratio"],
                    "drawdown": ["max_drawdown", "max_drawdown_duration", "recovery_time"],
                    "risk": ["var_95", "cvar_95", "downside_deviation"],
                    "stability": ["stability_of_returns", "tail_ratio", "common_sense_ratio"]
                },
                "visualizations": {
                    "equity_curve": True,
                    "drawdown_chart": True,
                    "returns_distribution": True,
                    "monthly_returns": True,
                    "rolling_metrics": True,
                    "trade_analysis": True,
                    "risk_dashboard": True
                },
                "output_formats": ["csv", "json", "html"],
                "save_trades": True,
                "save_signals": True,
                "save_metrics": True
            },
            
            # Performance Configuration
            "performance": {
                "chunk_size": 10000,  # Rows to process at once
                "cache_enabled": True,
                "cache_dir": ".cache",
                "parallel_processing": True,
                "memory_limit": "4GB",
                "profile_enabled": False
            }
        }
    
    def _load_config_file(self, config_file: str) -> None:
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                self._merge_configs(self.config, user_config)
                self.logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            self.logger.error(f"Failed to load config file {config_file}: {e}")
            raise
    
    def _merge_configs(self, base: Dict, update: Dict) -> None:
        """Recursively merge configuration dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Examples:
            config.get('risk.position_sizing.max_position_size')
            config.get('strategies.parameters.mse.macd_fast')
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def save(self, file_path: str) -> None:
        """Save current configuration to JSON file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.config, f, indent=2)
                self.logger.info(f"Saved configuration to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
            raise
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate configuration for consistency and completeness.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required paths exist
        data_dir = Path(self.get('system.data_pool_dir'))
        if not data_dir.exists():
            errors.append(f"Data directory does not exist: {data_dir}")
        
        # Validate risk parameters
        if self.get('risk.portfolio_limits.max_drawdown') <= 0:
            errors.append("Max drawdown must be positive")
        
        if self.get('risk.position_sizing.max_position_size') > 1:
            errors.append("Max position size cannot exceed 100%")
        
        # Validate transaction costs
        if self.get('costs.commission.rate') < 0:
            errors.append("Commission rate cannot be negative")
        
        # Validate options parameters
        if self.get('options.enabled'):
            if self.get('options.risk_free_rate') < 0:
                errors.append("Risk-free rate cannot be negative")
            
            if self.get('options.volatility.min_volatility') <= 0:
                errors.append("Minimum volatility must be positive")
        
        return len(errors) == 0, errors
    
    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """Get complete configuration for a specific strategy."""
        return {
            'parameters': self.get(f'strategies.parameters.{strategy_name}', {}),
            'risk': self.get('risk'),
            'costs': self.get('costs'),
            'bias_detection': self.get('bias_detection'),
            'reporting': self.get('reporting')
        }
    
    def create_template(self, file_path: str) -> None:
        """Create a configuration template file."""
        template = {
            "description": "Backtesting configuration template",
            "system": {
                "data_pool_dir": "path/to/data",
                "output_dir": "path/to/output"
            },
            "strategies": {
                "default": ["your_strategy"],
                "parameters": {
                    "your_strategy": {
                        "param1": 0,
                        "param2": 0
                    }
                }
            },
            "risk": {
                "enabled": True,
                "position_sizing": {
                    "method": "volatility_targeting",
                    "max_position_size": 0.1
                }
            },
            "costs": {
                "enabled": True,
                "commission": {
                    "rate": 0.0003
                }
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(template, f, indent=2)
        
        self.logger.info(f"Created configuration template at {file_path}")


# Global configuration instance
BACKTEST_CONFIG = BacktestConfig()


def load_config(config_file: Optional[str] = None) -> BacktestConfig:
    """
    Load configuration from file or environment.
    
    Priority order:
    1. Specified config file
    2. Environment variable BACKTEST_CONFIG_FILE
    3. Default config.json in current directory
    4. Built-in defaults
    """
    if config_file:
        return BacktestConfig(config_file)
    
    # Check environment variable
    env_config = os.environ.get('BACKTEST_CONFIG_FILE')
    if env_config and os.path.exists(env_config):
        return BacktestConfig(env_config)
    
    # Check default location
    default_config = 'config.json'
    if os.path.exists(default_config):
        return BacktestConfig(default_config)
    
    # Return with defaults
    return BacktestConfig()