# config/unified_config.py
"""
Unified Configuration System for Backtester

This module provides a comprehensive configuration management system that:
- Centralizes all configuration in a single place
- Provides validation and type checking
- Supports YAML-based configuration files
- Implements the Builder pattern for easy configuration
- Provides templates for different trading styles
"""

import yaml
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime, timedelta
import json

@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    data_pool_dir: str = "data/pools"
    timeframe_folders: Dict[str, str] = field(default_factory=lambda: {
        "1minute": "1minute",
        "5minute": "5minute", 
        "15minute": "15minute",
        "1hour": "1hour",
        "1day": "1day"
    })
    default_timeframe: str = "1minute"
    required_columns: List[str] = field(default_factory=lambda: [
        "timestamp", "open", "high", "low", "close", "volume"
    ])
    date_format: str = "%Y-%m-%d"
    timezone: str = "Asia/Kolkata"
    
@dataclass
class StrategyConfig:
    """Configuration for strategy parameters."""
    name: str = "mse"
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    description: str = ""
    risk_profile: str = "moderate"  # conservative, moderate, aggressive
    initial_capital: float = 1000000.0  # Default 1M capital
    
@dataclass
class RiskConfig:
    """Configuration for risk management."""
    enabled: bool = True            # Enable/disable risk management completely
    bypass_mode: bool = False       # Bypass all risk checks (for debugging/analysis)
    max_position_size: float = 0.1  # 10% of portfolio
    max_daily_loss: float = 0.02    # 2% daily loss limit
    max_drawdown: float = 0.15      # 15% maximum drawdown
    max_concentration: float = 0.5  # 50% maximum concentration per ticker
    stop_loss_pct: float = 0.05     # 5% stop loss
    take_profit_pct: float = 0.10   # 10% take profit
    position_timeout_minutes: int = 240  # 4 hours
    enable_stop_loss: bool = True
    enable_take_profit: bool = True
    enable_timeout: bool = True
    
@dataclass
class TransactionConfig:
    """Configuration for transaction costs."""
    enabled: bool = True  # Enable transaction cost modeling
    model_type: str = "advanced"  # basic, advanced, broker_specific
    brokerage_rate: float = 0.0003  # 0.03%
    fixed_cost: float = 0.0
    slippage_rate: float = 0.0001   # 0.01%
    market_impact_factor: float = 0.1
    enable_market_impact: bool = True
    
@dataclass
class OptionsConfig:
    """Configuration for options trading."""
    enabled: bool = False
    synthetic_enabled: bool = True
    volatility_model: str = "black_scholes"  # black_scholes, heston
    interest_rate: float = 0.05
    dividend_yield: float = 0.0
    days_to_expiry: int = 30
    strike_range: float = 0.1  # 10% around current price
    greeks_calculation: bool = True
    
@dataclass
class ValidationConfig:
    """Configuration for data validation and bias detection."""
    enabled: bool = True
    lookahead_bias_check: bool = True
    survivorship_bias_check: bool = True
    data_quality_check: bool = True
    min_data_points: int = 100
    max_missing_data_pct: float = 0.05  # 5%
    price_outlier_threshold: float = 3.0  # 3 standard deviations
    strict_mode: bool = False  # Enable strict validation mode
    
@dataclass
class OptimizationConfig:
    """Configuration for strategy optimization."""
    enabled: bool = False
    method: str = "grid_search"  # grid_search, random_search, bayesian
    max_iterations: int = 100
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    
@dataclass
class OutputConfig:
    """Configuration for output generation."""
    save_trades: bool = True
    save_metrics: bool = True
    save_plots: bool = True
    save_base_data: bool = True  # Save base data for analysis
    output_dir: str = "outputs"
    trade_file_format: str = "csv"  # csv, parquet, json
    base_file_format: str = "csv"  # csv, parquet, json
    metrics_file_format: str = "json"
    plot_format: str = "png"  # png, pdf, svg
    
    # Visualization trade source configuration
    visualization_trade_source: str = "auto"  # "strategy_trades", "risk_approved_trades", "auto"
    
@dataclass
class ExecutionConfig:
    """Configuration for execution and parallel processing."""
    parallel_processing: bool = True
    max_workers: int = 4
    cache_enabled: bool = True
    cache_dir: str = "cache"
    timeout_seconds: int = 3600  # 1 hour default timeout

@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    file_enabled: bool = True
    console_enabled: bool = True
    log_dir: str = "logs"
    max_file_size: str = "10MB"
    backup_count: int = 5
    
@dataclass
class BacktestConfig:
    """Unified configuration for the entire backtesting system."""
    # Component configurations
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    transaction: TransactionConfig = field(default_factory=TransactionConfig)
    options: OptionsConfig = field(default_factory=OptionsConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Global settings
    base_dir: str = str(Path(__file__).resolve().parent.parent)
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    # Legacy properties for backward compatibility
    @property
    def parallel_processing(self) -> bool:
        return self.execution.parallel_processing
    
    @property
    def max_workers(self) -> int:
        return self.execution.max_workers
    
    @property
    def cache_enabled(self) -> bool:
        return self.execution.cache_enabled
    
    @property
    def cache_dir(self) -> str:
        return self.execution.cache_dir
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        self.validate()
        self.setup_paths()
        
    def validate(self):
        """Validate configuration parameters."""
        errors = []
        
        # Validate risk parameters
        if not 0 <= self.risk.max_position_size <= 1:
            errors.append("max_position_size must be between 0 and 1")
        if not 0 <= self.risk.max_daily_loss <= 1:
            errors.append("max_daily_loss must be between 0 and 1")
        if not 0 <= self.risk.max_drawdown <= 1:
            errors.append("max_drawdown must be between 0 and 1")
            
        # Validate transaction parameters
        if self.transaction.brokerage_rate < 0:
            errors.append("brokerage_rate must be non-negative")
        if self.transaction.slippage_rate < 0:
            errors.append("slippage_rate must be non-negative")
            
        # Validate validation parameters
        if self.validation.min_data_points <= 0:
            errors.append("min_data_points must be positive")
        if not 0 <= self.validation.max_missing_data_pct <= 1:
            errors.append("max_missing_data_pct must be between 0 and 1")
            
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
            
    def setup_paths(self):
        """Set up required directory paths."""
        base_path = Path(self.base_dir)
        
        # Ensure all required directories exist
        required_dirs = [
            self.data.data_pool_dir,
            self.output.output_dir,
            self.logging.log_dir,
            self.execution.cache_dir
        ]
        
        for dir_path in required_dirs:
            full_path = base_path / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
        
    def to_yaml(self, file_path: Optional[str] = None) -> str:
        """Export configuration to YAML format."""
        yaml_content = yaml.dump(self.to_dict(), default_flow_style=False, indent=2)
        
        if file_path:
            with open(file_path, 'w') as f:
                f.write(yaml_content)
                
        return yaml_content
        
    @classmethod
    def from_yaml(cls, file_path: str) -> 'BacktestConfig':
        """Load configuration from YAML file."""
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
            
        return cls.from_dict(data)
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BacktestConfig':
        """Create configuration from dictionary."""
        # Create nested configurations
        config_kwargs = {}
        
        if 'data' in data:
            config_kwargs['data'] = DataConfig(**data['data'])
        if 'strategy' in data:
            config_kwargs['strategy'] = StrategyConfig(**data['strategy'])
        if 'risk' in data:
            config_kwargs['risk'] = RiskConfig(**data['risk'])
        if 'transaction' in data:
            config_kwargs['transaction'] = TransactionConfig(**data['transaction'])
        if 'options' in data:
            config_kwargs['options'] = OptionsConfig(**data['options'])
        if 'validation' in data:
            config_kwargs['validation'] = ValidationConfig(**data['validation'])
        if 'optimization' in data:
            config_kwargs['optimization'] = OptimizationConfig(**data['optimization'])
        if 'execution' in data:
            config_kwargs['execution'] = ExecutionConfig(**data['execution'])
        if 'output' in data:
            config_kwargs['output'] = OutputConfig(**data['output'])
        if 'logging' in data:
            config_kwargs['logging'] = LoggingConfig(**data['logging'])
            
        # Add any remaining top-level keys
        for key, value in data.items():
            if key not in config_kwargs:
                config_kwargs[key] = value
                
        return cls(**config_kwargs)

class ConfigBuilder:
    """Builder pattern for creating BacktestConfig instances."""
    
    def __init__(self):
        self.config = BacktestConfig()
        
    def with_data_config(self, **kwargs) -> 'ConfigBuilder':
        """Configure data settings."""
        for key, value in kwargs.items():
            if hasattr(self.config.data, key):
                setattr(self.config.data, key, value)
        return self
        
    def with_strategy_config(self, **kwargs) -> 'ConfigBuilder':
        """Configure strategy settings."""
        for key, value in kwargs.items():
            if hasattr(self.config.strategy, key):
                setattr(self.config.strategy, key, value)
        return self
    
    def with_risk_config(self, **kwargs) -> 'ConfigBuilder':
        """Configure risk settings."""
        for key, value in kwargs.items():
            if hasattr(self.config.risk, key):
                setattr(self.config.risk, key, value)
        return self
    
    def with_conservative_risk(self) -> 'ConfigBuilder':
        """Apply conservative risk settings."""
        self.config.strategy.risk_profile = "conservative"
        self.config.risk = RiskConfig(
            max_position_size=0.05,  # 5%
            max_daily_loss=0.01,     # 1%
            max_drawdown=0.10,       # 10%
            stop_loss_pct=0.03,      # 3%
            take_profit_pct=0.06,    # 6%
            position_timeout_minutes=120
        )
        return self
    
    def with_aggressive_risk(self) -> 'ConfigBuilder':
        """Apply aggressive risk settings."""
        self.config.strategy.risk_profile = "aggressive"
        self.config.risk = RiskConfig(
            max_position_size=0.20,  # 20%
            max_daily_loss=0.05,     # 5%
            max_drawdown=0.25,       # 25%
            stop_loss_pct=0.08,      # 8%
            take_profit_pct=0.15,    # 15%
            position_timeout_minutes=480
        )
        return self
        
    def with_options_enabled(self, **kwargs) -> 'ConfigBuilder':
        """Enable options trading with configuration."""
        self.config.options.enabled = True
        for key, value in kwargs.items():
            if hasattr(self.config.options, key):
                setattr(self.config.options, key, value)
        return self
        
    def build(self) -> BacktestConfig:
        """Build and return the final configuration."""
        return self.config

# Predefined configuration templates
def get_minimal_config() -> BacktestConfig:
    """Get a minimal risk configuration for learning and testing."""
    return (ConfigBuilder()            .with_strategy_config(name="mse", risk_profile="minimal")
            .with_risk_config(
                max_position_size=0.05,      # 5% max position
                max_daily_loss=0.01,         # 1% daily loss limit
                stop_loss_pct=0.02,          # 2% stop loss
                take_profit_pct=0.04,        # 4% take profit
                max_concurrent_positions=2   # Very limited positions
            )
            .build())

def get_conservative_config() -> BacktestConfig:
    """Get a conservative trading configuration."""
    return ConfigBuilder().with_conservative_risk().build()

def get_aggressive_config() -> BacktestConfig:
    """Get an aggressive trading configuration."""
    return ConfigBuilder().with_aggressive_risk().build()

def get_options_config() -> BacktestConfig:
    """Get a configuration with options trading enabled."""
    return (ConfigBuilder()
            .with_options_enabled(synthetic_enabled=True, greeks_calculation=True)
            .build())

# Standard calculation definitions for market consistency
MARKET_STANDARD_CALCULATIONS = {
    "MACD": {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9,
        "formula": "EMA(12) - EMA(26), Signal: EMA(9) of MACD"
    },
    "RSI": {
        "period": 14,
        "method": "wilders",  # Wilder's smoothing method
        "formula": "100 - (100 / (1 + RS)), RS = Average Gain / Average Loss"
    },
    "Bollinger_Bands": {
        "period": 20,
        "std_dev": 2,
        "formula": "Middle: SMA(20), Upper: Middle + 2*StdDev, Lower: Middle - 2*StdDev"
    },
    "Stochastic": {
        "k_period": 14,
        "d_period": 3,
        "smooth": 3,
        "formula": "%K = (Close - Low14) / (High14 - Low14) * 100, %D = SMA3(%K)"
    },
    "ATR": {
        "period": 14,
        "method": "wilders",
        "formula": "Average True Range using Wilder's smoothing"
    },
    "EMA": {
        "alpha_formula": "2 / (period + 1)",
        "formula": "EMA = (Close * Alpha) + (Previous_EMA * (1 - Alpha))"
    },
    "SMA": {
        "formula": "Sum of Close prices / Period"
    }
}

def get_calculation_standard(indicator: str) -> Dict[str, Any]:
    """Get market standard calculation parameters for an indicator."""
    return MARKET_STANDARD_CALCULATIONS.get(indicator.upper(), {})

# Example usage and testing
if __name__ == "__main__":
    # Test configuration creation
    config = ConfigBuilder().with_conservative_risk().build()
    print("Conservative config created successfully")
    
    # Test YAML export/import
    yaml_content = config.to_yaml()
    print("YAML export successful")
    
    # Test validation
    try:
        config.validate()
        print("Configuration validation passed")
    except ValueError as e:
        print(f"Validation error: {e}")
