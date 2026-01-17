# config/unified_config.py
from __future__ import annotations
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


def _build_indicator_map(raw_spec: Optional[Dict[str, Any]]) -> Dict[str, List[IndicatorSpec]]:
    indicator_map: Dict[str, List[IndicatorSpec]] = {"entry": [], "exit": []}
    if not raw_spec:
        return indicator_map
    for role, specs in raw_spec.items():
        cleaned = []
        for spec in specs or []:
            cleaned.append(spec if isinstance(spec, IndicatorSpec) else IndicatorSpec(**spec))
        indicator_map[role] = cleaned
    return indicator_map


def _build_exit_config(raw_exit: Optional[Dict[str, Any]]) -> ExitConfig:
    if not raw_exit:
        return ExitConfig()
    exit_dict = raw_exit.copy()
    if 'stop_loss' in exit_dict:
        exit_dict['stop_loss'] = ThresholdConfig(**exit_dict['stop_loss'])
    if 'take_profit' in exit_dict:
        exit_dict['take_profit'] = ThresholdConfig(**exit_dict['take_profit'])
    if 'timeout' in exit_dict:
        exit_dict['timeout'] = TimeoutConfig(**exit_dict['timeout'])
    if 'square_off' in exit_dict:
        exit_dict['square_off'] = SquareOffConfig(**exit_dict['square_off'])
    return ExitConfig(**exit_dict)


@dataclass
class TimeframeConfig:
    entry: List[str] = field(default_factory=lambda: ['1m'])
    exit: List[str] = field(default_factory=list)
    confirmation: List[str] = field(default_factory=list)


@dataclass
class IndicatorSpec:
    name: str
    type: str
    timeframe: str = '1m'
    role: str = 'entry'
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThresholdConfig:
    enabled: bool = False
    type: str = 'percent'
    value: float = 0.02
    indicator: Optional[str] = None
    operator: str = 'gte'
    multiplier: float = 1.0
    timeframe: Optional[str] = None


@dataclass
class TimeoutConfig:
    enabled: bool = False
    max_minutes: int = 0
    intraday_cutoff: Optional[str] = None


@dataclass
class SquareOffConfig:
    mode: str = 'none'
    intraday_cutoff: str = '15:20'
    delivery_horizon_days: int = 0


@dataclass
class ExitConfig:
    mode: str = 'manual'
    stop_loss: ThresholdConfig = field(default_factory=ThresholdConfig)
    take_profit: ThresholdConfig = field(default_factory=lambda: ThresholdConfig(enabled=False, value=0.04))
    timeout: TimeoutConfig = field(default_factory=TimeoutConfig)
    square_off: SquareOffConfig = field(default_factory=SquareOffConfig)

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
    """Configuration for strategy parameters and declarative behavior."""
    name: str = "open_source_baseline"
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    description: str = "Open source baseline trend + momentum strategy"
    risk_profile: str = "moderate"  # conservative, moderate, aggressive
    initial_capital: float = 1000000.0  # Default 1M capital
    timeframes: TimeframeConfig = field(default_factory=TimeframeConfig)
    indicators: Dict[str, List[IndicatorSpec]] = field(default_factory=lambda: {"entry": [], "exit": []})
    exit: ExitConfig = field(default_factory=ExitConfig)
    
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
    save_visualizations: bool = True  # Legacy alias for save_plots
    save_base_data: bool = True  # Save base data for analysis
    save_signals: bool = False   # Legacy flag (no-op but retained for compatibility)
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
    performance_logging: bool = False
    trade_logging: bool = False


@dataclass
class FetchConfig:
    """Configuration for broker data fetching."""
    timeframes: List[str] = field(default_factory=lambda: ['1m'])
    min_chunks_before_abort: int = 2
    failure_threshold: float = 0.8  # proportion of failed chunks before aborting ticker
    max_retries: int = 5
    validate_symbols: bool = True

@dataclass
class BrokerConfig:
    """Configuration for broker connections and data providers."""
    # Data Provider Settings
    default_provider: str = "upstox"
    available_providers: List[str] = field(default_factory=lambda: ["upstox", "zerodha", "binance"])
    
    # Upstox Configuration (full V3 API support)
    upstox_client_id: str = ""  # Set via environment: ${UPSTOX_CLIENT_ID}
    upstox_client_secret: str = ""  # Set via environment: ${UPSTOX_CLIENT_SECRET}
    upstox_redirect_uri: str = "https://127.0.0.1:5000/"
    upstox_auth_url: str = "https://api.upstox.com/v2/login/authorization/dialog"
    upstox_token_url: str = "https://api.upstox.com/v2/login/authorization/token"
    upstox_historical_api_base: str = "https://api.upstox.com/v3/historical-candle"
    upstox_expiry_api_url: str = "https://api.upstox.com/v2/expired-instruments/expiries"
    upstox_api_version: str = "v3"
    upstox_max_days_per_request: int = 200
    upstox_max_retries: int = 3
    upstox_retry_delay: int = 5
    upstox_request_timeout: int = 30
    
    # Upstox V3 API timeframe mappings (unit + interval format)
    upstox_timeframe_mappings: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        '1m': {'unit': 'minutes', 'interval': '1'},
        '2m': {'unit': 'minutes', 'interval': '2'},
        '3m': {'unit': 'minutes', 'interval': '3'},
        '5m': {'unit': 'minutes', 'interval': '5'},
        '10m': {'unit': 'minutes', 'interval': '10'},
        '15m': {'unit': 'minutes', 'interval': '15'},
        '30m': {'unit': 'minutes', 'interval': '30'},
        '1h': {'unit': 'hours', 'interval': '1'},
        '2h': {'unit': 'hours', 'interval': '2'},
        'day': {'unit': 'days', 'interval': '1'},
        'week': {'unit': 'weeks', 'interval': '1'},
        'month': {'unit': 'months', 'interval': '1'},
        '1minute': {'unit': 'minutes', 'interval': '1'},  # Legacy
        '30minute': {'unit': 'minutes', 'interval': '30'}  # Legacy
    })
    
    upstox_supported_timeframes: List[str] = field(default_factory=lambda: [
        '1m', '2m', '3m', '5m', '10m', '15m', '30m', '1h', '2h', 'day', 'week', 'month'
    ])
    
    # Zerodha Configuration
    zerodha_api_key: str = ""  # Set via environment: ${ZERODHA_API_KEY}
    zerodha_api_secret: str = ""  # Set via environment: ${ZERODHA_API_SECRET}
    zerodha_redirect_uri: str = "https://127.0.0.1:5000/"
    zerodha_segment: str = "NSE"
    zerodha_max_retries: int = 3
    zerodha_retry_delay: int = 5
    zerodha_request_timeout: int = 30
    
    zerodha_supported_timeframes: List[str] = field(default_factory=lambda: [
        'minute', '3minute', '5minute', '10minute', '15minute', '30minute', 'hour', 'day', 'week', 'month'
    ])
    
    zerodha_historical_limits: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        'minute': {'days': 60, 'candles_per_request': 60},
        '3minute': {'days': 100, 'candles_per_request': 100},
        '5minute': {'days': 100, 'candles_per_request': 100},
        '15minute': {'days': 200, 'candles_per_request': 200},
        '30minute': {'days': 200, 'candles_per_request': 200},
        'hour': {'days': 400, 'candles_per_request': 400},
        'day': {'days': 2000, 'candles_per_request': 2000},
        'week': {'days': 2000, 'candles_per_request': 2000},
        'month': {'days': 2000, 'candles_per_request': 2000}
    })
    
    # Binance Configuration
    binance_api_key: str = ""  # Public data doesn't require auth
    binance_api_secret: str = ""
    binance_testnet: bool = True
    
    binance_supported_timeframes: List[str] = field(default_factory=lambda: [
        '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '1d', '1w', '1M'
    ])
    
    # Token Management (unified across providers)
    token_dir: str = "config/access_tokens"
    token_refresh_enabled: bool = True
    auto_refresh_minutes: int = 60
    
    # Standard timeframe mappings for cross-provider compatibility
    standard_timeframes: Dict[str, Dict[str, Optional[str]]] = field(default_factory=lambda: {
        '1m': {'upstox': '1minute', 'zerodha': 'minute'},
        '3m': {'upstox': None, 'zerodha': '3minute'},
        '5m': {'upstox': None, 'zerodha': '5minute'},
        '10m': {'upstox': None, 'zerodha': '10minute'},
        '15m': {'upstox': None, 'zerodha': '15minute'},
        '30m': {'upstox': '30minute', 'zerodha': '30minute'},
        '1h': {'upstox': None, 'zerodha': 'hour'},
        'day': {'upstox': 'day', 'zerodha': 'day'},
        'week': {'upstox': 'week', 'zerodha': 'week'},
        'month': {'upstox': 'month', 'zerodha': 'month'}
    })
    
    # Timeframe folder naming convention
    timeframe_folders: Dict[str, str] = field(default_factory=lambda: {
        '1m': '1minute',
        '3m': '3minute',
        '5m': '5minute',
        '10m': '10minute',
        '15m': '15minute',
        '30m': '30minute',
        '1h': 'hour',
        'day': 'day',
        'week': 'week',
        'month': 'month'
    })
    
    # Instruments CSV file paths
    upstox_instruments_csv: str = "config/complete.csv"
    zerodha_instruments_csv: str = "config/zerodha_instruments.csv"

@dataclass
class AuditComplianceConfig:
    """Configuration for audit compliance requirements."""
    # From analysis report - enforce audit requirements
    warmup_minutes: int = 525  # Always 525 min (35├ù15-min bars)
    use_previous_bar: bool = True  # Always previous bar
    enable_two_bar_rule: bool = True  # Always two-bar rule
    cascade_prevention: bool = True  # Default enabled
    
    # Exit thresholds (can be overridden by strategy params)
    default_exit_threshold: float = 0.8  # 80% by default
    
    # Validation requirements
    enforce_compliance: bool = True  # Enforce all audit requirements
    compliance_checks: List[str] = field(default_factory=lambda: [
        "warmup_duration", "previous_bar_usage", "two_bar_execution", 
        "cascade_prevention", "parameter_parity"
    ])
    
    def validate_audit_compliance(self) -> List[str]:
        """Validate configuration meets audit requirements."""
        errors = []
        
        if self.warmup_minutes < 525:
            errors.append("Warmup period must be at least 525 minutes (audit requirement)")
        
        if not self.use_previous_bar:
            errors.append("Must use previous bar indicators (audit requirement)")
            
        if not self.enable_two_bar_rule:
            errors.append("Two-bar execution rule must be enabled (audit requirement)")
        
        return errors
    
@dataclass
class BacktestConfig:
    """Unified configuration for the entire backtesting system."""
    # Component configurations
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    transaction: TransactionConfig = field(default_factory=TransactionConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    fetch: FetchConfig = field(default_factory=FetchConfig)
    timeframes: List[str] = field(default_factory=lambda: ['1m'])
    fetch_max_retries: int = 5
    fetch_failure_threshold: float = 0.5
    fetch_min_chunks_before_abort: int = 5
    fetch_validate_symbols: bool = True
    replay_manifest: Optional[str] = None
    
    # Infrastructure configurations (absorbed from config.py)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    
    # Audit compliance configurations (from analysis report)
    compliance: AuditComplianceConfig = field(default_factory=AuditComplianceConfig)
    
    # Global settings
    base_dir: str = str(Path(__file__).resolve().parent.parent)
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_label: Optional[str] = None
    exit_template_path: Optional[str] = None
    
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
        
        # Validate audit compliance (from analysis report requirements)
        if self.compliance.enforce_compliance:
            compliance_errors = self.compliance.validate_audit_compliance()
            errors.extend(compliance_errors)
            
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
            
    def ensure_audit_compliance(self):
        """Enforce audit report requirements - override any non-compliant settings."""
        if self.compliance.enforce_compliance:
            # Force audit-compliant settings
            self.compliance.warmup_minutes = max(self.compliance.warmup_minutes, 525)
            self.compliance.use_previous_bar = True
            self.compliance.enable_two_bar_rule = True
            
            # Update any related strategy parameters
            if hasattr(self.strategy, 'parameters'):
                if 'warmup_minutes' in self.strategy.parameters:
                    self.strategy.parameters['warmup_minutes'] = max(
                        self.strategy.parameters.get('warmup_minutes', 525), 525
                    )
            
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
            strategy_data = data['strategy'].copy()
            timeframes = strategy_data.pop('timeframes', None)
            indicators = strategy_data.pop('indicators', None)
            exit_plan = strategy_data.pop('exit', None)

            strategy_data['timeframes'] = TimeframeConfig(**timeframes) if timeframes else TimeframeConfig()
            strategy_data['indicators'] = _build_indicator_map(indicators)
            strategy_data['exit'] = _build_exit_config(exit_plan)

            config_kwargs['strategy'] = StrategyConfig(**strategy_data)
        if 'risk' in data:
            config_kwargs['risk'] = RiskConfig(**data['risk'])
        if 'transaction' in data:
            config_kwargs['transaction'] = TransactionConfig(**data['transaction'])
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
        if 'fetch' in data:
            config_kwargs['fetch'] = FetchConfig(**data['fetch'])
        if 'broker' in data:
            config_kwargs['broker'] = BrokerConfig(**data['broker'])
        if 'compliance' in data:
            config_kwargs['compliance'] = AuditComplianceConfig(**data['compliance'])
            
        # Add any remaining top-level keys
        for key, value in data.items():
            if key not in ['data', 'strategy', 'risk', 'transaction', 'validation', 
                          'optimization', 'execution', 'output', 'logging', 'fetch', 'broker', 'compliance']:
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
            if key == 'timeframes':
                if isinstance(value, TimeframeConfig):
                    self.config.strategy.timeframes = value
                else:
                    self.config.strategy.timeframes = TimeframeConfig(**value)
            elif key == 'indicators':
                self.config.strategy.indicators = _build_indicator_map(value)
            elif key == 'exit':
                self.config.strategy.exit = value if isinstance(value, ExitConfig) else _build_exit_config(value)
            elif hasattr(self.config.strategy, key):
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

    def with_validation_config(self, **kwargs) -> 'ConfigBuilder':
        """Configure validation settings."""
        for key, value in kwargs.items():
            if hasattr(self.config.validation, key):
                setattr(self.config.validation, key, value)
        return self
        
    def build(self) -> BacktestConfig:
        """Build and return the final configuration."""
        return self.config

# Predefined configuration templates
def get_minimal_config() -> BacktestConfig:
    """Get a minimal risk configuration for learning and testing."""
    return (ConfigBuilder()
            .with_strategy_config(name="mse", risk_profile="minimal")
            .with_risk_config(
                max_position_size=0.05,      # 5% max position
                max_daily_loss=0.01,         # 1% daily loss limit
                stop_loss_pct=0.02,          # 2% stop loss
                take_profit_pct=0.04,        # 4% take profit
                max_concurrent_positions=2   # Very limited positions
            )
            .with_validation_config(enabled=True, strict_mode=True)
            .build())

def get_conservative_config() -> BacktestConfig:
    """Get a conservative trading configuration."""
    return ConfigBuilder().with_conservative_risk().build()

def get_aggressive_config() -> BacktestConfig:
    """Get an aggressive trading configuration."""
    return ConfigBuilder().with_aggressive_risk().build()

    return (ConfigBuilder()
            .build())

def get_debug_config() -> BacktestConfig:
    """Get a debug configuration for pure strategy testing - NO RISK MANAGEMENT."""
    return (ConfigBuilder()
            .with_strategy_config(name="mse", risk_profile="debug") 
            .with_risk_config(
                enabled=False,
                bypass_mode=True,
                max_position_size=1.0,
                max_daily_loss=1.0,
                max_drawdown=1.0,
                stop_loss_pct=0.0,
                take_profit_pct=0.0,
                enable_stop_loss=False,
                enable_take_profit=False,
                enable_timeout=False
            )
            .with_validation_config(
                enabled=False,
                lookahead_bias_check=False,
                survivorship_bias_check=False,
                strict_mode=False
            )
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
