# strategies/support/strategy_base.py
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from datetime import datetime, timedelta
import warnings

try:
    from config.unified_config import (
        StrategyConfig,
        TimeframeConfig,
        ExitConfig,
        MARKET_STANDARD_CALCULATIONS,
    )
except ImportError:
    StrategyConfig = None
    TimeframeConfig = None
    ExitConfig = None
    MARKET_STANDARD_CALCULATIONS = {}

from .indicator_registry import IndicatorRegistry
from .exit_manager import ExitManager

class StrategyBase(ABC):
    """
    Enhanced abstract base class for all trading strategies.
    
    Provides:
    - Common interface and functionality
    - Market-standard indicator calculations  
    - Configuration management
    - Performance monitoring
    - Error handling and validation
    - Extensible signal generation framework
    """
    
    def __init__(self, name: str, parameters: Dict[str, Any] = None, config: Optional[StrategyConfig] = None):
        """
        Initialize the strategy with enhanced configuration support.
        
        Args:
            name: Strategy name
            parameters: Dictionary of strategy parameters (legacy support)
            config: StrategyConfig object for unified configuration
        """
        self.name = name
        self.parameters = parameters or {}
        self.config = config
        self.logger = logging.getLogger(f"strategy.{name}")
        self.timeframe_config: Optional[TimeframeConfig] = TimeframeConfig() if TimeframeConfig else None
        self.indicator_specs: Dict[str, List[Dict[str, Any]]] = {"entry": [], "exit": []}
        self.exit_config: Optional[ExitConfig] = ExitConfig() if ExitConfig else None
        self.indicator_registry: Optional[IndicatorRegistry] = None
        self.exit_manager: Optional[ExitManager] = None
        self._required_timeframes: Optional[List[str]] = None
        
        if config:
            self.apply_strategy_config(config)
        else:
            self.risk_profile = "moderate"
            self.description = ""
            
        # Strategy metadata
        self.version = "1.0"
        self.description = getattr(config, 'description', '') if config else ""
        self.required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        self.warmup_period = self.parameters.get('warmup_period', 50)
        
        # Performance tracking
        self.last_execution_time = None
        self.total_signals_generated = 0
        self.errors_encountered = []
        
        # Market standard calculations cache
        self._indicator_cache = {}
        
        self.logger.info(f"Strategy {name} initialized with {self.risk_profile} risk profile")
        self.logger.info(f"Strategy requires timeframes: {self.required_timeframes}")

    def apply_strategy_config(self, strategy_config: StrategyConfig) -> None:
        if not strategy_config:
            return

        self.config = strategy_config
        self.risk_profile = strategy_config.risk_profile
        self.description = strategy_config.description

        if strategy_config.parameters:
            self.parameters.update(strategy_config.parameters)

        self.timeframe_config = strategy_config.timeframes or TimeframeConfig()
        self.indicator_specs = strategy_config.indicators or {"entry": [], "exit": []}
        self.exit_config = strategy_config.exit or ExitConfig()

        derived_frames = self._derive_required_timeframes()
        # Only override existing requirements when config explicitly declares frames
        if derived_frames:
            self._required_timeframes = derived_frames
        self.indicator_registry = IndicatorRegistry(self, self.indicator_specs)
        self.exit_manager = ExitManager(self.exit_config)
    
    @property
    def required_timeframes(self) -> List[str]:
        if self._required_timeframes is None:
            return []
        return self._required_timeframes
    
    @required_timeframes.setter
    def required_timeframes(self, timeframes: List[str]) -> None:
        """
        Set the required timeframes for this strategy.
        
        Args:
            timeframes: List of timeframe strings
        """
        self._required_timeframes = timeframes
        self.logger.info(f"Strategy {self.name} timeframe requirements updated: {timeframes}")

    def _derive_required_timeframes(self) -> List[str]:
        if not self.timeframe_config:
            return []
        frames = set(self.timeframe_config.entry or [])
        frames.update(self.timeframe_config.exit or [])
        frames.update(self.timeframe_config.confirmation or [])
        return list(frames) if frames else []

    def _validate_timeframes(self, available_timeframes: List[str]) -> bool:
        """
        Ensure that all declared required timeframes are present in the loaded data.

        Multi-timeframe strategies can override this for custom validation,
        but the default implementation keeps single-timeframe strategies safe
        when users request additional frames via CLI overrides.
        """
        required = self.required_timeframes or []
        if not required:
            self.logger.error(
                "No required timeframes declared for %s; strategy cannot execute without explicit frames",
                self.name,
            )
            return False
        missing = set(required) - set(available_timeframes)
        if missing:
            self.logger.error(
                "Missing required timeframes for %s: %s (available=%s)",
                self.name,
                sorted(missing),
                sorted(available_timeframes),
            )
            return False
        return True

    def _apply_configured_indicators(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]):
        if not self.indicator_registry:
            return data
        return self.indicator_registry.apply(data)
    
    @abstractmethod
    def prepare_data(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], ticker: str, pull_date: str) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Prepare data for the strategy. This includes calculating indicators,
        applying warmup periods, and any other data preparation steps.
        
        Args:
            data: Single DataFrame (legacy) or Dict of timeframe DataFrames (multi-timeframe)
                  e.g., {'1m': df1, '5m': df5, '15m': df15}
            ticker: Ticker symbol
            pull_date: Date for which the data is being prepared
            
        Returns:
            Single DataFrame (legacy) or Dict of prepared DataFrames (multi-timeframe)
        """
        pass
    
    @abstractmethod
    def generate_signals(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
        """
        Generate entry and exit signals based on the prepared data.
        
        Args:
            data: Single DataFrame (legacy) or Dict of prepared timeframe DataFrames
                  e.g., {'1m': prepared_df1, '5m': prepared_df5}
            
        Returns:
            DataFrame with entry and exit signals
        """
        pass
    
    def execute(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], ticker: str, pull_date: str) -> pd.DataFrame:
        """
        Execute the strategy on the provided data.
        
        Args:
            data: Single DataFrame (legacy) or Dict of timeframe DataFrames (multi-timeframe)
                  e.g., {'1m': df1, '5m': df5, '15m': df15}
            ticker: Ticker symbol
            pull_date: Date for which the strategy is being executed
            
        Returns:
            DataFrame with signals and indicators
        """
        self.logger.info(f"Executing strategy {self.name} for {ticker} on {pull_date}")
        
        # Handle both legacy single DataFrame and new multi-timeframe Dict
        if isinstance(data, pd.DataFrame):
            # Legacy single DataFrame mode
            self.logger.debug(f"Strategy {self.name} executing in legacy single-timeframe mode")
            prepared_data = self.prepare_data(data.copy(), ticker, pull_date)
            
            if prepared_data is None or (isinstance(prepared_data, pd.DataFrame) and prepared_data.empty):
                self.logger.warning(f"No data available after preparation for {ticker} on {pull_date}")
                return pd.DataFrame()
                
            prepared_data = self._apply_configured_indicators(prepared_data)
            with_signals = self.generate_signals(prepared_data)
            
        else:
            # Multi-timeframe mode
            timeframes = list(data.keys())
            self.logger.info(f"Strategy {self.name} executing in multi-timeframe mode with: {timeframes}")
            
            # Validate that provided timeframes match strategy requirements
            if not self._validate_timeframes(timeframes):
                raise ValueError(f"Provided timeframes {timeframes} don't match strategy requirements {self.required_timeframes}")
            
            # Prepare multi-timeframe data
            prepared_data = self.prepare_data(data, ticker, pull_date)
            
            if prepared_data is None:
                self.logger.warning(f"No data available after preparation for {ticker} on {pull_date}")
                return pd.DataFrame()
                
            prepared_data = self._apply_configured_indicators(prepared_data)
            with_signals = self.generate_signals(prepared_data)
        
        self.logger.info(f"Strategy execution completed for {ticker} on {pull_date}")
        return with_signals
    
    def optimize_parameters(self, df: pd.DataFrame, ticker: str, pull_date: str,
                            param_grid: Dict[str, List[Any]], metric: str = 'profit') -> Dict[str, Any]:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            df: DataFrame with OHLCV data
            ticker: Ticker symbol
            pull_date: Date for which the strategy is being optimized
            param_grid: Dictionary of parameter names and possible values
            metric: Metric to optimize ('profit', 'win_rate', etc.)
            
        Returns:
            Dictionary with optimized parameters
        """
        self.logger.info(f"Optimizing parameters for {self.name} strategy on {ticker}")
        
        # Generate all parameter combinations
        import itertools
        param_combinations = list(itertools.product(*param_grid.values()))
        param_keys = list(param_grid.keys())
        
        best_score = float('-inf')
        best_params = None
        
        # Test each parameter combination
        for params in param_combinations:
            param_dict = {param_keys[i]: params[i] for i in range(len(param_keys))}
            self.parameters = param_dict
            
            try:
                # Execute strategy with these parameters
                result_df = self.execute(df.copy(), ticker, pull_date)
                
                # Extract trades
                from src.strat_stats.strategy_executor import extract_trades
                trades = extract_trades(result_df, exit_manager=getattr(self, 'exit_manager', None))
                
                # Calculate metrics
                from src.strat_stats.statistics import calculate_metrics
                metrics = calculate_metrics(trades)
                
                # Get score based on the specified metric
                if metric == 'profit':
                    score = metrics.get('Average Profit (%)', 0)
                elif metric == 'win_rate':
                    score = metrics.get('Accuracy (%)', 0)
                else:
                    score = 0
                
                # Update best parameters if this combination is better
                if score > best_score:
                    best_score = score
                    best_params = param_dict.copy()
                    
                self.logger.debug(f"Parameters {param_dict} - Score: {score}")
                
            except Exception as e:
                self.logger.error(f"Error optimizing with parameters {param_dict}: {e}")
        
        self.logger.info(f"Best parameters: {best_params} with score: {best_score}")
        
        # Reset to best parameters
        self.parameters = best_params
        
        return best_params
    
    # Market-standard indicator calculations
    def calculate_sma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average using market standard."""
        return series.rolling(window=period).mean()
    
    def calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average using market standard (2/(period+1))."""
        alpha = 2 / (period + 1)
        return series.ewm(alpha=alpha, adjust=False).mean()
    
    def calculate_macd(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD using market standard parameters (12-26-9)."""
        ema_fast = self.calculate_ema(series, fast)
        ema_slow = self.calculate_ema(series, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI using Wilder's method (market standard)."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Use Wilder's smoothing (alpha = 1/period)
        alpha = 1.0 / period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_bollinger_bands(self, series: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands using market standard (20, 2)."""
        middle = self.calculate_sma(series, period)
        std = series.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                           k_period: int = 14, d_period: int = 3, smooth: int = 3) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator using market standard."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        k_percent_smooth = k_percent.rolling(window=smooth).mean()
        d_percent = k_percent_smooth.rolling(window=d_period).mean()
        
        return {
            'k_percent': k_percent_smooth,
            'd_percent': d_percent
        }
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range using Wilder's method."""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Use Wilder's smoothing
        alpha = 1.0 / period
        atr = true_range.ewm(alpha=alpha, adjust=False).mean()
        return atr
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate that the DataFrame has required columns and sufficient data."""
        try:
            # Check required columns
            missing_columns = [col for col in self.required_columns if col not in df.columns]
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            # Check for sufficient data
            if len(df) < self.warmup_period:
                self.logger.warning(f"Insufficient data: {len(df)} rows, need at least {self.warmup_period}")
                return False
            
            # Check for data quality
            if df[['open', 'high', 'low', 'close', 'volume']].isnull().any().any():
                self.logger.warning("Data contains null values")
                return False
            
            # Validate price relationships
            invalid_prices = (df['high'] < df['low']) | (df['close'] > df['high']) | (df['close'] < df['low'])
            if invalid_prices.any():
                self.logger.error("Invalid price relationships detected")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating data: {e}")
            return False
    
    def apply_warmup_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply warmup period to remove initial indicator calculation artifacts."""
        if len(df) <= self.warmup_period:
            self.logger.warning("Data length is less than or equal to warmup period")
            return df
        
        return df.iloc[self.warmup_period:].copy()
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get parameter value with fallback to default."""
        return self.parameters.get(key, default)
    
    def set_parameter(self, key: str, value: Any) -> None:
        """Set parameter value."""
        self.parameters[key] = value
        self.logger.debug(f"Parameter {key} set to {value}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get strategy performance metrics."""
        return {
            'name': self.name,
            'version': self.version,
            'risk_profile': self.risk_profile,
            'last_execution_time': self.last_execution_time,
            'total_signals_generated': self.total_signals_generated,
            'errors_encountered': len(self.errors_encountered),
            'parameters': self.parameters.copy()
        }
    
    def reset_performance_tracking(self) -> None:
        """Reset performance tracking metrics."""
        self.last_execution_time = None
        self.total_signals_generated = 0
        self.errors_encountered = []
        self.logger.info("Performance tracking reset")
    
    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"Strategy({self.name}, risk_profile={self.risk_profile}, params={len(self.parameters)})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"Strategy(name='{self.name}', version='{self.version}', risk_profile='{self.risk_profile}', parameters={self.parameters})"
