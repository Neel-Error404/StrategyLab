# strategies/strategy_base.py
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from datetime import datetime, timedelta
import warnings

# Import the unified configuration system
try:
    from config.unified_config import StrategyConfig, MARKET_STANDARD_CALCULATIONS
except ImportError:
    # Fallback for cases where unified config is not available
    StrategyConfig = None
    MARKET_STANDARD_CALCULATIONS = {}

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
        
        # Merge configuration if provided
        if config:
            # Override parameters with config values
            if config.parameters:
                self.parameters.update(config.parameters)
            self.risk_profile = config.risk_profile
        else:
            self.risk_profile = "moderate"
            
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
    
    @abstractmethod
    def prepare_data(self, df: pd.DataFrame, ticker: str, pull_date: str) -> pd.DataFrame:
        """
        Prepare data for the strategy. This includes calculating indicators,
        applying warmup periods, and any other data preparation steps.
        
        Args:
            df: DataFrame with OHLCV data
            ticker: Ticker symbol
            pull_date: Date for which the data is being prepared
            
        Returns:
            DataFrame with prepared data including indicators
        """
        pass
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate entry and exit signals based on the prepared data.
        
        Args:
            df: DataFrame with prepared data and indicators
            
        Returns:
            DataFrame with entry and exit signals
        """
        pass
    
    def execute(self, df: pd.DataFrame, ticker: str, pull_date: str) -> pd.DataFrame:
        """
        Execute the strategy on the provided data.
        
        Args:
            df: DataFrame with OHLCV data
            ticker: Ticker symbol
            pull_date: Date for which the strategy is being executed
            
        Returns:
            DataFrame with signals and indicators
        """
        self.logger.info(f"Executing strategy {self.name} for {ticker} on {pull_date}")
        
        # Prepare data (calculate indicators, etc.)
        prepared_df = self.prepare_data(df.copy(), ticker, pull_date)
        
        if prepared_df is None or prepared_df.empty:
            self.logger.warning(f"No data available after preparation for {ticker} on {pull_date}")
            return pd.DataFrame()
            
        # Generate signals
        with_signals = self.generate_signals(prepared_df)
        
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
                trades = extract_trades(result_df)
                
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