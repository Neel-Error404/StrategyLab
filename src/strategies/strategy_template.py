"""
Strategy Template for Backtesting Framework
==========================================

This template provides a complete guide for creating new trading strategies.
Copy this file and implement your strategy logic following the patterns shown.

Required Components:
1. Strategy class inheriting from StrategyBase
2. Parameter configuration with validation
3. Signal generation logic
4. Risk management integration
5. Strategy registration

Author: Generated Template
Version: 1.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from abc import abstractmethod

# Import the base strategy class
from .strategy_base import StrategyBase


class TemplateStrategy(StrategyBase):
    """
    Template Strategy Implementation
    
    This template demonstrates how to create a new strategy by:
    1. Inheriting from StrategyBase
    2. Implementing required methods
    3. Using built-in indicators and utilities
    4. Following risk management practices
    
    Replace 'Template' with your strategy name throughout this file.
    """
    
    def __init__(self, 
                 # Strategy-specific parameters (customize these)
                 lookback_period: int = 20,
                 threshold: float = 0.02,
                 min_volume: int = 1000,
                 max_positions: int = 5,
                 
                 # Standard parameters (usually keep these)
                 warmup_period: int = 50,
                 risk_free_rate: float = 0.02,
                 **kwargs):
        """
        Initialize your strategy with parameters.
        
        PARAMETER GUIDE:
        ===============
        Strategy-Specific Parameters (Customize these):
        - lookback_period: How many periods to look back for calculations
        - threshold: Decision threshold for buy/sell signals
        - min_volume: Minimum volume required for trades
        - max_positions: Maximum number of concurrent positions
        
        Standard Parameters (Usually keep as-is):
        - warmup_period: Number of periods needed before generating signals
        - risk_free_rate: Risk-free rate for Sharpe ratio calculations
        
        Add your own parameters following this pattern.
        """
        super().__init__(warmup_period=warmup_period, **kwargs)
        
        # Store strategy parameters
        self.lookback_period = lookback_period
        self.threshold = threshold
        self.min_volume = min_volume
        self.max_positions = max_positions
        self.risk_free_rate = risk_free_rate
        
        # Initialize strategy state
        self.positions = {}  # Track current positions
        self.signals_history = []  # Track signal history
        
        # Set up logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Validate parameters
        self._validate_parameters()
    
    def _validate_parameters(self):
        """
        Validate strategy parameters.
        Add your own validation logic here.
        """
        if self.lookback_period <= 0:
            raise ValueError("lookback_period must be positive")
        
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        
        if self.min_volume <= 0:
            raise ValueError("min_volume must be positive")
        
        if self.max_positions <= 0:
            raise ValueError("max_positions must be positive")
    
    def calculate_signals(self, 
                         market_data: pd.DataFrame, 
                         current_time: pd.Timestamp) -> Dict[str, int]:
        """
        Generate trading signals for all tickers.
        
        IMPLEMENTATION GUIDE:
        ====================
        1. Use self.get_historical_data() to get past data
        2. Use built-in indicators from StrategyBase (SMA, EMA, RSI, etc.)
        3. Implement your signal logic
        4. Return dictionary: {'TICKER': signal} where signal is -1, 0, or 1
        
        Args:
            market_data: Current market data for all tickers
            current_time: Current timestamp
            
        Returns:
            Dictionary mapping ticker symbols to signals (-1=sell, 0=hold, 1=buy)
        """
        signals = {}
        
        # Get list of available tickers
        tickers = market_data.columns.get_level_values(0).unique()
        
        for ticker in tickers:
            try:
                # Get historical data for this ticker
                hist_data = self.get_historical_data(
                    ticker, 
                    current_time, 
                    periods=self.lookback_period + 10  # Extra buffer
                )
                
                if len(hist_data) < self.lookback_period:
                    signals[ticker] = 0  # Not enough data
                    continue
                
                # IMPLEMENT YOUR SIGNAL LOGIC HERE
                # ================================
                
                # Example 1: Moving Average Crossover
                signal = self._example_ma_crossover_signal(hist_data, ticker)
                
                # Example 2: Mean Reversion
                # signal = self._example_mean_reversion_signal(hist_data, ticker)
                
                # Example 3: Momentum Strategy
                # signal = self._example_momentum_signal(hist_data, ticker)
                
                signals[ticker] = signal
                
            except Exception as e:
                self.logger.warning(f"Error calculating signal for {ticker}: {e}")
                signals[ticker] = 0
        
        # Apply position limits
        signals = self._apply_position_limits(signals)
        
        return signals
    
    def _example_ma_crossover_signal(self, data: pd.DataFrame, ticker: str) -> int:
        """
        Example Signal 1: Moving Average Crossover
        
        Buy when short MA crosses above long MA
        Sell when short MA crosses below long MA
        """
        try:
            # Calculate moving averages using built-in methods
            short_ma = self.calculate_sma(data['close'], window=5)
            long_ma = self.calculate_sma(data['close'], window=self.lookback_period)
            
            # Check volume requirement
            current_volume = data['volume'].iloc[-1]
            if current_volume < self.min_volume:
                return 0
            
            # Generate signal based on crossover
            if len(short_ma) >= 2 and len(long_ma) >= 2:
                # Current values
                short_current = short_ma.iloc[-1]
                long_current = long_ma.iloc[-1]
                
                # Previous values
                short_prev = short_ma.iloc[-2]
                long_prev = long_ma.iloc[-2]
                
                # Crossover detection
                if short_prev <= long_prev and short_current > long_current:
                    return 1  # Buy signal
                elif short_prev >= long_prev and short_current < long_current:
                    return -1  # Sell signal
            
            return 0  # Hold
            
        except Exception as e:
            self.logger.warning(f"Error in MA crossover for {ticker}: {e}")
            return 0
    
    def _example_mean_reversion_signal(self, data: pd.DataFrame, ticker: str) -> int:
        """
        Example Signal 2: Mean Reversion Strategy
        
        Buy when price is significantly below moving average
        Sell when price is significantly above moving average
        """
        try:
            # Calculate indicators
            sma = self.calculate_sma(data['close'], window=self.lookback_period)
            current_price = data['close'].iloc[-1]
            current_sma = sma.iloc[-1]
            
            # Calculate deviation
            deviation = (current_price - current_sma) / current_sma
            
            # Generate signals
            if deviation < -self.threshold:
                return 1  # Buy (price below average)
            elif deviation > self.threshold:
                return -1  # Sell (price above average)
            
            return 0  # Hold
            
        except Exception as e:
            self.logger.warning(f"Error in mean reversion for {ticker}: {e}")
            return 0
    
    def _example_momentum_signal(self, data: pd.DataFrame, ticker: str) -> int:
        """
        Example Signal 3: Momentum Strategy
        
        Buy when momentum is positive and accelerating
        Sell when momentum is negative and decelerating
        """
        try:
            # Calculate price change over lookback period
            price_change = (data['close'].iloc[-1] - data['close'].iloc[-self.lookback_period]) / data['close'].iloc[-self.lookback_period]
            
            # Calculate RSI for additional confirmation
            rsi = self.calculate_rsi(data['close'], window=14)
            current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
            
            # Generate signals
            if price_change > self.threshold and current_rsi < 70:
                return 1  # Buy (positive momentum, not overbought)
            elif price_change < -self.threshold and current_rsi > 30:
                return -1  # Sell (negative momentum, not oversold)
            
            return 0  # Hold
            
        except Exception as e:
            self.logger.warning(f"Error in momentum signal for {ticker}: {e}")
            return 0
    
    def _apply_position_limits(self, signals: Dict[str, int]) -> Dict[str, int]:
        """
        Apply position limits to signals.
        
        This ensures we don't exceed max_positions by filtering signals.
        """
        # Count current positions
        current_positions = len([k for k, v in self.positions.items() if v != 0])
        
        # Count new buy signals
        new_buys = len([k for k, v in signals.items() if v == 1])
        
        # If we would exceed limits, prioritize by some criteria
        if current_positions + new_buys > self.max_positions:
            # Simple approach: take first N signals
            # You could implement more sophisticated prioritization
            buy_tickers = [k for k, v in signals.items() if v == 1]
            allowed_buys = self.max_positions - current_positions
            
            for i, ticker in enumerate(buy_tickers):
                if i >= allowed_buys:
                    signals[ticker] = 0  # Convert to hold
        
        return signals
    
    def get_strategy_parameters(self) -> Dict[str, Any]:
        """
        Return strategy parameters for configuration and reporting.
        
        This method is used by the framework for:
        1. Parameter validation
        2. Configuration file generation
        3. Results reporting
        4. Strategy comparison
        """
        return {
            'strategy_name': 'template',  # CHANGE THIS to your strategy name
            'lookback_period': self.lookback_period,
            'threshold': self.threshold,
            'min_volume': self.min_volume,
            'max_positions': self.max_positions,
            'warmup_period': self.warmup_period,
            'risk_free_rate': self.risk_free_rate,
            
            # Add any additional parameters here
            'version': '1.0',
            'description': 'Template strategy for framework extension'
        }
    
    def get_required_indicators(self) -> List[str]:
        """
        Return list of indicators this strategy requires.
        
        This helps the framework optimize data preparation and caching.
        """
        return [
            'close',    # Always required
            'volume',   # Required for volume filtering
            'open',     # Often useful
            'high',     # For range calculations
            'low',      # For range calculations
            
            # Add any specific indicators you need:
            # 'sma_20', 'ema_12', 'rsi_14', etc.
        ]
    
    def validate_market_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that market data contains required columns and indicators.
        
        This method is called by the framework to ensure data quality
        before strategy execution.
        """
        required_columns = self.get_required_indicators()
        
        # Check if we have multi-level columns (ticker, indicator)
        if isinstance(data.columns, pd.MultiIndex):
            available_indicators = data.columns.get_level_values(1).unique()
        else:
            available_indicators = data.columns
        
        missing_indicators = [col for col in required_columns if col not in available_indicators]
        
        if missing_indicators:
            self.logger.error(f"Missing required indicators: {missing_indicators}")
            return False
        
        return True
    
    def __str__(self) -> str:
        """String representation of the strategy."""
        params = self.get_strategy_parameters()
        return f"TemplateStrategy(lookback={self.lookback_period}, threshold={self.threshold})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()


# DEPLOYMENT CHECKLIST FOR NEW STRATEGIES:
# ========================================
# 
# 1. Copy this template and rename the class
# 2. Implement your signal logic in calculate_signals()
# 3. Update get_strategy_parameters() with your strategy name and parameters
# 4. Update get_required_indicators() with indicators you need
# 5. Add parameter validation in _validate_parameters()
# 6. Register your strategy in register_strategies.py
# 7. Create configuration templates for your strategy
# 8. Update command-line defaults if needed
# 9. Add unit tests for your strategy
# 10. Test with sample data before production use
#
# FRAMEWORK FEATURES AVAILABLE:
# ============================
# 
# Built-in Indicators (from StrategyBase):
# - calculate_sma(data, window)
# - calculate_ema(data, window) 
# - calculate_rsi(data, window)
# - calculate_bollinger_bands(data, window)
# - calculate_macd(data, fast, slow, signal)
# - calculate_atr(data, window)
# - calculate_stochastic(data, window)
# 
# Data Access:
# - get_historical_data(ticker, current_time, periods)
# - validate_market_data(data)
# 
# Risk Management (automatic):
# - Position sizing validation
# - Drawdown limits
# - Risk metrics calculation
# 
# Transaction Costs (automatic):
# - Brokerage fees
# - Market impact
# - Slippage modeling
# 
# Analysis & Reporting (automatic):
# - Performance metrics
# - Risk-adjusted returns
# - Drawdown analysis
# - Bias detection
# - Visualization
