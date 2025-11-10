# strategies/strategy_sma.py
"""
Simple Moving Average (SMA) Strategy

A straightforward trend-following strategy using two moving averages:
- Short MA (20 periods): Fast signal
- Long MA (50 periods): Trend direction

Buy when short MA crosses above long MA
Sell when short MA crosses below long MA
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from .strategy_base import StrategyBase


class SMAStrategy(StrategyBase):
    """
    Simple Moving Average Crossover Strategy
    
    This strategy demonstrates a classic technical analysis approach using
    moving average crossovers to generate buy/sell signals.
    """
    
    def __init__(self, short_ma: int = 20, long_ma: int = 50, **kwargs):
        """
        Initialize SMA strategy
        
        Args:
            short_ma: Short moving average period (default: 20)
            long_ma: Long moving average period (default: 50)
        """
        super().__init__(**kwargs)
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.name = f"SMA_{short_ma}_{long_ma}"
        
        # Validation
        if short_ma >= long_ma:
            raise ValueError("Short MA period must be less than Long MA period")
        
        self.logger.info(f"Initialized SMA Strategy: {self.short_ma}/{self.long_ma}")
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate that we have enough data for the strategy
        
        Args:
            df: Market data DataFrame
            
        Returns:
            bool: True if data is sufficient
        """
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Check required columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        # Check minimum data length
        min_required = max(self.long_ma + 10, 100)  # Buffer for stable signals
        if len(df) < min_required:
            self.logger.error(f"Insufficient data: {len(df)} rows, need {min_required}")
            return False
        
        # Check for data quality
        if df['close'].isna().sum() > len(df) * 0.05:  # Max 5% missing data
            self.logger.error("Too many missing price values")
            return False
        
        return True
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate moving averages and crossover signals
        
        Args:
            df: Market data DataFrame
            
        Returns:
            DataFrame with calculated indicators
        """
        df = df.copy()
        
        # Calculate moving averages
        df[f'sma_{self.short_ma}'] = df['close'].rolling(window=self.short_ma).mean()
        df[f'sma_{self.long_ma}'] = df['close'].rolling(window=self.long_ma).mean()
        
        # Calculate signal strength (difference between MAs)
        df['ma_diff'] = df[f'sma_{self.short_ma}'] - df[f'sma_{self.long_ma}']
        df['ma_diff_pct'] = (df['ma_diff'] / df[f'sma_{self.long_ma}']) * 100
        
        # Detect crossovers
        df['short_above_long'] = df[f'sma_{self.short_ma}'] > df[f'sma_{self.long_ma}']
        df['crossover_up'] = (df['short_above_long'] == True) & (df['short_above_long'].shift(1) == False)
        df['crossover_down'] = (df['short_above_long'] == False) & (df['short_above_long'].shift(1) == True)
        
        # Calculate trend strength
        df['trend_strength'] = abs(df['ma_diff_pct'])
        
        # Price position relative to MAs
        df['price_above_short'] = df['close'] > df[f'sma_{self.short_ma}']
        df['price_above_long'] = df['close'] > df[f'sma_{self.long_ma}']
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on MA crossovers
        
        Args:
            df: Market data with calculated indicators
            
        Returns:
            DataFrame with signals
        """
        if not self.validate_data(df):
            return pd.DataFrame()
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Initialize signal column
        df['signal'] = 'HOLD'
        df['signal_strength'] = 0.0
        df['entry_reason'] = ''
        df['exit_reason'] = ''
        
        # Generate signals
        for i in range(len(df)):
            current = df.iloc[i]
            
            # Skip if not enough data for moving averages
            if pd.isna(current[f'sma_{self.long_ma}']):
                continue
            
            # Buy signal: Short MA crosses above Long MA
            if current['crossover_up']:
                # Additional confirmation: price should be above short MA
                if current['price_above_short']:
                    df.loc[i, 'signal'] = 'BUY'
                    df.loc[i, 'signal_strength'] = min(current['trend_strength'], 5.0)  # Cap at 5%
                    df.loc[i, 'entry_reason'] = f"MA Crossover Up: {current['ma_diff_pct']:.2f}%"
            
            # Sell signal: Short MA crosses below Long MA  
            elif current['crossover_down']:
                df.loc[i, 'signal'] = 'SELL'
                df.loc[i, 'signal_strength'] = min(current['trend_strength'], 5.0)
                df.loc[i, 'exit_reason'] = f"MA Crossover Down: {current['ma_diff_pct']:.2f}%"
        
        # Clean up signals
        signals_df = df[df['signal'].isin(['BUY', 'SELL'])].copy()
        
        if len(signals_df) == 0:
            self.logger.warning("No signals generated")
            return pd.DataFrame(columns=['timestamp', 'signal', 'price', 'quantity'])
        
        # Create standardized output
        result = pd.DataFrame({
            'timestamp': signals_df['timestamp'],
            'signal': signals_df['signal'],
            'price': signals_df['close'],
            'quantity': 1,  # Default quantity, will be adjusted by position sizing
            'signal_strength': signals_df['signal_strength'],
            'entry_reason': signals_df['entry_reason'],
            'exit_reason': signals_df['exit_reason'],
            'short_ma': signals_df[f'sma_{self.short_ma}'],
            'long_ma': signals_df[f'sma_{self.long_ma}'],
            'ma_diff_pct': signals_df['ma_diff_pct']
        })
        
        self.logger.info(f"Generated {len(result)} signals ({len(result[result['signal']=='BUY'])} BUY, {len(result[result['signal']=='SELL'])} SELL)")
        
        return result
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Return strategy information for reporting
        
        Returns:
            Dictionary with strategy details
        """
        return {
            'name': self.name,
            'type': 'Trend Following',
            'description': 'Simple Moving Average Crossover Strategy',
            'parameters': {
                'short_ma': self.short_ma,
                'long_ma': self.long_ma
            },
            'signals': 'BUY on upward crossover, SELL on downward crossover',
            'risk_level': 'Medium',
            'timeframe': 'Any (tested on 1-minute to daily)',
            'suitable_for': ['Trending markets', 'Medium to long-term trades'],
            'limitations': ['Whipsaws in sideways markets', 'Lag in trend changes']
        }
    
    def optimize_parameters(self, df: pd.DataFrame, objective: str = 'sharpe') -> Dict[str, Any]:
        """
        Simple parameter optimization
        
        Args:
            df: Historical data for optimization
            objective: Optimization objective ('sharpe', 'return', 'drawdown')
            
        Returns:
            Dictionary with optimal parameters
        """
        best_params = {'short_ma': self.short_ma, 'long_ma': self.long_ma}
        best_score = -np.inf
        
        # Simple grid search
        short_range = range(10, 31, 5)  # 10, 15, 20, 25, 30
        long_range = range(40, 81, 10)  # 40, 50, 60, 70, 80
        
        for short in short_range:
            for long in long_range:
                if short >= long:
                    continue
                
                # Test parameters
                temp_strategy = SMAStrategy(short_ma=short, long_ma=long)
                signals = temp_strategy.generate_signals(df.copy())
                
                if len(signals) < 5:  # Need minimum trades
                    continue
                
                # Simple score calculation (placeholder)
                # In real implementation, you'd run a full backtest
                score = len(signals) * (long - short) / 100  # Favor more signals and spread
                
                if score > best_score:
                    best_score = score
                    best_params = {'short_ma': short, 'long_ma': long}
        
        return {
            'optimal_parameters': best_params,
            'optimization_score': best_score,
            'method': 'Simple Grid Search',
            'objective': objective
        }


# Strategy factory registration helper
def create_sma_strategy(**kwargs) -> SMAStrategy:
    """
    Factory function for creating SMA strategy instances
    
    Returns:
        Configured SMAStrategy instance
    """
    return SMAStrategy(**kwargs)


# Default parameter sets for different market conditions
SMA_PRESETS = {
    'fast': {'short_ma': 10, 'long_ma': 30},      # More signals, more noise
    'standard': {'short_ma': 20, 'long_ma': 50},   # Balanced approach
    'slow': {'short_ma': 30, 'long_ma': 100},      # Fewer signals, more stable
    'scalping': {'short_ma': 5, 'long_ma': 15},    # Very fast for intraday
}
