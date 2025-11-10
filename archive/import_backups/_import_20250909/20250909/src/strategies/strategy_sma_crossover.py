"""
Simple Moving Average Crossover Strategy

A classic trend-following strategy that uses two moving averages:
- Fast SMA (default: 5 periods)
- Slow SMA (default: 20 periods)

Trading Logic:
- BUY when fast SMA crosses above slow SMA (golden cross)
- SELL when fast SMA crosses below slow SMA (death cross)

This strategy works well in trending markets but may generate false signals
in sideways/choppy market conditions.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from .strategy_base import StrategyBase

class SMAcrossoverStrategy(StrategyBase):
    """
    Simple Moving Average Crossover Strategy
    
    A beginner-friendly trend-following strategy that demonstrates:
    - Technical indicator calculation
    - Signal generation logic
    - Parameter customization
    - Risk management integration
    """
    
    def __init__(self, name: str, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize SMA Crossover strategy.
        
        Parameters:
            name: Strategy identifier
            parameters: Configuration parameters including:
                - fast_sma_period: Fast SMA period (default: 5)
                - slow_sma_period: Slow SMA period (default: 20)
                - min_volume: Minimum volume filter (default: 1000)
        """
        super().__init__(name, parameters)
        
        # Strategy parameters
        self.fast_sma_period = parameters.get('fast_sma_period', 5) if parameters else 5
        self.slow_sma_period = parameters.get('slow_sma_period', 20) if parameters else 20
        self.min_volume = parameters.get('min_volume', 1000) if parameters else 1000
          # Validation
        if self.fast_sma_period >= self.slow_sma_period:
            raise ValueError("Fast SMA period must be less than slow SMA period")
            
        logging.info(f"SMA Crossover Strategy initialized: "
                    f"Fast={self.fast_sma_period}, Slow={self.slow_sma_period}")
    
    def prepare_data(self, df: pd.DataFrame, ticker: str, pull_date: str) -> pd.DataFrame:
        """
        Prepare data by calculating moving averages and technical indicators.
        
        Args:
            df: OHLCV dataframe with columns ['open', 'high', 'low', 'close', 'volume']
            ticker: Stock ticker symbol
            pull_date: Date range being processed
            
        Returns:
            Enhanced dataframe with SMA indicators and signals
        """
        if df.empty:
            logging.warning("Empty dataframe provided to SMA Crossover strategy")
            return df
            
        df = df.copy()
        
        # Calculate moving averages
        df[f'sma_{self.fast_sma_period}'] = df['close'].rolling(
            window=self.fast_sma_period, min_periods=self.fast_sma_period
        ).mean()
        
        df[f'sma_{self.slow_sma_period}'] = df['close'].rolling(
            window=self.slow_sma_period, min_periods=self.slow_sma_period
        ).mean()
        
        # Calculate crossover signals
        df['sma_fast'] = df[f'sma_{self.fast_sma_period}']
        df['sma_slow'] = df[f'sma_{self.slow_sma_period}']
        
        # Previous period values for crossover detection
        df['sma_fast_prev'] = df['sma_fast'].shift(1)
        df['sma_slow_prev'] = df['sma_slow'].shift(1)
        
        # Volume filter
        df['volume_ok'] = df['volume'] >= self.min_volume
        
        # Trend direction
        df['trend_up'] = df['sma_fast'] > df['sma_slow']
        df['trend_down'] = df['sma_fast'] < df['sma_slow']
        
        logging.info(f"SMA Crossover data preparation complete. Shape: {df.shape}")
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on SMA crossover logic.
        
        Args:
            df: Prepared dataframe with SMA indicators
            
        Returns:
            Dataframe with entry/exit signals added
        """
        if df.empty:
            logging.warning("Empty dataframe provided for signal generation")
            return df
            
        df = df.copy()
        
        # Initialize signal columns
        df['entry_signal_buy'] = False
        df['entry_signal_sell'] = False
        df['exit_signal_buy'] = False
        df['exit_signal_sell'] = False
        
        # Remove rows with insufficient data
        valid_mask = (
            df['sma_fast'].notna() & 
            df['sma_slow'].notna() & 
            df['sma_fast_prev'].notna() & 
            df['sma_slow_prev'].notna() &
            df['volume_ok']
        )
        
        if not valid_mask.any():
            logging.warning("No valid data for signal generation")
            return df
        
        # Golden Cross: Fast SMA crosses above Slow SMA (BUY signal)
        golden_cross = (
            (df['sma_fast_prev'] <= df['sma_slow_prev']) &  # Was below or equal
            (df['sma_fast'] > df['sma_slow']) &             # Now above
            df['volume_ok']                                  # Volume filter
        )
        
        # Death Cross: Fast SMA crosses below Slow SMA (SELL signal)
        death_cross = (
            (df['sma_fast_prev'] >= df['sma_slow_prev']) &  # Was above or equal
            (df['sma_fast'] < df['sma_slow']) &             # Now below
            df['volume_ok']                                  # Volume filter
        )
        
        # Entry signals
        df.loc[golden_cross, 'entry_signal_buy'] = True
        df.loc[death_cross, 'entry_signal_sell'] = True
        
        # Exit signals (opposite of entry)
        df.loc[death_cross, 'exit_signal_buy'] = True
        df.loc[golden_cross, 'exit_signal_sell'] = True
        
        # Log signal summary
        buy_signals = df['entry_signal_buy'].sum()
        sell_signals = df['entry_signal_sell'].sum()
        logging.info(f"SMA Crossover signals generated: {buy_signals} BUY, {sell_signals} SELL")
        
        return df
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get strategy configuration and metadata.
        
        Returns:
            Dictionary with strategy information
        """
        return {
            'name': self.name,
            'type': 'Trend Following',
            'description': 'Simple Moving Average Crossover Strategy',
            'parameters': {
                'fast_sma_period': self.fast_sma_period,
                'slow_sma_period': self.slow_sma_period,
                'min_volume': self.min_volume
            },
            'indicators': [
                f'SMA_{self.fast_sma_period}',
                f'SMA_{self.slow_sma_period}'
            ],
            'signals': [
                'Golden Cross (BUY)',
                'Death Cross (SELL)'
            ],
            'market_type': 'Trending markets',
            'risk_level': 'Medium',
            'complexity': 'Beginner'
        }
