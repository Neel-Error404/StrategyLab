"""
Bollinger Bands Mean Reversion Strategy

A volatility-based mean reversion strategy that uses Bollinger Bands:
- Middle Band: Simple Moving Average (default: 20 periods)
- Upper/Lower Bands: ±2 standard deviations from middle band

Trading Logic:
- BUY when price touches or goes below lower band (oversold)
- SELL when price touches or goes above upper band (overbought)
- Exit when price returns to middle band

This strategy works well in range-bound markets but may generate losses
in strong trending conditions.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from .strategy_base import StrategyBase

class BollingerBandsStrategy(StrategyBase):
    """
    Bollinger Bands Mean Reversion Strategy
    
    A intermediate-level mean reversion strategy that demonstrates:
    - Statistical indicator calculation
    - Volatility-based signal generation
    - Parameter customization
    - Risk management integration
    """
    
    def __init__(self, name: str, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize Bollinger Bands strategy.
        
        Parameters:
            name: Strategy identifier
            parameters: Configuration parameters including:
                - bb_period: Bollinger Bands period (default: 20)
                - bb_std_dev: Standard deviation multiplier (default: 2.0)
                - min_volume: Minimum volume filter (default: 1000)
                - rsi_period: RSI confirmation period (default: 14)
        """
        super().__init__(name, parameters)
        
        # Strategy parameters
        self.bb_period = parameters.get('bb_period', 20) if parameters else 20
        self.bb_std_dev = parameters.get('bb_std_dev', 2.0) if parameters else 2.0
        self.min_volume = parameters.get('min_volume', 1000) if parameters else 1000
        self.rsi_period = parameters.get('rsi_period', 14) if parameters else 14
        
        # Validation
        if self.bb_period < 5:
            raise ValueError("Bollinger Bands period must be at least 5")
        if self.bb_std_dev <= 0:
            raise ValueError("Standard deviation multiplier must be positive")
            
        logging.info(f"Bollinger Bands Strategy initialized: "
                    f"Period={self.bb_period}, StdDev={self.bb_std_dev}")
    
    def prepare_data(self, df: pd.DataFrame, ticker: str, pull_date: str) -> pd.DataFrame:
        """
        Prepare data by calculating Bollinger Bands and technical indicators.
        
        Args:
            df: OHLCV dataframe with columns ['open', 'high', 'low', 'close', 'volume']
            ticker: Stock ticker symbol
            pull_date: Date range being processed
            
        Returns:
            Enhanced dataframe with Bollinger Bands indicators and signals
        """
        if df.empty:
            logging.warning("Empty dataframe provided to Bollinger Bands strategy")
            return df
            
        df = df.copy()
        
        # Calculate Bollinger Bands
        df['bb_middle'] = df['close'].rolling(
            window=self.bb_period, min_periods=self.bb_period
        ).mean()
        
        bb_std = df['close'].rolling(
            window=self.bb_period, min_periods=self.bb_period
        ).std()
        
        df['bb_upper'] = df['bb_middle'] + (self.bb_std_dev * bb_std)
        df['bb_lower'] = df['bb_middle'] - (self.bb_std_dev * bb_std)
        
        # Calculate band width and position
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Calculate RSI for confirmation
        df['rsi'] = self._calculate_rsi(df['close'], self.rsi_period)
        
        # Volume filter
        df['volume_ok'] = df['volume'] >= self.min_volume
        
        # Band touch detection
        df['touching_upper'] = df['close'] >= df['bb_upper']
        df['touching_lower'] = df['close'] <= df['bb_lower']
        df['near_middle'] = abs(df['close'] - df['bb_middle']) <= (df['bb_width'] * df['bb_middle'] * 0.1)
        
        logging.info(f"Bollinger Bands data preparation complete. Shape: {df.shape}")
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on Bollinger Bands logic.
        
        Args:
            df: Prepared dataframe with Bollinger Bands indicators
            
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
            df['bb_upper'].notna() & 
            df['bb_lower'].notna() & 
            df['bb_middle'].notna() &
            df['rsi'].notna() &
            df['volume_ok']
        )
        
        if not valid_mask.any():
            logging.warning("No valid data for signal generation")
            return df
        
        # BUY signals: Price at lower band + RSI oversold confirmation
        buy_conditions = (
            df['touching_lower'] &                    # Price at or below lower band
            (df['rsi'] < 30) &                       # RSI oversold
            df['volume_ok']                          # Volume filter
        )
        
        # SELL signals: Price at upper band + RSI overbought confirmation  
        sell_conditions = (
            df['touching_upper'] &                   # Price at or above upper band
            (df['rsi'] > 70) &                      # RSI overbought
            df['volume_ok']                         # Volume filter
        )
        
        # EXIT signals: Price returns to middle band
        exit_long_conditions = (
            df['near_middle'] &                     # Price near middle band
            (df['bb_position'] > 0.4) &            # Not too close to lower band
            (df['bb_position'] < 0.6)              # Not too close to upper band
        )
        
        exit_short_conditions = exit_long_conditions  # Same exit logic for both
        
        # Apply signals
        df.loc[buy_conditions, 'entry_signal_buy'] = True
        df.loc[sell_conditions, 'entry_signal_sell'] = True
        df.loc[exit_long_conditions, 'exit_signal_buy'] = True
        df.loc[exit_short_conditions, 'exit_signal_sell'] = True
        
        # Log signal summary
        buy_signals = df['entry_signal_buy'].sum()
        sell_signals = df['entry_signal_sell'].sum()
        logging.info(f"Bollinger Bands signals generated: {buy_signals} BUY, {sell_signals} SELL")
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index (RSI)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get strategy configuration and metadata.
        
        Returns:
            Dictionary with strategy information
        """
        return {
            'name': self.name,
            'type': 'Mean Reversion',
            'description': 'Bollinger Bands Mean Reversion Strategy',
            'parameters': {
                'bb_period': self.bb_period,
                'bb_std_dev': self.bb_std_dev,
                'min_volume': self.min_volume,
                'rsi_period': self.rsi_period
            },
            'indicators': [
                f'Bollinger_Bands_{self.bb_period}',
                f'RSI_{self.rsi_period}',
                'Band_Width',
                'Band_Position'
            ],
            'signals': [
                'Lower Band Touch + RSI Oversold (BUY)',
                'Upper Band Touch + RSI Overbought (SELL)',
                'Return to Middle Band (EXIT)'
            ],
            'market_type': 'Range-bound markets',
            'risk_level': 'Medium-High',
            'complexity': 'Intermediate'
        }
