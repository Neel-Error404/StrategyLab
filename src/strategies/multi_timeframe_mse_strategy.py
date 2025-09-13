#!/usr/bin/env python3
"""
MULTI-TIMEFRAME MSE STRATEGY - DEMONSTRATION TEMPLATE

This strategy demonstrates the complete multi-timeframe architecture:
- Strategy declares required timeframes: ['5m', '15m']
- Data loader provides exactly those timeframes
- Strategy receives Dict[str, pd.DataFrame] format
- Implements multi-timeframe signal generation

STRATEGY LOGIC:
- Uses 5m timeframe for trend direction (primary trend)
- Uses 1m timeframe for entry timing (secondary signals)
- Combines both timeframes for robust signal generation
- Maintains audit compliance (previous bar indicators, two-bar rule)

TIMEFRAME ROLES:
- 5m: Primary trend analysis (MACD + EMA for direction)
- 1m: Entry timing and momentum confirmation
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Union
from datetime import datetime, timedelta
from .strategy_base import StrategyBase

class MultiTimeframeMSEStrategy(StrategyBase):
    """
    Multi-timeframe MSE strategy demonstrating complete architecture.
    
    This strategy showcases:
    - Multi-timeframe data requirements ['1m', '5m']
    - Dict[str, pd.DataFrame] data handling
    - Cross-timeframe signal generation
    - Audit-compliant implementation
    """
    
    def __init__(self, name="Multi_MSE", parameters=None):
        super().__init__(name, parameters or {})
        
        # DECLARE MULTI-TIMEFRAME REQUIREMENTS (use available data)
        self.required_timeframes = ['1m', '5m']
        
        # [REQUIRED] WARMUP PERIODS PER TIMEFRAME
        self.warmup_periods = {
            '1m': 105,  # 105 bars of 1m = 105 minutes warmup  
            '5m': 35    # 35 bars of 5m = 175 minutes warmup
        }
        
        # Strategy parameters
        self.exit_threshold = 0.2  # 20% MACD histogram decay
        
        # Indicator parameters
        self.macd_params = {'fast': 12, 'slow': 26, 'signal': 9}
        self.ema_short = 9
        self.ema_long = 20
        
        # Position management
        self.position_state = {}  # Per-ticker position tracking
        
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.info(f"Multi-timeframe MSE strategy initialized with timeframes: {self.required_timeframes}")
        
    def _validate_timeframes(self, available_timeframes: List[str]) -> bool:
        """Validate that all required timeframes are available."""
        missing = set(self.required_timeframes) - set(available_timeframes)
        if missing:
            self.logger.error(f"Missing required timeframes: {missing}")
            return False
        return True
        
    def prepare_data(self, data: Dict[str, pd.DataFrame], ticker: str, pull_date: str) -> Dict[str, pd.DataFrame]:
        """
        Prepare multi-timeframe data with indicators for each timeframe.
        
        Args:
            data: Dict of timeframe DataFrames {'5m': df5, '15m': df15}
            ticker: Ticker symbol
            pull_date: Date for processing
            
        Returns:
            Dict of prepared DataFrames with indicators
        """
        self.logger.info(f"Preparing multi-timeframe data for {ticker}: {list(data.keys())}")
        
        if not isinstance(data, dict):
            raise ValueError(f"Multi-timeframe strategy expects Dict[str, pd.DataFrame], got {type(data)}")
        
        # Validate required timeframes are present
        missing_timeframes = set(self.required_timeframes) - set(data.keys())
        if missing_timeframes:
            raise ValueError(f"Missing required timeframes: {missing_timeframes}")
        
        prepared_data = {}
        
        # Process each timeframe
        for timeframe in self.required_timeframes:
            df = data[timeframe].copy()
            
            # Validate data quality
            if not self.validate_data(df):
                self.logger.error(f"Data validation failed for {ticker} at {timeframe}")
                continue
            
            # Ensure timestamp is datetime and sorted
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Compute indicators for this timeframe
            df = self._compute_indicators(df, timeframe)
            
            # Apply warmup period
            df = self._apply_timeframe_warmup(df, timeframe)
            
            prepared_data[timeframe] = df
            
            self.logger.info(f"Prepared {timeframe} data: {len(df)} bars after warmup")
        
        return prepared_data
    
    def _compute_indicators(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Compute MACD and EMA indicators for a specific timeframe."""
        
        # MACD calculation
        close_prices = df['close']
        ema_fast = close_prices.ewm(span=self.macd_params['fast'], adjust=False).mean()
        ema_slow = close_prices.ewm(span=self.macd_params['slow'], adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_params['signal'], adjust=False).mean()
        macd_hist = macd_line - signal_line
        
        # EMA calculation
        ema_short = close_prices.ewm(span=self.ema_short, adjust=False).mean()
        ema_long = close_prices.ewm(span=self.ema_long, adjust=False).mean()
        
        # Add indicators with timeframe prefix
        df['macd_line'] = macd_line.round(4)
        df['signal_line'] = signal_line.round(4)
        df['macd_hist'] = macd_hist.round(4)
        df['ema_short'] = ema_short.round(2)
        df['ema_long'] = ema_long.round(2)
        
        return df
    
    def _apply_timeframe_warmup(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Apply the enforced warmup period for this timeframe."""
        if timeframe not in self.warmup_periods:
            raise ValueError(f"MISSING WARMUP: No warmup period defined for {timeframe}")
        
        warmup_bars = self.warmup_periods[timeframe]
        
        if len(df) <= warmup_bars:
            self.logger.warning(f"Insufficient data for {timeframe} warmup: {len(df)} <= {warmup_bars}")
            return df
        
        return df.iloc[warmup_bars:].copy().reset_index(drop=True)
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate multi-timeframe signals combining 5m trend with 1m timing.
        
        Strategy Logic:
        1. 5m timeframe determines primary trend direction
        2. 1m timeframe provides entry timing signals
        3. Both timeframes must align for signal generation
        
        Args:
            data: Dict of prepared DataFrames with indicators
            
        Returns:
            DataFrame with combined signals (based on 1m timeframe for granularity)
        """
        self.logger.info("Generating multi-timeframe signals...")
        
        if not isinstance(data, dict):
            raise ValueError(f"Expected Dict[str, pd.DataFrame], got {type(data)}")
        
        # Get timeframe data
        df_1m = data['1m'].copy()
        df_5m = data['5m'].copy()
        
        # Align timeframes: map 5m signals to 1m timestamps
        df_1m = self._align_timeframes(df_1m, df_5m)
        
        # Generate trend signals from 5m data
        df_1m = self._generate_trend_signals(df_1m)
        
        # Generate timing signals from 1m data
        df_1m = self._generate_timing_signals(df_1m)
        
        # Combine signals with multi-timeframe logic
        df_1m = self._combine_multi_timeframe_signals(df_1m)
        
        # Apply two-bar execution rule (audit compliant)
        df_1m = self._apply_two_bar_rule(df_1m)
        
        signal_count = df_1m['final_buy_signal'].sum() + df_1m['final_sell_signal'].sum()
        self.logger.info(f"Generated {signal_count} total signals from multi-timeframe analysis")
        
        return df_1m
    
    def _align_timeframes(self, df_1m: pd.DataFrame, df_5m: pd.DataFrame) -> pd.DataFrame:
        """Align 5m signals with 1m timestamps."""
        
        # Add 5m trend information to 1m data
        df_1m['trend_direction'] = np.nan
        df_1m['trend_strength'] = np.nan
        
        for i, row_1m in df_1m.iterrows():
            timestamp_1m = row_1m['timestamp']
            
            # Find corresponding 5m bar (previous or current)
            df_5m_before = df_5m[df_5m['timestamp'] <= timestamp_1m]
            
            if not df_5m_before.empty:
                latest_5m = df_5m_before.iloc[-1]
                
                # Determine trend direction from 5m MACD and EMA
                if (latest_5m['macd_line'] > latest_5m['signal_line'] and 
                    latest_5m['ema_short'] > latest_5m['ema_long']):
                    trend_direction = 1  # Bullish
                elif (latest_5m['macd_line'] < latest_5m['signal_line'] and 
                      latest_5m['ema_short'] < latest_5m['ema_long']):
                    trend_direction = -1  # Bearish
                else:
                    trend_direction = 0  # Neutral
                
                # Trend strength based on MACD histogram
                trend_strength = abs(latest_5m['macd_hist'])
                
                df_1m.at[i, 'trend_direction'] = trend_direction
                df_1m.at[i, 'trend_strength'] = trend_strength
        
        return df_1m
    
    def _generate_trend_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trend signals based on 5m timeframe data."""
        
        # Trend signals (from 5m analysis)
        df['trend_bullish'] = df['trend_direction'] == 1
        df['trend_bearish'] = df['trend_direction'] == -1
        
        return df
    
    def _generate_timing_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate timing signals based on 1m timeframe data."""
        
        # Use previous bar indicators (audit compliant)
        df['timing_buy'] = ((df['macd_line'].shift(1) > df['signal_line'].shift(1)) & 
                           (df['ema_short'].shift(1) > df['ema_long'].shift(1)))
        
        df['timing_sell'] = ((df['macd_line'].shift(1) < df['signal_line'].shift(1)) & 
                            (df['ema_short'].shift(1) < df['ema_long'].shift(1)))
        
        return df
    
    def _combine_multi_timeframe_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combine 15m trend with 5m timing for final signals."""
        
        # Multi-timeframe signal logic:
        # BUY: 5m trend bullish AND 1m timing buy
        df['final_buy_signal'] = df['trend_bullish'] & df['timing_buy']
        
        # SELL: 5m trend bearish AND 1m timing sell
        df['final_sell_signal'] = df['trend_bearish'] & df['timing_sell']
        
        # Add signal strength based on trend strength
        df['signal_strength'] = df['trend_strength'] * (df['final_buy_signal'] | df['final_sell_signal'])
        
        return df
    
    def _apply_two_bar_rule(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply two-bar execution rule: signal on bar N, execute on bar N+1."""
        
        # Shift signals for next-bar execution (audit compliant)
        df['execute_buy'] = df['final_buy_signal'].shift(1).fillna(False)
        df['execute_sell'] = df['final_sell_signal'].shift(1).fillna(False)
        
        # Execution prices (open of execution bar)
        df['entry_price_buy'] = df['open'].where(df['execute_buy'], np.nan)
        df['entry_price_sell'] = df['open'].where(df['execute_sell'], np.nan)
        
        return df


# Test strategy instantiation
if __name__ == "__main__":
    strategy = MultiTimeframeMSEStrategy()
    print(f"Strategy: {strategy.name}")
    print(f"Required timeframes: {strategy.required_timeframes}")
    print("Multi-timeframe strategy created successfully!")