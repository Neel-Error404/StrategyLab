#!/usr/bin/env python3
"""
ENFORCED MULTI-TIMEFRAME STRATEGY TEMPLATE
==========================================

This template ENFORCES the non-negotiable requirements for multi-timeframe strategies:

NON-NEGOTIABLE REQUIREMENTS:
1. MUST declare required_timeframes in __init__
2. MUST specify warmup periods for each timeframe
3. MUST implement timeframe-specific data preparation
4. MUST implement cross-timeframe signal generation
5. MUST handle Dict[str, pd.DataFrame] data format

FRAMEWORK ENFORCEMENT:
- Strategy registration will FAIL if required_timeframes is not set
- Data loader will FAIL if timeframes are not declared
- Strategy execution will FAIL if warmup periods are insufficient

Copy this template and replace 'YourMultiTimeframe' with your strategy name.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Union
from datetime import datetime, timedelta
from .strategy_base import StrategyBase


class YourMultiTimeframeStrategyTemplate(StrategyBase):
    """
    ENFORCED Multi-Timeframe Strategy Template
    
    This template demonstrates the MANDATORY requirements for multi-timeframe strategies.
    All sections marked with [REQUIRED] MUST be implemented for the strategy to work.
    
    ARCHITECTURE:
    - Primary timeframe: Higher timeframe for trend/direction (e.g., 15m, 1h)
    - Secondary timeframe: Lower timeframe for entry timing (e.g., 1m, 5m)
    - Cross-timeframe logic: Combine both timeframes for robust signals
    """
    
    def __init__(self, name="YourMultiTimeframe", parameters=None):
        super().__init__(name, parameters or {})
        
        # [REQUIRED] DECLARE MULTI-TIMEFRAME REQUIREMENTS
        # ===============================================
        # This is NON-NEGOTIABLE. Strategy will FAIL without this.
        # Replace these with your actual required timeframes:
        self.required_timeframes = ['5m', '15m']  # CHANGE THIS to your timeframes
        
        # [REQUIRED] WARMUP PERIODS PER TIMEFRAME
        # =======================================
        # Each timeframe needs sufficient data for indicators to be valid.
        # These are the MINIMUM bars required before generating signals.
        self.warmup_periods = {
            '5m': 105,    # CHANGE THIS: 105 bars of 5m = 525 minutes warmup
            '15m': 35     # CHANGE THIS: 35 bars of 15m = 525 minutes warmup
        }
        
        # [REQUIRED] STRATEGY PARAMETERS
        # =============================
        # Define your strategy-specific parameters here:
        self.primary_timeframe = '15m'      # CHANGE THIS: Higher timeframe for trend
        self.secondary_timeframe = '5m'     # CHANGE THIS: Lower timeframe for timing
        
        # Example parameters (customize these):
        self.trend_lookback = 20            # Periods for trend analysis
        self.timing_lookback = 10           # Periods for entry timing
        self.signal_threshold = 0.02        # Minimum signal strength
        
        # [REQUIRED] INDICATOR PARAMETERS
        # ==============================
        # Define parameters for technical indicators:
        self.macd_params = {'fast': 12, 'slow': 26, 'signal': 9}
        self.ema_short = 9
        self.ema_long = 21
        self.rsi_period = 14
        
        # Position management
        self.position_state = {}  # Per-ticker position tracking
        
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.info(f"Multi-timeframe strategy initialized with timeframes: {self.required_timeframes}")
        
        # [REQUIRED] VALIDATION
        # ====================
        self._validate_configuration()
    
    def _validate_configuration(self):
        """
        [REQUIRED] Validate that all mandatory configuration is present.
        This method ENFORCES the non-negotiable requirements.
        """
        # Validate required_timeframes
        if not hasattr(self, 'required_timeframes') or not self.required_timeframes:
            raise ValueError("MISSING REQUIREMENT: required_timeframes must be declared in __init__")
        
        if not isinstance(self.required_timeframes, list) or len(self.required_timeframes) < 2:
            raise ValueError("INVALID REQUIREMENT: required_timeframes must be a list with at least 2 timeframes")
        
        # Validate warmup periods
        if not hasattr(self, 'warmup_periods') or not self.warmup_periods:
            raise ValueError("MISSING REQUIREMENT: warmup_periods must be declared for each timeframe")
        
        # Check warmup period for each required timeframe
        for timeframe in self.required_timeframes:
            if timeframe not in self.warmup_periods:
                raise ValueError(f"MISSING REQUIREMENT: warmup period not defined for timeframe {timeframe}")
            
            if self.warmup_periods[timeframe] <= 0:
                raise ValueError(f"INVALID REQUIREMENT: warmup period for {timeframe} must be positive")
        
        # Validate primary and secondary timeframes are in required list
        if hasattr(self, 'primary_timeframe') and self.primary_timeframe not in self.required_timeframes:
            raise ValueError(f"INVALID CONFIG: primary_timeframe {self.primary_timeframe} not in required_timeframes")
        
        if hasattr(self, 'secondary_timeframe') and self.secondary_timeframe not in self.required_timeframes:
            raise ValueError(f"INVALID CONFIG: secondary_timeframe {self.secondary_timeframe} not in required_timeframes")
        
        self.logger.info("✅ All non-negotiable requirements validated successfully")
    
    def _validate_timeframes(self, available_timeframes: List[str]) -> bool:
        """
        [REQUIRED] Validate that all required timeframes are available.
        This is called by the base class execute() method.
        """
        missing = set(self.required_timeframes) - set(available_timeframes)
        if missing:
            self.logger.error(f"Missing required timeframes: {missing}")
            return False
        return True
    
    def prepare_data(self, data: Dict[str, pd.DataFrame], ticker: str, pull_date: str) -> Dict[str, pd.DataFrame]:
        """
        [REQUIRED] Prepare multi-timeframe data with indicators for each timeframe.
        
        MANDATORY IMPLEMENTATION:
        1. Validate input is Dict[str, pd.DataFrame]
        2. Apply warmup periods for each timeframe
        3. Calculate indicators for each timeframe
        4. Return prepared data in same Dict format
        
        Args:
            data: Dict of timeframe DataFrames {'5m': df5, '15m': df15}
            ticker: Ticker symbol
            pull_date: Date for processing
            
        Returns:
            Dict of prepared DataFrames with indicators
        """
        self.logger.info(f"Preparing multi-timeframe data for {ticker}: {list(data.keys())}")
        
        # [REQUIRED] VALIDATE INPUT FORMAT
        if not isinstance(data, dict):
            raise ValueError(f"INVALID INPUT: Multi-timeframe strategy expects Dict[str, pd.DataFrame], got {type(data)}")
        
        # [REQUIRED] VALIDATE REQUIRED TIMEFRAMES ARE PRESENT
        missing_timeframes = set(self.required_timeframes) - set(data.keys())
        if missing_timeframes:
            raise ValueError(f"MISSING TIMEFRAMES: {missing_timeframes}")
        
        prepared_data = {}
        
        # [REQUIRED] PROCESS EACH TIMEFRAME
        for timeframe in self.required_timeframes:
            df = data[timeframe].copy()
            
            # Validate basic data quality
            if not self.validate_data(df):
                self.logger.error(f"Data validation failed for {ticker} at {timeframe}")
                continue
            
            # Ensure timestamp is datetime and sorted
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # [REQUIRED] CALCULATE INDICATORS FOR THIS TIMEFRAME
            df = self._compute_indicators(df, timeframe)
            
            # [REQUIRED] APPLY WARMUP PERIOD
            df = self._apply_warmup_period(df, timeframe)
            
            prepared_data[timeframe] = df
            self.logger.info(f"Prepared {timeframe} data: {len(df)} bars after warmup")
        
        return prepared_data
    
    def _compute_indicators(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        [REQUIRED] Compute technical indicators for a specific timeframe.
        
        CUSTOMIZE THIS METHOD with your indicator calculations.
        """
        # Example indicator calculations (customize these):
        
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
        
        # RSI calculation
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Add indicators to DataFrame
        df['macd_line'] = macd_line.round(4)
        df['signal_line'] = signal_line.round(4)
        df['macd_hist'] = macd_hist.round(4)
        df['ema_short'] = ema_short.round(2)
        df['ema_long'] = ema_long.round(2)
        df['rsi'] = rsi.round(2)
        
        # ADD YOUR CUSTOM INDICATORS HERE
        # ===============================
        # Example:
        # df['your_indicator'] = your_calculation(df)
        
        return df
    
    def _apply_warmup_period(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        [REQUIRED] Apply the mandatory warmup period for this timeframe.
        """
        if timeframe not in self.warmup_periods:
            raise ValueError(f"MISSING WARMUP: No warmup period defined for {timeframe}")
        
        warmup_bars = self.warmup_periods[timeframe]
        
        if len(df) <= warmup_bars:
            self.logger.warning(f"Insufficient data for {timeframe} warmup: {len(df)} <= {warmup_bars}")
            return df
        
        return df.iloc[warmup_bars:].copy().reset_index(drop=True)
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        [REQUIRED] Generate multi-timeframe signals.
        
        MANDATORY IMPLEMENTATION:
        1. Validate input is Dict[str, pd.DataFrame]
        2. Extract data from primary and secondary timeframes
        3. Align timeframes (map higher TF signals to lower TF timestamps)
        4. Generate trend signals from primary timeframe
        5. Generate timing signals from secondary timeframe  
        6. Combine signals with multi-timeframe logic
        7. Apply two-bar rule for audit compliance
        8. Return DataFrame based on secondary (granular) timeframe
        
        Args:
            data: Dict of prepared DataFrames with indicators
            
        Returns:
            DataFrame with combined signals (based on secondary timeframe)
        """
        self.logger.info("Generating multi-timeframe signals...")
        
        # [REQUIRED] VALIDATE INPUT FORMAT
        if not isinstance(data, dict):
            raise ValueError(f"INVALID INPUT: Expected Dict[str, pd.DataFrame], got {type(data)}")
        
        # [REQUIRED] GET TIMEFRAME DATA
        # Use your actual primary and secondary timeframes:
        primary_df = data[self.primary_timeframe].copy()      # Higher TF (trend)
        secondary_df = data[self.secondary_timeframe].copy()  # Lower TF (timing)
        
        # [REQUIRED] ALIGN TIMEFRAMES
        # Map primary timeframe signals to secondary timeframe timestamps
        secondary_df = self._align_timeframes(secondary_df, primary_df)
        
        # [REQUIRED] GENERATE TREND SIGNALS (from primary timeframe)
        secondary_df = self._generate_trend_signals(secondary_df)
        
        # [REQUIRED] GENERATE TIMING SIGNALS (from secondary timeframe)
        secondary_df = self._generate_timing_signals(secondary_df)
        
        # [REQUIRED] COMBINE MULTI-TIMEFRAME SIGNALS
        secondary_df = self._combine_signals(secondary_df)
        
        # [REQUIRED] APPLY TWO-BAR RULE (audit compliance)
        secondary_df = self._apply_two_bar_rule(secondary_df)
        
        signal_count = secondary_df['final_buy_signal'].sum() + secondary_df['final_sell_signal'].sum()
        self.logger.info(f"Generated {signal_count} total signals from multi-timeframe analysis")
        
        return secondary_df
    
    def _align_timeframes(self, secondary_df: pd.DataFrame, primary_df: pd.DataFrame) -> pd.DataFrame:
        """
        [REQUIRED] Align primary timeframe signals with secondary timeframe timestamps.
        
        CUSTOMIZE THIS METHOD with your alignment logic.
        """
        # Add primary timeframe information to secondary timeframe data
        secondary_df['trend_direction'] = np.nan
        secondary_df['trend_strength'] = np.nan
        
        for i, row in secondary_df.iterrows():
            timestamp = row['timestamp']
            
            # Find corresponding primary timeframe bar (previous or current)
            primary_before = primary_df[primary_df['timestamp'] <= timestamp]
            
            if not primary_before.empty:
                latest_primary = primary_before.iloc[-1]
                
                # CUSTOMIZE THIS LOGIC for your trend determination:
                # ================================================
                
                # Example: Determine trend from MACD + EMA
                if (latest_primary['macd_line'] > latest_primary['signal_line'] and 
                    latest_primary['ema_short'] > latest_primary['ema_long']):
                    trend_direction = 1  # Bullish
                elif (latest_primary['macd_line'] < latest_primary['signal_line'] and 
                      latest_primary['ema_short'] < latest_primary['ema_long']):
                    trend_direction = -1  # Bearish
                else:
                    trend_direction = 0  # Neutral
                
                # Trend strength from MACD histogram
                trend_strength = abs(latest_primary['macd_hist'])
                
                secondary_df.at[i, 'trend_direction'] = trend_direction
                secondary_df.at[i, 'trend_strength'] = trend_strength
        
        return secondary_df
    
    def _generate_trend_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [REQUIRED] Generate trend signals based on primary timeframe analysis.
        
        CUSTOMIZE THIS METHOD with your trend signal logic.
        """
        # Example trend signal generation (customize this):
        df['trend_bullish'] = df['trend_direction'] == 1
        df['trend_bearish'] = df['trend_direction'] == -1
        df['trend_neutral'] = df['trend_direction'] == 0
        
        # ADD YOUR CUSTOM TREND LOGIC HERE
        # ===============================
        
        return df
    
    def _generate_timing_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [REQUIRED] Generate timing signals based on secondary timeframe data.
        
        CUSTOMIZE THIS METHOD with your timing signal logic.
        """
        # Example timing signals using previous bar indicators (audit compliant)
        df['timing_buy'] = ((df['macd_line'].shift(1) > df['signal_line'].shift(1)) & 
                           (df['ema_short'].shift(1) > df['ema_long'].shift(1)) &
                           (df['rsi'].shift(1) < 70))  # Not overbought
        
        df['timing_sell'] = ((df['macd_line'].shift(1) < df['signal_line'].shift(1)) & 
                            (df['ema_short'].shift(1) < df['ema_long'].shift(1)) &
                            (df['rsi'].shift(1) > 30))  # Not oversold
        
        # ADD YOUR CUSTOM TIMING LOGIC HERE
        # ================================
        
        return df
    
    def _combine_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [REQUIRED] Combine primary timeframe trend with secondary timeframe timing.
        
        CUSTOMIZE THIS METHOD with your signal combination logic.
        """
        # Multi-timeframe signal logic:
        # BUY: Primary trend bullish AND Secondary timing buy
        df['final_buy_signal'] = df['trend_bullish'] & df['timing_buy']
        
        # SELL: Primary trend bearish AND Secondary timing sell
        df['final_sell_signal'] = df['trend_bearish'] & df['timing_sell']
        
        # Signal strength based on trend strength
        df['signal_strength'] = df['trend_strength'] * (df['final_buy_signal'] | df['final_sell_signal'])
        
        # ADD YOUR CUSTOM COMBINATION LOGIC HERE
        # =====================================
        
        return df
    
    def _apply_two_bar_rule(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [REQUIRED] Apply two-bar execution rule for audit compliance.
        
        This is NON-NEGOTIABLE for production strategies.
        Signal on bar N, execute on bar N+1.
        """
        # Shift signals for next-bar execution (audit compliant)
        df['execute_buy'] = df['final_buy_signal'].shift(1).fillna(False)
        df['execute_sell'] = df['final_sell_signal'].shift(1).fillna(False)
        
        # Execution prices (open of execution bar)
        df['entry_price_buy'] = df['open'].where(df['execute_buy'], np.nan)
        df['entry_price_sell'] = df['open'].where(df['execute_sell'], np.nan)
        
        return df


# [REQUIRED] STRATEGY REGISTRATION EXAMPLE
# =======================================
# Add this to your register_strategies.py:
#
# from .your_multi_timeframe_strategy import YourMultiTimeframeStrategy
# StrategyFactory.register_strategy('your_multi_tf', YourMultiTimeframeStrategy)


# [REQUIRED] TESTING YOUR STRATEGY
# ===============================
if __name__ == "__main__":
    # Test strategy instantiation
    strategy = YourMultiTimeframeStrategyTemplate()
    print(f"Strategy: {strategy.name}")
    print(f"Required timeframes: {strategy.required_timeframes}")
    print(f"Warmup periods: {strategy.warmup_periods}")
    print("✅ Multi-timeframe strategy template validation passed!")


# [REQUIRED] DEPLOYMENT CHECKLIST
# ===============================
# 1. ✅ Copy this template and rename class/file
# 2. ✅ Update required_timeframes with your actual timeframes
# 3. ✅ Update warmup_periods for each timeframe
# 4. ✅ Implement _compute_indicators() with your indicators
# 5. ✅ Implement _align_timeframes() with your alignment logic
# 6. ✅ Implement _generate_trend_signals() with trend logic
# 7. ✅ Implement _generate_timing_signals() with timing logic
# 8. ✅ Implement _combine_signals() with combination logic
# 9. ✅ Register strategy in register_strategies.py
# 10. ✅ Test with end-to-end test script
#
# FRAMEWORK GUARANTEES:
# ====================
# ✅ Strategy registration will validate timeframes
# ✅ Data loader will provide exactly required timeframes
# ✅ Framework will enforce warmup periods
# ✅ Audit compliance built into template
# ✅ Multi-timeframe data format guaranteed