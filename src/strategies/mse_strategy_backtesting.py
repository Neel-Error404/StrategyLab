"""
MSE Strategy - Backtesting Implementation
========================================

Multi-timeframe MSE (Moving averages + Signal + Entry) Strategy for backtesting
with strict look-ahead bias prevention and proper timing controls.

Key Features:
- 525-minute warmup period for MACD stability
- No look-ahead bias (uses .shift(1) for all decisions)  
- Two-bar execution rule (signal detection → pending → execute)
- 4-indicator entry system (5m + 15m MACD + EMA alignments)
- 80% peak/valley exit logic
- Single position enforcement
- Indian market compliance (15:15 cutoff)

Critical Implementation Notes:
- ALL indicator decisions use PREVIOUS bar data via .shift(1)
- Entry/Exit occurs at OPEN price of bar AFTER signal detection
- First trades occur only AFTER 525-minute warmup period
- Peak tracking initialized ONLY after entry completion

Author: Backtesting System
Date: September 2025
Version: 1.0 - Bias-Free Implementation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, time
import logging

from src.strategies.strategy_base import StrategyBase


class MSEStrategyBacktesting(StrategyBase):
    """
    MSE Strategy - Backtesting Implementation
    
    Multi-timeframe strategy using 5-minute and 15-minute data with:
    - MACD crossover signals on both timeframes
    - EMA trend confirmation on both timeframes
    - Threshold-based exits using MACD histogram peaks
    - Strict look-ahead bias prevention
    """
    
    def __init__(self, name: str = "MSE_Strategy_Backtesting", parameters: Dict[str, Any] = None, config = None):
        super().__init__(name, parameters, config)
        
        # Strategy identification
        self.version = "1.0"
        self.description = "Multi-timeframe MSE with bias prevention"
        
        # MANDATORY: Timeframe requirements (enforced by template)
        self.required_timeframes = ['5m', '15m']
        
        # MANDATORY: Warmup periods (enforced by template)
        # 15-minute MACD requires 35 periods = 35 * 15 = 525 minutes
        self.warmup_periods = {
            '5m': 175,   # 35 periods * 5 minutes = 175 minutes
            '15m': 525   # 35 periods * 15 minutes = 525 minutes (CRITICAL)
        }
        
        # Trade state variables
        self.in_buy_trade = False
        self.in_sell_trade = False
        self.buy_entry_pending = False
        self.sell_entry_pending = False
        self.buy_exit_pending = False
        self.sell_exit_pending = False
        
        # Peak/Valley tracking for exits
        self.buy_max_hist = 0.0
        self.sell_min_hist = 0.0
        self.buy_peak_initialized = False
        self.sell_peak_initialized = False
        
        # Strategy parameters
        default_exit_threshold = 0.80  # Exit when 20% of the move remains
        if parameters and 'exit_threshold' in parameters:
            try:
                default_exit_threshold = float(parameters['exit_threshold'])
            except (TypeError, ValueError):
                self.logger.warning(f"Invalid exit_threshold '{parameters['exit_threshold']}' supplied; falling back to {default_exit_threshold}")
        self.exit_threshold = default_exit_threshold
        self.logger.info(f"MSEStrategyBacktesting exit_threshold set to {self.exit_threshold:.2f}")
        
        # Indian market hours
        self.market_close = time(15, 15)  # 3:15 PM IST
        self.eod_decision_time = time(15, 15)  # EOD decision time
        
        # Exit reason override for EOD
        self.exit_reason_override = None
        
        # Trade tracking
        self.current_position = None
        self.last_trade_direction = None
        self.last_trade_date = None
        
        # Logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
    def _validate_timeframes(self, available_timeframes: List[str]) -> bool:
        """
        MANDATORY: Validate required timeframes are available
        Called by strategy factory during registration
        """
        missing = set(self.required_timeframes) - set(available_timeframes)
        if missing:
            self.logger.error(f"Missing required timeframes: {missing}")
            return False
        return True
        
    def get_warmup_period(self, timeframe: str) -> int:
        """
        MANDATORY: Return warmup period for given timeframe
        Called by backtesting engine to skip insufficient data
        """
        return self.warmup_periods.get(timeframe, 525)  # Default to max warmup
        
    def prepare_data(self, data: Union[Dict[str, pd.DataFrame], pd.DataFrame], ticker: str, pull_date: str) -> Dict[str, pd.DataFrame]:
        """
        Prepare data for MSE strategy (required by base class)
        
        Args:
            data: Raw OHLCV data (dict or dataframe)
            ticker: Ticker symbol
            pull_date: Date string for data pull
            
        Returns:
            Dictionary with timeframes and computed indicators
        """
        
        # Convert single DataFrame to dict format if needed
        if isinstance(data, pd.DataFrame):
            # Assume it's 1-minute data if single DataFrame provided
            data_dict = {'1m': data}
        else:
            data_dict = data
            
        # Compute indicators for all timeframes
        prepared_data = self.compute_indicators(data_dict)
        
        self.logger.info(f"Data prepared for {ticker} on {pull_date}")
        self.logger.info(f"Available timeframes: {list(prepared_data.keys())}")
        
        return prepared_data
        
    def compute_indicators(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Compute technical indicators for all required timeframes
        
        Args:
            data: Dictionary with timeframe keys ('5m', '15m') and OHLCV DataFrames
            
        Returns:
            Dictionary with same structure but indicators added
        """
        result = {}
        
        for timeframe, df in data.items():
            if timeframe not in self.required_timeframes:
                result[timeframe] = df
                continue
                
            df_with_indicators = df.copy()
            
            # Compute MACD (12, 26, 9)
            df_with_indicators = self._compute_macd(
                df_with_indicators, 
                fast=12, slow=26, signal=9,
                prefix=f'{timeframe}_'
            )
            
            # Compute EMAs (9, 20)
            df_with_indicators = self._compute_ema(
                df_with_indicators, 
                periods=[9, 20],
                prefix=f'{timeframe}_'
            )
            
            result[timeframe] = df_with_indicators
            
        return result
        
    def _compute_macd(self, df: pd.DataFrame, fast: int, slow: int, 
                     signal: int, prefix: str = '') -> pd.DataFrame:
        """Compute MACD indicators"""
        
        # MACD line = EMA(12) - EMA(26)  
        ema_fast = df['close'].ewm(span=fast).mean()
        ema_slow = df['close'].ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        
        # Signal line = EMA(9) of MACD line
        signal_line = macd_line.ewm(span=signal).mean()
        
        # MACD histogram = MACD line - Signal line
        macd_hist = macd_line - signal_line
        
        df[f'{prefix}macd_line'] = macd_line
        df[f'{prefix}signal_line'] = signal_line  
        df[f'{prefix}macd_hist'] = macd_hist
        
        return df
        
    def _compute_ema(self, df: pd.DataFrame, periods: List[int], 
                    prefix: str = '') -> pd.DataFrame:
        """Compute EMA indicators"""
        
        for period in periods:
            df[f'{prefix}ema_{period}'] = df['close'].ewm(span=period).mean()
            
        return df
        
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate buy/sell signals using 4-indicator system with look-ahead bias prevention
        
        MSE 4-Indicator System (ALL must align):
        1. 5m MACD line > 5m Signal line (bullish momentum)
        2. 5m EMA(9) > 5m EMA(20) (short-term uptrend)  
        3. 15m MACD line > 15m Signal line (longer-term momentum)
        4. 15m EMA(9) > 15m EMA(20) (longer-term uptrend)
        
        CRITICAL: Uses .shift(1) to prevent look-ahead bias
        """
        
        # Use 5m as base timeframe since we don't have 1m data
        base_df = data['5m'].copy()
        df_15m = data['15m'].copy()
        
        # Merge 15m data into 5m base timeframe with forward-fill
        merged_df = self._merge_timeframe_data_5m_base(base_df, df_15m)
        
        # Generate raw signals using PREVIOUS bar data (look-ahead bias prevention)
        raw_buy_signal = (
            (merged_df['5m_macd_line'].shift(1) > merged_df['5m_signal_line'].shift(1)) &      # 5min MACD bullish
            (merged_df['5m_ema_9'].shift(1) > merged_df['5m_ema_20'].shift(1)) &              # 5min EMA bullish  
            (merged_df['15m_macd_line'].shift(1) > merged_df['15m_signal_line'].shift(1)) &    # 15min MACD bullish
            (merged_df['15m_ema_9'].shift(1) > merged_df['15m_ema_20'].shift(1))              # 15min EMA bullish
        )
        
        raw_sell_signal = (
            (merged_df['5m_macd_line'].shift(1) < merged_df['5m_signal_line'].shift(1)) &      # 5min MACD bearish
            (merged_df['5m_ema_9'].shift(1) < merged_df['5m_ema_20'].shift(1)) &              # 5min EMA bearish
            (merged_df['15m_macd_line'].shift(1) < merged_df['15m_signal_line'].shift(1)) &    # 15min MACD bearish
            (merged_df['15m_ema_9'].shift(1) < merged_df['15m_ema_20'].shift(1))              # 15min EMA bearish
        )
        
        # Apply additional filters
        merged_df['entry_signal_buy'] = self._filter_signals(merged_df, raw_buy_signal, 'buy')
        merged_df['entry_signal_sell'] = self._filter_signals(merged_df, raw_sell_signal, 'sell')
        
        # Generate exit signals using 80% peak/valley logic (bias-free)
        merged_df = self._generate_exit_signals(merged_df)
        
        return merged_df
        
    def _merge_timeframe_data_5m_base(self, df_5m: pd.DataFrame, df_15m: pd.DataFrame) -> pd.DataFrame:
        """Merge 15m data into 5m base timeframe with forward-fill"""
        
        # Start with 5m data (already has 5m indicators)
        merged = df_5m.copy()
        
        # Forward-fill 15m data to 5m resolution
        merged = merged.merge(
            df_15m[['timestamp', '15m_macd_line', '15m_signal_line', '15m_macd_hist',
                    '15m_ema_9', '15m_ema_20']],
            on='timestamp', how='left'
        )
        
        # Forward-fill 15m indicators
        merged[['15m_macd_line', '15m_signal_line', '15m_macd_hist',
                '15m_ema_9', '15m_ema_20']] = merged[['15m_macd_line', '15m_signal_line',
                '15m_macd_hist', '15m_ema_9', '15m_ema_20']].ffill()
        
        return merged
        
    def _merge_timeframe_data(self, base_df: pd.DataFrame, df_5m: pd.DataFrame, 
                            df_15m: pd.DataFrame) -> pd.DataFrame:
        """Legacy merge method for 1m base timeframe (kept for compatibility)"""
        
        merged = base_df.copy()
        
        # Forward-fill 5m data
        merged = merged.merge(
            df_5m[['timestamp', '5m_macd_line', '5m_signal_line', '5m_macd_hist', 
                   '5m_ema_9', '5m_ema_20']],
            on='timestamp', how='left'
        )
        merged[['5m_macd_line', '5m_signal_line', '5m_macd_hist', 
                '5m_ema_9', '5m_ema_20']] = merged[['5m_macd_line', '5m_signal_line', 
                '5m_macd_hist', '5m_ema_9', '5m_ema_20']].ffill()
        
        # Forward-fill 15m data  
        merged = merged.merge(
            df_15m[['timestamp', '15m_macd_line', '15m_signal_line', '15m_macd_hist',
                    '15m_ema_9', '15m_ema_20']],
            on='timestamp', how='left'
        )
        merged[['15m_macd_line', '15m_signal_line', '15m_macd_hist',
                '15m_ema_9', '15m_ema_20']] = merged[['15m_macd_line', '15m_signal_line',
                '15m_macd_hist', '15m_ema_9', '15m_ema_20']].ffill()
        
        return merged
        
    def _filter_signals(self, df: pd.DataFrame, raw_signal: pd.Series, 
                       direction: str) -> pd.Series:
        """Apply additional signal filters"""
        
        filtered_signal = raw_signal.copy()
        
        # Market hours filter (before 15:15 IST)
        market_hours_mask = df['timestamp'].dt.time < self.market_close
        filtered_signal = filtered_signal & market_hours_mask
        
        # Cascade prevention: prevent multiple same-direction trades per day
        # This will be implemented in execute_strategy method
        
        return filtered_signal
    
    def _generate_exit_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate exit signals using 80% peak/valley logic with bias prevention
        
        Key improvements over original:
        - Uses 80% threshold (vs 20%) - let winners run longer
        - Bias-free: uses .shift(1) for previous bar data
        - EOD exits at 15:15 IST
        - No cascade prevention complexity
        """
        
        # Initialize exit signal columns
        df['exit_signal_buy'] = False
        df['exit_signal_sell'] = False
        
        # Track position state and peak/valley for exit logic
        in_buy_trade = False
        in_sell_trade = False
        buy_max_hist = 0.0
        sell_min_hist = 0.0
        
        for i in range(len(df)):
            row = df.iloc[i]
            current_time = row['timestamp'].time() if hasattr(row['timestamp'], 'time') else row['timestamp']
            
            # Use previous bar MACD histogram (bias prevention)
            prev_macd_hist = df.iloc[i-1]['15m_macd_hist'] if i > 0 else row['15m_macd_hist']
            
            # EOD Exit Logic: Exit any position at/after 15:15 IST
            if current_time >= self.market_close:
                if in_buy_trade:
                    df.iloc[i, df.columns.get_loc('exit_signal_buy')] = True
                    in_buy_trade = False
                if in_sell_trade:
                    df.iloc[i, df.columns.get_loc('exit_signal_sell')] = True
                    in_sell_trade = False
                continue
                
            # Buy Trade Logic
            if not in_buy_trade and not in_sell_trade and row['entry_signal_buy']:
                in_buy_trade = True
                buy_max_hist = prev_macd_hist
                
            elif in_buy_trade:
                # Update peak tracking
                buy_max_hist = max(buy_max_hist, prev_macd_hist)
                
                # 80% Peak Exit: Exit when MACD hist drops to 80% of peak
                if buy_max_hist > 0 and prev_macd_hist < self.exit_threshold * buy_max_hist:
                    df.iloc[i, df.columns.get_loc('exit_signal_buy')] = True
                    in_buy_trade = False
                    
            # Sell Trade Logic  
            elif not in_buy_trade and not in_sell_trade and row['entry_signal_sell']:
                in_sell_trade = True
                sell_min_hist = prev_macd_hist
                
            elif in_sell_trade:
                # Update valley tracking
                sell_min_hist = min(sell_min_hist, prev_macd_hist)
                
                # 80% Valley Exit: Exit when MACD hist rises to 80% of valley depth
                if sell_min_hist < 0 and prev_macd_hist > self.exit_threshold * sell_min_hist:
                    df.iloc[i, df.columns.get_loc('exit_signal_sell')] = True
                    in_sell_trade = False
                    
        return df
        
    def execute_strategy(self, df: pd.DataFrame, initial_capital: float = 100000) -> Tuple[pd.DataFrame, Dict]:
        """
        Execute strategy with two-bar rule and bias prevention
        
        Two-Bar Execution Rule:
        1. Bar N-1: Previous bar provides indicator data
        2. Bar N: Current bar detects signal, sets pending flag  
        3. Bar N+1: Next bar executes entry/exit at OPEN price
        
        Args:
            df: DataFrame with OHLCV data and signals
            initial_capital: Starting capital
            
        Returns:
            (trades_df, summary_stats)
        """
        
        trades = []
        capital = initial_capital
        
        # Reset trade state
        self._reset_trade_state()
        
        # Skip warmup period (CRITICAL for MACD stability)
        warmup_min = max(self.warmup_periods.values())  # 525 minutes
        cutoff_ts = df['timestamp'].min() + pd.Timedelta(minutes=warmup_min)
        
        self.logger.info(f"Starting strategy execution with {warmup_min} minute warmup period")
        self.logger.info(f"First potential trade after: {cutoff_ts}")
        
        for i, (_, row) in enumerate(df.iterrows()):
            
            # Skip warmup period
            if row['timestamp'] < cutoff_ts:
                continue
                
            current_time = row['timestamp']
            current_date = current_time.date()
            current_time_only = current_time.time()
            
            # Session detection
            is_new_session_bar = (i == 0) or (df.iloc[i-1]['timestamp'].date() != row['timestamp'].date())
            is_last_bar_today = (i == len(df)-1) or (df.iloc[i+1]['timestamp'].date() != row['timestamp'].date())
            
            # 1. EXECUTE PENDING ACTIONS (using OPEN price of current bar)
            
            # Execute pending buy entry
            if self.buy_entry_pending and not self.in_buy_trade and not self.in_sell_trade:
                entry_price = row['open']
                self.in_buy_trade = True
                self.buy_entry_pending = False
                
                # Initialize peak tracking AFTER entry
                self.buy_max_hist = 0.0
                self.buy_peak_initialized = False
                
                # Update cascade tracking AFTER successful entry execution
                self.last_trade_direction = 'buy'
                self.last_trade_date = current_date
                
                trade = {
                    'entry_timestamp': current_time,
                    'entry_price': entry_price,
                    'direction': 'buy',
                    'exit_timestamp': None,
                    'exit_price': None,
                    'exit_reason': None,
                    'pnl': None,
                    'pnl_percent': None
                }
                trades.append(trade)
                
                self.logger.info(f"BUY ENTRY at {current_time}: ${entry_price:.2f} (using OPEN price)")
                
            # Execute pending sell entry  
            elif self.sell_entry_pending and not self.in_buy_trade and not self.in_sell_trade:
                entry_price = row['open']
                self.in_sell_trade = True
                self.sell_entry_pending = False
                
                # Initialize valley tracking AFTER entry
                self.sell_min_hist = 0.0
                self.sell_peak_initialized = False
                
                # Update cascade tracking AFTER successful entry execution
                self.last_trade_direction = 'sell'
                self.last_trade_date = current_date
                
                trade = {
                    'entry_timestamp': current_time,
                    'entry_price': entry_price,
                    'direction': 'sell', 
                    'exit_timestamp': None,
                    'exit_price': None,
                    'exit_reason': None,
                    'pnl': None,
                    'pnl_percent': None
                }
                trades.append(trade)
                
                self.logger.info(f"SELL ENTRY at {current_time}: ${entry_price:.2f} (using OPEN price)")
                
            # Execute pending buy exit
            if self.buy_exit_pending and self.in_buy_trade:
                exit_price = row['open']
                self.in_buy_trade = False
                self.buy_exit_pending = False
                self.buy_peak_initialized = False
                
                # Update last completed trade
                if trades:
                    last_trade = trades[-1]
                    last_trade['exit_timestamp'] = current_time
                    last_trade['exit_price'] = exit_price
                    last_trade['pnl'] = exit_price - last_trade['entry_price']
                    last_trade['pnl_percent'] = (last_trade['pnl'] / last_trade['entry_price']) * 100
                    
                    # Use override exit reason if set
                    if self.exit_reason_override:
                        last_trade['exit_reason'] = self.exit_reason_override
                        self.exit_reason_override = None  # Clear after use
                    
                    self.logger.info(f"BUY EXIT at {current_time}: ${exit_price:.2f}, PnL: {last_trade['pnl']:.2f} ({last_trade['pnl_percent']:.2f}%) - Reason: {last_trade.get('exit_reason', 'N/A')}")
                    
            # Execute pending sell exit
            if self.sell_exit_pending and self.in_sell_trade:
                exit_price = row['open'] 
                self.in_sell_trade = False
                self.sell_exit_pending = False
                self.sell_peak_initialized = False
                
                # Update last completed trade
                if trades:
                    last_trade = trades[-1]
                    last_trade['exit_timestamp'] = current_time
                    last_trade['exit_price'] = exit_price
                    last_trade['pnl'] = last_trade['entry_price'] - exit_price  # Profit when price goes down
                    last_trade['pnl_percent'] = (last_trade['pnl'] / last_trade['entry_price']) * 100
                    
                    # Use override exit reason if set
                    if self.exit_reason_override:
                        last_trade['exit_reason'] = self.exit_reason_override
                        self.exit_reason_override = None  # Clear after use
                    
                    self.logger.info(f"SELL EXIT at {current_time}: ${exit_price:.2f}, PnL: {last_trade['pnl']:.2f} ({last_trade['pnl_percent']:.2f}%) - Reason: {last_trade.get('exit_reason', 'N/A')}")
                    
            # 2. EOD (END OF DAY) HANDLING - Decision at 15:15, Execute at next bar open
            
            # EOD decision gate (set pending exit at/after 15:15; execute on next bar open)
            if current_time_only >= self.eod_decision_time:
                # Block any new entries
                if self.buy_entry_pending:
                    self.buy_entry_pending = False
                    self.logger.info(f"EOD: Cancelled pending BUY entry at {current_time}")
                if self.sell_entry_pending:
                    self.sell_entry_pending = False
                    self.logger.info(f"EOD: Cancelled pending SELL entry at {current_time}")
                
                # Schedule exits (will execute on *next* row's open, per normal pending-exit logic)
                if self.in_buy_trade and not self.buy_exit_pending:
                    self.buy_exit_pending = True
                    self.exit_reason_override = 'EOD'
                    self.logger.info(f"EOD: Scheduled BUY exit at {current_time} (will execute next bar)")
                if self.in_sell_trade and not self.sell_exit_pending:
                    self.sell_exit_pending = True
                    self.exit_reason_override = 'EOD'
                    self.logger.info(f"EOD: Scheduled SELL exit at {current_time} (will execute next bar)")
            
            # Final safety on the last bar of the session (flat by close even if next-open didn't occur)
            if is_last_bar_today:
                # Cancel any leftover pending entries
                if self.buy_entry_pending:
                    self.buy_entry_pending = False
                    self.logger.info(f"EOD FINAL: Cancelled pending BUY entry at {current_time}")
                if self.sell_entry_pending:
                    self.sell_entry_pending = False
                    self.logger.info(f"EOD FINAL: Cancelled pending SELL entry at {current_time}")
                
                # Force-close open positions at CURRENT bar close (fallback)
                if self.in_buy_trade:
                    exit_price = row['close']
                    self.in_buy_trade = False
                    self.buy_exit_pending = False
                    self.buy_peak_initialized = False
                    
                    if trades:
                        last_trade = trades[-1]
                        last_trade.update({
                            'exit_timestamp': current_time,
                            'exit_price': exit_price,
                            'pnl': exit_price - last_trade['entry_price'],
                            'pnl_percent': (exit_price / last_trade['entry_price'] - 1) * 100,
                            'exit_reason': 'EOD close (fallback)'
                        })
                        self.logger.info(f"EOD FALLBACK: BUY EXIT at {current_time}: ${exit_price:.2f} (using CLOSE)")
                
                if self.in_sell_trade:
                    exit_price = row['close']
                    self.in_sell_trade = False
                    self.sell_exit_pending = False
                    self.sell_peak_initialized = False
                    
                    if trades:
                        last_trade = trades[-1]
                        last_trade.update({
                            'exit_timestamp': current_time,
                            'exit_price': exit_price,
                            'pnl': last_trade['entry_price'] - exit_price,
                            'pnl_percent': (last_trade['entry_price'] / exit_price - 1) * 100,
                            'exit_reason': 'EOD close (fallback)'
                        })
                        self.logger.info(f"EOD FALLBACK: SELL EXIT at {current_time}: ${exit_price:.2f} (using CLOSE)")
            
            # 3. DETECT NEW SIGNALS (for next bar execution)
            
            # New buy signal detection
            if (not self.in_buy_trade and not self.in_sell_trade and 
                not self.buy_entry_pending and not self.sell_entry_pending and
                row.get('entry_signal_buy', False)):
                
                # Cascade prevention: check if same direction as last trade today
                if not self._is_cascade_trade(current_date, 'buy'):
                    self.buy_entry_pending = True
                    self.logger.info(f"BUY SIGNAL detected at {current_time} - entry pending for next bar")
                
            # New sell signal detection  
            elif (not self.in_buy_trade and not self.in_sell_trade and
                  not self.buy_entry_pending and not self.sell_entry_pending and
                  row.get('entry_signal_sell', False)):
                
                # Cascade prevention: check if same direction as last trade today
                if not self._is_cascade_trade(current_date, 'sell'):
                    self.sell_entry_pending = True
                    self.logger.info(f"SELL SIGNAL detected at {current_time} - entry pending for next bar")
                    
            # 4. MONITOR EXIT CONDITIONS (using PREVIOUS bar data)
            
            if i > 0:  # Ensure we have previous bar data
                prev_row = df.iloc[i-1]  # Fixed: Use positional iloc for robustness
                
                # Buy position exit monitoring
                if self.in_buy_trade:
                    prev_macd_hist = prev_row.get('15m_macd_hist', 0)
                    
                    # Initialize or update peak tracking
                    if not self.buy_peak_initialized:
                        self.buy_max_hist = prev_macd_hist
                        self.buy_peak_initialized = True
                    else:
                        self.buy_max_hist = max(self.buy_max_hist, prev_macd_hist)
                        
                    # Check exit condition: MACD histogram drops to 80% of peak
                    if (self.buy_max_hist > 0 and 
                        prev_macd_hist < self.exit_threshold * self.buy_max_hist):
                        self.buy_exit_pending = True
                        last_trade = trades[-1] if trades else {}
                        last_trade['exit_reason'] = f"80% peak exit (peak: {self.buy_max_hist:.4f}, current: {prev_macd_hist:.4f})"
                        self.logger.info(f"BUY EXIT condition detected - exit pending for next bar")
                        
                # Sell position exit monitoring  
                elif self.in_sell_trade:
                    prev_macd_hist = prev_row.get('15m_macd_hist', 0)
                    
                    # Initialize or update valley tracking  
                    if not self.sell_peak_initialized:
                        self.sell_min_hist = prev_macd_hist
                        self.sell_peak_initialized = True
                    else:
                        self.sell_min_hist = min(self.sell_min_hist, prev_macd_hist)
                        
                    # Check exit condition: MACD histogram rises to 80% of valley depth
                    if (self.sell_min_hist < 0 and
                        prev_macd_hist > self.exit_threshold * self.sell_min_hist):
                        self.sell_exit_pending = True
                        last_trade = trades[-1] if trades else {}
                        last_trade['exit_reason'] = f"80% valley exit (valley: {self.sell_min_hist:.4f}, current: {prev_macd_hist:.4f})"
                        self.logger.info(f"SELL EXIT condition detected - exit pending for next bar")
                        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(trades_df, initial_capital)
        
        return trades_df, summary_stats
        
    def _reset_trade_state(self):
        """Reset all trade state variables"""
        self.in_buy_trade = False
        self.in_sell_trade = False
        self.buy_entry_pending = False
        self.sell_entry_pending = False
        self.buy_exit_pending = False
        self.sell_exit_pending = False
        self.buy_max_hist = 0.0
        self.sell_min_hist = 0.0
        self.buy_peak_initialized = False
        self.sell_peak_initialized = False
        self.last_trade_direction = None
        self.last_trade_date = None
        
    def _is_cascade_trade(self, current_date, direction: str) -> bool:
        """Check if this would be a cascade trade (same direction on same day)"""
        if (self.last_trade_date == current_date and 
            self.last_trade_direction == direction):
            self.logger.info(f"Cascade trade prevented: {direction} on {current_date}")
            return True
            
        # Do NOT update last trade info here - it's updated after successful entry execution
        return False
        
    def _calculate_summary_stats(self, trades_df: pd.DataFrame, 
                                initial_capital: float) -> Dict[str, Any]:
        """Calculate comprehensive summary statistics"""
        
        if trades_df.empty:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_return': 0.0,
                'total_return_percent': 0.0,
                'avg_trade_return': 0.0,
                'max_profit': 0.0,
                'max_loss': 0.0,
                'profit_factor': 0.0
            }
            
        # Filter completed trades only
        completed_trades = trades_df[trades_df['exit_price'].notna()].copy()
        
        if completed_trades.empty:
            return {'total_trades': len(trades_df), 'completed_trades': 0}
            
        total_trades = len(completed_trades)
        winning_trades = len(completed_trades[completed_trades['pnl'] > 0])
        losing_trades = len(completed_trades[completed_trades['pnl'] < 0])
        
        total_return = completed_trades['pnl'].sum()
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        gross_profit = completed_trades[completed_trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(completed_trades[completed_trades['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'total_return_percent': (total_return / initial_capital) * 100,
            'avg_trade_return': total_return / total_trades if total_trades > 0 else 0,
            'max_profit': completed_trades['pnl'].max() if total_trades > 0 else 0,
            'max_loss': completed_trades['pnl'].min() if total_trades > 0 else 0,
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }
        
    def get_strategy_info(self) -> Dict[str, Any]:
        """Return comprehensive strategy information"""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'required_timeframes': self.required_timeframes,
            'warmup_periods': self.warmup_periods,
            'parameters': {
                'exit_threshold': self.exit_threshold,
                'market_close': str(self.market_close),
            },
            'features': [
                '4-indicator entry system',
                '525-minute warmup for MACD stability',
                'Look-ahead bias prevention (.shift(1))',
                'Two-bar execution rule',
                '80% peak/valley exit logic',
                'Single position enforcement',
                'Cascade trade prevention',
                'Indian market compliance'
            ]
        }
