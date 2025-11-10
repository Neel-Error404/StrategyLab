# strategies/mse_80pct_no_cascade.py
# MSE Strategy: 80% Exit Threshold WITHOUT CASCADE PREVENTION

import logging
import pandas as pd
from typing import Optional
from .strategy_base import StrategyBase

#####################################
# Utility Functions
#####################################
def round2(x):
    try:
        return round(x, 2)
    except Exception:
        return x

def compute_macd(df: pd.DataFrame, column: str = 'close', prefix: str = '') -> pd.DataFrame:
    """
    Compute MACD indicators for the given column on df.
    Optionally use 'prefix' (e.g., '5m_' or '15m_') to label columns distinctly.
    """
    short_ema = df[column].ewm(span=12, adjust=False).mean()
    long_ema = df[column].ewm(span=26, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=9, adjust=False).mean()

    df[f'{prefix}macd_line'] = macd_line.apply(round2)
    df[f'{prefix}signal_line'] = signal_line.apply(round2)
    df[f'{prefix}macd_hist'] = (macd_line - signal_line).apply(round2)
    return df

def compute_ema(df: pd.DataFrame, span: int, column: str = 'close', prefix: str = '') -> pd.DataFrame:
    """
    Compute EMA for the given column on df, with optional column prefix.
    """
    df[f'{prefix}ema_{span}'] = df[column].ewm(span=span, adjust=False).mean().apply(round2)
    return df

def resample_ohlc(base_df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Resample 1-minute data (base_df) to the specified freq (e.g., '5min', '15min').
    Returns a DataFrame with columns: open, high, low, close, volume
    """
    # Ensure 'timestamp' is the index
    if 'timestamp' not in base_df.columns:
        logging.error("The base DataFrame does not contain a 'timestamp' column.")
        return pd.DataFrame()
    
    base_df = base_df.set_index('timestamp', drop=False)
    df_resampled = base_df.resample(freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().reset_index()
    return df_resampled

def forward_fill_to_1m(df_high_tf: pd.DataFrame, base_1m_df: pd.DataFrame, freq_label: str) -> pd.DataFrame:
    """
    Forward-fill the higher-timeframe (df_high_tf) indicator columns onto the 1-minute DataFrame.
    We assume df_high_tf['timestamp'] is the bar close time. We'll forward fill for the minutes that follow until next bar.
    freq_label is something like '5m_' or '15m_' to prefix columns if desired.
    """
    # Ensure 'timestamp' is datetime and sorted
    df_high_tf['timestamp'] = pd.to_datetime(df_high_tf['timestamp'])
    base_1m_df['timestamp'] = pd.to_datetime(base_1m_df['timestamp'])
    
    df_high_tf = df_high_tf.sort_values('timestamp')
    base_1m_df = base_1m_df.sort_values('timestamp')
    
    # Perform merge_asof with direction='backward'
    df_merged = pd.merge_asof(
        base_1m_df,
        df_high_tf,
        on='timestamp',
        direction='backward',
        suffixes=('', f'_{freq_label.rstrip("_")}')
    )
    
    # Prefix the columns from the higher timeframe
    indicator_columns = ['macd_line', 'signal_line', 'macd_hist', 'ema_9', 'ema_20']
    for col in indicator_columns:
        if col in df_merged.columns and col not in base_1m_df.columns:
            df_merged.rename(columns={col: f'{freq_label}{col}'}, inplace=True)
    
    return df_merged

#####################################
# Main Strategy Function
#####################################
class MSEStrategy(StrategyBase):
    """
    MSE Strategy: 80% Exit Threshold WITHOUT CASCADE PREVENTION
    
    CONFIGURATION:
    - Exit Threshold: 80% of peak (late exits - lets winners run)
    - Cascade Prevention: DISABLED
    - Entry: ALL 4 indicators must align (5min MACD, 15min MACD, 5min EMA, 15min EMA)
    """
   
    def __init__(self, name="MSE_80pct_NoCascade", parameters=None):
        super().__init__(name, parameters or {})
        # CASCADE PREVENTION: DISABLED
        self.enable_cascade_prevention = False
        # EXIT THRESHOLD: 80%
        self.exit_threshold = 0.8
        
    def prepare_data(self, df: pd.DataFrame, ticker: str, pull_date: str) -> pd.DataFrame:
        """
        Implements the required prepare_data method from StrategyBase.
        """
        return self.prepare_strategy_data(df, pull_date, ticker)
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on prepared data.
        This is actually already done in prepare_strategy_data, so just return the DataFrame.
        """
        # The signals are already generated in prepare_strategy_data,
        # so we just need to ensure they exist
        if 'entry_signal_buy' not in df.columns or 'entry_signal_sell' not in df.columns:
            self.logger.error("Signal columns missing from DataFrame")
            return pd.DataFrame()
        return df
        
    def prepare_strategy_data(
        self,
        base_1m: pd.DataFrame,
        pull_date: str,
        ticker: str,
        last_processed_timestamp: Optional[pd.Timestamp] = None
    ) -> Optional[pd.DataFrame]:
        """
        MSE Strategy Implementation
        """
        if base_1m is None or base_1m.empty:
            logging.warning(f"No base data for ticker '{ticker}' on '{pull_date}'.")
            return None

        if 'timestamp' not in base_1m.columns:
            logging.error(f"'timestamp' column not found in base data for {ticker} on '{pull_date}'.")
            return None

        logging.info(f"MSE Strategy 80% NO Cascade - LIVE MATCHING: Starting for {ticker}, date={pull_date}, total rows={len(base_1m)}")

        # Ensure 'timestamp' is datetime and sorted
        base_1m['timestamp'] = pd.to_datetime(base_1m['timestamp'])
        base_1m = base_1m.sort_values('timestamp').reset_index(drop=True)

        # 1) Resample to 5-minute
        df_5m = resample_ohlc(base_1m, '5min')
        if df_5m.empty:
            logging.error(f"Resampled 5-minute data is empty for {ticker} on '{pull_date}'.")
            return None

        # Compute MACD & EMAs on 5-minute
        df_5m = compute_macd(df_5m, 'close', prefix='5m_')
        df_5m = compute_ema(df_5m, span=9, prefix='5m_')
        df_5m = compute_ema(df_5m, span=20, prefix='5m_')

        # 2) Resample to 15-minute
        df_15m = resample_ohlc(base_1m, '15min')
        if df_15m.empty:
            logging.error(f"Resampled 15-minute data is empty for {ticker} on '{pull_date}'.")
            return None

        # Compute MACD & EMAs on 15-minute
        df_15m = compute_macd(df_15m, 'close', prefix='15m_')
        df_15m = compute_ema(df_15m, span=9, prefix='15m_')
        df_15m = compute_ema(df_15m, span=20, prefix='15m_')

        # 3) Merge them back to 1-minute via forward fill
        df_merged_5m = forward_fill_to_1m(df_5m, base_1m, '5m_')
        full_df = forward_fill_to_1m(df_15m, df_merged_5m, '15m_')
        
        # 4) Warm-up skip - CRITICAL: Must allow sufficient time for stable MACD calculations
        # 15min MACD requires 35 candles: 35 * 15 = 525 minutes (8.75 hours)
        # 5min MACD requires 35 candles: 35 * 5 = 175 minutes (2.9 hours)  
        # Use the longer requirement to ensure both timeframes have stable indicators
        skip_minutes = 525  # Proper warmup period for 15min MACD stability

        # Calculate the timestamp to start from
        if len(full_df) == 0:
            logging.error(f"Full DataFrame is empty after merging for {ticker} on '{pull_date}'.")
            return None

        first_timestamp = full_df['timestamp'].iloc[0]
        start_timestamp = first_timestamp + pd.Timedelta(minutes=skip_minutes)

        # Filter the DataFrame
        final_df = full_df[full_df['timestamp'] >= start_timestamp].copy()
        final_df.reset_index(drop=True, inplace=True)

        logging.info(f"MSE Strategy 80% NO Cascade - LIVE MATCHING: After warm-up, final rows={len(final_df)} for {ticker} on '{pull_date}'")

        # 5) Define buy/sell logic
        # Initialize exit signals
        final_df['exit_signal_buy'] = False
        final_df['exit_signal_sell'] = False
        
        # Generate raw technical signals using CURRENT bar indicators (like live) to match live system logic
        raw_buy_signal = (
            (final_df['5m_macd_line'] > final_df['5m_signal_line']) &
            (final_df['5m_ema_9'] > final_df['5m_ema_20']) &
            (final_df['15m_macd_line'] > final_df['15m_signal_line']) &
            (final_df['15m_ema_9'] > final_df['15m_ema_20'])
        )

        raw_sell_signal = (
            (final_df['5m_macd_line'] < final_df['5m_signal_line']) &
            (final_df['5m_ema_9'] < final_df['5m_ema_20']) &
            (final_df['15m_macd_line'] < final_df['15m_signal_line']) &
            (final_df['15m_ema_9'] < final_df['15m_ema_20'])
        )
        
        # Apply CASCADE PREVENTION filter to raw signals
        if self.enable_cascade_prevention:
            final_df['entry_signal_buy'], final_df['entry_signal_sell'] = self._apply_cascade_prevention(
                final_df, raw_buy_signal, raw_sell_signal, ticker, pull_date
            )
            logging.info(f"CASCADE PREVENTION applied for {ticker} on {pull_date}")
        else:
            final_df['entry_signal_buy'] = raw_buy_signal
            final_df['entry_signal_sell'] = raw_sell_signal

        # Check if 'entry_signal_buy' and 'entry_signal_sell' exist
        if 'entry_signal_buy' not in final_df.columns or 'entry_signal_sell' not in final_df.columns:
            logging.error("'entry_signal_buy' or 'entry_signal_sell' columns are missing in the final DataFrame.")
            return None

        # Initialize trade flags and trackers
        in_buy_trade = False
        in_sell_trade = False
        buy_entry_pending = False
        sell_entry_pending = False
        buy_exit_pending = False
        sell_exit_pending = False
        # Track indices where detection happened so we can shift entry signals to next bar
        buy_detect_idx: Optional[int] = None
        sell_detect_idx: Optional[int] = None
        buy_max_hist = 0
        sell_min_hist = 0
        buy_peak_initialized = False
        sell_peak_initialized = False

        # Columns to guide backtest trade extraction to use OPEN prices on the entry/exit bar
        final_df['use_open_for_entry'] = False
        final_df['use_open_for_exit'] = False

        # Iterate through the DataFrame to determine entry and exit signals
        for idx, row in final_df.iterrows():
            # Skip first bar entirely to avoid any look-ahead bias
            if idx == 0:
                continue
            # Handle pending entries (enter at OPEN price of bar AFTER signal detection)
            if buy_entry_pending and not in_buy_trade:
                # Enter at NEXT bar OPEN. Mark the entry on THIS row so extractor aligns with live.
                in_buy_trade = True
                buy_entry_pending = False
                buy_peak_initialized = False  # Will initialize peak tracking on first update
                entry_price = row['open']  # Enter at OPEN price
                # Shift the entry signal to this bar and clear detection bar signal to avoid early entries
                final_df.at[idx, 'entry_signal_buy'] = True
                final_df.at[idx, 'use_open_for_entry'] = True
                if buy_detect_idx is not None:
                    final_df.at[buy_detect_idx, 'entry_signal_buy'] = False
                    buy_detect_idx = None
                logging.info(f"Buy trade ENTERED at {row['timestamp']} OPEN price: {entry_price}. Peak tracking will initialize on first update.")
            
            if sell_entry_pending and not in_sell_trade:
                # Enter at NEXT bar OPEN. Mark the entry on THIS row so extractor aligns with live.
                in_sell_trade = True
                sell_entry_pending = False
                sell_peak_initialized = False  # Will initialize peak tracking on first update
                entry_price = row['open']  # Enter at OPEN price
                final_df.at[idx, 'entry_signal_sell'] = True
                final_df.at[idx, 'use_open_for_entry'] = True
                if sell_detect_idx is not None:
                    final_df.at[sell_detect_idx, 'entry_signal_sell'] = False
                    sell_detect_idx = None
                logging.info(f"Sell trade ENTERED at {row['timestamp']} OPEN price: {entry_price}. Peak tracking will initialize on first update.")

            # Handle pending exits (exit at OPEN price of bar AFTER exit condition detected)
            if buy_exit_pending and in_buy_trade:
                final_df.at[idx, 'exit_signal_buy'] = True
                final_df.at[idx, 'use_open_for_exit'] = True
                exit_price = row['open']  # Exit at OPEN price
                in_buy_trade = False
                buy_exit_pending = False
                logging.info(f"Buy trade EXITED at {row['timestamp']} OPEN price: {exit_price}")
                
            if sell_exit_pending and in_sell_trade:
                final_df.at[idx, 'exit_signal_sell'] = True
                final_df.at[idx, 'use_open_for_exit'] = True
                exit_price = row['open']  # Exit at OPEN price
                in_sell_trade = False
                sell_exit_pending = False
                logging.info(f"Sell trade EXITED at {row['timestamp']} OPEN price: {exit_price}")

            # Detect new entry signals (decision made at CLOSE, enter on NEXT bar's OPEN)
            if not in_buy_trade and not buy_entry_pending and row['entry_signal_buy']:
                buy_entry_pending = True
                buy_detect_idx = idx
                logging.info(f"Buy entry signal detected at {row['timestamp']} CLOSE - will enter next bar at OPEN")

            if not in_sell_trade and not sell_entry_pending and row['entry_signal_sell']:
                sell_entry_pending = True
                sell_detect_idx = idx
                logging.info(f"Sell entry signal detected at {row['timestamp']} CLOSE - will enter next bar at OPEN")

            # Update peak/valley tracking and check exit conditions using CURRENT bar data
            if in_buy_trade and not buy_exit_pending:
                # Use CURRENT bar MACD Histogram to match live system logic
                current_macd_hist = row['15m_macd_hist'] if idx > 0 else None
                
                if current_macd_hist is not None:
                    # Initialize peak tracking on first iteration after entry
                    if not buy_peak_initialized:
                        buy_max_hist = current_macd_hist
                        buy_peak_initialized = True
                        logging.info(f"Buy peak tracking initialized with CURRENT bar MACD: {buy_max_hist}")
                    
                    # Update maximum MACD Histogram since entry using CURRENT bar
                    if current_macd_hist > buy_max_hist:
                        buy_max_hist = current_macd_hist

                    # Check exit condition for Buy using CURRENT bar - CONFIGURABLE THRESHOLD
                    if buy_peak_initialized and current_macd_hist < self.exit_threshold * buy_max_hist:
                        buy_exit_pending = True
                        logging.info(f"Buy exit condition detected using CURRENT bar data - will exit at {row['timestamp']} OPEN. MACD Hist: {current_macd_hist}")

            if in_sell_trade and not sell_exit_pending:
                # Use CURRENT bar MACD Histogram to match live system logic
                current_macd_hist = row['15m_macd_hist'] if idx > 0 else None
                
                if current_macd_hist is not None:
                    # Initialize peak tracking on first iteration after entry
                    if not sell_peak_initialized:
                        sell_min_hist = current_macd_hist
                        sell_peak_initialized = True
                        logging.info(f"Sell peak tracking initialized with CURRENT bar MACD: {sell_min_hist}")
                    
                    # Update minimum MACD Histogram since entry using CURRENT bar
                    if current_macd_hist < sell_min_hist:
                        sell_min_hist = current_macd_hist

                    # Check exit condition for Sell using CURRENT bar - CONFIGURABLE THRESHOLD
                    if sell_peak_initialized and current_macd_hist > self.exit_threshold * sell_min_hist:
                        sell_exit_pending = True
                        logging.info(f"Sell exit condition detected using CURRENT bar data - will exit at {row['timestamp']} OPEN. MACD Hist: {current_macd_hist}")

        logging.info(f"MSE Strategy 80% NO Cascade - LIVE MATCHING: Completed for {ticker} on {pull_date}")

        return final_df

    def _apply_cascade_prevention(self, df: pd.DataFrame, raw_buy_signal: pd.Series, 
                                raw_sell_signal: pd.Series, ticker: str, pull_date: str):
        """
        Apply CASCADE PREVENTION logic to filter same-direction re-entries.
        """
        
        # Initialize filtered signals (start as copies of raw signals)
        filtered_buy = raw_buy_signal.copy()
        filtered_sell = raw_sell_signal.copy()
        
        # Process each timestamp to apply direction alternation rule PER DAY
        filtered_trades_count = 0
        daily_trade_log = {}  # Track trades per individual day: {date_string: [trade_directions]}
        
        for idx, row in df.iterrows():
            current_buy = raw_buy_signal.iloc[idx] if idx < len(raw_buy_signal) else False
            current_sell = raw_sell_signal.iloc[idx] if idx < len(raw_sell_signal) else False
            
            # If no signal, continue
            if not current_buy and not current_sell:
                continue
                
            # Extract INDIVIDUAL trading date from timestamp
            trade_date = row['timestamp'].strftime('%Y-%m-%d')
            
            # Initialize daily log if not exists
            if trade_date not in daily_trade_log:
                daily_trade_log[trade_date] = []
            
            trade_log = daily_trade_log[trade_date]
            
            # Determine signal direction
            signal_direction = 'BUY' if current_buy else 'SELL'
            
            # Apply CASCADE PREVENTION logic - SAME DAY ONLY
            if len(trade_log) == 0:
                # First trade of the day - always allowed
                trade_log.append(signal_direction)
                logging.info(f"CASCADE PREVENTION: First trade of day {trade_date} allowed - {signal_direction} for {ticker} at {row['timestamp']}")
                
            else:
                # Check if this direction matches the last trade OF THE SAME DAY
                last_direction = trade_log[-1]
                
                if signal_direction == last_direction:
                    # SAME DIRECTION ON SAME DAY - REJECT TRADE (CASCADE PREVENTION)
                    if current_buy:
                        filtered_buy.iloc[idx] = False
                    if current_sell:
                        filtered_sell.iloc[idx] = False
                    
                    filtered_trades_count += 1
                    logging.info(f"CASCADE PREVENTION: {signal_direction} trade REJECTED for {ticker} on {trade_date} at {row['timestamp']} - Same direction as last trade on same day ({last_direction})")
                    
                else:
                    # OPPOSITE DIRECTION ON SAME DAY - ALLOW TRADE
                    trade_log.append(signal_direction)
                    logging.info(f"CASCADE PREVENTION: {signal_direction} trade ALLOWED for {ticker} on {trade_date} at {row['timestamp']} - Opposite to last trade on same day ({last_direction})")
        
        # Log summary
        total_raw_signals = raw_buy_signal.sum() + raw_sell_signal.sum()
        total_filtered_signals = filtered_buy.sum() + filtered_sell.sum()
        total_days_processed = len(daily_trade_log)
        
        logging.info(f"CASCADE PREVENTION Summary for {ticker}:")
        logging.info(f"  Raw signals: {total_raw_signals}")
        logging.info(f"  Filtered signals: {total_filtered_signals}")
        logging.info(f"  Trades rejected: {filtered_trades_count}")
        logging.info(f"  Days processed: {total_days_processed}")
        
        return filtered_buy, filtered_sell
