# strategies/mse_20pct_no_cascade_with_macd_exit.py
# MSE Strategy: 20% Exit Threshold + MACD Crossover Exits WITHOUT CASCADE PREVENTION

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
    MSE Strategy: 20% Exit Threshold + MACD Crossover Exits WITHOUT CASCADE PREVENTION
    
    CONFIGURATION:
    - Exit Threshold: 20% of peak (early exits)
    - MACD Crossover Exits: 5min and 15min MACD crossovers trigger exits
    - Cascade Prevention: DISABLED
    - Entry: ALL 4 indicators must align (5min MACD, 15min MACD, 5min EMA, 15min EMA)
    
    EXIT CONDITIONS (ANY triggers exit):
    1. MACD histogram drops below 20% of peak (for BUY) / rises above 20% of valley (for SELL)
    2. 5-minute MACD line crosses below signal line (for BUY) / above signal line (for SELL)
    3. 15-minute MACD line crosses below signal line (for BUY) / above signal line (for SELL)
    """
   
    def __init__(self, name="MSE_20pct_NoCascade_WithMACDExit", parameters=None):
        super().__init__(name, parameters or {})
        # CASCADE PREVENTION: DISABLED
        self.enable_cascade_prevention = False
        # EXIT THRESHOLD: 20%
        self.exit_threshold = 0.2
        
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
        MSE Strategy Implementation with MACD Crossover Exits
        """
        if base_1m is None or base_1m.empty:
            logging.warning(f"No base data for ticker '{ticker}' on '{pull_date}'.")
            return None

        if 'timestamp' not in base_1m.columns:
            logging.error(f"'timestamp' column not found in base data for {ticker} on '{pull_date}'.")
            return None

        logging.info(f"MSE Strategy 20% NO Cascade + MACD Exit: Starting for {ticker}, date={pull_date}, total rows={len(base_1m)}")

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

        logging.info(f"MSE Strategy 20% NO Cascade + MACD Exit: After warm-up, final rows={len(final_df)} for {ticker} on '{pull_date}'")

        # 5) Define buy/sell logic
        # Initialize exit signals
        final_df['exit_signal_buy'] = False
        final_df['exit_signal_sell'] = False
        
        # Generate raw technical signals using PREVIOUS bar indicators to avoid look-ahead bias
        raw_buy_signal = (
            (final_df['5m_macd_line'].shift(1) > final_df['5m_signal_line'].shift(1)) &
            (final_df['5m_ema_9'].shift(1) > final_df['5m_ema_20'].shift(1)) &
            (final_df['15m_macd_line'].shift(1) > final_df['15m_signal_line'].shift(1)) &
            (final_df['15m_ema_9'].shift(1) > final_df['15m_ema_20'].shift(1))
        )

        raw_sell_signal = (
            (final_df['5m_macd_line'].shift(1) < final_df['5m_signal_line'].shift(1)) &
            (final_df['5m_ema_9'].shift(1) < final_df['5m_ema_20'].shift(1)) &
            (final_df['15m_macd_line'].shift(1) < final_df['15m_signal_line'].shift(1)) &
            (final_df['15m_ema_9'].shift(1) < final_df['15m_ema_20'].shift(1))
        )
        
        # No cascade prevention for this strategy
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
        buy_max_hist = 0
        sell_min_hist = 0
        buy_peak_initialized = False
        sell_peak_initialized = False

        # Store previous MACD values for crossover detection
        prev_5m_macd_line = None
        prev_5m_signal_line = None
        prev_15m_macd_line = None
        prev_15m_signal_line = None

        # Iterate through the DataFrame to determine entry and exit signals
        for idx, row in final_df.iterrows():
            # Skip first bar entirely to avoid any look-ahead bias
            if idx == 0:
                continue
            # Handle pending entries (enter at OPEN price of bar AFTER signal detection)
            if buy_entry_pending and not in_buy_trade:
                in_buy_trade = True
                buy_entry_pending = False
                buy_peak_initialized = False  # Will initialize peak tracking on first update
                entry_price = row['open']  # Enter at OPEN price
                logging.info(f"Buy trade ENTERED at {row['timestamp']} OPEN price: {entry_price}. Peak tracking will initialize on first update.")
            
            if sell_entry_pending and not in_sell_trade:
                in_sell_trade = True
                sell_entry_pending = False
                sell_peak_initialized = False  # Will initialize peak tracking on first update
                entry_price = row['open']  # Enter at OPEN price
                logging.info(f"Sell trade ENTERED at {row['timestamp']} OPEN price: {entry_price}. Peak tracking will initialize on first update.")

            # Handle pending exits (exit at OPEN price of bar AFTER exit condition detected)
            if buy_exit_pending and in_buy_trade:
                final_df.at[idx, 'exit_signal_buy'] = True
                exit_price = row['open']  # Exit at OPEN price
                in_buy_trade = False
                buy_exit_pending = False
                logging.info(f"Buy trade EXITED at {row['timestamp']} OPEN price: {exit_price}")
                
            if sell_exit_pending and in_sell_trade:
                final_df.at[idx, 'exit_signal_sell'] = True
                exit_price = row['open']  # Exit at OPEN price
                in_sell_trade = False
                sell_exit_pending = False
                logging.info(f"Sell trade EXITED at {row['timestamp']} OPEN price: {exit_price}")

            # Detect new entry signals (decision made at CLOSE, enter on NEXT bar's OPEN)
            if not in_buy_trade and not buy_entry_pending and row['entry_signal_buy']:
                buy_entry_pending = True
                logging.info(f"Buy entry signal detected at {row['timestamp']} CLOSE - will enter NEXT bar's OPEN")

            if not in_sell_trade and not sell_entry_pending and row['entry_signal_sell']:
                sell_entry_pending = True
                logging.info(f"Sell entry signal detected at {row['timestamp']} CLOSE - will enter NEXT bar's OPEN")

            # Update peak/valley tracking and check exit conditions using PREVIOUS bar data
            if in_buy_trade and not buy_exit_pending:
                # Use PREVIOUS bar MACD Histogram to avoid look-ahead bias
                prev_macd_hist = final_df.iloc[idx-1]['15m_macd_hist'] if idx > 0 else None
                
                if prev_macd_hist is not None:
                    # Initialize peak tracking on first iteration after entry
                    if not buy_peak_initialized:
                        buy_max_hist = prev_macd_hist
                        buy_peak_initialized = True
                        logging.info(f"Buy peak tracking initialized with PREVIOUS bar MACD: {buy_max_hist}")
                    
                    # Update maximum MACD Histogram since entry using PREVIOUS bar
                    if prev_macd_hist > buy_max_hist:
                        buy_max_hist = prev_macd_hist

                    # EXIT CONDITION 1: MACD histogram drops below threshold using PREVIOUS bar
                    threshold_exit = buy_peak_initialized and prev_macd_hist < self.exit_threshold * buy_max_hist
                    
                    # EXIT CONDITION 2: 5-minute MACD crossover using PREVIOUS bars (bearish for buy trades)
                    macd_5m_exit = False
                    prev_prev_5m_macd_line = final_df.iloc[idx-2]['5m_macd_line'] if idx > 1 else None
                    prev_prev_5m_signal_line = final_df.iloc[idx-2]['5m_signal_line'] if idx > 1 else None
                    prev_5m_macd_line = final_df.iloc[idx-1]['5m_macd_line'] if idx > 0 else None
                    prev_5m_signal_line = final_df.iloc[idx-1]['5m_signal_line'] if idx > 0 else None
                    
                    if (prev_prev_5m_macd_line is not None and prev_prev_5m_signal_line is not None and
                        prev_5m_macd_line is not None and prev_5m_signal_line is not None):
                        # Bar N-2: MACD line was above signal line
                        # Bar N-1: MACD line is below signal line = bearish crossover detected on previous bar
                        if (prev_prev_5m_macd_line >= prev_prev_5m_signal_line and 
                            prev_5m_macd_line < prev_5m_signal_line):
                            macd_5m_exit = True
                            logging.info(f"5-minute bearish MACD crossover detected using PREVIOUS bar data for BUY exit")
                    
                    # EXIT CONDITION 3: 15-minute MACD crossover using PREVIOUS bars (bearish for buy trades)
                    macd_15m_exit = False
                    prev_prev_15m_macd_line = final_df.iloc[idx-2]['15m_macd_line'] if idx > 1 else None
                    prev_prev_15m_signal_line = final_df.iloc[idx-2]['15m_signal_line'] if idx > 1 else None
                    prev_15m_macd_line = final_df.iloc[idx-1]['15m_macd_line'] if idx > 0 else None
                    prev_15m_signal_line = final_df.iloc[idx-1]['15m_signal_line'] if idx > 0 else None
                    
                    if (prev_prev_15m_macd_line is not None and prev_prev_15m_signal_line is not None and
                        prev_15m_macd_line is not None and prev_15m_signal_line is not None):
                        # Bar N-2: MACD line was above signal line
                        # Bar N-1: MACD line is below signal line = bearish crossover detected on previous bar
                        if (prev_prev_15m_macd_line >= prev_prev_15m_signal_line and 
                            prev_15m_macd_line < prev_15m_signal_line):
                            macd_15m_exit = True
                            logging.info(f"15-minute bearish MACD crossover detected using PREVIOUS bar data for BUY exit")

                    # Exit if ANY condition is met (only after peak is initialized)
                    if buy_peak_initialized and (threshold_exit or macd_5m_exit or macd_15m_exit):
                        buy_exit_pending = True
                        exit_reason = []
                        if threshold_exit:
                            exit_reason.append("Threshold")
                        if macd_5m_exit:
                            exit_reason.append("5m_MACD_Cross")
                        if macd_15m_exit:
                            exit_reason.append("15m_MACD_Cross")
                        logging.info(f"Buy exit condition detected using PREVIOUS bar data - will exit at {row['timestamp']} OPEN. Reasons: {exit_reason}. MACD Hist: {prev_macd_hist}")

            if in_sell_trade and not sell_exit_pending:
                # Use PREVIOUS bar MACD Histogram to avoid look-ahead bias
                prev_macd_hist = final_df.iloc[idx-1]['15m_macd_hist'] if idx > 0 else None
                
                if prev_macd_hist is not None:
                    # Initialize peak tracking on first iteration after entry
                    if not sell_peak_initialized:
                        sell_min_hist = prev_macd_hist
                        sell_peak_initialized = True
                        logging.info(f"Sell peak tracking initialized with PREVIOUS bar MACD: {sell_min_hist}")
                    
                    # Update minimum MACD Histogram since entry using PREVIOUS bar
                    if prev_macd_hist < sell_min_hist:
                        sell_min_hist = prev_macd_hist

                    # EXIT CONDITION 1: MACD histogram rises above threshold using PREVIOUS bar
                    threshold_exit = sell_peak_initialized and prev_macd_hist > self.exit_threshold * sell_min_hist
                    
                    # EXIT CONDITION 2: 5-minute MACD crossover using PREVIOUS bars (bullish for sell trades)
                    macd_5m_exit = False
                    prev_prev_5m_macd_line = final_df.iloc[idx-2]['5m_macd_line'] if idx > 1 else None
                    prev_prev_5m_signal_line = final_df.iloc[idx-2]['5m_signal_line'] if idx > 1 else None
                    prev_5m_macd_line = final_df.iloc[idx-1]['5m_macd_line'] if idx > 0 else None
                    prev_5m_signal_line = final_df.iloc[idx-1]['5m_signal_line'] if idx > 0 else None
                    
                    if (prev_prev_5m_macd_line is not None and prev_prev_5m_signal_line is not None and
                        prev_5m_macd_line is not None and prev_5m_signal_line is not None):
                        # Bar N-2: MACD line was below signal line
                        # Bar N-1: MACD line is above signal line = bullish crossover detected on previous bar
                        if (prev_prev_5m_macd_line <= prev_prev_5m_signal_line and 
                            prev_5m_macd_line > prev_5m_signal_line):
                            macd_5m_exit = True
                            logging.info(f"5-minute bullish MACD crossover detected using PREVIOUS bar data for SELL exit")
                    
                    # EXIT CONDITION 3: 15-minute MACD crossover using PREVIOUS bars (bullish for sell trades)
                    macd_15m_exit = False
                    prev_prev_15m_macd_line = final_df.iloc[idx-2]['15m_macd_line'] if idx > 1 else None
                    prev_prev_15m_signal_line = final_df.iloc[idx-2]['15m_signal_line'] if idx > 1 else None
                    prev_15m_macd_line = final_df.iloc[idx-1]['15m_macd_line'] if idx > 0 else None
                    prev_15m_signal_line = final_df.iloc[idx-1]['15m_signal_line'] if idx > 0 else None
                    
                    if (prev_prev_15m_macd_line is not None and prev_prev_15m_signal_line is not None and
                        prev_15m_macd_line is not None and prev_15m_signal_line is not None):
                        # Bar N-2: MACD line was below signal line
                        # Bar N-1: MACD line is above signal line = bullish crossover detected on previous bar
                        if (prev_prev_15m_macd_line <= prev_prev_15m_signal_line and 
                            prev_15m_macd_line > prev_15m_signal_line):
                            macd_15m_exit = True
                            logging.info(f"15-minute bullish MACD crossover detected using PREVIOUS bar data for SELL exit")

                    # Exit if ANY condition is met (only after peak is initialized)
                    if sell_peak_initialized and (threshold_exit or macd_5m_exit or macd_15m_exit):
                        sell_exit_pending = True
                        exit_reason = []
                        if threshold_exit:
                            exit_reason.append("Threshold")
                        if macd_5m_exit:
                            exit_reason.append("5m_MACD_Cross")
                        if macd_15m_exit:
                            exit_reason.append("15m_MACD_Cross")
                        logging.info(f"Sell exit condition detected using PREVIOUS bar data - will exit at {row['timestamp']} OPEN. Reasons: {exit_reason}. MACD Hist: {prev_macd_hist}")

            # Note: No longer need to store current MACD values since we use direct DataFrame lookups

        logging.info(f"MSE Strategy 20% NO Cascade + MACD Exit: Completed for {ticker} on {pull_date}")

        return final_df