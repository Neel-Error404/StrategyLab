# strategies/strategy_mse.py

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
    MSE (Mean Squared Error) strategy implementation.
    Wraps the functional implementation for the strategy factory pattern.
    """
   
    def __init__(self, name="MSE", parameters=None):
        super().__init__(name, parameters or {})
        
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
        MSE Strategy:
          1) We have 1-minute base data (base_1m).
          2) We'll resample to 5-minute and 15-minute, compute MACD + EMAs,
             forward-fill them onto the 1-minute DataFrame.
          3) Perform warm-up skip based on the largest indicator lookback.
          4) Define buy/sell logic.
          5) Return new signals based on recent data.
        """
        if base_1m is None or base_1m.empty:
            logging.warning(f"No base data for ticker '{ticker}' on '{pull_date}'.")
            return None

        if 'timestamp' not in base_1m.columns:
            logging.error(f"'timestamp' column not found in base data for {ticker} on '{pull_date}'.")
            return None

        logging.info(f"Strategy MSE: Starting for {ticker}, date={pull_date}, total rows={len(base_1m)}")

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
        full_df = forward_fill_to_1m(df_15m, df_merged_5m, '15m_')        # 4) Warm-up skip
        # For short-term testing, use a more reasonable warmup period
        # Original: 26 * 15 = 390 minutes (too much for single day data)
        # Adjusted: Use 30 minutes warmup to allow indicators to stabilize
        skip_minutes = 30  # Conservative warmup period

        # Calculate the timestamp to start from
        if len(full_df) == 0:
            logging.error(f"Full DataFrame is empty after merging for {ticker} on '{pull_date}'.")
            return None

        first_timestamp = full_df['timestamp'].iloc[0]
        start_timestamp = first_timestamp + pd.Timedelta(minutes=skip_minutes)

        # Filter the DataFrame
        final_df = full_df[full_df['timestamp'] >= start_timestamp].copy()
        final_df.reset_index(drop=True, inplace=True)

        logging.info(f"Strategy MSE: After warm-up, final rows={len(final_df)} for {ticker} on '{pull_date}'")

        # 5) Define buy/sell logic
        # Initialize exit signals
        final_df['exit_signal_buy'] = False
        final_df['exit_signal_sell'] = False        # Define entry signals using PREVIOUS bar indicators to avoid look-ahead bias
        # In real trading, you can't know current bar values until bar is complete
        final_df['entry_signal_buy'] = (
            (final_df['5m_macd_line'].shift(1) > final_df['5m_signal_line'].shift(1)) &
            (final_df['5m_ema_9'].shift(1) > final_df['5m_ema_20'].shift(1)) &
            (final_df['15m_macd_line'].shift(1) > final_df['15m_signal_line'].shift(1)) &
            (final_df['15m_ema_9'].shift(1) > final_df['15m_ema_20'].shift(1))
        )

        final_df['entry_signal_sell'] = (
            (final_df['5m_macd_line'].shift(1) < final_df['5m_signal_line'].shift(1)) &
            (final_df['5m_ema_9'].shift(1) < final_df['5m_ema_20'].shift(1)) &
            (final_df['15m_macd_line'].shift(1) < final_df['15m_signal_line'].shift(1)) &
            (final_df['15m_ema_9'].shift(1) < final_df['15m_ema_20'].shift(1))
        )

        # Log columns to verify
        logging.info(f"Final DF Columns: {final_df.columns.tolist()}")

        # Check if 'entry_signal_buy' and 'entry_signal_sell' exist
        if 'entry_signal_buy' not in final_df.columns or 'entry_signal_sell' not in final_df.columns:
            logging.error("'entry_signal_buy' or 'entry_signal_sell' columns are missing in the final DataFrame.")
            return None

        # Initialize trade flags and trackers
        in_buy_trade = False
        in_sell_trade = False
        buy_max_hist = 0
        sell_min_hist = 0

        # Iterate through the DataFrame to determine exit signals
        for idx, row in final_df.iterrows():
            # Buy Trade Logic
            if not in_buy_trade and row['entry_signal_buy']:
                in_buy_trade = True
                buy_max_hist = row['15m_macd_hist']
                final_df.at[idx, 'exit_signal_buy'] = False  # Explicitly setting, though default is False
                logging.info(f"Buy trade entered at {row['timestamp']}. MACD Hist: {row['15m_macd_hist']}")
            elif in_buy_trade:
                # Update maximum MACD Histogram since entry (use previous bar to avoid look-ahead)
                prev_macd_hist = final_df.iloc[idx-1]['15m_macd_hist'] if idx > 0 else row['15m_macd_hist']
                if prev_macd_hist > buy_max_hist:
                    buy_max_hist = prev_macd_hist
                    logging.debug(f"Updated buy_max_hist to {buy_max_hist} at {row['timestamp']}")

                # Check exit condition for Buy (use previous bar to avoid look-ahead)
                if prev_macd_hist < 0.2 * buy_max_hist:
                    final_df.at[idx, 'exit_signal_buy'] = True
                    in_buy_trade = False
                    logging.info(f"Buy trade exited at {row['timestamp']}. MACD Hist: {prev_macd_hist}")

            # Sell Trade Logic
            if not in_sell_trade and row['entry_signal_sell']:
                in_sell_trade = True
                sell_min_hist = row['15m_macd_hist']
                final_df.at[idx, 'exit_signal_sell'] = False  # Explicitly setting, though default is False
                logging.info(f"Sell trade entered at {row['timestamp']}. MACD Hist: {row['15m_macd_hist']}")
            elif in_sell_trade:
                # Update minimum MACD Histogram since entry (use previous bar to avoid look-ahead)
                prev_macd_hist = final_df.iloc[idx-1]['15m_macd_hist'] if idx > 0 else row['15m_macd_hist']
                if prev_macd_hist < sell_min_hist:
                    sell_min_hist = prev_macd_hist
                    logging.debug(f"Updated sell_min_hist to {sell_min_hist} at {row['timestamp']}")

                # Check exit condition for Sell (use previous bar to avoid look-ahead)
                if prev_macd_hist > 0.2 * sell_min_hist:
                    final_df.at[idx, 'exit_signal_sell'] = True
                    in_sell_trade = False
                    logging.info(f"Sell trade exited at {row['timestamp']}. MACD Hist: {prev_macd_hist}")

        logging.info(f"Strategy MSE: Completed for {ticker} on {pull_date}. Total trades processed.")

        return final_df
