import pandas as pd
from config import BACKTESTER_CONFIG
import os
import logging
import glob
from typing import Optional
from pathlib import Path

def load_base_data(pull_date: str, ticker: str) -> Optional[pd.DataFrame]:
    """
    Load base data for a given ticker and date from available timeframes.
    
    Args:
        pull_date: Date range string in format 'YYYY-MM-DD_to_YYYY-MM-DD'
        ticker: Ticker symbol to load data for
    
    Returns:
        DataFrame with historical price data or None if no data found
    """
    # Try multiple timeframes in order of preference
    timeframes_to_try = ["1m", "day", "5m", "15m", "1h"]
    csv_files = []
    
    for tf in timeframes_to_try:
        if tf in BACKTESTER_CONFIG['TIMEFRAME_FOLDERS']:
            base_file_pattern = os.path.join(
                BACKTESTER_CONFIG['DATA_POOL_DIR'],
                pull_date,
                BACKTESTER_CONFIG['TIMEFRAME_FOLDERS'][tf],
                f"{ticker}_*.csv"
            )
            csv_files = glob.glob(base_file_pattern)
            if csv_files:
                logging.info(f"Found {len(csv_files)} CSV files for {ticker} in {tf} timeframe")
                break

    if not csv_files:
        logging.warning(f"No CSV files found for ticker '{ticker}' on date range '{pull_date}' in any supported timeframe.")
        return None

    data_frames = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Identify and rename the timestamp column
            timestamp_cols = ['timestamp', 'datetime', 'time']
            found = False
            for col in timestamp_cols:
                if col in df.columns:
                    df.rename(columns={col: 'timestamp'}, inplace=True)
                    found = True
                    break
            if not found:
                logging.error(f"No recognized timestamp column found in file '{file}'. Expected one of {timestamp_cols}.")
                return None

            # Ensure 'timestamp' is in datetime format
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            if df['timestamp'].isnull().any():
                logging.error(f"Some 'timestamp' values could not be parsed in file '{file}'.")
                return None

            data_frames.append(df)
        except Exception as e:
            logging.error(f"Error reading file '{file}': {e}")

    if not data_frames:
        logging.warning(f"No valid data loaded for ticker '{ticker}' on date range '{pull_date}'.")
        return None

    combined_df = pd.concat(data_frames, ignore_index=True)
    combined_df.drop_duplicates(subset=['timestamp'], inplace=True)
    combined_df.sort_values('timestamp', inplace=True)
    combined_df.reset_index(drop=True, inplace=True)

    return combined_df