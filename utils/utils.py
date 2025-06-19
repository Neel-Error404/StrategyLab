# utils.py

import pandas as pd

def resample_data(minute_df: pd.DataFrame, timeframe: int) -> pd.DataFrame:
    """
    Resample 1-minute DataFrame to specified minute intervals in memory.
    
    E.g., timeframe=5 -> 5-minute bars
          timeframe=1440 -> daily bars
    """
    df = minute_df.copy()
    # Ensure 'timestamp' is the index
    df.set_index('timestamp', inplace=True, drop=True)

    if timeframe < 1440:
        rule = f"{timeframe}min"  # '5min' for 5 minutes, etc.
    else:
        rule = '1D'  # Daily bars

    # Resample using OHLC + volume
    resampled = df.resample(rule, label='right', closed='right').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'ticker': 'first'
    }).dropna(subset=['open'])

    # Reset index to bring 'timestamp' back as a column
    resampled.reset_index(inplace=True)
    resampled.sort_values('timestamp', inplace=True)

    return resampled
