# indicators.py

import pandas as pd

def compute_macd(df: pd.DataFrame, column: str = 'close') -> pd.DataFrame:
    """
    Compute MACD indicators.
    """
    df['ema_short'] = df[column].ewm(span=12, adjust=False).mean()
    df['ema_long'] = df[column].ewm(span=26, adjust=False).mean()
    df['macd_line'] = df['ema_short'] - df['ema_long']
    df['signal_line'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['signal_line']
    return df

def compute_ema(df: pd.DataFrame, span: int, column: str = 'close') -> pd.DataFrame:
    """
    Compute Exponential Moving Average (EMA).
    """
    df[f'ema_{span}'] = df[column].ewm(span=span, adjust=False).mean()
    return df

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all necessary indicators.
    """
    df = compute_macd(df)
    df = compute_ema(df, span=9)
    df = compute_ema(df, span=20)
    # Add more indicator computations here if needed
    return df
