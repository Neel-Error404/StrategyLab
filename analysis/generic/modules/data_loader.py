#!/usr/bin/env python3
"""
Data Loader Module
==================

Loads trade data and base data from configuration.

Usage:
    from modules.data_loader import load_trades, load_base_data

    trades_df = load_trades(config)
    base_data = load_base_data(config, ticker='RELIANCE')
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import glob

def load_trades(config: Dict, paths: Dict = None, sample_size: int = None) -> pd.DataFrame:
    """
    Load merged trades CSV file.

    Args:
        config: Configuration dictionary
        paths: Resolved paths dictionary (if None, will resolve from config)
        sample_size: Optional number of trades to sample (for testing)

    Returns:
        DataFrame with trade data

    Raises:
        FileNotFoundError: If merged trades file doesn't exist
    """
    if paths is None:
        from .config_loader import resolve_paths
        paths = resolve_paths(config)

    merged_file = Path(paths['merged_trades_file'])

    if not merged_file.exists():
        raise FileNotFoundError(
            f"Merged trades file not found: {merged_file}\n"
            f"\n"
            f"Run merge script first:\n"
            f"  python ../utils/merge_trades.py --config config.yaml"
        )

    print(f"📊 Loading trade data from: {merged_file.name}")

    # Load with optimized dtypes for memory efficiency
    dtype_dict = {
        'Trade Type': 'category',
        'Entry Price': 'float32',
        'High During Trade': 'float32',
        'Low During Trade': 'float32',
        'Exit Price': 'float32',
        'Profit (Currency)': 'float32',
        'Profit (%)': 'float32',
        'Trade Duration (min)': 'float32',
        'Drawdown (%)': 'float32',
        'RRR': 'float32',
        'Recovery Time (min)': 'float32',
        'ticker': 'category',
        'strategy_generated': 'bool',
        'risk_processed': 'bool'
    }

    df = pd.read_csv(
        merged_file,
        dtype=dtype_dict,
        parse_dates=['Entry Time', 'Exit Time']
    )

    print(f"✅ Loaded {len(df):,} trades")
    print(f"   Date range: {df['Entry Time'].min()} to {df['Entry Time'].max()}")
    print(f"   Tickers: {df['ticker'].nunique()}")

    # Sample if requested (useful for quick testing)
    if sample_size and len(df) > sample_size:
        print(f"📊 Sampling {sample_size:,} trades for faster analysis...")
        df = df.sample(n=sample_size, random_state=42).sort_values('Entry Time')

    return df

def load_base_data(config: Dict, ticker: str, paths: Dict = None) -> Optional[pd.DataFrame]:
    """
    Load base data (indicators) for a specific ticker.

    Base data contains:
    - OHLCV at each minute
    - Technical indicators (MACD, EMA, etc.)
    - Signal generation context

    Args:
        config: Configuration dictionary
        ticker: Ticker symbol (e.g., 'RELIANCE')
        paths: Resolved paths dictionary (if None, will resolve from config)

    Returns:
        DataFrame with base data, or None if file not found
    """
    if paths is None:
        from .config_loader import resolve_paths
        paths = resolve_paths(config)

    base_data_dir = Path(paths['base_data_dir'])

    if not base_data_dir.exists():
        print(f"⚠️  Base data directory not found: {base_data_dir}")
        return None

    # Find base data file for ticker
    pattern = f"{ticker}_Base_*.csv"
    files = list(base_data_dir.glob(pattern))

    if not files:
        print(f"⚠️  No base data found for {ticker}")
        return None

    # Use first match (should only be one)
    base_file = files[0]

    print(f"📊 Loading base data for {ticker}: {base_file.name}")

    df = pd.read_csv(base_file, parse_dates=['timestamp'])

    print(f"✅ Loaded {len(df):,} data points for {ticker}")

    return df

def load_all_base_data(config: Dict, paths: Dict = None) -> Dict[str, pd.DataFrame]:
    """
    Load base data for all tickers.

    Args:
        config: Configuration dictionary
        paths: Resolved paths dictionary

    Returns:
        Dictionary mapping ticker → base data DataFrame
    """
    if paths is None:
        from .config_loader import resolve_paths
        paths = resolve_paths(config)

    base_data_dir = Path(paths['base_data_dir'])

    if not base_data_dir.exists():
        print(f"⚠️  Base data directory not found: {base_data_dir}")
        return {}

    # Find all base data files
    pattern = "*_Base_*.csv"
    files = list(base_data_dir.glob(pattern))

    if not files:
        print(f"⚠️  No base data files found in {base_data_dir}")
        return {}

    print(f"📊 Loading base data for {len(files)} tickers...")

    base_data_dict = {}

    for base_file in files:
        # Extract ticker from filename: TICKER_Base_*.csv
        ticker = base_file.name.split('_Base_')[0]

        try:
            df = pd.read_csv(base_file, parse_dates=['timestamp'])
            base_data_dict[ticker] = df
            print(f"   ✅ {ticker}: {len(df):,} data points")
        except Exception as e:
            print(f"   ❌ Error loading {ticker}: {e}")
            continue

    print(f"✅ Loaded base data for {len(base_data_dict)} tickers")

    return base_data_dict

def get_trade_columns_info(df: pd.DataFrame) -> Dict:
    """
    Get information about available columns in trade data.

    Useful for debugging and validation.

    Args:
        df: Trade DataFrame

    Returns:
        Dictionary with column information
    """
    return {
        'total_columns': len(df.columns),
        'column_names': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'missing_values': {col: df[col].isna().sum() for col in df.columns if df[col].isna().sum() > 0}
    }

def validate_trade_data(df: pd.DataFrame) -> Dict:
    """
    Validate trade data integrity.

    Checks:
    - Required columns exist
    - No missing critical values
    - Timestamps are valid
    - Entry/Exit times are sequential

    Args:
        df: Trade DataFrame

    Returns:
        Dictionary with validation results
    """
    required_columns = [
        'ticker', 'Entry Time', 'Exit Time',
        'Entry Price', 'Exit Price',
        'Profit (Currency)', 'Trade Type'
    ]

    validation = {
        'valid': True,
        'errors': [],
        'warnings': []
    }

    # Check required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        validation['valid'] = False
        validation['errors'].append(f"Missing required columns: {missing_cols}")

    # Check for missing values in critical columns
    for col in required_columns:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                validation['warnings'].append(f"{col}: {missing_count} missing values")

    # Check timestamp validity
    if 'Entry Time' in df.columns and 'Exit Time' in df.columns:
        invalid_times = (df['Exit Time'] < df['Entry Time']).sum()
        if invalid_times > 0:
            validation['errors'].append(f"{invalid_times} trades have Exit Time before Entry Time")

    # Check profit calculation consistency
    if all(col in df.columns for col in ['Entry Price', 'Exit Price', 'Profit (Currency)', 'Trade Type']):
        # For Buy trades: profit = (Exit - Entry) * quantity
        # For Sell trades: profit = (Entry - Exit) * quantity
        # Simplified check: just verify direction matches
        buy_trades = df[df['Trade Type'] == 'Buy']
        sell_trades = df[df['Trade Type'] == 'Sell']

        buy_inconsistent = ((buy_trades['Exit Price'] > buy_trades['Entry Price']) & (buy_trades['Profit (Currency)'] < 0)).sum()
        sell_inconsistent = ((sell_trades['Entry Price'] > sell_trades['Exit Price']) & (sell_trades['Profit (Currency)'] < 0)).sum()

        if buy_inconsistent > 0:
            validation['warnings'].append(f"{buy_inconsistent} Buy trades: price increased but profit negative")
        if sell_inconsistent > 0:
            validation['warnings'].append(f"{sell_inconsistent} Sell trades: price decreased but profit negative")

    return validation
