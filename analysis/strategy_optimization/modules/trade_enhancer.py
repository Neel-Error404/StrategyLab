#!/usr/bin/env python3
"""
trade_enhancer.py - Reusable Trade Data Enhancement Endpoint
============================================================

Simple, clean API for enhancing any trade dataset with base data context.
Designed to be used by any analysis function as an input enhancement layer.

Key Features:
- Single function call: enhance_trades(trade_data, base_data_dir)
- Returns enhanced DataFrame ready for analysis
- Strategy-agnostic: works with any trade format
- High-performance vectorized operations
- Minimal dependencies and clean interface

Usage:
    from analysis.integration.core.trade_enhancer import enhance_trades

    enhanced_df = enhance_trades(your_trade_data, base_data_directory)
    # Now use enhanced_df in your existing analysis functions

Author: Financial Analysis Team
Date: September 2025
Version: 2.0 - Clean API Endpoint
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')

def enhance_trades(trade_data: pd.DataFrame,
                  base_data_dir: str,
                  required_columns: Optional[List[str]] = None,
                  cache_base_data: bool = True) -> pd.DataFrame:
    """
    Main API endpoint: Enhance trade data with base data context.

    Args:
        trade_data: DataFrame with trade records
        base_data_dir: Directory containing base data files
        required_columns: Additional required columns beyond defaults
        cache_base_data: Whether to cache loaded base data files

    Returns:
        Enhanced DataFrame with base data context added to each trade

    Example:
        # Basic usage
        enhanced = enhance_trades(trades_df, "path/to/base_data")

        # Now use in your analysis
        profit_by_macd = enhanced.groupby(pd.cut(enhanced['macd_change'], 5))['profit_currency'].mean()
    """

    # Validate input
    _validate_trade_data(trade_data, required_columns)

    print(f"🔗 Enhancing {len(trade_data):,} trades with base data context...")

    # Prepare trade data
    enhanced_trades = trade_data.copy()
    enhanced_trades['Entry Time'] = pd.to_datetime(enhanced_trades['Entry Time'])
    enhanced_trades['Exit Time'] = pd.to_datetime(enhanced_trades['Exit Time'])

    # Cache for base data
    base_data_cache = {} if cache_base_data else None

    # Process each unique ticker
    tickers = enhanced_trades['ticker'].unique()
    enhanced_records = []

    for ticker in tickers:
        ticker_trades = enhanced_trades[enhanced_trades['ticker'] == ticker]

        try:
            # Load base data for ticker
            base_data = _load_ticker_base_data(ticker, base_data_dir, base_data_cache)

            # Enhance trades for this ticker
            for _, trade in ticker_trades.iterrows():
                enhanced_trade = _enhance_single_trade(trade, base_data)
                enhanced_records.append(enhanced_trade)

        except Exception as e:
            print(f"⚠️ Skipping {ticker}: {e}")
            # Add original trades without enhancement
            for _, trade in ticker_trades.iterrows():
                enhanced_records.append(trade.to_dict())
            continue

    result_df = pd.DataFrame(enhanced_records)

    print(f"✅ Enhancement complete: {len(trade_data.columns)} → {len(result_df.columns)} columns")

    return result_df

def get_enhanced_columns() -> List[str]:
    """
    Get list of columns that will be added by enhancement.
    Useful for analysis functions to know what new data is available.

    Returns:
        List of column names added during enhancement
    """
    return [
        # Base data at entry
        'entry_open', 'entry_high', 'entry_low', 'entry_close', 'entry_volume',
        'entry_timestamp',

        # Base data at exit
        'exit_open', 'exit_high', 'exit_low', 'exit_close', 'exit_volume',
        'exit_timestamp',

        # Trade context
        'trade_duration_minutes', 'trade_intervals',
        'entry_time_alignment_seconds', 'exit_time_alignment_seconds',

        # Indicators (dynamically added based on available data)
        'entry_5m_macd', 'exit_5m_macd', 'macd_change',
        'entry_15m_macd', 'exit_15m_macd', 'macd_15m_change',
        'entry_5m_ema21', 'exit_5m_ema21', 'ema21_change',
        'entry_5m_ema50', 'exit_5m_ema50', 'ema50_change',

        # Signals (if available)
        'entry_buy_entry_signal', 'exit_buy_entry_signal',
        'entry_sell_entry_signal', 'exit_sell_entry_signal'
    ]

def quick_enhance_sample(trade_data: pd.DataFrame,
                        base_data_dir: str,
                        sample_size: int = 1000) -> pd.DataFrame:
    """
    Quick enhancement for sample data - useful for prototyping analysis.

    Args:
        trade_data: Full trade dataset
        base_data_dir: Base data directory
        sample_size: Number of trades to sample

    Returns:
        Enhanced sample DataFrame
    """
    sample_data = trade_data.sample(n=min(sample_size, len(trade_data)), random_state=42)
    return enhance_trades(sample_data, base_data_dir)

# Internal helper functions

def _validate_trade_data(trade_data: pd.DataFrame, additional_columns: Optional[List[str]]) -> None:
    """Validate that trade data has required columns."""
    required = ['ticker', 'Entry Time', 'Exit Time', 'Profit (Currency)', 'Trade Type']
    if additional_columns:
        required.extend(additional_columns)

    missing = [col for col in required if col not in trade_data.columns]
    if missing:
        raise ValueError(f"Trade data missing required columns: {missing}")

def _load_ticker_base_data(ticker: str, base_data_dir: str, cache: Optional[Dict]) -> pd.DataFrame:
    """Load and cache base data for a ticker."""

    if cache is not None and ticker in cache:
        return cache[ticker]

    # Find base data file
    base_dir = Path(base_data_dir)
    base_files = list(base_dir.glob(f"{ticker}_Base_*.csv"))

    if not base_files:
        raise FileNotFoundError(f"No base data file found for {ticker}")

    # Load and prepare base data
    base_data = pd.read_csv(base_files[0])
    base_data['timestamp'] = pd.to_datetime(base_data['timestamp']).dt.tz_localize(None)
    base_data = base_data.sort_values('timestamp').reset_index(drop=True)

    if cache is not None:
        cache[ticker] = base_data

    return base_data

def _enhance_single_trade(trade: pd.Series, base_data: pd.DataFrame) -> Dict:
    """Enhance a single trade with base data context."""

    entry_time = trade['Entry Time']
    exit_time = trade['Exit Time']

    # Find corresponding base data records
    entry_idx = _align_timestamp(entry_time, base_data)
    exit_idx = _align_timestamp(exit_time, base_data)

    entry_data = base_data.iloc[entry_idx]
    exit_data = base_data.iloc[exit_idx]

    # Build enhanced record
    enhanced = trade.to_dict()

    # Add base OHLCV data
    enhanced.update({
        'entry_open': entry_data['open'],
        'entry_high': entry_data['high'],
        'entry_low': entry_data['low'],
        'entry_close': entry_data['close'],
        'entry_volume': entry_data['volume'],
        'entry_timestamp': entry_data['timestamp'],

        'exit_open': exit_data['open'],
        'exit_high': exit_data['high'],
        'exit_low': exit_data['low'],
        'exit_close': exit_data['close'],
        'exit_volume': exit_data['volume'],
        'exit_timestamp': exit_data['timestamp'],

        'trade_duration_minutes': (exit_time - entry_time).total_seconds() / 60,
        'trade_intervals': abs(exit_idx - entry_idx),
        'entry_time_alignment_seconds': (entry_time - entry_data['timestamp']).total_seconds(),
        'exit_time_alignment_seconds': (exit_time - exit_data['timestamp']).total_seconds()
    })

    # Add available indicators
    for col in base_data.columns:
        if any(indicator in col.lower() for indicator in ['macd', 'ema', 'signal']):
            enhanced[f'entry_{col}'] = entry_data[col]
            enhanced[f'exit_{col}'] = exit_data[col]

            # Calculate change for numeric indicators (exclude boolean)
            if pd.api.types.is_numeric_dtype(base_data[col]) and not pd.api.types.is_bool_dtype(base_data[col]):
                change_val = exit_data[col] - entry_data[col]
                enhanced[f'{col}_change'] = change_val

    return enhanced

def _align_timestamp(trade_time: pd.Timestamp, base_data: pd.DataFrame) -> int:
    """Align trade timestamp to nearest base data record."""
    # Find record with timestamp <= trade_time (last available data)
    before_mask = base_data['timestamp'] <= trade_time
    if not before_mask.any():
        return 0
    return before_mask[before_mask].index[-1]

# Convenience functions for common analysis patterns

def add_indicator_efficiency_metrics(enhanced_data: pd.DataFrame) -> pd.DataFrame:
    """
    Add pre-calculated indicator efficiency metrics to enhanced data.

    Args:
        enhanced_data: DataFrame from enhance_trades()

    Returns:
        DataFrame with additional efficiency columns
    """
    result = enhanced_data.copy()

    # MACD efficiency
    if 'macd_change' in result.columns:
        profitable = result['Profit (Currency)'] > 0
        result['macd_efficiency'] = np.where(
            profitable,
            np.where(result['macd_change'] > 0, 'macd_positive_profitable', 'macd_negative_profitable'),
            np.where(result['macd_change'] > 0, 'macd_positive_losing', 'macd_negative_losing')
        )

    # Timing quality
    if 'entry_time_alignment_seconds' in result.columns:
        timing_quality = np.abs(result['entry_time_alignment_seconds']) / 60  # Convert to minutes
        result['timing_quality'] = pd.cut(timing_quality,
                                        bins=[0, 1, 2.5, float('inf')],
                                        labels=['excellent', 'good', 'poor'])

    return result

def get_trade_context_window(enhanced_data: pd.DataFrame,
                           trade_idx: int,
                           base_data_dir: str,
                           context_intervals: int = 10) -> pd.DataFrame:
    """
    Get detailed context window around a specific enhanced trade.

    Args:
        enhanced_data: DataFrame from enhance_trades()
        trade_idx: Index of trade to analyze
        base_data_dir: Base data directory
        context_intervals: Number of 5-minute intervals before/after trade

    Returns:
        DataFrame with context window data
    """
    trade = enhanced_data.iloc[trade_idx]
    ticker = trade['ticker']

    # Load base data
    base_data = _load_ticker_base_data(ticker, base_data_dir, None)

    # Find trade position in base data
    entry_time = pd.to_datetime(trade['entry_timestamp'])
    exit_time = pd.to_datetime(trade['exit_timestamp'])

    entry_idx = _align_timestamp(entry_time, base_data)
    exit_idx = _align_timestamp(exit_time, base_data)

    # Define context window
    start_idx = max(0, entry_idx - context_intervals)
    end_idx = min(len(base_data), exit_idx + context_intervals + 1)

    context = base_data.iloc[start_idx:end_idx].copy()

    # Mark trade boundaries
    context['trade_phase'] = 'before'
    context.loc[entry_idx:exit_idx, 'trade_phase'] = 'during'
    context.loc[exit_idx+1:, 'trade_phase'] = 'after'

    return context

# Usage examples and documentation

def example_usage():
    """
    Example of how to use the trade enhancement API in analysis functions.
    """
    print("Example usage patterns:")
    print("""
    # 1. Basic enhancement
    from analysis.integration.core.trade_enhancer import enhance_trades
    enhanced_df = enhance_trades(trade_data, base_data_dir)

    # 2. Use in existing analysis
    def analyze_macd_efficiency(trade_data, base_data_dir):
        enhanced = enhance_trades(trade_data, base_data_dir)
        return enhanced.groupby('macd_efficiency')['Profit (Currency)'].mean()

    # 3. Sample for quick prototyping
    from analysis.integration.core.trade_enhancer import quick_enhance_sample
    sample_enhanced = quick_enhance_sample(large_trade_data, base_data_dir, 500)

    # 4. Get context for specific trades
    context = get_trade_context_window(enhanced_data, trade_idx=42, base_data_dir)
    """)

if __name__ == "__main__":
    example_usage()