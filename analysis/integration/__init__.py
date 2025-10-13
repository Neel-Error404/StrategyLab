#!/usr/bin/env python3
"""
Integration Module - Clean API for Trade-Base Data Enhancement
==============================================================

SIMPLE USER API - Only 1 function, user provides explicit paths:

    from analysis.integration import enhance_trades

    # User provides explicit paths (no guessing!)
    enhanced_data = enhance_trades(
        trade_file="path/to/trades.csv",
        base_data_dir="path/to/base_data"
    )

That's it! Your trade data now has 20+ additional columns with indicator context.

Author: Financial Analysis Team
Version: 3.0 - User-Explicit Paths API
"""

import pandas as pd
from pathlib import Path
from .core.trade_enhancer import enhance_trades as _enhance_trades

def enhance_trades(trade_file: str,
                  base_data_dir: str,
                  sample_size: int = None) -> pd.DataFrame:
    """
    **MAIN API FUNCTION** - Enhance trade data with base data context.

    Args:
        trade_file: Path to CSV file with trade records
        base_data_dir: Directory containing base data CSV files
        sample_size: Optional limit to N trades (useful for large datasets)

    Returns:
        Enhanced DataFrame with 20+ additional columns:
        - OHLCV data at entry/exit points
        - Indicator values (MACD, EMA) at entry/exit
        - Trade timing and duration metrics
        - Signal quality assessments

    Required Trade File Columns:
        - ticker: Stock symbol
        - Entry Time: Trade entry timestamp
        - Exit Time: Trade exit timestamp
        - Profit (Currency): Trade P&L
        - Trade Type: BUY/SELL

    Required Base Data Directory Structure:
        base_data_dir/
        ├── TICKER1_Base_*.csv
        ├── TICKER2_Base_*.csv
        └── ...

    Example:
        enhanced = enhance_trades(
            trade_file="/path/to/all_trade_merged.csv",
            base_data_dir="/path/to/outputs/.../data/base_data"
        )

        # Sample for quick analysis
        enhanced = enhance_trades(
            trade_file="/path/to/trades.csv",
            base_data_dir="/path/to/base_data",
            sample_size=1000
        )

        # Now analyze with enhanced context
        macd_analysis = enhanced.groupby('ticker')['macd_change'].mean()
    """

    # Validate inputs
    trade_path = Path(trade_file)
    base_path = Path(base_data_dir)

    if not trade_path.exists():
        raise FileNotFoundError(f"Trade file not found: {trade_file}")

    if not base_path.exists():
        raise FileNotFoundError(f"Base data directory not found: {base_data_dir}")

    # Load trade data
    print(f"📊 Loading trade data from: {trade_file}")
    trade_data = pd.read_csv(trade_file)

    # Sample data if requested
    if sample_size and len(trade_data) > sample_size:
        print(f"📊 Sampling {sample_size:,} trades from {len(trade_data):,} total")
        trade_data = trade_data.sample(n=sample_size, random_state=42)

    # Call the core enhancement function
    return _enhance_trades(trade_data, base_data_dir)

# Expose main function at package level
__all__ = ['enhance_trades']