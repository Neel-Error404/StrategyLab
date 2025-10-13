#!/usr/bin/env python3
"""
ANTI-CASCADING TRADES SUBSET (Config-Driven)
=============================================

Purpose: Create anti-cascading dataset using PERCENTAGE calculations
- Based on foundation analysis (Script 00), using Anti-Cascading Top 50
- Filter for affordable tickers (configurable price threshold)
- Exclude cascading (CONSECUTIVE_SAME_DIRECTION) trades

Input: Merged trades + TOP50_ANTICASCADING_TRADES.csv from foundation analysis
Output: Filtered trades CSV for portfolio construction

Author: Portfolio Construction Team
Version: 2.0 - Config-Driven (migrated October 2025)
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PORTFOLIO_DIR = SCRIPT_DIR.parent
ANALYSIS_DIR = PORTFOLIO_DIR.parent
sys.path.insert(0, str(ANALYSIS_DIR))

from generic.modules.config_loader import load_config, resolve_paths, get_output_dir, get_module_spec
from generic.modules.data_loader import load_trades, validate_trade_data


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Anti-Cascading Subset Filter')
    parser.add_argument('--config', required=True, help='Path to YAML configuration file')
    return parser.parse_args()


def load_anticascading_top50(config):
    """
    Load the Anti-Cascading Top 50 from foundation analysis output
    """

    print("🔧 LOADING ANTI-CASCADING TOP 50 FROM FOUNDATION ANALYSIS")
    print("=" * 75)

    # Get output directory from previous module (ticker_ranking)
    prev_output_dir = Path(get_output_dir(config, 'ticker_ranking', category='portfolio'))

    try:
        anticascading_top50_file = prev_output_dir / "TOP50_ANTICASCADING_TRADES.csv"
        anticascading_top50_df = pd.read_csv(anticascading_top50_file)
        correct_top50_tickers = anticascading_top50_df['ticker'].tolist()

        print(f"✅ Loaded Anti-Cascading Top 50 list")
        print(f"✅ Top 10 performers: {', '.join(correct_top50_tickers[:10])}")
        print(f"📁 Source: {anticascading_top50_file}")

        return correct_top50_tickers, anticascading_top50_df

    except FileNotFoundError:
        print(f"❌ Foundation analysis output not found at: {prev_output_dir}")
        print("❌ Please run 00_foundation_cascade_vs_anticascade_analysis.py first!")
        raise


def filter_trades_for_top50(trades_df, correct_top50_tickers):
    """
    Filter trades for correct Anti-Cascading Top 50 tickers only
    """

    print(f"\n📊 FILTERING TRADES FOR TOP 50 TICKERS")
    print("=" * 50)

    print(f"✅ Original dataset: {len(trades_df):,} total trades")
    print(f"✅ Date range: {trades_df['Entry Time'].min()} to {trades_df['Exit Time'].max()}")

    # Filter for Top 50 tickers only
    trades_top50 = trades_df[trades_df['ticker'].isin(correct_top50_tickers)].copy()
    print(f"✅ Filtered to {len(trades_top50):,} trades from Top 50 tickers")

    return trades_top50


def identify_affordable_tickers(trades_top50, anticascading_top50_df, price_threshold=2000):
    """
    Find tickers under price threshold from the Anti-Cascading Top 50
    """

    print(f"\n🔍 IDENTIFYING AFFORDABLE TICKERS (Under ₹{price_threshold:,.0f})")
    print("=" * 60)

    # Get last trade for each ticker to determine current price
    last_trades = trades_top50.groupby('ticker').last().reset_index()

    # Create price analysis
    price_analysis = []
    for _, trade in last_trades.iterrows():
        ticker = trade['ticker']
        current_price = trade['Exit Price']  # Last known price
        last_date = trade['Exit Time'].date() if hasattr(trade['Exit Time'], 'date') else trade['Exit Time']

        # Get performance metrics from Anti-Cascading analysis
        ticker_metrics = anticascading_top50_df[anticascading_top50_df['ticker'] == ticker]
        if len(ticker_metrics) > 0:
            rank = ticker_metrics.iloc[0]['rank']
            composite_score = ticker_metrics.iloc[0]['composite_score']
            profit_factor = ticker_metrics.iloc[0]['profit_factor']
            sharpe_like_ratio = ticker_metrics.iloc[0]['sharpe_like_ratio']
        else:
            rank, composite_score, profit_factor, sharpe_like_ratio = 0, 0, 0, 0

        price_analysis.append({
            'ticker': ticker,
            'current_price': current_price,
            'last_trade_date': last_date,
            'under_threshold': current_price < price_threshold,
            'anticascading_rank': rank,
            'composite_score': composite_score,
            'profit_factor': profit_factor,
            'sharpe_like_ratio': sharpe_like_ratio,
            'price_category': 'Under ₹500' if current_price < 500
                           else 'Under ₹1000' if current_price < 1000
                           else f'Under ₹{price_threshold}' if current_price < price_threshold
                           else f'Over ₹{price_threshold}'
        })

    price_df = pd.DataFrame(price_analysis)
    price_df = price_df.sort_values('current_price')

    # Filter for affordable tickers
    affordable_tickers = price_df[price_df['current_price'] < price_threshold]

    print(f"📊 PRICE ANALYSIS RESULTS:")
    print(f"   Total Top 50 tickers: {len(price_df)}")
    print(f"   Tickers under ₹{price_threshold:,.0f}: {len(affordable_tickers)} ({len(affordable_tickers)/len(price_df)*100:.1f}%)")

    print(f"\n📋 AFFORDABLE TICKERS:")
    print(f"   {'Ticker':12} | {'Price':8} | {'Rank':4} | {'PF':5} | {'Sharpe':6} | {'Category':15}")
    print("   " + "-" * 70)

    for _, row in affordable_tickers.iterrows():
        print(f"   {row['ticker']:12} | ₹{row['current_price']:6.2f} | {row['anticascading_rank']:4.0f} | {row['profit_factor']:5.2f} | {row['sharpe_like_ratio']:6.3f} | {row['price_category']}")

    return affordable_tickers


def apply_anti_cascading_filter(trades_top50, affordable_tickers):
    """
    Apply anti-cascading filter: exclude CONSECUTIVE_SAME_DIRECTION trades
    """

    print(f"\n🎯 APPLYING ANTI-CASCADING FILTER")
    print("=" * 60)

    # Filter trades for affordable tickers only
    ticker_list = affordable_tickers['ticker'].tolist()
    trades_filtered = trades_top50[trades_top50['ticker'].isin(ticker_list)].copy()

    print(f"📊 Working with {len(trades_filtered):,} trades from {len(ticker_list)} affordable tickers")

    # Sort by ticker and entry time for sequential analysis
    trades_sorted = trades_filtered.sort_values(['ticker', 'Entry Time']).reset_index(drop=True)
    trades_sorted['Entry Date'] = trades_sorted['Entry Time'].dt.date

    # Create previous trade reference for cascade detection
    trades_sorted['prev_ticker'] = trades_sorted['ticker'].shift(1)
    trades_sorted['prev_entry_date'] = trades_sorted['Entry Date'].shift(1)
    trades_sorted['prev_trade_type'] = trades_sorted['Trade Type'].shift(1)

    # Categorize each trade
    def categorize_trade(row):
        if pd.isna(row['prev_ticker']):
            return 'FIRST_TRADE_OVERALL'
        elif row['ticker'] != row['prev_ticker']:
            return 'FIRST_TRADE_FOR_TICKER'
        elif row['Entry Date'] != row['prev_entry_date']:
            return 'FIRST_TRADE_OF_DAY'
        elif row['Trade Type'] == row['prev_trade_type']:
            return 'CONSECUTIVE_SAME_DIRECTION'  # ❌ CASCADING - EXCLUDE
        else:
            return 'CONSECUTIVE_OPPOSITE_DIRECTION'  # ✅ INCLUDE

    print("🔍 Categorizing trades for cascade detection...")
    trades_sorted['trade_category'] = trades_sorted.apply(categorize_trade, axis=1)

    # Show categorization results
    category_counts = trades_sorted['trade_category'].value_counts()
    print(f"\n📊 TRADE CATEGORIZATION RESULTS:")
    for category, count in category_counts.items():
        percentage = (count / len(trades_sorted)) * 100
        status = "❌ EXCLUDE" if category == 'CONSECUTIVE_SAME_DIRECTION' else "✅ INCLUDE"
        print(f"   {category:30} | {count:8,} ({percentage:5.1f}%) | {status}")

    # Apply anti-cascading filter: Exclude ONLY consecutive same-direction trades
    anti_cascading_mask = trades_sorted['trade_category'] != 'CONSECUTIVE_SAME_DIRECTION'
    anti_cascading_trades = trades_sorted[anti_cascading_mask].copy()

    # Calculate reduction statistics
    original_count = len(trades_sorted)
    filtered_count = len(anti_cascading_trades)
    excluded_count = original_count - filtered_count
    reduction_percentage = (excluded_count / original_count) * 100

    print(f"\n🎯 ANTI-CASCADING FILTER RESULTS:")
    print(f"   Original trades: {original_count:,}")
    print(f"   Excluded (cascading): {excluded_count:,} ({reduction_percentage:.1f}%)")
    print(f"   Remaining (anti-cascading): {filtered_count:,} ({100-reduction_percentage:.1f}%)")

    return anti_cascading_trades


def save_filtered_dataset(config, anti_cascading_trades, affordable_tickers):
    """
    Save the filtered anti-cascading dataset and metadata
    """

    print(f"\n💾 SAVING FILTERED DATASET")
    print("=" * 50)

    # Get output directory from config
    output_dir = Path(get_output_dir(config, 'anti_cascade_filter', category='portfolio'))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save main filtered trades dataset
    output_file = output_dir / "anti_cascading_trades_filtered.csv"
    anti_cascading_trades.to_csv(output_file, index=False)

    # Save ticker metadata
    metadata_file = output_dir / "affordable_tickers_metadata.csv"
    affordable_tickers.to_csv(metadata_file, index=False)

    # Save summary report
    summary_file = output_dir / "anti_cascade_filter_summary.md"
    with open(summary_file, 'w') as f:
        f.write("# ANTI-CASCADING FILTER SUMMARY\n\n")
        f.write(f"**Filter Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## FILTERED DATASET STATISTICS\n\n")
        f.write(f"- Total trades after filter: {len(anti_cascading_trades):,}\n")
        f.write(f"- Number of affordable tickers: {len(affordable_tickers)}\n")
        f.write(f"- Date range: {anti_cascading_trades['Entry Time'].min()} to {anti_cascading_trades['Exit Time'].max()}\n\n")

        f.write("## AFFORDABLE TICKERS INCLUDED\n\n")
        f.write("| Ticker | Price | Rank | Profit Factor | Sharpe |\n")
        f.write("|--------|-------|------|---------------|--------|\n")
        for _, row in affordable_tickers.iterrows():
            f.write(f"| {row['ticker']} | ₹{row['current_price']:.2f} | {row['anticascading_rank']:.0f} | {row['profit_factor']:.2f} | {row['sharpe_like_ratio']:.3f} |\n")

    print(f"✅ Filtered trades saved: {output_file.name}")
    print(f"✅ Ticker metadata saved: {metadata_file.name}")
    print(f"✅ Summary report saved: {summary_file.name}")
    print(f"📁 Location: {output_dir}")

    # Final verification
    verification_df = pd.read_csv(output_file)
    print(f"\n🔍 VERIFICATION:")
    print(f"   File size: {len(verification_df):,} trades")
    print(f"   Memory usage: {verification_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    return output_file


def main():
    """
    Execute the anti-cascading subset creation process
    """

    # Parse arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    paths = resolve_paths(config)
    module_config = get_module_spec(config, 'anti_cascade_filter', category='portfolio')

    print("🚀 STARTING ANTI-CASCADING SUBSET CREATION")
    print("=" * 80)
    print(f"📁 Config: {args.config}")
    print(f"📊 Strategy: {config['run']['strategy']}")
    print(f"📅 Date Range: {config['run']['date_range']}")
    print("=" * 80)

    try:
        # Get price threshold from module config (default: 2000)
        price_threshold = module_config.get('config', {}).get('price_threshold', 2000)

        # Load Anti-Cascading Top 50 from foundation analysis
        correct_top50_tickers, anticascading_top50_df = load_anticascading_top50(config)

        # Load all trades
        trades_df = load_trades(config, paths)
        validation = validate_trade_data(trades_df)

        if not validation['valid']:
            print(f"❌ Data validation failed: {validation['errors']}")
            return None

        # Filter trades for Top 50 universe
        trades_top50 = filter_trades_for_top50(trades_df, correct_top50_tickers)

        # Identify affordable tickers from Top 50
        affordable_tickers = identify_affordable_tickers(trades_top50, anticascading_top50_df, price_threshold)

        # Apply anti-cascading filter
        anti_cascading_trades = apply_anti_cascading_filter(trades_top50, affordable_tickers)

        # Save filtered dataset
        output_file = save_filtered_dataset(config, anti_cascading_trades, affordable_tickers)

        print(f"\n🏆 ANTI-CASCADING SUBSET CREATION COMPLETED!")
        print(f"📊 Filtered dataset: {len(anti_cascading_trades):,} trades")
        print(f"🎯 Next step: Sector classification and correlation analysis")

        return {
            'output_file': str(output_file),
            'affordable_tickers': affordable_tickers,
            'anti_cascading_trades': anti_cascading_trades
        }

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = main()
