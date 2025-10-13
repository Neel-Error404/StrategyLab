"""
MAE/MFE Calculator Module
=========================

Purpose:
--------
Calculate Maximum Adverse Excursion (MAE) and Maximum Favorable Excursion (MFE)
for trades using full intra-trade bar data from base_data files.

Key Metrics:
------------
- MFE: Best price we could have exited at during trade
- MAE: Worst price we hit during trade
- MFE Capture Ratio: (Actual Profit / MFE) × 100
- Exit Efficiency Score: MFE_Capture_Ratio - (MAE/MFE × 50)
- Potential Left on Table: MFE - Actual Profit

Author: Strategy Optimization Pipeline
Date: 2025-10-04
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Import local trade_enhancer
from trade_enhancer import enhance_trades, get_trade_context_window

def calculate_mae_mfe(trade_data: pd.DataFrame,
                     base_data_dir: str,
                     progress_callback=None) -> pd.DataFrame:
    """
    Calculate MAE/MFE metrics for all trades.

    Parameters:
    -----------
    trade_data : pd.DataFrame
        Trade data with columns: Ticker, Trade Type, Entry Time, Entry Price,
                                Exit Time, Exit Price, percentage_return
    base_data_dir : str
        Path to base_data directory with 5min bar files
    progress_callback : callable, optional
        Function to call with progress updates (trade_idx, total_trades)

    Returns:
    --------
    pd.DataFrame
        Enhanced trade data with MAE/MFE columns added
    """

    print(f"\n🔬 Calculating MAE/MFE for {len(trade_data):,} trades...")

    # Step 1: Enhance trades with base_data context
    print("   Step 1: Enhancing trades with base_data...")
    enhanced = enhance_trades(trade_data.copy(), base_data_dir, cache_base_data=True)
    print(f"   ✓ Enhanced {len(enhanced):,} trades")

    # Initialize MAE/MFE columns
    enhanced['MFE_pct'] = np.nan
    enhanced['MAE_pct'] = np.nan
    enhanced['MFE_Capture_Ratio'] = np.nan
    enhanced['Exit_Efficiency_Score'] = np.nan
    enhanced['Potential_Left_on_Table_pct'] = np.nan
    enhanced['MAE_MFE_Ratio'] = np.nan

    # Step 2: Calculate MAE/MFE for each trade
    print("   Step 2: Calculating MAE/MFE for each trade...")

    successful_calcs = 0
    failed_calcs = 0

    for idx in range(len(enhanced)):
        try:
            trade = enhanced.iloc[idx]

            # Get intra-trade bars (bars during the trade only)
            context = get_trade_context_window(
                enhanced_data=enhanced,
                trade_idx=idx,
                base_data_dir=base_data_dir,
                context_intervals=0  # Only bars during trade
            )

            # Filter to bars during trade
            trade_bars = context[context['trade_phase'] == 'during'].copy()

            if len(trade_bars) == 0:
                # No intra-trade data available
                failed_calcs += 1
                continue

            # Calculate MFE and MAE based on trade type
            entry_price = trade['Entry Price']
            exit_price = trade['Exit Price']
            trade_type = trade['Trade Type']

            if trade_type == 'Buy':
                # For Buy trades:
                # MFE = highest price during trade (best exit opportunity)
                # MAE = lowest price during trade (worst drawdown)
                best_price = trade_bars['high'].max()
                worst_price = trade_bars['low'].min()

                MFE_pct = ((best_price - entry_price) / entry_price) * 100
                MAE_pct = ((entry_price - worst_price) / entry_price) * 100

            else:  # Sell trades
                # For Sell trades:
                # MFE = lowest price during trade (best exit opportunity)
                # MAE = highest price during trade (worst drawdown)
                best_price = trade_bars['low'].min()
                worst_price = trade_bars['high'].max()

                MFE_pct = ((entry_price - best_price) / entry_price) * 100
                MAE_pct = ((worst_price - entry_price) / entry_price) * 100

            # Actual profit from trade
            actual_profit_pct = trade['percentage_return']

            # Calculate derived metrics
            if MFE_pct > 0:
                MFE_Capture_Ratio = (actual_profit_pct / MFE_pct) * 100
                MAE_MFE_Ratio = MAE_pct / MFE_pct if MFE_pct > 0 else 0

                # Exit Efficiency Score = MFE_Capture_Ratio - (MAE/MFE × 50)
                # Penalizes high drawdown relative to profit potential
                Exit_Efficiency_Score = MFE_Capture_Ratio - (MAE_MFE_Ratio * 50)

                # Potential left on table
                Potential_Left = MFE_pct - actual_profit_pct
            else:
                # If MFE is 0 or negative (trade never went profitable)
                MFE_Capture_Ratio = 0
                MAE_MFE_Ratio = 0
                Exit_Efficiency_Score = -100  # Terrible
                Potential_Left = 0

            # Update the DataFrame
            enhanced.loc[enhanced.index[idx], 'MFE_pct'] = MFE_pct
            enhanced.loc[enhanced.index[idx], 'MAE_pct'] = MAE_pct
            enhanced.loc[enhanced.index[idx], 'MFE_Capture_Ratio'] = MFE_Capture_Ratio
            enhanced.loc[enhanced.index[idx], 'Exit_Efficiency_Score'] = Exit_Efficiency_Score
            enhanced.loc[enhanced.index[idx], 'Potential_Left_on_Table_pct'] = Potential_Left
            enhanced.loc[enhanced.index[idx], 'MAE_MFE_Ratio'] = MAE_MFE_Ratio

            successful_calcs += 1

            # Progress callback
            if progress_callback and (idx + 1) % 100 == 0:
                progress_callback(idx + 1, len(enhanced))

        except Exception as e:
            # Silently continue on error (trade will have NaN values)
            failed_calcs += 1
            continue

    print(f"   ✓ Successfully calculated MAE/MFE for {successful_calcs:,} trades")
    if failed_calcs > 0:
        print(f"   ⚠ Failed to calculate for {failed_calcs:,} trades (no intra-trade data)")

    return enhanced


def get_mae_mfe_summary(enhanced_data: pd.DataFrame) -> dict:
    """
    Calculate summary statistics for MAE/MFE metrics.

    Parameters:
    -----------
    enhanced_data : pd.DataFrame
        Trade data with MAE/MFE columns

    Returns:
    --------
    dict
        Summary statistics
    """

    # Filter to trades with valid MAE/MFE data
    valid_trades = enhanced_data.dropna(subset=['MFE_pct', 'MAE_pct'])

    if len(valid_trades) == 0:
        return {
            'error': 'No valid MAE/MFE data available',
            'total_trades': len(enhanced_data),
            'valid_trades': 0
        }

    summary = {
        'total_trades': len(enhanced_data),
        'valid_trades': len(valid_trades),
        'coverage_pct': (len(valid_trades) / len(enhanced_data)) * 100,

        # MFE statistics
        'avg_MFE_pct': valid_trades['MFE_pct'].mean(),
        'median_MFE_pct': valid_trades['MFE_pct'].median(),
        'std_MFE_pct': valid_trades['MFE_pct'].std(),

        # MAE statistics
        'avg_MAE_pct': valid_trades['MAE_pct'].mean(),
        'median_MAE_pct': valid_trades['MAE_pct'].median(),
        'std_MAE_pct': valid_trades['MAE_pct'].std(),

        # Capture ratio statistics
        'avg_MFE_Capture_Ratio': valid_trades['MFE_Capture_Ratio'].mean(),
        'median_MFE_Capture_Ratio': valid_trades['MFE_Capture_Ratio'].median(),

        # Exit efficiency statistics
        'avg_Exit_Efficiency_Score': valid_trades['Exit_Efficiency_Score'].mean(),
        'median_Exit_Efficiency_Score': valid_trades['Exit_Efficiency_Score'].median(),

        # Potential left on table
        'avg_Potential_Left_on_Table_pct': valid_trades['Potential_Left_on_Table_pct'].mean(),
        'total_Potential_Left_pct': valid_trades['Potential_Left_on_Table_pct'].sum(),

        # MAE/MFE ratio
        'avg_MAE_MFE_Ratio': valid_trades['MAE_MFE_Ratio'].mean(),
        'median_MAE_MFE_Ratio': valid_trades['MAE_MFE_Ratio'].median(),

        # Distribution
        'efficiency_score_excellent': (valid_trades['Exit_Efficiency_Score'] > 70).sum(),
        'efficiency_score_good': ((valid_trades['Exit_Efficiency_Score'] >= 50) &
                                  (valid_trades['Exit_Efficiency_Score'] <= 70)).sum(),
        'efficiency_score_poor': ((valid_trades['Exit_Efficiency_Score'] >= 30) &
                                 (valid_trades['Exit_Efficiency_Score'] < 50)).sum(),
        'efficiency_score_terrible': (valid_trades['Exit_Efficiency_Score'] < 30).sum(),
    }

    return summary


def print_mae_mfe_summary(summary: dict):
    """Pretty print MAE/MFE summary statistics"""

    if 'error' in summary:
        print(f"\n❌ {summary['error']}")
        return

    print("\n" + "="*70)
    print("MAE/MFE SUMMARY STATISTICS")
    print("="*70)

    print(f"\nData Coverage:")
    print(f"   Total Trades: {summary['total_trades']:,}")
    print(f"   Valid MAE/MFE Data: {summary['valid_trades']:,} ({summary['coverage_pct']:.1f}%)")

    print(f"\nMaximum Favorable Excursion (MFE) - Best Price Available:")
    print(f"   Average: {summary['avg_MFE_pct']:.2f}%")
    print(f"   Median: {summary['median_MFE_pct']:.2f}%")
    print(f"   Std Dev: {summary['std_MFE_pct']:.2f}%")

    print(f"\nMaximum Adverse Excursion (MAE) - Worst Drawdown:")
    print(f"   Average: {summary['avg_MAE_pct']:.2f}%")
    print(f"   Median: {summary['median_MAE_pct']:.2f}%")
    print(f"   Std Dev: {summary['std_MAE_pct']:.2f}%")

    print(f"\nMFE Capture Ratio - % of Available Profit Captured:")
    print(f"   Average: {summary['avg_MFE_Capture_Ratio']:.2f}%")
    print(f"   Median: {summary['median_MFE_Capture_Ratio']:.2f}%")

    print(f"\nExit Efficiency Score - Overall Exit Quality:")
    print(f"   Average: {summary['avg_Exit_Efficiency_Score']:.2f}")
    print(f"   Median: {summary['median_Exit_Efficiency_Score']:.2f}")

    print(f"\nExit Efficiency Distribution:")
    print(f"   Excellent (>70): {summary['efficiency_score_excellent']:,} trades")
    print(f"   Good (50-70): {summary['efficiency_score_good']:,} trades")
    print(f"   Poor (30-50): {summary['efficiency_score_poor']:,} trades")
    print(f"   Terrible (<30): {summary['efficiency_score_terrible']:,} trades")

    print(f"\nPotential Left on Table:")
    print(f"   Average per Trade: {summary['avg_Potential_Left_on_Table_pct']:.2f}%")
    print(f"   Total Across All Trades: {summary['total_Potential_Left_pct']:.2f}%")

    print(f"\nMAE/MFE Ratio - Drawdown vs Profit Potential:")
    print(f"   Average: {summary['avg_MAE_MFE_Ratio']:.2f}")
    print(f"   Median: {summary['median_MAE_MFE_Ratio']:.2f}")

    print("="*70)
