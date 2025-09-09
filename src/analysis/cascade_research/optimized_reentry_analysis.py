#!/usr/bin/env python3
"""
Optimized Re-Entry Cascade Analysis
===================================

High-performance version using vectorized operations for large datasets.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path):
    """Load and prepare the trades data for analysis."""
    print("Loading trades data...")
    
    # Load with optimized dtypes
    dtypes = {
        'Trade Type': 'category',
        'ticker': 'category',
        'Profit (Currency)': 'float32',
        'Profit (%)': 'float32'
    }
    
    df = pd.read_csv(file_path, dtype=dtypes)
    
    # Convert datetime columns
    df['Entry Time'] = pd.to_datetime(df['Entry Time'])
    df['Exit Time'] = pd.to_datetime(df['Exit Time'])
    
    # Sort by ticker and entry time
    df = df.sort_values(['ticker', 'Entry Time']).reset_index(drop=True)
    
    print(f"Loaded {len(df):,} trades covering {df['ticker'].nunique()} tickers")
    print(f"Date range: {df['Entry Time'].min().date()} to {df['Exit Time'].max().date()}")
    
    return df

def identify_sequential_trades_vectorized(df):
    """Vectorized identification of consecutive same-direction trades."""
    print("Identifying sequential same-direction trades...")
    
    # Create shift columns for comparison
    df['prev_ticker'] = df['ticker'].shift(1)
    df['prev_trade_type'] = df['Trade Type'].shift(1)
    df['prev_exit_time'] = df['Exit Time'].shift(1)
    
    # Identify re-entry trades (same ticker, same direction as previous)
    df['is_reentry'] = (
        (df['ticker'] == df['prev_ticker']) & 
        (df['Trade Type'] == df['prev_trade_type'])
    )
    
    # Group consecutive re-entries
    df['sequence_change'] = (
        (df['ticker'] != df['prev_ticker']) | 
        (df['Trade Type'] != df['prev_trade_type'])
    )
    
    df['sequence_id'] = df['sequence_change'].cumsum()
    
    # Identify sequences with 2+ trades
    sequence_counts = df.groupby('sequence_id').size()
    valid_sequences = sequence_counts[sequence_counts >= 2].index
    
    sequences_df = df[df['sequence_id'].isin(valid_sequences)].copy()
    
    print(f"Found {len(valid_sequences):,} sequences with 2+ same-direction trades")
    print(f"Total trades in sequences: {len(sequences_df):,}")
    
    return sequences_df

def analyze_performance_vectorized(sequences_df, df):
    """Vectorized performance analysis."""
    print("Analyzing sequence performance...")
    
    if len(sequences_df) == 0:
        print("No sequences found!")
        return None
    
    # Mark first trades in each sequence
    sequences_df['is_first_in_sequence'] = ~sequences_df['is_reentry'].shift(1, fill_value=False) | (
        sequences_df['sequence_id'] != sequences_df['sequence_id'].shift(1, fill_value=-1)
    )
    
    # Separate first trades and re-entries
    first_trades = sequences_df[sequences_df['is_first_in_sequence']]
    reentry_trades = sequences_df[~sequences_df['is_first_in_sequence']]
    
    # Calculate statistics
    results = {}
    
    # First trade stats
    first_profits = first_trades['Profit (Currency)'].values
    first_wins = (first_profits > 0).sum()
    first_accuracy = first_wins / len(first_profits) if len(first_profits) > 0 else 0
    
    # Re-entry stats  
    reentry_profits = reentry_trades['Profit (Currency)'].values
    reentry_wins = (reentry_profits > 0).sum()
    reentry_accuracy = reentry_wins / len(reentry_profits) if len(reentry_profits) > 0 else 0
    
    results = {
        'total_sequences': first_trades['sequence_id'].nunique(),
        'total_first_trades': len(first_trades),
        'total_reentry_trades': len(reentry_trades),
        'first_trade_pnl': first_profits.sum(),
        'reentry_trade_pnl': reentry_profits.sum(),
        'first_trade_avg_pnl': first_profits.mean() if len(first_profits) > 0 else 0,
        'reentry_trade_avg_pnl': reentry_profits.mean() if len(reentry_profits) > 0 else 0,
        'first_trade_accuracy': first_accuracy,
        'reentry_trade_accuracy': reentry_accuracy,
        'buy_sequences': len(first_trades[first_trades['Trade Type'] == 'Buy']),
        'sell_sequences': len(first_trades[first_trades['Trade Type'] == 'Sell']),
    }
    
    return results, sequences_df, first_trades, reentry_trades

def analyze_timing_patterns(sequences_df):
    """Analyze timing patterns in re-entries."""
    print("Analyzing timing patterns...")
    
    # Calculate time gaps for re-entries
    reentries = sequences_df[sequences_df['is_reentry']].copy()
    
    if len(reentries) == 0:
        return
    
    # Time gap from previous trade's exit to current entry
    reentries['time_gap_hours'] = (
        reentries['Entry Time'] - reentries['prev_exit_time']
    ).dt.total_seconds() / 3600
    
    time_gaps = reentries['time_gap_hours'].dropna()
    
    print(f"\nTiming Pattern Analysis:")
    print(f"  Average time gap between re-entries: {time_gaps.mean():.1f} hours")
    print(f"  Median time gap: {time_gaps.median():.1f} hours")
    print(f"  Re-entries within 1 hour: {(time_gaps < 1).sum():,} ({(time_gaps < 1).mean()*100:.1f}%)")
    print(f"  Re-entries within 24 hours: {(time_gaps < 24).sum():,} ({(time_gaps < 24).mean()*100:.1f}%)")
    print(f"  Re-entries within 1 week: {(time_gaps < 168).sum():,} ({(time_gaps < 168).mean()*100:.1f}%)")

def analyze_ticker_impact(sequences_df, reentry_trades):
    """Analyze which tickers are most affected."""
    print("Analyzing ticker impact...")
    
    # Group by ticker
    ticker_stats = reentry_trades.groupby('ticker').agg({
        'Profit (Currency)': ['count', 'sum', 'mean'],
        'sequence_id': 'nunique'
    }).round(2)
    
    ticker_stats.columns = ['reentry_count', 'total_pnl', 'avg_pnl', 'sequences']
    ticker_stats = ticker_stats.sort_values('total_pnl')
    
    print(f"\nTop 10 Tickers Most Negatively Affected by Re-entry Cascade:")
    worst_tickers = ticker_stats.head(10)
    for i, (ticker, row) in enumerate(worst_tickers.iterrows(), 1):
        print(f"  {i:2d}. {ticker}: {row['sequences']:.0f} sequences, "
              f"{row['reentry_count']:.0f} re-entries, "
              f"Total P&L: {row['total_pnl']:.2f}, "
              f"Avg P&L: {row['avg_pnl']:.2f}")

def simulate_filter_solution(df, sequences_df):
    """Simulate removing re-entry trades."""
    print("Simulating proposed filter solution...")
    
    # Current performance
    total_trades = len(df)
    current_pnl = df['Profit (Currency)'].sum()
    current_avg = current_pnl / total_trades
    current_accuracy = (df['Profit (Currency)'] > 0).mean()
    
    # Trades that would be removed (re-entries)
    reentry_indices = sequences_df[sequences_df['is_reentry']].index
    trades_to_remove = len(reentry_indices)
    removed_pnl = sequences_df[sequences_df['is_reentry']]['Profit (Currency)'].sum()
    
    # Remaining performance
    remaining_trades = total_trades - trades_to_remove
    remaining_pnl = current_pnl - removed_pnl
    remaining_avg = remaining_pnl / remaining_trades if remaining_trades > 0 else 0
    
    # Calculate remaining accuracy
    remaining_df = df.drop(reentry_indices)
    remaining_accuracy = (remaining_df['Profit (Currency)'] > 0).mean()
    
    print(f"\nFilter Solution Impact:")
    print(f"  Current Performance:")
    print(f"    Total trades: {total_trades:,}")
    print(f"    Total P&L: {current_pnl:.2f}")
    print(f"    Average P&L per trade: {current_avg:.2f}")
    print(f"    Win rate: {current_accuracy*100:.1f}%")
    
    print(f"\n  Trades to Remove (Re-entries):")
    print(f"    Count: {trades_to_remove:,} ({trades_to_remove/total_trades*100:.1f}% of all trades)")
    print(f"    Total P&L: {removed_pnl:.2f}")
    print(f"    Average P&L: {removed_pnl/trades_to_remove:.2f}")
    
    print(f"\n  After Filter Performance:")
    print(f"    Remaining trades: {remaining_trades:,}")
    print(f"    Total P&L: {remaining_pnl:.2f}")
    print(f"    Average P&L per trade: {remaining_avg:.2f}")
    print(f"    Win rate: {remaining_accuracy*100:.1f}%")
    
    print(f"\n  Performance Improvement:")
    print(f"    P&L per trade improvement: {remaining_avg - current_avg:.2f}")
    print(f"    Win rate improvement: {(remaining_accuracy - current_accuracy)*100:.1f} percentage points")
    print(f"    Total P&L improvement: {(remaining_avg - current_avg) * remaining_trades:.2f}")

def main():
    """Main analysis function."""
    file_path = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/back_tester/backtester/consolidated_trades.csv"
    
    print("="*70)
    print("RE-ENTRY CASCADE ANALYSIS")
    print("="*70)
    
    # Load data
    df = load_and_prepare_data(file_path)
    
    # Identify sequential trades
    sequences_df = identify_sequential_trades_vectorized(df)
    
    if len(sequences_df) == 0:
        print("No sequential same-direction trades found!")
        return
    
    # Analyze performance
    results, sequences_df, first_trades, reentry_trades = analyze_performance_vectorized(sequences_df, df)
    
    if results is None:
        return
    
    # Print main results
    print("\n" + "="*70)
    print("MAIN RESULTS")
    print("="*70)
    
    print(f"\n1. SEQUENTIAL TRADE DETECTION:")
    print(f"   Total sequences found: {results['total_sequences']:,}")
    print(f"   Buy sequences: {results['buy_sequences']:,}")
    print(f"   Sell sequences: {results['sell_sequences']:,}")
    print(f"   Total first trades: {results['total_first_trades']:,}")
    print(f"   Total re-entry trades: {results['total_reentry_trades']:,}")
    
    print(f"\n2. PERFORMANCE IMPACT:")
    print(f"   First Trade Performance:")
    print(f"     Total P&L: {results['first_trade_pnl']:.2f}")
    print(f"     Average P&L: {results['first_trade_avg_pnl']:.2f}")
    print(f"     Win Rate: {results['first_trade_accuracy']*100:.1f}%")
    
    print(f"   Re-entry Trade Performance:")
    print(f"     Total P&L: {results['reentry_trade_pnl']:.2f}")
    print(f"     Average P&L: {results['reentry_trade_avg_pnl']:.2f}")
    print(f"     Win Rate: {results['reentry_trade_accuracy']*100:.1f}%")
    
    print(f"   Performance Gap:")
    performance_gap = results['first_trade_avg_pnl'] - results['reentry_trade_avg_pnl']
    accuracy_gap = results['first_trade_accuracy'] - results['reentry_trade_accuracy']
    print(f"     P&L per trade difference: {performance_gap:.2f}")
    print(f"     Win rate difference: {accuracy_gap*100:.1f} percentage points")
    print(f"     Total profit lost to re-entries: {results['reentry_trade_pnl']:.2f}")
    
    # Additional analyses
    analyze_timing_patterns(sequences_df)
    analyze_ticker_impact(sequences_df, reentry_trades)
    simulate_filter_solution(df, sequences_df)
    
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    print(f"\n• Found {results['total_reentry_trades']:,} re-entry trades across {results['total_sequences']:,} sequences")
    print(f"• Re-entries show {results['reentry_trade_accuracy']*100:.1f}% accuracy vs {results['first_trade_accuracy']*100:.1f}% for first trades")
    print(f"• Each re-entry loses an average of {performance_gap:.2f} compared to first trades")
    print(f"• Total profit lost to re-entry cascade: {results['reentry_trade_pnl']:.2f}")
    print(f"• Filtering out re-entries would remove {results['total_reentry_trades']:,} trades ({results['total_reentry_trades']/len(df)*100:.1f}% of total)")
    print(f"• This would significantly improve overall strategy performance")

if __name__ == "__main__":
    main()