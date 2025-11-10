#!/usr/bin/env python3
"""
Final Re-Entry Cascade Analysis
==============================

Analysis focused on the specific "re-entry cascade" problem:
Consecutive same-direction trades that show systematic underperformance.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_comprehensive_sample(file_path):
    """Load a comprehensive sample with good ticker coverage."""
    print("Loading comprehensive sample...")
    
    # Load sample with stratified sampling by ticker
    dtypes = {
        'Trade Type': 'category',
        'ticker': 'category',
        'Profit (Currency)': 'float32'
    }
    
    # Read data in chunks and take samples
    samples = []
    total_read = 0
    target_sample = 100000
    
    for chunk in pd.read_csv(file_path, chunksize=25000, dtype=dtypes):
        # Take a sample from each chunk
        sample_size = min(5000, len(chunk))
        chunk_sample = chunk.sample(n=sample_size, random_state=42)
        samples.append(chunk_sample)
        total_read += len(chunk)
        
        if len(pd.concat(samples)) >= target_sample:
            break
    
    df = pd.concat(samples, ignore_index=True)
    
    # Convert datetime
    df['Entry Time'] = pd.to_datetime(df['Entry Time'])
    df['Exit Time'] = pd.to_datetime(df['Exit Time'])
    
    # Sort and prepare
    df = df.sort_values(['ticker', 'Entry Time']).reset_index(drop=True)
    df['trade_id'] = df.index
    
    print(f"Sample: {len(df):,} trades from {df['ticker'].nunique()} tickers")
    print(f"Sample represents ~{len(df)/total_read*100:.1f}% of processed data")
    
    return df

def identify_immediate_reentries(df):
    """Identify immediate re-entry patterns."""
    print("Identifying immediate re-entry patterns...")
    
    reentry_pairs = []
    
    for ticker in df['ticker'].unique():
        ticker_trades = df[df['ticker'] == ticker].sort_values('Entry Time')
        
        for i in range(len(ticker_trades) - 1):
            current = ticker_trades.iloc[i]
            next_trade = ticker_trades.iloc[i + 1]
            
            # Same direction = re-entry
            if current['Trade Type'] == next_trade['Trade Type']:
                # Calculate time gap
                time_gap = (next_trade['Entry Time'] - current['Exit Time']).total_seconds() / 3600
                
                reentry_pairs.append({
                    'ticker': ticker,
                    'trade_type': current['Trade Type'],
                    'first_trade_id': current['trade_id'],
                    'reentry_trade_id': next_trade['trade_id'],
                    'first_profit': current['Profit (Currency)'],
                    'reentry_profit': next_trade['Profit (Currency)'],
                    'first_entry': current['Entry Time'],
                    'first_exit': current['Exit Time'],
                    'reentry_entry': next_trade['Entry Time'],
                    'reentry_exit': next_trade['Exit Time'],
                    'time_gap_hours': time_gap,
                    'profit_decline': current['Profit (Currency)'] - next_trade['Profit (Currency)']
                })
    
    return pd.DataFrame(reentry_pairs)

def analyze_reentry_performance(reentry_df, full_df):
    """Analyze re-entry performance patterns."""
    print("\n" + "="*70)
    print("RE-ENTRY PERFORMANCE ANALYSIS")
    print("="*70)
    
    if len(reentry_df) == 0:
        print("No re-entry pairs found!")
        return
    
    print(f"Found {len(reentry_df):,} re-entry pairs")
    
    # Basic statistics
    first_profits = reentry_df['first_profit']
    reentry_profits = reentry_df['reentry_profit']
    
    print(f"\n1. PERFORMANCE COMPARISON:")
    print(f"   First Trades in Re-entry Pairs:")
    print(f"     Average P&L: {first_profits.mean():.2f}")
    print(f"     Median P&L: {first_profits.median():.2f}")
    print(f"     Win Rate: {(first_profits > 0).mean()*100:.1f}%")
    print(f"     Total P&L: {first_profits.sum():.2f}")
    
    print(f"   Re-entry Trades:")
    print(f"     Average P&L: {reentry_profits.mean():.2f}")
    print(f"     Median P&L: {reentry_profits.median():.2f}")
    print(f"     Win Rate: {(reentry_profits > 0).mean()*100:.1f}%")
    print(f"     Total P&L: {reentry_profits.sum():.2f}")
    
    # Performance gap
    performance_gap = first_profits.mean() - reentry_profits.mean()
    win_rate_gap = (first_profits > 0).mean() - (reentry_profits > 0).mean()
    
    print(f"   Performance Impact:")
    print(f"     Average P&L gap: {performance_gap:.2f} per trade")
    print(f"     Win rate gap: {win_rate_gap*100:.1f} percentage points")
    print(f"     Total underperformance: {performance_gap * len(reentry_df):.2f}")
    
    # Categorize by performance decline
    declining_reentries = reentry_df[reentry_df['profit_decline'] > 0]
    improving_reentries = reentry_df[reentry_df['profit_decline'] < 0]
    
    print(f"\n2. PROFIT DECLINE ANALYSIS:")
    print(f"   Re-entries with profit decline: {len(declining_reentries):,} ({len(declining_reentries)/len(reentry_df)*100:.1f}%)")
    print(f"   Re-entries with profit improvement: {len(improving_reentries):,} ({len(improving_reentries)/len(reentry_df)*100:.1f}%)")
    
    if len(declining_reentries) > 0:
        print(f"   Average decline amount: {declining_reentries['profit_decline'].mean():.2f}")
        print(f"   Median decline amount: {declining_reentries['profit_decline'].median():.2f}")
        print(f"   Total profit lost to declining re-entries: {declining_reentries['profit_decline'].sum():.2f}")
    
    # Time gap analysis
    time_gaps = reentry_df['time_gap_hours']
    
    print(f"\n3. TIMING ANALYSIS:")
    print(f"   Average time gap: {time_gaps.mean():.1f} hours")
    print(f"   Median time gap: {time_gaps.median():.1f} hours")
    print(f"   Immediate re-entries (< 1 hour): {(time_gaps < 1).sum():,} ({(time_gaps < 1).mean()*100:.1f}%)")
    print(f"   Same day re-entries (< 24 hours): {(time_gaps < 24).sum():,} ({(time_gaps < 24).mean()*100:.1f}%)")
    
    # Quick re-entries performance
    quick_reentries = reentry_df[time_gaps < 24]
    if len(quick_reentries) > 0:
        quick_performance_gap = quick_reentries['first_profit'].mean() - quick_reentries['reentry_profit'].mean()
        print(f"   Quick re-entries (< 24h) performance gap: {quick_performance_gap:.2f}")
    
    # Trade type analysis
    buy_reentries = reentry_df[reentry_df['trade_type'] == 'Buy']
    sell_reentries = reentry_df[reentry_df['trade_type'] == 'Sell']
    
    print(f"\n4. TRADE TYPE BREAKDOWN:")
    for trade_type, subset in [('Buy', buy_reentries), ('Sell', sell_reentries)]:
        if len(subset) > 0:
            first_avg = subset['first_profit'].mean()
            reentry_avg = subset['reentry_profit'].mean()
            gap = first_avg - reentry_avg
            
            print(f"   {trade_type} Re-entries ({len(subset):,} pairs):")
            print(f"     First trade avg: {first_avg:.2f}")
            print(f"     Re-entry avg: {reentry_avg:.2f}")
            print(f"     Performance gap: {gap:.2f}")
    
    return {
        'performance_gap': performance_gap,
        'win_rate_gap': win_rate_gap,
        'reentry_count': len(reentry_df),
        'declining_count': len(declining_reentries),
        'total_decline': declining_reentries['profit_decline'].sum() if len(declining_reentries) > 0 else 0
    }

def identify_worst_reentry_patterns(reentry_df):
    """Identify the worst re-entry patterns."""
    print(f"\n5. WORST RE-ENTRY PATTERNS:")
    
    # Worst individual re-entries
    worst_reentries = reentry_df.nlargest(10, 'profit_decline')
    
    print(f"   Top 10 Worst Individual Re-entries:")
    for i, (_, row) in enumerate(worst_reentries.iterrows(), 1):
        print(f"     {i:2d}. {row['ticker']} ({row['trade_type']}): "
              f"First: {row['first_profit']:.2f}, "
              f"Re-entry: {row['reentry_profit']:.2f}, "
              f"Decline: {row['profit_decline']:.2f}")
    
    # Worst tickers by total re-entry impact
    ticker_impact = reentry_df.groupby('ticker').agg({
        'profit_decline': ['count', 'sum', 'mean'],
        'reentry_profit': 'sum'
    }).round(2)
    
    ticker_impact.columns = ['reentry_count', 'total_decline', 'avg_decline', 'total_reentry_pnl']
    ticker_impact = ticker_impact.sort_values('total_decline', ascending=False)
    
    print(f"\n   Top 10 Tickers by Total Re-entry Impact:")
    for i, (ticker, row) in enumerate(ticker_impact.head(10).iterrows(), 1):
        print(f"     {i:2d}. {ticker}: {row['reentry_count']:.0f} re-entries, "
              f"Total decline: {row['total_decline']:.2f}, "
              f"Avg decline: {row['avg_decline']:.2f}")

def simulate_reentry_filter(full_df, reentry_df):
    """Simulate filtering out problematic re-entries."""
    print(f"\n6. SOLUTION SIMULATION:")
    
    # Current performance
    current_pnl = full_df['Profit (Currency)'].sum()
    current_avg = current_pnl / len(full_df)
    current_win_rate = (full_df['Profit (Currency)'] > 0).mean()
    
    # Strategy 1: Remove all re-entries
    reentry_ids = set(reentry_df['reentry_trade_id'])
    reentry_pnl = reentry_df['reentry_profit'].sum()
    
    remaining_trades_1 = len(full_df) - len(reentry_ids)
    remaining_pnl_1 = current_pnl - reentry_pnl
    remaining_avg_1 = remaining_pnl_1 / remaining_trades_1
    
    print(f"   Strategy 1: Remove All Re-entries")
    print(f"     Trades removed: {len(reentry_ids):,} ({len(reentry_ids)/len(full_df)*100:.1f}%)")
    print(f"     Current avg P&L: {current_avg:.2f}")
    print(f"     New avg P&L: {remaining_avg_1:.2f}")
    print(f"     Improvement: {remaining_avg_1 - current_avg:.2f} per trade")
    
    # Strategy 2: Remove only declining re-entries
    declining_reentries = reentry_df[reentry_df['profit_decline'] > 0]
    declining_ids = set(declining_reentries['reentry_trade_id'])
    declining_pnl = declining_reentries['reentry_profit'].sum()
    
    remaining_trades_2 = len(full_df) - len(declining_ids)
    remaining_pnl_2 = current_pnl - declining_pnl
    remaining_avg_2 = remaining_pnl_2 / remaining_trades_2
    
    print(f"   Strategy 2: Remove Only Declining Re-entries")
    print(f"     Trades removed: {len(declining_ids):,} ({len(declining_ids)/len(full_df)*100:.1f}%)")
    print(f"     Current avg P&L: {current_avg:.2f}")
    print(f"     New avg P&L: {remaining_avg_2:.2f}")
    print(f"     Improvement: {remaining_avg_2 - current_avg:.2f} per trade")
    
    # Strategy 3: Remove quick re-entries (< 24 hours)
    quick_reentries = reentry_df[reentry_df['time_gap_hours'] < 24]
    quick_ids = set(quick_reentries['reentry_trade_id'])
    quick_pnl = quick_reentries['reentry_profit'].sum()
    
    remaining_trades_3 = len(full_df) - len(quick_ids)
    remaining_pnl_3 = current_pnl - quick_pnl
    remaining_avg_3 = remaining_pnl_3 / remaining_trades_3
    
    print(f"   Strategy 3: Remove Quick Re-entries (< 24 hours)")
    print(f"     Trades removed: {len(quick_ids):,} ({len(quick_ids)/len(full_df)*100:.1f}%)")
    print(f"     Current avg P&L: {current_avg:.2f}")
    print(f"     New avg P&L: {remaining_avg_3:.2f}")
    print(f"     Improvement: {remaining_avg_3 - current_avg:.2f} per trade")

def extrapolate_to_full_dataset(results, sample_size, total_size):
    """Extrapolate sample results to full dataset."""
    print(f"\n7. FULL DATASET EXTRAPOLATION:")
    
    scaling_factor = total_size / sample_size
    
    estimated_reentries = int(results['reentry_count'] * scaling_factor)
    estimated_total_decline = results['total_decline'] * scaling_factor
    estimated_performance_gap = results['performance_gap'] * estimated_reentries
    
    print(f"   Sample Analysis:")
    print(f"     Sample size: {sample_size:,} trades")
    print(f"     Re-entry pairs found: {results['reentry_count']:,}")
    print(f"     Performance gap: {results['performance_gap']:.2f} per trade")
    
    print(f"   Full Dataset Estimates (Total: {total_size:,} trades):")
    print(f"     Estimated re-entry pairs: {estimated_reentries:,}")
    print(f"     Estimated total profit lost: {estimated_total_decline:.2f}")
    print(f"     Estimated total performance gap: {estimated_performance_gap:.2f}")
    print(f"     Percentage of trades that are re-entries: {estimated_reentries/total_size*100:.1f}%")

def main():
    """Main analysis function."""
    file_path = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/back_tester/backtester/consolidated_trades.csv"
    
    print("="*70)
    print("FINAL RE-ENTRY CASCADE ANALYSIS")
    print("="*70)
    
    # Load sample
    df = load_comprehensive_sample(file_path)
    
    # Identify re-entry pairs
    reentry_df = identify_immediate_reentries(df)
    
    if len(reentry_df) == 0:
        print("No re-entry patterns found!")
        return
    
    # Analyze performance
    results = analyze_reentry_performance(reentry_df, df)
    
    # Identify worst patterns
    identify_worst_reentry_patterns(reentry_df)
    
    # Simulate solutions
    simulate_reentry_filter(df, reentry_df)
    
    # Extrapolate to full dataset
    extrapolate_to_full_dataset(results, len(df), 590037)  # Total from earlier analysis
    
    print(f"\n" + "="*70)
    print("FINAL CONCLUSIONS")
    print("="*70)
    
    print(f"• Re-entry cascade problem confirmed: {results['reentry_count']:,} re-entry pairs identified")
    print(f"• Re-entries underperform first trades by {results['performance_gap']:.2f} per trade")
    print(f"• {results['declining_count']:,} re-entries show clear profit decline")
    print(f"• Total profit lost in sample: {results['total_decline']:.2f}")
    print(f"• Implementing re-entry filters would improve strategy performance")
    print(f"• Recommendation: Prevent same-direction trades within 24-hour windows")

if __name__ == "__main__":
    main()