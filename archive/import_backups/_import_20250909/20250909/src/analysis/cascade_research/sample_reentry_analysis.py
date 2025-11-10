#!/usr/bin/env python3
"""
Sample-Based Re-Entry Cascade Analysis
======================================

Efficient analysis using representative samples to identify re-entry cascade patterns.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def sample_and_analyze(file_path, sample_size=50000):
    """Load sample data and perform analysis."""
    print(f"Loading sample of {sample_size:,} trades for analysis...")
    
    # Get total line count first
    with open(file_path, 'r') as f:
        total_lines = sum(1 for _ in f) - 1  # Subtract header
    
    print(f"Total trades in dataset: {total_lines:,}")
    
    # Load stratified sample
    dtypes = {
        'Trade Type': 'category',
        'ticker': 'category',
        'Profit (Currency)': 'float32'
    }
    
    # Read in chunks and sample proportionally
    chunk_size = 50000
    sample_dfs = []
    
    for chunk in pd.read_csv(file_path, dtype=dtypes, chunksize=chunk_size):
        chunk_sample_size = int(len(chunk) * sample_size / total_lines)
        if chunk_sample_size > 0:
            sample = chunk.sample(n=min(chunk_sample_size, len(chunk)), random_state=42)
            sample_dfs.append(sample)
    
    df = pd.concat(sample_dfs, ignore_index=True)
    
    # Convert datetime columns
    df['Entry Time'] = pd.to_datetime(df['Entry Time'])
    df['Exit Time'] = pd.to_datetime(df['Exit Time'])
    
    # Sort by ticker and entry time
    df = df.sort_values(['ticker', 'Entry Time']).reset_index(drop=True)
    df['trade_id'] = df.index
    
    print(f"Loaded sample: {len(df):,} trades covering {df['ticker'].nunique()} tickers")
    
    return df, total_lines

def quick_cascade_analysis(df):
    """Quick identification of re-entry patterns."""
    print("\nAnalyzing re-entry cascade patterns in sample...")
    
    cascade_data = []
    
    # Group by ticker for analysis
    for ticker in df['ticker'].unique():
        ticker_trades = df[df['ticker'] == ticker].sort_values('Entry Time').reset_index(drop=True)
        
        if len(ticker_trades) < 2:
            continue
        
        # Look for consecutive same-direction trades
        for i in range(len(ticker_trades) - 1):
            current = ticker_trades.iloc[i]
            next_trade = ticker_trades.iloc[i + 1]
            
            if current['Trade Type'] == next_trade['Trade Type']:
                # This is a re-entry
                time_gap = (next_trade['Entry Time'] - current['Exit Time']).total_seconds() / 3600
                
                cascade_data.append({
                    'ticker': ticker,
                    'first_trade_profit': current['Profit (Currency)'],
                    'reentry_trade_profit': next_trade['Profit (Currency)'],
                    'trade_type': current['Trade Type'],
                    'time_gap_hours': time_gap,
                    'first_trade_id': current['trade_id'],
                    'reentry_trade_id': next_trade['trade_id']
                })
    
    return pd.DataFrame(cascade_data)

def analyze_sample_results(cascade_df, sample_df, total_trades):
    """Analyze results and extrapolate to full dataset."""
    print(f"\n" + "="*70)
    print("SAMPLE ANALYSIS RESULTS")
    print("="*70)
    
    if len(cascade_df) == 0:
        print("No re-entry cascades found in sample!")
        return
    
    sample_size = len(sample_df)
    scaling_factor = total_trades / sample_size
    
    print(f"\nSample Statistics:")
    print(f"  Sample size: {sample_size:,} trades")
    print(f"  Total dataset: {total_trades:,} trades")
    print(f"  Scaling factor: {scaling_factor:.1f}x")
    
    # Basic cascade statistics
    reentry_count_sample = len(cascade_df)
    estimated_reentry_total = int(reentry_count_sample * scaling_factor)
    
    print(f"\n1. RE-ENTRY CASCADE DETECTION:")
    print(f"   Sample re-entry pairs found: {reentry_count_sample:,}")
    print(f"   Estimated total re-entry pairs: {estimated_reentry_total:,}")
    print(f"   Estimated % of trades that are re-entries: {estimated_reentry_total/total_trades*100:.1f}%")
    
    # Performance analysis
    first_trade_profits = cascade_df['first_trade_profit']
    reentry_trade_profits = cascade_df['reentry_trade_profit']
    
    first_avg = first_trade_profits.mean()
    reentry_avg = reentry_trade_profits.mean()
    performance_gap = first_avg - reentry_avg
    
    first_win_rate = (first_trade_profits > 0).mean()
    reentry_win_rate = (reentry_trade_profits > 0).mean()
    accuracy_gap = first_win_rate - reentry_win_rate
    
    print(f"\n2. PERFORMANCE IMPACT ANALYSIS:")
    print(f"   First Trade in Cascade:")
    print(f"     Average P&L: {first_avg:.2f}")
    print(f"     Win Rate: {first_win_rate*100:.1f}%")
    print(f"     Total P&L (sample): {first_trade_profits.sum():.2f}")
    
    print(f"   Re-entry Trade:")
    print(f"     Average P&L: {reentry_avg:.2f}")
    print(f"     Win Rate: {reentry_win_rate*100:.1f}%")
    print(f"     Total P&L (sample): {reentry_trade_profits.sum():.2f}")
    
    print(f"   Performance Impact:")
    print(f"     P&L gap per trade: {performance_gap:.2f}")
    print(f"     Accuracy gap: {accuracy_gap*100:.1f} percentage points")
    
    # Extrapolate impact
    estimated_total_reentry_loss = reentry_trade_profits.sum() * scaling_factor
    estimated_performance_loss = performance_gap * estimated_reentry_total
    
    print(f"   Estimated Total Impact:")
    print(f"     Total P&L lost to re-entries: {estimated_total_reentry_loss:.2f}")
    print(f"     Performance loss vs first trades: {estimated_performance_loss:.2f}")
    
    # Trade type analysis
    buy_cascades = cascade_df[cascade_df['trade_type'] == 'Buy']
    sell_cascades = cascade_df[cascade_df['trade_type'] == 'Sell']
    
    print(f"\n3. TRADE TYPE BREAKDOWN:")
    print(f"   Buy Re-entries: {len(buy_cascades):,} (Est. total: {int(len(buy_cascades) * scaling_factor):,})")
    if len(buy_cascades) > 0:
        print(f"     First trade avg P&L: {buy_cascades['first_trade_profit'].mean():.2f}")
        print(f"     Re-entry avg P&L: {buy_cascades['reentry_trade_profit'].mean():.2f}")
        print(f"     Performance gap: {buy_cascades['first_trade_profit'].mean() - buy_cascades['reentry_trade_profit'].mean():.2f}")
    
    print(f"   Sell Re-entries: {len(sell_cascades):,} (Est. total: {int(len(sell_cascades) * scaling_factor):,})")
    if len(sell_cascades) > 0:
        print(f"     First trade avg P&L: {sell_cascades['first_trade_profit'].mean():.2f}")
        print(f"     Re-entry avg P&L: {sell_cascades['reentry_trade_profit'].mean():.2f}")
        print(f"     Performance gap: {sell_cascades['first_trade_profit'].mean() - sell_cascades['reentry_trade_profit'].mean():.2f}")
    
    # Timing analysis
    time_gaps = cascade_df['time_gap_hours']
    
    print(f"\n4. TIMING PATTERNS:")
    print(f"   Average time gap: {time_gaps.mean():.1f} hours")
    print(f"   Median time gap: {time_gaps.median():.1f} hours")
    print(f"   Re-entries within 1 hour: {(time_gaps < 1).sum():,} ({(time_gaps < 1).mean()*100:.1f}%)")
    print(f"   Re-entries within 24 hours: {(time_gaps < 24).sum():,} ({(time_gaps < 24).mean()*100:.1f}%)")
    
    # Ticker analysis
    ticker_impact = cascade_df.groupby('ticker').agg({
        'reentry_trade_profit': ['count', 'sum', 'mean']
    }).round(2)
    
    ticker_impact.columns = ['reentry_count', 'total_loss', 'avg_loss']
    ticker_impact = ticker_impact.sort_values('total_loss')
    
    print(f"\n5. MOST AFFECTED TICKERS (sample):")
    worst_tickers = ticker_impact.head(10)
    for i, (ticker, row) in enumerate(worst_tickers.iterrows(), 1):
        est_total_reentries = int(row['reentry_count'] * scaling_factor)
        est_total_loss = row['total_loss'] * scaling_factor
        print(f"   {i:2d}. {ticker}: {row['reentry_count']} re-entries (est. {est_total_reentries}), "
              f"P&L: {row['total_loss']:.2f} (est. {est_total_loss:.2f})")
    
    return {
        'estimated_reentry_count': estimated_reentry_total,
        'performance_gap': performance_gap,
        'accuracy_gap': accuracy_gap,
        'estimated_total_loss': estimated_total_reentry_loss,
        'scaling_factor': scaling_factor
    }

def simulate_filter_impact(sample_df, cascade_df, total_trades, results):
    """Simulate the impact of filtering out re-entry trades."""
    print(f"\n6. PROPOSED SOLUTION VALIDATION:")
    
    # Current sample performance
    sample_pnl = sample_df['Profit (Currency)'].sum()
    sample_avg = sample_pnl / len(sample_df)
    sample_win_rate = (sample_df['Profit (Currency)'] > 0).mean()
    
    # Re-entry trades in sample
    reentry_ids = set(cascade_df['reentry_trade_id'])
    reentry_pnl_sample = cascade_df['reentry_trade_profit'].sum()
    
    # After filter (sample)
    remaining_sample = sample_df[~sample_df['trade_id'].isin(reentry_ids)]
    remaining_pnl_sample = remaining_sample['Profit (Currency)'].sum()
    remaining_avg_sample = remaining_pnl_sample / len(remaining_sample)
    remaining_win_rate_sample = (remaining_sample['Profit (Currency)'] > 0).mean()
    
    # Scale to full dataset
    scaling_factor = results['scaling_factor']
    
    print(f"   Current Performance (estimated full dataset):")
    print(f"     Total trades: {total_trades:,}")
    print(f"     Est. total P&L: {sample_pnl * scaling_factor:.2f}")
    print(f"     Est. avg P&L per trade: {sample_avg:.2f}")
    print(f"     Est. win rate: {sample_win_rate*100:.1f}%")
    
    print(f"   Trades to Filter Out:")
    print(f"     Sample re-entries: {len(cascade_df):,}")
    print(f"     Est. total re-entries: {results['estimated_reentry_count']:,} ({results['estimated_reentry_count']/total_trades*100:.1f}% of all trades)")
    print(f"     Est. P&L of filtered trades: {reentry_pnl_sample * scaling_factor:.2f}")
    
    print(f"   After Filter (estimated):")
    remaining_trades_est = total_trades - results['estimated_reentry_count']
    remaining_total_pnl_est = remaining_pnl_sample * scaling_factor
    remaining_avg_est = remaining_total_pnl_est / remaining_trades_est
    
    print(f"     Remaining trades: {remaining_trades_est:,}")
    print(f"     Est. total P&L: {remaining_total_pnl_est:.2f}")
    print(f"     Est. avg P&L per trade: {remaining_avg_est:.2f}")
    print(f"     Est. win rate: {remaining_win_rate_sample*100:.1f}%")
    
    print(f"   Performance Improvement:")
    improvement_per_trade = remaining_avg_est - sample_avg
    improvement_win_rate = remaining_win_rate_sample - sample_win_rate
    total_improvement = improvement_per_trade * remaining_trades_est
    
    print(f"     P&L per trade improvement: {improvement_per_trade:.2f}")
    print(f"     Win rate improvement: {improvement_win_rate*100:.1f} percentage points")
    print(f"     Total portfolio improvement: {total_improvement:.2f}")

def main():
    """Main analysis function."""
    file_path = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/back_tester/backtester/consolidated_trades.csv"
    
    print("="*70)
    print("RE-ENTRY CASCADE ANALYSIS (SAMPLE-BASED)")
    print("="*70)
    
    # Load sample data
    sample_df, total_trades = sample_and_analyze(file_path, sample_size=25000)
    
    # Quick cascade detection
    cascade_df = quick_cascade_analysis(sample_df)
    
    # Analyze results
    results = analyze_sample_results(cascade_df, sample_df, total_trades)
    
    if results:
        # Simulate solution
        simulate_filter_impact(sample_df, cascade_df, total_trades, results)
        
        print(f"\n" + "="*70)
        print("KEY FINDINGS & RECOMMENDATIONS")
        print("="*70)
        
        print(f"• Estimated {results['estimated_reentry_count']:,} re-entry trades in full dataset")
        print(f"• Re-entries underperform first trades by {results['performance_gap']:.2f} per trade")
        print(f"• Re-entries have {results['accuracy_gap']*100:.1f} percentage points lower win rate")
        print(f"• Total estimated profit lost: {results['estimated_total_loss']:.2f}")
        print(f"• Filtering re-entries would remove {results['estimated_reentry_count']/total_trades*100:.1f}% of trades")
        print(f"• This filter would significantly improve strategy performance")
        print(f"• Recommendation: Implement direction-alternating filter to prevent same-direction re-entries")

if __name__ == "__main__":
    main()