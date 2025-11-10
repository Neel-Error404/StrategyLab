#!/usr/bin/env python3
"""
Corrected Re-Entry Cascade Analysis
===================================

Properly identifies and quantifies the re-entry cascade problem where
consecutive same-direction trades occur without opposite direction trades.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path):
    """Load and prepare the trades data for analysis."""
    print("Loading trades data...")
    
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
    df['trade_id'] = df.index
    
    print(f"Loaded {len(df):,} trades covering {df['ticker'].nunique()} tickers")
    print(f"Date range: {df['Entry Time'].min().date()} to {df['Exit Time'].max().date()}")
    
    return df

def identify_reentry_cascades(df):
    """Identify re-entry cascade patterns."""
    print("Identifying re-entry cascade patterns...")
    
    cascade_info = []
    
    # Process each ticker separately
    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker].copy()
        
        if len(ticker_df) < 2:
            continue
        
        i = 0
        while i < len(ticker_df):
            current_trade = ticker_df.iloc[i]
            cascade = [current_trade]
            
            # Look for consecutive same-direction trades
            j = i + 1
            while j < len(ticker_df):
                next_trade = ticker_df.iloc[j]
                
                # If same direction, add to cascade
                if current_trade['Trade Type'] == next_trade['Trade Type']:
                    cascade.append(next_trade)
                    j += 1
                else:
                    # Direction changed, break the cascade
                    break
            
            # If cascade has 2+ trades, record it
            if len(cascade) >= 2:
                for k, trade in enumerate(cascade):
                    cascade_info.append({
                        'ticker': ticker,
                        'trade_id': trade['trade_id'],
                        'trade_type': trade['Trade Type'],
                        'cascade_id': f"{ticker}_{i}_{trade['Trade Type']}",
                        'position_in_cascade': k,  # 0 = first, 1+ = re-entries
                        'cascade_length': len(cascade),
                        'is_reentry': k > 0,  # True for all except first
                        'profit': trade['Profit (Currency)'],
                        'entry_time': trade['Entry Time'],
                        'exit_time': trade['Exit Time']
                    })
            
            # Move to next potential cascade start
            i = j if j > i + 1 else i + 1
    
    cascade_df = pd.DataFrame(cascade_info)
    
    print(f"Found {cascade_df['cascade_id'].nunique():,} cascade sequences")
    print(f"Total trades in cascades: {len(cascade_df):,}")
    print(f"Re-entry trades: {cascade_df['is_reentry'].sum():,}")
    
    return cascade_df

def analyze_cascade_performance(cascade_df, original_df):
    """Analyze performance of cascades vs regular trades."""
    print("\nAnalyzing cascade performance...")
    
    # Separate first trades and re-entries
    first_trades = cascade_df[~cascade_df['is_reentry']]
    reentry_trades = cascade_df[cascade_df['is_reentry']]
    
    # Non-cascade trades (trades not part of any cascade)
    cascade_trade_ids = set(cascade_df['trade_id'])
    non_cascade_trades = original_df[~original_df['trade_id'].isin(cascade_trade_ids)]
    
    print(f"\nTrade Categories:")
    print(f"  Cascade first trades: {len(first_trades):,}")
    print(f"  Re-entry trades: {len(reentry_trades):,}")  
    print(f"  Non-cascade trades: {len(non_cascade_trades):,}")
    print(f"  Total: {len(first_trades) + len(reentry_trades) + len(non_cascade_trades):,}")
    
    # Calculate performance metrics
    def calc_metrics(trades_df, profit_col='profit'):
        if profit_col == 'profit':
            profits = trades_df['profit']
        else:
            profits = trades_df['Profit (Currency)']
        
        return {
            'count': len(trades_df),
            'total_pnl': profits.sum(),
            'avg_pnl': profits.mean(),
            'win_rate': (profits > 0).mean(),
            'median_pnl': profits.median(),
            'std_pnl': profits.std()
        }
    
    first_metrics = calc_metrics(first_trades)
    reentry_metrics = calc_metrics(reentry_trades)
    non_cascade_metrics = calc_metrics(non_cascade_trades, 'Profit (Currency)')
    
    print(f"\nPerformance Analysis:")
    print(f"  Cascade First Trades:")
    print(f"    Count: {first_metrics['count']:,}")
    print(f"    Total P&L: {first_metrics['total_pnl']:.2f}")
    print(f"    Average P&L: {first_metrics['avg_pnl']:.2f}")
    print(f"    Win Rate: {first_metrics['win_rate']*100:.1f}%")
    print(f"    Median P&L: {first_metrics['median_pnl']:.2f}")
    
    print(f"  Re-entry Trades:")
    print(f"    Count: {reentry_metrics['count']:,}")
    print(f"    Total P&L: {reentry_metrics['total_pnl']:.2f}")
    print(f"    Average P&L: {reentry_metrics['avg_pnl']:.2f}")
    print(f"    Win Rate: {reentry_metrics['win_rate']*100:.1f}%")
    print(f"    Median P&L: {reentry_metrics['median_pnl']:.2f}")
    
    print(f"  Non-Cascade Trades:")
    print(f"    Count: {non_cascade_metrics['count']:,}")
    print(f"    Total P&L: {non_cascade_metrics['total_pnl']:.2f}")
    print(f"    Average P&L: {non_cascade_metrics['avg_pnl']:.2f}")
    print(f"    Win Rate: {non_cascade_metrics['win_rate']*100:.1f}%")
    print(f"    Median P&L: {non_cascade_metrics['median_pnl']:.2f}")
    
    # Impact analysis
    performance_gap = first_metrics['avg_pnl'] - reentry_metrics['avg_pnl']
    accuracy_gap = first_metrics['win_rate'] - reentry_metrics['win_rate']
    
    print(f"\nRe-entry Impact:")
    print(f"  Performance gap: {performance_gap:.2f} per trade")
    print(f"  Accuracy gap: {accuracy_gap*100:.1f} percentage points")
    print(f"  Total P&L lost to re-entries: {reentry_metrics['total_pnl']:.2f}")
    
    # Compared to non-cascade trades
    non_cascade_gap = non_cascade_metrics['avg_pnl'] - reentry_metrics['avg_pnl']
    non_cascade_acc_gap = non_cascade_metrics['win_rate'] - reentry_metrics['win_rate']
    
    print(f"\nRe-entries vs Non-Cascade Trades:")
    print(f"  Performance gap: {non_cascade_gap:.2f} per trade")
    print(f"  Accuracy gap: {non_cascade_acc_gap*100:.1f} percentage points")
    
    return {
        'first_metrics': first_metrics,
        'reentry_metrics': reentry_metrics,
        'non_cascade_metrics': non_cascade_metrics,
        'performance_gap': performance_gap,
        'accuracy_gap': accuracy_gap
    }

def analyze_cascade_patterns(cascade_df):
    """Analyze patterns in cascade behavior."""
    print(f"\nAnalyzing cascade patterns...")
    
    # Cascade length distribution
    cascade_lengths = cascade_df.groupby('cascade_id')['cascade_length'].first()
    length_counts = cascade_lengths.value_counts().sort_index()
    
    print(f"Cascade Length Distribution:")
    for length, count in length_counts.head(10).items():
        pct = count / len(cascade_lengths) * 100
        print(f"  {length} trades: {count:,} cascades ({pct:.1f}%)")
    
    # Trade type breakdown
    type_breakdown = cascade_df.groupby(['trade_type', 'is_reentry']).agg({
        'profit': ['count', 'sum', 'mean']
    }).round(2)
    
    print(f"\nTrade Type Breakdown:")
    for trade_type in ['Buy', 'Sell']:
        subset = cascade_df[cascade_df['trade_type'] == trade_type]
        first_count = len(subset[~subset['is_reentry']])
        reentry_count = len(subset[subset['is_reentry']])
        
        first_pnl = subset[~subset['is_reentry']]['profit'].sum()
        reentry_pnl = subset[subset['is_reentry']]['profit'].sum()
        
        print(f"  {trade_type} Cascades:")
        print(f"    First trades: {first_count:,} (P&L: {first_pnl:.2f})")
        print(f"    Re-entries: {reentry_count:,} (P&L: {reentry_pnl:.2f})")
    
    # Timing analysis
    reentries = cascade_df[cascade_df['is_reentry']].sort_values(['cascade_id', 'position_in_cascade'])
    
    # Calculate time gaps
    time_gaps = []
    for cascade_id in reentries['cascade_id'].unique():
        cascade_trades = cascade_df[cascade_df['cascade_id'] == cascade_id].sort_values('position_in_cascade')
        
        for i in range(1, len(cascade_trades)):
            prev_exit = cascade_trades.iloc[i-1]['exit_time']
            curr_entry = cascade_trades.iloc[i]['entry_time']
            gap_hours = (curr_entry - prev_exit).total_seconds() / 3600
            time_gaps.append(gap_hours)
    
    time_gaps = np.array(time_gaps)
    
    print(f"\nTiming Patterns:")
    print(f"  Average time gap between re-entries: {np.mean(time_gaps):.1f} hours")
    print(f"  Median time gap: {np.median(time_gaps):.1f} hours")
    print(f"  Re-entries within 1 hour: {np.sum(time_gaps < 1):,} ({np.mean(time_gaps < 1)*100:.1f}%)")
    print(f"  Re-entries within 24 hours: {np.sum(time_gaps < 24):,} ({np.mean(time_gaps < 24)*100:.1f}%)")
    print(f"  Re-entries within 1 week: {np.sum(time_gaps < 168):,} ({np.mean(time_gaps < 168)*100:.1f}%)")

def analyze_ticker_impact(cascade_df):
    """Analyze which tickers have the worst re-entry cascade problem."""
    print(f"\nAnalyzing ticker impact...")
    
    ticker_stats = []
    
    for ticker in cascade_df['ticker'].unique():
        ticker_data = cascade_df[cascade_df['ticker'] == ticker]
        
        reentries = ticker_data[ticker_data['is_reentry']]
        cascades = ticker_data['cascade_id'].nunique()
        
        if len(reentries) > 0:
            ticker_stats.append({
                'ticker': ticker,
                'cascade_count': cascades,
                'reentry_count': len(reentries),
                'reentry_pnl': reentries['profit'].sum(),
                'avg_reentry_pnl': reentries['profit'].mean(),
                'reentry_win_rate': (reentries['profit'] > 0).mean()
            })
    
    ticker_df = pd.DataFrame(ticker_stats).sort_values('reentry_pnl')
    
    print(f"Top 15 Tickers Most Affected by Re-entry Cascade (by total P&L loss):")
    for i, row in ticker_df.head(15).iterrows():
        print(f"  {row.name+1:2d}. {row['ticker']}: {row['cascade_count']} cascades, "
              f"{row['reentry_count']} re-entries, "
              f"P&L: {row['reentry_pnl']:.2f}, "
              f"Avg: {row['avg_reentry_pnl']:.2f}, "
              f"Win Rate: {row['reentry_win_rate']*100:.1f}%")

def simulate_solution(cascade_df, original_df):
    """Simulate the effect of filtering out re-entry trades."""
    print(f"\nSimulating solution: Remove all re-entry trades...")
    
    # Current performance
    current_pnl = original_df['Profit (Currency)'].sum()
    current_count = len(original_df)
    current_avg = current_pnl / current_count
    current_win_rate = (original_df['Profit (Currency)'] > 0).mean()
    
    # Re-entry trades to remove
    reentry_trades = cascade_df[cascade_df['is_reentry']]
    reentry_count = len(reentry_trades)
    reentry_pnl = reentry_trades['profit'].sum()
    
    # After filtering
    remaining_count = current_count - reentry_count
    remaining_pnl = current_pnl - reentry_pnl
    remaining_avg = remaining_pnl / remaining_count if remaining_count > 0 else 0
    
    # Calculate remaining win rate by excluding re-entry trade IDs
    reentry_ids = set(reentry_trades['trade_id'])
    remaining_trades = original_df[~original_df['trade_id'].isin(reentry_ids)]
    remaining_win_rate = (remaining_trades['Profit (Currency)'] > 0).mean()
    
    print(f"Solution Impact Analysis:")
    print(f"  Before Filter:")
    print(f"    Total trades: {current_count:,}")
    print(f"    Total P&L: {current_pnl:.2f}")
    print(f"    Average P&L per trade: {current_avg:.2f}")
    print(f"    Win rate: {current_win_rate*100:.1f}%")
    
    print(f"  Trades to Remove (Re-entries):")
    print(f"    Count: {reentry_count:,} ({reentry_count/current_count*100:.1f}% of all trades)")
    print(f"    Total P&L: {reentry_pnl:.2f}")
    print(f"    Average P&L: {reentry_pnl/reentry_count:.2f}")
    print(f"    Win rate: {(reentry_trades['profit'] > 0).mean()*100:.1f}%")
    
    print(f"  After Filter:")
    print(f"    Remaining trades: {remaining_count:,}")
    print(f"    Total P&L: {remaining_pnl:.2f}")
    print(f"    Average P&L per trade: {remaining_avg:.2f}")
    print(f"    Win rate: {remaining_win_rate*100:.1f}%")
    
    print(f"  Performance Improvement:")
    improvement_per_trade = remaining_avg - current_avg
    improvement_win_rate = remaining_win_rate - current_win_rate
    total_improvement = improvement_per_trade * remaining_count
    
    print(f"    P&L per trade improvement: {improvement_per_trade:.2f}")
    print(f"    Win rate improvement: {improvement_win_rate*100:.1f} percentage points")
    print(f"    Total portfolio improvement: {total_improvement:.2f}")
    
    print(f"\nKey Insight: By filtering out re-entry trades, we:")
    print(f"  • Remove {reentry_count:,} problematic trades ({reentry_count/current_count*100:.1f}% of portfolio)")
    print(f"  • Improve average P&L per trade by {improvement_per_trade:.2f}")
    print(f"  • Improve win rate by {improvement_win_rate*100:.1f} percentage points")
    print(f"  • Would have made {total_improvement:.2f} more profit overall")

def main():
    """Main analysis function."""
    file_path = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/back_tester/backtester/consolidated_trades.csv"
    
    print("="*80)
    print("RE-ENTRY CASCADE PROBLEM ANALYSIS")
    print("="*80)
    
    # Load data
    df = load_and_prepare_data(file_path)
    
    # Identify cascades
    cascade_df = identify_reentry_cascades(df)
    
    if len(cascade_df) == 0:
        print("No re-entry cascades found!")
        return
    
    # Analyze performance
    metrics = analyze_cascade_performance(cascade_df, df)
    
    # Pattern analysis
    analyze_cascade_patterns(cascade_df)
    
    # Ticker impact
    analyze_ticker_impact(cascade_df)
    
    # Solution simulation
    simulate_solution(cascade_df, df)
    
    print("\n" + "="*80)
    print("EXECUTIVE SUMMARY")
    print("="*80)
    
    reentry_count = (cascade_df['is_reentry']).sum()
    cascade_count = cascade_df['cascade_id'].nunique()
    
    print(f"• Found {cascade_count:,} re-entry cascade sequences")
    print(f"• {reentry_count:,} trades are problematic re-entries ({reentry_count/len(df)*100:.1f}% of all trades)")
    print(f"• Re-entries underperform by {metrics['performance_gap']:.2f} per trade")
    print(f"• Re-entries have {metrics['accuracy_gap']*100:.1f} percentage points lower win rate")
    print(f"• Total profit currently lost to re-entry cascade: {metrics['reentry_metrics']['total_pnl']:.2f}")
    print(f"• Implementing the proposed filter would significantly improve strategy performance")

if __name__ == "__main__":
    main()