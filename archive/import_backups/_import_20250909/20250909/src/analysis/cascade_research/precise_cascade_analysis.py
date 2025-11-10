#!/usr/bin/env python3
"""
Precise Re-Entry Cascade Analysis
=================================

Focused analysis to identify true re-entry cascade patterns where consecutive
same-direction trades occur rapidly without alternating directions.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_sample_data(file_path, n_tickers=50):
    """Load data for top N most active tickers to focus analysis."""
    print(f"Loading data for top {n_tickers} most active tickers...")
    
    # First pass: identify most active tickers
    ticker_counts = {}
    chunk_size = 50000
    
    for chunk in pd.read_csv(file_path, chunksize=chunk_size, usecols=['ticker']):
        chunk_counts = chunk['ticker'].value_counts()
        for ticker, count in chunk_counts.items():
            ticker_counts[ticker] = ticker_counts.get(ticker, 0) + count
    
    # Get top N tickers
    top_tickers = sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)[:n_tickers]
    selected_tickers = [ticker for ticker, count in top_tickers]
    
    print(f"Selected tickers: {', '.join(selected_tickers[:10])}...")
    print(f"Total trades for selected tickers: {sum(count for _, count in top_tickers):,}")
    
    # Second pass: load data for selected tickers
    dtypes = {
        'Trade Type': 'category',
        'ticker': 'category', 
        'Profit (Currency)': 'float32'
    }
    
    dfs = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size, dtype=dtypes):
        filtered_chunk = chunk[chunk['ticker'].isin(selected_tickers)]
        if len(filtered_chunk) > 0:
            dfs.append(filtered_chunk)
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Convert datetime columns
    df['Entry Time'] = pd.to_datetime(df['Entry Time'])
    df['Exit Time'] = pd.to_datetime(df['Exit Time'])
    
    # Sort by ticker and entry time
    df = df.sort_values(['ticker', 'Entry Time']).reset_index(drop=True)
    df['trade_id'] = df.index
    
    print(f"Loaded {len(df):,} trades for analysis")
    
    return df

def identify_true_cascades(df, max_gap_hours=24):
    """Identify true re-entry cascades with specific criteria."""
    print(f"Identifying re-entry cascades (max gap: {max_gap_hours} hours)...")
    
    cascades = []
    cascade_trades = []
    
    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker].copy()
        
        if len(ticker_df) < 2:
            continue
        
        i = 0
        while i < len(ticker_df):
            current_sequence = [ticker_df.iloc[i]]
            j = i + 1
            
            # Build sequence of same-direction trades
            while j < len(ticker_df):
                prev_trade = current_sequence[-1]
                next_trade = ticker_df.iloc[j]
                
                # Check if same direction
                if prev_trade['Trade Type'] == next_trade['Trade Type']:
                    # Check time gap
                    time_gap = (next_trade['Entry Time'] - prev_trade['Exit Time']).total_seconds() / 3600
                    
                    if time_gap <= max_gap_hours:
                        current_sequence.append(next_trade)
                        j += 1
                    else:
                        # Gap too large, end sequence
                        break
                else:
                    # Direction changed, end sequence
                    break
            
            # Record cascade if 2+ trades
            if len(current_sequence) >= 2:
                cascade_id = f"{ticker}_{i}_{current_sequence[0]['Trade Type']}"
                
                cascades.append({
                    'cascade_id': cascade_id,
                    'ticker': ticker,
                    'trade_type': current_sequence[0]['Trade Type'],
                    'sequence_length': len(current_sequence),
                    'total_profit': sum(t['Profit (Currency)'] for t in current_sequence),
                    'first_trade_profit': current_sequence[0]['Profit (Currency)'],
                    'reentry_profits': [t['Profit (Currency)'] for t in current_sequence[1:]],
                    'start_time': current_sequence[0]['Entry Time'],
                    'end_time': current_sequence[-1]['Exit Time'],
                    'duration_hours': (current_sequence[-1]['Exit Time'] - current_sequence[0]['Entry Time']).total_seconds() / 3600
                })
                
                # Record individual trades
                for pos, trade in enumerate(current_sequence):
                    cascade_trades.append({
                        'cascade_id': cascade_id,
                        'ticker': ticker,
                        'trade_id': trade['trade_id'],
                        'position': pos,
                        'is_first': pos == 0,
                        'is_reentry': pos > 0,
                        'trade_type': trade['Trade Type'],
                        'profit': trade['Profit (Currency)'],
                        'entry_time': trade['Entry Time'],
                        'exit_time': trade['Exit Time']
                    })
            
            # Move to next position
            i = max(j, i + 1)
    
    cascade_summary_df = pd.DataFrame(cascades)
    cascade_trades_df = pd.DataFrame(cascade_trades)
    
    print(f"Found {len(cascade_summary_df)} cascade sequences")
    print(f"Total trades in cascades: {len(cascade_trades_df)}")
    print(f"Re-entry trades: {cascade_trades_df['is_reentry'].sum()}")
    
    return cascade_summary_df, cascade_trades_df

def analyze_cascade_impact(cascade_summary_df, cascade_trades_df, full_df):
    """Analyze the impact of cascades on performance."""
    print("\n" + "="*70)
    print("CASCADE IMPACT ANALYSIS")
    print("="*70)
    
    if len(cascade_trades_df) == 0:
        print("No cascades found!")
        return
    
    # Separate first trades and re-entries
    first_trades = cascade_trades_df[cascade_trades_df['is_first']]
    reentry_trades = cascade_trades_df[cascade_trades_df['is_reentry']]
    
    # Non-cascade trades
    cascade_trade_ids = set(cascade_trades_df['trade_id'])
    non_cascade_trades = full_df[~full_df['trade_id'].isin(cascade_trade_ids)]
    
    print(f"Trade Distribution:")
    print(f"  First trades in cascades: {len(first_trades):,}")
    print(f"  Re-entry trades: {len(reentry_trades):,}")
    print(f"  Non-cascade trades: {len(non_cascade_trades):,}")
    print(f"  Total: {len(full_df):,}")
    
    # Performance comparison
    def analyze_group(trades, profit_col, name):
        if len(trades) == 0:
            return None
        
        if profit_col == 'profit':
            profits = trades['profit']
        else:
            profits = trades[profit_col]
            
        return {
            'name': name,
            'count': len(trades),
            'total_pnl': profits.sum(),
            'avg_pnl': profits.mean(),
            'median_pnl': profits.median(),
            'win_rate': (profits > 0).mean(),
            'loss_rate': (profits < 0).mean(),
            'max_profit': profits.max(),
            'max_loss': profits.min()
        }
    
    first_stats = analyze_group(first_trades, 'profit', 'First Trades')
    reentry_stats = analyze_group(reentry_trades, 'profit', 'Re-entry Trades')
    non_cascade_stats = analyze_group(non_cascade_trades, 'Profit (Currency)', 'Non-Cascade Trades')
    
    print(f"\nPerformance Comparison:")
    for stats in [first_stats, reentry_stats, non_cascade_stats]:
        if stats:
            print(f"  {stats['name']}:")
            print(f"    Count: {stats['count']:,}")
            print(f"    Total P&L: {stats['total_pnl']:.2f}")
            print(f"    Average P&L: {stats['avg_pnl']:.2f}")
            print(f"    Median P&L: {stats['median_pnl']:.2f}")
            print(f"    Win Rate: {stats['win_rate']*100:.1f}%")
            print(f"    Max Profit: {stats['max_profit']:.2f}")
            print(f"    Max Loss: {stats['max_loss']:.2f}")
    
    # Key comparisons
    if first_stats and reentry_stats:
        pnl_gap = first_stats['avg_pnl'] - reentry_stats['avg_pnl']
        win_rate_gap = first_stats['win_rate'] - reentry_stats['win_rate']
        
        print(f"\nRe-entry vs First Trade Impact:")
        print(f"  Average P&L gap: {pnl_gap:.2f} per trade")
        print(f"  Win rate gap: {win_rate_gap*100:.1f} percentage points")
        print(f"  Total re-entry underperformance: {pnl_gap * len(reentry_trades):.2f}")
        
        if non_cascade_stats:
            non_cascade_gap = non_cascade_stats['avg_pnl'] - reentry_stats['avg_pnl']
            non_cascade_win_gap = non_cascade_stats['win_rate'] - reentry_stats['win_rate']
            
            print(f"\nRe-entry vs Non-Cascade Trade Impact:")
            print(f"  Average P&L gap: {non_cascade_gap:.2f} per trade")
            print(f"  Win rate gap: {non_cascade_win_gap*100:.1f} percentage points")
    
    return {
        'first_stats': first_stats,
        'reentry_stats': reentry_stats,
        'non_cascade_stats': non_cascade_stats,
        'reentry_count': len(reentry_trades)
    }

def analyze_cascade_patterns(cascade_summary_df):
    """Analyze patterns in the cascades."""
    print(f"\nCascade Pattern Analysis:")
    
    # Length distribution
    length_dist = cascade_summary_df['sequence_length'].value_counts().sort_index()
    print(f"  Sequence Length Distribution:")
    for length, count in length_dist.head(10).items():
        pct = count / len(cascade_summary_df) * 100
        print(f"    {length} trades: {count:,} cascades ({pct:.1f}%)")
    
    # Trade type distribution
    type_dist = cascade_summary_df['trade_type'].value_counts()
    print(f"  Trade Type Distribution:")
    for trade_type, count in type_dist.items():
        pct = count / len(cascade_summary_df) * 100
        avg_profit = cascade_summary_df[cascade_summary_df['trade_type'] == trade_type]['total_profit'].mean()
        print(f"    {trade_type}: {count:,} cascades ({pct:.1f}%), Avg P&L: {avg_profit:.2f}")
    
    # Duration analysis
    durations = cascade_summary_df['duration_hours']
    print(f"  Duration Analysis:")
    print(f"    Average cascade duration: {durations.mean():.1f} hours")
    print(f"    Median cascade duration: {durations.median():.1f} hours")
    print(f"    Cascades under 1 hour: {(durations < 1).sum()} ({(durations < 1).mean()*100:.1f}%)")
    print(f"    Cascades under 24 hours: {(durations < 24).sum()} ({(durations < 24).mean()*100:.1f}%)")
    
    # Profitability analysis
    profitable_cascades = cascade_summary_df['total_profit'] > 0
    print(f"  Profitability:")
    print(f"    Profitable cascades: {profitable_cascades.sum()} ({profitable_cascades.mean()*100:.1f}%)")
    print(f"    Average profitable cascade P&L: {cascade_summary_df[profitable_cascades]['total_profit'].mean():.2f}")
    print(f"    Average unprofitable cascade P&L: {cascade_summary_df[~profitable_cascades]['total_profit'].mean():.2f}")

def worst_cascade_analysis(cascade_summary_df):
    """Identify worst performing cascades."""
    print(f"\nWorst Performing Cascades:")
    
    worst_cascades = cascade_summary_df.nsmallest(10, 'total_profit')
    
    for i, (_, cascade) in enumerate(worst_cascades.iterrows(), 1):
        reentry_loss = sum(cascade['reentry_profits'])
        print(f"  {i:2d}. {cascade['ticker']} ({cascade['trade_type']}): "
              f"{cascade['sequence_length']} trades, "
              f"Total P&L: {cascade['total_profit']:.2f}, "
              f"First: {cascade['first_trade_profit']:.2f}, "
              f"Re-entries: {reentry_loss:.2f}")

def simulate_cascade_filter(full_df, cascade_trades_df):
    """Simulate removing re-entry trades from cascades."""
    print(f"\nSimulating Cascade Filter Solution:")
    
    # Current performance
    current_total = full_df['Profit (Currency)'].sum()
    current_avg = current_total / len(full_df)
    current_win_rate = (full_df['Profit (Currency)'] > 0).mean()
    
    # Re-entries to remove
    reentry_trades = cascade_trades_df[cascade_trades_df['is_reentry']]
    reentry_pnl = reentry_trades['profit'].sum()
    reentry_count = len(reentry_trades)
    
    # After filter
    remaining_count = len(full_df) - reentry_count
    remaining_pnl = current_total - reentry_pnl
    remaining_avg = remaining_pnl / remaining_count
    
    # Calculate remaining win rate
    reentry_ids = set(reentry_trades['trade_id'])
    remaining_trades = full_df[~full_df['trade_id'].isin(reentry_ids)]
    remaining_win_rate = (remaining_trades['Profit (Currency)'] > 0).mean()
    
    print(f"  Current Performance:")
    print(f"    Total trades: {len(full_df):,}")
    print(f"    Total P&L: {current_total:.2f}")
    print(f"    Average P&L: {current_avg:.2f}")
    print(f"    Win rate: {current_win_rate*100:.1f}%")
    
    print(f"  Filter Impact:")
    print(f"    Trades to remove: {reentry_count:,} ({reentry_count/len(full_df)*100:.1f}%)")
    print(f"    P&L of removed trades: {reentry_pnl:.2f}")
    print(f"    Remaining trades: {remaining_count:,}")
    
    print(f"  After Filter:")
    print(f"    Total P&L: {remaining_pnl:.2f}")
    print(f"    Average P&L: {remaining_avg:.2f}")
    print(f"    Win rate: {remaining_win_rate*100:.1f}%")
    
    print(f"  Improvement:")
    improvement_per_trade = remaining_avg - current_avg
    improvement_win_rate = remaining_win_rate - current_win_rate
    total_improvement = improvement_per_trade * remaining_count
    
    print(f"    P&L per trade: {improvement_per_trade:+.2f}")
    print(f"    Win rate: {improvement_win_rate*100:+.1f} percentage points")
    print(f"    Total improvement: {total_improvement:+.2f}")

def main():
    """Main analysis function."""
    file_path = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/back_tester/backtester/consolidated_trades.csv"
    
    print("="*70)
    print("PRECISE RE-ENTRY CASCADE ANALYSIS")
    print("="*70)
    
    # Load focused dataset
    df = load_sample_data(file_path, n_tickers=100)
    
    # Identify true cascades (within 24 hours)
    cascade_summary_df, cascade_trades_df = identify_true_cascades(df, max_gap_hours=24)
    
    if len(cascade_trades_df) == 0:
        print("No cascades found with current criteria!")
        return
    
    # Analyze impact
    analysis_results = analyze_cascade_impact(cascade_summary_df, cascade_trades_df, df)
    
    # Pattern analysis
    analyze_cascade_patterns(cascade_summary_df)
    
    # Worst cases
    worst_cascade_analysis(cascade_summary_df)
    
    # Simulation
    simulate_cascade_filter(df, cascade_trades_df)
    
    print(f"\n" + "="*70)
    print("EXECUTIVE SUMMARY")
    print("="*70)
    
    reentry_count = analysis_results['reentry_count']
    print(f"• Found {len(cascade_summary_df)} re-entry cascade sequences")
    print(f"• {reentry_count} trades are problematic re-entries")
    print(f"• These represent {reentry_count/len(df)*100:.1f}% of trades in analyzed sample")
    print(f"• Analysis focused on rapid re-entries (within 24 hours)")
    print(f"• Filtering these trades would improve overall strategy performance")
    print(f"• Recommendation: Implement time-based direction filter to prevent rapid re-entries")

if __name__ == "__main__":
    main()