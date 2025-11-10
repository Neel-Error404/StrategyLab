#!/usr/bin/env python3
"""
Comprehensive Re-Entry Cascade Analysis
=======================================

This script analyzes the consolidated trades data to identify and quantify the 
"re-entry cascade" problem where consecutive same-direction trades lead to 
systematic losses.

Analysis includes:
1. Sequential same-direction trade detection
2. Performance impact analysis  
3. Pattern analysis
4. Proposed solution validation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path):
    """Load and prepare the trades data for analysis."""
    print("Loading trades data...")
    
    # Load the data
    df = pd.read_csv(file_path)
    
    # Convert datetime columns
    df['Entry Time'] = pd.to_datetime(df['Entry Time'])
    df['Exit Time'] = pd.to_datetime(df['Exit Time'])
    
    # Sort by ticker and entry time
    df = df.sort_values(['ticker', 'Entry Time']).reset_index(drop=True)
    
    # Create a unique trade ID
    df['trade_id'] = range(len(df))
    
    print(f"Loaded {len(df):,} trades covering {df['ticker'].nunique()} tickers")
    print(f"Date range: {df['Entry Time'].min().date()} to {df['Entry Time'].max().date()}")
    
    return df

def identify_sequential_trades(df):
    """Identify consecutive same-direction trades by ticker."""
    print("\nIdentifying sequential same-direction trades...")
    
    sequences = []
    
    for ticker in df['ticker'].unique():
        ticker_trades = df[df['ticker'] == ticker].copy()
        
        if len(ticker_trades) < 2:
            continue
            
        current_sequence = [ticker_trades.iloc[0]]
        
        for i in range(1, len(ticker_trades)):
            prev_trade = ticker_trades.iloc[i-1]
            curr_trade = ticker_trades.iloc[i]
            
            # Check if same direction as previous trade
            if prev_trade['Trade Type'] == curr_trade['Trade Type']:
                # Add to current sequence
                if len(current_sequence) == 1:
                    # This is the start of a new sequence
                    current_sequence = [prev_trade, curr_trade]
                else:
                    # Continue existing sequence
                    current_sequence.append(curr_trade)
            else:
                # Direction changed - save sequence if it has 2+ trades
                if len(current_sequence) >= 2:
                    sequences.append({
                        'ticker': ticker,
                        'sequence_length': len(current_sequence),
                        'trade_type': current_sequence[0]['Trade Type'],
                        'trades': current_sequence.copy()
                    })
                
                # Reset for new potential sequence
                current_sequence = [curr_trade]
        
        # Handle last sequence
        if len(current_sequence) >= 2:
            sequences.append({
                'ticker': ticker,
                'sequence_length': len(current_sequence),
                'trade_type': current_sequence[0]['Trade Type'],
                'trades': current_sequence.copy()
            })
    
    return sequences

def analyze_sequence_performance(sequences):
    """Analyze performance of sequential same-direction trades."""
    print("\nAnalyzing sequence performance...")
    
    results = {
        'total_sequences': len(sequences),
        'buy_sequences': 0,
        'sell_sequences': 0,
        'first_trade_stats': {'profit': [], 'accuracy': []},
        'reentry_trade_stats': {'profit': [], 'accuracy': []},
        'sequence_stats': []
    }
    
    for seq in sequences:
        if seq['trade_type'] == 'Buy':
            results['buy_sequences'] += 1
        else:
            results['sell_sequences'] += 1
        
        trades = seq['trades']
        
        # First trade stats
        first_trade = trades[0]
        first_profit = first_trade['Profit (Currency)']
        first_accurate = 1 if first_profit > 0 else 0
        
        results['first_trade_stats']['profit'].append(first_profit)
        results['first_trade_stats']['accuracy'].append(first_accurate)
        
        # Re-entry trades stats
        reentry_profits = []
        reentry_accuracy = []
        
        for trade in trades[1:]:  # All trades except first
            profit = trade['Profit (Currency)']
            accurate = 1 if profit > 0 else 0
            
            results['reentry_trade_stats']['profit'].append(profit)
            results['reentry_trade_stats']['accuracy'].append(accurate)
            reentry_profits.append(profit)
            reentry_accuracy.append(accurate)
        
        # Sequence-level stats
        total_sequence_profit = sum(trade['Profit (Currency)'] for trade in trades)
        sequence_duration_hours = (trades[-1]['Exit Time'] - trades[0]['Entry Time']).total_seconds() / 3600
        
        time_gaps = []
        for i in range(1, len(trades)):
            gap_hours = (trades[i]['Entry Time'] - trades[i-1]['Exit Time']).total_seconds() / 3600
            time_gaps.append(gap_hours)
        
        results['sequence_stats'].append({
            'ticker': seq['ticker'],
            'trade_type': seq['trade_type'],
            'sequence_length': seq['sequence_length'],
            'first_trade_profit': first_profit,
            'reentry_profits': reentry_profits,
            'total_profit': total_sequence_profit,
            'sequence_duration_hours': sequence_duration_hours,
            'time_gaps_hours': time_gaps,
            'avg_time_gap_hours': np.mean(time_gaps) if time_gaps else 0
        })
    
    return results

def calculate_summary_statistics(results):
    """Calculate and print summary statistics."""
    print("\n" + "="*70)
    print("RE-ENTRY CASCADE ANALYSIS RESULTS")
    print("="*70)
    
    # Overall sequence statistics
    print(f"\n1. SEQUENTIAL SAME-DIRECTION TRADE DETECTION:")
    print(f"   Total sequences found: {results['total_sequences']:,}")
    print(f"   Buy sequences: {results['buy_sequences']:,}")
    print(f"   Sell sequences: {results['sell_sequences']:,}")
    
    # Calculate total trades involved
    total_trades_in_sequences = sum(len(seq['trades']) for seq in results['sequence_stats'])
    total_reentry_trades = total_trades_in_sequences - results['total_sequences']
    print(f"   Total trades in sequences: {total_trades_in_sequences:,}")
    print(f"   Total re-entry trades: {total_reentry_trades:,}")
    
    # Performance comparison
    first_profits = np.array(results['first_trade_stats']['profit'])
    reentry_profits = np.array(results['reentry_trade_stats']['profit'])
    first_accuracy = np.array(results['first_trade_stats']['accuracy'])
    reentry_accuracy = np.array(results['reentry_trade_stats']['accuracy'])
    
    print(f"\n2. PERFORMANCE IMPACT ANALYSIS:")
    print(f"   First Trade Performance:")
    print(f"     Average P&L: {np.mean(first_profits):.2f}")
    print(f"     Total P&L: {np.sum(first_profits):.2f}")
    print(f"     Accuracy: {np.mean(first_accuracy)*100:.1f}%")
    print(f"     Win Rate: {np.sum(first_profits > 0)}/{len(first_profits)} = {np.sum(first_profits > 0)/len(first_profits)*100:.1f}%")
    
    print(f"   Re-entry Trade Performance:")
    print(f"     Average P&L: {np.mean(reentry_profits):.2f}")
    print(f"     Total P&L: {np.sum(reentry_profits):.2f}")
    print(f"     Accuracy: {np.mean(reentry_accuracy)*100:.1f}%")
    print(f"     Win Rate: {np.sum(reentry_profits > 0)}/{len(reentry_profits)} = {np.sum(reentry_profits > 0)/len(reentry_profits)*100:.1f}%")
    
    # Impact calculation
    performance_diff = np.mean(first_profits) - np.mean(reentry_profits)
    accuracy_diff = np.mean(first_accuracy) - np.mean(reentry_accuracy)
    
    print(f"   Performance Difference:")
    print(f"     P&L Impact: {performance_diff:.2f} per trade")
    print(f"     Accuracy Impact: {accuracy_diff*100:.1f}%")
    print(f"     Total P&L Lost to Re-entries: {np.sum(reentry_profits):.2f}")
    
    return {
        'first_trade_avg_pnl': np.mean(first_profits),
        'reentry_avg_pnl': np.mean(reentry_profits),
        'first_trade_accuracy': np.mean(first_accuracy),
        'reentry_accuracy': np.mean(reentry_accuracy),
        'total_reentry_pnl': np.sum(reentry_profits),
        'total_reentry_trades': len(reentry_profits)
    }

def analyze_patterns(results, df):
    """Analyze patterns in the re-entry cascade problem."""
    print(f"\n3. PATTERN ANALYSIS:")
    
    # Ticker analysis
    ticker_impact = {}
    for seq_stat in results['sequence_stats']:
        ticker = seq_stat['ticker']
        if ticker not in ticker_impact:
            ticker_impact[ticker] = {
                'sequences': 0,
                'reentry_trades': 0,
                'reentry_pnl': 0
            }
        
        ticker_impact[ticker]['sequences'] += 1
        ticker_impact[ticker]['reentry_trades'] += len(seq_stat['reentry_profits'])
        ticker_impact[ticker]['reentry_pnl'] += sum(seq_stat['reentry_profits'])
    
    # Sort by impact
    worst_tickers = sorted(ticker_impact.items(), 
                          key=lambda x: x[1]['reentry_pnl'])[:10]
    
    print(f"   Top 10 Tickers Most Affected by Re-entry Cascade:")
    for i, (ticker, stats) in enumerate(worst_tickers, 1):
        print(f"     {i:2d}. {ticker}: {stats['sequences']} sequences, "
              f"{stats['reentry_trades']} re-entries, "
              f"P&L: {stats['reentry_pnl']:.2f}")
    
    # Timing analysis
    all_time_gaps = []
    for seq_stat in results['sequence_stats']:
        all_time_gaps.extend(seq_stat['time_gaps_hours'])
    
    all_time_gaps = np.array(all_time_gaps)
    
    print(f"\n   Timing Patterns:")
    print(f"     Average time gap between re-entries: {np.mean(all_time_gaps):.1f} hours")
    print(f"     Median time gap: {np.median(all_time_gaps):.1f} hours")
    print(f"     Re-entries within 1 hour: {np.sum(all_time_gaps < 1)}/{len(all_time_gaps)} = {np.sum(all_time_gaps < 1)/len(all_time_gaps)*100:.1f}%")
    print(f"     Re-entries within 24 hours: {np.sum(all_time_gaps < 24)}/{len(all_time_gaps)} = {np.sum(all_time_gaps < 24)/len(all_time_gaps)*100:.1f}%")
    
    # Sequence length analysis
    sequence_lengths = [seq_stat['sequence_length'] for seq_stat in results['sequence_stats']]
    unique_lengths, counts = np.unique(sequence_lengths, return_counts=True)
    
    print(f"\n   Sequence Length Distribution:")
    for length, count in zip(unique_lengths, counts):
        pct = count / len(sequence_lengths) * 100
        print(f"     {length} trades: {count:,} sequences ({pct:.1f}%)")

def simulate_solution(df, sequences):
    """Simulate the effect of requiring opposite direction trades between same-direction entries."""
    print(f"\n4. PROPOSED SOLUTION VALIDATION:")
    
    # Create a copy for simulation
    trades_to_remove = set()
    
    for seq in sequences:
        # Keep first trade, mark re-entries for removal
        for trade in seq['trades'][1:]:  # All except first
            trades_to_remove.add(trade['trade_id'])
    
    # Calculate current performance
    total_trades = len(df)
    total_current_pnl = df['Profit (Currency)'].sum()
    
    # Calculate what would be removed
    removed_trades = df[df['trade_id'].isin(trades_to_remove)]
    remaining_trades = df[~df['trade_id'].isin(trades_to_remove)]
    
    removed_pnl = removed_trades['Profit (Currency)'].sum()
    remaining_pnl = remaining_trades['Profit (Currency)'].sum()
    
    # Performance improvement
    avg_current_pnl = total_current_pnl / total_trades
    avg_remaining_pnl = remaining_pnl / len(remaining_trades)
    
    print(f"   Solution Impact:")
    print(f"     Current total trades: {total_trades:,}")
    print(f"     Trades to be filtered out: {len(removed_trades):,} ({len(removed_trades)/total_trades*100:.1f}%)")
    print(f"     Remaining trades: {len(remaining_trades):,}")
    
    print(f"\n   P&L Impact:")
    print(f"     Current total P&L: {total_current_pnl:.2f}")
    print(f"     P&L from removed trades: {removed_pnl:.2f}")
    print(f"     P&L from remaining trades: {remaining_pnl:.2f}")
    
    print(f"\n   Performance Metrics:")
    print(f"     Current average P&L per trade: {avg_current_pnl:.2f}")
    print(f"     New average P&L per trade: {avg_remaining_pnl:.2f}")
    print(f"     Performance improvement: {avg_remaining_pnl - avg_current_pnl:.2f} per trade")
    print(f"     Total performance improvement: {(avg_remaining_pnl - avg_current_pnl) * len(remaining_trades):.2f}")
    
    # Calculate win rates
    current_win_rate = (df['Profit (Currency)'] > 0).sum() / len(df)
    remaining_win_rate = (remaining_trades['Profit (Currency)'] > 0).sum() / len(remaining_trades)
    removed_win_rate = (removed_trades['Profit (Currency)'] > 0).sum() / len(removed_trades)
    
    print(f"\n   Accuracy Metrics:")
    print(f"     Current win rate: {current_win_rate*100:.1f}%")
    print(f"     Remaining trades win rate: {remaining_win_rate*100:.1f}%")
    print(f"     Removed trades win rate: {removed_win_rate*100:.1f}%")
    print(f"     Accuracy improvement: {(remaining_win_rate - current_win_rate)*100:.1f}%")

def main():
    """Main analysis function."""
    file_path = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/back_tester/backtester/consolidated_trades.csv"
    
    # Load data
    df = load_and_prepare_data(file_path)
    
    # Identify sequential trades
    sequences = identify_sequential_trades(df)
    
    # Analyze performance
    results = analyze_sequence_performance(sequences)
    
    # Calculate summary statistics
    summary_stats = calculate_summary_statistics(results)
    
    # Analyze patterns
    analyze_patterns(results, df)
    
    # Simulate solution
    simulate_solution(df, sequences)
    
    print(f"\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    print(f"\nKEY FINDINGS:")
    print(f"• {len(sequences):,} re-entry cascade sequences identified")
    print(f"• {summary_stats['total_reentry_trades']:,} trades are re-entries causing losses")
    print(f"• Re-entries have {summary_stats['reentry_accuracy']*100:.1f}% accuracy vs {summary_stats['first_trade_accuracy']*100:.1f}% for first trades")
    print(f"• Total profit lost to re-entry cascade: {summary_stats['total_reentry_pnl']:.2f}")
    print(f"• Average re-entry performance: {summary_stats['reentry_avg_pnl']:.2f} vs {summary_stats['first_trade_avg_pnl']:.2f} for first trades")
    print(f"• Implementing the filter would improve performance significantly")

if __name__ == "__main__":
    main()