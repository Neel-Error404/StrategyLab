#!/usr/bin/env python3
"""
Trade Duration Analysis for specific tickers
Analyzes trade patterns, durations, and timing for TATASTEEL, JSWSTEEL, BIOCON, KOTAKBANK
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def load_and_filter_data(file_path, target_tickers):
    """Load CSV data and filter for target tickers"""
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # Filter for target tickers
    df_filtered = df[df['ticker'].isin(target_tickers)].copy()
    
    # Convert datetime columns
    df_filtered['Entry Time'] = pd.to_datetime(df_filtered['Entry Time'])
    df_filtered['Exit Time'] = pd.to_datetime(df_filtered['Exit Time'])
    
    # Extract dates for analysis
    df_filtered['Entry Date'] = df_filtered['Entry Time'].dt.date
    df_filtered['Exit Date'] = df_filtered['Exit Time'].dt.date
    
    # Add helper columns
    df_filtered['Trade Duration (hours)'] = df_filtered['Trade Duration (min)'] / 60
    df_filtered['Same Day Trade'] = df_filtered['Entry Date'] == df_filtered['Exit Date']
    
    print(f"Loaded {len(df_filtered)} trades for analysis")
    print(f"Tickers: {df_filtered['ticker'].value_counts()}")
    
    return df_filtered

def calculate_duration_stats(df):
    """Calculate comprehensive duration statistics"""
    stats = {}
    
    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker]
        durations_min = ticker_data['Trade Duration (min)']
        durations_hours = ticker_data['Trade Duration (hours)']
        
        # Basic statistics
        stats[ticker] = {
            'count': len(ticker_data),
            'avg_duration_min': durations_min.mean(),
            'median_duration_min': durations_min.median(),
            'min_duration_min': durations_min.min(),
            'max_duration_min': durations_min.max(),
            'std_duration_min': durations_min.std(),
            
            # Duration distribution buckets
            'under_60min': len(ticker_data[durations_min < 60]),
            'min_60_240': len(ticker_data[(durations_min >= 60) & (durations_min <= 240)]),  # 1-4 hours
            'over_240min': len(ticker_data[durations_min > 240]),  # > 4 hours
            'intraday_trades': len(ticker_data[ticker_data['Same Day Trade']]),
            'multiday_trades': len(ticker_data[~ticker_data['Same Day Trade']]),
            
            # Percentages
            'pct_under_60min': len(ticker_data[durations_min < 60]) / len(ticker_data) * 100,
            'pct_1_4_hours': len(ticker_data[(durations_min >= 60) & (durations_min <= 240)]) / len(ticker_data) * 100,
            'pct_over_4_hours': len(ticker_data[durations_min > 240]) / len(ticker_data) * 100,
            'pct_intraday': len(ticker_data[ticker_data['Same Day Trade']]) / len(ticker_data) * 100,
            'pct_multiday': len(ticker_data[~ticker_data['Same Day Trade']]) / len(ticker_data) * 100,
        }
    
    return stats

def analyze_daily_patterns(df):
    """Analyze daily trading patterns"""
    daily_stats = {}
    
    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker]
        
        # Trades per day
        daily_counts = ticker_data.groupby('Entry Date').size()
        
        # Consecutive same-direction trades analysis
        ticker_data_sorted = ticker_data.sort_values(['Entry Date', 'Entry Time'])
        consecutive_patterns = []
        
        dates = ticker_data_sorted['Entry Date'].unique()
        for date in dates[:10]:  # Sample first 10 dates
            day_trades = ticker_data_sorted[ticker_data_sorted['Entry Date'] == date]
            if len(day_trades) > 1:
                directions = day_trades['Trade Type'].tolist()
                times = day_trades['Entry Time'].dt.strftime('%H:%M').tolist()
                consecutive_patterns.append({
                    'date': date,
                    'trades': list(zip(directions, times)),
                    'count': len(day_trades)
                })
        
        daily_stats[ticker] = {
            'avg_trades_per_day': daily_counts.mean(),
            'median_trades_per_day': daily_counts.median(),
            'max_trades_per_day': daily_counts.max(),
            'min_trades_per_day': daily_counts.min(),
            'total_trading_days': len(daily_counts),
            'sample_consecutive_patterns': consecutive_patterns[:5]  # First 5 examples
        }
    
    return daily_stats

def analyze_cascade_patterns(df):
    """Analyze potential cascade patterns within and across days"""
    cascade_analysis = {}
    
    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker].sort_values('Entry Time')
        
        same_day_cascades = 0
        cross_day_cascades = 0
        cascade_examples = []
        
        # Look for consecutive trades in same direction
        for i in range(len(ticker_data) - 1):
            current_trade = ticker_data.iloc[i]
            next_trade = ticker_data.iloc[i + 1]
            
            # Check if same direction
            if current_trade['Trade Type'] == next_trade['Trade Type']:
                time_diff = (next_trade['Entry Time'] - current_trade['Entry Time']).total_seconds() / 60
                
                # Same day cascade (within 24 hours)
                if current_trade['Entry Date'] == next_trade['Entry Date']:
                    same_day_cascades += 1
                    cascade_type = 'same_day'
                else:
                    cross_day_cascades += 1
                    cascade_type = 'cross_day'
                
                # Collect examples for first 10 cascades
                if len(cascade_examples) < 10:
                    cascade_examples.append({
                        'type': cascade_type,
                        'direction': current_trade['Trade Type'],
                        'entry1': current_trade['Entry Time'].strftime('%Y-%m-%d %H:%M'),
                        'entry2': next_trade['Entry Time'].strftime('%Y-%m-%d %H:%M'),
                        'time_diff_min': time_diff,
                        'profit1': current_trade['Profit (%)'],
                        'profit2': next_trade['Profit (%)']
                    })
        
        cascade_analysis[ticker] = {
            'same_day_cascades': same_day_cascades,
            'cross_day_cascades': cross_day_cascades,
            'total_cascades': same_day_cascades + cross_day_cascades,
            'cascade_examples': cascade_examples,
            'pct_same_day_cascades': same_day_cascades / (same_day_cascades + cross_day_cascades) * 100 if (same_day_cascades + cross_day_cascades) > 0 else 0
        }
    
    return cascade_analysis

def print_comprehensive_report(df, duration_stats, daily_patterns, cascade_analysis):
    """Print comprehensive analysis report"""
    print("\n" + "="*80)
    print("COMPREHENSIVE TRADE DURATION ANALYSIS")
    print("="*80)
    
    # Overall summary
    print(f"\nDATA OVERVIEW:")
    print(f"Analysis Period: {df['Entry Time'].min().date()} to {df['Entry Time'].max().date()}")
    print(f"Total Trades Analyzed: {len(df)}")
    print(f"Tickers: {', '.join(df['ticker'].unique())}")
    
    # Duration analysis for each ticker
    print(f"\n1. TRADE DURATION ANALYSIS BY TICKER:")
    print("-" * 50)
    
    for ticker in sorted(duration_stats.keys()):
        stats = duration_stats[ticker]
        print(f"\n{ticker} ({stats['count']} trades):")
        print(f"  Average Duration: {stats['avg_duration_min']:.1f} minutes ({stats['avg_duration_min']/60:.1f} hours)")
        print(f"  Median Duration:  {stats['median_duration_min']:.1f} minutes ({stats['median_duration_min']/60:.1f} hours)")
        print(f"  Min Duration:     {stats['min_duration_min']:.0f} minutes")
        print(f"  Max Duration:     {stats['max_duration_min']:.0f} minutes ({stats['max_duration_min']/60:.1f} hours)")
        print(f"  Std Deviation:    {stats['std_duration_min']:.1f} minutes")
        
        print(f"\n  Duration Distribution:")
        print(f"    < 60 minutes:     {stats['under_60min']:4d} trades ({stats['pct_under_60min']:.1f}%)")
        print(f"    1-4 hours:        {stats['min_60_240']:4d} trades ({stats['pct_1_4_hours']:.1f}%)")
        print(f"    > 4 hours:        {stats['over_240min']:4d} trades ({stats['pct_over_4_hours']:.1f}%)")
        
        print(f"\n  Intraday vs Multi-day:")
        print(f"    Same-day trades:  {stats['intraday_trades']:4d} trades ({stats['pct_intraday']:.1f}%)")
        print(f"    Multi-day trades: {stats['multiday_trades']:4d} trades ({stats['pct_multiday']:.1f}%)")
    
    # Daily patterns
    print(f"\n2. DAILY TRADING PATTERNS:")
    print("-" * 30)
    
    for ticker in sorted(daily_patterns.keys()):
        patterns = daily_patterns[ticker]
        print(f"\n{ticker}:")
        print(f"  Trading Days: {patterns['total_trading_days']}")
        print(f"  Avg Trades/Day: {patterns['avg_trades_per_day']:.1f}")
        print(f"  Median Trades/Day: {patterns['median_trades_per_day']:.1f}")
        print(f"  Max Trades/Day: {patterns['max_trades_per_day']}")
        print(f"  Min Trades/Day: {patterns['min_trades_per_day']}")
        
        # Sample consecutive patterns
        if patterns['sample_consecutive_patterns']:
            print(f"\n  Sample Daily Trade Sequences:")
            for pattern in patterns['sample_consecutive_patterns'][:3]:
                trades_str = ', '.join([f"{direction}@{time}" for direction, time in pattern['trades']])
                print(f"    {pattern['date']}: {trades_str} ({pattern['count']} trades)")
    
    # Cascade analysis
    print(f"\n3. CASCADE PATTERN ANALYSIS:")
    print("-" * 35)
    
    for ticker in sorted(cascade_analysis.keys()):
        cascade = cascade_analysis[ticker]
        print(f"\n{ticker}:")
        print(f"  Total Consecutive Same-Direction Trades: {cascade['total_cascades']}")
        print(f"  Same-day Cascades:  {cascade['same_day_cascades']} ({cascade['pct_same_day_cascades']:.1f}%)")
        print(f"  Cross-day Cascades: {cascade['cross_day_cascades']} ({100-cascade['pct_same_day_cascades']:.1f}%)")
        
        if cascade['cascade_examples']:
            print(f"\n  Sample Cascade Sequences:")
            for example in cascade['cascade_examples'][:5]:
                print(f"    {example['type'].upper()}: {example['direction']} trades")
                print(f"      {example['entry1']} → {example['entry2']} ({example['time_diff_min']:.0f}min gap)")
                print(f"      P&L: {example['profit1']:.2f}% → {example['profit2']:.2f}%")
    
    # Overall insights
    print(f"\n4. KEY INSIGHTS:")
    print("-" * 15)
    
    # Calculate overall statistics
    avg_durations = {ticker: stats['avg_duration_min'] for ticker, stats in duration_stats.items()}
    longest_avg = max(avg_durations, key=avg_durations.get)
    shortest_avg = min(avg_durations, key=avg_durations.get)
    
    intraday_pcts = {ticker: stats['pct_intraday'] for ticker, stats in duration_stats.items()}
    most_intraday = max(intraday_pcts, key=intraday_pcts.get)
    least_intraday = min(intraday_pcts, key=intraday_pcts.get)
    
    total_same_day_cascades = sum(c['same_day_cascades'] for c in cascade_analysis.values())
    total_cross_day_cascades = sum(c['cross_day_cascades'] for c in cascade_analysis.values())
    
    print(f"\n• TYPICAL HOLDING PERIODS:")
    print(f"  Longest average hold: {longest_avg} ({avg_durations[longest_avg]:.1f} min / {avg_durations[longest_avg]/60:.1f} hours)")
    print(f"  Shortest average hold: {shortest_avg} ({avg_durations[shortest_avg]:.1f} min / {avg_durations[shortest_avg]/60:.1f} hours)")
    
    print(f"\n• INTRADAY vs MULTI-DAY TRADING:")
    print(f"  Most intraday-focused: {most_intraday} ({intraday_pcts[most_intraday]:.1f}% same-day)")
    print(f"  Least intraday-focused: {least_intraday} ({intraday_pcts[least_intraday]:.1f}% same-day)")
    
    print(f"\n• CASCADE PROBLEM ANALYSIS:")
    print(f"  Total consecutive same-direction trades: {total_same_day_cascades + total_cross_day_cascades}")
    print(f"  Same-day cascades: {total_same_day_cascades} ({total_same_day_cascades/(total_same_day_cascades + total_cross_day_cascades)*100:.1f}%)")
    print(f"  Cross-day cascades: {total_cross_day_cascades} ({total_cross_day_cascades/(total_same_day_cascades + total_cross_day_cascades)*100:.1f}%)")
    
    if total_same_day_cascades > total_cross_day_cascades:
        print("  → CASCADE ISSUE IS PRIMARILY INTRADAY - multiple trades in same direction within same trading day")
    else:
        print("  → CASCADE ISSUE IS PRIMARILY CROSS-DAY - consecutive trades spanning multiple days")
    
    # Duration recommendations
    overall_avg_duration = df['Trade Duration (min)'].mean()
    overall_intraday_pct = (df['Same Day Trade'].sum() / len(df)) * 100
    
    print(f"\n• OVERALL PATTERNS:")
    print(f"  Average trade duration across all tickers: {overall_avg_duration:.1f} minutes ({overall_avg_duration/60:.1f} hours)")
    print(f"  Percentage of intraday trades: {overall_intraday_pct:.1f}%")
    print(f"  Most common duration range: ", end="")
    
    total_under_60 = sum(stats['under_60min'] for stats in duration_stats.values())
    total_1_4_hours = sum(stats['min_60_240'] for stats in duration_stats.values())
    total_over_4_hours = sum(stats['over_240min'] for stats in duration_stats.values())
    
    if total_under_60 > total_1_4_hours and total_under_60 > total_over_4_hours:
        print("< 60 minutes (quick scalp trades)")
    elif total_1_4_hours > total_over_4_hours:
        print("1-4 hours (intraday swing trades)")
    else:
        print("> 4 hours (position/swing trades)")

def main():
    """Main analysis function"""
    file_path = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/back_tester/backtester/consolidated_trades.csv"
    target_tickers = ['TATASTEEL', 'JSWSTEEL', 'BIOCON', 'KOTAKBANK']
    
    # Load and process data
    df = load_and_filter_data(file_path, target_tickers)
    
    # Perform analyses
    duration_stats = calculate_duration_stats(df)
    daily_patterns = analyze_daily_patterns(df)
    cascade_analysis = analyze_cascade_patterns(df)
    
    # Print comprehensive report
    print_comprehensive_report(df, duration_stats, daily_patterns, cascade_analysis)
    
    # Additional summary statistics
    print(f"\n5. DETAILED STATISTICS TABLE:")
    print("-" * 35)
    print(f"{'Ticker':<12} {'Trades':<8} {'Avg(min)':<10} {'Med(min)':<10} {'Intraday%':<12} {'Cascades':<10}")
    print("-" * 70)
    
    for ticker in sorted(target_tickers):
        if ticker in duration_stats:
            stats = duration_stats[ticker]
            cascades = cascade_analysis[ticker]
            print(f"{ticker:<12} {stats['count']:<8} {stats['avg_duration_min']:<10.1f} {stats['median_duration_min']:<10.1f} {stats['pct_intraday']:<12.1f} {cascades['total_cascades']:<10}")

if __name__ == "__main__":
    main()