#!/usr/bin/env python3
"""
Detailed timing analysis to understand the cascade problem better
"""

import pandas as pd
from datetime import datetime, timedelta

def analyze_timing_details(file_path, target_tickers):
    """Detailed timing analysis"""
    df = pd.read_csv(file_path)
    df_filtered = df[df['ticker'].isin(target_tickers)].copy()
    
    # Convert datetime columns
    df_filtered['Entry Time'] = pd.to_datetime(df_filtered['Entry Time'])
    df_filtered['Exit Time'] = pd.to_datetime(df_filtered['Exit Time'])
    df_filtered['Entry Date'] = df_filtered['Entry Time'].dt.date
    df_filtered['Entry Hour'] = df_filtered['Entry Time'].dt.hour
    
    print("DETAILED TIMING ANALYSIS")
    print("=" * 50)
    
    # 1. Hourly distribution of trade entries
    print("\n1. HOURLY DISTRIBUTION OF TRADE ENTRIES:")
    print("-" * 45)
    
    for ticker in sorted(target_tickers):
        ticker_data = df_filtered[df_filtered['ticker'] == ticker]
        hourly_counts = ticker_data.groupby('Entry Hour').size().sort_index()
        
        print(f"\n{ticker}:")
        peak_hour = hourly_counts.idxmax()
        peak_count = hourly_counts.max()
        print(f"  Peak trading hour: {peak_hour}:00 ({peak_count} trades)")
        print(f"  Distribution: ", end="")
        
        # Show top 5 hours
        top_hours = hourly_counts.nlargest(5)
        for hour, count in top_hours.items():
            print(f"{hour}h({count}) ", end="")
        print()
    
    # 2. Same-day cascade timing analysis
    print("\n\n2. SAME-DAY CASCADE TIMING PATTERNS:")
    print("-" * 42)
    
    for ticker in sorted(target_tickers):
        ticker_data = df_filtered[df_filtered['ticker'] == ticker].sort_values('Entry Time')
        same_day_gaps = []
        
        for i in range(len(ticker_data) - 1):
            current = ticker_data.iloc[i]
            next_trade = ticker_data.iloc[i + 1]
            
            # Same day, same direction
            if (current['Entry Date'] == next_trade['Entry Date'] and 
                current['Trade Type'] == next_trade['Trade Type']):
                gap_minutes = (next_trade['Entry Time'] - current['Entry Time']).total_seconds() / 60
                same_day_gaps.append(gap_minutes)
        
        if same_day_gaps:
            print(f"\n{ticker} - Same-day cascade gaps:")
            print(f"  Average gap: {sum(same_day_gaps)/len(same_day_gaps):.1f} minutes")
            print(f"  Median gap:  {sorted(same_day_gaps)[len(same_day_gaps)//2]:.1f} minutes")
            print(f"  Quick cascades (< 30min): {sum(1 for gap in same_day_gaps if gap < 30)}")
            print(f"  Medium cascades (30-120min): {sum(1 for gap in same_day_gaps if 30 <= gap <= 120)}")
            print(f"  Delayed cascades (> 120min): {sum(1 for gap in same_day_gaps if gap > 120)}")
    
    # 3. Multi-day patterns
    print("\n\n3. MULTI-DAY TRADE PATTERNS:")
    print("-" * 32)
    
    for ticker in sorted(target_tickers):
        ticker_data = df_filtered[df_filtered['ticker'] == ticker]
        multiday_trades = ticker_data[ticker_data['Entry Date'] != ticker_data['Exit Time'].dt.date]
        
        print(f"\n{ticker}:")
        print(f"  Multi-day trades: {len(multiday_trades)} ({len(multiday_trades)/len(ticker_data)*100:.1f}%)")
        
        if len(multiday_trades) > 0:
            durations = multiday_trades['Trade Duration (min)'] / (60 * 24)  # Convert to days
            print(f"  Average duration: {durations.mean():.1f} days")
            print(f"  Longest trade: {durations.max():.1f} days")
            
            # Weekend effect
            weekend_entries = multiday_trades[multiday_trades['Entry Time'].dt.dayofweek >= 5]  # Sat/Sun
            if len(weekend_entries) > 0:
                print(f"  Weekend entries: {len(weekend_entries)} trades")
    
    # 4. Profitable vs Losing trade timing
    print("\n\n4. PROFITABLE vs LOSING TRADE TIMING:")
    print("-" * 40)
    
    for ticker in sorted(target_tickers):
        ticker_data = df_filtered[df_filtered['ticker'] == ticker]
        profitable = ticker_data[ticker_data['Profit (%)'] > 0]
        losing = ticker_data[ticker_data['Profit (%)'] <= 0]
        
        print(f"\n{ticker}:")
        print(f"  Profitable trades: {len(profitable)} ({len(profitable)/len(ticker_data)*100:.1f}%)")
        print(f"    Avg duration: {profitable['Trade Duration (min)'].mean():.0f} min")
        print(f"    Avg same-day: {(profitable['Entry Date'] == profitable['Exit Time'].dt.date).sum()/len(profitable)*100:.1f}%")
        
        print(f"  Losing trades: {len(losing)} ({len(losing)/len(ticker_data)*100:.1f}%)")
        print(f"    Avg duration: {losing['Trade Duration (min)'].mean():.0f} min")
        print(f"    Avg same-day: {(losing['Entry Date'] == losing['Exit Time'].dt.date).sum()/len(losing)*100:.1f}%")

    # 5. Sample of problematic cascade sequences
    print("\n\n5. SAMPLE PROBLEMATIC CASCADE SEQUENCES:")
    print("-" * 45)
    
    for ticker in sorted(target_tickers):
        ticker_data = df_filtered[df_filtered['ticker'] == ticker].sort_values('Entry Time')
        
        print(f"\n{ticker} - Consecutive losing trades:")
        losing_sequences = []
        current_sequence = []
        
        for _, trade in ticker_data.iterrows():
            if trade['Profit (%)'] <= 0:
                current_sequence.append(trade)
            else:
                if len(current_sequence) >= 2:  # 2+ consecutive losses
                    losing_sequences.append(current_sequence.copy())
                current_sequence = []
        
        # Add final sequence if it exists
        if len(current_sequence) >= 2:
            losing_sequences.append(current_sequence)
        
        # Show first few sequences
        for i, sequence in enumerate(losing_sequences[:3]):
            total_loss = sum(trade['Profit (%)'] for trade in sequence)
            start_time = sequence[0]['Entry Time'].strftime('%Y-%m-%d %H:%M')
            end_time = sequence[-1]['Exit Time'].strftime('%Y-%m-%d %H:%M')
            
            print(f"  Sequence {i+1}: {len(sequence)} consecutive losses")
            print(f"    Period: {start_time} to {end_time}")
            print(f"    Total loss: {total_loss:.2f}%")
            print(f"    Trades: ", end="")
            for trade in sequence:
                print(f"{trade['Trade Type']}({trade['Profit (%)']:.1f}%) ", end="")
            print()

def main():
    file_path = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/back_tester/backtester/consolidated_trades.csv"
    target_tickers = ['TATASTEEL', 'JSWSTEEL', 'BIOCON', 'KOTAKBANK']
    
    analyze_timing_details(file_path, target_tickers)

if __name__ == "__main__":
    main()