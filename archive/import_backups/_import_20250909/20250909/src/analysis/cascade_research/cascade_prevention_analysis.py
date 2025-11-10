#!/usr/bin/env python3
"""
Comprehensive analysis comparing CASCADE PREVENTION results vs original trades
for the same 30 tickers to measure improvement
"""

import pandas as pd
import numpy as np
from datetime import datetime

def analyze_cascade_improvement():
    """Compare CASCADE PREVENTION results with original trades for same tickers"""
    
    print("🔍 CASCADE PREVENTION IMPROVEMENT ANALYSIS")
    print("=" * 60)
    
    # Load datasets
    original_file = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/back_tester/backtester/consolidated_trades.csv"
    cascade_file = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/back_tester/backtester/cascade_prevention_trades.csv"
    
    print("📁 Loading datasets...")
    original_df = pd.read_csv(original_file)
    cascade_df = pd.read_csv(cascade_file)
    
    # Get the 30 tickers used in CASCADE PREVENTION test
    cascade_tickers = set(cascade_df['ticker'].unique())
    print(f"🎯 Analyzing {len(cascade_tickers)} tickers: {sorted(cascade_tickers)}")
    
    # Filter original data to same tickers for fair comparison
    original_subset = original_df[original_df['ticker'].isin(cascade_tickers)].copy()
    
    print(f"\n📊 DATASET COMPARISON:")
    print(f"Original trades (30 tickers): {len(original_subset):,}")
    print(f"CASCADE PREVENTION trades:    {len(cascade_df):,}")
    print(f"Trades eliminated:           {len(original_subset) - len(cascade_df):,} ({(len(original_subset) - len(cascade_df))/len(original_subset)*100:.1f}%)")
    
    # Performance Analysis
    print(f"\n🎯 PERFORMANCE COMPARISON:")
    print("-" * 40)
    
    # Original performance
    orig_total_pnl = original_subset['Profit (%)'].sum()
    orig_avg_profit = original_subset['Profit (%)'].mean()
    orig_profitable = (original_subset['Profit (%)'] > 0).sum()
    orig_win_rate = orig_profitable / len(original_subset) * 100
    
    # CASCADE PREVENTION performance  
    casc_total_pnl = cascade_df['Profit (%)'].sum()
    casc_avg_profit = cascade_df['Profit (%)'].mean()
    casc_profitable = (cascade_df['Profit (%)'] > 0).sum()
    casc_win_rate = casc_profitable / len(cascade_df) * 100
    
    print(f"📈 ORIGINAL STRATEGY:")
    print(f"  Total P&L:        {orig_total_pnl:,.2f}%")
    print(f"  Avg per trade:    {orig_avg_profit:.3f}%")
    print(f"  Win rate:         {orig_win_rate:.1f}% ({orig_profitable:,}/{len(original_subset):,})")
    
    print(f"\n🚀 CASCADE PREVENTION:")
    print(f"  Total P&L:        {casc_total_pnl:,.2f}%")
    print(f"  Avg per trade:    {casc_avg_profit:.3f}%") 
    print(f"  Win rate:         {casc_win_rate:.1f}% ({casc_profitable:,}/{len(cascade_df):,})")
    
    # Calculate improvements
    pnl_improvement = casc_total_pnl - orig_total_pnl
    avg_improvement = casc_avg_profit - orig_avg_profit
    win_rate_improvement = casc_win_rate - orig_win_rate
    
    print(f"\n✨ IMPROVEMENTS:")
    print(f"  Total P&L gain:   {pnl_improvement:+,.2f}% ({pnl_improvement/abs(orig_total_pnl)*100:+.1f}%)")
    print(f"  Avg per trade:    {avg_improvement:+.3f}% ({avg_improvement/abs(orig_avg_profit)*100:+.1f}%)")
    print(f"  Win rate gain:    {win_rate_improvement:+.1f} percentage points")
    
    # Cascade Analysis
    print(f"\n🔄 CASCADE PROBLEM ANALYSIS:")
    print("-" * 35)
    
    def analyze_cascades(df, name):
        """Analyze consecutive same-direction trades"""
        cascades = 0
        same_day_cascades = 0
        cross_day_cascades = 0
        
        # Convert Entry Time to datetime
        df['Entry Time'] = pd.to_datetime(df['Entry Time'])
        df['Entry Date'] = df['Entry Time'].dt.date
        
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker].sort_values('Entry Time')
            
            for i in range(len(ticker_data) - 1):
                current = ticker_data.iloc[i]
                next_trade = ticker_data.iloc[i + 1]
                
                if current['Trade Type'] == next_trade['Trade Type']:
                    cascades += 1
                    if current['Entry Date'] == next_trade['Entry Date']:
                        same_day_cascades += 1
                    else:
                        cross_day_cascades += 1
        
        return cascades, same_day_cascades, cross_day_cascades
    
    orig_cascades, orig_same_day, orig_cross_day = analyze_cascades(original_subset, "Original")
    casc_cascades, casc_same_day, casc_cross_day = analyze_cascades(cascade_df, "CASCADE PREVENTION")
    
    print(f"📊 ORIGINAL STRATEGY CASCADES:")
    print(f"  Total cascades:    {orig_cascades:,}")
    print(f"  Same-day:          {orig_same_day:,} ({orig_same_day/orig_cascades*100:.1f}%)")
    print(f"  Cross-day:         {orig_cross_day:,} ({orig_cross_day/orig_cascades*100:.1f}%)")
    print(f"  Cascade rate:      {orig_cascades/len(original_subset)*100:.1f}%")
    
    print(f"\n🛡️ CASCADE PREVENTION CASCADES:")
    print(f"  Total cascades:    {casc_cascades:,}")
    print(f"  Same-day:          {casc_same_day:,} ({casc_same_day/casc_cascades*100:.1f}% if any)")
    print(f"  Cross-day:         {casc_cross_day:,} ({casc_cross_day/casc_cascades*100:.1f}%)")
    print(f"  Cascade rate:      {casc_cascades/len(cascade_df)*100:.1f}%")
    
    cascades_eliminated = orig_cascades - casc_cascades
    same_day_eliminated = orig_same_day - casc_same_day
    
    print(f"\n🎯 CASCADE ELIMINATION:")
    print(f"  Total eliminated:  {cascades_eliminated:,} ({cascades_eliminated/orig_cascades*100:.1f}%)")
    print(f"  Same-day elim:     {same_day_eliminated:,} ({same_day_eliminated/orig_same_day*100:.1f}%)")
    print(f"  SUCCESS: {same_day_eliminated/orig_same_day*100:.1f}% of same-day cascades eliminated!")
    
    # Risk Analysis
    print(f"\n📉 RISK ANALYSIS:")
    print("-" * 20)
    
    orig_losses = original_subset[original_subset['Profit (%)'] < 0]
    casc_losses = cascade_df[cascade_df['Profit (%)'] < 0]
    
    orig_avg_loss = orig_losses['Profit (%)'].mean() if len(orig_losses) > 0 else 0
    casc_avg_loss = casc_losses['Profit (%)'].mean() if len(casc_losses) > 0 else 0
    
    orig_max_loss = original_subset['Profit (%)'].min()
    casc_max_loss = cascade_df['Profit (%)'].min()
    
    print(f"Average loss per losing trade:")
    print(f"  Original:     {orig_avg_loss:.3f}%")
    print(f"  With CASCADE: {casc_avg_loss:.3f}%")
    print(f"  Improvement:  {casc_avg_loss - orig_avg_loss:+.3f}%")
    
    print(f"\nMaximum single trade loss:")
    print(f"  Original:     {orig_max_loss:.3f}%") 
    print(f"  With CASCADE: {casc_max_loss:.3f}%")
    print(f"  Improvement:  {casc_max_loss - orig_max_loss:+.3f}%")
    
    # Annualized Performance Estimate
    print(f"\n💰 ANNUALIZED IMPACT ESTIMATE:")
    print("-" * 35)
    
    # Assume same date range for both datasets
    orig_start = pd.to_datetime(original_subset['Entry Time']).min()
    orig_end = pd.to_datetime(original_subset['Entry Time']).max()
    years = (orig_end - orig_start).days / 365.25
    
    annual_pnl_improvement = pnl_improvement / years
    
    print(f"Analysis period:     {years:.1f} years")
    print(f"Annual P&L gain:     {annual_pnl_improvement:+.2f}%")
    
    # If we assume this represents a $1M portfolio
    portfolio_value = 1000000  # $1M
    annual_dollar_improvement = (annual_pnl_improvement / 100) * portfolio_value
    
    print(f"On $1M portfolio:    ${annual_dollar_improvement:+,.0f} per year")
    
    # Summary
    print(f"\n🏆 EXECUTIVE SUMMARY:")
    print("=" * 25)
    print(f"✅ CASCADE PREVENTION successfully eliminated {same_day_eliminated/orig_same_day*100:.1f}% of same-day cascades")
    print(f"✅ Improved average profit per trade by {avg_improvement/abs(orig_avg_profit)*100:+.1f}%")
    print(f"✅ Increased win rate by {win_rate_improvement:+.1f} percentage points")
    print(f"✅ Reduced total trades by {(len(original_subset) - len(cascade_df))/len(original_subset)*100:.1f}% (quality over quantity)")
    print(f"✅ Estimated annual improvement: ${annual_dollar_improvement:+,.0f} on $1M portfolio")
    
    return {
        'original_trades': len(original_subset),
        'cascade_trades': len(cascade_df), 
        'trades_eliminated': len(original_subset) - len(cascade_df),
        'pnl_improvement': pnl_improvement,
        'cascades_eliminated': cascades_eliminated,
        'annual_dollar_improvement': annual_dollar_improvement
    }

if __name__ == "__main__":
    results = analyze_cascade_improvement()