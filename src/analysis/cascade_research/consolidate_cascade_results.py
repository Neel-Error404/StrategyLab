#!/usr/bin/env python3
"""
Consolidate CASCADE PREVENTION backtest results into single file for comparison
with original consolidated_trades.csv
"""

import pandas as pd
import glob
import os

def consolidate_cascade_results():
    """Consolidate all CASCADE PREVENTION strategy trades into one file"""
    
    # Path to the strategy trades folder
    strategy_trades_path = "/path/to/outputs/20250829_230157/open_source_baseline/2022-01-01_to_2025-07-07/data/strategy_trades/"
    
    # Get all CSV files
    csv_files = glob.glob(os.path.join(strategy_trades_path, "*.csv"))
    
    print(f"Found {len(csv_files)} CSV files to consolidate")
    
    all_trades = []
    
    for csv_file in csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Extract ticker name from filename
            filename = os.path.basename(csv_file)
            ticker = filename.split('_')[0]
            
            # Add ticker column if not exists
            if 'ticker' not in df.columns:
                df['ticker'] = ticker
                
            print(f"Processing {ticker}: {len(df)} trades")
            all_trades.append(df)
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    if all_trades:
        # Concatenate all dataframes
        consolidated_df = pd.concat(all_trades, ignore_index=True)
        
        # Sort by Entry Time for chronological order
        consolidated_df = consolidated_df.sort_values('Entry Time').reset_index(drop=True)
        
        # Save consolidated file
        output_path = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/back_tester/backtester/cascade_prevention_trades.csv"
        consolidated_df.to_csv(output_path, index=False)
        
        print(f"\n✅ SUCCESS: Consolidated {len(consolidated_df)} trades from {len(all_trades)} tickers")
        print(f"📁 Output file: {output_path}")
        
        # Summary statistics
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"Total trades: {len(consolidated_df)}")
        print(f"Date range: {consolidated_df['Entry Time'].min()} to {consolidated_df['Entry Time'].max()}")
        print(f"Tickers: {sorted(consolidated_df['ticker'].unique())}")
        print(f"Trade types: {consolidated_df['Trade Type'].value_counts().to_dict()}")
        
        # Performance metrics
        total_pnl = consolidated_df['Profit (%)'].sum()
        avg_profit = consolidated_df['Profit (%)'].mean()
        profitable_trades = (consolidated_df['Profit (%)'] > 0).sum()
        total_trades = len(consolidated_df)
        win_rate = profitable_trades / total_trades * 100
        
        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"Total P&L: {total_pnl:.2f}%")
        print(f"Average profit per trade: {avg_profit:.3f}%")
        print(f"Win rate: {win_rate:.1f}% ({profitable_trades}/{total_trades})")
        
        return output_path
    else:
        print("❌ ERROR: No valid trade files found")
        return None

if __name__ == "__main__":
    consolidate_cascade_results()
