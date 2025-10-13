#!/usr/bin/env python3
"""
Script to merge all individual ticker trade CSV files into a single consolidated CSV.
"""

import pandas as pd
import glob
import os
from pathlib import Path

def merge_trade_files():
    """Merge all ticker trade CSV files into a consolidated CSV."""

    # Define paths
    source_dir = "outputs/20250915_121714/mse_backtesting/2022-01-01_to_2025-08-31/data/strategy_trades/"
    output_dir = "outputs/20250915_121714/mse_backtesting/2022-01-01_to_2025-08-31/data/"
    output_file = os.path.join(output_dir, "all_trade_mereged.csv")

    # Find all CSV files
    csv_pattern = os.path.join(source_dir, "*_StrategyTrades_*.csv")
    csv_files = glob.glob(csv_pattern)

    print(f"Found {len(csv_files)} CSV files to merge...")

    if not csv_files:
        print("No CSV files found to merge!")
        return

    # Initialize list to store dataframes
    dataframes = []

    # Read and combine all CSV files
    for i, file_path in enumerate(csv_files, 1):
        try:
            df = pd.read_csv(file_path)
            dataframes.append(df)

            if i % 50 == 0:
                print(f"Processed {i}/{len(csv_files)} files...")

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    if not dataframes:
        print("No valid CSV files could be read!")
        return

    # Concatenate all dataframes
    print("Concatenating all dataframes...")
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Sort by entry time for better organization
    print("Sorting trades by entry time...")
    merged_df['Entry Time'] = pd.to_datetime(merged_df['Entry Time'])
    merged_df = merged_df.sort_values('Entry Time').reset_index(drop=True)

    # Save merged file
    print(f"Saving merged file to {output_file}...")
    merged_df.to_csv(output_file, index=False)

    # Print summary statistics
    print("\n" + "="*50)
    print("MERGE SUMMARY")
    print("="*50)
    print(f"Total files processed: {len(csv_files)}")
    print(f"Total trades: {len(merged_df):,}")
    print(f"Date range: {merged_df['Entry Time'].min()} to {merged_df['Entry Time'].max()}")
    print(f"Unique tickers: {merged_df['ticker'].nunique()}")
    print(f"Output file: {output_file}")
    print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")

    # Show trade type distribution
    print("\nTrade Type Distribution:")
    print(merged_df['Trade Type'].value_counts())

    # Show top tickers by trade count
    print("\nTop 10 Tickers by Trade Count:")
    print(merged_df['ticker'].value_counts().head(10))

    print("\nMerge completed successfully!")

if __name__ == "__main__":
    merge_trade_files()