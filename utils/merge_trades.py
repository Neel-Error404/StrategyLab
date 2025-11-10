#!/usr/bin/env python3
"""
Trade File Merger - YAML Config Driven
=======================================

Merges individual ticker trade CSV files into a single consolidated file.
Uses YAML configuration for flexible, reusable workflows.

Usage:
    python utils/merge_trades.py --config analysis/config.yaml

    # Or run from analysis directory
    cd analysis
    python ../utils/merge_trades.py --config config.yaml

Author: StrategyLab Team
Version: 2.0 - YAML Config Driven
"""

import pandas as pd
import glob
import os
import yaml
import argparse
from pathlib import Path
from datetime import datetime

def load_config(config_path):
    """Load YAML configuration file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"[OK] Loaded config from: {config_path}")
        return config
    except FileNotFoundError:
        print(f"[ERROR] Config file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        print(f"[ERROR] Error parsing YAML: {e}")
        raise

def resolve_paths(config):
    """
    Resolve paths from config variables.

    Priority:
    1. Use data_sources paths if available (for cross-workspace analysis)
    2. Fall back to constructing paths from run metadata (for local analysis)
    
    Supports template variables like:
        {run_id}, {strategy}, {date_range}
    """
    run_id = config['run']['run_id']
    strategy = config['run']['strategy']
    date_range = config['run']['date_range']
    trade_source = config['run']['trade_source']

    # Check if data_sources section exists (for cross-workspace support)
    if 'data_sources' in config and config['data_sources'].get('strategy_trades_dir'):
        # Use explicit data_sources paths
        if trade_source == 'strategy_trades':
            source_dir = config['data_sources']['strategy_trades_dir']
            file_pattern = "*_StrategyTrades_*.csv"
        elif trade_source == 'risk_approved_trades':
            source_dir = config['data_sources'].get('risk_approved_trades_dir', '')
            file_pattern = "*_RiskApprovedTrades_*.csv"
        else:
            raise ValueError(f"Invalid trade_source: {trade_source}")
        
        base_data_dir = config['data_sources'].get('base_data_dir', '')
        outputs_base = os.path.dirname(os.path.dirname(source_dir))  # Go up two levels
    else:
        # Construct paths from run metadata (backward compatibility)
        outputs_base = f"outputs/{run_id}/{strategy}/{date_range}"

        # Determine source directory based on trade_source
        if trade_source == 'strategy_trades':
            source_dir = f"{outputs_base}/data/strategy_trades"
            file_pattern = "*_StrategyTrades_*.csv"
        elif trade_source == 'risk_approved_trades':
            source_dir = f"{outputs_base}/data/risk_approved_trades"
            file_pattern = "*_RiskApprovedTrades_*.csv"
        else:
            raise ValueError(f"Invalid trade_source: {trade_source}")

        base_data_dir = f"{outputs_base}/data/base_data"

    # Output merged file path (use output config)
    output_filename = config['output']['merged_filename']
    output_root = config['output']['root_dir']
    output_dir = f"{output_root}/{strategy}/{run_id}/data"
    output_path = f"{output_dir}/{output_filename}"

    paths = {
        'source_dir': source_dir,
        'file_pattern': file_pattern,
        'output_path': output_path,
        'base_data_dir': base_data_dir,
        'outputs_base': outputs_base
    }

    return paths

def merge_trade_files(paths, config):
    """Merge all individual ticker trade CSV files"""

    source_dir = os.path.normpath(paths['source_dir'])
    pattern = paths['file_pattern']
    output_file = paths['output_path']

    # Find all CSV files
    csv_pattern = os.path.join(source_dir, pattern)
    csv_files = glob.glob(csv_pattern)

    print(f"\n{'='*60}")
    print(f"TRADE FILE MERGER")
    print(f"{'='*60}")
    print(f"Source: {source_dir}")
    print(f"Pattern: {pattern}")
    print(f"Found {len(csv_files)} files to merge")

    if not csv_files:
        print(f"\n[ERROR] No CSV files found matching pattern!")
        print(f"   Directory: {source_dir}")
        print(f"   Pattern: {pattern}")
        return None

    # Show first few files
    print(f"\nSample files:")
    for f in csv_files[:5]:
        print(f"  - {os.path.basename(f)}")
    if len(csv_files) > 5:
        print(f"  ... and {len(csv_files) - 5} more")

    # Read and combine all CSV files
    dataframes = []
    errors = []

    print(f"\nReading files...")
    for i, file_path in enumerate(csv_files, 1):
        try:
            df = pd.read_csv(file_path)
            dataframes.append(df)

            if i % 50 == 0 or i == len(csv_files):
                print(f"  Processed {i}/{len(csv_files)} files...")
        except Exception as e:
            errors.append((file_path, str(e)))
            print(f"  [WARNING] Error reading {os.path.basename(file_path)}: {e}")
            continue

    if not dataframes:
        print(f"\n[ERROR] No valid CSV files could be read!")
        return None

    # Concatenate all dataframes
    print(f"\nMerging dataframes...")
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Sort by entry time for better organization
    print(f"Sorting by entry time...")
    merged_df['Entry Time'] = pd.to_datetime(merged_df['Entry Time'])
    merged_df = merged_df.sort_values('Entry Time').reset_index(drop=True)

    # Save merged file
    print(f"Saving to: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged_df.to_csv(output_file, index=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"MERGE SUMMARY")
    print(f"{'='*60}")
    print(f"[OK] Files processed: {len(csv_files)}")
    if errors:
        print(f"[WARNING] Files with errors: {len(errors)}")
    print(f"[OK] Total trades: {len(merged_df):,}")
    print(f"Date range: {merged_df['Entry Time'].min()} to {merged_df['Entry Time'].max()}")
    print(f"Unique tickers: {merged_df['ticker'].nunique()}")
    print(f"Output file: {output_file}")
    print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")

    # Trade type distribution
    if 'Trade Type' in merged_df.columns:
        print(f"\nTrade Type Distribution:")
        for trade_type, count in merged_df['Trade Type'].value_counts().items():
            print(f"   {trade_type}: {count:,} ({count/len(merged_df)*100:.1f}%)")

    # Top tickers
    print(f"\nTop 10 Tickers by Trade Count:")
    for ticker, count in merged_df['ticker'].value_counts().head(10).items():
        print(f"   {ticker}: {count:,}")

    # Profitability summary
    if 'Profit (Currency)' in merged_df.columns:
        total_profit = merged_df['Profit (Currency)'].sum()
        winning_trades = (merged_df['Profit (Currency)'] > 0).sum()
        win_rate = winning_trades / len(merged_df) * 100

        print(f"\nProfitability Summary:")
        print(f"   Total P&L: Rs {total_profit:,.2f}")
        print(f"   Winning Trades: {winning_trades:,} ({win_rate:.1f}%)")
        print(f"   Losing Trades: {len(merged_df) - winning_trades:,} ({100-win_rate:.1f}%)")

    print(f"\n[SUCCESS] Merge completed successfully!")
    print(f"{'='*60}\n")

    # Update config with merged file path (for next steps)
    config['paths'] = {
        'merged_trades_file': output_file,
        'base_data_dir': paths['base_data_dir'],
        'outputs_base': paths['outputs_base']
    }

    return merged_df, config

def save_updated_config(config, config_path):
    """Save updated config with resolved paths"""
    # Create a copy for saving
    updated_config = config.copy()

    # Add timestamp of merge
    updated_config['merge_info'] = {
        'merged_at': datetime.now().isoformat(),
        'merged_file': config['paths']['merged_trades_file']
    }

    # Save to same directory as config
    config_dir = os.path.dirname(config_path)
    updated_config_path = os.path.join(config_dir, 'config_with_paths.yaml')

    with open(updated_config_path, 'w') as f:
        yaml.dump(updated_config, f, default_flow_style=False, sort_keys=False)

    print(f"Updated config saved to: {updated_config_path}")
    print(f"   (Contains resolved paths for next analysis steps)")

def main():
    parser = argparse.ArgumentParser(
        description="Merge individual ticker trade CSV files using YAML config"
    )
    parser.add_argument(
        '--config',
        required=True,
        help="Path to YAML config file (e.g., analysis/config.yaml)"
    )
    parser.add_argument(
        '--save-config',
        action='store_true',
        help="Save updated config with resolved paths"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Resolve paths from config
    paths = resolve_paths(config)

    # Merge files
    result = merge_trade_files(paths, config)

    if result is None:
        print("\n[ERROR] Merge failed!")
        return 1

    merged_df, updated_config = result

    # Save updated config if requested
    if args.save_config:
        save_updated_config(updated_config, args.config)

    print("\n[INFO] Merge finished. You can now run analysis scripts using this merged file.")
    print(f"   Merged file: {paths['output_path']}")

    return 0

if __name__ == "__main__":
    exit(main())
