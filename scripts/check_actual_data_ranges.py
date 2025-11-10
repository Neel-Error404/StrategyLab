#!/usr/bin/env python3
"""
Check Actual Data Ranges in Data Pools
Validates the real date coverage vs folder names
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

def analyze_pool(pool_path):
    """Analyze actual data ranges in a pool"""
    
    pool = Path(pool_path)
    if not pool.exists():
        print(f"ERROR: Pool path does not exist: {pool_path}")
        return None
    
    tickers = sorted([d.name for d in pool.iterdir() if d.is_dir()])
    
    print('=' * 80)
    print('ACTUAL DATA RANGE ANALYSIS')
    print('=' * 80)
    print(f'\nPool Folder: {pool.name}')
    print(f'Total Tickers: {len(tickers)}\n')
    
    date_ranges = {}
    for ticker in tickers:
        try:
            # Check both 5m and 15m files
            file_5m = pool / ticker / '5m.parquet'
            file_15m = pool / ticker / '15m.parquet'
            
            if file_5m.exists():
                df = pd.read_parquet(file_5m)
                start = pd.to_datetime(df['timestamp'].min())
                end = pd.to_datetime(df['timestamp'].max())
                date_ranges[ticker] = {
                    'start': start,
                    'end': end,
                    'rows_5m': len(df),
                    'file': '5m.parquet'
                }
            elif file_15m.exists():
                df = pd.read_parquet(file_15m)
                start = pd.to_datetime(df['timestamp'].min())
                end = pd.to_datetime(df['timestamp'].max())
                date_ranges[ticker] = {
                    'start': start,
                    'end': end,
                    'rows_15m': len(df),
                    'file': '15m.parquet'
                }
            else:
                date_ranges[ticker] = {
                    'start': None,
                    'end': None,
                    'rows': 0,
                    'error': 'No parquet files found'
                }
        except Exception as e:
            date_ranges[ticker] = {'error': str(e)}
    
    # Find min and max dates across all tickers
    valid_ranges = [v for v in date_ranges.values() if v.get('start') and v.get('end')]
    
    if valid_ranges:
        overall_start = min(v['start'] for v in valid_ranges)
        overall_end = max(v['end'] for v in valid_ranges)
        min_end = min(v['end'] for v in valid_ranges)
        
        print(f'📊 Overall Data Range:')
        print(f'  Earliest Start: {overall_start.date()}')
        print(f'  Latest End: {overall_end.date()}')
        print(f'  Common End (earliest): {min_end.date()}')
        
        # Calculate suggested folder name
        suggested_folder = f"{overall_start.date()}_to_{overall_end.date()}"
        actual_folder = pool.name
        
        print(f'\n📁 Folder Name Analysis:')
        print(f'  Current Folder: {actual_folder}')
        print(f'  Suggested Folder: {suggested_folder}')
        
        if actual_folder != suggested_folder:
            print(f'  ⚠️  MISMATCH DETECTED!')
        else:
            print(f'  ✅ Folder name matches data!')
        
        print(f'\n📋 Ticker-wise End Dates (sorted):')
        print(f'{"Ticker":<20} {"End Date":<15} {"Rows":<10} {"File"}')
        print('-' * 80)
        
        for ticker, info in sorted(date_ranges.items(), key=lambda x: (x[1].get('end') or datetime.min)):
            if info.get('end'):
                rows = info.get('rows_5m', info.get('rows_15m', 0))
                file_type = info.get('file', 'unknown')
                end_date_str = str(info['end'].date())
                print(f'{ticker:<20} {end_date_str:<15} {rows:<10,} {file_type}')
            elif info.get('error'):
                print(f'{ticker:<20} {"ERROR":<15} {info["error"]}')
        
        # Group by end date
        print(f'\n📅 Tickers Grouped by End Date:')
        from collections import defaultdict
        grouped = defaultdict(list)
        for ticker, info in date_ranges.items():
            if info.get('end'):
                grouped[info['end'].date()].append(ticker)
        
        for end_date in sorted(grouped.keys()):
            tickers_on_date = grouped[end_date]
            print(f'\n  {end_date} ({len(tickers_on_date)} tickers):')
            print(f'    {", ".join(sorted(tickers_on_date))}')
        
        return {
            'overall_start': overall_start,
            'overall_end': overall_end,
            'min_end': min_end,
            'suggested_folder': suggested_folder,
            'actual_folder': actual_folder,
            'mismatch': actual_folder != suggested_folder,
            'ticker_count': len(tickers),
            'date_ranges': date_ranges
        }
    else:
        print('❌ No valid data found in pool!')
        return None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        pool_path = sys.argv[1]
    else:
        # Default to the pool mentioned in user's query
        pool_path = "data/pools/2022-01-01_to_2025-08-31"
    
    result = analyze_pool(pool_path)
    
    if result and result['mismatch']:
        print(f'\n\n⚠️  RECOMMENDATION: Rename folder to: {result["suggested_folder"]}')
        print(f'\nCommand to rename:')
        print(f'  Rename-Item -Path "data/pools/{result["actual_folder"]}" -NewName "{result["suggested_folder"]}"')
