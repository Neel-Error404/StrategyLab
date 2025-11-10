#!/usr/bin/env python3
"""
MSE Data Management Utilities

Helper utilities for managing MSE strategy data:
- Ticker analysis and deduplication
- Data validation and verification
- Progress monitoring and resumption
- Data cleanup and organization

Usage:
    python mse_data_utils.py analyze-tickers
    python mse_data_utils.py validate-data --date-range 2022-01-01_to_2025-07-07
    python mse_data_utils.py check-coverage --ticker RELIANCE
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Set, List, Dict, Any, Optional
import pandas as pd
import json
from datetime import datetime, timedelta
import re

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.config import BACKTESTER_CONFIG

class MSEDataUtils:
    """Utility class for MSE data management."""
    
    def __init__(self):
        self.logger = logging.getLogger("MSEDataUtils")
        
    def analyze_tickers_file(self, tickers_file: Path = None) -> Dict[str, Any]:
        """
        Analyze the Tickers.txt file for duplicates, invalid entries, etc.
        
        Args:
            tickers_file: Path to tickers file
            
        Returns:
            Analysis results dictionary
        """
        if not tickers_file:
            tickers_file = project_root / "Tickers.txt"
            
        if not tickers_file.exists():
            raise FileNotFoundError(f"Tickers file not found: {tickers_file}")
            
        print(f"📊 Analyzing tickers file: {tickers_file}")
        
        with open(tickers_file, 'r') as f:
            content = f.read()
            
        # Split by commas and analyze
        raw_tickers = content.replace('\n', ',').split(',')
        
        stats = {
            'total_raw_entries': len(raw_tickers),
            'empty_entries': 0,
            'duplicate_entries': 0,
            'invalid_entries': [],
            'unique_tickers': set(),
            'ticker_lengths': {},
            'special_characters': set()
        }
        
        seen_tickers = {}
        
        for ticker in raw_tickers:
            cleaned = ticker.strip().upper()
            
            if not cleaned:
                stats['empty_entries'] += 1
                continue
                
            # Track duplicates
            if cleaned in seen_tickers:
                stats['duplicate_entries'] += 1
                seen_tickers[cleaned] += 1
            else:
                seen_tickers[cleaned] = 1
                
            # Validate ticker format
            if not re.match(r'^[A-Z0-9&-]+$', cleaned):
                stats['invalid_entries'].append(cleaned)
                # Track special characters
                for char in cleaned:
                    if not char.isalnum():
                        stats['special_characters'].add(char)
            else:
                stats['unique_tickers'].add(cleaned)
                
            # Track length distribution
            length = len(cleaned)
            stats['ticker_lengths'][length] = stats['ticker_lengths'].get(length, 0) + 1
            
        stats['total_unique'] = len(stats['unique_tickers'])
        stats['most_common_duplicates'] = [(k, v) for k, v in seen_tickers.items() if v > 1][:10]
        
        return stats
        
    def print_ticker_analysis(self, stats: Dict[str, Any]):
        """Print formatted ticker analysis results."""
        print("\n" + "="*60)
        print("              TICKER FILE ANALYSIS")
        print("="*60)
        print(f"📊 Total Raw Entries: {stats['total_raw_entries']}")
        print(f"🎯 Unique Valid Tickers: {stats['total_unique']}")
        print(f"🔄 Duplicate Entries: {stats['duplicate_entries']}")
        print(f"📭 Empty Entries: {stats['empty_entries']}")
        print(f"⚠️  Invalid Entries: {len(stats['invalid_entries'])}")
        
        if stats['special_characters']:
            print(f"🔤 Special Characters Found: {', '.join(sorted(stats['special_characters']))}")
            
        print(f"\n📏 Ticker Length Distribution:")
        for length in sorted(stats['ticker_lengths'].keys()):
            count = stats['ticker_lengths'][length]
            print(f"   {length} chars: {count} tickers")
            
        if stats['most_common_duplicates']:
            print(f"\n🔄 Most Common Duplicates:")
            for ticker, count in stats['most_common_duplicates'][:5]:
                print(f"   {ticker}: {count} times")
                
        if stats['invalid_entries'][:10]:
            print(f"\n⚠️  Sample Invalid Entries:")
            for ticker in stats['invalid_entries'][:10]:
                print(f"   '{ticker}'")
                
        print("="*60)
        
    def save_unique_tickers(self, stats: Dict[str, Any], output_file: Path = None):
        """
        Save unique tickers to a clean file.
        
        Args:
            stats: Ticker analysis stats
            output_file: Output file path
        """
        if not output_file:
            output_file = project_root / "Tickers_unique.txt"
            
        unique_tickers = sorted(stats['unique_tickers'])
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(unique_tickers))
            
        print(f"💾 Saved {len(unique_tickers)} unique tickers to: {output_file}")
        
    def validate_existing_data(self, date_range: str = None, 
                              tickers: List[str] = None) -> Dict[str, Any]:
        """
        Validate existing data files for completeness and quality.
        
        Args:
            date_range: Date range string (e.g., '2022-01-01_to_2025-07-07')
            tickers: List of tickers to validate
            
        Returns:
            Validation results
        """
        if not date_range:
            date_range = "2022-01-01_to_2025-07-07"
            
        data_dir = Path(BACKTESTER_CONFIG['DATA_POOL_DIR']) / date_range / "1minute"
        
        if not data_dir.exists():
            return {'error': f"Data directory not found: {data_dir}"}
            
        print(f"🔍 Validating data in: {data_dir}")
        
        # Get all CSV files
        csv_files = list(data_dir.glob("*.csv"))
        
        validation_results = {
            'total_files': len(csv_files),
            'valid_files': 0,
            'invalid_files': [],
            'file_stats': {},
            'ticker_coverage': set(),
            'date_coverage': {},
            'total_candles': 0,
            'size_stats': {
                'total_size_mb': 0,
                'avg_size_mb': 0,
                'min_candles': float('inf'),
                'max_candles': 0
            }
        }
        
        for csv_file in csv_files:
            try:
                # Extract ticker from filename
                ticker_match = re.search(r'([A-Z0-9&-]+)_1m_', csv_file.name)
                if not ticker_match:
                    validation_results['invalid_files'].append({
                        'file': csv_file.name,
                        'error': 'Invalid filename format'
                    })
                    continue
                    
                ticker = ticker_match.group(1)
                validation_results['ticker_coverage'].add(ticker)
                
                # Load and validate file
                df = pd.read_csv(csv_file)
                
                # Basic validation
                required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    validation_results['invalid_files'].append({
                        'file': csv_file.name,
                        'error': f'Missing columns: {missing_columns}'
                    })
                    continue
                    
                # Data quality checks
                candles_count = len(df)
                file_size_mb = csv_file.stat().st_size / (1024 * 1024)
                
                # Date range analysis
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                min_date = df['timestamp'].min().date()
                max_date = df['timestamp'].max().date()
                
                validation_results['file_stats'][ticker] = {
                    'candles': candles_count,
                    'size_mb': round(file_size_mb, 2),
                    'date_range': f"{min_date} to {max_date}",
                    'min_date': min_date,
                    'max_date': max_date
                }
                
                # Update coverage stats
                if min_date not in validation_results['date_coverage']:
                    validation_results['date_coverage'][min_date] = 0
                validation_results['date_coverage'][min_date] += 1
                
                # Update size stats
                validation_results['size_stats']['total_size_mb'] += file_size_mb
                validation_results['total_candles'] += candles_count
                validation_results['size_stats']['min_candles'] = min(
                    validation_results['size_stats']['min_candles'], candles_count
                )
                validation_results['size_stats']['max_candles'] = max(
                    validation_results['size_stats']['max_candles'], candles_count
                )
                
                validation_results['valid_files'] += 1
                
            except Exception as e:
                validation_results['invalid_files'].append({
                    'file': csv_file.name,
                    'error': str(e)
                })
                
        # Calculate averages
        if validation_results['valid_files'] > 0:
            validation_results['size_stats']['avg_size_mb'] = round(
                validation_results['size_stats']['total_size_mb'] / validation_results['valid_files'], 2
            )
            
        return validation_results
        
    def print_validation_results(self, results: Dict[str, Any]):
        """Print formatted validation results."""
        if 'error' in results:
            print(f"❌ {results['error']}")
            return
            
        print("\n" + "="*60)
        print("              DATA VALIDATION RESULTS")
        print("="*60)
        print(f"📁 Total Files: {results['total_files']}")
        print(f"✅ Valid Files: {results['valid_files']}")
        print(f"❌ Invalid Files: {len(results['invalid_files'])}")
        print(f"🎯 Tickers Covered: {len(results['ticker_coverage'])}")
        print(f"🕯️  Total Candles: {results['total_candles']:,}")
        print(f"💾 Total Size: {results['size_stats']['total_size_mb']:.2f} MB")
        print(f"📊 Avg File Size: {results['size_stats']['avg_size_mb']:.2f} MB")
        
        if results['size_stats']['min_candles'] != float('inf'):
            print(f"📈 Candles Range: {results['size_stats']['min_candles']:,} - {results['size_stats']['max_candles']:,}")
            
        if results['invalid_files']:
            print(f"\n❌ Invalid Files (first 5):")
            for error_info in results['invalid_files'][:5]:
                print(f"   {error_info['file']}: {error_info['error']}")
                
        print("="*60)
        
    def check_ticker_coverage(self, ticker: str, date_range: str = None) -> Dict[str, Any]:
        """
        Check data coverage for a specific ticker.
        
        Args:
            ticker: Ticker symbol
            date_range: Date range string
            
        Returns:
            Coverage analysis
        """
        if not date_range:
            date_range = "2022-01-01_to_2025-07-07"
            
        data_dir = Path(BACKTESTER_CONFIG['DATA_POOL_DIR']) / date_range / "1minute"
        ticker_file = data_dir / f"{ticker}_1m_{date_range}.csv"
        
        if not ticker_file.exists():
            return {
                'ticker': ticker,
                'file_exists': False,
                'error': f"File not found: {ticker_file}"
            }
            
        try:
            df = pd.read_csv(ticker_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Analyze coverage
            min_date = df['timestamp'].min()
            max_date = df['timestamp'].max()
            total_candles = len(df)
            
            # Expected vs actual candles (rough estimate)
            expected_days = (max_date.date() - min_date.date()).days
            expected_candles = expected_days * 375  # ~375 minutes per trading day
            coverage_percentage = (total_candles / expected_candles) * 100 if expected_candles > 0 else 0
            
            # Gap analysis
            df['time_diff'] = df['timestamp'].diff()
            large_gaps = df[df['time_diff'] > pd.Timedelta(minutes=10)]
            
            results = {
                'ticker': ticker,
                'file_exists': True,
                'total_candles': total_candles,
                'date_range': f"{min_date.date()} to {max_date.date()}",
                'coverage_percentage': round(coverage_percentage, 2),
                'large_gaps': len(large_gaps),
                'file_size_mb': round(ticker_file.stat().st_size / (1024 * 1024), 2),
                'first_candle': str(min_date),
                'last_candle': str(max_date)
            }
            
            if len(large_gaps) > 0:
                results['gap_details'] = [
                    {
                        'date': str(row['timestamp'].date()),
                        'gap_hours': row['time_diff'].total_seconds() / 3600
                    }
                    for _, row in large_gaps.head(5).iterrows()
                ]
                
            return results
            
        except Exception as e:
            return {
                'ticker': ticker,
                'file_exists': True,
                'error': f"Error analyzing file: {e}"
            }
            
    def print_coverage_analysis(self, results: Dict[str, Any]):
        """Print ticker coverage analysis."""
        ticker = results['ticker']
        
        print(f"\n🎯 Coverage Analysis for {ticker}")
        print("-" * 40)
        
        if not results['file_exists']:
            print(f"❌ {results['error']}")
            return
            
        if 'error' in results:
            print(f"❌ {results['error']}")
            return
            
        print(f"📅 Date Range: {results['date_range']}")
        print(f"🕯️  Total Candles: {results['total_candles']:,}")
        print(f"📊 Coverage: {results['coverage_percentage']:.1f}%")
        print(f"💾 File Size: {results['file_size_mb']} MB")
        print(f"⚠️  Large Gaps: {results['large_gaps']}")
        
        if 'gap_details' in results and results['gap_details']:
            print("📉 Gap Details (first 5):")
            for gap in results['gap_details']:
                print(f"   {gap['date']}: {gap['gap_hours']:.1f} hours")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="MSE Data Management Utilities")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze tickers command
    analyze_parser = subparsers.add_parser('analyze-tickers', help='Analyze tickers file')
    analyze_parser.add_argument('--save-unique', action='store_true', help='Save unique tickers to file')
    
    # Validate data command
    validate_parser = subparsers.add_parser('validate-data', help='Validate existing data files')
    validate_parser.add_argument('--date-range', type=str, help='Date range to validate')
    
    # Check coverage command
    coverage_parser = subparsers.add_parser('check-coverage', help='Check ticker coverage')
    coverage_parser.add_argument('--ticker', type=str, required=True, help='Ticker to check')
    coverage_parser.add_argument('--date-range', type=str, help='Date range to check')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
        
    utils = MSEDataUtils()
    
    try:
        if args.command == 'analyze-tickers':
            stats = utils.analyze_tickers_file()
            utils.print_ticker_analysis(stats)
            
            if args.save_unique:
                utils.save_unique_tickers(stats)
                
        elif args.command == 'validate-data':
            results = utils.validate_existing_data(args.date_range)
            utils.print_validation_results(results)
            
        elif args.command == 'check-coverage':
            results = utils.check_ticker_coverage(args.ticker, args.date_range)
            utils.print_coverage_analysis(results)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
