#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 0.2: Data Baseline Establishment
=======================================

Purpose: Pull fresh data for test tickers to establish clean baseline for testing.

Test Tickers:
- RELIANCE (Equity, Liquid)
- NIFTY (Index)
- INFY (Equity, IT)
- BANKNIFTY (Index)
- TCS (Equity, IT)

Data Requirements:
- 1-minute: Last 30 calendar days
- 5-minute: Last 30 calendar days
- 1-day: Last 2 years
- Options chains: Last 3 months (weekly expiry)

Success Criteria:
- All 5 tickers fetched successfully
- Zero gaps in data
- Data integrity: no duplicates, no missing timestamps
- SHA256 hash generated for baseline
- Metadata recorded (fetch date, row counts, file sizes)

Expected Runtime: 10-15 minutes

Author: QA Team
Date: October 16, 2025
"""

import sys
import os

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import hashlib
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.core.etl.data_fetcher import DataFetcher
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DataBaselineEstablisher:
    """Establishes fresh data baseline for QA testing"""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.baseline_dir = self.project_root / 'data' / 'pools' / 'qa_testing_baseline'
        self.results: Dict[str, Dict] = {}
        
        # Test tickers
        self.tickers = ['RELIANCE', 'NIFTY', 'INFY', 'BANKNIFTY', 'TCS']
        
        # Date ranges (use datetime objects, not date)
        self.end_date = datetime.now()
        self.start_intraday = self.end_date - timedelta(days=30)
        self.start_daily = self.end_date - timedelta(days=2*365)  # 2 years
        self.start_options = self.end_date - timedelta(days=90)  # 3 months
        
    def setup_baseline_directory(self):
        """Create clean baseline directory"""
        print("Setting up baseline directory...")
        
        # Remove existing baseline if present
        if self.baseline_dir.exists():
            import shutil
            shutil.rmtree(self.baseline_dir)
            print(f"  Removed existing baseline: {self.baseline_dir}")
        
        # Create fresh structure
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        for timeframe in ['1minute', '5minute', '1day', 'options']:
            (self.baseline_dir / timeframe).mkdir(exist_ok=True)
        
        print(f"  ✅ Created: {self.baseline_dir}")
    
    def fetch_all_data(self):
        """Fetch data for all tickers and timeframes"""
        print("\n" + "=" * 70)
        print("Fetching Data for All Tickers")
        print("=" * 70)
        
        try:
            # Initialize data fetcher
            fetcher = DataFetcher()
            
            # Fetch intraday data (1m, 5m)
            print(f"\n📥 Fetching intraday data (1m, 5m) for {len(self.tickers)} tickers...")
            print(f"   Date range: {self.start_intraday} to {self.end_date}")
            
            result_intraday = fetcher.fetch_historical_data(
                tickers=self.tickers,
                timeframes=['1m', '5m'],
                start_date=self.start_intraday,
                end_date=self.end_date,
                output_dir=self.baseline_dir,
                use_ticker_first_storage=True
            )
            
            # Fetch daily data (1d)
            print(f"\n📥 Fetching daily data (1d) for {len(self.tickers)} tickers...")
            print(f"   Date range: {self.start_daily} to {self.end_date}")
            
            result_daily = fetcher.fetch_historical_data(
                tickers=self.tickers,
                timeframes=['day'],
                start_date=self.start_daily,
                end_date=self.end_date,
                output_dir=self.baseline_dir,
                use_ticker_first_storage=True
            )
            
            # Process results
            print("\n" + "=" * 70)
            print("Data Fetch Results")
            print("=" * 70)
            
            for ticker in self.tickers:
                self.results[ticker] = {}
                print(f"\n{ticker}:")
                
                # Check 1m
                if ticker in result_intraday and '1m' in result_intraday[ticker]:
                    file_path = result_intraday[ticker]['1m']
                    df = pd.read_parquet(file_path)
                    self.results[ticker]['1minute'] = {
                        'success': True,
                        'message': f"{len(df)} rows fetched",
                        'row_count': len(df),
                        'date_range': f"{self.start_intraday} to {self.end_date}",
                    }
                    print(f"  ✅ 1m: {len(df)} bars")
                else:
                    self.results[ticker]['1minute'] = {
                        'success': False,
                        'message': "Failed to fetch",
                        'row_count': 0,
                        'date_range': f"{self.start_intraday} to {self.end_date}",
                    }
                    print(f"  ❌ 1m: FAILED")
                
                # Check 5m
                if ticker in result_intraday and '5m' in result_intraday[ticker]:
                    file_path = result_intraday[ticker]['5m']
                    df = pd.read_parquet(file_path)
                    self.results[ticker]['5minute'] = {
                        'success': True,
                        'message': f"{len(df)} rows fetched",
                        'row_count': len(df),
                        'date_range': f"{self.start_intraday} to {self.end_date}",
                    }
                    print(f"  ✅ 5m: {len(df)} bars")
                else:
                    self.results[ticker]['5minute'] = {
                        'success': False,
                        'message': "Failed to fetch",
                        'row_count': 0,
                        'date_range': f"{self.start_intraday} to {self.end_date}",
                    }
                    print(f"  ❌ 5m: FAILED")
                
                # Check 1d
                if ticker in result_daily and 'day' in result_daily[ticker]:
                    file_path = result_daily[ticker]['day']
                    df = pd.read_parquet(file_path)
                    self.results[ticker]['1day'] = {
                        'success': True,
                        'message': f"{len(df)} rows fetched",
                        'row_count': len(df),
                        'date_range': f"{self.start_daily} to {self.end_date}",
                    }
                    print(f"  ✅ 1d: {len(df)} bars")
                else:
                    self.results[ticker]['1day'] = {
                        'success': False,
                        'message': "Failed to fetch",
                        'row_count': 0,
                        'date_range': f"{self.start_daily} to {self.end_date}",
                    }
                    print(f"  ❌ 1d: FAILED")
                    
        except Exception as e:
            print(f"\n❌ ERROR during data fetch: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Mark all as failed
            for ticker in self.tickers:
                self.results[ticker] = {}
                for timeframe in ['1minute', '5minute', '1day']:
                    self.results[ticker][timeframe] = {
                        'success': False,
                        'message': f"Error: {str(e)}",
                        'row_count': 0,
                        'date_range': '',
                    }
    
    def validate_data_integrity(self):
        """Validate fetched data for integrity issues"""
        print("\n" + "=" * 70)
        print("Validating Data Integrity")
        print("=" * 70)
        
        issues = []
        
        # Map timeframe names to file names
        timeframe_file_map = {
            '1minute': '1m.parquet',
            '5minute': '5m.parquet',
            '1day': 'day.parquet',
        }
        
        for timeframe, filename in timeframe_file_map.items():
            for ticker in self.tickers:
                # Ticker-first structure: baseline_dir/TICKER/timeframe.parquet
                file_path = self.baseline_dir / ticker / filename
                
                if not file_path.exists():
                    issues.append(f"Missing: {ticker} {timeframe}")
                    continue
                
                try:
                    df = pd.read_parquet(file_path)
                    
                    # Check for duplicates
                    if df.duplicated(subset=['timestamp']).any():
                        issues.append(f"Duplicates: {ticker} {timeframe}")
                    
                    # Check for missing timestamps (gaps)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp')
                    
                    # Calculate expected frequency
                    freq_map = {
                        '1minute': '1min',
                        '5minute': '5min',
                        '1day': '1D',
                    }
                    expected_freq = freq_map[timeframe]
                    
                    # For intraday, only check market hours (9:15 to 15:30)
                    if timeframe in ['1minute', '5minute']:
                        df = df[df['timestamp'].dt.time >= pd.Timestamp('09:15:00').time()]
                        df = df[df['timestamp'].dt.time <= pd.Timestamp('15:30:00').time()]
                    
                    # Simple gap check: look for large time jumps
                    df['time_diff'] = df['timestamp'].diff()
                    
                    if timeframe == '1minute':
                        max_gap = pd.Timedelta(minutes=5)
                    elif timeframe == '5minute':
                        max_gap = pd.Timedelta(minutes=15)
                    else:
                        max_gap = pd.Timedelta(days=7)
                    
                    gaps = df[df['time_diff'] > max_gap]
                    if not gaps.empty:
                        gap_count = len(gaps)
                        issues.append(f"Gaps ({gap_count}): {ticker} {timeframe}")
                    
                except Exception as e:
                    issues.append(f"Error reading: {ticker} {timeframe} - {str(e)}")
        
        if issues:
            print("❌ INTEGRITY ISSUES FOUND:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("✅ ALL DATA PASSED INTEGRITY CHECKS")
        
        return len(issues) == 0
    
    def generate_baseline_hash(self) -> str:
        """Generate SHA256 hash of entire baseline for reproducibility"""
        print("\n" + "=" * 70)
        print("Generating Baseline Hash")
        print("=" * 70)
        
        hasher = hashlib.sha256()
        
        # Map timeframe names to file names
        timeframe_file_map = {
            '1minute': '1m.parquet',
            '5minute': '5m.parquet',
            '1day': 'day.parquet',
        }
        
        # Hash all parquet files in sorted order (ticker-first structure)
        for ticker in sorted(self.tickers):
            for timeframe in sorted(['1minute', '5minute', '1day']):
                filename = timeframe_file_map[timeframe]
                file_path = self.baseline_dir / ticker / filename
                
                if file_path.exists():
                    with open(file_path, 'rb') as f:
                        hasher.update(f.read())
        
        baseline_hash = hasher.hexdigest()
        print(f"  Baseline SHA256: {baseline_hash[:16]}...{baseline_hash[-16:]}")
        
        return baseline_hash
    
    def generate_metadata(self, baseline_hash: str):
        """Generate metadata file for baseline"""
        metadata = {
            'creation_date': datetime.now().isoformat(),
            'tickers': self.tickers,
            'date_ranges': {
                'intraday': f"{self.start_intraday} to {self.end_date}",
                'daily': f"{self.start_daily} to {self.end_date}",
            },
            'baseline_hash': baseline_hash,
            'results': self.results,
        }
        
        metadata_file = self.baseline_dir / 'BASELINE_METADATA.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n📄 Metadata saved: {metadata_file}")
    
    def generate_summary_report(self):
        """Generate human-readable summary"""
        report_file = self.project_root / 'outputs' / 'qa_phase0.2_data_baseline_report.txt'
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("Phase 0.2: Data Baseline Establishment Report\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Creation Date: {datetime.now()}\n")
            f.write(f"Baseline Directory: {self.baseline_dir}\n\n")
            
            f.write("Tickers: " + ", ".join(self.tickers) + "\n\n")
            
            f.write("Date Ranges:\n")
            f.write(f"  - Intraday (1m, 5m): {self.start_intraday} to {self.end_date}\n")
            f.write(f"  - Daily (1d): {self.start_daily} to {self.end_date}\n\n")
            
            f.write("Results by Ticker:\n")
            for ticker, timeframes in self.results.items():
                f.write(f"\n{ticker}:\n")
                for timeframe, result in timeframes.items():
                    status = "✅" if result['success'] else "❌"
                    f.write(f"  {status} {timeframe}: {result['row_count']} rows\n")
        
        print(f"📄 Report saved: {report_file}")
    
    def run_all_steps(self) -> bool:
        """Execute full baseline establishment workflow"""
        print("=" * 70)
        print("Phase 0.2: Data Baseline Establishment")
        print("=" * 70)
        
        # Step 1: Setup directory
        self.setup_baseline_directory()
        
        # Step 2: Fetch all data
        self.fetch_all_data()
        
        # Step 3: Validate integrity
        integrity_passed = self.validate_data_integrity()
        
        # Step 4: Generate hash
        baseline_hash = self.generate_baseline_hash()
        
        # Step 5: Generate metadata
        self.generate_metadata(baseline_hash)
        
        # Step 6: Generate report
        self.generate_summary_report()
        
        print("\n" + "=" * 70)
        if integrity_passed:
            print("✅ BASELINE ESTABLISHMENT COMPLETE")
            print(f"   Location: {self.baseline_dir}")
            print(f"   Hash: {baseline_hash[:16]}...{baseline_hash[-16:]}")
        else:
            print("⚠️  BASELINE COMPLETE WITH WARNINGS")
            print("   Review integrity issues before proceeding")
        print("=" * 70)
        
        return integrity_passed


def main():
    """Main test entry point"""
    establisher = DataBaselineEstablisher()
    success = establisher.run_all_steps()
    
    print("\n📝 Manual Action Required:")
    print("   1. Update QA_TESTING_JOURNAL.md with test results")
    print("   2. Review baseline data in qa_testing_baseline/")
    print("   3. Proceed to Phase 1 (Core Backtester) if all passed")
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
