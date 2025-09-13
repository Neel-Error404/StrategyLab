#!/usr/bin/env python3
"""
MSE Strategy Data Puller

This script pulls 1-minute historical data for MSE strategy backtesting.
Handles large date ranges, unique ticker extraction, and progress monitoring.

Usage:
    python mse_data_puller.py --mode validate  # Test with small sample
    python mse_data_puller.py --mode full      # Full data pull
    python mse_data_puller.py --mode custom --tickers RELIANCE,TCS --dates 2024-01-01,2024-01-31
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Set, Dict, Any, Optional
import pandas as pd
import json
import time
import traceback

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.config import BACKTESTER_CONFIG, setup_logging
from src.core.etl.data_fetcher import DataFetcher

class MSEDataPuller:
    """Data puller specifically designed for MSE strategy requirements."""
    
    def __init__(self):
        self.logger = logging.getLogger("MSEDataPuller")
        self.unique_tickers: Set[str] = set()
        self.failed_tickers: List[str] = []
        self.successful_tickers: List[str] = []
        self.total_files_created = 0
        self.start_time = None
        
        # MSE Strategy requirements  
        self.required_timeframe = "1m"  # MSE needs 1-minute base data
        self.date_range = {
            'start': '2022-01-01',  # Upstox V3 1-minute data available from Jan 2022
            'end': '2025-07-07'
        }
        
    def extract_unique_tickers(self, tickers_file: Path = None) -> Set[str]:
        """
        Extract unique tickers from Tickers.txt file.
        
        Args:
            tickers_file: Path to tickers file (defaults to Tickers.txt)
            
        Returns:
            Set of unique ticker symbols
        """
        if not tickers_file:
            tickers_file = project_root / "Tickers.txt"
            
        if not tickers_file.exists():
            raise FileNotFoundError(f"Tickers file not found: {tickers_file}")
            
        self.logger.info(f"Reading tickers from: {tickers_file}")
        
        try:
            with open(tickers_file, 'r') as f:
                content = f.read()
                
            # Split by commas and clean up
            raw_tickers = content.replace('\n', ',').split(',')
            unique_tickers = set()
            
            for ticker in raw_tickers:
                cleaned = ticker.strip().upper()
                if cleaned and cleaned.isalnum():  # Only alphanumeric tickers
                    unique_tickers.add(cleaned)
                    
            self.unique_tickers = unique_tickers
            self.logger.info(f"Extracted {len(unique_tickers)} unique tickers")
            return unique_tickers
            
        except Exception as e:
            self.logger.error(f"Error reading tickers file: {e}")
            raise
            
    def estimate_data_requirements(self, tickers: Set[str]) -> Dict[str, Any]:
        """
        Estimate data volume and time requirements.
        
        Args:
            tickers: Set of ticker symbols
            
        Returns:
            Dictionary with estimates
        """
        start_date = datetime.strptime(self.date_range['start'], '%Y-%m-%d')
        end_date = datetime.strptime(self.date_range['end'], '%Y-%m-%d')
        total_days = (end_date - start_date).days
        
        # Estimate market days (roughly 250 trading days per year)
        market_days = int(total_days * (250/365))
        
        # 1-minute data: ~375 candles per trading day (6.5 hours * 60 minutes)
        candles_per_ticker = market_days * 375
        total_candles = candles_per_ticker * len(tickers)
        
        # Estimate file size (assuming ~100 bytes per candle in CSV)
        estimated_size_gb = (total_candles * 100) / (1024**3)
        
        # API calls needed (max 200 days per call)
        api_calls_per_ticker = max(1, total_days // 200)
        total_api_calls = api_calls_per_ticker * len(tickers)
        
        # Estimated time (assume 2 seconds per API call + processing)
        estimated_hours = (total_api_calls * 2) / 3600
        
        estimates = {
            'total_tickers': len(tickers),
            'total_days': total_days,
            'market_days': market_days,
            'total_candles': total_candles,
            'estimated_size_gb': round(estimated_size_gb, 2),
            'total_api_calls': total_api_calls,
            'estimated_hours': round(estimated_hours, 2),
            'date_range': f"{self.date_range['start']} to {self.date_range['end']}"
        }
        
        return estimates
        
    def validate_setup(self) -> bool:
        """
        Validate that the system is ready for data pulling.
        
        Returns:
            True if setup is valid
        """
        self.logger.info("🔍 Validating system setup...")
        
        try:
            # Check data fetcher
            fetcher = DataFetcher()
            if not fetcher.provider:
                self.logger.error("❌ No data provider available")
                return False
                
            self.logger.info(f"✅ Data provider: {fetcher.provider_name}")
            
            # Check authentication
            if hasattr(fetcher.provider, 'authenticated') and not fetcher.provider.authenticated:
                self.logger.info("🔐 Attempting authentication...")
                if not fetcher.provider.authenticate():
                    self.logger.error("❌ Authentication failed")
                    return False
                self.logger.info("✅ Authentication successful")
            
            # Check timeframe support
            supported_timeframes = BACKTESTER_CONFIG.get('SUPPORTED_TIMEFRAMES', [])
            if self.required_timeframe not in supported_timeframes:
                self.logger.error(f"❌ Timeframe {self.required_timeframe} not supported")
                return False
                
            self.logger.info(f"✅ Timeframe {self.required_timeframe} supported")
            
            # Check output directory
            data_pool_dir = Path(BACKTESTER_CONFIG['DATA_POOL_DIR'])
            if not data_pool_dir.exists():
                data_pool_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"✅ Created data pool directory: {data_pool_dir}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Setup validation failed: {e}")
            return False
            
    def run_validation_test(self, test_tickers: List[str] = None, 
                           test_days: int = 5) -> bool:
        """
        Run a small validation test with limited data.
        
        Args:
            test_tickers: List of tickers to test (defaults to first 3 unique tickers)
            test_days: Number of days to test (default: 5)
            
        Returns:
            True if test successful
        """
        if not test_tickers:
            test_tickers = list(self.unique_tickers)[:3]
            
        test_end = datetime.now() - timedelta(days=1)  # Yesterday
        test_start = test_end - timedelta(days=test_days)
        
        self.logger.info(f"🧪 Running validation test:")
        self.logger.info(f"   Tickers: {test_tickers}")
        self.logger.info(f"   Date range: {test_start.date()} to {test_end.date()}")
        
        try:
            fetcher = DataFetcher()
            
            for ticker in test_tickers:
                self.logger.info(f"   Testing ticker: {ticker}")
                
                df = fetcher.provider.fetch_historical_data(
                    symbol=ticker,
                    start_date=test_start,
                    end_date=test_end,
                    timeframe=self.required_timeframe
                )
                
                if df.empty:
                    self.logger.warning(f"   ⚠️ No data for {ticker}")
                else:
                    self.logger.info(f"   ✅ {ticker}: {len(df)} candles")
                    
            self.logger.info("✅ Validation test completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Validation test failed: {e}")
            return False
            
    def create_date_chunks(self, start_date: str, end_date: str, 
                          chunk_days: int = 28) -> List[Dict[str, str]]:
        """
        Create date chunks for efficient API calls.
        
        Args:
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            chunk_days: Days per chunk (default: 28, under 30-day API limit for 1-minute data)
            
        Returns:
            List of date chunk dictionaries
        """
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        chunks = []
        current_start = start_dt
        
        while current_start <= end_dt:
            current_end = min(current_start + timedelta(days=chunk_days - 1), end_dt)
            
            chunks.append({
                'start': current_start.strftime('%Y-%m-%d'),
                'end': current_end.strftime('%Y-%m-%d'),
                'days': (current_end - current_start).days + 1
            })
            
            current_start = current_end + timedelta(days=1)
            
        return chunks
        
    def pull_ticker_data(self, ticker: str, date_chunks: List[Dict[str, str]], 
                        fetcher: DataFetcher) -> bool:
        """
        Pull data for a single ticker across all date chunks.
        
        Args:
            ticker: Ticker symbol
            date_chunks: List of date chunks
            fetcher: DataFetcher instance
            
        Returns:
            True if successful
        """
        try:
            self.logger.info(f"📥 Pulling data for {ticker} ({len(date_chunks)} chunks)")
            
            all_data = []
            
            for i, chunk in enumerate(date_chunks, 1):
                self.logger.debug(f"   Chunk {i}/{len(date_chunks)}: {chunk['start']} to {chunk['end']}")
                
                start_dt = datetime.strptime(chunk['start'], '%Y-%m-%d')
                end_dt = datetime.strptime(chunk['end'], '%Y-%m-%d')
                
                # Fetch data for this chunk
                df = fetcher.provider.fetch_historical_data(
                    symbol=ticker,
                    start_date=start_dt,
                    end_date=end_dt,
                    timeframe=self.required_timeframe
                )
                
                if not df.empty:
                    all_data.append(df)
                    self.logger.debug(f"   ✅ Chunk {i}: {len(df)} candles")
                else:
                    self.logger.warning(f"   ⚠️ Chunk {i}: No data")
                
                # Small delay to respect API limits
                time.sleep(0.5)
                
            if all_data:
                # Combine all chunks
                combined_df = pd.concat(all_data, ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                
                # Save to file
                date_range_str = f"{self.date_range['start']}_to_{self.date_range['end']}"
                output_dir = Path(BACKTESTER_CONFIG['DATA_POOL_DIR']) / date_range_str / "1minute"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                output_file = output_dir / f"{ticker}_{self.required_timeframe}_{date_range_str}.csv"
                combined_df.to_csv(output_file, index=False)
                
                self.logger.info(f"✅ {ticker}: {len(combined_df)} candles saved to {output_file}")
                self.total_files_created += 1
                return True
            else:
                self.logger.warning(f"⚠️ {ticker}: No data found for any chunks")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error pulling data for {ticker}: {e}")
            return False
            
    def run_full_data_pull(self, tickers: Set[str] = None) -> Dict[str, Any]:
        """
        Run the full data pulling process.
        
        Args:
            tickers: Set of tickers to process (defaults to all unique tickers)
            
        Returns:
            Summary dictionary
        """
        if not tickers:
            tickers = self.unique_tickers
            
        if not tickers:
            raise ValueError("No tickers to process")
            
        self.start_time = datetime.now()
        self.logger.info(f"🚀 Starting full data pull for {len(tickers)} tickers")
        
        # Create date chunks
        date_chunks = self.create_date_chunks(
            self.date_range['start'], 
            self.date_range['end']
        )
        self.logger.info(f"📅 Created {len(date_chunks)} date chunks")
        
        # Initialize data fetcher
        fetcher = DataFetcher()
        
        # Process each ticker
        for i, ticker in enumerate(sorted(tickers), 1):
            self.logger.info(f"🔄 Processing ticker {i}/{len(tickers)}: {ticker}")
            
            try:
                success = self.pull_ticker_data(ticker, date_chunks, fetcher)
                
                if success:
                    self.successful_tickers.append(ticker)
                else:
                    self.failed_tickers.append(ticker)
                    
            except Exception as e:
                self.logger.error(f"❌ Failed to process {ticker}: {e}")
                self.failed_tickers.append(ticker)
                
            # Progress update every 10 tickers
            if i % 10 == 0:
                elapsed = datetime.now() - self.start_time
                self.logger.info(f"📊 Progress: {i}/{len(tickers)} tickers, "
                               f"elapsed: {elapsed}, files: {self.total_files_created}")
                
        # Generate summary
        end_time = datetime.now()
        elapsed_time = end_time - self.start_time
        
        summary = {
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'elapsed_time': str(elapsed_time),
            'total_tickers': len(tickers),
            'successful_tickers': len(self.successful_tickers),
            'failed_tickers': len(self.failed_tickers),
            'total_files_created': self.total_files_created,
            'success_rate': round(len(self.successful_tickers) / len(tickers) * 100, 2),
            'failed_tickers_list': self.failed_tickers[:10]  # First 10 failures
        }
        
        # Save summary
        summary_file = Path(BACKTESTER_CONFIG['DATA_POOL_DIR']) / f"mse_data_pull_summary_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        self.logger.info(f"📋 Summary saved to: {summary_file}")
        return summary
        
    def print_summary_report(self, summary: Dict[str, Any]):
        """Print a formatted summary report."""
        print("\n" + "="*60)
        print("              MSE DATA PULL SUMMARY")
        print("="*60)
        print(f"📅 Date Range: {self.date_range['start']} to {self.date_range['end']}")
        print(f"⏱️  Duration: {summary['elapsed_time']}")
        print(f"📊 Total Tickers: {summary['total_tickers']}")
        print(f"✅ Successful: {summary['successful_tickers']} ({summary['success_rate']}%)")
        print(f"❌ Failed: {summary['failed_tickers']}")
        print(f"📁 Files Created: {summary['total_files_created']}")
        
        if summary['failed_tickers'] > 0:
            print(f"\n❌ First 10 Failed Tickers:")
            for ticker in summary['failed_tickers_list']:
                print(f"   - {ticker}")
        
        print("="*60)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="MSE Strategy Data Puller")
    parser.add_argument('--mode', choices=['validate', 'full', 'custom', 'estimate'], 
                       required=True, help="Execution mode")
    parser.add_argument('--tickers', type=str, help="Comma-separated ticker list for custom mode")
    parser.add_argument('--dates', type=str, help="Start,End dates for custom mode (YYYY-MM-DD,YYYY-MM-DD)")
    parser.add_argument('--test-size', type=int, default=5, help="Number of tickers for validation test")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging('mse_data_puller')
    
    try:
        puller = MSEDataPuller()
        
        # Extract unique tickers
        unique_tickers = puller.extract_unique_tickers()
        print(f"📋 Found {len(unique_tickers)} unique tickers")
        
        if args.mode == 'estimate':
            # Show estimates only
            estimates = puller.estimate_data_requirements(unique_tickers)
            print("\n" + "="*50)
            print("         DATA REQUIREMENTS ESTIMATE")
            print("="*50)
            print(f"🎯 Total Tickers: {estimates['total_tickers']}")
            print(f"📅 Date Range: {estimates['date_range']}")
            print(f"📊 Total Days: {estimates['total_days']} ({estimates['market_days']} market days)")
            print(f"🕯️  Total Candles: {estimates['total_candles']:,}")
            print(f"💾 Estimated Size: {estimates['estimated_size_gb']} GB")
            print(f"🌐 API Calls Needed: {estimates['total_api_calls']:,}")
            print(f"⏱️  Estimated Time: {estimates['estimated_hours']} hours")
            print("="*50)
            print("⚠️  WARNING: This is a massive data pull!")
            print("   Consider starting with validation mode first.")
            
        elif args.mode == 'validate':
            # Validation mode
            if not puller.validate_setup():
                print("❌ Setup validation failed. Please fix issues before proceeding.")
                return
                
            print("✅ Setup validation passed!")
            
            # Run test with small sample
            test_tickers = list(unique_tickers)[:args.test_size]
            if puller.run_validation_test(test_tickers):
                print("✅ Validation test successful! System is ready for full data pull.")
            else:
                print("❌ Validation test failed. Please review logs.")
                
        elif args.mode == 'full':
            # Full data pull
            if not puller.validate_setup():
                print("❌ Setup validation failed.")
                return
                
            # Show estimates - auto-confirm for full pull
            estimates = puller.estimate_data_requirements(unique_tickers)
            print(f"\n⚠️  Starting full data pull: {estimates['estimated_size_gb']} GB of data!")
            print(f"   Estimated processing time: {estimates['estimated_hours']} hours.")
            print("   Auto-confirming for full data pull...")
            
            # Auto-confirm - no user input needed
                
            summary = puller.run_full_data_pull()
            puller.print_summary_report(summary)
            
        elif args.mode == 'custom':
            # Custom mode with specified tickers/dates
            if args.tickers:
                custom_tickers = set(args.tickers.upper().split(','))
                custom_tickers = custom_tickers.intersection(unique_tickers)
                if not custom_tickers:
                    print("❌ No valid tickers found in custom list")
                    return
            else:
                custom_tickers = unique_tickers
                
            if args.dates:
                start_date, end_date = args.dates.split(',')
                puller.date_range = {'start': start_date, 'end': end_date}
                
            print(f"🎯 Custom mode: {len(custom_tickers)} tickers, "
                  f"{puller.date_range['start']} to {puller.date_range['end']}")
                  
            if not puller.validate_setup():
                print("❌ Setup validation failed.")
                return
                
            summary = puller.run_full_data_pull(custom_tickers)
            puller.print_summary_report(summary)
            
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
