#!/usr/bin/env python3
"""
Comprehensive Historical Data Pulling Script
Fetches daily data for all tickers from 1996 to 2025 using Upstox V3 API

Features:
- Progress tracking with detailed logging
- Robust error handling and retry logic
- Data validation and quality checks
- Efficient chunking for large date ranges
- CSV output with standardized format
- Resume capability for interrupted runs
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import time
import json
from typing import List, Dict, Tuple, Optional

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.etl.data_provider.upstox_provider import UpstoxDataProvider
from config.config import UPSTOX_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_pulling.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveDataPuller:
    """Comprehensive data pulling system for historical market data"""
    
    def __init__(self, output_dir: str = "historical_data"):
        self.provider = None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configuration
        self.start_date = datetime(1996, 1, 1)
        self.end_date = datetime(2025, 7, 12)
        self.timeframe = "day"
        self.max_retries = 3
        self.delay_between_requests = 0.5  # Respect API rate limits
        
        # Progress tracking
        self.progress_file = self.output_dir / "progress.json"
        self.error_log_file = self.output_dir / "errors.json"
        self.summary_file = self.output_dir / "summary.json"
        
        # Statistics
        self.stats = {
            'total_tickers': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_data_points': 0,
            'start_time': None,
            'end_time': None,
            'errors': []
        }
        
        logger.info(f"Initialized data puller for {self.start_date.date()} to {self.end_date.date()}")
        
    def load_tickers(self, file_path: str) -> List[str]:
        """Load ticker symbols from file"""
        try:
            with open(file_path, 'r') as f:
                content = f.read().strip()
                # Handle comma-separated format
                tickers = [ticker.strip() for ticker in content.split(',') if ticker.strip()]
                
            logger.info(f"Loaded {len(tickers)} tickers from {file_path}")
            return tickers
            
        except Exception as e:
            logger.error(f"Error loading tickers from {file_path}: {e}")
            return []
    
    def setup_provider(self) -> bool:
        """Initialize and authenticate Upstox provider"""
        try:
            logger.info("Setting up Upstox V3 provider...")
            self.provider = UpstoxDataProvider(UPSTOX_CONFIG)
            
            if self.provider.authenticate():
                logger.info("✅ Upstox authentication successful")
                return True
            else:
                logger.error("❌ Upstox authentication failed")
                return False
                
        except Exception as e:
            logger.error(f"Error setting up provider: {e}")
            return False
    
    def load_progress(self) -> Dict:
        """Load progress from previous run"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                logger.info(f"Resumed from previous run: {len(progress)} tickers already processed")
                return progress
            except Exception as e:
                logger.warning(f"Could not load progress: {e}")
        return {}
    
    def save_progress(self, progress: Dict):
        """Save current progress"""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving progress: {e}")
    
    def fetch_ticker_data(self, ticker: str) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Fetch historical data for a single ticker with retry logic
        
        Returns:
            Tuple of (DataFrame, status_message)
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Fetching data for {ticker} (attempt {attempt + 1}/{self.max_retries})")
                
                df = self.provider.fetch_historical_data(
                    symbol=ticker,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    timeframe=self.timeframe
                )
                
                if df.empty:
                    logger.warning(f"No data returned for {ticker}")
                    return None, "No data available"
                
                # Data validation
                if self.validate_data(df, ticker):
                    logger.info(f"✅ Successfully fetched {len(df)} data points for {ticker}")
                    return df, "Success"
                else:
                    logger.warning(f"Data validation failed for {ticker}")
                    return None, "Data validation failed"
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {ticker}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay_between_requests * (attempt + 1))  # Exponential backoff
                else:
                    return None, f"Failed after {self.max_retries} attempts: {str(e)}"
        
        return None, "Max retries exceeded"
    
    def validate_data(self, df: pd.DataFrame, ticker: str) -> bool:
        """Validate the quality of fetched data"""
        try:
            # Check if DataFrame has required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Missing required columns for {ticker}")
                return False
            
            # Check for reasonable data ranges
            if df['close'].max() <= 0 or df['volume'].min() < 0:
                logger.error(f"Invalid price/volume data for {ticker}")
                return False
            
            # Check for reasonable OHLC relationships
            invalid_ohlc = (
                (df['high'] < df['low']) |
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['low'] > df['open']) |
                (df['low'] > df['close'])
            ).any()
            
            if invalid_ohlc:
                logger.warning(f"Invalid OHLC relationships found for {ticker}")
                # Don't fail validation for this, just warn
            
            # Check date range coverage
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            earliest_date = df['timestamp'].min()
            latest_date = df['timestamp'].max()
            
            logger.info(f"Data range for {ticker}: {earliest_date.date()} to {latest_date.date()}")
            
            return True
            
        except Exception as e:
            logger.error(f"Data validation error for {ticker}: {e}")
            return False
    
    def save_ticker_data(self, ticker: str, df: pd.DataFrame):
        """Save ticker data to CSV file"""
        try:
            # Ensure timestamp is properly formatted
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Create ticker-specific output file
            output_file = self.output_dir / f"{ticker}.csv"
            
            # Save with proper formatting
            df.to_csv(output_file, index=False, date_format='%Y-%m-%d %H:%M:%S%z')
            
            logger.info(f"Saved {len(df)} records for {ticker} to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving data for {ticker}: {e}")
            raise
    
    def update_statistics(self, ticker: str, success: bool, data_points: int = 0, error: str = None):
        """Update running statistics"""
        if success:
            self.stats['successful_downloads'] += 1
            self.stats['total_data_points'] += data_points
        else:
            self.stats['failed_downloads'] += 1
            if error:
                self.stats['errors'].append({
                    'ticker': ticker,
                    'error': error,
                    'timestamp': datetime.now().isoformat()
                })
    
    def save_summary(self):
        """Save final summary report"""
        try:
            self.stats['end_time'] = datetime.now().isoformat()
            
            if self.stats['start_time']:
                duration = datetime.now() - datetime.fromisoformat(self.stats['start_time'])
                self.stats['duration_seconds'] = duration.total_seconds()
                self.stats['duration_formatted'] = str(duration)
            
            # Calculate success rate
            total_processed = self.stats['successful_downloads'] + self.stats['failed_downloads']
            if total_processed > 0:
                self.stats['success_rate'] = (self.stats['successful_downloads'] / total_processed) * 100
            
            with open(self.summary_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
            
            logger.info("📊 FINAL SUMMARY:")
            logger.info(f"  Total tickers: {self.stats['total_tickers']}")
            logger.info(f"  Successful: {self.stats['successful_downloads']}")
            logger.info(f"  Failed: {self.stats['failed_downloads']}")
            logger.info(f"  Success rate: {self.stats.get('success_rate', 0):.1f}%")
            logger.info(f"  Total data points: {self.stats['total_data_points']:,}")
            logger.info(f"  Duration: {self.stats.get('duration_formatted', 'Unknown')}")
            
        except Exception as e:
            logger.error(f"Error saving summary: {e}")
    
    def run_data_collection(self, tickers_file: str):
        """Main data collection process"""
        try:
            # Initialize
            self.stats['start_time'] = datetime.now().isoformat()
            
            # Load tickers
            tickers = self.load_tickers(tickers_file)
            if not tickers:
                logger.error("No tickers loaded. Exiting.")
                return False
            
            self.stats['total_tickers'] = len(tickers)
            
            # Setup provider
            if not self.setup_provider():
                logger.error("Failed to setup data provider. Exiting.")
                return False
            
            # Load previous progress
            progress = self.load_progress()
            
            # Process each ticker
            logger.info(f"🚀 Starting data collection for {len(tickers)} tickers...")
            logger.info(f"📅 Date range: {self.start_date.date()} to {self.end_date.date()}")
            
            for i, ticker in enumerate(tickers, 1):
                # Skip if already processed
                if ticker in progress and progress[ticker].get('status') == 'success':
                    logger.info(f"[{i}/{len(tickers)}] Skipping {ticker} (already processed)")
                    self.stats['successful_downloads'] += 1
                    self.stats['total_data_points'] += progress[ticker].get('data_points', 0)
                    continue
                
                logger.info(f"[{i}/{len(tickers)}] Processing {ticker}...")
                
                # Fetch data
                df, status_message = self.fetch_ticker_data(ticker)
                
                if df is not None:
                    # Save data
                    self.save_ticker_data(ticker, df)
                    
                    # Update progress
                    progress[ticker] = {
                        'status': 'success',
                        'data_points': len(df),
                        'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
                        'processed_at': datetime.now().isoformat()
                    }
                    
                    self.update_statistics(ticker, True, len(df))
                    
                else:
                    # Handle failure
                    progress[ticker] = {
                        'status': 'failed',
                        'error': status_message,
                        'processed_at': datetime.now().isoformat()
                    }
                    
                    self.update_statistics(ticker, False, error=status_message)
                
                # Save progress periodically
                if i % 10 == 0:
                    self.save_progress(progress)
                    logger.info(f"Progress saved. Processed {i}/{len(tickers)} tickers.")
                
                # Rate limiting
                time.sleep(self.delay_between_requests)
            
            # Final save
            self.save_progress(progress)
            self.save_summary()
            
            logger.info("🎉 Data collection completed successfully!")
            return True
            
        except KeyboardInterrupt:
            logger.info("⚠️ Data collection interrupted by user")
            self.save_progress(progress)
            self.save_summary()
            return False
            
        except Exception as e:
            logger.error(f"Fatal error in data collection: {e}")
            return False

def main():
    """Main execution function"""
    print("🚀 COMPREHENSIVE HISTORICAL DATA COLLECTION")
    print("=" * 60)
    print(f"📅 Date Range: 1996-01-01 to 2025-07-12")
    print(f"📊 Data Provider: Upstox V3 API")
    print(f"⏱️ Timeframe: Daily")
    print("=" * 60)
    
    # Initialize data puller
    puller = ComprehensiveDataPuller()
    
    # Run data collection
    tickers_file = "Tickers.txt"
    success = puller.run_data_collection(tickers_file)
    
    if success:
        print("\n✅ Data collection completed successfully!")
        print(f"📁 Output directory: {puller.output_dir}")
        print(f"📋 Summary file: {puller.summary_file}")
    else:
        print("\n❌ Data collection failed or was interrupted")
        print(f"📋 Check logs for details: data_pulling.log")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
