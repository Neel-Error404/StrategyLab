# src/etl/data_fetcher.py
import os
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
import pytz
from typing import List, Dict, Any, Union, Optional
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

# Optional parquet support
try:
    import pyarrow
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

from config.config import BACKTESTER_CONFIG
from .data_provider.provider_factory import DataProviderFactory

# Set up IST timezone for consistent timestamping
IST = pytz.timezone("Asia/Kolkata")

class DataFetcher:
    """
    Enhanced data fetcher with support for multiple data providers.
    """
    
    def __init__(self, config: Dict[str, Any] = None, provider_name: str = None):
        """
        Initialize the data fetcher with enhanced provider management.
        
        Args:
            config: Configuration dictionary (defaults to BACKTESTER_CONFIG)
            provider_name: Name of the data provider to use (defaults to config value or auto-detect)
        """
        self.config = config or BACKTESTER_CONFIG
        self.logger = logging.getLogger("DataFetcher")
        
        # Enhanced provider selection with auto-detection fallback
        if provider_name:
            self.provider_name = provider_name
            self.provider = DataProviderFactory.get_provider(provider_name)
        else:
            # Try configured provider first, then auto-detect
            configured_provider = self.config.get('DATA_PROVIDER', 'upstox')
            self.provider = DataProviderFactory.get_provider(configured_provider)
            
            if not self.provider:
                self.logger.warning(f"Configured provider '{configured_provider}' failed, trying auto-detection")
                self.provider = DataProviderFactory.get_provider(auto_detect=True)
            
            if not self.provider:
                self.logger.warning("Auto-detection failed, trying fallback")
                self.provider = DataProviderFactory.get_provider_with_fallback(configured_provider)
        
        if not self.provider:
            available_providers = DataProviderFactory.list_providers()
            self.logger.error("Failed to initialize any data provider")
            self.logger.error(f"Available providers: {list(available_providers.keys())}")
            raise ValueError("No data provider could be initialized")
        
        # Get provider name from the initialized provider
        self.provider_name = getattr(self.provider, 'provider_name', 
                                   self.provider.__class__.__name__.replace('DataProvider', '').lower())
        
        self.logger.info(f"Initialized data fetcher with provider: {self.provider_name}")
        
        # Authenticate the provider
        if not self.provider.authenticate():
            self.logger.error(f"Failed to authenticate with {self.provider_name}")
            raise ValueError(f"Authentication failed for provider '{self.provider_name}'")
    
    def fetch_historical_data(self, 
                             tickers: List[str], 
                             timeframes: List[str], 
                             start_date: Union[str, datetime], 
                             end_date: Union[str, datetime], 
                             output_dir: Optional[Path] = None,
                             use_ticker_first_storage: bool = True) -> Dict[str, Dict[str, Path]]:
        """
        Fetch historical data for multiple tickers and timeframes with enhanced storage options.
        
        Args:
            tickers: List of ticker symbols
            timeframes: List of timeframes (e.g., '1m', '5m', 'day')
            start_date: Start date for the data
            end_date: End date for the data
            output_dir: Output directory (defaults to DATA_POOL_DIR/current_date)
            use_ticker_first_storage: Use ticker-first directory structure with parquet files
            
        Returns:
            Dictionary mapping tickers to timeframes to saved file paths
        """
        # Convert dates to datetime if they're strings
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
          # Create output directory
        if not output_dir:
            date_range = f"{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}"
            data_pool_dir = Path(self.config.get('DATA_POOL_DIR'))
            output_dir = data_pool_dir / date_range
        
        # Check parquet support
        use_parquet = use_ticker_first_storage and PARQUET_AVAILABLE
        if use_ticker_first_storage and not PARQUET_AVAILABLE:
            self.logger.warning("pyarrow not available, falling back to CSV format")
        
        result = {}
        total_combinations = len(tickers) * len(timeframes)
        current_progress = 0
        failed_combinations = []
        
        self.logger.info(f"Starting enhanced data fetch: {len(tickers)} tickers × {len(timeframes)} timeframes = {total_combinations} combinations")
        self.logger.info(f"Storage mode: {'Ticker-first with ' + ('Parquet' if use_parquet else 'CSV') if use_ticker_first_storage else 'Timeframe-first CSV'}")
        
        # Process each ticker
        for ticker in tickers:
            result[ticker] = {}
            
            # Process each timeframe
            for timeframe in timeframes:
                current_progress += 1
                # Create appropriate directory structure
                if use_ticker_first_storage:
                    # Ticker-first: data/pools/date_range/TICKER/
                    ticker_dir = output_dir / ticker
                    ticker_dir.mkdir(parents=True, exist_ok=True)
                    target_dir = ticker_dir
                else:
                    # Timeframe-first: data/pools/date_range/timeframe/
                    timeframe_folder = self.config.get('TIMEFRAME_FOLDERS', {}).get(timeframe, timeframe)
                    timeframe_dir = output_dir / timeframe_folder
                    timeframe_dir.mkdir(parents=True, exist_ok=True)
                    target_dir = timeframe_dir
                
                # Enhanced progress logging
                progress_pct = (current_progress / total_combinations) * 100
                self.logger.info(f"[{current_progress}/{total_combinations}] ({progress_pct:.1f}%) Fetching {timeframe} data for {ticker} from {start_date.date()} to {end_date.date()}")
                
                try:
                    df = self.provider.fetch_historical_data(ticker, start_date, end_date, timeframe)
                    
                    if df.empty:
                        self.logger.warning(f"No data returned for {ticker} at {timeframe} timeframe")
                        failed_combinations.append(f"{ticker}_{timeframe}_no_data")
                        continue
                    
                    # Create filename based on storage mode
                    if use_ticker_first_storage:
                        filename = f"{timeframe}.{'parquet' if use_parquet else 'csv'}"
                        file_path = target_dir / filename
                    else:
                        filename = f"{ticker}_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}.csv"
                        file_path = target_dir / filename

                    # Ensure destination directory still exists (guards against race conditions or external cleanup)
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Save data with appropriate format
                    if use_ticker_first_storage and use_parquet:
                        df.to_parquet(file_path, index=False)
                        format_info = "parquet"
                    else:
                        df.to_csv(file_path, index=False)
                        format_info = "CSV"
                    
                    self.logger.info(f"✅ Saved {len(df)} records for {ticker}@{timeframe} as {format_info} to {file_path}")
                    
                    result[ticker][timeframe] = file_path
                    
                    # Memory cleanup
                    del df
                    
                except Exception as e:
                    error_msg = f"{ticker}_{timeframe}_{str(e)[:50]}"
                    failed_combinations.append(error_msg)
                    self.logger.error(f"❌ Error fetching data for {ticker} at {timeframe} timeframe: {e}")
        
        # Final summary report
        successful_combinations = total_combinations - len(failed_combinations)
        success_rate = (successful_combinations / total_combinations * 100) if total_combinations > 0 else 0
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"📊 ENHANCED DATA FETCH SUMMARY")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total combinations: {total_combinations}")
        self.logger.info(f"Successful: {successful_combinations} ({success_rate:.1f}%)")
        self.logger.info(f"Failed: {len(failed_combinations)}")
        
        if failed_combinations:
            self.logger.warning(f"Failed combinations: {failed_combinations[:5]}{'...' if len(failed_combinations) > 5 else ''}")
        
        storage_mode = "Ticker-first " + ("Parquet" if use_parquet else "CSV") if use_ticker_first_storage else "Timeframe-first CSV"
        self.logger.info(f"Storage format: {storage_mode}")
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"{'='*60}")
            
        return result
    
    def get_user_inputs(self) -> Dict[str, Any]:
        """
        Prompt the user for inputs to fetch historical data.
        
        Returns:
            Dictionary with user inputs
        """
        DEFAULT_TICKER = self.config.get('DEFAULT_TICKER', ["RELIANCE", "TCS", "INFY"])
        DEFAULT_TIMEFRAME = self.config.get('DEFAULT_TIMEFRAME', ['1m'])
        SUPPORTED_TIMEFRAMES = self.config.get('SUPPORTED_TIMEFRAMES', ['1m', '5m', '15m', '30m', '1h', 'day', 'week', 'month'])

        # Get tickers
        tickers_input = input(f"Enter ticker names (comma-separated, default: {DEFAULT_TICKER}): ").strip()
        tickers = [t.strip() for t in tickers_input.split(",")] if tickers_input else DEFAULT_TICKER

        # Get timeframes
        timeframes_input = input(f"Enter timeframes (comma-separated, default: {DEFAULT_TIMEFRAME}): ").strip()
        timeframes = [tf.strip() for tf in timeframes_input.split(",") if tf.strip() in SUPPORTED_TIMEFRAMES] if timeframes_input else DEFAULT_TIMEFRAME

        # Get date range
        start_date_str = input("Enter start date (YYYY-MM-DD, default: 7 days ago): ").strip()
        end_date_str = input("Enter end date (YYYY-MM-DD, default: today): ").strip()

        # Parse dates
        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d") if start_date_str else (datetime.now(IST) - timedelta(days=7))
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d") if end_date_str else datetime.now(IST)
        except ValueError as e:
            self.logger.error(f"Invalid date format: {e}")
            start_date = datetime.now(IST) - timedelta(days=7)
            end_date = datetime.now(IST)
            
        return {
            'tickers': tickers,
            'timeframes': timeframes,
            'start_date': start_date,
            'end_date': end_date
        }

def update_pool_workflow(pool_path: str, target_end_date: str = None, provider_name: str = 'upstox',
                        backup: bool = True, dry_run: bool = False, validate_only: bool = False,
                        yes_flag: bool = False):
    """
    Update existing data pool with incremental data fetch

    Args:
        pool_path: Path to existing pool directory
        target_end_date: Target end date (default: today)
        provider_name: Data provider to use
        backup: Create backup before merge
        dry_run: Preview changes without executing
        validate_only: Only validate pool integrity
        yes_flag: Skip confirmation prompt (for unattended updates)

    Returns:
        True if successful, False otherwise
    """
    from .pool_inspector import inspect_pool, print_pool_summary
    from .gap_calculator import calculate_gaps, print_gap_report
    from .data_merger import merge_parquet_files

    logger = logging.getLogger("DataFetcher")

    print("\n" + "="*70)
    print(">>> INCREMENTAL POOL UPDATE WORKFLOW")
    print("="*70)

    try:
        # Step 1: Inspect pool
        logger.info("Step 1: Inspecting existing pool...")
        pool_metadata = inspect_pool(pool_path, validate=True)
        print_pool_summary(pool_metadata)

        if validate_only:
            logger.info("✅ Validation complete (--validate-only mode)")
            return True

        # Step 2: Calculate gaps
        logger.info("\nStep 2: Calculating gaps...")
        gap_report = calculate_gaps(pool_metadata, target_end_date=target_end_date)
        print_gap_report(gap_report)

        if dry_run:
            logger.info("✅ Dry-run complete (no changes made)")
            return True

        # Step 3: Confirmation (skip if --yes flag)
        if not yes_flag:
            print("\n" + "-"*70)
            response = input("❓ Proceed with update? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                logger.info("❌ Update cancelled by user")
                return False
        else:
            logger.info("Proceeding with update (--yes flag provided)")
            print()  # Empty line for formatting

        # Step 4: Initialize data fetcher
        logger.info("\nStep 3: Initializing data fetcher...")
        fetcher = DataFetcher(provider_name=provider_name)

        # Step 5: Fetch missing data for each ticker/timeframe
        logger.info("\nStep 4: Fetching missing data...")
        new_data_map = {}  # (ticker, timeframe) -> DataFrame

        total_gaps = len(gap_report.gaps)
        for i, ((ticker, timeframe), (gap_start, gap_end)) in enumerate(gap_report.gaps.items(), 1):
            logger.info(f"   [{i}/{total_gaps}] Fetching {ticker} @ {timeframe} from {gap_start.date()} to {gap_end.date()}")

            try:
                df = fetcher.provider.fetch_historical_data(ticker, gap_start, gap_end, timeframe)

                if df.empty:
                    logger.warning(f"   ⚠️  No data returned for {ticker} @ {timeframe}")
                else:
                    new_data_map[(ticker, timeframe)] = df
                    logger.info(f"   ✅ Fetched {len(df):,} records")

            except Exception as e:
                logger.error(f"   ❌ Error fetching {ticker} @ {timeframe}: {str(e)}")

        # Step 6: Merge data files
        logger.info(f"\nStep 5: Merging data files...")
        merge_results = {}

        for (ticker, timeframe), new_df in new_data_map.items():
            # Find old file path (handle both ticker-first and timeframe-first structures)
            pool_path_obj = Path(pool_path)

            # Try ticker-first structure first
            old_file = pool_path_obj / ticker / f"{timeframe}.parquet"
            if not old_file.exists():
                # Try timeframe-first structure
                old_file = pool_path_obj / timeframe / f"{ticker}.parquet"

            if not old_file.exists():
                logger.error(f"   ❌ Old file not found for {ticker} @ {timeframe}")
                merge_results[(ticker, timeframe)] = False
                continue

            try:
                success = merge_parquet_files(
                    str(old_file),
                    new_df,
                    strategy='append',
                    backup=backup,
                    validate=True
                )
                merge_results[(ticker, timeframe)] = success
            except Exception as e:
                logger.error(f"   ❌ Merge failed for {ticker} @ {timeframe}: {str(e)}")
                merge_results[(ticker, timeframe)] = False

        # Step 7: Summary
        successful = sum(1 for v in merge_results.values() if v)
        failed = len(merge_results) - successful

        print("\n" + "="*70)
        print("📊 UPDATE SUMMARY")
        print("="*70)
        print(f"Total files updated: {len(merge_results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")

        if failed == 0:
            print(f"\n🎉 Pool update complete!")
            print(f"   Old range: {pool_metadata.date_range[0]} to {pool_metadata.date_range[1]}")
            print(f"   New range: {pool_metadata.date_range[0]} to {target_end_date or datetime.now().strftime('%Y-%m-%d')}")
            print("="*70)
            return True
        else:
            print(f"\n⚠️  Update completed with {failed} failures")
            print("="*70)
            return False

    except Exception as e:
        logger.error(f"❌ Update workflow failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main(provider=None, timeframe=None, days=None, force_token_refresh=False):
    """
    Main function to run the data fetcher.
    If provider, timeframe, or days are provided, it uses those values; otherwise, it runs interactively.

    Args:
        provider: Name of data provider to use (upstox/zerodha)
        timeframe: Comma-separated list of timeframes to process
        days: Number of days to fetch data for
        force_token_refresh: Whether to force refresh of access token
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"    )
    logger = logging.getLogger("DataFetcher")
    
    print("\nWELCOME TO ENHANCED DATA PULL INTERFACE!\n")
    
    # Use the passed provider if provided; otherwise, prompt for input.
    if provider is None:
        provider_name = input("Which data provider would you like to use? (upstox/zerodha/binance, default: upstox): ").strip().lower()
        if not provider_name:
            provider_name = 'upstox'
    else:
        provider_name = provider

    if provider_name not in ['upstox', 'zerodha', 'binance']:
        logger.error(f"Unsupported provider: {provider_name}. Using upstox instead.")
        provider_name = 'upstox'
    
    try:        # Handle token refresh if requested
        if force_token_refresh:
            from .token_manager import clear_provider_token
            
            if clear_provider_token(provider_name):
                logger.info(f"Successfully cleared {provider_name} tokens to force refresh")
            else:
                logger.warning(f"Failed to clear {provider_name} tokens or no tokens found")
            
        # Initialize data fetcher with the selected provider
        fetcher = DataFetcher(provider_name=provider_name)
        
        # If timeframe or days are provided, override interactive inputs:
        if timeframe is not None or days is not None:
            # Use default tickers from config if not provided via another mechanism
            tickers = fetcher.config.get('DEFAULT_TICKER', ["RELIANCE", "TCS", "INFY"])
            # Parse the comma-separated timeframe string
            if timeframe is not None:
                timeframes = [tf.strip() for tf in timeframe.split(",")]
            else:
                timeframes = fetcher.config.get('DEFAULT_TIMEFRAME', ['1m'])
            # Calculate dates based on days if provided
            if days is not None:
                start_date = datetime.now(IST) - timedelta(days=days)
                end_date = datetime.now(IST)
            else:
                # Fall back to defaults
                start_date = datetime.now(IST) - timedelta(days=7)
                end_date = datetime.now(IST)
        else:
            # Otherwise, prompt the user for inputs interactively
            inputs = fetcher.get_user_inputs()
            tickers = inputs['tickers']
            timeframes = inputs['timeframes']
            start_date = inputs['start_date']
            end_date = inputs['end_date']
        
        # Fetch historical data
        result = fetcher.fetch_historical_data(
            tickers=tickers,
            timeframes=timeframes,
            start_date=start_date,
            end_date=end_date
        )
        
        total_files = sum(len(tf_dict) for tf_dict in result.values())
        print(f"\nFetched data for {len(result)} tickers across {len(timeframes)} timeframes.")
        print(f"Total files saved: {total_files}")
        
    except Exception as e:
        logger.error(f"Error in data fetching: {e}", exc_info=True)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Data Fetcher - Fetch or update historical market data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch mode (original behavior)
  python data_fetcher.py --mode fetch
  python data_fetcher.py --mode fetch --provider upstox --timeframe 1m,5m --days 7

  # Update mode (incremental update)
  python data_fetcher.py --mode update --pool-path data/pools/2022-01-01_to_2025-08-31/
  python data_fetcher.py --mode update --pool-path data/pools/2022-01-01_to_2025-08-31/ --extend-to 2025-10-08
  python data_fetcher.py --mode update --pool-path data/pools/2022-01-01_to_2025-08-31/ --dry-run
  python data_fetcher.py --mode update --pool-path data/pools/2022-01-01_to_2025-08-31/ --validate-only
        """
    )

    # Mode selection
    parser.add_argument(
        '--mode',
        choices=['fetch', 'update'],
        default='fetch',
        help='Operation mode: fetch (new data) or update (incremental)'
    )

    # Common arguments
    parser.add_argument('--provider', help='Data provider (upstox/zerodha/binance)')
    parser.add_argument('--force-token-refresh', action='store_true', help='Force refresh access token')

    # Fetch mode arguments
    parser.add_argument('--timeframe', help='Comma-separated timeframes (e.g., 1m,5m)')
    parser.add_argument('--days', type=int, help='Number of days to fetch')

    # Update mode arguments
    parser.add_argument('--pool-path', help='Path to existing pool to update')
    parser.add_argument('--extend-to', help='Target end date (YYYY-MM-DD, default: today)')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without executing')
    parser.add_argument('--validate-only', action='store_true', help='Only validate pool integrity')
    parser.add_argument('--no-backup', action='store_true', help='Skip backup creation')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation prompt (for unattended updates)')

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    if args.mode == 'update':
        # Update mode workflow
        if not args.pool_path:
            print("❌ Error: --pool-path is required for update mode")
            print("\nExample: python data_fetcher.py --mode update --pool-path data/pools/2022-01-01_to_2025-08-31/")
            sys.exit(1)

        provider_name = args.provider or 'upstox'
        success = update_pool_workflow(
            pool_path=args.pool_path,
            target_end_date=args.extend_to,
            provider_name=provider_name,
            backup=not args.no_backup,
            dry_run=args.dry_run,
            validate_only=args.validate_only,
            yes_flag=args.yes
        )

        sys.exit(0 if success else 1)

    else:
        # Fetch mode (original behavior)
        main(
            provider=args.provider,
            timeframe=args.timeframe,
            days=args.days,
            force_token_refresh=args.force_token_refresh
        )
    
