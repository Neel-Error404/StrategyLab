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
                             output_dir: Optional[Path] = None) -> Dict[str, Dict[str, Path]]:
        """
        Fetch historical data for multiple tickers and timeframes.
        
        Args:
            tickers: List of ticker symbols
            timeframes: List of timeframes (e.g., '1m', '5m', 'day')
            start_date: Start date for the data
            end_date: End date for the data
            output_dir: Output directory (defaults to DATA_POOL_DIR/current_date)
            
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
        
        result = {}
        
        # Process each ticker
        for ticker in tickers:
            result[ticker] = {}
            
            # Process each timeframe
            for timeframe in timeframes:
                # Create timeframe directory
                timeframe_folder = self.config.get('TIMEFRAME_FOLDERS', {}).get(timeframe, timeframe)
                timeframe_dir = output_dir / timeframe_folder
                timeframe_dir.mkdir(parents=True, exist_ok=True)
                
                # Fetch data
                self.logger.info(f"Fetching {timeframe} data for {ticker} from {start_date.date()} to {end_date.date()}")
                
                try:
                    df = self.provider.fetch_historical_data(ticker, start_date, end_date, timeframe)
                    
                    if df.empty:
                        self.logger.warning(f"No data returned for {ticker} at {timeframe} timeframe")
                        continue
                    
                    # Create filename
                    filename = f"{ticker}_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}.csv"
                    file_path = timeframe_dir / filename
                    
                    # Save data
                    df.to_csv(file_path, index=False)
                    
                    self.logger.info(f"Saved {len(df)} records for {ticker} at {timeframe} timeframe to {file_path}")
                    
                    result[ticker][timeframe] = file_path
                    
                except Exception as e:
                    self.logger.error(f"Error fetching data for {ticker} at {timeframe} timeframe: {e}")
            
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
        provider_name = input("Which data provider would you like to use? (upstox/zerodha, default: upstox): ").strip().lower()
        if not provider_name:
            provider_name = 'upstox'
    else:
        provider_name = provider

    if provider_name not in ['upstox', 'zerodha']:
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
    main()
    