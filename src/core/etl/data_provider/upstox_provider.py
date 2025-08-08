# src/etl/data_providers/upstox_provider.py
import pandas as pd
from datetime import datetime, timedelta
import requests
import logging
import time
from pathlib import Path

from .base_provider import DataProvider
from ..token_manager import load_provider_token, save_provider_token
from config import create_session, BACKTESTER_CONFIG

class UpstoxDataProvider(DataProvider):
    """Upstox V3 API implementation of the DataProvider interface."""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.session = None
        self.instruments_df = None
        self.provider_name = 'upstox_v3'  # Indicate V3 API usage
    
    def authenticate(self) -> bool:
        """Authenticate with Upstox API."""
        access_token = load_provider_token('upstox')
        if not access_token:
            auth_code = self._get_auth_code()
            access_token = self._get_access_token(auth_code)

        if not access_token:
            self.logger.error("Unable to authenticate with Upstox. Exiting.")
            return False

        # Create a session with the access token
        self.session = create_session(access_token)
        self.authenticated = True
        return True
    
    def _get_auth_code(self):
        """Generates the authorization URL and retrieves the auth code from the user."""
        CLIENT_ID = self.config.get('CLIENT_ID')
        REDIRECT_URI = self.config.get('REDIRECT_URI')
        
        if not CLIENT_ID:
            raise ValueError(
                "❌ UPSTOX_CLIENT_ID environment variable is not set!\n"
                "Please set your Upstox API credentials:\n"
                "  • UPSTOX_CLIENT_ID=your_client_id\n"
                "  • UPSTOX_CLIENT_SECRET=your_client_secret\n"
                "Refer to docs/BROKER_SETUP.md for detailed setup instructions."
            )
        
        if not REDIRECT_URI:
            REDIRECT_URI = "https://127.0.0.1:5000/"  # Default fallback
        
        auth_url = (
            f"https://api.upstox.com/v2/login/authorization/dialog"
            f"?client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}"
            f"&response_type=code&state=random_state_123"
        )
        print("Please visit the following URL to authorize the application:")
        print(auth_url)
        print("\nAfter authorization, you will be redirected to your redirect_uri with a 'code' parameter.")
        auth_code = input("Enter the authorization code from the URL: ").strip()
        return auth_code

    def _get_access_token(self, auth_code):
        """Exchanges the auth code for an access token and saves it to a file."""
        CLIENT_ID = self.config.get('CLIENT_ID')
        CLIENT_SECRET = self.config.get('CLIENT_SECRET')
        REDIRECT_URI = self.config.get('REDIRECT_URI')
        
        if not CLIENT_ID or not CLIENT_SECRET:
            raise ValueError(
                "❌ Missing Upstox API credentials!\n"
                "Please set these environment variables:\n"
                "  • UPSTOX_CLIENT_ID=your_client_id\n"
                "  • UPSTOX_CLIENT_SECRET=your_client_secret\n"
                "Refer to docs/BROKER_SETUP.md for detailed setup instructions."
            )
        
        if not REDIRECT_URI:
            REDIRECT_URI = "https://127.0.0.1:5000/"  # Default fallback
        
        token_url = self.config.get('TOKEN_URL', "https://api.upstox.com/v2/login/authorization/token")
        payload = {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "redirect_uri": REDIRECT_URI,
            "code": auth_code,
            "grant_type": "authorization_code"
        }
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json'
        }
        response = requests.post(token_url, headers=headers, data=payload)
        if response.status_code == 200:
            token_data = response.json()
            save_provider_token('upstox', token_data)
            return token_data.get("access_token")
        else:
            self.logger.error(f"Error fetching access token: {response.text}")
            return None
    
    def map_timeframe(self, standard_timeframe: str) -> dict:
        """Map standardized timeframe to Upstox V3 API unit and interval."""
        # Get mapping from config or use defaults
        mappings = self.config.get('TIMEFRAME_MAPPINGS', {})
        
        if standard_timeframe in mappings:
            return mappings[standard_timeframe]
        
        # Fallback mappings if not in config
        fallback_mappings = {
            '1m': {'unit': 'minutes', 'interval': '1'},
            '2m': {'unit': 'minutes', 'interval': '2'},
            '3m': {'unit': 'minutes', 'interval': '3'},
            '5m': {'unit': 'minutes', 'interval': '5'},
            '10m': {'unit': 'minutes', 'interval': '10'},
            '15m': {'unit': 'minutes', 'interval': '15'},
            '30m': {'unit': 'minutes', 'interval': '30'},
            '1h': {'unit': 'hours', 'interval': '1'},
            '2h': {'unit': 'hours', 'interval': '2'},
            'day': {'unit': 'days', 'interval': '1'},
            'week': {'unit': 'weeks', 'interval': '1'},
            'month': {'unit': 'months', 'interval': '1'},
            # Legacy compatibility
            '1minute': {'unit': 'minutes', 'interval': '1'},
            '30minute': {'unit': 'minutes', 'interval': '30'}
        }
        
        mapping = fallback_mappings.get(standard_timeframe)
        if not mapping:
            self.logger.warning(f"Unknown timeframe '{standard_timeframe}', defaulting to 1 minute")
            mapping = {'unit': 'minutes', 'interval': '1'}
        
        return mapping
    
    def get_available_symbols(self) -> list:
        """Get list of available trading symbols."""
        if not self.authenticated:
            self.authenticate()
            
        try:
            instruments = self.fetch_instrument_details()
            return instruments['tradingsymbol'].tolist()
        except Exception as e:
            self.logger.error(f"Error fetching available symbols: {e}")
            return []
    
    def fetch_instrument_details(self, symbols=None) -> pd.DataFrame:
        if not self.authenticated:
            self.authenticate()

        # If we already loaded the DataFrame, reuse it
        if self.instruments_df is not None:
            df = self.instruments_df
        else:
            instruments_csv = self.config.get('INSTRUMENTS_CSV')
            df = pd.read_csv(instruments_csv)

            # Filter out any exchanges you don't want (e.g., BSE)
            allowed_exchanges = ["NSE_FO","NSE_EQ", "NSE_INDEX", "MCX_FO", "MCX_INDEX"]
            df = df[df['exchange'].isin(allowed_exchanges)]

            self.instruments_df = df  # Cache for reuse

        # Optionally filter by specific tickers
        if symbols:
            df = df[df['tradingsymbol'].isin(symbols)]

        return df


    def symbol_to_instrument_id(self, symbol: str) -> str:
        if not self.authenticated:
            self.authenticate()

        instruments = self.fetch_instrument_details([symbol])

        if instruments.empty:
            self.logger.error(f"Symbol '{symbol}' not found in allowed instruments data.")
            return None

        match = instruments[instruments['tradingsymbol'] == symbol]

        if match.empty:
            self.logger.error(f"Symbol '{symbol}' not found in instruments data after filtering.")
            return None

        return match.iloc[0]['instrument_key']

    
    def fetch_historical_data(self, symbol: str, start_date: datetime, end_date: datetime, 
                             timeframe: str) -> pd.DataFrame:
        """Fetch historical data from Upstox API month by month."""
        if not self.authenticated:
            self.authenticate()
            
        instrument_id = self.symbol_to_instrument_id(symbol)
        if not instrument_id:
            self.logger.error(f"Could not find instrument key for ticker: {symbol}")
            return pd.DataFrame()
        
        timeframe_mapping = self.map_timeframe(timeframe)
        unit = timeframe_mapping['unit']
        interval = timeframe_mapping['interval']
        
        # Get historical API base URL
        historical_base = self.config.get('HISTORICAL_API_BASE', 'https://api.upstox.com/v3/historical-candle')
        
        full_data = []
        current_start_date = start_date
        
        # Smart chunking strategy based on empirical API limits
        chunk_days = self._get_optimal_chunk_size(unit, interval)
        
        # Progress tracking variables
        total_days = (end_date - start_date).days + 1
        days_processed = 0
        last_month_logged = None
        failed_days = []
            
        while current_start_date <= end_date:
            chunk_end_date = min(current_start_date + timedelta(days=chunk_days - 1), end_date)
            chunk_start_str = current_start_date.strftime("%Y-%m-%d")
            chunk_end_str = chunk_end_date.strftime("%Y-%m-%d")

            # Progress logging at monthly intervals
            current_month = current_start_date.strftime('%Y-%m')
            if current_month != last_month_logged:
                progress_pct = (days_processed / total_days) * 100
                self.logger.info(f"📅 Monthly Progress - {symbol}: Processing {current_month} ({progress_pct:.1f}% complete)")
                last_month_logged = current_month
            
            # Make API call with retry logic
            chunk_data = self._fetch_chunk_with_retry(historical_base, instrument_id, unit, interval, 
                                                     chunk_start_str, chunk_end_str, symbol)
            
            if chunk_data:
                full_data.extend(chunk_data)
                success_days = (chunk_end_date - current_start_date).days + 1
                days_processed += success_days
            else:
                # Track failed days but continue
                failed_days.append(f"{chunk_start_str}_to_{chunk_end_str}")
                days_processed += (chunk_end_date - current_start_date).days + 1

            # Move to next chunk
            current_start_date = chunk_end_date + timedelta(days=1)
        
        # Final progress report
        if failed_days:
            self.logger.warning(f"⚠️  {symbol}: {len(failed_days)} date ranges failed: {failed_days[:3]}{'...' if len(failed_days) > 3 else ''}")
        
        success_pct = ((days_processed - len(failed_days)) / days_processed * 100) if days_processed > 0 else 0
        self.logger.info(f"✅ {symbol}: Completed fetching {len(full_data)} total candles ({success_pct:.1f}% success rate)")
        
        if not full_data:
            return pd.DataFrame()
            
        df = pd.DataFrame(full_data)
        
        # Apply standard data normalization
        df = self.normalize_data(df)
        
        return df
    
    def _get_optimal_chunk_size(self, unit: str, interval: str) -> int:
        """
        Determine optimal chunk size based on timeframe and empirical API limits.
        
        Args:
            unit: Time unit (e.g., 'minutes', 'hours', 'days')
            interval: Interval value (e.g., '1', '5', '30')
            
        Returns:
            Number of days per chunk
        """
        if unit == 'minutes':
            interval_int = int(interval)
            if interval_int <= 15:
                # For 1-15 minute intervals: use 1-day chunks (confirmed working)
                return 1
            elif interval_int <= 60:
                # For 16-60 minute intervals: try 7-day chunks
                return 7
            else:
                # For higher minute intervals: try 30-day chunks
                return 30
        elif unit == 'hours':
            # For hourly data: 30-day chunks
            return 30
        else:
            # For daily/weekly/monthly: 90-day chunks
            return 90
    
    def _fetch_chunk_with_retry(self, historical_base: str, instrument_id: str, 
                               unit: str, interval: str, start_str: str, end_str: str, 
                               symbol: str, max_retries: int = 3) -> list:
        """
        Fetch data chunk with retry logic and exponential backoff.
        
        Args:
            historical_base: Base API URL
            instrument_id: Instrument identifier
            unit: Time unit
            interval: Interval value
            start_str: Start date string
            end_str: End date string
            symbol: Symbol name for logging
            max_retries: Maximum retry attempts
            
        Returns:
            List of candle data or empty list on failure
        """
        url = f"{historical_base}/{instrument_id}/{unit}/{interval}/{end_str}/{start_str}"
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get('status') == 'success':
                        candles = data.get('data', {}).get('candles', [])
                        
                        if candles:
                            # Convert to our format
                            chunk_data = []
                            for candle in candles:
                                chunk_data.append({
                                    'timestamp': candle[0],
                                    'open': candle[1],
                                    'high': candle[2],
                                    'low': candle[3],
                                    'close': candle[4],
                                    'volume': candle[5],
                                    'ticker': symbol
                                })
                            return chunk_data
                        else:
                            # No data for this date range (e.g., weekend/holiday)
                            return []
                    else:
                        error_msg = data.get('message', 'Unknown API error')
                        if 'too high for interval' in error_msg.lower():
                            # Try smaller chunk if this is the first attempt
                            if attempt == 0 and unit == 'minutes' and int(interval) <= 15:
                                self.logger.warning(f"Chunk too large for {symbol} {start_str}-{end_str}, this shouldn't happen with 1-day chunks")
                            return []  # Don't retry for this specific error
                        else:
                            self.logger.warning(f"API error for {symbol} (attempt {attempt + 1}): {error_msg}")
                else:
                    self.logger.warning(f"HTTP {response.status_code} for {symbol} (attempt {attempt + 1}): {response.text[:200]}")
                
                # Exponential backoff before retry
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 1, 2, 4 seconds
                    time.sleep(wait_time)
                    
            except Exception as e:
                self.logger.error(f"Exception fetching {symbol} {start_str}-{end_str} (attempt {attempt + 1}): {e}")
                
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        # All retries failed
        self.logger.error(f"❌ Failed to fetch {symbol} for {start_str} to {end_str} after {max_retries} attempts")
        return []
    
    def fetch_expiry_dates(self, instrument_key: str) -> list:
        """
        Fetch all available expiry dates for a given instrument.
        
        Args:
            instrument_key: The unique identifier for the financial instrument
            
        Returns:
            List of expiry dates in YYYY-MM-DD format
        """
        if not self.authenticated:
            self.authenticate()
        
        try:
            # Use query parameter format as per API documentation examples
            expiry_url = self.config.get('EXPIRY_API_URL', 'https://api.upstox.com/v2/expired-instruments/expiries')
            params = {'instrument_key': instrument_key}
            
            self.logger.info(f"Fetching expiry dates for instrument: {instrument_key}")
            response = self.session.get(expiry_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    expiry_dates = data.get('data', [])
                    self.logger.info(f"Found {len(expiry_dates)} expiry dates for {instrument_key}")
                    return expiry_dates
                else:
                    self.logger.error(f"API returned error status for {instrument_key}: {data}")
                    return []
            else:
                self.logger.error(f"Error fetching expiry dates for {instrument_key}: {response.text}")
                return []
                
        except Exception as e:
            self.logger.error(f"Exception while fetching expiry dates for {instrument_key}: {e}")
            return []