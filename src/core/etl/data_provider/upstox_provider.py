# src/etl/data_providers/upstox_provider.py
import pandas as pd
from datetime import datetime, timedelta
import requests
import logging
from pathlib import Path

from .base_provider import DataProvider
from ..token_manager import load_provider_token, save_provider_token
from config import create_session, BACKTESTER_CONFIG

class UpstoxDataProvider(DataProvider):
    """Upstox implementation of the DataProvider interface."""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.session = None
        self.instruments_df = None
    
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
    
    def map_timeframe(self, standard_timeframe: str) -> str:
        """Map standardized timeframe to Upstox-specific timeframe."""
        timeframe_mapping = {
            '1m': '1minute',
            '5m': '5minute',  # Not supported by Upstox
            '15m': '15minute',  # Not supported by Upstox
            '30m': '30minute',
            '1h': '60minute',  # Not supported by Upstox
            'day': 'day',
            'week': 'week',
            'month': 'month'
        }
        
        return timeframe_mapping.get(standard_timeframe, standard_timeframe)
    
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
        """Fetch historical data from Upstox API."""
        if not self.authenticated:
            self.authenticate()
            
        instrument_id = self.symbol_to_instrument_id(symbol)
        if not instrument_id:
            self.logger.error(f"Could not find instrument key for ticker: {symbol}")
            return pd.DataFrame()
        
        timeframe = self.map_timeframe(timeframe)
        
        # Upstox API has a limit on the number of days per request
        MAX_DAYS_PER_REQUEST = self.config.get('MAX_DAYS_PER_REQUEST', 200)
        
        # Convert dates to string format
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        full_data = []
        current_start_date = start_date
        
        while current_start_date <= end_date:
            chunk_end_date = min(current_start_date + timedelta(days=MAX_DAYS_PER_REQUEST - 1), end_date)
            chunk_start_str = current_start_date.strftime("%Y-%m-%d")
            chunk_end_str = chunk_end_date.strftime("%Y-%m-%d")

            self.logger.info(f"Pulling data: Ticker={symbol}, Time Frame={timeframe}, Start={chunk_start_str}, End={chunk_end_str}")

            url = f"https://api.upstox.com/v2/historical-candle/{instrument_id}/{timeframe}/{chunk_end_str}/{chunk_start_str}"

            response = self.session.get(url)
            if response.status_code == 200:
                data = response.json()
                candles = data.get('data', {}).get('candles', [])
                
                for candle in candles:
                    full_data.append({
                        'timestamp': candle[0],
                        'open': candle[1],
                        'high': candle[2],
                        'low': candle[3],
                        'close': candle[4],
                        'volume': candle[5],
                        'ticker': symbol
                    })
            else:
                self.logger.error(f"Error fetching data for {symbol}: {response.text}")
                return pd.DataFrame()

            # Update current_start_date for the next chunk
            current_start_date = chunk_end_date + timedelta(days=1)
        
        if not full_data:
            return pd.DataFrame()
            
        df = pd.DataFrame(full_data)
        
        # Apply standard data normalization
        df = self.normalize_data(df)
        
        return df