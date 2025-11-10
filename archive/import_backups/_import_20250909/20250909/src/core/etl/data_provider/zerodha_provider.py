# src/etl/data_providers/zerodha_provider.py
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
import os
import csv
import time
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .base_provider import DataProvider
from ..token_manager import load_provider_token, save_provider_token
from kiteconnect import KiteConnect, KiteTicker
from config import ZERODHA_CONFIG

class ZerodhaDataProvider(DataProvider):
    """Zerodha implementation of the DataProvider interface."""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.kite = None
        self.symbol_token_map = {}
        self.token_symbol_map = {}
    
    def authenticate(self) -> bool:
        """Authenticate with Zerodha's Kite API."""
        try:
            # Check for required API credentials first
            api_key = self.config.get('API_KEY')
            api_secret = self.config.get('API_SECRET')
            
            if not api_key or not api_secret:
                raise ValueError(
                    "❌ Missing Zerodha API credentials!\n"
                    "Please set these environment variables:\n"
                    "  • ZERODHA_API_KEY=your_api_key\n"
                    "  • ZERODHA_API_SECRET=your_api_secret\n"
                    "Refer to docs/BROKER_SETUP.md for detailed setup instructions."
                )
            
            # Check for existing access token
            access_token = load_provider_token('zerodha')

            if access_token:
                # Use existing access token
                self.kite = KiteConnect(api_key=api_key)
                self.kite.set_access_token(access_token)
            else:
                # Initiate auth flow for new token
                print(f"\n\nAUTH site link is https://kite.zerodha.com/connect/login?v=3&api_key={api_key}")
                request_token = input("\nEnter request token: ").strip()

                # Generate access token
                self.kite = KiteConnect(api_key=api_key)
                resp = self.kite.generate_session(
                    request_token, 
                    api_secret=api_secret
                )
                access_token = resp["access_token"]

                # Save token with date information
                save_provider_token('zerodha', access_token, 
                                  request_token=request_token, api_key=api_key)

                # Set the access token
                self.kite.set_access_token(access_token)

            # Verify authentication
            try:
                self.kite.profile()
                self.authenticated = True
                return True
            except Exception as e:
                self.logger.error(f"Authentication verification failed: {e}")
                return False

        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            return False
        
    def _create_kite_connect_instance(self, candidate_key: str, is_access_key: bool = False, timeout: int = 20) -> KiteConnect:
        """Create and initialize a KiteConnect instance."""
        api_key = self.config.get('API_KEY')
        api_secret = self.config.get('API_SECRET')
        key_csv_file = self.config.get('KEY_CSV_LOCATION', 'access_token.csv')
        
        kite_instance = KiteConnect(api_key=api_key, timeout=timeout)
        
        if is_access_key:
            access_key = candidate_key
        else:
            # Generate access token from request key
            access_key = kite_instance.generate_session(
                candidate_key, 
                api_secret=api_secret
            )["access_token"]
            
            # Save access token to file
            os.makedirs(os.path.dirname(key_csv_file), exist_ok=True)
            with open(key_csv_file, 'w+') as f:
                writer = csv.writer(f)
                writer.writerow([access_key])
        
        kite_instance.set_access_token(access_key)
        
        return kite_instance
    
    def map_timeframe(self, standard_timeframe: str) -> str:
        """Map standardized timeframe to Zerodha-specific timeframe."""
        timeframe_mapping = {
            '1m': 'minute',
            '3m': '3minute',
            '5m': '5minute',
            '10m': '10minute',
            '15m': '15minute',
            '30m': '30minute',
            '1h': 'hour',
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
        instruments_csv = self.config.get('INSTRUMENTS_CSV')
        if instruments_csv and Path(instruments_csv).exists():
            self.logger.info(f"Loading Zerodha instruments from CSV: {instruments_csv}")
            instruments = pd.read_csv(instruments_csv)
        else:
            self.logger.info("CSV not found; fetching Zerodha instruments from Kite API.")
            segment = self.config.get('SEGMENT', 'NSE')
            instruments = pd.DataFrame(self.kite.instruments(segment))

        if symbols:
            instruments = instruments[instruments['tradingsymbol'].isin(symbols)]

        # Build the mapping
        for _, row in instruments.iterrows():
            self.symbol_token_map[row['tradingsymbol']] = row['instrument_token']
            self.token_symbol_map[row['instrument_token']] = row['tradingsymbol']

        return instruments
    
    def symbol_to_instrument_id(self, symbol: str) -> int:
        """Convert symbol to Zerodha instrument token."""
        if not self.authenticated:
            self.authenticate()
            
        # Check if symbol is already in the mapping
        if symbol in self.symbol_token_map:
            return self.symbol_token_map[symbol]
            
        # If not, fetch instrument details
        instruments = self.fetch_instrument_details([symbol])
        
        if instruments.empty:
            return None
            
        match = instruments[instruments['tradingsymbol'] == symbol]
        
        if match.empty:
            self.logger.error(f"Symbol '{symbol}' not found in instruments data.")
            return None
            
        instrument_token = match.iloc[0]['instrument_token']
        
        # Cache the mapping
        self.symbol_token_map[symbol] = instrument_token
        self.token_symbol_map[instrument_token] = symbol
        
        return instrument_token
    
    def fetch_historical_data(self, symbol: str, start_date: datetime, end_date: datetime, 
                              timeframe: str) -> pd.DataFrame:
        """Fetch historical data from Zerodha's Kite API."""
        if not self.authenticated:
            self.authenticate()
            
        instrument_token = self.symbol_to_instrument_id(symbol)
        if not instrument_token:
            self.logger.error(f"Could not find instrument token for symbol: {symbol}")
            return pd.DataFrame()
        
        kite_timeframe = self.map_timeframe(timeframe)
        
        # Zerodha has limits on the number of candles per request
        # Different limits for different timeframes
        limits_dict = {
            'minute': 60,
            '3minute': 100,
            '5minute': 100,
            '10minute': 100,
            '15minute': 200,
            '30minute': 200,
            'hour': 400,
            'day': 2000,
            'week': 2000,
            'month': 2000
        }
        
        limit = limits_dict.get(kite_timeframe, 60)
        days_diff = (end_date - start_date).days + 1
        
        # Calculate number of iterations needed
        n_iters = days_diff // limit
        rem_days = days_diff % limit
        
        try:
            # Fetch data in chunks
            df_list = []
            temp_end = end_date
            
            for i in range(n_iters):
                temp_start = temp_end - timedelta(days=limit-1)
                self.logger.info(f"Fetching data chunk {i+1}/{n_iters} for {symbol} from {temp_start.date()} to {temp_end.date()}")
                
                data = self.kite.historical_data(
                    instrument_token,
                    from_date=temp_start.strftime('%Y-%m-%d'),
                    to_date=temp_end.strftime('%Y-%m-%d'),
                    interval=kite_timeframe,
                    continuous=False,
                    oi=False
                )
                
                if data:
                    df_list.append(pd.DataFrame(data))
                
                # Add delay to avoid rate limiting
                time.sleep(0.25)
                temp_end = temp_start - timedelta(days=1)
            
            # Get remaining days
            if rem_days > 0 and temp_end >= start_date:
                temp_start = max(start_date, temp_end - timedelta(days=rem_days-1))
                self.logger.info(f"Fetching final chunk for {symbol} from {temp_start.date()} to {temp_end.date()}")
                
                data = self.kite.historical_data(
                    instrument_token,
                    from_date=temp_start.strftime('%Y-%m-%d'),
                    to_date=temp_end.strftime('%Y-%m-%d'),
                    interval=kite_timeframe,
                    continuous=False,
                    oi=False
                )
                
                if data:
                    df_list.append(pd.DataFrame(data))
            
            # Combine all chunks
            if not df_list:
                return pd.DataFrame()
                
            df = pd.concat(df_list)
            
            # Add symbol column
            df['ticker'] = symbol
            
            # Apply standard data normalization
            df = self.normalize_data(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()