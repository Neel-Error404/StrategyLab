# src/core/etl/data_provider/binance_provider.py
import pandas as pd
from datetime import datetime, timedelta
import logging
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from .base_provider import DataProvider

try:
    from binance.client import Client
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    Client = None


class BinanceDataProvider(DataProvider):
    """Binance API implementation of the DataProvider interface for cryptocurrency data."""
    
    # Top 30 cryptocurrency symbols with USDT pairs (based on 2025 trading volume & market cap)
    SUPPORTED_SYMBOLS = [
        # Top 10 - Highest volume and market cap
        'BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'BNBUSDT', 'SOLUSDT',
        'DOGEUSDT', 'ADAUSDT', 'TRXUSDT', 'AVAXUSDT', 'SHIBUSDT',
        
        # Top 11-20 - High volume platforms and DeFi
        'TONUSDT', 'MATICUSDT', 'DOTUSDT', 'BCHUSDT', 'NEARUSDT',
        'UNIUSDT', 'LTCUSDT', 'ATOMUSDT', 'LINKUSDT', 'FILUSDT',
        
        # Top 21-30 - Emerging and specialized platforms
        'ETCUSDT', 'APTUSDT', 'ARBUSDT', 'OPUSDT', 'HBARUSDT',
        'VETUSDT', 'FTMUSDT', 'USDCUSDT', 'TUSDUSDT', 'BUSDUSDT',
        
        # Additional high-volume pairs for comprehensive coverage
        'PEPEUSDT', 'WIFUSDT', 'FLOKIUSDT', 'BONKUSDT', 'INJUSDT'
    ]
    
    # Mapping from standard timeframes to Binance intervals
    TIMEFRAME_MAPPING = {
        '1m': Client.KLINE_INTERVAL_1MINUTE if Client else '1m',
        '3m': Client.KLINE_INTERVAL_3MINUTE if Client else '3m', 
        '5m': Client.KLINE_INTERVAL_5MINUTE if Client else '5m',
        '15m': Client.KLINE_INTERVAL_15MINUTE if Client else '15m',
        '30m': Client.KLINE_INTERVAL_30MINUTE if Client else '30m',
        '1h': Client.KLINE_INTERVAL_1HOUR if Client else '1h',
        '2h': Client.KLINE_INTERVAL_2HOUR if Client else '2h',
        '4h': Client.KLINE_INTERVAL_4HOUR if Client else '4h',
        '6h': Client.KLINE_INTERVAL_6HOUR if Client else '6h',
        '8h': Client.KLINE_INTERVAL_8HOUR if Client else '8h',
        '12h': Client.KLINE_INTERVAL_12HOUR if Client else '12h',
        'day': Client.KLINE_INTERVAL_1DAY if Client else '1d',
        '1d': Client.KLINE_INTERVAL_1DAY if Client else '1d',
        '3d': Client.KLINE_INTERVAL_3DAY if Client else '3d',
        'week': Client.KLINE_INTERVAL_1WEEK if Client else '1w',
        '1w': Client.KLINE_INTERVAL_1WEEK if Client else '1w',
        'month': Client.KLINE_INTERVAL_1MONTH if Client else '1M',
        '1M': Client.KLINE_INTERVAL_1MONTH if Client else '1M'
    }
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not BINANCE_AVAILABLE:
            raise ImportError(
                "python-binance library is not installed. "
                "Please install it using: pip install python-binance"
            )
        
        # Initialize Binance client without API credentials (public API access only)
        self.client = Client("", "")  # No API key needed for historical data
        self.provider_name = 'binance'
        
        # Set authenticated to True since we don't need auth for historical data
        self.authenticated = True
        
        self.logger.info("Binance data provider initialized for cryptocurrency data")
    
    def authenticate(self) -> bool:
        """
        Authenticate with Binance API.
        For historical data, no authentication is required.
        """
        try:
            # Test connection by getting server time
            server_time = self.client.get_server_time()
            if server_time:
                self.authenticated = True
                self.logger.info("Binance API connection successful")
                return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Binance API: {e}")
            self.authenticated = False
            return False
        
        return False
    
    def _calculate_chunk_size(self, timeframe: str) -> int:
        """
        Calculate optimal chunk size based on timeframe to respect API limits.
        
        Args:
            timeframe: Data timeframe
            
        Returns:
            Number of periods per chunk
        """
        # Binance allows up to 1500 klines per request (maximum limit)
        # Use 1500 for optimal API efficiency
        chunk_sizes = {
            '1m': 1500,    # ~25 hours
            '3m': 1500,    # ~75 hours  
            '5m': 1500,    # ~125 hours
            '15m': 1500,   # ~15.6 days
            '30m': 1500,   # ~31.3 days
            '1h': 1500,    # ~62.5 days
            '2h': 1500,    # ~125 days
            '4h': 1500,    # ~250 days
            '6h': 1500,    # ~375 days
            '12h': 1500,   # ~750 days
            'day': 1500,   # ~4.1 years
            '1d': 1500,    # ~4.1 years
            'week': 1500,  # ~28.8 years
            '1w': 1500,    # ~28.8 years
        }
        return chunk_sizes.get(timeframe, 1500)
    
    def _get_chunk_timedelta(self, timeframe: str, chunk_size: int) -> timedelta:
        """
        Calculate timedelta for a chunk based on timeframe.
        
        Args:
            timeframe: Data timeframe
            chunk_size: Number of periods in chunk
            
        Returns:
            Timedelta representing the chunk duration
        """
        timeframe_minutes = {
            '1m': 1,
            '3m': 3,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '2h': 120,
            '4h': 240,
            '6h': 360,
            '12h': 720,
            'day': 1440,
            '1d': 1440,
            'week': 10080,
            '1w': 10080,
        }
        
        minutes = timeframe_minutes.get(timeframe, 60)
        return timedelta(minutes=minutes * chunk_size)
    
    def _fetch_chunk(self, symbol: str, start_date: datetime, end_date: datetime, 
                     timeframe: str) -> List[Dict]:
        """
        Fetch a single chunk of data with rate limiting.
        
        Args:
            symbol: Normalized symbol
            start_date: Chunk start date
            end_date: Chunk end date
            timeframe: Data timeframe
            
        Returns:
            List of kline data
        """
        # Get Binance interval
        binance_interval = self.TIMEFRAME_MAPPING.get(timeframe, '1h')
        
        # Format dates for Binance API
        start_str = start_date.strftime("%d %b, %Y")
        end_str = end_date.strftime("%d %b, %Y")
        
        try:
            # Add rate limiting delay - more conservative for large fetches
            # 0.1 seconds = 600 requests/minute (safe under 6000 weight limit)
            time.sleep(0.1)
            
            self.logger.debug(f"Fetching chunk: {symbol} {timeframe} from {start_str} to {end_str}")
            
            klines = self.client.get_historical_klines(
                symbol, 
                binance_interval, 
                start_str, 
                end_str
            )
            
            return klines
            
        except Exception as e:
            self.logger.error(f"Failed to fetch chunk for {symbol}: {e}")
            # Add exponential backoff on error
            time.sleep(1)
            return []
    
    def _estimate_api_calls(self, start_date: datetime, end_date: datetime, timeframe: str) -> int:
        """
        Estimate number of API calls needed for the date range.
        
        Args:
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe
            
        Returns:
            Estimated number of API calls
        """
        total_days = (end_date - start_date).days
        chunk_size = self._calculate_chunk_size(timeframe)
        chunk_timedelta = self._get_chunk_timedelta(timeframe, chunk_size)
        
        # Calculate how many chunks needed
        estimated_chunks = total_days / chunk_timedelta.days
        return max(1, int(estimated_chunks))
    
    def _fetch_monthly_chunks(self, symbol: str, start_date: datetime, end_date: datetime, 
                              timeframe: str) -> pd.DataFrame:
        """
        Fetch data using monthly chunks for optimal performance on large date ranges.
        
        Args:
            symbol: Normalized symbol
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe
            
        Returns:
            Combined DataFrame from all monthly chunks
        """
        all_klines = []
        current_date = start_date.replace(day=1)  # Start at beginning of month
        month_count = 0
        
        # Calculate total months for progress tracking
        total_months = ((end_date.year - start_date.year) * 12 + 
                       (end_date.month - start_date.month)) + 1
        
        self.logger.info(f"Fetching {total_months} months of data using monthly chunks...")
        
        while current_date < end_date:
            month_count += 1
            
            # Calculate month boundaries
            if current_date.month == 12:
                next_month = current_date.replace(year=current_date.year + 1, month=1)
            else:
                next_month = current_date.replace(month=current_date.month + 1)
            
            month_end = min(next_month, end_date)
            
            # Estimate API calls for this month
            month_calls = self._estimate_api_calls(current_date, month_end, timeframe)
            
            self.logger.info(f"Fetching month {month_count}/{total_months}: "
                           f"{current_date.strftime('%Y-%m')} "
                           f"(~{month_calls} API calls)")
            
            # Fetch month data using regular chunking within the month
            month_data = self._fetch_month_data(symbol, current_date, month_end, timeframe)
            
            if month_data:
                all_klines.extend(month_data)
                self.logger.info(f"Month {month_count}: {len(month_data)} records fetched")
            else:
                self.logger.warning(f"No data for month {month_count}")
            
            current_date = next_month
        
        if not all_klines:
            self.logger.warning(f"No data returned for {symbol} across all monthly chunks")
            return pd.DataFrame()
        
        self.logger.info(f"Successfully fetched {len(all_klines)} total records across "
                        f"{month_count} months")
        return self._process_klines(all_klines, symbol)
    
    def _fetch_month_data(self, symbol: str, start_date: datetime, end_date: datetime, 
                          timeframe: str) -> List[Dict]:
        """
        Fetch data for a single month, handling internal chunking if needed.
        
        Args:
            symbol: Normalized symbol
            start_date: Month start date
            end_date: Month end date  
            timeframe: Data timeframe
            
        Returns:
            List of kline data for the month
        """
        month_klines = []
        chunk_size = self._calculate_chunk_size(timeframe)
        chunk_timedelta = self._get_chunk_timedelta(timeframe, chunk_size)
        
        current_start = start_date
        chunk_count = 0
        
        while current_start < end_date:
            chunk_end = min(current_start + chunk_timedelta, end_date)
            chunk_count += 1
            
            self.logger.debug(f"  Fetching chunk {chunk_count}: {current_start.date()} to {chunk_end.date()}")
            
            klines = self._fetch_chunk(symbol, current_start, chunk_end, timeframe)
            
            if klines:
                month_klines.extend(klines)
            
            current_start = chunk_end
        
        return month_klines
    
    def _load_exchange_info(self) -> Dict[str, Any]:
        """
        Load Binance exchange information and cache it.
        
        Returns:
            Exchange info dictionary
        """
        cache_file = Path(__file__).parent.parent.parent.parent / "config" / "binance_exchange_info.json"
        
        # Try to load from cache first
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    exchange_info = json.load(f)
                    # Check if cache is recent (less than 24 hours old)
                    if time.time() - exchange_info.get('cached_at', 0) < 86400:
                        return exchange_info
            except Exception as e:
                self.logger.warning(f"Failed to load cached exchange info: {e}")
        
        # Fetch fresh exchange info
        try:
            self.logger.info("Fetching Binance exchange information...")
            exchange_info = self.client.get_exchange_info()
            exchange_info['cached_at'] = time.time()
            
            # Cache the result
            cache_file.parent.mkdir(exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(exchange_info, f, indent=2)
                
            self.logger.info(f"Cached {len(exchange_info.get('symbols', []))} Binance symbols")
            return exchange_info
            
        except Exception as e:
            self.logger.error(f"Failed to fetch exchange info: {e}")
            return {}
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of all available symbols from Binance.
        
        Returns:
            List of available symbols
        """
        exchange_info = self._load_exchange_info()
        symbols = []
        
        for symbol_info in exchange_info.get('symbols', []):
            if symbol_info.get('status') == 'TRADING':
                symbols.append(symbol_info['symbol'])
                
        return sorted(symbols)
    
    def create_symbol_mapping(self) -> Dict[str, str]:
        """
        Create mapping from user-friendly names to actual symbols.
        
        Returns:
            Dictionary mapping friendly names to symbols
        """
        exchange_info = self._load_exchange_info()
        mapping = {}
        
        for symbol_info in exchange_info.get('symbols', []):
            if symbol_info.get('status') == 'TRADING':
                symbol = symbol_info['symbol']
                base_asset = symbol_info['baseAsset']
                quote_asset = symbol_info['quoteAsset']
                
                # Add various mapping formats
                mapping[symbol] = symbol  # Exact match
                mapping[base_asset] = symbol  # Base asset (BTC -> BTCUSDT if available)
                mapping[f"{base_asset}USD"] = symbol
                mapping[f"{base_asset}USDT"] = symbol
                mapping[f"{base_asset}/{quote_asset}"] = symbol
                mapping[f"{base_asset}-{quote_asset}"] = symbol
                
        return mapping
    
    def fetch_historical_data(self, symbol: str, start_date: datetime, end_date: datetime, 
                              timeframe: str) -> pd.DataFrame:
        """
        Fetch historical OHLCV data with automatic chunking for large date ranges.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTCUSDT', 'BTC', 'ETHUSDT', 'ETH')
            start_date: Start date for historical data
            end_date: End date for historical data  
            timeframe: Data timeframe (e.g., '1m', '1h', 'day')
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume, ticker
        """
        if not self.authenticated:
            if not self.authenticate():
                self.logger.error("Authentication failed")
                return pd.DataFrame()
        
        # Normalize symbol to Binance format
        normalized_symbol = self._normalize_symbol(symbol)
        if not normalized_symbol:
            self.logger.error(f"Invalid or unsupported symbol: {symbol}")
            return pd.DataFrame()
        
        # Map timeframe to Binance interval
        binance_interval = self.map_timeframe(timeframe)
        
        # Calculate date range and determine if chunking is needed
        total_days = (end_date - start_date).days
        chunk_size = self._calculate_chunk_size(timeframe)
        chunk_timedelta = self._get_chunk_timedelta(timeframe, chunk_size)
        
        # Estimate total API calls needed for intelligent chunking
        estimated_calls = self._estimate_api_calls(start_date, end_date, timeframe)
        
        # For large requests (>6 months), use monthly chunking
        if total_days > 180:  # 6 months
            self.logger.info(f"Large date range detected ({total_days} days, ~{estimated_calls} API calls). "
                           f"Using monthly chunking for optimal performance...")
            return self._fetch_monthly_chunks(normalized_symbol, start_date, end_date, timeframe)
        
        # If the date range is small enough, fetch in one go
        if total_days <= chunk_timedelta.days:
            self.logger.info(f"Fetching Binance data: Symbol={normalized_symbol}, "
                           f"Interval={binance_interval}, {total_days} days")
            
            klines = self._fetch_chunk(normalized_symbol, start_date, end_date, timeframe)
            
            if not klines:
                self.logger.warning(f"No data returned for {normalized_symbol}")
                return pd.DataFrame()
                
            return self._process_klines(klines, normalized_symbol)
        
        # Large date range - use chunking
        self.logger.info(f"Large date range detected ({total_days} days). Using chunked fetching...")
        all_klines = []
        current_start = start_date
        chunk_count = 0
        
        while current_start < end_date:
            chunk_end = min(current_start + chunk_timedelta, end_date)
            chunk_count += 1
            
            self.logger.info(f"Fetching chunk {chunk_count}: {current_start.date()} to {chunk_end.date()}")
            
            klines = self._fetch_chunk(normalized_symbol, current_start, chunk_end, timeframe)
            
            if klines:
                all_klines.extend(klines)
                self.logger.debug(f"Chunk {chunk_count}: {len(klines)} records")
            else:
                self.logger.warning(f"No data for chunk {chunk_count}")
            
            current_start = chunk_end
        
        if not all_klines:
            self.logger.warning(f"No data returned for {normalized_symbol} across all chunks")
            return pd.DataFrame()
        
        self.logger.info(f"Successfully fetched {len(all_klines)} total records in {chunk_count} chunks")
        return self._process_klines(all_klines, normalized_symbol)
    
    def _process_klines(self, klines: List, symbol: str) -> pd.DataFrame:
        """
        Process raw klines data into a standardized DataFrame.
        
        Args:
            klines: List of kline data from Binance API
            symbol: Symbol name for the ticker column
            
        Returns:
            Processed DataFrame
        """
        try:
            # Convert to DataFrame
            df_data = []
            for kline in klines:
                # Binance kline format: [timestamp, open, high, low, close, volume, 
                #                       close_time, quote_asset_volume, number_of_trades,
                #                       taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore]
                df_data.append({
                    'timestamp': pd.to_datetime(kline[0], unit='ms'),
                    'open': float(kline[1]),
                    'high': float(kline[2]), 
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5]),
                    'ticker': symbol
                })
            
            df = pd.DataFrame(df_data)
            
            if df.empty:
                self.logger.warning(f"No data processed for {symbol}")
                return df
            
            # Remove duplicates and sort by timestamp
            df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            
            # Apply standard data normalization
            df = self.normalize_data(df)
            
            self.logger.info(f"Successfully processed {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing klines for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def map_timeframe(self, standard_timeframe: str) -> str:
        """
        Map standardized timeframe to Binance-specific interval.
        
        Args:
            standard_timeframe: Standard timeframe string
            
        Returns:
            Binance interval string
        """
        # Normalize input
        timeframe = standard_timeframe.lower().strip()
        
        # Direct mapping from our predefined mappings
        binance_interval = self.TIMEFRAME_MAPPING.get(timeframe)
        
        if binance_interval:
            return binance_interval
        
        # Fallback mappings for alternative formats
        fallback_mappings = {
            '1minute': self.TIMEFRAME_MAPPING['1m'],
            '5minute': self.TIMEFRAME_MAPPING['5m'],
            '30minute': self.TIMEFRAME_MAPPING['30m'],
            '1hour': self.TIMEFRAME_MAPPING['1h'],
            'daily': self.TIMEFRAME_MAPPING['day'],
            'weekly': self.TIMEFRAME_MAPPING['week'],
            'monthly': self.TIMEFRAME_MAPPING['month']
        }
        
        binance_interval = fallback_mappings.get(timeframe)
        if binance_interval:
            return binance_interval
        
        # Default fallback
        self.logger.warning(f"Unknown timeframe '{standard_timeframe}', defaulting to 1 hour")
        return self.TIMEFRAME_MAPPING['1h']
    
    def symbol_to_instrument_id(self, symbol: str) -> str:
        """
        Convert symbol to Binance-specific instrument identifier.
        For Binance, symbols are used directly as instrument IDs.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Normalized Binance symbol
        """
        return self._normalize_symbol(symbol) or symbol
    
    def fetch_instrument_details(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fetch instrument details for cryptocurrency symbols.
        
        Args:
            symbols: Optional list of symbols to filter
            
        Returns:
            DataFrame with instrument details
        """
        # Create instrument details for supported symbols
        instruments = []
        
        symbol_list = symbols if symbols else self.SUPPORTED_SYMBOLS
        
        for symbol in symbol_list:
            if symbol in self.SUPPORTED_SYMBOLS or self._normalize_symbol(symbol):
                normalized = self._normalize_symbol(symbol) or symbol
                
                # Extract base and quote currencies
                if normalized.endswith('USDT'):
                    base_currency = normalized[:-4]
                    quote_currency = 'USDT'
                else:
                    base_currency = normalized
                    quote_currency = 'USDT'
                
                instruments.append({
                    'tradingsymbol': normalized,
                    'exchange': 'BINANCE',
                    'instrument_type': 'CRYPTO',
                    'base_currency': base_currency,
                    'quote_currency': quote_currency,
                    'lot_size': 1,
                    'tick_size': 0.00000001,
                    'instrument_token': normalized
                })
        
        return pd.DataFrame(instruments)
    
    def _normalize_symbol(self, symbol: str) -> Optional[str]:
        """
        Normalize symbol to Binance format.
        
        Args:
            symbol: Input symbol (e.g., 'BTC', 'BTCUSDT', 'btc')
            
        Returns:
            Normalized Binance symbol or None if invalid
        """
        if not symbol:
            return None
            
        symbol = symbol.upper().strip()
        
        # If already in USDT format and supported, return as-is
        if symbol in self.SUPPORTED_SYMBOLS:
            return symbol
        
        # If it's a base currency, append USDT
        if symbol + 'USDT' in self.SUPPORTED_SYMBOLS:
            return symbol + 'USDT'
        
        # Check if it's a known base currency from our supported list
        for supported_symbol in self.SUPPORTED_SYMBOLS:
            if supported_symbol.startswith(symbol + 'USDT'):
                return supported_symbol
        
        self.logger.warning(f"Symbol '{symbol}' not found in supported symbols")
        return None
