"""
Upstox API client for fetching expired options historical data.

Wraps the Upstox v2 expired-instruments API endpoints.
"""

import csv
import requests
import pandas as pd
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple
import time
import logging
from pathlib import Path
from functools import lru_cache
from collections import deque

# Import from parent modules
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from config.config import UPSTOX_CONFIG
from src.core.etl.token_manager import load_provider_token


class UpstoxOptionsAPI:
    """
    Client for Upstox expired options historical data API.

    Endpoints used:
    - GET /v2/expired-instruments/expiries
    - GET /v2/expired-instruments/option/contract
    - GET /v2/expired-instruments/historical-candle/{instrument_key}/{interval}/{to_date}/{from_date}
    """

    BASE_URL = "https://api.upstox.com/v2"

    def __init__(
        self,
        access_token: Optional[str] = None,
        rate_limit: Optional[int] = None,
        requests_per_minute: Optional[int] = None,
        retry_attempts: Optional[int] = None,
        retry_backoff_factor: Optional[float] = None
    ):
        """
        Initialize Upstox Options API client.

        Args:
            access_token: Upstox access token (if None, will try to load from token manager)
            rate_limit: Requests per second (default: 5)
            retry_attempts: Number of retry attempts on failure (default: 3)
            retry_backoff_factor: Exponential backoff factor (default: 2)
        """
        self.logger = logging.getLogger(__name__)

        # Get access token
        if access_token is None:
            # FIX Issue #1: load_provider_token returns string directly, not dict
            access_token = load_provider_token('upstox')

        if not access_token:
            raise ValueError(
                "No Upstox access token found. Please authenticate first.\n"
                "Run: python src/core/etl/data_fetcher.py --mode fetch --provider upstox"
            )

        self.access_token = access_token
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.access_token}',
            'Accept': 'application/json'
        })

        # Rate limiting (configurable, default 5 req/sec, 300 req/min)
        self.requests_per_second = rate_limit or 5
        default_per_minute = (self.requests_per_second * 60) if self.requests_per_second else 300
        self.requests_per_minute = requests_per_minute or default_per_minute
        self.last_request_time = 0
        self._request_times = deque()

        # Retry configuration
        self.retry_attempts = retry_attempts or 3
        self.retry_backoff_factor = retry_backoff_factor or 2
        self._spread_warning_logged = False

        # FIX Issue #2: Load timeframe mappings from config
        try:
            self.timeframe_mappings = UPSTOX_CONFIG.get('TIMEFRAME_MAPPINGS', {})
        except:
            self.timeframe_mappings = {}

        # Fallback mappings if config not available
        if not self.timeframe_mappings:
            self.timeframe_mappings = {
                '1m': {'unit': 'minutes', 'interval': '1'},
                '5m': {'unit': 'minutes', 'interval': '5'},
                '15m': {'unit': 'minutes', 'interval': '15'},
                '30m': {'unit': 'minutes', 'interval': '30'},
                '1h': {'unit': 'hours', 'interval': '1'},
                '1day': {'unit': 'days', 'interval': '1'},
                'day': {'unit': 'days', 'interval': '1'},
            }

    def _map_timeframe_to_upstox(self, timeframe: str) -> str:
        """
        Map standardized timeframe to Upstox API format.

        Args:
            timeframe: Standard timeframe (e.g., '1day', '5m', '1h')

        Returns:
            Upstox API interval string (e.g., 'day', '5minute', '1hour')

        FIX Issue #2: Proper timeframe mapping for all intervals
        """
        # Direct mappings for Upstox expired-instruments API
        upstox_format_map = {
            '1m': '1minute',
            '5m': '5minute',
            '15m': '15minute',
            '30m': '30minute',
            '1h': '1hour',
            '1day': 'day',
            'day': 'day',
            '1minute': '1minute',  # Pass-through if already correct
            '5minute': '5minute',
            '15minute': '15minute',
            '30minute': '30minute',
            '1hour': '1hour',
        }

        mapped = upstox_format_map.get(timeframe)

        if mapped is None:
            self.logger.warning(f"Unknown timeframe '{timeframe}', defaulting to 'day'")
            return 'day'

        return mapped

    def _rate_limit(self):
        """Enforce per-second and per-minute rate limits."""
        now = time.time()

        if self.requests_per_second:
            min_interval = 1.0 / self.requests_per_second
            elapsed = now - self.last_request_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
                now = time.time()

        if self.requests_per_minute:
            cutoff = now - 60
            while self._request_times and self._request_times[0] < cutoff:
                self._request_times.popleft()
            if len(self._request_times) >= self.requests_per_minute:
                sleep_time = max(0.0, 60 - (now - self._request_times[0]))
                if sleep_time > 0:
                    self.logger.debug(f"Per-minute rate limit reached, sleeping {sleep_time:.2f}s")
                    time.sleep(sleep_time)
                    now = time.time()
                    cutoff = now - 60
                    while self._request_times and self._request_times[0] < cutoff:
                        self._request_times.popleft()

        now = time.time()
        self.last_request_time = now
        self._request_times.append(now)

    def _make_request(self, method: str, url: str, **kwargs) -> Dict:
        """
        Make HTTP request with rate limiting, retry logic, and error handling.

        Args:
            method: HTTP method (GET, POST)
            url: Full URL
            **kwargs: Additional arguments for requests

        Returns:
            Response JSON

        Raises:
            Exception: If request fails after all retries
        """
        last_exception = None

        for attempt in range(self.retry_attempts):
            try:
                self._rate_limit()
                response = self.session.request(method, url, **kwargs)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as e:
                last_exception = e
                self.logger.warning(f"HTTP Error (attempt {attempt + 1}/{self.retry_attempts}): {e}")
                if response.status_code in [429, 503]:  # Rate limit or service unavailable
                    sleep_time = self.retry_backoff_factor ** attempt
                    self.logger.info(f"Retrying after {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    # Non-retryable error (400, 401, 404, etc.)
                    self.logger.error(f"Non-retryable error: {response.text}")
                    raise
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Request failed (attempt {attempt + 1}/{self.retry_attempts}): {e}")
                if attempt < self.retry_attempts - 1:
                    sleep_time = self.retry_backoff_factor ** attempt
                    self.logger.info(f"Retrying after {sleep_time}s...")
                    time.sleep(sleep_time)

        # All retries exhausted
        self.logger.error(f"Request failed after {self.retry_attempts} attempts")
        raise last_exception

    def get_instrument_key(self, ticker: str, exchange: str = "NSE") -> str:
        """
        Construct instrument key for options.

        For NSE equity options: NSE_FO|{ticker}
        For indices: NSE_INDEX|{ticker}

        Args:
            ticker: Ticker symbol (e.g., "RELIANCE", "NIFTY")
            exchange: Exchange (default "NSE")

        Returns:
            Instrument key string
        """
        # Indices use NSE_INDEX segment
        instrument_key = self._resolve_instrument_key(ticker)
        if instrument_key:
            return instrument_key

        # Fallback heuristic
        if ticker in {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}:
            return f"{exchange}_INDEX|{ticker}"

        return f"{exchange}_FO|{ticker}"

    @lru_cache(maxsize=None)
    def _instrument_lookup(self) -> Dict[str, list]:
        """Load instrument master data for instrument key resolution."""
        instruments_csv = UPSTOX_CONFIG.get('INSTRUMENTS_CSV')
        mapping: Dict[str, list] = {}

        if not instruments_csv:
            self.logger.warning("UPSTOX_CONFIG missing INSTRUMENTS_CSV; instrument lookups disabled")
            return mapping

        path = Path(instruments_csv)
        if not path.exists():
            self.logger.warning(f"Instrument CSV not found at {path}; instrument lookups disabled")
            return mapping

        try:
            with path.open('r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ts = row.get('tradingsymbol')
                    if not ts:
                        continue
                    mapping.setdefault(ts.strip().upper(), []).append(row)
        except Exception as exc:
            self.logger.warning(f"Failed to load instrument master from {path}: {exc}")

        return mapping

    def _resolve_instrument_key(self, ticker: str) -> Optional[str]:
        """Resolve Upstox instrument_key for the given ticker from instrument master."""
        lookup = self._instrument_lookup()
        if not lookup:
            return None

        ticker_key = ticker.upper()

        index_aliases = {
            "NIFTY": "NIFTY 50",
            "BANKNIFTY": "NIFTY BANK",
            "FINNIFTY": "NIFTY FIN SERVICE",
            "MIDCPNIFTY": "NIFTY MID SELECT"
        }

        candidates = []

        rows = lookup.get(ticker_key, [])
        if rows:
            rows_sorted = sorted(
                rows,
                key=lambda r: (
                    0 if r.get('exchange') == 'NSE_EQ' else
                    1 if r.get('exchange', '').startswith('NSE') else 2
                )
            )
            candidates.extend(rows_sorted)

        alias = index_aliases.get(ticker_key)
        if alias:
            candidates.extend(lookup.get(alias.upper(), []))

        for row in candidates:
            instrument_key = row.get('instrument_key')
            if not instrument_key:
                continue

            exchange = row.get('exchange', '')
            instrument_type = row.get('instrument_type', '')

            if ticker_key in index_aliases and exchange == 'NSE_INDEX':
                return instrument_key

            if exchange == 'NSE_EQ' and instrument_type in {'EQUITY', 'EQ'}:
                return instrument_key

            if exchange == 'NSE_FO' and instrument_type in {'FUTSTK'}:
                return instrument_key

        if candidates:
            candidate_key = candidates[0].get('instrument_key')
            if candidate_key:
                return candidate_key

        self.logger.debug(f"Unable to resolve instrument key from master for {ticker}")
        return None

    def get_expiries(self, instrument_key: str) -> List[date]:
        """
        Get list of expired option expiry dates for an instrument.

        API: GET /v2/expired-instruments/expiries?instrument_key={instrument_key}

        Args:
            instrument_key: Instrument key (e.g., "NSE_FO|RELIANCE")

        Returns:
            List of expiry dates (sorted, most recent first)

        Note:
            Upstox returns up to 6 months of historical expiries
        """
        url = f"{self.BASE_URL}/expired-instruments/expiries"
        params = {'instrument_key': instrument_key}

        self.logger.info(f"Fetching expiries for {instrument_key}")

        response = self._make_request('GET', url, params=params)

        if response.get('status') != 'success':
            raise Exception(f"Failed to fetch expiries: {response}")

        # Parse expiry dates
        expiries_str = response.get('data', [])
        expiries = [datetime.strptime(exp, '%Y-%m-%d').date() for exp in expiries_str]

        # Sort descending (most recent first)
        expiries.sort(reverse=True)

        self.logger.info(f"Found {len(expiries)} expiries for {instrument_key}")

        return expiries

    def get_option_contracts(self, instrument_key: str, expiry_date: date) -> pd.DataFrame:
        """
        Get all option contracts (strikes) for a specific expiry.

        API: GET /v2/expired-instruments/option/contract?instrument_key={}&expiry_date={}

        Args:
            instrument_key: Instrument key
            expiry_date: Expiry date

        Returns:
            DataFrame with columns: strike, option_type, instrument_key, trading_symbol, lot_size, tick_size
        """
        url = f"{self.BASE_URL}/expired-instruments/option/contract"
        params = {
            'instrument_key': instrument_key,
            'expiry_date': expiry_date.strftime('%Y-%m-%d')
        }

        self.logger.info(f"Fetching option contracts for {instrument_key} expiry {expiry_date}")

        response = self._make_request('GET', url, params=params)

        if response.get('status') != 'success':
            raise Exception(f"Failed to fetch option contracts: {response}")

        contracts_data = response.get('data', [])

        if not contracts_data:
            self.logger.warning(f"No contracts found for {instrument_key} {expiry_date}")
            return pd.DataFrame()

        # Parse contracts
        contracts = []
        for contract in contracts_data:
            contracts.append({
                'strike': contract.get('strike_price'),
                'option_type': contract.get('instrument_type'),  # CE or PE
                'instrument_key': contract.get('instrument_key'),
                'trading_symbol': contract.get('trading_symbol'),
                'lot_size': contract.get('lot_size'),
                'tick_size': contract.get('tick_size'),
                'segment': contract.get('segment'),
                'exchange': contract.get('exchange')
            })

        df = pd.DataFrame(contracts)

        self.logger.info(f"Found {len(df)} option contracts ({len(df[df['option_type']=='CE'])} calls, {len(df[df['option_type']=='PE'])} puts)")

        return df

    def get_historical_candle(
        self,
        instrument_key: str,
        interval: str,
        from_date: date,
        to_date: date
    ) -> pd.DataFrame:
        """
        Get historical OHLC data for an expired option contract.

        API: GET /v2/expired-instruments/historical-candle/{instrument_key}/{interval}/{to_date}/{from_date}

        Args:
            instrument_key: Option contract instrument key (e.g., "NSE_FO|RELIANCE25JAN2850CE")
            interval: Candle interval ('1day', '5m', '1h', etc.) - will be converted to Upstox format
            from_date: Start date
            to_date: End date

        Returns:
            DataFrame with OHLC data

        Note:
            - URL path format: {instrument_key}/{interval}/{to_date}/{from_date}
            - Notice: to_date comes before from_date in URL (API quirk)
            - FIX Issue #2: Timeframe is mapped to Upstox format before API call
        """
        # FIX Issue #2: Map timeframe to Upstox API format
        upstox_interval = self._map_timeframe_to_upstox(interval)

        # Format dates as YYYY-MM-DD
        from_date_str = from_date.strftime('%Y-%m-%d')
        to_date_str = to_date.strftime('%Y-%m-%d')

        # URL encode instrument key (| becomes %7C)
        instrument_key_encoded = requests.utils.quote(instrument_key, safe='')

        # Build URL (note: to_date before from_date!)
        url = (
            f"{self.BASE_URL}/expired-instruments/historical-candle/"
            f"{instrument_key_encoded}/{upstox_interval}/{to_date_str}/{from_date_str}"
        )

        self.logger.debug(f"Fetching candles: {instrument_key} {interval} {from_date} to {to_date}")

        response = self._make_request('GET', url)

        if response.get('status') != 'success':
            self.logger.warning(f"No data for {instrument_key}: {response.get('message', 'Unknown error')}")
            return pd.DataFrame()

        candles = response.get('data', {}).get('candles', [])

        if not candles:
            self.logger.warning(f"No candle data for {instrument_key}")
            return pd.DataFrame()

        # Parse candles
        # Upstox format: [timestamp, open, high, low, close, volume, open_interest]
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'open_interest'])

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Kolkata')

        # Convert numeric columns
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)
        df['open_interest'] = pd.to_numeric(df['open_interest'], errors='coerce').fillna(0).astype(int)

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        self.logger.debug(f"Fetched {len(df)} candles for {instrument_key}")

        return df

    def fetch_options_for_expiry(
        self,
        ticker: str,
        expiry_date: date,
        interval: str = '1day',
        reference_price: Optional[float] = None,
        strike_range_pct: float = 0.20,
        min_open_interest: int = 100,
        min_volume: Optional[int] = None,
        max_spread_pct: Optional[float] = None,
        max_strikes: Optional[int] = None,
        exclude_dte_below: Optional[int] = None,
        exclude_dte_above: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch all option data for a specific expiry (high-level convenience method).

        Args:
            ticker: Ticker symbol (e.g., "RELIANCE")
            expiry_date: Expiry date
            interval: Candle interval (default 'day')
            reference_price: Current underlying price (for strike filtering). If None, fetches all strikes.
            strike_range_pct: Fetch strikes within ±X% of reference price (default 20%)
            min_open_interest: Minimum OI to include strike (default 100)

        Returns:
            DataFrame with all option data for this expiry (all strikes, calls & puts)
            Columns: timestamp, strike, option_type, open, high, low, close, volume, open_interest, ...
        """
        # Get instrument key
        instrument_key = self.get_instrument_key(ticker)

        # Get option contracts for this expiry
        contracts_df = self.get_option_contracts(instrument_key, expiry_date)

        if contracts_df.empty:
            self.logger.warning(f"No contracts found for {ticker} {expiry_date}")
            return pd.DataFrame()

        # Filter strikes if reference price provided
        if reference_price is not None:
            lower_bound = reference_price * (1 - strike_range_pct)
            upper_bound = reference_price * (1 + strike_range_pct)

            contracts_df = contracts_df[
                (contracts_df['strike'] >= lower_bound) &
                (contracts_df['strike'] <= upper_bound)
            ]

            self.logger.info(f"Filtered to {len(contracts_df)} contracts within {strike_range_pct*100}% of {reference_price}")

        if max_strikes and max_strikes > 0 and not contracts_df.empty:
            if reference_price is not None:
                contracts_df = contracts_df.assign(
                    _distance=(contracts_df['strike'] - reference_price).abs()
                )
            else:
                median_strike = contracts_df['strike'].median()
                contracts_df = contracts_df.assign(
                    _distance=(contracts_df['strike'] - median_strike).abs()
                )
            limited_df = contracts_df.nsmallest(int(max_strikes), '_distance').drop(columns=['_distance'])
            if len(limited_df) < len(contracts_df):
                self.logger.info(
                    "Limited contracts to %s based on max_strikes setting (original %s)",
                    len(limited_df),
                    len(contracts_df)
                )
            contracts_df = limited_df

        if contracts_df.empty:
            self.logger.warning(f"No contracts left for {ticker} {expiry_date} after filtering")
            return pd.DataFrame()

        # Fetch historical data for each contract
        all_data = []

        # Data range: from 2 months before expiry to expiry date
        # (options typically list ~2 months before expiry)
        from_date = expiry_date - timedelta(days=60)
        to_date = expiry_date

        for _, contract in contracts_df.iterrows():
            contract_key = contract['instrument_key']
            strike = contract['strike']
            option_type = contract['option_type']

            if not contract_key:
                self.logger.debug(
                    "Skipping contract with empty instrument_key: %s %s",
                    strike,
                    option_type,
                )
                continue

            try:
                # Fetch OHLC data
                candles_df = self.get_historical_candle(
                    contract_key,
                    interval,
                    from_date,
                    to_date
                )

                if candles_df.empty:
                    continue

                # Filter by OI if specified
                if min_open_interest > 0:
                    candles_df = candles_df[candles_df['open_interest'] >= min_open_interest]

                if candles_df.empty:
                    self.logger.debug(f"No data with OI >= {min_open_interest} for {strike} {option_type}")
                    continue

                # Filter by minimum volume
                if min_volume is not None and min_volume > 0:
                    candles_df = candles_df[candles_df['volume'] >= min_volume]
                    if candles_df.empty:
                        self.logger.debug(f"No data with volume >= {min_volume} for {strike} {option_type}")
                        continue

                # Filter by max spread percentage when bid/ask available
                if max_spread_pct is not None:
                    if {'bid', 'ask'}.issubset(candles_df.columns):
                        mid = (candles_df['bid'] + candles_df['ask']) / 2.0
                        spread = (candles_df['ask'] - candles_df['bid']).abs()
                        with_mid = mid.replace(0, pd.NA)
                        spread_pct = spread / with_mid
                        candles_df = candles_df[spread_pct <= max_spread_pct]
                        if candles_df.empty:
                            self.logger.debug(
                                f"No data with spread <= {max_spread_pct:.2f} for {strike} {option_type}"
                            )
                            continue
                    elif not hasattr(self, "_spread_warning_logged"):
                        self.logger.warning(
                            "Bid/ask data not available from Upstox response; "
                            "max_spread_pct filter skipped."
                        )
                        self._spread_warning_logged = True

                # Filter by days-to-expiry bounds
                if exclude_dte_below is not None or exclude_dte_above is not None:
                    dte_series = (expiry_date - candles_df['timestamp'].dt.date).apply(lambda delta: delta.days)
                    mask = pd.Series(True, index=candles_df.index)
                    if exclude_dte_below is not None:
                        mask &= dte_series >= exclude_dte_below
                    if exclude_dte_above is not None:
                        mask &= dte_series <= exclude_dte_above
                    candles_df = candles_df[mask]
                    if candles_df.empty:
                        self.logger.debug(
                            f"No data within DTE bounds for {strike} {option_type} "
                            f"(min={exclude_dte_below}, max={exclude_dte_above})"
                        )
                        continue

                # Add metadata
                candles_df['strike'] = strike
                candles_df['option_type'] = option_type
                candles_df['ticker'] = ticker
                candles_df['expiry'] = expiry_date
                candles_df['lot_size'] = contract['lot_size']

                all_data.append(candles_df)

            except Exception as e:
                self.logger.error(f"Failed to fetch {ticker} {strike} {option_type}: {e}")
                continue

        if not all_data:
            self.logger.warning(f"No option data fetched for {ticker} {expiry_date}")
            return pd.DataFrame()

        # Combine all contracts
        combined_df = pd.concat(all_data, ignore_index=True)

        # Reorder columns
        column_order = [
            'timestamp', 'strike', 'option_type', 'open', 'high', 'low', 'close',
            'volume', 'open_interest', 'ticker', 'expiry', 'lot_size'
        ]
        combined_df = combined_df[column_order]

        # Sort
        combined_df = combined_df.sort_values(['timestamp', 'strike', 'option_type']).reset_index(drop=True)

        self.logger.info(
            f"Fetched {len(combined_df)} rows for {ticker} {expiry_date} "
            f"({len(combined_df['strike'].unique())} strikes)"
        )

        return combined_df


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    api = UpstoxOptionsAPI()

    # Test: Get expiries for RELIANCE
    instrument_key = api.get_instrument_key("RELIANCE")
    print(f"Instrument key: {instrument_key}")

    expiries = api.get_expiries(instrument_key)
    print(f"\nFound {len(expiries)} expiries:")
    for exp in expiries[:5]:  # Show first 5
        print(f"  - {exp}")

    # Test: Get contracts for latest expiry
    if expiries:
        latest_expiry = expiries[0]
        contracts = api.get_option_contracts(instrument_key, latest_expiry)
        print(f"\nContracts for {latest_expiry}:")
        print(contracts.head())
