"""
Data schemas for options contracts and historical data.

Defines the structure of options data storage, validation, and retrieval.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, date
from typing import List, Optional, Dict, Literal
from enum import Enum
import pandas as pd


class OptionType(str, Enum):
    """Option contract type."""
    CALL = "CE"  # Call option
    PUT = "PE"   # Put option


class Exchange(str, Enum):
    """Supported exchanges."""
    NSE = "NSE"
    BSE = "BSE"
    MCX = "MCX"


@dataclass
class OptionContract:
    """
    Represents a single option contract specification.

    This is the metadata about the contract itself (not price data).
    """
    ticker: str                    # Underlying ticker (e.g., "NIFTY")
    strike: float                  # Strike price
    expiry: date                   # Expiry date
    option_type: OptionType        # CE or PE
    exchange: Exchange             # NSE, BSE, MCX
    lot_size: int                  # Contracts per lot

    # Upstox-specific identifiers
    instrument_key: Optional[str] = None   # e.g., "NSE_FO|NIFTY24JAN24350CE"
    trading_symbol: Optional[str] = None   # e.g., "NIFTY24JAN24350CE"

    # Additional metadata
    segment: Optional[str] = None          # FO (Futures & Options), CDS, etc.
    tick_size: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d['option_type'] = self.option_type.value
        d['exchange'] = self.exchange.value
        d['expiry'] = self.expiry.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> 'OptionContract':
        """Create from dictionary."""
        data = data.copy()
        data['option_type'] = OptionType(data['option_type'])
        data['exchange'] = Exchange(data['exchange'])
        data['expiry'] = date.fromisoformat(data['expiry']) if isinstance(data['expiry'], str) else data['expiry']
        return cls(**data)

    def __str__(self) -> str:
        """Human-readable string."""
        return f"{self.ticker} {self.strike} {self.option_type.value} {self.expiry}"


@dataclass
class OptionOHLC:
    """
    Single OHLC bar for an option contract.

    This is a single row of price data.
    """
    timestamp: datetime
    strike: float
    option_type: OptionType
    open: float
    high: float
    low: float
    close: float
    volume: int
    open_interest: int

    # Context (for storage)
    ticker: str
    expiry: date
    lot_size: int

    # Optional (if available)
    bid: Optional[float] = None
    ask: Optional[float] = None

    @property
    def mid(self) -> Optional[float]:
        """Calculate mid price if bid/ask available."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2.0
        return None

    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            'timestamp': self.timestamp,
            'strike': self.strike,
            'option_type': self.option_type.value,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'open_interest': self.open_interest,
            'ticker': self.ticker,
            'expiry': self.expiry,
            'lot_size': self.lot_size,
            'bid': self.bid,
            'ask': self.ask,
            'mid': self.mid
        }


@dataclass
class ExpiryMetadata:
    """
    Metadata for a single expiry date.

    Stored as {expiry}/metadata.json for quick lookups.
    """
    ticker: str
    expiry: date
    expiry_type: Literal["weekly", "monthly", "quarterly"]
    lot_size: int

    # Available strikes for this expiry
    call_strikes: List[float]
    put_strikes: List[float]

    # Data availability
    data_start_date: Optional[date] = None
    data_end_date: Optional[date] = None
    total_trading_days: Optional[int] = None

    # Contract identifiers (for reference)
    contracts: Optional[List[Dict]] = None  # List of OptionContract dicts

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON storage."""
        return {
            'ticker': self.ticker,
            'expiry': self.expiry.isoformat(),
            'expiry_type': self.expiry_type,
            'lot_size': self.lot_size,
            'call_strikes': self.call_strikes,
            'put_strikes': self.put_strikes,
            'data_start_date': self.data_start_date.isoformat() if self.data_start_date else None,
            'data_end_date': self.data_end_date.isoformat() if self.data_end_date else None,
            'total_trading_days': self.total_trading_days,
            'contracts': self.contracts
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ExpiryMetadata':
        """Create from dictionary."""
        data = data.copy()
        data['expiry'] = date.fromisoformat(data['expiry'])
        if data.get('data_start_date'):
            data['data_start_date'] = date.fromisoformat(data['data_start_date'])
        if data.get('data_end_date'):
            data['data_end_date'] = date.fromisoformat(data['data_end_date'])
        return cls(**data)


# DataFrame Schema Definitions

OPTION_OHLC_SCHEMA = {
    'timestamp': 'datetime64[ns, Asia/Kolkata]',
    'strike': 'float64',
    'option_type': 'str',  # CE or PE
    'open': 'float64',
    'high': 'float64',
    'low': 'float64',
    'close': 'float64',
    'volume': 'int64',
    'open_interest': 'int64',
    'ticker': 'str',
    'expiry': 'object',  # date object
    'lot_size': 'int64',
    'bid': 'float64',
    'ask': 'float64',
    'mid': 'float64'
}


EQUITY_OHLC_SCHEMA = {
    'timestamp': 'datetime64[ns, Asia/Kolkata]',
    'open': 'float64',
    'high': 'float64',
    'low': 'float64',
    'close': 'float64',
    'volume': 'int64',
    'ticker': 'str'
}


def validate_option_ohlc_df(df: pd.DataFrame) -> bool:
    """
    Validate that a DataFrame conforms to option OHLC schema.

    Args:
        df: DataFrame to validate

    Returns:
        True if valid

    Raises:
        ValueError: If validation fails
    """
    required_columns = [
        'timestamp', 'strike', 'option_type', 'open', 'high', 'low',
        'close', 'volume', 'open_interest', 'ticker', 'expiry', 'lot_size'
    ]

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Validate option_type values
    valid_types = {OptionType.CALL.value, OptionType.PUT.value}
    invalid_types = set(df['option_type'].unique()) - valid_types
    if invalid_types:
        raise ValueError(f"Invalid option_type values: {invalid_types}. Must be {valid_types}")

    # Check for negative prices
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if (df[col] < 0).any():
            raise ValueError(f"Negative prices found in column: {col}")

    # Check OHLC consistency
    if not ((df['low'] <= df['open']) &
            (df['low'] <= df['close']) &
            (df['high'] >= df['open']) &
            (df['high'] >= df['close'])).all():
        raise ValueError("OHLC consistency check failed (low/high bounds violated)")

    return True


def create_empty_option_df() -> pd.DataFrame:
    """Create an empty DataFrame with correct option OHLC schema."""
    df = pd.DataFrame(columns=list(OPTION_OHLC_SCHEMA.keys()))
    return df


# File path helpers

def get_expiry_dir_name(expiry: date) -> str:
    """
    Generate directory name for an expiry.

    Args:
        expiry: Expiry date

    Returns:
        Directory name like "expiry_2025-01-30"
    """
    return f"expiry_{expiry.isoformat()}"


def get_option_file_path(
    base_dir: str,
    ticker: str,
    expiry: date,
    timeframe: str,
    date_range: Optional[str] = None
) -> str:
    """
    Generate file path for option data parquet.

    UPDATED: Matches actual OptionsDataStorage format.

    Args:
        base_dir: Base directory (e.g., "data/pools/options")
        ticker: Ticker symbol
        expiry: Expiry date
        timeframe: Timeframe (e.g., "1day", "5m")
        date_range: Optional date range folder (e.g., "2025-04-01_to_2025-10-08")

    Returns:
        Full path like "data/pools/options/2025-04-01_to_2025-10-08/NIFTY/1day/expiry_2025-05-29.parquet"
    """
    import os

    # FIXED: {ticker}/{timeframe}/expiry_{date}.parquet (not {ticker}/expiry_{date}/{timeframe}.parquet)
    expiry_filename = f"expiry_{expiry.isoformat()}.parquet"

    if date_range:
        path = os.path.join(base_dir, date_range, ticker, timeframe, expiry_filename)
    else:
        path = os.path.join(base_dir, ticker, timeframe, expiry_filename)

    return path


def get_metadata_path(
    base_dir: str,
    ticker: str,
    expiry: date,
    date_range: Optional[str] = None
) -> str:
    """
    Generate file path for expiry metadata JSON.

    UPDATED: Matches actual OptionsDataStorage format.

    Returns:
        Path like "data/pools/options/2025-04-01_to_2025-10-08/NIFTY/metadata/expiry_2025-05-29.json"
    """
    import os

    # FIXED: {ticker}/metadata/expiry_{date}.json (not {ticker}/expiry_{date}/metadata.json)
    expiry_filename = f"expiry_{expiry.isoformat()}.json"

    if date_range:
        path = os.path.join(base_dir, date_range, ticker, "metadata", expiry_filename)
    else:
        path = os.path.join(base_dir, ticker, "metadata", expiry_filename)

    return path


# Validation dataset configuration

VALIDATION_TICKERS = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY"]

# Date range for validation (6 months from today backwards)
VALIDATION_DATE_RANGE = "2025-04-01_to_2025-10-08"  # Approximately 6 months

TICKER_TO_EXCHANGE = {
    "NIFTY": Exchange.NSE,
    "BANKNIFTY": Exchange.NSE,
    "FINNIFTY": Exchange.NSE,
    "RELIANCE": Exchange.NSE,
    "TCS": Exchange.NSE,
    "INFY": Exchange.NSE,
    "HDFCBANK": Exchange.NSE,
    "ICICIBANK": Exchange.NSE
}
