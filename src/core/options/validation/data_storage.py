"""
Data storage and retrieval utilities for options data.

Handles saving/loading options data in the structured format:
data/pools/options/{date_range}/{ticker}/{timeframe}/expiry_{date}.parquet
"""

import pandas as pd
import json
from pathlib import Path
from datetime import date, datetime
from typing import Optional, List, Dict
import logging

# Import schemas
from src.core.options.data.schemas import (
    validate_option_ohlc_df,
    ExpiryMetadata,
    get_expiry_dir_name
)


class OptionsDataStorage:
    """
    Manages storage and retrieval of options historical data.

    Directory structure:
        data/pools/options/{date_range}/
        └── {TICKER}/
            ├── 1day/
            │   ├── expiry_2025-05-29.parquet
            │   ├── expiry_2025-06-26.parquet
            │   └── ...
            ├── 5m/  (optional)
            └── metadata/
                ├── expiry_2025-05-29.json
                └── ...
    """

    def __init__(self, base_dir: str = "data/pools/options"):
        """
        Initialize options data storage.

        Args:
            base_dir: Base directory for options data (default: data/pools/options)
        """
        self.base_dir = Path(base_dir)
        self.logger = logging.getLogger(__name__)

    def _get_file_path(
        self,
        ticker: str,
        expiry: date,
        timeframe: str,
        date_range: str,
        file_type: str = 'data'
    ) -> Path:
        """
        Get file path for options data or metadata.

        Args:
            ticker: Ticker symbol
            expiry: Expiry date
            timeframe: Timeframe (e.g., '1day', '5m')
            date_range: Date range string (e.g., '2025-04-01_to_2025-10-08')
            file_type: 'data' or 'metadata'

        Returns:
            Path object
        """
        expiry_filename = f"expiry_{expiry.isoformat()}"

        if file_type == 'data':
            # data/pools/options/{date_range}/{ticker}/{timeframe}/expiry_2025-05-29.parquet
            path = self.base_dir / date_range / ticker / timeframe / f"{expiry_filename}.parquet"
        elif file_type == 'metadata':
            # data/pools/options/{date_range}/{ticker}/metadata/expiry_2025-05-29.json
            path = self.base_dir / date_range / ticker / "metadata" / f"{expiry_filename}.json"
        else:
            raise ValueError(f"Unknown file_type: {file_type}")

        return path

    def save_expiry_data(
        self,
        df: pd.DataFrame,
        ticker: str,
        expiry: date,
        timeframe: str,
        date_range: str,
        metadata: Optional[ExpiryMetadata] = None
    ):
        """
        Save options data for a single expiry.

        Args:
            df: DataFrame with option OHLC data
            ticker: Ticker symbol
            expiry: Expiry date
            timeframe: Timeframe (e.g., '1day', '5m')
            date_range: Date range string
            metadata: Optional ExpiryMetadata object (will be auto-generated if None)
        """
        if df.empty:
            self.logger.warning(f"Empty DataFrame for {ticker} {expiry}, skipping save")
            return

        # Validate schema
        try:
            validate_option_ohlc_df(df)
        except ValueError as e:
            self.logger.error(f"Data validation failed for {ticker} {expiry}: {e}")
            raise

        # Get file paths
        data_path = self._get_file_path(ticker, expiry, timeframe, date_range, 'data')
        metadata_path = self._get_file_path(ticker, expiry, timeframe, date_range, 'metadata')

        # Create directories
        data_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        # Save data
        df.to_parquet(data_path, index=False, compression='gzip')
        self.logger.info(f"Saved {len(df)} rows to {data_path}")

        # Generate or save metadata
        if metadata is None:
            metadata = self._generate_metadata(df, ticker, expiry, timeframe)

        def _sanitize(obj):
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sanitize(item) for item in obj]
            if isinstance(obj, (pd.Timestamp, datetime)):
                return obj.isoformat()
            if isinstance(obj, (pd.Timedelta, )):
                return obj.total_seconds()
            if isinstance(obj, (pd.Int64Dtype, pd.Float64Dtype)):
                return obj.item()
            if hasattr(obj, "item"):
                try:
                    return obj.item()
                except Exception:
                    return obj
            return obj

        metadata_dict = metadata.to_dict()
        metadata_dict = _sanitize(metadata_dict)

        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)

        self.logger.info(f"Saved metadata to {metadata_path}")

    def load_expiry_data(
        self,
        ticker: str,
        expiry: date,
        timeframe: str,
        date_range: str
    ) -> Optional[pd.DataFrame]:
        """
        Load options data for a single expiry.

        Args:
            ticker: Ticker symbol
            expiry: Expiry date
            timeframe: Timeframe
            date_range: Date range string

        Returns:
            DataFrame or None if file doesn't exist
        """
        data_path = self._get_file_path(ticker, expiry, timeframe, date_range, 'data')

        if not data_path.exists():
            self.logger.warning(f"Data file not found: {data_path}")
            return None

        df = pd.read_parquet(data_path)
        self.logger.info(f"Loaded {len(df)} rows from {data_path}")

        return df

    def load_metadata(
        self,
        ticker: str,
        expiry: date,
        timeframe: str,
        date_range: str
    ) -> Optional[ExpiryMetadata]:
        """
        Load metadata for a single expiry.

        Args:
            ticker: Ticker symbol
            expiry: Expiry date
            timeframe: Timeframe
            date_range: Date range string

        Returns:
            ExpiryMetadata object or None if file doesn't exist
        """
        metadata_path = self._get_file_path(ticker, expiry, timeframe, date_range, 'metadata')

        if not metadata_path.exists():
            self.logger.warning(f"Metadata file not found: {metadata_path}")
            return None

        with open(metadata_path, 'r') as f:
            data = json.load(f)

        metadata = ExpiryMetadata.from_dict(data)
        return metadata

    def list_expiries(
        self,
        ticker: str,
        timeframe: str,
        date_range: str
    ) -> List[date]:
        """
        List all available expiries for a ticker/timeframe.

        Args:
            ticker: Ticker symbol
            timeframe: Timeframe
            date_range: Date range string

        Returns:
            List of expiry dates (sorted)
        """
        timeframe_dir = self.base_dir / date_range / ticker / timeframe

        if not timeframe_dir.exists():
            return []

        # Find all expiry_*.parquet files
        expiry_files = list(timeframe_dir.glob("expiry_*.parquet"))

        # Extract dates from filenames
        expiries = []
        for file in expiry_files:
            # expiry_2025-05-29.parquet → 2025-05-29
            date_str = file.stem.replace("expiry_", "")
            try:
                expiry_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                expiries.append(expiry_date)
            except ValueError:
                self.logger.warning(f"Invalid expiry filename: {file.name}")
                continue

        expiries.sort()
        return expiries

    def load_all_expiries(
        self,
        ticker: str,
        timeframe: str,
        date_range: str,
        expiries: Optional[List[date]] = None
    ) -> pd.DataFrame:
        """
        Load and combine data from multiple expiries.

        Args:
            ticker: Ticker symbol
            timeframe: Timeframe
            date_range: Date range string
            expiries: List of specific expiries to load (default: all available)

        Returns:
            Combined DataFrame
        """
        if expiries is None:
            expiries = self.list_expiries(ticker, timeframe, date_range)

        if not expiries:
            self.logger.warning(f"No expiries found for {ticker} {timeframe} in {date_range}")
            return pd.DataFrame()

        dfs = []
        for expiry in expiries:
            df = self.load_expiry_data(ticker, expiry, timeframe, date_range)
            if df is not None:
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.sort_values(['timestamp', 'expiry', 'strike', 'option_type']).reset_index(drop=True)

        self.logger.info(f"Loaded {len(combined_df)} total rows from {len(dfs)} expiries")

        return combined_df

    def _generate_metadata(
        self,
        df: pd.DataFrame,
        ticker: str,
        expiry: date,
        timeframe: str
    ) -> ExpiryMetadata:
        """
        Generate metadata from DataFrame.

        Args:
            df: Options data DataFrame
            ticker: Ticker symbol
            expiry: Expiry date
            timeframe: Timeframe

        Returns:
            ExpiryMetadata object
        """
        # Extract unique strikes
        call_strikes = sorted(df[df['option_type'] == 'CE']['strike'].unique().tolist())
        put_strikes = sorted(df[df['option_type'] == 'PE']['strike'].unique().tolist())

        # Data availability
        timestamps = pd.to_datetime(df['timestamp'])
        data_start = timestamps.min().date()
        data_end = timestamps.max().date()
        total_days = len(timestamps.dt.date.unique())

        # Lot size (should be consistent)
        lot_size = df['lot_size'].iloc[0] if 'lot_size' in df.columns else None

        # Determine expiry type (heuristic)
        # Weekly: typically 1 week apart, Monthly: ~1 month apart
        # For now, default to weekly
        expiry_type = "weekly"  # Could be enhanced with smarter logic

        metadata = ExpiryMetadata(
            ticker=ticker,
            expiry=expiry,
            expiry_type=expiry_type,
            lot_size=lot_size,
            call_strikes=call_strikes,
            put_strikes=put_strikes,
            data_start_date=data_start,
            data_end_date=data_end,
            total_trading_days=total_days
        )

        return metadata

    def get_storage_stats(self, date_range: str) -> Dict:
        """
        Get statistics about stored options data.

        Args:
            date_range: Date range string

        Returns:
            Dictionary with statistics
        """
        stats = {
            'date_range': date_range,
            'tickers': [],
            'total_files': 0,
            'total_size_mb': 0.0
        }

        date_range_dir = self.base_dir / date_range

        if not date_range_dir.exists():
            return stats

        # Iterate through tickers
        for ticker_dir in date_range_dir.iterdir():
            if not ticker_dir.is_dir():
                continue

            ticker = ticker_dir.name
            ticker_stats = {
                'ticker': ticker,
                'timeframes': {},
                'total_expiries': 0
            }

            # Iterate through timeframes
            for item in ticker_dir.iterdir():
                if not item.is_dir() or item.name == 'metadata':
                    continue

                timeframe = item.name
                expiries = self.list_expiries(ticker, timeframe, date_range)

                ticker_stats['timeframes'][timeframe] = {
                    'expiries_count': len(expiries),
                    'expiries': [exp.isoformat() for exp in expiries]
                }
                ticker_stats['total_expiries'] += len(expiries)

                # Count files
                stats['total_files'] += len(expiries)

            stats['tickers'].append(ticker_stats)

        # Calculate total size
        for file in date_range_dir.rglob("*.parquet"):
            stats['total_size_mb'] += file.stat().st_size / (1024 * 1024)

        return stats


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    storage = OptionsDataStorage()

    # Test: Create sample data and save
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2025-04-01', periods=10, freq='D', tz='Asia/Kolkata'),
        'strike': [2850] * 10,
        'option_type': ['CE'] * 10,
        'open': [45.0] * 10,
        'high': [47.0] * 10,
        'low': [44.0] * 10,
        'close': [46.0] * 10,
        'volume': [1000] * 10,
        'open_interest': [5000] * 10,
        'ticker': ['RELIANCE'] * 10,
        'expiry': [date(2025, 5, 29)] * 10,
        'lot_size': [505] * 10,
        'bid': [None] * 10,
        'ask': [None] * 10,
        'mid': [None] * 10
    })

    # Save
    storage.save_expiry_data(
        sample_data,
        ticker='RELIANCE',
        expiry=date(2025, 5, 29),
        timeframe='1day',
        date_range='2025-04-01_to_2025-10-08'
    )

    # Load
    loaded = storage.load_expiry_data(
        ticker='RELIANCE',
        expiry=date(2025, 5, 29),
        timeframe='1day',
        date_range='2025-04-01_to_2025-10-08'
    )

    print(f"Loaded {len(loaded)} rows")
    print(loaded.head())
