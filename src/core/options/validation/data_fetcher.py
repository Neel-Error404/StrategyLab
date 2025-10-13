"""
Options historical data fetcher for Phase 1 validation.

Orchestrates fetching options data from Upstox and storing in structured format.
"""

import json
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path
import logging
from typing import List, Optional
from tqdm import tqdm
import sys

# Add repository root to Python path so `src.*` imports resolve
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.core.options.validation.upstox_options_api import UpstoxOptionsAPI
from src.core.options.validation.data_storage import OptionsDataStorage
from src.core.options.validation.config_loader import get_validation_config
from src.core.options.data.schemas import VALIDATION_TICKERS, VALIDATION_DATE_RANGE

def configure_logging(level: str, log_to_file: bool, log_file: Optional[str]) -> None:
    """
    Configure application logging based on config/CLI settings.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO) if isinstance(level, str) else level
    root_logger = logging.getLogger()

    # Clear existing handlers to avoid duplicate logs
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    root_logger.setLevel(numeric_level)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    if log_to_file and log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


class OptionsDataFetcher:
    """
    Fetches historical options data for validation.

    Workflow:
    1. Get list of expiries for ticker
    2. For each expiry:
       a. Get reference price from equity data (for strike filtering)
       b. Fetch option contracts (strikes)
       c. Fetch historical OHLC for each contract
       d. Save to parquet + metadata
    """

    def __init__(
        self,
        base_dir: str = "data/pools/options",
        equity_data_dir: Optional[str] = None,
        date_range: str = VALIDATION_DATE_RANGE,
        config_path: Optional[str] = None
    ):
        """
        Initialize options data fetcher.

        Args:
            base_dir: Base directory for options data
            equity_data_dir: Directory with equity data (for reference prices). If None, auto-detects.
            date_range: Date range string for organizing data
            config_path: Path to validation config (default: auto-detect)
        """
        self.logger = logging.getLogger(__name__)

        # FIX Issue #6: Load validation config
        self.config = get_validation_config(config_path)

        # Initialize API with rate limit and retry config
        self.api = UpstoxOptionsAPI(
            rate_limit=self.config.requests_per_second,
            requests_per_minute=self.config.requests_per_minute,
            retry_attempts=self.config.retry_attempts,
            retry_backoff_factor=self.config.retry_backoff_factor
        )
        self.storage = OptionsDataStorage(base_dir=base_dir)

        # FIX Issue #3: Use config equity_pool, then param, then auto-detect
        if equity_data_dir is None:
            equity_data_dir = self.config.equity_pool or self._auto_detect_equity_pool()

        self.equity_data_dir = Path(equity_data_dir) if equity_data_dir else None

        # FIX: Use config date_range if available, otherwise use parameter
        self.date_range = self.config.date_range or date_range

        # Cache for equity data
        self._equity_cache = {}
        self._equity_start_dates = {}

    def _auto_detect_equity_pool(self) -> Optional[str]:
        """
        Auto-detect the latest equity data pool directory.

        FIX Issue #3: Don't hard-code equity pool path

        Returns:
            Path to equity pool directory or None if not found
        """
        pools_dir = Path("data/pools")

        if not pools_dir.exists():
            self.logger.warning("data/pools directory not found")
            return None

        # Find all date range directories (format: YYYY-MM-DD_to_YYYY-MM-DD)
        import re
        date_range_pattern = re.compile(r'\d{4}-\d{2}-\d{2}_to_\d{4}-\d{2}-\d{2}')

        pool_dirs = [
            d for d in pools_dir.iterdir()
            if d.is_dir() and date_range_pattern.match(d.name) and d.name != 'options'
        ]

        if not pool_dirs:
            self.logger.warning("No equity data pools found in data/pools/")
            return None

        # Sort by end date (latest first)
        def get_end_date(dir_name):
            try:
                end_str = dir_name.split('_to_')[-1]
                return datetime.strptime(end_str, '%Y-%m-%d')
            except:
                return datetime.min

        pool_dirs.sort(key=lambda d: get_end_date(d.name), reverse=True)
        latest_pool = pool_dirs[0]

        self.logger.info(f"Auto-detected equity pool: {latest_pool}")
        return str(latest_pool)

    def _get_equity_start_date(self, ticker: str) -> Optional[date]:
        """
        Determine the earliest available equity date for a ticker.

        Returns:
            Earliest trading date as date object, or None if unavailable.
        """
        if self.equity_data_dir is None:
            return None

        if ticker in self._equity_start_dates:
            return self._equity_start_dates[ticker]

        equity_file = self.equity_data_dir / ticker / "1day.parquet"
        if not equity_file.exists():
            return None

        try:
            df = pd.read_parquet(equity_file, columns=['timestamp'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            if df.empty:
                return None
            start_date = df['timestamp'].min().date()
            self._equity_start_dates[ticker] = start_date
            return start_date
        except Exception as exc:
            self.logger.warning(f"Unable to determine start date for {ticker}: {exc}")
            return None

    def get_reference_price(self, ticker: str, target_date: date) -> Optional[float]:
        """
        Get underlying equity price on a specific date (for strike filtering).

        Args:
            ticker: Ticker symbol
            target_date: Date to get price for

        Returns:
            Close price or None if not available

        FIX Issue #4: Handle missing/empty equity data gracefully
        """
        # Check cache
        cache_key = (ticker, target_date)
        if cache_key in self._equity_cache:
            return self._equity_cache[cache_key]

        # Manual override (from config)
        manual_price = self.config.manual_reference_prices.get(ticker.upper()) or self.config.manual_reference_prices.get(ticker)
        if manual_price is not None:
            self.logger.info(
                "Using manual reference price %.2f for %s (config override).",
                manual_price,
                ticker,
            )
            self._equity_cache[cache_key] = float(manual_price)
            return float(manual_price)

        # Check if equity dir exists
        if self.equity_data_dir is None:
            self.logger.warning("No equity data directory configured")
            return None

        # Load equity data
        equity_file = self.equity_data_dir / ticker / "1day.parquet"

        if not equity_file.exists():
            self.logger.warning(f"Equity data not found for {ticker} at {equity_file}")
            return None

        try:
            df = pd.read_parquet(equity_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Find price on target date (or nearest before)
            df['date'] = df['timestamp'].dt.date
            filtered_df = df[df['date'] <= target_date]

            # FIX Issue #4: Check if filtered dataframe is empty before .iloc
            if filtered_df.empty:
                self.logger.warning(
                    f"No equity data for {ticker} before {target_date}. "
                    f"Available data starts from {df['date'].min()}"
                )
                return None

            price_row = filtered_df.iloc[-1]
            price = float(price_row['close'])
            self._equity_cache[cache_key] = price

            return price

        except Exception as e:
            self.logger.error(f"Error loading equity data for {ticker}: {e}")
            return None

    def fetch_ticker_data(
        self,
        ticker: str,
        timeframe: Optional[str] = None,
        max_expiries: Optional[int] = None,
        strike_range_pct: Optional[float] = None,
        min_open_interest: Optional[int] = None,
        min_volume: Optional[int] = None,
        max_spread_pct: Optional[float] = None,
        exclude_dte_below: Optional[int] = None,
        exclude_dte_above: Optional[int] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ):
        """
        Fetch all options data for a ticker.

        Args:
            ticker: Ticker symbol (e.g., "RELIANCE")
            timeframe: Data timeframe ('1day', '5m', etc.) - defaults from config
            max_expiries: Maximum number of expiries to fetch (for testing)
            strike_range_pct: Fetch strikes within ±X% of current price - defaults from config
            min_open_interest: Minimum OI to include - defaults from config
            start_date: Only fetch expiries after this date (optional)
            end_date: Only fetch expiries before this date (optional)
        """
        # FIX Issue #6: Use config defaults if not specified
        if timeframe is None:
            timeframe = self.config.timeframe
        if strike_range_pct is None:
            strike_range_pct = self.config.strike_range_pct
        if min_open_interest is None:
            min_open_interest = self.config.min_open_interest
        if min_volume is None:
            min_volume = self.config.min_volume
        if max_spread_pct is None:
            max_spread_pct = self.config.max_spread_pct
        if exclude_dte_below is None:
            exclude_dte_below = self.config.exclude_dte_below
        if exclude_dte_above is None:
            exclude_dte_above = self.config.exclude_dte_above
        max_strikes = self.config.max_strikes

        self.logger.info(f"=" * 80)
        self.logger.info(f"Fetching options data for {ticker}")
        self.logger.info(f"=" * 80)

        # Get instrument key
        instrument_key = self.api.get_instrument_key(ticker)
        self.logger.info(f"Instrument key: {instrument_key}")

        # Get expiries
        expiries = self.api.get_expiries(instrument_key)
        self.logger.info(f"Found {len(expiries)} expiries")

        # Filter by date range
        if start_date:
            expiries = [exp for exp in expiries if exp >= start_date]
            self.logger.info(f"Filtered to {len(expiries)} expiries after {start_date}")

        if end_date:
            expiries = [exp for exp in expiries if exp <= end_date]
            self.logger.info(f"Filtered to {len(expiries)} expiries before {end_date}")

        # Limit for testing
        if max_expiries:
            expiries = expiries[:max_expiries]
            self.logger.info(f"Limited to {len(expiries)} expiries for testing")

        if not expiries:
            self.logger.warning("No expiries to fetch")
            return

        # Fetch data for each expiry
        expiry_iter = tqdm(
            expiries,
            desc=f"{ticker} expiries",
            unit="expiry",
            disable=len(expiries) <= 1
        )

        for i, expiry in enumerate(expiry_iter, 1):
            self.logger.info(f"\n[{i}/{len(expiries)}] Processing expiry: {expiry}")

            try:
                self._fetch_expiry_data(
                    ticker=ticker,
                    expiry=expiry,
                    timeframe=timeframe,
                    strike_range_pct=strike_range_pct,
                    min_open_interest=min_open_interest,
                    min_volume=min_volume,
                    max_spread_pct=max_spread_pct,
                    max_strikes=max_strikes,
                    exclude_dte_below=exclude_dte_below,
                    exclude_dte_above=exclude_dte_above
                )
            except Exception as e:
                self.logger.error(f"Failed to fetch expiry {expiry}: {e}", exc_info=True)
                continue

        self.logger.info(f"\n{'=' * 80}")
        self.logger.info(f"Completed fetching {ticker}")
        self.logger.info(f"{'=' * 80}")

    def _fetch_expiry_data(
        self,
        ticker: str,
        expiry: date,
        timeframe: str,
        strike_range_pct: float,
        min_open_interest: int,
        min_volume: int,
        max_spread_pct: Optional[float],
        max_strikes: Optional[int],
        exclude_dte_below: Optional[int],
        exclude_dte_above: Optional[int]
    ):
        """
        Fetch and save data for a single expiry.

        Args:
            ticker: Ticker symbol
            expiry: Expiry date
            timeframe: Data timeframe
            strike_range_pct: Strike filtering range
            min_open_interest: Minimum OI filter
        """
        # Check if already exists
        existing = self.storage.load_expiry_data(ticker, expiry, timeframe, self.date_range)
        if existing is not None and not existing.empty:
            self.logger.info(f"Data already exists for {ticker} {expiry}, skipping")
            return

        # Get reference price (1 month before expiry or at start of data range)
        # Use a date when the options were likely already trading
        # FIX: Use config.reference_date_start instead of hard-coded date
        config_start_date = self.config.reference_date_start
        if config_start_date is None:
            config_start_date = self._get_equity_start_date(ticker)
        if config_start_date is None:
            config_start_date = expiry - timedelta(days=60)
        reference_date = max(
            expiry - timedelta(days=30),  # 1 month before expiry
            config_start_date              # Start from config (or fallback)
        )

        reference_price = self.get_reference_price(ticker, reference_date)

        if reference_price is None:
            self.logger.warning(f"No reference price for {ticker} on {reference_date}, fetching all strikes")
            reference_price = None  # Fetch all strikes (no filtering)

        self.logger.info(f"Reference price: {reference_price} (date: {reference_date})")

        # Fetch options data
        df = self.api.fetch_options_for_expiry(
            ticker=ticker,
            expiry_date=expiry,
            interval=timeframe if timeframe != '1day' else 'day',  # API uses 'day' not '1day'
            reference_price=reference_price,
            strike_range_pct=strike_range_pct,
            min_open_interest=min_open_interest,
            min_volume=min_volume,
            max_spread_pct=max_spread_pct,
            max_strikes=max_strikes,
            exclude_dte_below=exclude_dte_below,
            exclude_dte_above=exclude_dte_above
        )

        if df.empty:
            self.logger.warning(f"No data fetched for {ticker} {expiry}")
            return

        # Log summary
        num_strikes = df['strike'].nunique()
        num_bars = len(df)
        date_range_str = f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}"

        self.logger.info(
            f"Fetched {num_bars} rows ({num_strikes} strikes) "
            f"from {date_range_str}"
        )

        # Save data
        self.storage.save_expiry_data(
            df=df,
            ticker=ticker,
            expiry=expiry,
            timeframe=timeframe,
            date_range=self.date_range
        )

        self.logger.info(f"Saved data for {ticker} {expiry}")

    def fetch_validation_dataset(
        self,
        tickers: Optional[List[str]] = None,
        timeframe: Optional[str] = None,
        max_expiries_per_ticker: Optional[int] = None
    ):
        """
        Fetch validation dataset for multiple tickers.

        Args:
            tickers: List of tickers (default: from config or VALIDATION_TICKERS)
            timeframe: Data timeframe (default: from config)
            max_expiries_per_ticker: Limit expiries per ticker (for testing)
        """
        # FIX Issue #6: Use config defaults if not specified
        if tickers is None:
            tickers = self.config.tickers if self.config.tickers else VALIDATION_TICKERS
        if timeframe is None:
            timeframe = self.config.timeframe

        self.logger.info(f"\n{'#' * 80}")
        self.logger.info(f"# Fetching Validation Dataset")
        self.logger.info(f"# Tickers: {', '.join(tickers)}")
        self.logger.info(f"# Timeframe: {timeframe}")
        self.logger.info(f"# Date Range: {self.date_range}")
        self.logger.info(f"{'#' * 80}\n")

        # Determine date range for expiries
        # Parse date range string: "2025-04-01_to_2025-10-08"
        start_str, end_str = self.date_range.split("_to_")
        start_date = datetime.strptime(start_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_str, '%Y-%m-%d').date()

        for i, ticker in enumerate(tickers, 1):
            self.logger.info(f"\n{'*' * 80}")
            self.logger.info(f"* Ticker {i}/{len(tickers)}: {ticker}")
            self.logger.info(f"{'*' * 80}")

            try:
                self.fetch_ticker_data(
                    ticker=ticker,
                    timeframe=timeframe,
                    max_expiries=max_expiries_per_ticker,
                    start_date=start_date,
                    end_date=end_date
                )
            except Exception as e:
                self.logger.error(f"Failed to fetch {ticker}: {e}", exc_info=True)
                continue

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print summary of fetched data."""
        stats = self.storage.get_storage_stats(self.date_range)

        self.logger.info(f"\n{'=' * 80}")
        self.logger.info(f"FETCH SUMMARY")
        self.logger.info(f"{'=' * 80}")
        self.logger.info(f"Date Range: {stats['date_range']}")
        self.logger.info(f"Total Files: {stats['total_files']}")
        self.logger.info(f"Total Size: {stats['total_size_mb']:.2f} MB")
        self.logger.info(f"\nTickers:")

        for ticker_stats in stats['tickers']:
            self.logger.info(f"\n  {ticker_stats['ticker']}:")
            for timeframe, tf_stats in ticker_stats['timeframes'].items():
                self.logger.info(f"    {timeframe}: {tf_stats['expiries_count']} expiries")

        # Persist summary if configured
        output_dir = self.config.output_dir
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            summary_file = output_path / f"fetch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)
            self.logger.info(f"\nSummary saved to {summary_file}")

        self.logger.info(f"\n{'=' * 80}\n")


def main():
    """Main entry point for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Fetch options historical data for validation")
    parser.add_argument('--ticker', type=str, help='Ticker to fetch (default: from config or RELIANCE)')
    parser.add_argument('--timeframe', type=str, help='Timeframe (default: from config or 1day)')
    parser.add_argument('--max-expiries', type=int, default=None, help='Max expiries to fetch (for testing)')
    parser.add_argument('--all', action='store_true', help='Fetch all validation tickers')
    parser.add_argument('--config', type=str, default=None, help='Path to validation config (default: auto-detect)')
    parser.add_argument('--log-level', type=str, default=None, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level (default: from config or INFO)')

    args = parser.parse_args()

    # Load config upfront for logging setup
    config = get_validation_config(args.config)

    # Setup logging using config defaults unless overridden via CLI
    log_level = args.log_level or config.log_level
    configure_logging(
        level=log_level,
        log_to_file=config.log_to_file,
        log_file=config.log_file
    )

    # FIX Issue #6: Load config first to get defaults
    fetcher = OptionsDataFetcher(config_path=args.config)

    if args.all:
        # Fetch all validation tickers (tickers come from config)
        fetcher.fetch_validation_dataset(
            timeframe=args.timeframe,  # None = use config default
            max_expiries_per_ticker=args.max_expiries
        )
    else:
        # Fetch single ticker (default to RELIANCE if not specified)
        ticker = args.ticker or 'RELIANCE'
        fetcher.fetch_ticker_data(
            ticker=ticker,
            timeframe=args.timeframe,  # None = use config default
            max_expiries=args.max_expiries
        )

        # Print summary
        fetcher.print_summary()


if __name__ == "__main__":
    main()
