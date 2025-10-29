"""
Pool Inspector - Analyze existing data pools

This module inspects existing data pools to extract metadata needed for
incremental updates: tickers, timeframes, last dates, schema, integrity.

Author: StrategyLab
Created: 2025-10-08
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field

import pandas as pd
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PoolMetadata:
    """Metadata about a data pool"""
    pool_path: str
    tickers: List[str]
    timeframes: List[str]
    last_dates: Dict[Tuple[str, str], datetime]  # (ticker, timeframe) -> last_date
    first_dates: Dict[Tuple[str, str], datetime]  # (ticker, timeframe) -> first_date
    schema: Dict[str, Any]
    row_counts: Dict[Tuple[str, str], int]
    file_sizes: Dict[Tuple[str, str], float]  # in MB
    health_status: str
    date_range: Tuple[str, str]  # (start, end) from pool path
    total_records: int = 0
    total_size_mb: float = 0.0
    issues: List[str] = field(default_factory=list)


def detect_pool_layout(pool_path: Path) -> str:
    """
    Auto-detect whether pool uses ticker-first or timeframe-first layout

    Args:
        pool_path: Path to pool directory

    Returns:
        'ticker-first' or 'timeframe-first'

    Raises:
        ValueError: If layout cannot be determined
    """
    # Sample first few directories
    subdirs = [d for d in pool_path.iterdir() if d.is_dir() and not d.name.startswith('.')]

    if not subdirs:
        raise ValueError(f"No subdirectories found in pool: {pool_path}")

    # Check first directory
    first_dir = subdirs[0]
    parquet_files = list(first_dir.glob("*.parquet"))

    if parquet_files:
        # Has parquet files directly ÔåÆ could be either layout
        # Check file naming pattern to distinguish
        sample_file = parquet_files[0]

        # Ticker-first: files named like "5m.parquet", "1minute.parquet", "1day.parquet"
        # Timeframe-first: files named like "RELIANCE.parquet", "HDFCBANK.parquet"

        # Heuristic: ticker names are usually uppercase, timeframe names have numbers or lowercase
        # Check if file name looks like a timeframe (has numbers or common timeframe patterns)
        file_stem = sample_file.stem

        # Timeframe patterns: 5m, 15m, 1minute, 30minute, 1day, etc.
        if any(char.isdigit() for char in file_stem) or file_stem in ['minute', 'day', 'week', 'month', 'hour']:
            # Likely a timeframe ÔåÆ directory is ticker-first
            return 'ticker-first'
        elif file_stem.isupper() or len(file_stem) > 8:
            # Likely a ticker name ÔåÆ directory is timeframe-first
            return 'timeframe-first'
        else:
            # Default to ticker-first as it's more common
            return 'ticker-first'
    else:
        # No parquet files directly ÔåÆ must be ticker-first with nested structure
        # But let's verify by checking one level deeper
        nested_dirs = [d for d in first_dir.iterdir() if d.is_dir()]
        if nested_dirs:
            raise ValueError(f"Unexpected nested directory structure in {first_dir}")

        # No parquet files at all in first directory
        raise ValueError(f"No parquet files found in {first_dir}")


def inspect_pool(pool_path: str, validate: bool = True) -> PoolMetadata:
    """
    Inspect existing data pool and extract metadata

    Supports both ticker-first and timeframe-first layouts:
    - Ticker-first: data/pools/date_range/RELIANCE/5m.parquet
    - Timeframe-first: data/pools/date_range/1minute/RELIANCE.parquet

    Args:
        pool_path: Path to existing pool (e.g., 'data/pools/2022-01-01_to_2025-08-31/')
        validate: Whether to run integrity validation

    Returns:
        PoolMetadata object containing pool information

    Raises:
        ValueError: If pool path doesn't exist or is invalid
    """
    logger.info(f"­ƒöì Inspecting pool: {pool_path}")

    # Validate pool path exists
    pool_path = Path(pool_path)
    if not pool_path.exists():
        raise ValueError(f"Pool path does not exist: {pool_path}")

    if not pool_path.is_dir():
        raise ValueError(f"Pool path is not a directory: {pool_path}")

    # Extract date range from pool path
    pool_name = pool_path.name
    date_range = extract_date_range_from_path(pool_name)

    # Auto-detect pool layout
    layout = detect_pool_layout(pool_path)
    logger.info(f"   Detected layout: {layout}")

    # Scan all files and extract metadata
    tickers = set()
    timeframes = set()
    last_dates = {}
    first_dates = {}
    row_counts = {}
    file_sizes = {}
    schema_samples = []
    issues = []

    # Scan based on layout
    subdirs = [d for d in pool_path.iterdir() if d.is_dir() and not d.name.startswith('.')]

    for subdir in subdirs:
        # Get all parquet files, excluding backup files
        all_parquet_files = list(subdir.glob("*.parquet"))
        parquet_files = [f for f in all_parquet_files if '.backup' not in f.name]

        if not parquet_files:
            issues.append(f"No parquet files in {subdir.name}/")
            continue

        if layout == 'ticker-first':
            # subdir.name is the ticker (e.g., RELIANCE)
            ticker = subdir.name
            tickers.add(ticker)

            logger.info(f"   Scanning {len(parquet_files)} timeframes for {ticker}...")

            for file_path in parquet_files:
                timeframe = file_path.stem  # e.g., "5m", "1minute"
                timeframes.add(timeframe)

                try:
                    # Get file metadata efficiently
                    metadata = get_parquet_metadata(str(file_path))

                    last_dates[(ticker, timeframe)] = metadata['last_date']
                    first_dates[(ticker, timeframe)] = metadata['first_date']
                    row_counts[(ticker, timeframe)] = metadata['row_count']
                    file_sizes[(ticker, timeframe)] = metadata['file_size_mb']

                    # Collect schema sample
                    if not schema_samples:
                        schema_samples.append(metadata['schema'])

                except Exception as e:
                    issues.append(f"Error reading {ticker}/{timeframe}.parquet: {str(e)}")
                    logger.warning(f"   ÔÜá´©Å  Error reading {ticker}/{timeframe}: {str(e)}")

        else:  # timeframe-first
            # subdir.name is the timeframe (e.g., 1minute, 5minute)
            timeframe = subdir.name
            timeframes.add(timeframe)

            logger.info(f"   Scanning {len(parquet_files)} tickers in {timeframe}/...")

            for file_path in parquet_files:
                ticker = file_path.stem  # e.g., "RELIANCE", "HDFCBANK"
                tickers.add(ticker)

                try:
                    # Get file metadata efficiently
                    metadata = get_parquet_metadata(str(file_path))

                    last_dates[(ticker, timeframe)] = metadata['last_date']
                    first_dates[(ticker, timeframe)] = metadata['first_date']
                    row_counts[(ticker, timeframe)] = metadata['row_count']
                    file_sizes[(ticker, timeframe)] = metadata['file_size_mb']

                    # Collect schema sample
                    if not schema_samples:
                        schema_samples.append(metadata['schema'])

                except Exception as e:
                    issues.append(f"Error reading {timeframe}/{ticker}.parquet: {str(e)}")
                    logger.warning(f"   ÔÜá´©Å  Error reading {ticker}: {str(e)}")

    tickers = sorted(list(tickers))
    timeframes = sorted(list(timeframes))
    logger.info(f"   Found {len(tickers)} tickers across {len(timeframes)} timeframes")

    # Calculate totals
    total_records = sum(row_counts.values())
    total_size_mb = sum(file_sizes.values())

    # Determine schema (use first sample)
    schema = schema_samples[0] if schema_samples else {}

    # Determine health status
    if validate:
        health_status, validation_issues = validate_pool_integrity(
            pool_path, tickers, timeframes, last_dates, first_dates, row_counts
        )
        issues.extend(validation_issues)
    else:
        health_status = "UNKNOWN"

    # Fallback: Extract date range from actual data if path-based extraction failed
    if date_range == ("UNKNOWN", "UNKNOWN") and last_dates and first_dates:
        logger.info(f"   ­ƒôà Pool name has no date info, extracting from data...")
        date_range = extract_date_range_from_data(pool_path, first_dates, last_dates)
        logger.info(f"   ­ƒôà Detected date range: {date_range[0]} to {date_range[1]}")

    # Create metadata object
    metadata = PoolMetadata(
        pool_path=str(pool_path),
        tickers=tickers,
        timeframes=timeframes,
        last_dates=last_dates,
        first_dates=first_dates,
        schema=schema,
        row_counts=row_counts,
        file_sizes=file_sizes,
        health_status=health_status,
        date_range=date_range,
        total_records=total_records,
        total_size_mb=total_size_mb,
        issues=issues
    )

    logger.info(f"   Ô£à Inspection complete")
    logger.info(f"      Total records: {total_records:,}")
    logger.info(f"      Total size: {total_size_mb:.1f} MB")
    logger.info(f"      Health: {health_status}")

    return metadata


def get_parquet_metadata(file_path: str) -> Dict[str, Any]:
    """
    Efficiently extract metadata from parquet file without loading full data

    Args:
        file_path: Path to parquet file

    Returns:
        Dict with keys: last_date, first_date, row_count, file_size_mb, schema
    """
    # Use pyarrow for efficient metadata reading
    parquet_file = pq.ParquetFile(file_path)

    # Get row count from metadata
    row_count = parquet_file.metadata.num_rows

    # Get file size
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

    # Get schema (use schema_arrow for proper PyArrow schema access)
    arrow_schema = parquet_file.schema_arrow
    schema = {
        'columns': arrow_schema.names,
        'dtypes': {name: str(arrow_schema.field(name).type) for name in arrow_schema.names}
    }

    # Read only first and last row to get date range efficiently
    # Read first row
    first_batch = parquet_file.read_row_group(0, columns=['timestamp'])
    first_df = first_batch.to_pandas()
    first_date = first_df['timestamp'].iloc[0]

    # Read last row
    last_row_group = parquet_file.num_row_groups - 1
    last_batch = parquet_file.read_row_group(last_row_group, columns=['timestamp'])
    last_df = last_batch.to_pandas()
    last_date = last_df['timestamp'].iloc[-1]

    # Convert to datetime if not already
    if not isinstance(first_date, datetime):
        first_date = pd.to_datetime(first_date)
    if not isinstance(last_date, datetime):
        last_date = pd.to_datetime(last_date)

    return {
        'first_date': first_date,
        'last_date': last_date,
        'row_count': row_count,
        'file_size_mb': file_size_mb,
        'schema': schema
    }


def extract_date_range_from_path(pool_name: str) -> Tuple[str, str]:
    """
    Extract start and end dates from pool directory name

    Args:
        pool_name: Pool directory name (e.g., '2022-01-01_to_2025-08-31')

    Returns:
        Tuple of (start_date, end_date) as strings
    """
    try:
        parts = pool_name.split('_to_')
        if len(parts) == 2:
            return (parts[0], parts[1])
        else:
            # Try to parse dates if format is different
            return ("UNKNOWN", "UNKNOWN")
    except:
        return ("UNKNOWN", "UNKNOWN")


def extract_date_range_from_data(pool_path: Path, first_dates: Dict, last_dates: Dict) -> Tuple[str, str]:
    """
    Extract date range from actual data when pool name doesn't contain dates
    
    Args:
        pool_path: Path to pool directory
        first_dates: Dict of (ticker, timeframe) -> first_date
        last_dates: Dict of (ticker, timeframe) -> last_date
        
    Returns:
        Tuple of (start_date, end_date) as strings
    """
    if not first_dates or not last_dates:
        return ("UNKNOWN", "UNKNOWN")
    
    try:
        # Find earliest and latest dates across all data
        all_first_dates = [date for date in first_dates.values() if date is not None]
        all_last_dates = [date for date in last_dates.values() if date is not None]
        
        if all_first_dates and all_last_dates:
            earliest_date = min(all_first_dates)
            latest_date = max(all_last_dates)
            
            # Convert to string format
            start_str = earliest_date.strftime('%Y-%m-%d')
            end_str = latest_date.strftime('%Y-%m-%d')
            
            return (start_str, end_str)
        else:
            return ("UNKNOWN", "UNKNOWN")
    except Exception as e:
        logger.warning(f"Error extracting date range from data: {e}")
        return ("UNKNOWN", "UNKNOWN")


def validate_pool_integrity(
    pool_path: Path,
    tickers: List[str],
    timeframes: List[str],
    last_dates: Dict,
    first_dates: Dict,
    row_counts: Dict = None
) -> Tuple[str, List[str]]:
    """
    Validate pool integrity

    Args:
        pool_path: Path to pool
        tickers: List of tickers
        timeframes: List of timeframes
        last_dates: Dict of (ticker, tf) -> last_date
        first_dates: Dict of (ticker, tf) -> first_date
        row_counts: Dict of (ticker, tf) -> row_count

    Returns:
        Tuple of (health_status, list_of_issues)
    """
    issues = []

    # Check 1: All tickers have all timeframes
    for ticker in tickers:
        for timeframe in timeframes:
            key = (ticker, timeframe)
            if key not in last_dates:
                issues.append(f"Missing data for {ticker} in {timeframe}/")

    # Check 2: Date consistency across timeframes for same ticker
    for ticker in tickers:
        last_dates_for_ticker = [
            last_dates.get((ticker, tf))
            for tf in timeframes
            if (ticker, tf) in last_dates
        ]

        if last_dates_for_ticker:
            # Check if all last dates are within 3 days of each other
            min_date = min(last_dates_for_ticker)
            max_date = max(last_dates_for_ticker)
            diff = (max_date - min_date).days

            if diff > 3:
                issues.append(
                    f"{ticker}: Last dates vary by {diff} days across timeframes "
                    f"({min_date.date()} to {max_date.date()})"
                )

    # Check 3: No empty files (use row_counts, not last_dates)
    if row_counts:
        for (ticker, tf), count in row_counts.items():
            if count == 0:
                issues.append(f"Empty file: {ticker} @ {tf} (0 rows)")

    # Determine overall health
    if not issues:
        health_status = "HEALTHY"
    elif len(issues) <= 3:
        health_status = "WARNING"
    else:
        health_status = "UNHEALTHY"

    return health_status, issues


def print_pool_summary(metadata: PoolMetadata):
    """Print human-readable pool summary"""
    print("\n" + "="*60)
    print(f"POOL INSPECTION SUMMARY")
    print("="*60)
    print(f"­ƒôü Path: {metadata.pool_path}")
    print(f"­ƒôà Date Range: {metadata.date_range[0]} to {metadata.date_range[1]}")
    print(f"­ƒÅÀ´©Å  Tickers: {len(metadata.tickers)} ({', '.join(metadata.tickers[:5])}{'...' if len(metadata.tickers) > 5 else ''})")
    print(f"ÔÅ▒´©Å  Timeframes: {', '.join(metadata.timeframes)}")
    print(f"­ƒôè Total Records: {metadata.total_records:,}")
    print(f"­ƒÆ¥ Total Size: {metadata.total_size_mb:.1f} MB")
    print(f"­ƒÅÑ Health: {metadata.health_status}")

    if metadata.issues:
        print(f"\nÔÜá´©Å  Issues Found ({len(metadata.issues)}):")
        for i, issue in enumerate(metadata.issues[:5], 1):
            print(f"   {i}. {issue}")
        if len(metadata.issues) > 5:
            print(f"   ... and {len(metadata.issues) - 5} more")

    # Show date ranges for each ticker/timeframe
    print(f"\n­ƒôà Last Dates by Ticker (sample):")
    sample_tickers = metadata.tickers[:3]
    for ticker in sample_tickers:
        for tf in metadata.timeframes:
            key = (ticker, tf)
            if key in metadata.last_dates:
                last_date = metadata.last_dates[key].strftime('%Y-%m-%d')
                records = metadata.row_counts[key]
                size = metadata.file_sizes[key]
                print(f"   {ticker:12} {tf:10} ÔåÆ {last_date} ({records:>8,} records, {size:>6.1f} MB)")

    print("="*60 + "\n")


# CLI interface for testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pool_inspector.py <pool_path>")
        print("Example: python pool_inspector.py data/pools/2022-01-01_to_2025-08-31/")
        sys.exit(1)

    pool_path = sys.argv[1]

    try:
        metadata = inspect_pool(pool_path, validate=True)
        print_pool_summary(metadata)

        # Exit with appropriate code
        if metadata.health_status == "HEALTHY":
            sys.exit(0)
        elif metadata.health_status == "WARNING":
            sys.exit(1)
        else:
            sys.exit(2)

    except Exception as e:
        logger.error(f"ÔØî Error: {str(e)}")
        sys.exit(3)
