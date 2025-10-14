"""
Gap Calculator - Determine missing data for incremental updates

This module calculates which data needs to be fetched to update an existing
data pool from its last date to a target date.

Author: StrategyLab
Created: 2025-10-08
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GapReport:
    """Report of data gaps to be filled"""
    gaps: Dict[Tuple[str, str], Tuple[datetime, datetime]]  # (ticker, tf) -> (start, end)
    total_calendar_days: int
    total_trading_days_estimate: int
    total_records_estimate: int
    estimated_size_mb: float
    fetch_time_estimate_min: int
    validation_status: str
    validation_messages: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def calculate_gaps(
    pool_metadata,
    target_end_date: str = None,
    buffer_days: int = 0
) -> GapReport:
    """
    Calculate missing date ranges per ticker/timeframe

    Args:
        pool_metadata: PoolMetadata object from pool_inspector
        target_end_date: Desired end date (default: today). Format: 'YYYY-MM-DD'
        buffer_days: Extra days to subtract from target (for safety)

    Returns:
        GapReport containing all gaps and estimates

    Raises:
        ValueError: If gaps are invalid or target date is before pool end
    """
    logger.info(f"📊 Calculating gaps...")

    # Parse target date
    if target_end_date is None:
        target_date = datetime.now().date()
    else:
        target_date = pd.to_datetime(target_end_date).date()

    # Apply buffer
    if buffer_days > 0:
        target_date = target_date - timedelta(days=buffer_days)
        logger.info(f"   Applied {buffer_days} day buffer → {target_date}")

    # Find the earliest last_date across all ticker/timeframe combinations
    # (most conservative - ensures we fetch enough data for all files)
    if not pool_metadata.last_dates:
        raise ValueError("No last dates found in pool metadata")

    min_last_date = min(pool_metadata.last_dates.values()).date()
    max_last_date = max(pool_metadata.last_dates.values()).date()

    logger.info(f"   Pool last dates range: {min_last_date} to {max_last_date}")
    logger.info(f"   Target end date: {target_date}")

    # Validate target date
    validation_status = "VALID"
    validation_messages = []
    warnings = []

    if target_date <= max_last_date:
        validation_status = "ERROR"
        validation_messages.append(
            f"Target date {target_date} is not after pool's last date {max_last_date}"
        )
        raise ValueError(
            f"Target date ({target_date}) must be after pool's last date ({max_last_date}). "
            f"Pool is already up to date or target date is in the past."
        )

    # Check if gap is very large (>180 days)
    gap_days = (target_date - max_last_date).days
    if gap_days > 180:
        warnings.append(
            f"Large gap detected: {gap_days} days. Consider splitting into smaller updates."
        )

    # Calculate gaps for each ticker/timeframe
    gaps = {}

    for ticker in pool_metadata.tickers:
        for timeframe in pool_metadata.timeframes:
            key = (ticker, timeframe)

            if key in pool_metadata.last_dates:
                last_date = pool_metadata.last_dates[key].date()

                # Gap starts the day after last_date
                gap_start = last_date + timedelta(days=1)
                gap_end = target_date

                gaps[key] = (
                    datetime.combine(gap_start, datetime.min.time()),
                    datetime.combine(gap_end, datetime.max.time())
                )

    if not gaps:
        raise ValueError("No gaps calculated - something went wrong")

    # Calculate estimates
    total_calendar_days = (target_date - max_last_date).days
    total_trading_days = estimate_trading_days(max_last_date, target_date)

    # Estimate records per ticker/timeframe based on historical data
    estimates = estimate_data_volume(
        pool_metadata,
        gaps,
        total_trading_days
    )

    # Create report
    report = GapReport(
        gaps=gaps,
        total_calendar_days=total_calendar_days,
        total_trading_days_estimate=total_trading_days,
        total_records_estimate=estimates['total_records'],
        estimated_size_mb=estimates['total_size_mb'],
        fetch_time_estimate_min=estimates['fetch_time_min'],
        validation_status=validation_status,
        validation_messages=validation_messages,
        warnings=warnings
    )

    logger.info(f"   ✅ Gap calculation complete")
    logger.info(f"      Calendar days: {total_calendar_days}")
    logger.info(f"      Trading days (est): {total_trading_days}")
    logger.info(f"      Records (est): {estimates['total_records']:,}")
    logger.info(f"      Size (est): {estimates['total_size_mb']:.1f} MB")

    return report


def estimate_trading_days(start_date, end_date) -> int:
    """
    Estimate number of trading days between two dates

    Assumes ~252 trading days per year (excludes weekends and major holidays)

    Args:
        start_date: Start date
        end_date: End date

    Returns:
        Estimated number of trading days
    """
    calendar_days = (end_date - start_date).days

    # Rough estimate: ~72% of days are trading days (252/365)
    # This accounts for weekends (~104 days) and holidays (~9 days)
    trading_days = int(calendar_days * 0.72)

    return max(1, trading_days)  # At least 1 day


def estimate_data_volume(
    pool_metadata,
    gaps: Dict,
    trading_days: int
) -> Dict[str, float]:
    """
    Estimate data volume to be fetched based on historical pool statistics

    Args:
        pool_metadata: PoolMetadata object
        gaps: Dict of (ticker, tf) -> (start, end)
        trading_days: Number of trading days to fetch

    Returns:
        Dict with keys: total_records, total_size_mb, fetch_time_min
    """
    # Calculate average records per trading day from existing pool
    if pool_metadata.total_records == 0:
        # Fallback estimates if pool is empty (shouldn't happen)
        records_per_day_per_ticker = {
            '1minute': 375,    # ~6.25 hours * 60 minutes
            '5minute': 75,     # ~6.25 hours * 12 (5-min bars)
            '1day': 1,
            'default': 100
        }
    else:
        # Calculate from pool's date range
        pool_start_date = pd.to_datetime(pool_metadata.date_range[0]).date()
        pool_end_date = pd.to_datetime(pool_metadata.date_range[1]).date()
        pool_trading_days = estimate_trading_days(pool_start_date, pool_end_date)

        # Records per day across all tickers/timeframes
        records_per_day = pool_metadata.total_records / max(1, pool_trading_days)

        # Estimate per timeframe based on pool statistics
        records_per_day_per_ticker = {}
        for tf in pool_metadata.timeframes:
            # Count total records for this timeframe
            tf_records = sum(
                count for (ticker, timeframe), count in pool_metadata.row_counts.items()
                if timeframe == tf
            )
            tf_tickers = len([t for t in pool_metadata.tickers])
            records_per_day_per_ticker[tf] = tf_records / (pool_trading_days * tf_tickers)

    # Estimate for gaps
    total_records = 0
    total_size_mb = 0.0

    for (ticker, timeframe), (gap_start, gap_end) in gaps.items():
        # Get records per day for this timeframe
        records_per_day = records_per_day_per_ticker.get(
            timeframe,
            records_per_day_per_ticker.get('default', 100)
        )

        # Estimate records for this gap
        gap_records = int(records_per_day * trading_days)
        total_records += gap_records

        # Estimate size (rough: ~200 bytes per row for typical OHLCV + indicators)
        gap_size_mb = (gap_records * 200) / (1024 * 1024)
        total_size_mb += gap_size_mb

    # Estimate fetch time
    # Rough: 30 seconds per ticker per timeframe + 2 seconds per day
    num_tickers = len(pool_metadata.tickers)
    num_timeframes = len(pool_metadata.timeframes)
    fetch_time_min = int((num_tickers * num_timeframes * 30 + trading_days * 2) / 60)
    fetch_time_min = max(1, fetch_time_min)  # At least 1 minute

    return {
        'total_records': total_records,
        'total_size_mb': total_size_mb,
        'fetch_time_min': fetch_time_min
    }


def validate_gap(gap_start: datetime, gap_end: datetime, last_pool_date: datetime) -> Tuple[bool, str]:
    """
    Validate that a gap makes sense

    Args:
        gap_start: Start of gap
        gap_end: End of gap
        last_pool_date: Last date in existing pool

    Returns:
        Tuple of (is_valid, message)
    """
    # Check 1: Gap start should be day after last_pool_date
    expected_start = last_pool_date.date() + timedelta(days=1)
    if gap_start.date() != expected_start:
        return False, f"Gap start {gap_start.date()} should be {expected_start}"

    # Check 2: Gap end should be in the future relative to pool
    if gap_end.date() <= last_pool_date.date():
        return False, f"Gap end {gap_end.date()} is not after pool last date {last_pool_date.date()}"

    # Check 3: Gap should not be too far in the future
    today = datetime.now().date()
    if gap_end.date() > today + timedelta(days=7):
        return False, f"Gap end {gap_end.date()} is more than 7 days in the future"

    return True, "Valid gap"


def print_gap_report(report: GapReport):
    """Print human-readable gap report"""
    print("\n" + "="*60)
    print(f"GAP ANALYSIS REPORT")
    print("="*60)

    print(f"📅 Gap Duration: {report.total_calendar_days} calendar days")
    print(f"📈 Trading Days (est): {report.total_trading_days_estimate}")
    print(f"📊 Records to Fetch (est): {report.total_records_estimate:,}")
    print(f"💾 Data Size (est): {report.estimated_size_mb:.1f} MB")
    print(f"⏱️  Fetch Time (est): {report.fetch_time_estimate_min} minutes")
    print(f"✅ Status: {report.validation_status}")

    if report.warnings:
        print(f"\n⚠️  Warnings:")
        for i, warning in enumerate(report.warnings, 1):
            print(f"   {i}. {warning}")

    if report.validation_messages:
        print(f"\n❌ Validation Issues:")
        for i, msg in enumerate(report.validation_messages, 1):
            print(f"   {i}. {msg}")

    # Show sample gaps
    print(f"\n📋 Sample Gaps (first 3 tickers, all timeframes):")
    sample_keys = list(report.gaps.keys())[:9]  # 3 tickers * 3 timeframes
    for (ticker, timeframe), (start, end) in [(k, report.gaps[k]) for k in sample_keys]:
        print(f"   {ticker:12} {timeframe:10} → {start.date()} to {end.date()}")

    if len(report.gaps) > 9:
        print(f"   ... and {len(report.gaps) - 9} more gaps")

    print("="*60 + "\n")


# CLI interface for testing
if __name__ == "__main__":
    import sys
    from pool_inspector import inspect_pool

    if len(sys.argv) < 2:
        print("Usage: python gap_calculator.py <pool_path> [target_date]")
        print("Example: python gap_calculator.py data/pools/2022-01-01_to_2025-08-31/ 2025-10-08")
        sys.exit(1)

    pool_path = sys.argv[1]
    target_date = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        # Inspect pool first
        print("Step 1: Inspecting pool...")
        metadata = inspect_pool(pool_path, validate=False)

        # Calculate gaps
        print("\nStep 2: Calculating gaps...")
        report = calculate_gaps(metadata, target_end_date=target_date)

        # Print report
        print_gap_report(report)

        sys.exit(0 if report.validation_status == "VALID" else 1)

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
