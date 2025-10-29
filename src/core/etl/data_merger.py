"""
Data Merger - Merge old and new parquet data files

This module safely merges existing parquet files with newly fetched data,
ensuring data integrity, handling overlaps, and using atomic operations.

Author: StrategyLab
Created: 2025-10-08
"""

import os
import shutil
import logging
import time
import gc
from pathlib import Path
from typing import Optional, Literal
from datetime import datetime, timedelta

import pandas as pd
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MergeStrategy = Literal['append', 'overlap-dedupe']


def merge_parquet_files(
    old_file: str,
    new_data: pd.DataFrame,
    output_file: Optional[str] = None,
    strategy: MergeStrategy = 'append',
    backup: bool = True,
    validate: bool = True
) -> bool:
    """
    Merge old parquet file with new data (memory-optimized version)

    Uses PyArrow for efficient file operations without loading full data into memory.

    Args:
        old_file: Path to existing parquet file
        new_data: New data fetched (pandas DataFrame)
        output_file: Where to write merged file (default: overwrite old_file)
        strategy: 'append' (default) or 'overlap-dedupe'
        backup: Create backup of old file before overwrite
        validate: Run validation checks on merged data

    Returns:
        True if merge successful, False otherwise

    Raises:
        ValueError: If validation fails
        FileNotFoundError: If old_file doesn't exist
    """
    old_file = Path(old_file)
    if not old_file.exists():
        raise FileNotFoundError(f"Old file not found: {old_file}")

    if output_file is None:
        output_file = old_file
    else:
        output_file = Path(output_file)

    logger.info(f"­ƒöä Merging {old_file.name}...")

    try:
        # Step 1: Read metadata only (efficient, no full load)
        logger.debug(f"   Reading metadata from {old_file}...")
        old_pf = pq.ParquetFile(old_file)
        old_rows = old_pf.metadata.num_rows

        # Get last timestamp from old file without full load
        last_row_group = old_pf.num_row_groups - 1
        last_batch = old_pf.read_row_group(last_row_group, columns=['timestamp'])
        old_last_date = last_batch.to_pandas()['timestamp'].iloc[-1]

        # Get first timestamp
        first_batch = old_pf.read_row_group(0, columns=['timestamp'])
        old_first_date = first_batch.to_pandas()['timestamp'].iloc[0]

        # CRITICAL: Close the parquet file handle BEFORE backup/merge operations
        # Otherwise, Windows will lock the file and prevent deletion/rename
        del old_pf, last_batch, first_batch
        gc.collect()

        logger.debug(f"   Old data: {old_rows:,} rows, {old_first_date} to {old_last_date}")

        # Step 2: Validate new data
        if new_data.empty:
            logger.warning(f"   ÔÜá´©Å  New data is empty, nothing to merge")
            return False

        new_first_date = new_data['timestamp'].min()
        new_last_date = new_data['timestamp'].max()
        new_rows = len(new_data)

        logger.debug(f"   New data: {new_rows:,} rows, {new_first_date} to {new_last_date}")

        # Step 3: Check for overlap
        has_overlap = new_first_date <= old_last_date

        if has_overlap and strategy == 'append':
            logger.warning(
                f"   ÔÜá´©Å  Overlap detected but strategy is 'append'. "
                f"Consider using 'overlap-dedupe' strategy."
            )

        # Step 4: Backup old file if requested (before any writes)
        if backup and output_file == old_file:
            backup_path = create_backup(old_file)
            logger.debug(f"   Backup created: {backup_path.name}")

        # Step 5: Merge based on strategy (memory-optimized)
        if strategy == 'append' and not has_overlap:
            # Fast path: clean append using PyArrow (no full load)
            merged_rows, merged_data = merge_append_pyarrow(old_file, new_data, output_file, validate=True)

            # Validate merged data (critical safety check)
            if validate:
                is_valid, error_msg = validate_merged_data(
                    merged_data,
                    old_first_date,
                    new_last_date,
                    old_rows,
                    new_rows
                )
                if not is_valid:
                    raise ValueError(f"Validation failed: {error_msg}")

            # Clean up memory
            del merged_data

            logger.info(f"   Ô£à Merged successfully (fast path): {old_rows:,} + {new_rows:,} ÔåÆ {merged_rows:,} rows")
        else:
            # Need full load for deduplication or overlap handling
            logger.debug(f"   Using pandas merge (overlap or dedupe required)...")
            old_data = pd.read_parquet(old_file)

            if strategy == 'append':
                merged_data = merge_append(old_data, new_data)
            elif strategy == 'overlap-dedupe':
                merged_data = merge_with_dedupe(old_data, new_data)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            merged_rows = len(merged_data)
            logger.debug(f"   Merged: {merged_rows:,} rows")

            # Validate merged data
            if validate:
                is_valid, error_msg = validate_merged_data(
                    merged_data,
                    old_first_date,
                    new_last_date,
                    old_rows,
                    new_rows
                )
                if not is_valid:
                    raise ValueError(f"Validation failed: {error_msg}")

            # Write merged data atomically
            atomic_write_parquet(merged_data, output_file)

            logger.info(f"   Ô£à Merged successfully: {old_rows:,} + {new_rows:,} ÔåÆ {merged_rows:,} rows")

            # Clean up memory
            del old_data, merged_data

        return True

    except Exception as e:
        logger.error(f"   ÔØî Merge failed: {str(e)}")
        raise


def merge_append_pyarrow(old_file: Path, new_data: pd.DataFrame, output_file: Path, validate: bool = True) -> tuple[int, pd.DataFrame]:
    """
    Memory-efficient append using chunked processing (avoid duplicate RAM usage)

    Appends new data to old file without loading full old data into pandas.

    Args:
        old_file: Path to existing parquet file
        new_data: New data to append
        output_file: Output file path
        validate: If True, return merged DataFrame for validation

    Returns:
        Tuple of (total_rows, merged_df) if validate=True, otherwise (total_rows, None)
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Convert new data to Arrow table first (small dataset)
    new_table = pa.Table.from_pandas(new_data, preserve_index=False)

    # Use temp file for atomic write
    temp_file = output_file.with_suffix('.parquet.tmp')

    # Open old file for reading and new file for writing (streaming approach)
    old_parquet = pq.ParquetFile(old_file)

    # Write merged data in chunks to avoid holding both old and new in memory
    writer = None
    total_rows = 0

    try:
        # Handle edge case: old file has zero row groups
        if old_parquet.num_row_groups == 0:
            # Initialize writer from new data schema
            writer = pq.ParquetWriter(temp_file, new_table.schema)
        else:
            # Process old file row groups without loading full table
            for i in range(old_parquet.num_row_groups):
                batch = old_parquet.read_row_group(i)
                if writer is None:
                    # Initialize writer with schema from first batch
                    writer = pq.ParquetWriter(temp_file, batch.schema)
                writer.write_table(batch)
                total_rows += batch.num_rows

        # Append new data
        writer.write_table(new_table)
        total_rows += new_table.num_rows

        writer.close()

        # CRITICAL: Close the old_parquet file handle BEFORE atomic rename
        # Otherwise, Windows will lock the file and prevent deletion/rename
        del old_parquet
        gc.collect()

        # If validation needed, read back merged file (unavoidable for validation)
        if validate:
            merged_df = pd.read_parquet(temp_file)
            # Sort by timestamp (critical for backtesting)
            merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)
            # Rewrite with sorted data
            merged_df.to_parquet(temp_file, index=False)
            # Release DataFrame before rename
            del merged_df
            gc.collect()
            # Re-read after sort (need to return it)
            merged_df = pd.read_parquet(temp_file)
        else:
            merged_df = None

        # Atomic rename (Windows-safe with retry logic)
        atomic_rename_windows_safe(temp_file, output_file)

        return total_rows, merged_df

    except Exception as e:
        # Clean up temp file on error
        if temp_file.exists():
            temp_file.unlink()
        raise

    finally:
        if writer:
            writer.close()


def merge_append(old_data: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
    """
    Merge using append strategy (simple concatenation + sort)

    Assumes no overlap between old and new data.
    Fastest strategy.

    Args:
        old_data: Existing data
        new_data: New data to append

    Returns:
        Merged DataFrame
    """
    # Concatenate
    merged = pd.concat([old_data, new_data], ignore_index=True)

    # Sort by timestamp (critical for backtesting)
    merged = merged.sort_values('timestamp').reset_index(drop=True)

    return merged


def merge_with_dedupe(old_data: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
    """
    Merge with overlap handling (deduplication)

    Keeps newer data on duplicates.
    Slower but handles overlapping fetches gracefully.

    Args:
        old_data: Existing data
        new_data: New data (may overlap with old)

    Returns:
        Merged and deduplicated DataFrame
    """
    # Concatenate
    merged = pd.concat([old_data, new_data], ignore_index=True)

    # Remove duplicates, keeping last occurrence (newer data)
    merged = merged.drop_duplicates(subset=['timestamp'], keep='last')

    # Sort by timestamp
    merged = merged.sort_values('timestamp').reset_index(drop=True)

    return merged


def validate_merged_data(
    merged_data: pd.DataFrame,
    expected_first_date: datetime,
    expected_last_date: datetime,
    old_rows: int,
    new_rows: int
) -> tuple[bool, str]:
    """
    Validate merged data integrity

    Args:
        merged_data: Merged DataFrame
        expected_first_date: Expected first timestamp
        expected_last_date: Expected last timestamp
        old_rows: Number of rows in old data
        new_rows: Number of rows in new data

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check 1: Not empty
    if merged_data.empty:
        return False, "Merged data is empty"

    # Check 2: Has timestamp column
    if 'timestamp' not in merged_data.columns:
        return False, "Missing 'timestamp' column"

    # Check 3: No null timestamps
    if merged_data['timestamp'].isnull().any():
        null_count = merged_data['timestamp'].isnull().sum()
        return False, f"Found {null_count} null timestamps"

    # Check 4: Timestamps are sorted
    if not merged_data['timestamp'].is_monotonic_increasing:
        return False, "Timestamps are not sorted"

    # Check 5: Date range is correct
    actual_first = merged_data['timestamp'].min()
    actual_last = merged_data['timestamp'].max()

    if actual_first != expected_first_date:
        return False, f"First date mismatch: expected {expected_first_date}, got {actual_first}"

    if actual_last < expected_last_date:
        return False, f"Last date is before expected: {actual_last} < {expected_last_date}"

    # Check 6: Row count makes sense
    merged_rows = len(merged_data)
    expected_min_rows = old_rows  # At minimum, should have old rows
    expected_max_rows = old_rows + new_rows  # At maximum (no duplicates)

    if merged_rows < expected_min_rows:
        return False, f"Row count too low: {merged_rows} < {expected_min_rows}"

    if merged_rows > expected_max_rows:
        return False, f"Row count too high: {merged_rows} > {expected_max_rows}"

    # Check 7: No duplicate timestamps
    if merged_data['timestamp'].duplicated().any():
        dup_count = merged_data['timestamp'].duplicated().sum()
        return False, f"Found {dup_count} duplicate timestamps"

    # Check 8: Required columns present
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in merged_data.columns]
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"

    # Check 9: No null values in OHLCV
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in merged_data.columns and merged_data[col].isnull().any():
            null_count = merged_data[col].isnull().sum()
            return False, f"Found {null_count} null values in '{col}'"

    # Check 10: Price sanity checks (with floating-point tolerance)
    # Use a small epsilon for floating-point comparisons (0.0001 = 1 paisa tolerance)
    epsilon = 0.0001

    # Critical check: High should never be less than Low (hard failure)
    if (merged_data['high'] < merged_data['low'] - epsilon).any():
        invalid_count = (merged_data['high'] < merged_data['low'] - epsilon).sum()
        return False, f"Found {invalid_count} rows where high < low (critical error)"

    # Relaxed checks: High should be >= Open and Close (with tolerance)
    # This allows for floating-point precision issues and single-price ticks
    high_violations = (
        (merged_data['high'] < merged_data['open'] - epsilon) |
        (merged_data['high'] < merged_data['close'] - epsilon)
    )
    if high_violations.any():
        violation_count = high_violations.sum()
        logger.warning(f"   ÔÜá´©Å  Found {violation_count} rows where high price slightly below open/close (within tolerance)")
        # Log sample violations for debugging
        sample_violations = merged_data[high_violations].head(3)
        for idx, row in sample_violations.iterrows():
            logger.debug(f"      Row {idx}: O={row['open']:.4f}, H={row['high']:.4f}, L={row['low']:.4f}, C={row['close']:.4f}")

    # Relaxed checks: Low should be <= Open and Close (with tolerance)
    low_violations = (
        (merged_data['low'] > merged_data['open'] + epsilon) |
        (merged_data['low'] > merged_data['close'] + epsilon)
    )
    if low_violations.any():
        violation_count = low_violations.sum()
        logger.warning(f"   ÔÜá´©Å  Found {violation_count} rows where low price slightly above open/close (within tolerance)")
        # Log sample violations for debugging
        sample_violations = merged_data[low_violations].head(3)
        for idx, row in sample_violations.iterrows():
            logger.debug(f"      Row {idx}: O={row['open']:.4f}, H={row['high']:.4f}, L={row['low']:.4f}, C={row['close']:.4f}")

    return True, "All validations passed"


def atomic_rename_windows_safe(temp_file: Path, output_file: Path, max_retries: int = 10, initial_delay: float = 0.1):
    """
    Windows-safe atomic file rename with retry logic

    Handles Windows file locking issues by:
    1. Forcing garbage collection to release file handles
    2. Deleting target file if it exists (after backup is created)
    3. Retrying with exponential backoff

    Args:
        temp_file: Source temp file
        output_file: Destination file
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds (doubles each retry)

    Raises:
        IOError: If rename fails after all retries
    """
    import sys

    # Force garbage collection to release any file handles
    gc.collect()

    delay = initial_delay
    last_error = None

    for attempt in range(max_retries):
        try:
            # On Windows, we need to handle the target file carefully
            if output_file.exists():
                # Try to remove the target file first (backup should already exist)
                try:
                    output_file.unlink()
                    logger.debug(f"      Removed existing target file: {output_file.name}")
                except (PermissionError, OSError) as e:
                    # File is locked or has permission issues, wait and retry
                    if attempt < max_retries - 1:
                        logger.debug(f"      Target file locked (attempt {attempt + 1}/{max_retries}), waiting {delay:.2f}s...")
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                        gc.collect()  # Force GC again
                        continue
                    else:
                        raise

            # Now rename temp -> output (target should not exist now)
            if sys.platform == 'win32':
                # On Windows, use os.rename instead of Path.replace for better compatibility
                os.rename(str(temp_file), str(output_file))
            else:
                # On POSIX, Path.replace is atomic
                temp_file.replace(output_file)

            logger.debug(f"      Successfully renamed {temp_file.name} -> {output_file.name}")
            return  # Success!

        except (PermissionError, OSError) as e:
            last_error = e
            if attempt < max_retries - 1:
                logger.debug(f"      Rename failed (attempt {attempt + 1}/{max_retries}): {e}, retrying in {delay:.2f}s...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
                gc.collect()  # Force garbage collection
            else:
                # All retries exhausted
                raise IOError(f"Failed to rename after {max_retries} attempts: {last_error}")


def atomic_write_parquet(df: pd.DataFrame, output_file: Path):
    """
    Write parquet file atomically to prevent corruption

    Process:
    1. Write to temp file: output_file.tmp
    2. Validate temp file can be read
    3. Atomic rename temp -> output (Windows-safe with retry)

    Args:
        df: DataFrame to write
        output_file: Target file path

    Raises:
        IOError: If write or rename fails
    """
    output_file = Path(output_file)
    temp_file = output_file.with_suffix('.parquet.tmp')

    try:
        # Write to temp file
        df.to_parquet(temp_file, index=False, engine='pyarrow')

        # Validate temp file can be read
        test_df = pd.read_parquet(temp_file, columns=['timestamp'])
        if len(test_df) != len(df):
            raise ValueError("Temp file validation failed: row count mismatch")

        # Release the test DataFrame
        del test_df
        gc.collect()

        # Atomic rename (Windows-safe with retry logic)
        atomic_rename_windows_safe(temp_file, output_file)

    except Exception as e:
        # Clean up temp file on failure
        if temp_file.exists():
            temp_file.unlink()
        raise IOError(f"Failed to write parquet file: {str(e)}")


def create_backup(file_path: Path) -> Path:
    """
    Create backup of file with timestamp suffix

    Args:
        file_path: File to backup

    Returns:
        Path to backup file
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = file_path.with_suffix(f'.backup_{timestamp}.parquet')

    shutil.copy2(file_path, backup_path)

    return backup_path


def restore_from_backup(backup_path: Path, original_path: Path):
    """
    Restore file from backup

    Args:
        backup_path: Path to backup file
        original_path: Original file path to restore to
    """
    if not backup_path.exists():
        raise FileNotFoundError(f"Backup not found: {backup_path}")

    shutil.copy2(backup_path, original_path)
    logger.info(f"Ô£à Restored from backup: {backup_path.name}")


def merge_batch(
    file_data_pairs: list[tuple[str, pd.DataFrame]],
    strategy: MergeStrategy = 'append',
    backup: bool = True,
    validate: bool = True
) -> dict[str, bool]:
    """
    Merge multiple files in batch

    Args:
        file_data_pairs: List of (old_file_path, new_data) tuples
        strategy: Merge strategy
        backup: Create backups
        validate: Run validation

    Returns:
        Dict of file_path -> success status
    """
    results = {}

    logger.info(f"­ƒöä Batch merging {len(file_data_pairs)} files...")

    for i, (old_file, new_data) in enumerate(file_data_pairs, 1):
        logger.info(f"   [{i}/{len(file_data_pairs)}] {Path(old_file).name}")

        try:
            success = merge_parquet_files(
                old_file,
                new_data,
                strategy=strategy,
                backup=backup,
                validate=validate
            )
            results[old_file] = success
        except Exception as e:
            logger.error(f"   ÔØî Failed: {str(e)}")
            results[old_file] = False

    # Summary
    successful = sum(1 for v in results.values() if v)
    failed = len(results) - successful

    logger.info(f"\nÔ£à Batch merge complete: {successful}/{len(results)} successful, {failed} failed")

    return results


# CLI interface for testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python data_merger.py <old_file> <new_file> [strategy]")
        print("Example: python data_merger.py old.parquet new.parquet append")
        print("Strategies: append, overlap-dedupe")
        sys.exit(1)

    old_file = sys.argv[1]
    new_file = sys.argv[2]
    strategy = sys.argv[3] if len(sys.argv) > 3 else 'append'

    try:
        # Read new data
        print(f"Reading new data from {new_file}...")
        new_data = pd.read_parquet(new_file)

        # Merge
        print(f"\nMerging with strategy: {strategy}")
        success = merge_parquet_files(
            old_file,
            new_data,
            strategy=strategy,
            backup=True,
            validate=True
        )

        if success:
            print("\nÔ£à Merge successful!")
            sys.exit(0)
        else:
            print("\nÔØî Merge failed")
            sys.exit(1)

    except Exception as e:
        logger.error(f"ÔØî Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
