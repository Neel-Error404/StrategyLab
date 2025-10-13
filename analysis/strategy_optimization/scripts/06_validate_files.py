"""
Quick validation script to check if trade files are ready for Stage 6 comparison.

Usage:
    python scripts/06_validate_files.py \
        --baseline "outputs/20251005_121223/mse_backtesting/2022-01-01_to_2025-08-31/all_trade_merged.csv" \
        --optimal "outputs/20251005_124708/mse_backtesting/2022-01-01_to_2025-08-31/all_trade_merged.csv"
"""

import sys
from pathlib import Path
import pandas as pd
import argparse

# Valid tickers used in optimization
VALID_TICKERS = [
    "RELIANCE", "TCS", "INFY", "HINDUNILVR", "ITC", "SBIN", "KOTAKBANK", "LT",
    "ASIANPAINT", "AXISBANK", "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO",
    "WIPRO", "NESTLEIND", "HCLTECH", "POWERGRID", "NTPC", "ONGC",
    "TATASTEEL", "JSWSTEEL", "ADANIPORTS", "TECHM"
]


def validate_file(file_path: str, label: str) -> bool:
    """Validate a single trade file."""

    print(f"\n{'='*60}")
    print(f"Validating {label}")
    print(f"{'='*60}")

    path = Path(file_path)

    # Check exists
    if not path.exists():
        print(f"❌ File not found: {path}")
        return False
    print(f"✅ File exists: {path.name}")

    # Check size
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"   Size: {size_mb:.2f} MB")

    # Load file
    try:
        df = pd.read_csv(file_path)
        print(f"✅ File loaded: {len(df):,} total trades")
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        return False

    # Check columns
    required_cols = ['Entry Time', 'Exit Time', 'ticker', 'percentage_return']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"❌ Missing columns: {missing}")
        return False
    print(f"✅ Required columns present")

    # Convert dates
    try:
        df['Entry Time'] = pd.to_datetime(df['Entry Time'])
        df['Exit Time'] = pd.to_datetime(df['Exit Time'])
    except Exception as e:
        print(f"❌ Date conversion failed: {e}")
        return False

    # Check date range
    min_date = df['Entry Time'].min()
    max_date = df['Entry Time'].max()
    print(f"   Date range: {min_date.date()} to {max_date.date()}")

    # Filter to valid tickers
    df_valid = df[df['ticker'].isin(VALID_TICKERS)]
    print(f"✅ Valid tickers (24): {len(df_valid):,} trades")

    # Check test period
    test_df = df_valid[
        (df_valid['Entry Time'] >= '2024-07-01') &
        (df_valid['Entry Time'] <= '2025-08-31')
    ]
    print(f"✅ Test period (2024-07-01 to 2025-08-31): {len(test_df):,} trades")

    if len(test_df) == 0:
        print(f"❌ No trades in test period!")
        return False

    # Show ticker distribution
    ticker_counts = test_df['ticker'].value_counts()
    print(f"   Tickers in test period: {len(ticker_counts)} / 24")
    print(f"   Top 5: {ticker_counts.head().to_dict()}")

    return True


def main():
    parser = argparse.ArgumentParser(description='Validate trade files for Stage 6')
    parser.add_argument('--baseline', required=True, help='Path to baseline (0.80) trades')
    parser.add_argument('--optimal', required=True, help='Path to optimal (0.95) trades')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("STAGE 6 FILE VALIDATION")
    print("="*60)

    # Validate baseline
    baseline_ok = validate_file(args.baseline, "BASELINE (0.80)")

    # Validate optimal
    optimal_ok = validate_file(args.optimal, "OPTIMAL (0.95)")

    # Final summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")

    if baseline_ok and optimal_ok:
        print("\n✅ ✅ ✅ VALIDATION PASSED ✅ ✅ ✅")
        print("\nBoth files are ready for comparison.")
        print("\nRun Stage 6 comparison:")
        print(f"\n   python scripts/06_final_test_comparison.py \\")
        print(f"     --baseline \"{args.baseline}\" \\")
        print(f"     --optimal \"{args.optimal}\"")
        print()
        return 0
    else:
        print("\n❌ ❌ ❌ VALIDATION FAILED ❌ ❌ ❌")
        print("\nIssues found:")
        if not baseline_ok:
            print("  - Baseline file has issues")
        if not optimal_ok:
            print("  - Optimal file has issues")
        print("\nFix issues before running comparison.")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
