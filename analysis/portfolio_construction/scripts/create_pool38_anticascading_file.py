"""
Create Anti-Cascading Trades File for Pool38
=============================================
Quick script to filter merged trades to only 38 tickers and save as anti-cascading file.
This bypasses the 3-hour ticker_ranking module.

Usage:
    py analysis/portfolio_construction/scripts/create_pool38_anticascading_file.py
"""

import pandas as pd
from pathlib import Path

# Define the 38 tickers
POOL38_TICKERS = [
    'AXISBANK', 'MAXHEALTH', 'INDHOTEL', 'POCL', 'NMDC', 'PNB', 'UBL', 'HERCULES',
    'ITC', 'PSUBANK', 'NTPC', 'PIDILITIND', 'DELHIVERY', 'JINDALSTEL', 'FEDERALBNK',
    'SUNPHARMA', 'IPCALAB', 'ABFRL', 'INFY', 'POWERGRID', 'IRFC', 'DABUR', 'PETRONET',
    'VIMTALABS', 'TATASTEEL', 'MSUMI', 'RAMCOCEM', 'GRAPHITE', 'TATAMOTORS', 'PFC',
    'SBICARD', 'NESTLEIND', 'WIPRO', 'VGUARD', 'TATACHEM', 'TECHM', 'SHYAMMETL', 'CIPLA'
]

def main():
    print("=" * 80)
    print("CREATE ANTI-CASCADING TRADES FILE FOR POOL38")
    print("=" * 80)
    print()

    # Paths
    merged_file = Path("analysis/output/mse_strategy_backtesting/pool38_base/data/pool38_trades_merged.csv")
    output_dir = Path("analysis/output/mse_strategy_backtesting/pool38_base/portfolio/anti_cascade_filter")
    output_file = output_dir / "anti_cascading_trades_filtered.csv"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Input file: {merged_file}")
    print(f"[INFO] Output file: {output_file}")
    print(f"[INFO] Filtering to {len(POOL38_TICKERS)} tickers")
    print()

    # Load merged trades
    print("[STEP 1] Loading merged trades...")
    trades_df = pd.read_csv(merged_file)
    print(f"[OK] Loaded {len(trades_df):,} trades")
    print()

    # Filter to 38 tickers
    print("[STEP 2] Filtering to Pool38 tickers...")
    filtered_df = trades_df[trades_df['ticker'].isin(POOL38_TICKERS)].copy()
    print(f"[OK] Filtered to {len(filtered_df):,} trades")
    print()

    # Show ticker breakdown
    print("[STEP 3] Ticker breakdown:")
    ticker_counts = filtered_df['ticker'].value_counts().sort_index()
    for ticker in POOL38_TICKERS:
        count = ticker_counts.get(ticker, 0)
        print(f"  {ticker:12s}: {count:>6,} trades")
    print()

    # Save
    print("[STEP 4] Saving anti-cascading trades file...")
    filtered_df.to_csv(output_file, index=False)
    print(f"[OK] Saved to: {output_file}")
    print()

    # Also create metadata file (affordable tickers)
    print("[STEP 5] Creating metadata file...")
    metadata_file = output_dir / "affordable_tickers_metadata.csv"

    # Get last price for each ticker (from last exit price)
    filtered_df['Exit Time'] = pd.to_datetime(filtered_df['Exit Time'])
    last_prices = filtered_df.sort_values('Exit Time').groupby('ticker').last()[['Exit Price']].reset_index()
    last_prices.columns = ['ticker', 'current_price']

    last_prices.to_csv(metadata_file, index=False)
    print(f"[OK] Saved metadata to: {metadata_file}")
    print()

    print("=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"Anti-cascading file ready: {output_file}")
    print(f"You can now run portfolio construction with ticker_ranking disabled")
    print()


if __name__ == "__main__":
    main()
