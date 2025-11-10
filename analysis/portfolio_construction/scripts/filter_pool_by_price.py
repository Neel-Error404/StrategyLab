"""
Filter Ticker Pool by Price Threshold
======================================

Filters the selected 60-ticker pool to only include tickers ≤ specified price.

Rational: Minimum capital per ticker = ₹1,000
Therefore: Max price = ₹2,000 (allows minimum 0.5 shares)

Author: MSE Strategy Research Team
Date: 2025-11-08
"""

import pandas as pd
from pathlib import Path
import sys

def main():
    print("=" * 80)
    print("FILTER TICKER POOL BY PRICE")
    print("=" * 80)
    print()

    # Configuration
    price_threshold = 2000
    run_id = "20251107_154802_full"
    strategy = "mse_strategy_backtesting"

    # Paths
    input_file = Path(f"analysis/output/{strategy}/{run_id}/ticker_pool_selection/selected_ticker_pool_60.csv")
    output_dir = Path(f"analysis/output/{strategy}/{run_id}/ticker_pool_selection")

    print(f"[INFO] Price threshold: ₹{price_threshold:,.0f}")
    print(f"[INFO] Input file: {input_file}")
    print()

    # Load pool
    print("[STEP 1] Loading 60-ticker pool...")
    pool_df = pd.read_csv(input_file)
    print(f"[OK] Loaded {len(pool_df)} tickers")
    print()

    # Filter by price
    print(f"[STEP 2] Filtering to tickers ≤₹{price_threshold:,.0f}...")
    filtered_df = pool_df[pool_df['current_price'] <= price_threshold].copy()

    removed_count = len(pool_df) - len(filtered_df)

    print(f"[OK] Filtered pool: {len(filtered_df)} tickers")
    print(f"[INFO] Removed: {removed_count} tickers (price >₹{price_threshold:,.0f})")
    print()

    # Show removed tickers
    if removed_count > 0:
        removed_df = pool_df[pool_df['current_price'] > price_threshold]
        print(f"[INFO] Removed tickers (>₹{price_threshold:,.0f}):")
        for _, row in removed_df.head(10).iterrows():
            print(f"       - {row['ticker']:12s} ₹{row['current_price']:>10,.2f}  (Rank #{int(row['selection_rank'])}, Sharpe {row['sharpe_like_ratio']:.4f})")
        if removed_count > 10:
            print(f"       ... and {removed_count - 10} more")
        print()

    # Show statistics
    print("[STEP 3] Filtered pool statistics...")
    print(f"  Total tickers: {len(filtered_df)}")
    print(f"  Avg price: ₹{filtered_df['current_price'].mean():,.2f}")
    print(f"  Median price: ₹{filtered_df['current_price'].median():,.2f}")
    print(f"  Price range: ₹{filtered_df['current_price'].min():,.2f} - ₹{filtered_df['current_price'].max():,.2f}")
    print()
    print(f"  Avg Win Rate: {filtered_df['win_rate'].mean():.2f}%")
    print(f"  Avg Profit Factor: {filtered_df['profit_factor'].mean():.4f}")
    print(f"  Avg Sharpe: {filtered_df['sharpe_like_ratio'].mean():.4f}")
    print(f"  Avg Composite Score: {filtered_df['composite_score'].mean():.4f}")
    print()

    # Save filtered pool
    output_file = output_dir / f"selected_ticker_pool_{len(filtered_df)}_under{price_threshold}.csv"
    filtered_df.to_csv(output_file, index=False)

    print(f"[OK] Saved filtered pool: {output_file}")
    print()

    # Create ticker list for config
    ticker_list = filtered_df['ticker'].tolist()
    ticker_list_file = output_dir / f"ticker_list_{len(filtered_df)}.txt"

    with open(ticker_list_file, 'w') as f:
        f.write('\n'.join(ticker_list))

    print(f"[OK] Saved ticker list: {ticker_list_file}")
    print()

    print("=" * 80)
    print("FILTER COMPLETE")
    print("=" * 80)
    print()
    print(f"Filtered pool: {len(filtered_df)} tickers ≤₹{price_threshold:,.0f}")
    print(f"Ready for portfolio construction")
    print()


if __name__ == "__main__":
    main()
