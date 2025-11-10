import pandas as pd
import os
from datetime import datetime

# Paths
candles_dir = r'd:\Balcony\Trading\backtester\candles'
hist_data_dir = r'd:\Balcony\Trading\backtester\hist_data'

# Tickers and intervals
tickers = ['BIOCON', 'JSWSTEEL', 'KOTAKBANK']
intervals = ['5min', '15min']
interval_files = {'5min': '5m.csv', '15min': '15m.csv'}

results = []

for ticker in tickers:
    for interval in intervals:
        print(f"\nProcessing {ticker} {interval}...")

        # Load historical data
        hist_path = os.path.join(hist_data_dir, ticker, interval_files[interval])
        if not os.path.exists(hist_path):
            print(f"Historical file not found: {hist_path}")
            continue
        hist_df = pd.read_csv(hist_path)
        hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'])
        hist_df.set_index('timestamp', inplace=True)

        # Load and merge candles data
        pattern = f"{ticker}_{interval}_"
        candles_files = [f for f in os.listdir(candles_dir) if f.startswith(pattern) and f.endswith('.csv')]
        if not candles_files:
            print(f"No candles files found for {ticker} {interval}")
            continue
        candles_df_list = []
        for file in candles_files:
            df = pd.read_csv(os.path.join(candles_dir, file))
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            candles_df_list.append(df)
        candles_df = pd.concat(candles_df_list)
        candles_df = candles_df[~candles_df.index.duplicated(keep='first')]

        # Merge and compare
        comparison_df = hist_df.join(candles_df, how='inner', lsuffix='_hist', rsuffix='_live')

        if comparison_df.empty:
            print(f"No overlapping data for {ticker} {interval}")
            continue

        # Calculate differences
        cols = ['open', 'high', 'low', 'close', 'volume']
        for col in cols:
            comparison_df[f'{col}_diff'] = comparison_df[f'{col}_hist'] - comparison_df[f'{col}_live']
            comparison_df[f'{col}_pct_diff'] = (comparison_df[f'{col}_diff'] / comparison_df[f'{col}_hist']) * 100

        # Stats
        total_rows = len(comparison_df)
        ohlc_accuracy = (comparison_df[['open_pct_diff', 'high_pct_diff', 'low_pct_diff', 'close_pct_diff']].abs() < 1).all(axis=1).sum() / total_rows * 100
        volume_mean_pct = comparison_df['volume_pct_diff'].mean()
        volume_max_pct = comparison_df['volume_pct_diff'].abs().max()

        results.append({
            'Ticker': ticker,
            'Interval': interval,
            'Overlaps': total_rows,
            'OHLC_Accuracy_%': round(ohlc_accuracy, 2),
            'Volume_Mean_Pct_%': round(volume_mean_pct, 2),
            'Volume_Max_Pct_%': round(volume_max_pct, 2)
        })

        print(f"Overlaps: {total_rows}, OHLC Accuracy: {ohlc_accuracy:.2f}%, Volume Mean: {volume_mean_pct:.2f}%, Max: {volume_max_pct:.2f}%")

# Summary DataFrame
results_df = pd.DataFrame(results)
print("\n" + "="*50)
print("SUMMARY ACROSS ALL TICKERS AND INTERVALS:")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('multi_ticker_analysis_summary.csv', index=False)
print("Summary saved to multi_ticker_analysis_summary.csv")