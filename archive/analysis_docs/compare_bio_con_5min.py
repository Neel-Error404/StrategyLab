import pandas as pd
import os
from datetime import datetime

# Paths
candles_dir = r'd:\Balcony\Trading\backtester\candles'
hist_data_dir = r'd:\Balcony\Trading\backtester\hist_data\BIOCON\5m.csv'

# Load historical data
hist_df = pd.read_csv(hist_data_dir)
hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'])
hist_df.set_index('timestamp', inplace=True)

# Load and merge candles data for BIOCON 5min
candles_files = [f for f in os.listdir(candles_dir) if f.startswith('BIOCON_5min_') and f.endswith('.csv')]
candles_df_list = []

for file in candles_files:
    df = pd.read_csv(os.path.join(candles_dir, file))
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    candles_df_list.append(df)

if candles_df_list:
    candles_df = pd.concat(candles_df_list)
    # Remove duplicates if any
    candles_df = candles_df[~candles_df.index.duplicated(keep='first')]
else:
    print("No candles files found for BIOCON 5min")
    exit()

# Compare: Join on timestamp
comparison_df = hist_df.join(candles_df, how='inner', lsuffix='_hist', rsuffix='_candles')

# Calculate differences
for col in ['open', 'high', 'low', 'close', 'volume']:
    comparison_df[f'{col}_diff'] = comparison_df[f'{col}_hist'] - comparison_df[f'{col}_candles']
    comparison_df[f'{col}_pct_diff'] = (comparison_df[f'{col}_diff'] / comparison_df[f'{col}_hist']) * 100

# Summary
total_rows = len(comparison_df)
matching_rows = (comparison_df[['open_diff', 'high_diff', 'low_diff', 'close_diff', 'volume_diff']] == 0).all(axis=1).sum()
mismatches = total_rows - matching_rows

print(f"Total overlapping timestamps: {total_rows}")
print(f"Perfect matches: {matching_rows}")
print(f"Mismatches: {mismatches}")

if mismatches > 0:
    print("\nSample mismatches:")
    print(comparison_df[comparison_df[['open_diff', 'high_diff', 'low_diff', 'close_diff', 'volume_diff']].ne(0).any(axis=1)].head())

# Save merged candles data
candles_df.to_csv(r'd:\Balcony\Trading\backtester\merged_bio_con_5min.csv')
print("Merged candles data saved to merged_bio_con_5min.csv")