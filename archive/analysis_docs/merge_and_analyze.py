import pandas as pd

# Load merged candles data (live data)
live_df = pd.read_csv('merged_bio_con_5min.csv')
live_df['timestamp'] = pd.to_datetime(live_df['timestamp'])
live_df.set_index('timestamp', inplace=True)

# Load historical data
hist_df = pd.read_csv(r'hist_data\BIOCON\5m.csv')
hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'])
hist_df.set_index('timestamp', inplace=True)

# Merge on timestamp (inner join for overlapping)
merged_df = hist_df.join(live_df, how='inner', lsuffix='_hist', rsuffix='_live')

# Calculate differences and percentages
cols = ['open', 'high', 'low', 'close', 'volume']
for col in cols:
    merged_df[f'{col}_diff'] = merged_df[f'{col}_hist'] - merged_df[f'{col}_live']
    merged_df[f'{col}_pct_diff'] = (merged_df[f'{col}_diff'] / merged_df[f'{col}_hist']) * 100

# Summary statistics
print("Disparity Analysis (Historical vs Live):")
print(f"Total overlapping records: {len(merged_df)}")

for col in cols:
    mean_diff = merged_df[f'{col}_diff'].mean()
    median_diff = merged_df[f'{col}_diff'].median()
    mean_pct = merged_df[f'{col}_pct_diff'].mean()
    median_pct = merged_df[f'{col}_pct_diff'].median()
    max_pct = merged_df[f'{col}_pct_diff'].abs().max()
    print(f"\n{col.upper()}:")
    print(f"  Mean diff: {mean_diff:.4f}, Median diff: {median_diff:.4f}")
    print(f"  Mean % diff: {mean_pct:.2f}%, Median % diff: {median_pct:.2f}%, Max % diff: {max_pct:.2f}%")

# Accuracy: Percentage of records with <1% difference in OHLC
ohlc_cols = ['open_pct_diff', 'high_pct_diff', 'low_pct_diff', 'close_pct_diff']
accurate_ohlc = (merged_df[ohlc_cols].abs() < 1).all(axis=1).sum()
accuracy_rate = (accurate_ohlc / len(merged_df)) * 100
print(f"\nOHLC Accuracy (<1% diff): {accuracy_rate:.2f}%")

# Save detailed comparison
merged_df.to_csv('bio_con_5min_comparison.csv')
print("Detailed comparison saved to bio_con_5min_comparison.csv")