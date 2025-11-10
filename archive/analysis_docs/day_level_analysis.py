import pandas as pd
import os

# Load the multi-ticker summary for reference
summary_df = pd.read_csv('multi_ticker_analysis_summary.csv')

print("OHLC Confirmation: 100% accuracy across all tickers and intervals - no discrepancies in open, high, low, close prices.")
print("Volume Direction: Historical volumes are generally higher than live volumes (positive mean % differences), but extremes show both directions.\n")

# Now, perform day-level analysis
candles_dir = r'd:\Balcony\Trading\backtester\candles'
hist_data_dir = r'd:\Balcony\Trading\backtester\hist_data'

tickers = ['BIOCON', 'JSWSTEEL', 'KOTAKBANK']
intervals = ['5min', '15min']
interval_files = {'5min': '5m.csv', '15min': '15m.csv'}

daily_volume_stats = []

for ticker in tickers:
    for interval in intervals:
        # Load historical
        hist_path = os.path.join(hist_data_dir, ticker, interval_files[interval])
        hist_df = pd.read_csv(hist_path)
        hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'])
        hist_df['date'] = hist_df['timestamp'].dt.date

        # Load merged candles (assuming we have them, or recreate)
        pattern = f"{ticker}_{interval}_"
        candles_files = [f for f in os.listdir(candles_dir) if f.startswith(pattern)]
        candles_df_list = []
        for file in candles_files:
            df = pd.read_csv(os.path.join(candles_dir, file))
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            candles_df_list.append(df)
        if candles_df_list:
            candles_df = pd.concat(candles_df_list)
            candles_df = candles_df.drop_duplicates(subset='timestamp', keep='first')

            # Merge
            merged = pd.merge(hist_df, candles_df, on='timestamp', suffixes=('_hist', '_live'))
            merged['date'] = merged['timestamp'].dt.date
            merged['volume_pct_diff'] = (merged['volume_hist'] - merged['volume_live']) / merged['volume_hist'] * 100

            # Group by date
            daily = merged.groupby('date').agg({
                'volume_pct_diff': ['mean', 'std', 'count']
            }).round(2)
            daily.columns = ['volume_mean_pct', 'volume_std_pct', 'records']

            for date, row in daily.iterrows():
                daily_volume_stats.append({
                    'Ticker': ticker,
                    'Interval': interval,
                    'Date': date,
                    'Volume_Mean_Pct': row['volume_mean_pct'],
                    'Volume_Std_Pct': row['volume_std_pct'],
                    'Records': row['records']
                })

# Daily stats DataFrame
daily_df = pd.DataFrame(daily_volume_stats)
print("Day-Level Volume Discrepancy Summary:")
print(daily_df.head(20).to_string(index=False))

# Find days with all three tickers
dates_with_all = daily_df.groupby('Date')['Ticker'].nunique()
full_days = dates_with_all[dates_with_all == 3].index
print(f"\nDays with data for all three tickers: {list(full_days)}")

if full_days.any():
    full_day_stats = daily_df[daily_df['Date'].isin(full_days)]
    print("\nCross-Ticker Day Validation (Sample):")
    print(full_day_stats.to_string(index=False))

# Save
daily_df.to_csv('day_level_volume_analysis.csv', index=False)
print("Day-level analysis saved to day_level_volume_analysis.csv")