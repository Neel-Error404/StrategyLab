import pandas as pd

df = pd.read_csv('merged_bio_con_5min.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'] = df['timestamp'].dt.date
days = df['date'].nunique()
print(f'Number of days in merged BIOCON 5min data: {days}')
print(f'Total rows: {len(df)}')
print(f'Unique dates: {sorted(df["date"].unique())}')
diffs = df['timestamp'].diff().dt.total_seconds() / 60
print(f'Most common interval: {diffs.mode()[0]} minutes')