#!/usr/bin/env python3
"""
Quick comparison script: OLD Top 50 (absolute) vs NEW Top 50 (percentage)
"""

import pandas as pd

# Load OLD and NEW Top 50
old_df = pd.read_csv('data/TOP50_ANTICASCADING_TRADES_v1_absolute.csv')
new_df = pd.read_csv('data/TOP50_ANTICASCADING_TRADES.csv')

print("=" * 80)
print("🔍 COMPARISON: OLD Top 50 (Absolute Currency) vs NEW Top 50 (Percentage Returns)")
print("=" * 80)

# Extract ticker lists
old_tickers = set(old_df['ticker'].tolist())
new_tickers = set(new_df['ticker'].tolist())

# Calculate overlap
overlap = old_tickers & new_tickers
only_in_old = old_tickers - new_tickers
only_in_new = new_tickers - old_tickers

print(f"\n📊 OVERLAP ANALYSIS:")
print(f"   Tickers in BOTH lists: {len(overlap)}")
print(f"   Only in OLD (dropped): {len(only_in_old)}")
print(f"   Only in NEW (entered): {len(only_in_new)}")
print(f"   Overlap percentage: {len(overlap)/50*100:.1f}%")

if overlap:
    print(f"\n✅ Tickers appearing in BOTH lists:")
    for ticker in sorted(overlap):
        old_rank = old_df[old_df['ticker'] == ticker]['rank'].values[0]
        new_rank = new_df[new_df['ticker'] == ticker]['rank'].values[0]
        old_pf = old_df[old_df['ticker'] == ticker]['profit_factor'].values[0]
        new_pf = new_df[new_df['ticker'] == ticker]['profit_factor'].values[0]
        print(f"      {ticker:<15} | Old: Rank #{old_rank:>2} PF={old_pf:.2f} | New: Rank #{new_rank:>2} PF={new_pf:.2f}")

print(f"\n❌ Tickers DROPPED from OLD list (top 10):")
dropped_tickers = old_df[old_df['ticker'].isin(only_in_old)].head(10)
for _, row in dropped_tickers.iterrows():
    print(f"      {row['ticker']:<15} | Rank #{row['rank']:>2} | PF={row['profit_factor']:.2f} | Score={row['composite_score']:.2f}")

print(f"\n✨ Tickers ENTERED in NEW list (top 10):")
entered_tickers = new_df[new_df['ticker'].isin(only_in_new)].head(10)
for _, row in entered_tickers.iterrows():
    print(f"      {row['ticker']:<15} | Rank #{row['rank']:>2} | PF={row['profit_factor']:.2f} | Score={row['composite_score']:.2f}")

print(f"\n📈 PROFIT FACTOR COMPARISON:")
print(f"   OLD List (Absolute): Min={old_df['profit_factor'].min():.2f} | Max={old_df['profit_factor'].max():.2f} | Avg={old_df['profit_factor'].mean():.2f}")
print(f"   NEW List (Percentage): Min={new_df['profit_factor'].min():.2f} | Max={new_df['profit_factor'].max():.2f} | Avg={new_df['profit_factor'].mean():.2f}")

print(f"\n🎯 TOP 10 COMPARISON:")
print("-" * 80)
print(f"{'OLD Top 10 (Absolute)':<40} {'NEW Top 10 (Percentage)':<40}")
print("-" * 80)
for i in range(10):
    old_row = old_df.iloc[i]
    new_row = new_df.iloc[i]
    print(f"{old_row['ticker']:<15} PF={old_row['profit_factor']:>5.2f} | {new_row['ticker']:<15} PF={new_row['profit_factor']:>5.2f}")

print("\n" + "=" * 80)
print("🎓 KEY INSIGHT:")
print("   Percentage-based ranking reveals TRUE capital efficiency")
print("   High absolute profit ≠ High percentage return")
print("=" * 80)