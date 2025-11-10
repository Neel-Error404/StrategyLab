"""
Verify Sector Mapping Coverage
===============================

Checks if all 38 tickers in the pool have sector classifications.
Reports missing mappings that need to be added.

Author: MSE Strategy Research Team
Date: 2025-11-08
"""

import pandas as pd
from pathlib import Path

def get_sector_mapping():
    """
    Define sector classification for NSE tickers
    Returns dictionary mapping ticker → sector
    """
    return {
        # Banking & Financial Services
        'KOTAKBANK': 'Banking & Financial Services',
        'AXISBANK': 'Banking & Financial Services',
        'UBL': 'Banking & Financial Services',
        'AUBANK': 'Banking & Financial Services',
        'PNB': 'Banking & Financial Services',
        'SBICARD': 'Banking & Financial Services',
        'FEDERALBNK': 'Banking & Financial Services',
        'IRFC': 'Banking & Financial Services',
        'PFC': 'Banking & Financial Services',

        # Pharmaceuticals & Healthcare
        'MAXHEALTH': 'Pharmaceuticals & Healthcare',
        'LALPATHLAB': 'Pharmaceuticals & Healthcare',
        'SUNPHARMA': 'Pharmaceuticals & Healthcare',
        'IPCALAB': 'Pharmaceuticals & Healthcare',
        'VIMTALABS': 'Pharmaceuticals & Healthcare',
        'CIPLA': 'Pharmaceuticals & Healthcare',

        # Information Technology
        'INFY': 'Information Technology',
        'TCS': 'Information Technology',
        'WIPRO': 'Information Technology',
        'TECHM': 'Information Technology',

        # Consumer Goods & FMCG
        'ITC': 'Consumer Goods & FMCG',
        'BRITANNIA': 'Consumer Goods & FMCG',
        'ABFRL': 'Consumer Goods & FMCG',
        'DABUR': 'Consumer Goods & FMCG',
        'NESTLEIND': 'Consumer Goods & FMCG',
        'VGUARD': 'Consumer Goods & FMCG',

        # Automotive & Auto Components
        'EICHERMOT': 'Automotive & Auto Components',
        'ESCORTS': 'Automotive & Auto Components',
        'MARUTI': 'Automotive & Auto Components',
        'BAJAJ-AUTO': 'Automotive & Auto Components',
        'MSUMI': 'Automotive & Auto Components',
        'TATAMOTORS': 'Automotive & Auto Components',

        # Chemicals & Materials
        'PIDILITIND': 'Chemicals & Materials',
        'TATACHEM': 'Chemicals & Materials',
        'GRAPHITE': 'Chemicals & Materials',
        'HERCULES': 'Chemicals & Materials',
        'POCL': 'Chemicals & Materials',

        # Energy & Power
        'NTPC': 'Energy & Power',
        'POWERGRID': 'Energy & Power',
        'PETRONET': 'Energy & Power',

        # Metals & Mining
        'NMDC': 'Metals & Mining',
        'TATASTEEL': 'Metals & Mining',
        'JINDALSTEL': 'Metals & Mining',
        'SHYAMMETL': 'Metals & Mining',

        # Hotels & Hospitality
        'INDHOTEL': 'Hotels & Hospitality',

        # Cement
        'RAMCOCEM': 'Cement',

        # Transportation & Logistics
        'DELHIVERY': 'Transportation & Logistics',

        # PSU Bank (ETF/Index)
        'PSUBANK': 'Banking & Financial Services',
    }

def main():
    print("=" * 80)
    print("SECTOR MAPPING VERIFICATION")
    print("=" * 80)
    print()

    # Load the 38-ticker pool
    pool_file = Path("analysis/output/mse_strategy_backtesting/20251107_154802_full/ticker_pool_selection/selected_ticker_pool_38_under2000.csv")

    pool_df = pd.read_csv(pool_file)
    tickers = pool_df['ticker'].tolist()

    print(f"[INFO] Loaded {len(tickers)} tickers from pool")
    print()

    # Get sector mapping
    sector_map = get_sector_mapping()

    print(f"[INFO] Sector mapping dictionary has {len(sector_map)} tickers")
    print()

    # Check coverage
    mapped = []
    unmapped = []

    for ticker in tickers:
        if ticker in sector_map:
            mapped.append((ticker, sector_map[ticker]))
        else:
            unmapped.append(ticker)

    # Report
    print("=" * 80)
    print("COVERAGE ANALYSIS")
    print("=" * 80)
    print()
    print(f"[OK] Mapped tickers: {len(mapped)}/{len(tickers)} ({len(mapped)/len(tickers)*100:.1f}%)")
    print(f"[MISSING] Unmapped tickers: {len(unmapped)}/{len(tickers)} ({len(unmapped)/len(tickers)*100:.1f}%)")
    print()

    if unmapped:
        print("=" * 80)
        print("UNMAPPED TICKERS (NEED SECTOR CLASSIFICATION)")
        print("=" * 80)
        print()

        # Get details for unmapped tickers
        unmapped_details = pool_df[pool_df['ticker'].isin(unmapped)][['ticker', 'current_price', 'win_rate', 'profit_factor', 'sharpe_like_ratio', 'selection_rank']]

        for _, row in unmapped_details.iterrows():
            print(f"  {row['ticker']:15s} | Price: ₹{row['current_price']:>8,.2f} | Rank: #{int(row['selection_rank']):2d} | Sharpe: {row['sharpe_like_ratio']:>6.4f} | PF: {row['profit_factor']:.4f}")

        print()
        print("Suggested sector classifications (manual research needed):")
        print()

        # Provide hints based on ticker names
        hints = {
            'NMDC': 'Metals & Mining (National Mineral Development Corporation)',
            'PNB': 'Banking & Financial Services (Punjab National Bank)',
            'HERCULES': 'Chemicals & Materials (Hercules Hoists)',
            'PSUBANK': 'Banking & Financial Services (PSU Bank ETF)',
            'NTPC': 'Energy & Power (National Thermal Power Corporation)',
            'DELHIVERY': 'Transportation & Logistics',
            'FEDERALBNK': 'Banking & Financial Services (Federal Bank)',
            'SUNPHARMA': 'Pharmaceuticals & Healthcare (Sun Pharma)',
            'IPCALAB': 'Pharmaceuticals & Healthcare (IPCA Laboratories)',
            'ABFRL': 'Consumer Goods & FMCG (Aditya Birla Fashion & Retail)',
            'POWERGRID': 'Energy & Power (Power Grid Corporation)',
            'IRFC': 'Banking & Financial Services (Indian Railway Finance Corporation)',
            'DABUR': 'Consumer Goods & FMCG',
            'PETRONET': 'Energy & Power (Petronet LNG)',
            'VIMTALABS': 'Pharmaceuticals & Healthcare (Vimta Labs)',
            'TATASTEEL': 'Metals & Mining',
            'MSUMI': 'Automotive & Auto Components (Motherson Sumi)',
            'RAMCOCEM': 'Cement (Ramco Cements)',
            'GRAPHITE': 'Chemicals & Materials (Graphite India)',
            'TATAMOTORS': 'Automotive & Auto Components',
            'PFC': 'Banking & Financial Services (Power Finance Corporation)',
            'NESTLEIND': 'Consumer Goods & FMCG (Nestle India)',
            'WIPRO': 'Information Technology',
            'VGUARD': 'Consumer Goods & FMCG (V-Guard Industries)',
            'TECHM': 'Information Technology (Tech Mahindra)',
            'SHYAMMETL': 'Metals & Mining (Shyam Metalics)',
            'CIPLA': 'Pharmaceuticals & Healthcare',
            'INDHOTEL': 'Hotels & Hospitality (Indian Hotels)',
            'POCL': 'Chemicals & Materials (POCL - Pen',
            'JINDALSTEL': 'Metals & Mining (Jindal Steel)',
        }

        for ticker in unmapped:
            if ticker in hints:
                print(f"  '{ticker}': '{hints[ticker]}',")

        print()

    else:
        print("[SUCCESS] All tickers have sector classifications!")
        print()

    # Show sector distribution
    if mapped:
        print("=" * 80)
        print("SECTOR DISTRIBUTION")
        print("=" * 80)
        print()

        sector_counts = {}
        for ticker, sector in mapped:
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        for sector, count in sorted(sector_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {sector:40s} | {count:2d} tickers")

        print()


if __name__ == "__main__":
    main()
