#!/usr/bin/env python3
"""
SECTOR CLASSIFICATION & CORRELATION ANALYSIS (Config-Driven)
============================================================

Purpose: Sector classification and correlation analysis for portfolio construction
- Create sector mapping for affordable tickers
- Calculate correlation matrix from actual trade returns
- Prepare foundation for diversification rules

Input: Anti-cascading trades + affordable tickers metadata from Step 01
Output: Sector mapping, correlation matrix, daily returns data

Author: Portfolio Construction Team
Version: 2.0 - Config-Driven (migrated October 2025)
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PORTFOLIO_DIR = SCRIPT_DIR.parent
ANALYSIS_DIR = PORTFOLIO_DIR.parent
sys.path.insert(0, str(ANALYSIS_DIR))

from generic.modules.config_loader import load_config, resolve_paths, get_output_dir, get_module_spec


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Sector Classification & Correlation Analysis')
    parser.add_argument('--config', required=True, help='Path to YAML configuration file')
    return parser.parse_args()


def load_anti_cascading_data(config):
    """
    Load the anti-cascading dataset and metadata from Step 01
    """

    print("🏗️ SECTOR CLASSIFICATION & CORRELATION ANALYSIS")
    print("=" * 80)
    print("📊 Loading anti-cascading trades dataset...")

    # Get output directory from previous module (anti_cascade_filter)
    prev_output_dir = Path(get_output_dir(config, 'anti_cascade_filter', category='portfolio'))

    # Load anti-cascading trades
    trades_file = prev_output_dir / "anti_cascading_trades_filtered.csv"
    trades_df = pd.read_csv(trades_file)

    # Load affordable tickers metadata
    tickers_file = prev_output_dir / "affordable_tickers_metadata.csv"
    affordable_tickers = pd.read_csv(tickers_file)

    print(f"✅ Loaded {len(trades_df):,} anti-cascading trades")
    print(f"✅ Working with {len(affordable_tickers)} affordable tickers")
    print(f"📁 Source: {prev_output_dir}")

    # Convert datetime
    trades_df['Entry Time'] = pd.to_datetime(trades_df['Entry Time'])
    trades_df['Exit Time'] = pd.to_datetime(trades_df['Exit Time'])

    return trades_df, affordable_tickers


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
        'AADHARHFC': 'Banking & Financial Services',
        'HDFCAMC': 'Banking & Financial Services',
        'SBIN': 'Banking & Financial Services',
        'PNB': 'Banking & Financial Services',
        'INDUSINDBK': 'Banking & Financial Services',
        'CANBK': 'Banking & Financial Services',
        'UNIONBANK': 'Banking & Financial Services',
        'BANKINDIA': 'Banking & Financial Services',
        'BANDHANBNK': 'Banking & Financial Services',
        'LICHSGFIN': 'Banking & Financial Services',
        'SBICARD': 'Banking & Financial Services',
        'OFSS': 'Banking & Financial Services',
        'RECLTD': 'Banking & Financial Services',

        # Pharmaceuticals & Healthcare
        'HCG': 'Pharmaceuticals & Healthcare',
        'FORTIS': 'Pharmaceuticals & Healthcare',
        'MAXHEALTH': 'Pharmaceuticals & Healthcare',
        'ARTEMISMED': 'Pharmaceuticals & Healthcare',
        'GLENMARK': 'Pharmaceuticals & Healthcare',
        'NH': 'Pharmaceuticals & Healthcare',
        'BIOCON': 'Pharmaceuticals & Healthcare',
        'APOLLOHOSP': 'Pharmaceuticals & Healthcare',
        'ABBOTINDIA': 'Pharmaceuticals & Healthcare',
        'LALPATHLAB': 'Pharmaceuticals & Healthcare',

        # Information Technology
        'HCLTECH': 'Information Technology',
        'INFY': 'Information Technology',
        'NEWGEN': 'Information Technology',
        'INTELLECT': 'Information Technology',
        'CREATIVE': 'Information Technology',
        'TCS': 'Information Technology',
        'TATAELXSI': 'Information Technology',

        # Consumer Goods & FMCG
        'EMAMILTD': 'Consumer Goods & FMCG',
        'VBL': 'Consumer Goods & FMCG',
        'WHIRLPOOL': 'Consumer Goods & FMCG',
        'GODFRYPHLP': 'Consumer Goods & FMCG',
        'HINDUNILVR': 'Consumer Goods & FMCG',
        'ASIANPAINT': 'Consumer Goods & FMCG',
        'BRITANNIA': 'Consumer Goods & FMCG',
        'ITC': 'Consumer Goods & FMCG',
        'JUBLFOOD': 'Consumer Goods & FMCG',

        # Automotive & Auto Components
        'BHARATFORG': 'Automotive & Auto Components',
        'KIRLOSBROS': 'Automotive & Auto Components',
        'EICHERMOT': 'Automotive & Auto Components',
        'M&M': 'Automotive & Auto Components',
        'MARUTI': 'Automotive & Auto Components',
        'BALKRISIND': 'Automotive & Auto Components',
        'CEATLTD': 'Automotive & Auto Components',
        'ESCORTS': 'Automotive & Auto Components',
        'BAJAJ-AUTO': 'Automotive & Auto Components',
        'ASHOKLEY': 'Automotive & Auto Components',
        'SONACOMS': 'Automotive & Auto Components',

        # Chemicals & Materials
        'BERGEPAINT': 'Chemicals & Materials',
        'CCL': 'Chemicals & Materials',
        'KAJARIACER': 'Chemicals & Materials',
        'PIDILITIND': 'Chemicals & Materials',
        'SUPREMEIND': 'Chemicals & Materials',
        'COROMANDEL': 'Chemicals & Materials',
        'TATACHEM': 'Chemicals & Materials',
        'ATUL': 'Chemicals & Materials',
        'DEEPAKNTR': 'Chemicals & Materials',
        'AARTIIND': 'Chemicals & Materials',
        'BALRAMCHIN': 'Chemicals & Materials',
        'GSFC': 'Chemicals & Materials',

        # Infrastructure & Construction
        'LT': 'Infrastructure & Construction',
        'JSWSTEEL': 'Infrastructure & Construction',

        # Insurance
        'ICICIGI': 'Insurance',
        'LICI': 'Insurance',
        'NIACL': 'Insurance',

        # Metals & Mining
        'VEDL': 'Metals & Mining',

        # Hotels & Hospitality
        'EIHOTEL': 'Hotels & Hospitality',

        # Textiles
        'BHARTIHEXA': 'Textiles',

        # Energy & Power
        'TATAPOWER': 'Energy & Power',
        'POWERINDIA': 'Energy & Power',
        'ADANIENT': 'Energy & Power',
        'RELIANCE': 'Energy & Power',

        # Diversified
        'BAJAJHLDNG': 'Diversified',
        'AMBER': 'Diversified',
        'ABB': 'Diversified',
        'BEML': 'Diversified',
        'MCX': 'Diversified',
        '3MINDIA': 'Diversified',
        'HONAUT': 'Diversified',

        # Cement
        'ULTRACEMCO': 'Cement',
        'SHREECEM': 'Cement',

        # Retail & E-commerce
        'PAYTM': 'Retail & E-commerce',
        'JUSTDIAL': 'Retail & E-commerce',

        # Transportation
        'IRCTC': 'Transportation',
        'IEX': 'Transportation',
        'GMDCLTD': 'Transportation',

        # Paper & Packaging
        'PAGEIND': 'Paper & Packaging',

        # Specialty Finance
        'MRF': 'Specialty Finance',
    }


def create_sector_mapping(affordable_tickers, sector_mapping_dict):
    """
    Create sector classification for affordable tickers
    """

    print(f"\n🎯 STEP 1: SECTOR CLASSIFICATION MAPPING")
    print("=" * 60)

    # Create sector analysis dataframe
    sector_data = []
    for _, ticker_row in affordable_tickers.iterrows():
        ticker = ticker_row['ticker']
        sector = sector_mapping_dict.get(ticker, 'Unclassified')

        sector_data.append({
            'ticker': ticker,
            'current_price': ticker_row['current_price'],
            'price_category': ticker_row['price_category'],
            'anticascading_rank': ticker_row['anticascading_rank'],
            'composite_score': ticker_row['composite_score'],
            'profit_factor': ticker_row['profit_factor'],
            'sharpe_like_ratio': ticker_row['sharpe_like_ratio'],
            'sector': sector
        })

    sector_df = pd.DataFrame(sector_data)

    # Analyze sector distribution
    sector_counts = sector_df['sector'].value_counts()

    print(f"📊 SECTOR DISTRIBUTION ANALYSIS:")
    print(f"   Total sectors identified: {len(sector_counts)}")
    print(f"   Total tickers classified: {len(sector_df)}")

    print(f"\n📋 SECTOR BREAKDOWN:")
    for sector, count in sector_counts.items():
        percentage = (count / len(sector_df)) * 100
        tickers_in_sector = sector_df[sector_df['sector'] == sector]['ticker'].tolist()
        avg_sharpe = sector_df[sector_df['sector'] == sector]['sharpe_like_ratio'].mean()
        print(f"   {sector:35} | {count:2} tickers ({percentage:4.1f}%) | Avg Sharpe: {avg_sharpe:.3f}")
        print(f"      → {', '.join(tickers_in_sector)}")

    # Check diversification potential
    print(f"\n🎯 DIVERSIFICATION ANALYSIS:")
    max_sector_concentration = sector_counts.max() / len(sector_df) * 100
    print(f"   Maximum sector concentration: {max_sector_concentration:.1f}%")
    print(f"   Minimum sectors for diversified portfolio: {min(3, len(sector_counts))}")
    print(f"   Available for sector-balanced portfolios: {'✅ YES' if len(sector_counts) >= 3 else '❌ NO'}")

    # Performance analysis by sector
    print(f"\n⭐ SECTOR PERFORMANCE ANALYSIS:")
    sector_performance = sector_df.groupby('sector').agg({
        'sharpe_like_ratio': 'mean',
        'profit_factor': 'mean',
        'composite_score': 'mean'
    }).round(3)

    for sector, metrics in sector_performance.iterrows():
        print(f"   {sector:35} | Sharpe: {metrics['sharpe_like_ratio']:.3f} | PF: {metrics['profit_factor']:.2f} | Score: {metrics['composite_score']:.1f}")

    return sector_df


def calculate_correlation_matrix(trades_df, sector_df):
    """
    Calculate correlation matrix from actual daily trade returns
    """

    print(f"\n📊 STEP 2: CORRELATION MATRIX CALCULATION")
    print("=" * 60)
    print("🔍 Calculating correlations from actual daily trade returns...")

    # Calculate daily returns for each ticker
    print("   Processing daily returns per ticker...")

    daily_returns_data = {}
    ticker_list = sector_df['ticker'].tolist()

    for ticker in ticker_list:
        print(f"   {ticker:12} → ", end='')

        ticker_trades = trades_df[trades_df['ticker'] == ticker].copy()

        if len(ticker_trades) < 10:  # Minimum trades threshold
            print(f"❌ Insufficient trades ({len(ticker_trades)})")
            continue

        # Create daily returns
        ticker_trades['trade_date'] = ticker_trades['Entry Time'].dt.date
        ticker_trades['trade_return'] = (ticker_trades['Exit Price'] / ticker_trades['Entry Price'] - 1) * 100

        # Average returns per day (if multiple trades per day)
        daily_returns = ticker_trades.groupby('trade_date')['trade_return'].mean()

        daily_returns_data[ticker] = daily_returns
        print(f"✅ {len(daily_returns)} days | Avg: {daily_returns.mean():.3f}% | Std: {daily_returns.std():.3f}%")

    print(f"\n   Building correlation matrix...")

    # Create correlation matrix from daily returns
    if len(daily_returns_data) < 2:
        print("❌ Insufficient tickers with adequate trade data for correlation analysis")
        return None, None

    # Align all daily returns to common dates
    correlation_df = pd.DataFrame(daily_returns_data).fillna(0)

    print(f"   Correlation matrix dimensions: {correlation_df.shape}")
    print(f"   Total trading days analyzed: {len(correlation_df)}")
    print(f"   Date range: {correlation_df.index.min()} to {correlation_df.index.max()}")

    # Calculate correlation matrix
    correlation_matrix = correlation_df.corr()

    # Analyze correlation statistics
    correlation_values = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            correlation_values.append(abs(correlation_matrix.iloc[i, j]))

    print(f"\n📊 CORRELATION STATISTICS:")
    print(f"   Average correlation: {np.mean(correlation_values):.3f}")
    print(f"   Maximum correlation: {np.max(correlation_values):.3f}")
    print(f"   Minimum correlation: {np.min(correlation_values):.3f}")
    print(f"   Median correlation: {np.median(correlation_values):.3f}")

    return correlation_matrix, daily_returns_data


def analyze_sector_correlations(correlation_matrix, sector_df):
    """
    Analyze within-sector vs cross-sector correlations
    """

    if correlation_matrix is None:
        return

    print(f"\n🔍 STEP 3: SECTOR CORRELATION ANALYSIS")
    print("=" * 60)

    # Group tickers by sector
    sectors = sector_df.groupby('sector')['ticker'].apply(list).to_dict()

    # Calculate within-sector and cross-sector correlations
    within_sector_corrs = []
    cross_sector_corrs = []

    for sector1, tickers1 in sectors.items():
        for ticker1 in tickers1:
            if ticker1 not in correlation_matrix.columns:
                continue

            for sector2, tickers2 in sectors.items():
                for ticker2 in tickers2:
                    if ticker2 not in correlation_matrix.columns or ticker1 == ticker2:
                        continue

                    corr_value = abs(correlation_matrix.loc[ticker1, ticker2])

                    if sector1 == sector2:
                        within_sector_corrs.append(corr_value)
                    else:
                        cross_sector_corrs.append(corr_value)

    if within_sector_corrs and cross_sector_corrs:
        print(f"📊 DIVERSIFICATION INSIGHTS:")
        print(f"   Average within-sector correlation: {np.mean(within_sector_corrs):.3f}")
        print(f"   Average cross-sector correlation:  {np.mean(cross_sector_corrs):.3f}")

        diversification_benefit = np.mean(within_sector_corrs) - np.mean(cross_sector_corrs)
        print(f"   Diversification benefit: {diversification_benefit:.3f}")

        if diversification_benefit > 0.1:
            print("   ✅ Strong diversification potential across sectors")
        elif diversification_benefit > 0.05:
            print("   ✅ Good diversification potential across sectors")
        else:
            print("   ⚠️  Limited diversification benefit across sectors")


def save_sector_correlation_data(config, sector_df, correlation_matrix, daily_returns_data):
    """
    Save sector and correlation analysis results
    """

    print(f"\n💾 STEP 4: SAVING SECTOR & CORRELATION DATA")
    print("=" * 60)

    # Get output directory from config
    output_dir = Path(get_output_dir(config, 'sector_classification', category='portfolio'))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save sector mapping
    sector_file = output_dir / "sector_mapping.csv"
    sector_df.to_csv(sector_file, index=False)
    print(f"✅ Sector mapping saved: {sector_file.name}")

    # Save correlation matrix
    if correlation_matrix is not None:
        corr_file = output_dir / "correlation_matrix.csv"
        correlation_matrix.to_csv(corr_file)
        print(f"✅ Correlation matrix saved: {corr_file.name}")

        # Save daily returns data for future use
        if daily_returns_data:
            returns_df = pd.DataFrame(daily_returns_data).fillna(0)
            returns_file = output_dir / "daily_returns_data.csv"
            returns_df.to_csv(returns_file)
            print(f"✅ Daily returns data saved: {returns_file.name}")

    # Create summary report
    summary_file = output_dir / "sector_correlation_summary.md"
    with open(summary_file, 'w') as f:
        f.write("# SECTOR CLASSIFICATION & CORRELATION ANALYSIS SUMMARY\n\n")
        f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Tickers Analyzed:** {len(sector_df)}\n")
        f.write(f"**Sectors Identified:** {sector_df['sector'].nunique()}\n\n")

        if correlation_matrix is not None:
            f.write(f"**Correlation Matrix Size:** {correlation_matrix.shape[0]}x{correlation_matrix.shape[1]}\n\n")

            # Calculate correlation stats
            correlation_values = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    correlation_values.append(abs(correlation_matrix.iloc[i, j]))

            f.write("## CORRELATION STATISTICS\n\n")
            f.write(f"- Average Correlation: {np.mean(correlation_values):.3f}\n")
            f.write(f"- Maximum Correlation: {np.max(correlation_values):.3f}\n")
            f.write(f"- Minimum Correlation: {np.min(correlation_values):.3f}\n")
            f.write(f"- Median Correlation: {np.median(correlation_values):.3f}\n\n")

        f.write("## SECTOR DISTRIBUTION\n\n")
        f.write("| Sector | Ticker Count |\n")
        f.write("|--------|-------------|\n")
        for sector, count in sector_df['sector'].value_counts().items():
            f.write(f"| {sector} | {count} |\n")

        f.write("\n## PERFORMANCE VALIDATION\n\n")
        f.write(f"- Sharpe Ratio Range: {sector_df['sharpe_like_ratio'].min():.3f} to {sector_df['sharpe_like_ratio'].max():.3f}\n")
        f.write(f"- Profit Factor Range: {sector_df['profit_factor'].min():.2f} to {sector_df['profit_factor'].max():.2f}\n")

    print(f"✅ Analysis summary saved: {summary_file.name}")
    print(f"📁 Location: {output_dir}")

    print(f"\n🎉 SECTOR CLASSIFICATION & CORRELATION ANALYSIS COMPLETED!")
    print(f"📊 Ready for next step: Intelligent Combination Generation")


def main():
    """
    Execute the complete sector classification and correlation analysis
    """

    # Parse arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    paths = resolve_paths(config)
    module_config = get_module_spec(config, 'sector_classification', category='portfolio')

    print("🚀 STARTING SECTOR CLASSIFICATION & CORRELATION ANALYSIS")
    print("=" * 80)
    print(f"📁 Config: {args.config}")
    print(f"📊 Strategy: {config['run']['strategy']}")
    print(f"📅 Date Range: {config['run']['date_range']}")
    print("=" * 80)

    try:
        # Load anti-cascading data
        trades_df, affordable_tickers = load_anti_cascading_data(config)

        # Get sector mapping dictionary
        sector_mapping_dict = get_sector_mapping()

        # Create sector mapping
        sector_df = create_sector_mapping(affordable_tickers, sector_mapping_dict)

        # Calculate correlation matrix from actual trades
        correlation_matrix, daily_returns_data = calculate_correlation_matrix(trades_df, sector_df)

        # Analyze sector correlations
        analyze_sector_correlations(correlation_matrix, sector_df)

        # Save results
        save_sector_correlation_data(config, sector_df, correlation_matrix, daily_returns_data)

        print(f"\n🏆 SECTOR ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"🎯 Next: Intelligent Combination Generation")

        return {
            'sector_df': sector_df,
            'correlation_matrix': correlation_matrix,
            'daily_returns_data': daily_returns_data
        }

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = main()
