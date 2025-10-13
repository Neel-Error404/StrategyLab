#!/usr/bin/env python3
"""
INTELLIGENT COMBINATION GENERATION (Config-Driven)
================================================

Purpose: Generate portfolio combinations with intelligent pre-filtering
- Apply sector diversification and correlation constraints
- Generate combinations for multiple portfolio sizes (4-8 tickers)
- Use trade-level performance metrics for validation

Input: Sector mapping, correlation matrix, filtered trades from previous steps
Output: Valid combinations CSV for different portfolio sizes

Author: Portfolio Construction Team
Version: 2.0 - Config-Driven (migrated October 2025)
"""

import argparse
import pandas as pd
import numpy as np
import itertools
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
    parser = argparse.ArgumentParser(description='Intelligent Combination Generation')
    parser.add_argument('--config', required=True, help='Path to YAML configuration file')
    return parser.parse_args()


def load_phase2_data(config):
    """
    Load results from Phase 2: sector mapping, correlation, and trades
    """

    print("🔧 INTELLIGENT COMBINATION GENERATION")
    print("=" * 80)
    print("📊 Loading Phase 2 results...")

    # Get sector classification output directory
    sector_output_dir = Path(get_output_dir(config, 'sector_classification', category='portfolio'))

    # Load sector mapping
    sector_df = pd.read_csv(sector_output_dir / "sector_mapping.csv")

    # Load correlation matrix
    correlation_matrix = pd.read_csv(sector_output_dir / "correlation_matrix.csv", index_col=0)

    # Get anti-cascading output directory
    filter_output_dir = Path(get_output_dir(config, 'anti_cascade_filter', category='portfolio'))

    # Load anti-cascading trades for trade count analysis
    trades_df = pd.read_csv(filter_output_dir / "anti_cascading_trades_filtered.csv")

    print(f"✅ Loaded {len(sector_df)} tickers with sector mapping")
    print(f"✅ Loaded {correlation_matrix.shape[0]}x{correlation_matrix.shape[1]} correlation matrix")
    print(f"✅ Loaded {len(trades_df):,} anti-cascading trades")

    return sector_df, correlation_matrix, trades_df


def calculate_individual_metrics(trades_df, sector_df):
    """
    Calculate individual ticker performance metrics from trade-level data
    """

    print(f"\n🎯 STEP 1: INDIVIDUAL TICKER PERFORMANCE ANALYSIS")
    print("=" * 70)

    individual_metrics = {}

    for _, ticker_row in sector_df.iterrows():
        ticker = ticker_row['ticker']

        # Get ticker trades
        ticker_trades = trades_df[trades_df['ticker'] == ticker].copy()

        if len(ticker_trades) < 10:  # Skip tickers with too few trades
            print(f"   {ticker:12} | ❌ Insufficient trades ({len(ticker_trades)})")
            continue

        # Calculate trade-level percentage returns
        ticker_trades['percentage_return'] = (ticker_trades['Exit Price'] / ticker_trades['Entry Price'] - 1) * 100

        # Use trade-level returns directly
        trade_returns = ticker_trades['percentage_return']

        # Calculate metrics from individual trades
        if len(trade_returns) > 0 and trade_returns.std() > 0:
            avg_return_per_trade = trade_returns.mean()
            volatility_per_trade = trade_returns.std()
            sharpe_ratio = avg_return_per_trade / volatility_per_trade if volatility_per_trade > 0 else 0

            # Profit factor (trade-level)
            winning_trades = trade_returns[trade_returns > 0]
            losing_trades = trade_returns[trade_returns < 0]
            profit_factor = abs(winning_trades.sum() / losing_trades.sum()) if len(losing_trades) > 0 else float('inf')

            # Win rate (accuracy)
            win_rate = len(winning_trades) / len(trade_returns) if len(trade_returns) > 0 else 0

            individual_metrics[ticker] = {
                'trade_count': len(ticker_trades),
                'avg_return_per_trade': avg_return_per_trade,
                'volatility_per_trade': volatility_per_trade,
                'sharpe_ratio': sharpe_ratio,
                'profit_factor': profit_factor,
                'win_rate': win_rate
            }

            print(f"   {ticker:12} | Trades: {len(ticker_trades):4} | Sharpe: {sharpe_ratio:6.3f} | PF: {profit_factor:5.2f} | WR: {win_rate*100:5.1f}%")

    print(f"\n✅ Calculated metrics for {len(individual_metrics)} tickers")

    return individual_metrics


def apply_pre_filtering(sector_df, correlation_matrix, individual_metrics, min_trades=200):
    """
    Apply pre-filtering to select valid tickers for portfolio generation
    """

    print(f"\n🔍 STEP 2: PRE-FILTERING TICKER UNIVERSE")
    print("=" * 70)

    valid_tickers = []

    print(f"\n📊 FILTER: MINIMUM TRADE THRESHOLD ({min_trades:,} trades)")
    for ticker, metrics in individual_metrics.items():
        if metrics['trade_count'] >= min_trades:
            valid_tickers.append(ticker)
            print(f"   {ticker:12} | ✅ {metrics['trade_count']:,} trades")
        else:
            print(f"   {ticker:12} | ❌ {metrics['trade_count']:,} trades (excluded)")

    print(f"\n   Result: {len(valid_tickers)}/{len(individual_metrics)} tickers pass minimum trade threshold")

    print(f"\n📊 FINAL VALID TICKER UNIVERSE: {len(valid_tickers)} tickers")
    for ticker in valid_tickers:
        metrics = individual_metrics[ticker]
        accuracy_pct = metrics['win_rate'] * 100
        print(f"   {ticker:12} | PF: {metrics['profit_factor']:5.2f} | Acc: {accuracy_pct:5.1f}% | Sharpe: {metrics['sharpe_ratio']:6.3f}")

    # Show Sharpe distribution
    if valid_tickers:
        sharpe_values = [individual_metrics[t]['sharpe_ratio'] for t in valid_tickers]
        positive_sharpe = len([s for s in sharpe_values if s > 0])
        print(f"\n📈 SHARPE RATIO DISTRIBUTION:")
        print(f"   Positive Sharpe: {positive_sharpe}/{len(sharpe_values)} tickers ({positive_sharpe/len(sharpe_values)*100:.1f}%)")
        print(f"   Range: {min(sharpe_values):.3f} to {max(sharpe_values):.3f}")

    return valid_tickers


def check_sector_diversification(ticker_combination, sector_df, max_sector_weight=0.6):
    """
    Check if combination meets sector diversification constraint
    """

    # Get sectors for this combination
    sectors = []
    for ticker in ticker_combination:
        sector = sector_df[sector_df['ticker'] == ticker]['sector'].iloc[0]
        sectors.append(sector)

    # Calculate sector weights (equal weight assumption)
    sector_counts = {}
    for sector in sectors:
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    # Check maximum sector weight
    max_weight = max(sector_counts.values()) / len(ticker_combination)

    return max_weight <= max_sector_weight


def check_correlation_constraint(ticker_combination, correlation_matrix, max_avg_correlation=0.75):
    """
    Check if combination meets correlation constraint
    """

    if len(ticker_combination) < 2:
        return True

    portfolio_correlations = []

    for i in range(len(ticker_combination)):
        for j in range(i+1, len(ticker_combination)):
            ticker1 = ticker_combination[i]
            ticker2 = ticker_combination[j]

            if ticker1 in correlation_matrix.index and ticker2 in correlation_matrix.columns:
                corr = correlation_matrix.loc[ticker1, ticker2]
                portfolio_correlations.append(abs(corr))

    if not portfolio_correlations:
        return True

    avg_correlation = np.mean(portfolio_correlations)

    return avg_correlation <= max_avg_correlation


def generate_portfolio_combinations(valid_tickers, sector_df, correlation_matrix, portfolio_size,
                                    max_sector_weight=0.6, max_avg_correlation=0.75):
    """
    Generate valid portfolio combinations with diversification filters
    """

    print(f"\n🎯 STEP 3: GENERATING {portfolio_size}-TICKER COMBINATIONS")
    print("=" * 70)

    # Calculate total possible combinations
    total_possible = len(list(itertools.combinations(valid_tickers, portfolio_size)))
    print(f"📊 Total possible {portfolio_size}-ticker combinations: {total_possible:,}")

    valid_combinations = []
    tested = 0
    sector_pass = 0
    correlation_pass = 0

    print(f"\n🔄 Generating and filtering combinations...")
    print(f"   Diversification filters:")
    print(f"   - Max sector concentration: {max_sector_weight*100:.0f}%")
    print(f"   - Max average correlation: {max_avg_correlation:.2f}")

    for combination in itertools.combinations(valid_tickers, portfolio_size):
        tested += 1

        # Filter 1: Sector diversification
        if not check_sector_diversification(combination, sector_df, max_sector_weight):
            continue
        sector_pass += 1

        # Filter 2: Correlation constraint
        if not check_correlation_constraint(combination, correlation_matrix, max_avg_correlation):
            continue
        correlation_pass += 1

        valid_combinations.append(combination)

        # Progress update every 10,000 combinations
        if tested % 10000 == 0:
            print(f"   Tested: {tested:,} | Valid: {len(valid_combinations):,} ({len(valid_combinations)/tested*100:.1f}%)")

    print(f"\n📊 GENERATION RESULTS:")
    print(f"   Total tested: {tested:,}")
    print(f"   Passed sector filter: {sector_pass:,} ({sector_pass/tested*100:.1f}%)")
    print(f"   Passed correlation filter: {correlation_pass:,} ({correlation_pass/sector_pass*100:.1f}% of sector-valid)")
    print(f"   Final valid combinations: {len(valid_combinations):,} ({len(valid_combinations)/tested*100:.1f}%)")

    return valid_combinations


def save_valid_combinations(config, valid_combinations, portfolio_size, valid_tickers, sector_df):
    """
    Save valid combinations to CSV
    """

    print(f"\n💾 STEP 4: SAVING VALID COMBINATIONS")
    print("=" * 60)

    # Get output directory
    output_dir = Path(get_output_dir(config, 'combination_generator', category='portfolio'))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare combinations data
    combinations_data = []
    for i, combination in enumerate(valid_combinations):
        # Get sector information
        sectors = []
        for ticker in combination:
            sector = sector_df[sector_df['ticker'] == ticker]['sector'].iloc[0]
            sectors.append(sector)

        combinations_data.append({
            'combination_id': f"{portfolio_size}T_{i+1:06d}",
            'portfolio_size': portfolio_size,
            'tickers': '|'.join(combination),
            'sectors': '|'.join(sectors),
            'unique_sectors': len(set(sectors))
        })

    # Save combinations
    combinations_df = pd.DataFrame(combinations_data)
    combinations_file = output_dir / f"valid_combinations_{portfolio_size}ticker.csv"
    combinations_df.to_csv(combinations_file, index=False)

    # Save summary
    summary_file = output_dir / f"combination_generation_summary_{portfolio_size}ticker.md"
    with open(summary_file, 'w') as f:
        f.write(f"# VALID COMBINATIONS GENERATION SUMMARY - {portfolio_size} TICKERS\n\n")
        f.write(f"**Generation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Portfolio Size:** {portfolio_size} tickers\n")
        f.write(f"**Valid Ticker Universe:** {len(valid_tickers)} tickers\n")
        f.write(f"**Valid Combinations Generated:** {len(valid_combinations):,}\n")
        f.write(f"**Average Unique Sectors per Portfolio:** {combinations_df['unique_sectors'].mean():.1f}\n\n")
        f.write(f"## TICKER UNIVERSE USED\n\n")
        for ticker in sorted(valid_tickers):
            f.write(f"- {ticker}\n")

    print(f"✅ Valid combinations saved: {combinations_file.name}")
    print(f"✅ Summary saved: {summary_file.name}")
    print(f"📁 Location: {output_dir}")
    print(f"\n📊 Ready for portfolio optimization with {len(valid_combinations):,} combinations")

    return str(combinations_file)


def main():
    """
    Execute the complete intelligent combination generation
    """

    # Parse arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    paths = resolve_paths(config)
    module_config = get_module_spec(config, 'combination_generator', category='portfolio')

    print("🚀 STARTING INTELLIGENT COMBINATION GENERATION")
    print("=" * 80)
    print(f"📁 Config: {args.config}")
    print(f"📊 Strategy: {config['run']['strategy']}")
    print(f"📅 Date Range: {config['run']['date_range']}")
    print("=" * 80)

    try:
        # Get configuration parameters
        cfg = module_config.get('config', {})
        portfolio_sizes = cfg.get('portfolio_sizes', [4, 5, 6, 7, 8])
        max_sector_concentration = cfg.get('max_sector_concentration', 0.6)
        max_correlation = cfg.get('max_correlation', 0.75)
        min_trades = cfg.get('min_trades_per_ticker', 200)

        # Load Phase 2 results
        sector_df, correlation_matrix, trades_df = load_phase2_data(config)

        # Calculate individual metrics
        individual_metrics = calculate_individual_metrics(trades_df, sector_df)

        # Apply pre-filtering
        valid_tickers = apply_pre_filtering(sector_df, correlation_matrix, individual_metrics, min_trades)

        # Generate combinations for different portfolio sizes
        results = {}
        for size in portfolio_sizes:
            print(f"\n" + "="*80)
            print(f"PROCESSING {size}-TICKER PORTFOLIOS")
            print("="*80)

            valid_combinations = generate_portfolio_combinations(
                valid_tickers, sector_df, correlation_matrix, size,
                max_sector_concentration, max_correlation
            )

            if valid_combinations:
                combinations_file = save_valid_combinations(
                    config, valid_combinations, size, valid_tickers, sector_df
                )
                results[size] = {
                    'combinations_count': len(valid_combinations),
                    'combinations_file': combinations_file
                }

        print(f"\n🏆 COMBINATION GENERATION COMPLETED SUCCESSFULLY!")
        print(f"\n📊 GENERATION SUMMARY:")
        for size, result in results.items():
            if result:
                print(f"   {size}-ticker portfolios: {result['combinations_count']:,} valid combinations")

        print(f"\n🎯 Next: Portfolio Optimization")

        return results

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
