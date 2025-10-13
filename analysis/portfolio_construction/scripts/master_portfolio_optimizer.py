#!/usr/bin/env python3
"""
MASTER PORTFOLIO OPTIMIZER
============================
Orchestrates Scripts 3 & 4 across multiple portfolio sizes to find optimal diversification

Runs: 4, 5, 6, 7, 8 ticker portfolios
Outputs: Performance comparison + equity curves for each size
Goal: Identify optimal portfolio size for best risk-adjusted returns

Author: Portfolio Construction Team
Date: September 2025
"""

import pandas as pd
import numpy as np
import itertools
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import functions from existing scripts
import sys
sys.path.append('/mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/StrategyLab-master/analysis/portfolio_construction')

def load_base_data():
    """
    Load the foundational data from Scripts 1-2
    This data is portfolio-size agnostic
    """
    print("📊 Loading foundational data (Scripts 1-2 outputs)...")

    data_dir = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/StrategyLab-master/analysis/portfolio_construction/data"

    # Load anti-cascading trades
    trades_df = pd.read_csv(f"{data_dir}/CORRECTED_anti_cascading_trades_under2k.csv")
    trades_df['Entry Time'] = pd.to_datetime(trades_df['Entry Time'])
    trades_df['Exit Time'] = pd.to_datetime(trades_df['Exit Time'])

    # Load sector mapping
    sector_df = pd.read_csv(f"{data_dir}/CORRECTED_sector_mapping.csv")

    # Load correlation matrix
    correlation_matrix = pd.read_csv(f"{data_dir}/CORRECTED_correlation_matrix.csv", index_col=0)

    print(f"✅ Loaded {len(trades_df):,} trades from {len(sector_df)} tickers")
    print(f"✅ Correlation matrix: {correlation_matrix.shape[0]}x{correlation_matrix.shape[1]}")

    return trades_df, sector_df, correlation_matrix


def generate_combinations_for_size(valid_tickers, sector_df, correlation_matrix, portfolio_size):
    """
    Generate valid portfolio combinations for a given size
    (Adapted from Script 3)
    """
    print(f"\n{'='*80}")
    print(f"🎯 GENERATING {portfolio_size}-TICKER COMBINATIONS")
    print(f"{'='*80}")

    data_dir = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/StrategyLab-master/analysis/portfolio_construction/data"

    total_possible = len(list(itertools.combinations(valid_tickers, portfolio_size)))
    print(f"📊 Total possible {portfolio_size}-ticker combinations: {total_possible:,}")

    # Diversification filters
    max_sector_concentration = 0.6  # Max 60% from one sector
    max_correlation = 0.7  # Max pairwise correlation

    valid_combinations = []
    tested = 0
    sector_pass = 0
    correlation_pass = 0

    start_time = time.time()

    for combination in itertools.combinations(valid_tickers, portfolio_size):
        tested += 1

        # Filter 1: Sector diversification
        sectors = sector_df[sector_df['ticker'].isin(combination)]['sector'].value_counts()
        max_sector_pct = sectors.max() / portfolio_size

        if max_sector_pct > max_sector_concentration:
            continue

        sector_pass += 1

        # Filter 2: Correlation constraint
        combo_tickers = list(combination)
        max_corr = 0

        for i in range(len(combo_tickers)):
            for j in range(i+1, len(combo_tickers)):
                ticker_i = combo_tickers[i]
                ticker_j = combo_tickers[j]

                if ticker_i in correlation_matrix.index and ticker_j in correlation_matrix.columns:
                    corr = abs(correlation_matrix.loc[ticker_i, ticker_j])
                    max_corr = max(max_corr, corr)

        if max_corr > max_correlation:
            continue

        correlation_pass += 1
        valid_combinations.append(combination)

        # Progress update
        if tested % 10000 == 0:
            elapsed = time.time() - start_time
            rate = tested / elapsed if elapsed > 0 else 0
            remaining = (total_possible - tested) / rate if rate > 0 else 0
            print(f"   Progress: {tested:,}/{total_possible:,} ({tested/total_possible*100:.1f}%) | Valid: {len(valid_combinations):,} | ETA: {remaining/60:.1f} min")

    elapsed_time = time.time() - start_time

    print(f"\n📊 COMBINATION FILTERING RESULTS:")
    print(f"   Combinations tested: {tested:,}")
    print(f"   Passed sector filter: {sector_pass:,} ({sector_pass/tested*100:.1f}%)")
    print(f"   Passed correlation filter: {correlation_pass:,} ({correlation_pass/tested*100:.1f}%)")
    print(f"   Final valid combinations: {len(valid_combinations):,} ({len(valid_combinations)/tested*100:.1f}%)")
    print(f"   Processing time: {elapsed_time/60:.1f} minutes")

    # Save combinations
    if len(valid_combinations) > 0:
        combinations_data = []
        for i, combo in enumerate(valid_combinations):
            combinations_data.append({
                'combination_id': f"PCT_{portfolio_size}T_{i+1:06d}",
                'portfolio_size': portfolio_size,
                'tickers': '|'.join(combo),
                'num_sectors': len(sector_df[sector_df['ticker'].isin(combo)]['sector'].unique()),
                'max_sector_concentration': sector_df[sector_df['ticker'].isin(combo)]['sector'].value_counts().max() / portfolio_size
            })

        combinations_df = pd.DataFrame(combinations_data)
        output_file = f"{data_dir}/PERCENTAGE_valid_combinations_{portfolio_size}ticker.csv"
        combinations_df.to_csv(output_file, index=False)
        print(f"✅ Saved {len(valid_combinations):,} combinations to: {output_file}")

        return combinations_df
    else:
        print(f"❌ No valid combinations found for {portfolio_size} tickers")
        return None


def calculate_portfolio_performance_for_size(combinations_df, trades_df, portfolio_size):
    """
    Calculate portfolio performance for all combinations of given size
    (Adapted from Script 4)
    """
    print(f"\n{'='*80}")
    print(f"🎯 CALCULATING PORTFOLIO PERFORMANCE ({portfolio_size} TICKERS)")
    print(f"{'='*80}")

    data_dir = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/StrategyLab-master/analysis/portfolio_construction/data"

    # Calculate percentage returns
    trades_df = trades_df.copy()
    trades_df['percentage_return'] = ((trades_df['Exit Price'] / trades_df['Entry Price']) - 1) * 100
    trades_df['trade_date'] = trades_df['Entry Time'].dt.date

    print(f"📊 Processing {len(combinations_df):,} portfolio combinations...")

    # Process portfolios (first 10,000 for quick benchmark)
    max_portfolios = min(10000, len(combinations_df))
    print(f"💡 Evaluating first {max_portfolios:,} portfolios for benchmark")

    results = []

    for idx, row in combinations_df.head(max_portfolios).iterrows():
        ticker_list = row['tickers'].split('|')

        # Get portfolio trades
        portfolio_trades = trades_df[trades_df['ticker'].isin(ticker_list)].copy()

        if len(portfolio_trades) == 0:
            continue

        # Daily returns (average across all portfolio trades on that day)
        daily_returns = portfolio_trades.groupby('trade_date')['percentage_return'].mean()

        # Performance metrics
        total_return = daily_returns.sum()
        mean_daily_return = daily_returns.mean()
        std_daily_return = daily_returns.std()

        # Sharpe ratio (annualized)
        if std_daily_return > 0:
            sharpe_ratio = (mean_daily_return * 252) / (std_daily_return * np.sqrt(252))
        else:
            sharpe_ratio = 0

        # Profit factor
        wins = portfolio_trades[portfolio_trades['percentage_return'] > 0]['percentage_return'].sum()
        losses = abs(portfolio_trades[portfolio_trades['percentage_return'] < 0]['percentage_return'].sum())
        profit_factor = wins / losses if losses > 0 else 0

        # Win rate
        win_rate = (portfolio_trades['percentage_return'] > 0).sum() / len(portfolio_trades) * 100

        # Annualized return and volatility
        annual_return = mean_daily_return * 252
        annual_volatility = std_daily_return * np.sqrt(252)

        # Max drawdown
        cumulative = (1 + daily_returns / 100).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        max_drawdown = drawdown.min()

        results.append({
            'combination_id': row['combination_id'],
            'tickers': row['tickers'],
            'portfolio_size': portfolio_size,
            'portfolio_sharpe': sharpe_ratio,
            'portfolio_pf': profit_factor,
            'portfolio_win_rate': win_rate,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'max_drawdown': max_drawdown,
            'total_trades': len(portfolio_trades),
            'num_sectors': row['num_sectors']
        })

        if (idx + 1) % 5000 == 0:
            print(f"   Processed: {idx+1:,}/{max_portfolios:,} ({(idx+1)/max_portfolios*100:.1f}%) | Valid: {len(results):,}")

    print(f"\n✅ Portfolio performance calculation complete!")
    print(f"   Total combinations processed: {max_portfolios:,}")
    print(f"   Valid portfolios with data: {len(results):,}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Sort by Sharpe ratio
    results_df = results_df.sort_values('portfolio_sharpe', ascending=False).reset_index(drop=True)
    results_df['rank'] = results_df.index + 1

    # Save results
    all_results_file = f"{data_dir}/portfolio_performance_{portfolio_size}ticker_ALL.csv"
    results_df.to_csv(all_results_file, index=False)
    print(f"✅ All results saved: {all_results_file}")

    top_results_file = f"{data_dir}/portfolio_performance_{portfolio_size}ticker_TOP50.csv"
    results_df.head(50).to_csv(top_results_file, index=False)
    print(f"✅ Top 50 results saved: {top_results_file}")

    # Print top performers
    print(f"\n🏆 TOP 10 PORTFOLIOS ({portfolio_size} TICKERS):")
    print("-" * 120)
    print(f"{'Rank':<6} {'Sharpe':>8} {'PF':>6} {'WinRate':>8} {'Ann.Ret':>9} {'Ann.Vol':>9} {'MaxDD':>8} Tickers")
    print("-" * 120)

    for _, row in results_df.head(10).iterrows():
        print(f"{row['rank']:<6} {row['portfolio_sharpe']:>8.3f} {row['portfolio_pf']:>6.2f} {row['portfolio_win_rate']:>7.1f}% "
              f"{row['annual_return']:>8.2f}% {row['annual_volatility']:>8.2f}% {row['max_drawdown']:>7.1f}% {row['tickers']}")

    return results_df


def generate_comparison_report(all_results):
    """
    Generate comparative analysis across all portfolio sizes
    """
    print(f"\n{'='*80}")
    print(f"📊 CROSS-SIZE COMPARISON ANALYSIS")
    print(f"{'='*80}")

    data_dir = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/StrategyLab-master/analysis/portfolio_construction/data"

    comparison_data = []

    for size, results_df in all_results.items():
        if results_df is None or len(results_df) == 0:
            continue

        comparison_data.append({
            'portfolio_size': size,
            'total_evaluated': len(results_df),
            'best_sharpe': results_df['portfolio_sharpe'].max(),
            'avg_sharpe': results_df['portfolio_sharpe'].mean(),
            'median_sharpe': results_df['portfolio_sharpe'].median(),
            'best_annual_return': results_df['annual_return'].max(),
            'avg_annual_return': results_df['annual_return'].mean(),
            'best_pf': results_df['portfolio_pf'].max(),
            'avg_pf': results_df['portfolio_pf'].mean(),
            'best_win_rate': results_df['portfolio_win_rate'].max(),
            'avg_win_rate': results_df['portfolio_win_rate'].mean(),
            'best_max_drawdown': results_df['max_drawdown'].max(),  # Least negative
            'avg_max_drawdown': results_df['max_drawdown'].mean(),
        })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('portfolio_size')

    # Save comparison
    comparison_file = f"{data_dir}/portfolio_size_comparison_report.csv"
    comparison_df.to_csv(comparison_file, index=False)

    # Print comparison
    print(f"\n🎯 PORTFOLIO SIZE COMPARISON:")
    print("-" * 120)
    print(f"{'Size':<6} {'Evaluated':>10} {'Best Sharpe':>12} {'Avg Sharpe':>11} {'Best Ret%':>10} {'Avg Ret%':>9} {'Best PF':>8} {'Avg DD%':>9}")
    print("-" * 120)

    for _, row in comparison_df.iterrows():
        print(f"{row['portfolio_size']:<6} {row['total_evaluated']:>10,} {row['best_sharpe']:>12.3f} {row['avg_sharpe']:>11.3f} "
              f"{row['best_annual_return']:>9.2f}% {row['avg_annual_return']:>8.2f}% {row['best_pf']:>8.2f} {row['avg_max_drawdown']:>8.1f}%")

    # Recommendation
    print("\n" + "="*120)
    print("💡 RECOMMENDATION:")

    best_sharpe_size = comparison_df.loc[comparison_df['best_sharpe'].idxmax(), 'portfolio_size']
    best_avg_sharpe_size = comparison_df.loc[comparison_df['avg_sharpe'].idxmax(), 'portfolio_size']
    best_return_size = comparison_df.loc[comparison_df['best_annual_return'].idxmax(), 'portfolio_size']

    print(f"   🏆 Best Peak Sharpe Ratio: {best_sharpe_size}-ticker portfolios")
    print(f"   📊 Best Average Sharpe: {best_avg_sharpe_size}-ticker portfolios")
    print(f"   💰 Best Peak Return: {best_return_size}-ticker portfolios")
    print("="*120)

    return comparison_df


def main():
    """
    Master orchestration: Run portfolio optimization for sizes 4-8
    """
    print("\n" + "="*80)
    print("🚀 MASTER PORTFOLIO OPTIMIZER")
    print("="*80)
    print("📊 Goal: Find optimal portfolio size for best risk-adjusted returns")
    print("🎯 Testing portfolio sizes: 4, 5, 6, 7, 8 tickers")
    print("⏱️  Estimated total runtime: ~100 minutes")
    print("="*80)

    start_time = time.time()

    # Load base data (Scripts 1-2 outputs)
    trades_df, sector_df, correlation_matrix = load_base_data()
    valid_tickers = sector_df['ticker'].tolist()

    # Portfolio sizes to test
    portfolio_sizes = [4, 5, 6, 7, 8]

    all_results = {}

    for size in portfolio_sizes:
        size_start = time.time()

        print(f"\n\n{'#'*80}")
        print(f"# PROCESSING {size}-TICKER PORTFOLIOS")
        print(f"{'#'*80}\n")

        # Step 1: Generate combinations (Script 3 logic)
        combinations_df = generate_combinations_for_size(valid_tickers, sector_df, correlation_matrix, size)

        if combinations_df is None or len(combinations_df) == 0:
            print(f"⚠️  Skipping {size}-ticker portfolios (no valid combinations)")
            all_results[size] = None
            continue

        # Step 2: Calculate performance (Script 4 logic)
        results_df = calculate_portfolio_performance_for_size(combinations_df, trades_df, size)

        all_results[size] = results_df

        size_elapsed = time.time() - size_start
        print(f"\n✅ {size}-ticker analysis complete in {size_elapsed/60:.1f} minutes")

    # Step 3: Generate comparison report
    comparison_df = generate_comparison_report(all_results)

    total_elapsed = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"🎉 MASTER PORTFOLIO OPTIMIZATION COMPLETE!")
    print(f"⏱️  Total runtime: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.1f} hours)")
    print(f"📁 Results saved in: data/portfolio_performance_*ticker_*.csv")
    print(f"📊 Comparison report: data/portfolio_size_comparison_report.csv")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
