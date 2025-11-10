#!/usr/bin/env python3
"""
PORTFOLIO OPTIMIZATION ENGINE (Config-Driven)
============================================

Purpose: Evaluate all valid portfolio combinations and identify best performers
- Calculate portfolio-level performance metrics for each combination
- Rank portfolios by risk-adjusted returns (Sharpe ratio)
- Identify top N performers for further analysis

Input: Valid combinations + filtered trades from previous steps
Output: Ranked portfolio performance metrics

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
    parser = argparse.ArgumentParser(description='Portfolio Optimization Engine')
    parser.add_argument('--config', required=True, help='Path to YAML configuration file')
    return parser.parse_args()


def load_combinations_and_trades(config):
    """
    Load valid combinations and trade data from previous modules
    """

    print("🚀 PORTFOLIO OPTIMIZATION ENGINE")
    print("=" * 80)
    print("📊 Loading valid combinations and trade data...")

    # Get combination generator output directory
    combo_output_dir = Path(get_output_dir(config, 'combination_generator', category='portfolio'))

    # Load combinations for all portfolio sizes
    all_combinations = []
    portfolio_sizes = []

    for size in [4, 5, 6, 7, 8]:  # Check all possible sizes
        combo_file = combo_output_dir / f"valid_combinations_{size}ticker.csv"
        if combo_file.exists():
            combo_df = pd.read_csv(combo_file)
            combo_df['portfolio_size'] = size
            all_combinations.append(combo_df)
            portfolio_sizes.append(size)
            print(f"✅ Loaded {len(combo_df):,} valid {size}-ticker combinations")

    if not all_combinations:
        raise FileNotFoundError(f"No combination files found in {combo_output_dir}")

    # Combine all sizes
    combinations_df = pd.concat(all_combinations, ignore_index=True)
    print(f"\n✅ Total combinations to evaluate: {len(combinations_df):,}")
    print(f"✅ Portfolio sizes: {portfolio_sizes}")

    # Get anti-cascading trades
    filter_output_dir = Path(get_output_dir(config, 'anti_cascade_filter', category='portfolio'))
    trades_file = filter_output_dir / "anti_cascading_trades_filtered.csv"

    trades_df = pd.read_csv(trades_file)
    trades_df['Entry Time'] = pd.to_datetime(trades_df['Entry Time'])
    trades_df['Exit Time'] = pd.to_datetime(trades_df['Exit Time'])

    print(f"✅ Loaded {len(trades_df):,} anti-cascading trades")
    print(f"✅ Date range: {trades_df['Entry Time'].min().date()} to {trades_df['Exit Time'].max().date()}")

    return combinations_df, trades_df


def calculate_portfolio_performance(ticker_list, trades_df):
    """
    Calculate portfolio-level performance metrics
    Equal-weight allocation: 1/N per ticker
    """

    # Filter trades for this portfolio's tickers
    portfolio_trades = trades_df[trades_df['ticker'].isin(ticker_list)].copy()

    if len(portfolio_trades) < 10:
        return None

    # Use the correct Profit (%) column from backtest data (handles SHORT trades correctly)
    # DO NOT recalculate - the backtest already computed this properly
    if 'Profit (%)' not in portfolio_trades.columns:
        raise ValueError("Missing 'Profit (%)' column in trades data")

    # Group by date to get daily portfolio returns
    portfolio_trades['trade_date'] = pd.to_datetime(portfolio_trades['Entry Time']).dt.date

    # Equal-weight portfolio: average returns across all tickers each day
    daily_returns = portfolio_trades.groupby('trade_date')['Profit (%)'].mean()

    if len(daily_returns) < 10 or daily_returns.std() == 0:
        return None

    # Calculate portfolio-level metrics
    avg_daily_return = daily_returns.mean()
    daily_volatility = daily_returns.std()

    # Portfolio Sharpe ratio (annualized with risk-free rate)
    rf_rate = 0.065  # 6.5% annual risk-free rate (India)
    annual_return = avg_daily_return * 252
    annual_volatility = daily_volatility * np.sqrt(252)
    portfolio_sharpe = (annual_return - rf_rate * 100) / annual_volatility if annual_volatility > 0 else 0

    # Portfolio profit factor
    winning_days = daily_returns[daily_returns > 0]
    losing_days = daily_returns[daily_returns < 0]
    portfolio_pf = abs(winning_days.sum() / losing_days.sum()) if len(losing_days) > 0 else float('inf')

    # Portfolio win rate
    portfolio_win_rate = len(winning_days) / len(daily_returns) if len(daily_returns) > 0 else 0

    # Total return (cumulative)
    cumulative_return = (1 + daily_returns / 100).prod() - 1
    total_return_pct = cumulative_return * 100

    # Maximum drawdown
    cumulative_returns = (1 + daily_returns / 100).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    return {
        'portfolio_sharpe': portfolio_sharpe,
        'portfolio_pf': portfolio_pf,
        'portfolio_win_rate': portfolio_win_rate * 100,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'total_return_pct': total_return_pct,
        'max_drawdown_pct': max_drawdown,
        'trading_days': len(daily_returns),
        'total_trades': len(portfolio_trades)
    }


def optimize_portfolios(combinations_df, trades_df, max_portfolios=None):
    """
    Calculate performance for all portfolio combinations and rank them
    """

    print(f"\n🎯 PORTFOLIO OPTIMIZATION IN PROGRESS")
    print("=" * 80)
    print(f"💡 Evaluating portfolio-level performance for all combinations")

    if max_portfolios:
        print(f"⚡ Quick mode: Evaluating first {max_portfolios:,} portfolios")
        combinations_df = combinations_df.head(max_portfolios)

    portfolio_results = []
    total_combinations = len(combinations_df)

    print(f"\n📊 Processing {total_combinations:,} portfolio combinations...")

    for idx, row in combinations_df.iterrows():
        # Extract ticker list from pipe-separated string
        ticker_list = row['tickers'].split('|')

        # Calculate portfolio performance
        performance = calculate_portfolio_performance(ticker_list, trades_df)

        if performance:
            result = {
                'combination_id': row.get('combination_id', idx),
                'portfolio_size': row.get('portfolio_size', len(ticker_list)),
                'tickers': ', '.join(ticker_list),
                'ticker_list': '|'.join(ticker_list),  # Keep pipe-separated for later use
                **performance
            }
            portfolio_results.append(result)

        # Progress reporting
        if (idx + 1) % 1000 == 0:
            progress = (idx + 1) / total_combinations * 100
            print(f"   Processed: {idx + 1:,}/{total_combinations:,} ({progress:.1f}%) | Valid: {len(portfolio_results):,}")

    print(f"\n✅ Portfolio optimization complete!")
    print(f"   Total combinations processed: {total_combinations:,}")
    print(f"   Valid portfolios with performance data: {len(portfolio_results):,}")

    results_df = pd.DataFrame(portfolio_results)

    return results_df


def analyze_top_performers(results_df, top_n=50):
    """
    Analyze and display top performing portfolios
    """

    print(f"\n🏆 TOP {top_n} PORTFOLIO PERFORMERS")
    print("=" * 80)
    print(f"📊 Ranked by Portfolio-Level Sharpe Ratio")

    # Sort by Sharpe ratio (descending)
    top_portfolios = results_df.nlargest(top_n, 'portfolio_sharpe')

    print(f"\n📈 TOP {top_n} PORTFOLIOS BY SHARPE RATIO:")
    print("-" * 130)
    print(f"{'Rank':<6} {'Size':<6} {'Sharpe':<8} {'PF':<6} {'WinRate':<9} {'Ann.Ret':<10} {'Ann.Vol':<10} {'MaxDD':<8} {'Tickers':<50}")
    print("-" * 130)

    for rank, (i, row) in enumerate(top_portfolios.head(top_n).iterrows(), 1):
        print(f"{rank:<6} {row['portfolio_size']:<6} {row['portfolio_sharpe']:>7.3f} {row['portfolio_pf']:>5.2f} {row['portfolio_win_rate']:>8.1f}% "
              f"{row['annual_return']:>9.2f}% {row['annual_volatility']:>9.2f}% {row['max_drawdown_pct']:>7.1f}% "
              f"{row['tickers']:<50}")

    # Summary statistics
    print(f"\n📊 PERFORMANCE DISTRIBUTION SUMMARY:")
    print(f"   Sharpe Ratio   - Mean: {results_df['portfolio_sharpe'].mean():.3f} | Median: {results_df['portfolio_sharpe'].median():.3f} | Max: {results_df['portfolio_sharpe'].max():.3f}")
    print(f"   Annual Return  - Mean: {results_df['annual_return'].mean():.2f}% | Median: {results_df['annual_return'].median():.2f}% | Max: {results_df['annual_return'].max():.2f}%")
    print(f"   Annual Vol     - Mean: {results_df['annual_volatility'].mean():.2f}% | Median: {results_df['annual_volatility'].median():.2f}%")
    print(f"   Max Drawdown   - Mean: {results_df['max_drawdown_pct'].mean():.2f}% | Median: {results_df['max_drawdown_pct'].median():.2f}%")

    # Best performers by size
    print(f"\n🎯 BEST PORTFOLIO BY SIZE:")
    for size in sorted(results_df['portfolio_size'].unique()):
        size_portfolios = results_df[results_df['portfolio_size'] == size]
        if len(size_portfolios) > 0:
            best = size_portfolios.nlargest(1, 'portfolio_sharpe').iloc[0]
            print(f"   {size}-ticker: Sharpe={best['portfolio_sharpe']:.3f} | {best['tickers']}")

    return top_portfolios


def save_optimization_results(config, results_df, top_portfolios):
    """
    Save optimization results to config-specified directory
    """

    print(f"\n💾 SAVING OPTIMIZATION RESULTS")
    print("=" * 60)

    # Get output directory
    output_dir = Path(get_output_dir(config, 'portfolio_optimizer', category='portfolio'))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save all portfolio performance results
    all_results_file = output_dir / "portfolio_performance_all.csv"
    results_df.to_csv(all_results_file, index=False)
    print(f"✅ All portfolio results saved: {all_results_file.name}")

    # Save top N performers
    top_n_file = output_dir / "portfolio_performance_top50.csv"
    top_portfolios.to_csv(top_n_file, index=False)
    print(f"✅ Top 50 portfolios saved: {top_n_file.name}")

    # Save summary report
    summary_file = output_dir / "portfolio_optimization_summary.md"
    with open(summary_file, 'w') as f:
        f.write("# PORTFOLIO OPTIMIZATION SUMMARY\n\n")
        f.write(f"**Optimization Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Portfolios Evaluated:** {len(results_df):,}\n")
        f.write(f"**Portfolio Sizes:** {sorted(results_df['portfolio_size'].unique())}\n\n")

        f.write("## PERFORMANCE STATISTICS\n\n")
        f.write(f"- **Sharpe Ratio Range:** {results_df['portfolio_sharpe'].min():.3f} to {results_df['portfolio_sharpe'].max():.3f}\n")
        f.write(f"- **Annual Return Range:** {results_df['annual_return'].min():.2f}% to {results_df['annual_return'].max():.2f}%\n")
        f.write(f"- **Annual Volatility Range:** {results_df['annual_volatility'].min():.2f}% to {results_df['annual_volatility'].max():.2f}%\n\n")

        f.write("## TOP 10 PORTFOLIOS\n\n")
        f.write("| Rank | Size | Sharpe | Ann. Return | Ann. Vol | Max DD | Tickers |\n")
        f.write("|------|------|--------|-------------|----------|--------|----------|\n")
        for rank, (_, row) in enumerate(top_portfolios.head(10).iterrows(), 1):
            f.write(f"| {rank} | {row['portfolio_size']} | {row['portfolio_sharpe']:.3f} | {row['annual_return']:.2f}% | "
                   f"{row['annual_volatility']:.2f}% | {row['max_drawdown_pct']:.2f}% | {row['tickers']} |\n")

    print(f"✅ Optimization summary saved: {summary_file.name}")
    print(f"📁 Location: {output_dir}")

    print(f"\n🎉 PORTFOLIO OPTIMIZATION COMPLETED!")
    print(f"📊 {len(results_df):,} portfolios evaluated")
    print(f"🏆 Top 50 best performers identified")

    return str(all_results_file), str(top_n_file)


def main():
    """
    Execute the complete portfolio optimization
    """

    # Parse arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    paths = resolve_paths(config)
    module_config = get_module_spec(config, 'portfolio_optimizer', category='portfolio')

    print("🚀 STARTING PORTFOLIO OPTIMIZATION ENGINE")
    print("=" * 80)
    print(f"📁 Config: {args.config}")
    print(f"📊 Strategy: {config['run']['strategy']}")
    print(f"📅 Date Range: {config['run']['date_range']}")
    print("=" * 80)

    try:
        # Get configuration parameters
        cfg = module_config.get('config', {})
        top_n = cfg.get('top_n', 50)
        max_portfolios = cfg.get('max_portfolios', None)  # For quick testing

        # Load combinations and trade data
        combinations_df, trades_df = load_combinations_and_trades(config)

        # Optimize portfolios
        results_df = optimize_portfolios(combinations_df, trades_df, max_portfolios)

        # Analyze top performers
        top_portfolios = analyze_top_performers(results_df, top_n)

        # Save results
        all_results_file, top_n_file = save_optimization_results(config, results_df, top_portfolios)

        print(f"\n🎯 Next: PyPortfolioOpt Weight Optimization & Equity Curve Generation")

        return {
            'results_df': results_df,
            'top_portfolios': top_portfolios,
            'all_results_file': all_results_file,
            'top_n_file': top_n_file
        }

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = main()
