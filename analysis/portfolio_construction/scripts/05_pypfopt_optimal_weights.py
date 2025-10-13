#!/usr/bin/env python3
"""
PYPFOPT OPTIMAL WEIGHT ALLOCATION (Config-Driven)
==================================================

Purpose: Calculate optimal portfolio weights using Mean-Variance Optimization (Markowitz)
- Compares Equal-weight (1/N) vs Optimized allocations
- Methods: Max Sharpe Ratio, Min Volatility, Efficient Risk
- Uses PyPortfolioOpt library for optimization

Input: Top portfolios from portfolio_optimizer module + anti-cascading trades
Output: Optimal weights for each portfolio and optimization method

Author: Portfolio Construction Team
Version: 2.0 - Config-Driven (migrated October 2025)
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PORTFOLIO_DIR = SCRIPT_DIR.parent
ANALYSIS_DIR = PORTFOLIO_DIR.parent
sys.path.insert(0, str(ANALYSIS_DIR))

from generic.modules.config_loader import load_config, resolve_paths, get_output_dir, get_module_spec

# PyPortfolioOpt imports
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    from pypfopt import objective_functions
    PYPFOPT_AVAILABLE = True
except ImportError:
    print("⚠️  PyPortfolioOpt not available. Install with: pip install PyPortfolioOpt")
    PYPFOPT_AVAILABLE = False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='PyPortfolioOpt Optimal Weight Allocation')
    parser.add_argument('--config', required=True, help='Path to YAML configuration file')
    return parser.parse_args()


def load_anti_cascading_trades(config):
    """Load anti-cascading trades from previous module"""
    filter_output_dir = Path(get_output_dir(config, 'anti_cascade_filter', category='portfolio'))
    trades_file = filter_output_dir / "anti_cascading_trades_filtered.csv"

    if not trades_file.exists():
        raise FileNotFoundError(f"Anti-cascading trades not found: {trades_file}")

    trades_df = pd.read_csv(trades_file)
    trades_df['Entry Time'] = pd.to_datetime(trades_df['Entry Time'])
    trades_df['Exit Time'] = pd.to_datetime(trades_df['Exit Time'])

    return trades_df


def load_top_portfolios(config, portfolio_size):
    """Load top portfolios from portfolio optimizer module"""
    optimizer_output_dir = Path(get_output_dir(config, 'portfolio_optimizer', category='portfolio'))

    # Try loading top 50 file
    top_file = optimizer_output_dir / "portfolio_performance_top50.csv"

    if not top_file.exists():
        raise FileNotFoundError(f"Top portfolios not found: {top_file}")

    portfolios_df = pd.read_csv(top_file)

    # Filter for specified portfolio size
    if portfolio_size:
        portfolios_df = portfolios_df[portfolios_df['portfolio_size'] == portfolio_size]

    return portfolios_df


def create_returns_matrix(trades_df, tickers):
    """
    Create returns matrix for PyPortfolioOpt

    PyPortfolioOpt expects a DataFrame with:
    - Index: Dates (time periods)
    - Columns: Asset tickers
    - Values: Returns for each asset in each period
    """

    # Ensure Entry Time is datetime
    if not pd.api.types.is_datetime64_any_dtype(trades_df['Entry Time']):
        trades_df['Entry Time'] = pd.to_datetime(trades_df['Entry Time'])

    trades_df = trades_df.sort_values('Entry Time')

    # Calculate percentage returns if not already present
    if 'percentage_return' not in trades_df.columns:
        trades_df['percentage_return'] = (
            (trades_df['Exit Price'] / trades_df['Entry Price'] - 1) * 100
        )

    # Create daily returns matrix
    returns_dict = {}

    for ticker in tickers:
        ticker_trades = trades_df[trades_df['ticker'] == ticker].copy()
        ticker_trades = ticker_trades.set_index('Entry Time')

        # Use percentage returns
        ticker_returns = ticker_trades[['percentage_return']].copy()
        ticker_returns.columns = [ticker]

        # Resample to daily frequency
        ticker_returns = ticker_returns.resample('D').mean()

        returns_dict[ticker] = ticker_returns[ticker]

    # Combine all ticker returns
    returns_df = pd.DataFrame(returns_dict)

    # Drop days where any ticker has NaN
    returns_df = returns_df.dropna()

    # Convert from percentage to decimal (PyPortfolioOpt expects decimal)
    returns_df = returns_df / 100

    return returns_df


def calculate_equal_weight_metrics(tickers, returns_df):
    """Calculate portfolio metrics with equal weights (1/N)"""
    n = len(tickers)
    equal_weights = {ticker: 1/n for ticker in tickers}

    # Calculate portfolio returns
    portfolio_returns = (returns_df * pd.Series(equal_weights)).sum(axis=1)

    # Calculate metrics (annualized)
    annual_return = portfolio_returns.mean() * 252
    annual_volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0

    return {
        'method': 'Equal Weight (1/N)',
        'weights': equal_weights,
        'expected_annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio
    }


def calculate_max_sharpe_weights(returns_df, gamma=0.1):
    """Calculate optimal weights using Max Sharpe Ratio"""

    # Calculate expected returns and covariance matrix
    mu = expected_returns.mean_historical_return(returns_df, frequency=252)
    S = risk_models.sample_cov(returns_df, frequency=252)

    # Optimize for maximum Sharpe ratio
    ef = EfficientFrontier(mu, S)

    # Add L2 regularization to prevent extreme weights
    ef.add_objective(objective_functions.L2_reg, gamma=gamma)

    # Get optimal weights
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    # Get portfolio performance
    performance = ef.portfolio_performance(verbose=False)

    return {
        'method': 'Max Sharpe Ratio',
        'weights': cleaned_weights,
        'expected_annual_return': performance[0],
        'annual_volatility': performance[1],
        'sharpe_ratio': performance[2]
    }


def calculate_min_volatility_weights(returns_df):
    """Calculate optimal weights using Minimum Volatility"""

    # Calculate expected returns and covariance matrix
    mu = expected_returns.mean_historical_return(returns_df, frequency=252)
    S = risk_models.sample_cov(returns_df, frequency=252)

    # Optimize for minimum volatility
    ef = EfficientFrontier(mu, S)

    # Get optimal weights
    weights = ef.min_volatility()
    cleaned_weights = ef.clean_weights()

    # Get portfolio performance
    performance = ef.portfolio_performance(verbose=False)

    return {
        'method': 'Min Volatility',
        'weights': cleaned_weights,
        'expected_annual_return': performance[0],
        'annual_volatility': performance[1],
        'sharpe_ratio': performance[2]
    }


def calculate_efficient_risk_weights(returns_df, target_volatility=0.15):
    """Calculate optimal weights for a target volatility level"""

    # Calculate expected returns and covariance matrix
    mu = expected_returns.mean_historical_return(returns_df, frequency=252)
    S = risk_models.sample_cov(returns_df, frequency=252)

    # Optimize for efficient risk
    ef = EfficientFrontier(mu, S)

    try:
        # Get optimal weights for target volatility
        weights = ef.efficient_risk(target_volatility)
        cleaned_weights = ef.clean_weights()

        # Get portfolio performance
        performance = ef.portfolio_performance(verbose=False)

        return {
            'method': f'Efficient Risk (σ={target_volatility})',
            'weights': cleaned_weights,
            'expected_annual_return': performance[0],
            'annual_volatility': performance[1],
            'sharpe_ratio': performance[2]
        }
    except:
        # If target volatility is unachievable, return None
        return None


def optimize_portfolio(portfolio_tickers, trades_df, config_params):
    """
    Optimize weights for a given portfolio using multiple methods

    Returns:
        List of optimization results (one per method)
    """

    # Filter trades for portfolio tickers
    portfolio_trades = trades_df[trades_df['ticker'].isin(portfolio_tickers)].copy()

    # Create returns matrix
    returns_df = create_returns_matrix(portfolio_trades, portfolio_tickers)

    if len(returns_df) < 30:
        print(f"⚠️  Insufficient data: {len(returns_df)} days")
        return []

    results = []

    # Method 1: Equal Weight (baseline)
    results.append(calculate_equal_weight_metrics(portfolio_tickers, returns_df))

    # Method 2: Max Sharpe Ratio
    try:
        gamma = config_params.get('l2_regularization', 0.1)
        results.append(calculate_max_sharpe_weights(returns_df, gamma=gamma))
    except Exception as e:
        print(f"⚠️  Max Sharpe failed: {e}")

    # Method 3: Min Volatility
    try:
        results.append(calculate_min_volatility_weights(returns_df))
    except Exception as e:
        print(f"⚠️  Min Volatility failed: {e}")

    # Method 4: Efficient Risk
    try:
        target_vol = config_params.get('target_volatility', 0.15)
        result = calculate_efficient_risk_weights(returns_df, target_volatility=target_vol)
        if result:
            results.append(result)
    except Exception as e:
        print(f"⚠️  Efficient Risk failed: {e}")

    return results


def process_portfolios(config, trades_df, portfolios_df, portfolio_size):
    """Process all portfolios and calculate optimal weights"""

    print(f"\n{'='*80}")
    print(f"📊 PROCESSING {portfolio_size}-TICKER PORTFOLIOS")
    print(f"{'='*80}")
    print(f"Total portfolios to optimize: {len(portfolios_df)}")

    # Get optimization config parameters
    module_config = get_module_spec(config, 'pypfopt_weights', category='portfolio')
    config_params = module_config.get('config', {})

    all_results = []

    for idx, row in portfolios_df.iterrows():
        # Extract tickers
        ticker_list_str = row.get('ticker_list') or row.get('tickers', '')

        # Handle both pipe-separated and comma-separated formats
        if '|' in ticker_list_str:
            tickers = ticker_list_str.split('|')
        elif ',' in ticker_list_str:
            tickers = [t.strip() for t in ticker_list_str.split(',')]
        else:
            tickers = [ticker_list_str]

        portfolio_id = f"Portfolio_{idx+1}"
        print(f"\n{portfolio_id}: {', '.join(tickers)}")

        # Optimize weights
        optimization_results = optimize_portfolio(tickers, trades_df, config_params)

        # Store results
        for result in optimization_results:
            result_row = {
                'portfolio_id': portfolio_id,
                'portfolio_size': portfolio_size,
                'tickers': ', '.join(tickers),
                'ticker_list': '|'.join(tickers),
                'optimization_method': result['method'],
                'expected_annual_return': result['expected_annual_return'],
                'annual_volatility': result['annual_volatility'],
                'sharpe_ratio': result['sharpe_ratio'],
                'original_sharpe': row.get('portfolio_sharpe', 0)
            }

            # Add individual ticker weights
            for ticker, weight in result['weights'].items():
                result_row[f'weight_{ticker}'] = weight

            all_results.append(result_row)

            print(f"  {result['method']:30s} → Sharpe: {result['sharpe_ratio']:.4f}, "
                  f"Return: {result['expected_annual_return']:.2%}, Vol: {result['annual_volatility']:.2%}")

    return pd.DataFrame(all_results)


def analyze_optimization_results(results_df, portfolio_size):
    """Generate summary analysis of optimization results"""

    if len(results_df) == 0:
        print(f"\n⚠️  No optimization results generated")
        return

    print(f"\n{'='*80}")
    print(f"📈 OPTIMIZATION SUMMARY ({portfolio_size} tickers)")
    print(f"{'='*80}")

    for method in results_df['optimization_method'].unique():
        method_df = results_df[results_df['optimization_method'] == method]
        avg_sharpe = method_df['sharpe_ratio'].mean()
        avg_return = method_df['expected_annual_return'].mean()
        avg_vol = method_df['annual_volatility'].mean()

        print(f"\n{method}:")
        print(f"  Avg Sharpe: {avg_sharpe:.4f}")
        print(f"  Avg Return: {avg_return:.2%}")
        print(f"  Avg Volatility: {avg_vol:.2%}")

    # Compare with equal-weight baseline
    baseline_df = results_df[results_df['optimization_method'] == 'Equal Weight (1/N)']
    if len(baseline_df) > 0:
        print(f"\n{'='*80}")
        print(f"💡 IMPROVEMENT OVER EQUAL WEIGHT")
        print(f"{'='*80}")

        baseline_sharpe = baseline_df['sharpe_ratio'].mean()

        for method in results_df['optimization_method'].unique():
            if method == 'Equal Weight (1/N)':
                continue

            method_sharpe = results_df[results_df['optimization_method'] == method]['sharpe_ratio'].mean()
            improvement = ((method_sharpe / baseline_sharpe) - 1) * 100

            print(f"{method:30s} → {improvement:+.2f}% Sharpe improvement")


def save_optimization_results(config, results_df, portfolio_size):
    """Save optimization results to config-specified directory"""

    print(f"\n💾 SAVING OPTIMIZATION RESULTS")
    print("=" * 60)

    # Get output directory
    output_dir = Path(get_output_dir(config, 'pypfopt_weights', category='portfolio'))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save optimal weights
    output_file = output_dir / f"optimal_weights_{portfolio_size}ticker.csv"
    results_df.to_csv(output_file, index=False)
    print(f"✅ Optimal weights saved: {output_file.name}")

    # Save summary report
    summary_file = output_dir / f"pypfopt_summary_{portfolio_size}ticker.md"
    with open(summary_file, 'w') as f:
        f.write(f"# PYPFOPT OPTIMIZATION SUMMARY - {portfolio_size} Tickers\n\n")
        f.write(f"**Optimization Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Portfolios Optimized:** {len(results_df['portfolio_id'].unique())}\n\n")

        f.write("## OPTIMIZATION METHODS\n\n")
        for method in results_df['optimization_method'].unique():
            method_df = results_df[results_df['optimization_method'] == method]
            f.write(f"### {method}\n")
            f.write(f"- **Portfolios:** {len(method_df)}\n")
            f.write(f"- **Avg Sharpe:** {method_df['sharpe_ratio'].mean():.4f}\n")
            f.write(f"- **Avg Return:** {method_df['expected_annual_return'].mean():.2%}\n")
            f.write(f"- **Avg Volatility:** {method_df['annual_volatility'].mean():.2%}\n\n")

        # Top 10 best performers
        f.write("## TOP 10 OPTIMIZED PORTFOLIOS (BY SHARPE)\n\n")
        f.write("| Rank | Portfolio | Method | Sharpe | Return | Vol | Tickers |\n")
        f.write("|------|-----------|--------|--------|--------|-----|-------|\n")

        top_10 = results_df.nlargest(10, 'sharpe_ratio')
        for rank, (_, row) in enumerate(top_10.iterrows(), 1):
            f.write(f"| {rank} | {row['portfolio_id']} | {row['optimization_method']} | "
                   f"{row['sharpe_ratio']:.4f} | {row['expected_annual_return']:.2%} | "
                   f"{row['annual_volatility']:.2%} | {row['tickers']} |\n")

    print(f"✅ Summary report saved: {summary_file.name}")
    print(f"📁 Location: {output_dir}")

    return str(output_file)


def main():
    """Execute PyPortfolioOpt optimization"""

    if not PYPFOPT_AVAILABLE:
        print("❌ PyPortfolioOpt library not available. Install with: pip install PyPortfolioOpt")
        return None

    # Parse arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    paths = resolve_paths(config)
    module_config = get_module_spec(config, 'pypfopt_weights', category='portfolio')

    print("🚀 STARTING PYPFOPT OPTIMAL WEIGHT ALLOCATION")
    print("=" * 80)
    print(f"📁 Config: {args.config}")
    print(f"📊 Strategy: {config['run']['strategy']}")
    print(f"📅 Date Range: {config['run']['date_range']}")
    print("=" * 80)

    try:
        # Get configuration parameters
        cfg = module_config.get('config', {})
        portfolio_sizes = cfg.get('portfolio_sizes', [5, 6, 7])  # Default to 5, 6, 7
        top_n = cfg.get('top_n', 50)  # How many portfolios to optimize per size

        print(f"📊 Portfolio sizes to optimize: {portfolio_sizes}")
        print(f"📊 Top N portfolios per size: {top_n}")

        # Load anti-cascading trades
        print("\n📂 Loading anti-cascading trade data...")
        trades_df = load_anti_cascading_trades(config)
        print(f"✅ Loaded {len(trades_df):,} anti-cascading trades")

        # Process each portfolio size
        all_results = []

        for size in portfolio_sizes:
            try:
                print(f"\n{'='*80}")
                print(f"Processing {size}-ticker portfolios...")
                print(f"{'='*80}")

                # Load top portfolios for this size
                portfolios_df = load_top_portfolios(config, size)

                # Limit to top N
                portfolios_df = portfolios_df.head(top_n)

                if len(portfolios_df) == 0:
                    print(f"⚠️  No portfolios found for size {size}")
                    continue

                print(f"✅ Loaded {len(portfolios_df)} portfolios for optimization")

                # Process and optimize
                results_df = process_portfolios(config, trades_df, portfolios_df, size)

                # Analyze results
                analyze_optimization_results(results_df, size)

                # Save results
                save_optimization_results(config, results_df, size)

                all_results.append(results_df)

            except Exception as e:
                print(f"\n❌ Error processing {size}-ticker portfolios: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n🎉 PYPFOPT OPTIMIZATION COMPLETED!")
        print(f"📊 Processed {len(portfolio_sizes)} portfolio sizes")

        return {
            'results': pd.concat(all_results, ignore_index=True) if all_results else None,
            'portfolio_sizes': portfolio_sizes
        }

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = main()
