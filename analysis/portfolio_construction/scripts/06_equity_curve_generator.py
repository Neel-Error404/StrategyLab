#!/usr/bin/env python3
"""
EQUITY CURVE & VISUALIZATION GENERATOR (Config-Driven)
======================================================

Purpose: Create comprehensive visualizations for top portfolios
- Cumulative return curves
- Drawdown analysis
- Monthly return heatmaps
- Portfolio comparison charts
- Sector allocation analysis

Input: Top portfolios from portfolio_optimizer + trade data
Output: PNG charts and summary statistics

Author: Portfolio Construction Team
Version: 2.0 - Config-Driven (migrated October 2025)
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# Set plot style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Equity Curve & Visualization Generator')
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

    # Use the correct Profit (%) column from backtest data (handles SHORT trades correctly)
    if 'Profit (%)' not in trades_df.columns:
        raise ValueError("Missing 'Profit (%)' column in trades data")

    return trades_df


def load_sector_mapping(config):
    """Load sector mapping from sector classification module"""
    sector_output_dir = Path(get_output_dir(config, 'sector_classification', category='portfolio'))
    sector_file = sector_output_dir / "sector_mapping.csv"

    if not sector_file.exists():
        raise FileNotFoundError(f"Sector mapping not found: {sector_file}")

    return pd.read_csv(sector_file)


def load_top_portfolios(config, portfolio_size):
    """Load top portfolios from portfolio optimizer module"""
    optimizer_output_dir = Path(get_output_dir(config, 'portfolio_optimizer', category='portfolio'))
    top_file = optimizer_output_dir / "portfolio_performance_top50.csv"

    if not top_file.exists():
        raise FileNotFoundError(f"Top portfolios not found: {top_file}")

    portfolios_df = pd.read_csv(top_file)

    # Filter for specified portfolio size if requested
    if portfolio_size:
        portfolios_df = portfolios_df[portfolios_df['portfolio_size'] == portfolio_size]

    return portfolios_df


def load_portfolio_trades(portfolio_tickers, trades_df):
    """Filter trades for a given portfolio"""
    portfolio_trades = trades_df[trades_df['ticker'].isin(portfolio_tickers)].copy()
    return portfolio_trades.sort_values('Entry Time')


def calculate_equity_curve(trades_df, initial_capital=100000):
    """
    Calculate cumulative equity curve from trades
    Assumes equal-weight allocation (1/N per ticker)
    """
    # Group trades by entry date
    daily_trades = trades_df.copy()
    daily_trades['Entry Date'] = daily_trades['Entry Time'].dt.date

    # Calculate daily portfolio returns (average across all trades that day)
    daily_returns = daily_trades.groupby('Entry Date')['Profit (%)'].mean() / 100

    # Calculate cumulative returns
    cumulative_returns = (1 + daily_returns).cumprod()
    equity_curve = initial_capital * cumulative_returns

    # Create time series
    equity_series = pd.Series(equity_curve.values, index=pd.to_datetime(equity_curve.index))

    return equity_series, daily_returns


def calculate_drawdown(equity_curve):
    """Calculate drawdown series from equity curve"""
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max * 100
    return drawdown


def plot_equity_curves(portfolios_data, title, output_dir, output_filename):
    """Plot cumulative equity curves for multiple portfolios"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Top panel: Equity curves
    for name, equity_curve in portfolios_data.items():
        ax1.plot(equity_curve.index, equity_curve.values, label=name, linewidth=2)

    ax1.set_title(f'{title} - Cumulative Equity Curves', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Portfolio Value (₹)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=100000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/1000:.0f}K'))

    # Bottom panel: Drawdowns
    for name, equity_curve in portfolios_data.items():
        drawdown = calculate_drawdown(equity_curve)
        ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3)
        ax2.plot(drawdown.index, drawdown.values, linewidth=1.5)

    ax2.set_title('Underwater Equity (Drawdown)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    plt.tight_layout()
    plt.savefig(output_dir / output_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {output_filename}")


def plot_monthly_returns_heatmap(trades_df, title, output_dir, output_filename):
    """Create monthly returns heatmap"""
    # Calculate monthly returns
    trades_df['YearMonth'] = trades_df['Entry Time'].dt.to_period('M')
    monthly_returns = trades_df.groupby('YearMonth')['Profit (%)'].mean()

    # Create year-month matrix
    monthly_returns_df = monthly_returns.reset_index()
    monthly_returns_df['Year'] = monthly_returns_df['YearMonth'].dt.year
    monthly_returns_df['Month'] = monthly_returns_df['YearMonth'].dt.month

    pivot_table = monthly_returns_df.pivot(index='Year', columns='Month', values='Profit (%)')

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Avg Return (%)'}, ax=ax, linewidths=0.5)

    ax.set_title(f'{title} - Monthly Average Returns Heatmap', fontsize=16, fontweight='bold')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Year', fontsize=12)

    # Month labels
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticklabels(month_labels)

    plt.tight_layout()
    plt.savefig(output_dir / output_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {output_filename}")


def plot_rolling_sharpe(equity_curve, daily_returns, window, title, output_dir, output_filename):
    """Plot rolling Sharpe ratio"""
    # Calculate rolling Sharpe (annualized)
    rolling_return = daily_returns.rolling(window).mean() * 252
    rolling_vol = daily_returns.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = rolling_return / rolling_vol

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2, color='darkblue')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Sharpe = 1.0')
    ax.axhline(y=2, color='darkgreen', linestyle='--', alpha=0.5, label='Sharpe = 2.0')

    ax.set_title(f'{title} - Rolling Sharpe Ratio ({window} days)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / output_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {output_filename}")


def plot_sector_allocation(portfolio_tickers, portfolio_name, sector_mapping, output_dir):
    """Plot sector allocation pie chart for a portfolio"""
    # Get sectors for portfolio tickers
    portfolio_sectors = sector_mapping[sector_mapping['ticker'].isin(portfolio_tickers)]
    sector_counts = portfolio_sectors['sector'].value_counts()

    # Plot pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = sns.color_palette('Set3', n_colors=len(sector_counts))
    wedges, texts, autotexts = ax.pie(sector_counts.values, labels=sector_counts.index,
                                        autopct='%1.1f%%', startangle=90, colors=colors,
                                        textprops={'fontsize': 10})

    ax.set_title(f'{portfolio_name} - Sector Allocation', fontsize=16, fontweight='bold')

    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    plt.tight_layout()
    filename = f"sector_allocation_{portfolio_name.replace(' ', '_').replace('#', '')}.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {filename}")


def generate_summary_stats(portfolios_dict, output_dir, output_filename="portfolio_summary_stats.csv"):
    """Generate summary statistics table"""
    stats_list = []

    for name, equity_curve in portfolios_dict.items():
        # Calculate metrics
        daily_returns = equity_curve.pct_change().dropna()

        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
        annualized_return = ((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (252 / len(equity_curve)) - 1) * 100
        annualized_vol = daily_returns.std() * np.sqrt(252) * 100
        sharpe_ratio = (annualized_return / annualized_vol) if annualized_vol > 0 else 0

        drawdown = calculate_drawdown(equity_curve)
        max_drawdown = drawdown.min()

        win_rate = (daily_returns > 0).sum() / len(daily_returns) * 100

        stats_list.append({
            'Portfolio': name,
            'Total Return (%)': round(total_return, 2),
            'Annualized Return (%)': round(annualized_return, 2),
            'Annualized Volatility (%)': round(annualized_vol, 2),
            'Sharpe Ratio': round(sharpe_ratio, 3),
            'Max Drawdown (%)': round(max_drawdown, 2),
            'Win Rate (%)': round(win_rate, 2),
            'Final Value (₹)': round(equity_curve.iloc[-1], 0)
        })

    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv(output_dir / output_filename, index=False)

    print(f"\n✅ Summary Statistics:")
    print(stats_df.to_string(index=False))
    print(f"\n✅ Saved: {output_filename}\n")

    return stats_df


def process_portfolio_size(config, trades_df, sector_mapping, portfolio_size, top_n, output_dir):
    """Process visualizations for a specific portfolio size"""

    print(f"\n{'='*80}")
    print(f"📊 PROCESSING {portfolio_size}-TICKER PORTFOLIOS")
    print(f"{'='*80}")

    # Load top portfolios
    portfolios_df = load_top_portfolios(config, portfolio_size)

    if len(portfolios_df) == 0:
        print(f"⚠️  No portfolios found for size {portfolio_size}")
        return None

    # Limit to top N
    portfolios_df = portfolios_df.head(top_n)

    print(f"📈 Processing Top {top_n} portfolios from {portfolio_size}-ticker results...")

    # Generate equity curves
    equity_curves = {}

    for idx in range(min(top_n, len(portfolios_df))):
        row = portfolios_df.iloc[idx]

        # Extract tickers
        ticker_list_str = row.get('ticker_list') or row.get('tickers', '')
        if '|' in ticker_list_str:
            tickers = ticker_list_str.split('|')
        elif ',' in ticker_list_str:
            tickers = [t.strip() for t in ticker_list_str.split(',')]
        else:
            tickers = [ticker_list_str]

        portfolio_name = f"{portfolio_size}T-#{idx+1} (Sharpe {row['portfolio_sharpe']:.2f})"

        print(f"\nProcessing: {portfolio_name}")
        print(f"  Tickers: {', '.join(tickers)}")

        # Get trades for this portfolio
        portfolio_trades = load_portfolio_trades(tickers, trades_df)
        equity_curve, daily_returns = calculate_equity_curve(portfolio_trades)
        equity_curves[portfolio_name] = equity_curve

        # Individual visualizations for top portfolio only
        if idx == 0:
            plot_monthly_returns_heatmap(
                portfolio_trades, portfolio_name, output_dir,
                f'monthly_returns_{portfolio_size}T_top1.png'
            )
            plot_rolling_sharpe(
                equity_curve, daily_returns, window=63,
                title=portfolio_name, output_dir=output_dir,
                output_filename=f'rolling_sharpe_{portfolio_size}T_top1.png'
            )
            plot_sector_allocation(tickers, portfolio_name, sector_mapping, output_dir)

    # Combined equity curve plot
    plot_equity_curves(
        equity_curves,
        f"Top {top_n} {portfolio_size}-Ticker Portfolios",
        output_dir,
        f'equity_curves_{portfolio_size}ticker.png'
    )

    return equity_curves


def main():
    """Execute equity curve and visualization generation"""

    # Parse arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    paths = resolve_paths(config)
    module_config = get_module_spec(config, 'equity_curves', category='portfolio')

    print("🚀 STARTING EQUITY CURVE & VISUALIZATION GENERATOR")
    print("=" * 80)
    print(f"📁 Config: {args.config}")
    print(f"📊 Strategy: {config['run']['strategy']}")
    print(f"📅 Date Range: {config['run']['date_range']}")
    print("=" * 80)

    try:
        # Get configuration parameters
        cfg = module_config.get('config', {})
        portfolio_sizes = cfg.get('portfolio_sizes', [5, 6, 7])
        top_n = cfg.get('top_n', 5)

        print(f"📊 Portfolio sizes: {portfolio_sizes}")
        print(f"📊 Top N per size: {top_n}")

        # Get output directory
        output_dir = Path(get_output_dir(config, 'equity_curves', category='portfolio'))
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"📁 Output directory: {output_dir}")

        # Load data
        print("\n📂 Loading data...")
        trades_df = load_anti_cascading_trades(config)
        sector_mapping = load_sector_mapping(config)

        print(f"✅ Loaded {len(trades_df):,} trades")
        print(f"✅ Loaded sector mapping for {len(sector_mapping)} tickers")

        # Process each portfolio size
        all_equity_curves = {}

        for size in portfolio_sizes:
            try:
                equity_curves = process_portfolio_size(
                    config, trades_df, sector_mapping, size, top_n, output_dir
                )

                if equity_curves:
                    all_equity_curves.update(equity_curves)

            except Exception as e:
                print(f"\n❌ Error processing {size}-ticker portfolios: {e}")
                import traceback
                traceback.print_exc()

        # Generate combined summary statistics
        if all_equity_curves:
            print(f"\n📊 Generating summary statistics...")
            generate_summary_stats(all_equity_curves, output_dir, "portfolio_summary_stats.csv")

        print("\n" + "=" * 80)
        print(f"✅ VISUALIZATION COMPLETE")
        print(f"📁 All visualizations saved to: {output_dir.absolute()}")
        print("=" * 80)

        return {
            'output_dir': str(output_dir),
            'portfolio_sizes': portfolio_sizes,
            'charts_generated': len(list(output_dir.glob('*.png')))
        }

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = main()
