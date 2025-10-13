#!/usr/bin/env python3
"""
Basic Exploratory Data Analysis (EDA)
======================================

Comprehensive analysis of trade data to understand overall performance,
patterns, and characteristics.

✅ GENERIC - Works with ANY strategy's trade data
❌ No strategy-specific logic required

**Analysis Performed**:
1. Overall Statistics (win rate, profit factor, total P&L)
2. Trade Distribution (Buy vs Sell, duration, timing)
3. Profitability Patterns (win/loss distribution, outliers)
4. Temporal Analysis (time-of-day, day-of-week patterns)
5. Ticker-Level Summary (performance by ticker)

**Usage**:
    python 01_basic_eda.py --config ../config.yaml

**Outputs**:
    - output/basic_eda_statistics.json
    - output/basic_eda_summary.csv
    - reports/BASIC_EDA_REPORT.md

Author: StrategyLab Team
Version: 2.0 - YAML Config Driven
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import argparse
import json
from datetime import datetime
from typing import Any, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from modules.config_loader import (
    get_analysis_config,
    load_config,
    resolve_artifact_path,
    resolve_paths,
)
from modules.data_loader import load_trades, validate_trade_data

MODULE_NAME = "basic_eda"

def calculate_overall_statistics(trades_df: pd.DataFrame) -> dict:
    """Calculate overall performance statistics."""

    print("\n📊 OVERALL STATISTICS")
    print("=" * 60)

    total_trades = len(trades_df)
    winning_trades = (trades_df['Profit (Currency)'] > 0).sum()
    losing_trades = (trades_df['Profit (Currency)'] < 0).sum()
    breakeven_trades = (trades_df['Profit (Currency)'] == 0).sum()

    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    total_profit = trades_df['Profit (Currency)'].sum()
    avg_profit = trades_df['Profit (Currency)'].mean()

    gross_profit = trades_df[trades_df['Profit (Currency)'] > 0]['Profit (Currency)'].sum()
    gross_loss = abs(trades_df[trades_df['Profit (Currency)'] < 0]['Profit (Currency)'].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf

    avg_win = trades_df[trades_df['Profit (Currency)'] > 0]['Profit (Currency)'].mean()
    avg_loss = trades_df[trades_df['Profit (Currency)'] < 0]['Profit (Currency)'].mean()

    avg_duration = trades_df['Trade Duration (min)'].mean()

    stats = {
        'total_trades': int(total_trades),
        'winning_trades': int(winning_trades),
        'losing_trades': int(losing_trades),
        'breakeven_trades': int(breakeven_trades),
        'win_rate': float(win_rate),
        'total_profit': float(total_profit),
        'avg_profit': float(avg_profit),
        'gross_profit': float(gross_profit),
        'gross_loss': float(gross_loss),
        'profit_factor': float(profit_factor) if profit_factor != np.inf else None,
        'avg_win': float(avg_win) if not pd.isna(avg_win) else None,
        'avg_loss': float(avg_loss) if not pd.isna(avg_loss) else None,
        'avg_duration_minutes': float(avg_duration)
    }

    # Print summary
    print(f"Total Trades: {total_trades:,}")
    print(f"  Winning: {winning_trades:,} ({win_rate:.1f}%)")
    print(f"  Losing: {losing_trades:,} ({100-win_rate:.1f}%)")
    print(f"  Breakeven: {breakeven_trades:,}")
    print(f"\nProfitability:")
    print(f"  Total P&L: ₹{total_profit:,.2f}")
    print(f"  Average P&L: ₹{avg_profit:.2f}")
    print(f"  Profit Factor: {profit_factor:.2f}" if profit_factor != np.inf else "  Profit Factor: ∞ (no losses)")
    print(f"  Avg Win: ₹{avg_win:.2f}" if not pd.isna(avg_win) else "  Avg Win: N/A")
    print(f"  Avg Loss: ₹{avg_loss:.2f}" if not pd.isna(avg_loss) else "  Avg Loss: N/A")
    print(f"\nDuration:")
    print(f"  Average: {avg_duration:.1f} minutes ({avg_duration/60:.1f} hours)")

    return stats

def analyze_trade_distribution(trades_df: pd.DataFrame) -> dict:
    """Analyze trade type distribution (Buy vs Sell)."""

    print("\n📊 TRADE TYPE DISTRIBUTION")
    print("=" * 60)

    buy_trades = trades_df[trades_df['Trade Type'] == 'Buy']
    sell_trades = trades_df[trades_df['Trade Type'] == 'Sell']

    buy_stats = {
        'count': int(len(buy_trades)),
        'win_rate': float((buy_trades['Profit (Currency)'] > 0).mean() * 100),
        'total_profit': float(buy_trades['Profit (Currency)'].sum()),
        'avg_profit': float(buy_trades['Profit (Currency)'].mean())
    }

    sell_stats = {
        'count': int(len(sell_trades)),
        'win_rate': float((sell_trades['Profit (Currency)'] > 0).mean() * 100),
        'total_profit': float(sell_trades['Profit (Currency)'].sum()),
        'avg_profit': float(sell_trades['Profit (Currency)'].mean())
    }

    print(f"Buy Trades:")
    print(f"  Count: {buy_stats['count']:,} ({buy_stats['count']/len(trades_df)*100:.1f}%)")
    print(f"  Win Rate: {buy_stats['win_rate']:.1f}%")
    print(f"  Total Profit: ₹{buy_stats['total_profit']:,.2f}")
    print(f"  Avg Profit: ₹{buy_stats['avg_profit']:.2f}")

    print(f"\nSell Trades:")
    print(f"  Count: {sell_stats['count']:,} ({sell_stats['count']/len(trades_df)*100:.1f}%)")
    print(f"  Win Rate: {sell_stats['win_rate']:.1f}%")
    print(f"  Total Profit: ₹{sell_stats['total_profit']:,.2f}")
    print(f"  Avg Profit: ₹{sell_stats['avg_profit']:.2f}")

    return {'buy': buy_stats, 'sell': sell_stats}

def analyze_ticker_performance(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance by ticker."""

    print("\n📊 TICKER-LEVEL PERFORMANCE")
    print("=" * 60)

    ticker_stats = trades_df.groupby('ticker').agg({
        'Profit (Currency)': ['count', 'sum', 'mean'],
        'Trade Duration (min)': 'mean'
    }).round(2)

    # Calculate win rate per ticker
    win_rates = trades_df.groupby('ticker').apply(
        lambda x: (x['Profit (Currency)'] > 0).sum() / len(x) * 100
    )

    ticker_stats['win_rate'] = win_rates

    ticker_stats.columns = ['trade_count', 'total_profit', 'avg_profit', 'avg_duration', 'win_rate']
    ticker_stats = ticker_stats.sort_values('total_profit', ascending=False)

    print(f"\nTop 10 Tickers by Total Profit:")
    print(ticker_stats.head(10).to_string())

    print(f"\nBottom 5 Tickers by Total Profit:")
    print(ticker_stats.tail(5).to_string())

    return ticker_stats


def analyze_time_of_day(trades_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Analyze performance by entry hour."""

    if 'Entry Time' not in trades_df.columns:
        print("⚠️ 'Entry Time' column not found. Skipping time-of-day analysis.")
        return None

    print("\n📅 TIME-OF-DAY PERFORMANCE")
    print("=" * 60)

    df = trades_df[['Entry Time', 'Profit (Currency)']].copy()
    df['entry_hour'] = df['Entry Time'].dt.hour
    hourly = df.groupby('entry_hour').agg(
        trade_count=pd.NamedAgg(column='Profit (Currency)', aggfunc='count'),
        avg_profit=pd.NamedAgg(column='Profit (Currency)', aggfunc='mean'),
        win_rate=pd.NamedAgg(column='Profit (Currency)', aggfunc=lambda x: (x > 0).mean() * 100),
    ).round(2)
    hourly = hourly.sort_index()

    print("Top 5 Hours by Win Rate:")
    for hour, row in hourly.sort_values('win_rate', ascending=False).head(5).iterrows():
        print(f"  {int(hour):02d}:00 → {row['win_rate']:.1f}% win rate (₹{row['avg_profit']:.2f} avg)")

    return hourly

def generate_report(
    config: dict,
    module_cfg: dict,
    stats: dict,
    distribution: dict,
    ticker_stats: Optional[pd.DataFrame],
    time_of_day: Optional[pd.DataFrame]
) -> None:
    report_path = Path(resolve_artifact_path(config, MODULE_NAME, 'report', artifact_type='markdown'))

    run_cfg = config.get('run', {})
    include_tickers = module_cfg.get('include_ticker_breakdown', True) and ticker_stats is not None
    include_time = module_cfg.get('include_time_of_day_stats', True) and time_of_day is not None

    lines = [
        "# Basic EDA Report",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Run ID**: {run_cfg.get('run_id', 'N/A')}",
        f"**Strategy**: {run_cfg.get('strategy', 'N/A')}",
    ]
    if run_cfg.get('date_range'):
        lines.append(f"**Date Range**: {run_cfg['date_range']}")
    lines.append("\n---\n")

    lines.append("## Overall Performance\n")
    lines.append(f"- **Total Trades**: {stats['total_trades']:,}")
    lines.append(f"- **Win Rate**: {stats['win_rate']:.1f}%")
    lines.append(f"- **Total P&L**: ₹{stats['total_profit']:,.2f}")
    if stats['profit_factor']:
        lines.append(f"- **Profit Factor**: {stats['profit_factor']:.2f}")
    else:
        lines.append("- **Profit Factor**: ∞ (no losses)")
    lines.append(f"- **Avg Trade Duration**: {stats['avg_duration_minutes']:.1f} minutes\n")

    lines.append("## Trade Distribution\n")
    lines.append(f"- **Buy Trades**: {distribution['buy']['count']:,} ({distribution['buy']['win_rate']:.1f}% WR)")
    lines.append(f"- **Sell Trades**: {distribution['sell']['count']:,} ({distribution['sell']['win_rate']:.1f}% WR)\n")

    if include_tickers:
        lines.append("## Top Performing Tickers\n")
        lines.append("| Ticker | Trades | Total Profit | Win Rate | Avg Profit |")
        lines.append("|--------|--------|--------------|----------|------------|")
        for ticker, row in ticker_stats.head(10).iterrows():
            lines.append(
                f"| {ticker} | {int(row['trade_count']):,} | ₹{row['total_profit']:,.2f} | {row['win_rate']:.1f}% | ₹{row['avg_profit']:.2f} |"
            )
        lines.append("")

    if include_time and time_of_day is not None:
        lines.append("## Time-of-Day Performance\n")
        lines.append("| Hour | Trades | Win Rate | Avg P&L |")
        lines.append("|------|--------|----------|----------|")
        for hour, row in time_of_day.iterrows():
            lines.append(
                f"| {int(hour):02d}:00 | {int(row['trade_count']):,} | {row['win_rate']:.1f}% | ₹{row['avg_profit']:.2f} |"
            )
        lines.append("")

    lines.append("## Key Insights\n")
    if stats['win_rate'] > 50:
        lines.append("- ✅ **Win rate above 50%** – strategy shows positive edge")
    else:
        lines.append("- ⚠️ **Win rate below 50%** – consider refining entry/exit rules")

    if stats['profit_factor'] and stats['profit_factor'] > 1.5:
        lines.append("- ✅ **Strong profit factor** – wins significantly outweigh losses")
    elif stats['profit_factor'] and stats['profit_factor'] < 1.2:
        lines.append("- ⚠️ **Low profit factor** – small edge, consider optimization")

    lines.append("\n---\n")
    lines.append("**Next Steps**: Run cascade analysis and ticker ranking for deeper insights.")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    print(f"\n📄 Report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Basic EDA - Exploratory Data Analysis"
    )
    parser.add_argument(
        '--config',
        required=True,
        help="Path to YAML config file"
    )
    parser.add_argument(
        '--sample',
        type=int,
        help="Sample N trades for faster testing"
    )

    args = parser.parse_args()

    print("="*60)
    print("BASIC EDA - Exploratory Data Analysis")
    print("="*60)

    # Load configuration
    config = load_config(args.config)
    paths = resolve_paths(config)
    module_cfg = get_analysis_config(config, MODULE_NAME) or {}

    # Load trade data
    sample_size = args.sample or module_cfg.get('sample_size')
    trades_df = load_trades(config, paths, sample_size=sample_size)

    # Validate data
    print("\n🔍 Validating trade data...")
    validation = validate_trade_data(trades_df)
    if not validation['valid']:
        print(f"❌ Data validation failed: {validation['errors']}")
        return 1

    # Perform analyses
    overall_stats = calculate_overall_statistics(trades_df)
    distribution = analyze_trade_distribution(trades_df)
    ticker_stats = None
    if module_cfg.get('include_ticker_breakdown', True):
        ticker_stats = analyze_ticker_performance(trades_df)

    time_of_day_stats = None
    if module_cfg.get('include_time_of_day_stats', True):
        time_of_day_stats = analyze_time_of_day(trades_df)

    # Save statistics JSON
    summary_path = Path(resolve_artifact_path(config, MODULE_NAME, 'summary', artifact_type='json'))
    summary = {
        'overall': overall_stats,
        'distribution': distribution,
    }
    if ticker_stats is not None:
        summary['top_tickers'] = (
            ticker_stats.head(20)
            .reset_index()
            .replace({np.nan: None})
            .to_dict(orient='records')
        )
    if time_of_day_stats is not None:
        summary['time_of_day'] = (
            time_of_day_stats.reset_index()
            .replace({np.nan: None})
            .to_dict(orient='records')
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n💾 Saved statistics to: {summary_path}")

    # Save ticker stats CSV if enabled
    if ticker_stats is not None:
        ticker_path = Path(resolve_artifact_path(config, MODULE_NAME, 'ticker_performance', artifact_type='csv'))
        ticker_stats.to_csv(ticker_path)
        print(f"💾 Saved ticker performance to: {ticker_path}")

    # Generate report
    generate_report(config, module_cfg, overall_stats, distribution, ticker_stats, time_of_day_stats)

    print("\n" + "="*60)
    print("✅ BASIC EDA COMPLETE!")
    print("="*60)

    return 0

if __name__ == "__main__":
    exit(main())
