#!/usr/bin/env python3
"""
02_trade_type_analysis.py - Comprehensive Buy vs Sell Trade Analysis
====================================================================

Deep dive analysis comparing Buy and Sell trade performance to understand:
- Why SELL trades outperform BUY trades (51.2% vs 45.8% win rate)
- Risk-adjusted returns by direction
- Intra-trade behavior (peak/valley analysis)
- Duration and timing differences
- Ticker-specific directional biases
- Consecutive trade patterns

Author: Financial Analysis AI
Date: September 2025
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from modules.config_loader import (
    get_analysis_config,
    load_config,
    resolve_artifact_path,
    resolve_paths,
)
from modules.data_loader import load_trades

MODULE_NAME = "trade_type_analysis"

def comprehensive_directional_analysis(df: pd.DataFrame, metrics_filter: Optional[List[str]] = None):
    """Comprehensive comparison of Buy vs Sell performance"""
    print("\n" + "="*90)
    print("🔄 COMPREHENSIVE BUY vs SELL TRADE ANALYSIS")
    print("="*90)

    buy_trades = df[df['Trade Type'] == 'Buy'].copy()
    sell_trades = df[df['Trade Type'] == 'Sell'].copy()

    print(f"📊 Dataset Split:")
    print(f"  Buy Trades: {len(buy_trades):,} ({len(buy_trades)/len(df)*100:.1f}%)")
    print(f"  Sell Trades: {len(sell_trades):,} ({len(sell_trades)/len(df)*100:.1f}%)")

    # Comprehensive metrics comparison
    metrics = {}

    for trade_type, trades_df in [('Buy', buy_trades), ('Sell', sell_trades)]:
        # Basic performance
        total_pnl = trades_df['Profit (Currency)'].sum()
        win_rate = (trades_df['Profit (Currency)'] > 0).sum() / len(trades_df) * 100
        avg_pnl = trades_df['Profit (Currency)'].mean()
        median_pnl = trades_df['Profit (Currency)'].median()

        # Win/Loss breakdown
        winners = trades_df[trades_df['Profit (Currency)'] > 0]
        losers = trades_df[trades_df['Profit (Currency)'] < 0]
        avg_win = winners['Profit (Currency)'].mean() if len(winners) > 0 else 0
        avg_loss = losers['Profit (Currency)'].mean() if len(losers) > 0 else 0

        # Profit factor
        gross_profit = winners['Profit (Currency)'].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers['Profit (Currency)'].sum()) if len(losers) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Risk metrics
        max_profit = trades_df['Profit (Currency)'].max()
        max_loss = trades_df['Profit (Currency)'].min()
        std_dev = trades_df['Profit (Currency)'].std()

        # Duration and efficiency
        avg_duration = trades_df['Trade Duration (min)'].mean()
        avg_drawdown = trades_df['Drawdown (%)'].mean()

        # Percentile analysis
        p25 = trades_df['Profit (Currency)'].quantile(0.25)
        p75 = trades_df['Profit (Currency)'].quantile(0.75)

        metrics[trade_type] = {
            'count': len(trades_df),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'median_pnl': median_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'std_dev': std_dev,
            'avg_duration': avg_duration,
            'avg_drawdown': avg_drawdown,
            'p25': p25,
            'p75': p75,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }

    # Display detailed comparison
    print(f"\n{'='*50}")
    print(f"{'METRIC':<25} {'BUY':<15} {'SELL':<15} {'WINNER':<10}")
    print(f"{'='*50}")

    comparison_metrics = [
        ('Total P&L', 'total_pnl', '₹{:,.0f}'),
        ('Win Rate', 'win_rate', '{:.1f}%'),
        ('Avg P&L/Trade', 'avg_pnl', '₹{:.2f}'),
        ('Median P&L', 'median_pnl', '₹{:.2f}'),
        ('Avg Win', 'avg_win', '₹{:.2f}'),
        ('Avg Loss', 'avg_loss', '₹{:.2f}'),
        ('Profit Factor', 'profit_factor', '{:.2f}'),
        ('Max Profit', 'max_profit', '₹{:,.0f}'),
        ('Max Loss', 'max_loss', '₹{:,.0f}'),
        ('Std Deviation', 'std_dev', '₹{:.2f}'),
        ('Avg Duration', 'avg_duration', '{:.1f} min'),
        ('Avg Drawdown', 'avg_drawdown', '{:.2f}%'),
        ('25th Percentile', 'p25', '₹{:.2f}'),
        ('75th Percentile', 'p75', '₹{:.2f}')
    ]

    if metrics_filter:
        metrics_filter_set = set(metrics_filter)
        comparison_metrics = [entry for entry in comparison_metrics if entry[1] in metrics_filter_set]

    for metric_name, key, fmt in comparison_metrics:
        buy_val = metrics['Buy'][key]
        sell_val = metrics['Sell'][key]

        if key in ['win_rate', 'avg_pnl', 'profit_factor', 'avg_win', 'max_profit', 'p75', 'total_pnl']:
            winner = 'SELL' if sell_val > buy_val else 'BUY'
        elif key in ['avg_loss', 'max_loss', 'avg_drawdown', 'std_dev', 'avg_duration', 'p25']:
            winner = 'BUY' if buy_val > sell_val else 'SELL'  # Better when lower
        else:
            winner = 'SELL' if sell_val > buy_val else 'BUY'

        print(f"{metric_name:<25} {fmt.format(buy_val):<15} {fmt.format(sell_val):<15} {winner:<10}")

    return metrics, buy_trades, sell_trades

def intra_trade_analysis(buy_trades, sell_trades):
    """Analyze intra-trade behavior using High/Low during trade data"""
    print("\n" + "="*90)
    print("📈 INTRA-TRADE BEHAVIOR ANALYSIS (Peak/Valley Patterns)")
    print("="*90)

    # Buy trade analysis: Entry vs High vs Exit
    print(f"\n🟢 BUY TRADE INTRA-ANALYSIS:")
    buy_analysis = analyze_buy_intra_trade(buy_trades)

    print(f"\n🔴 SELL TRADE INTRA-ANALYSIS:")
    sell_analysis = analyze_sell_intra_trade(sell_trades)

    return buy_analysis, sell_analysis

def analyze_buy_intra_trade(buy_trades):
    """Detailed analysis of BUY trade intra-behavior"""
    # Calculate potential profits and actual profits
    buy_trades = buy_trades.copy()

    # Potential profit if exited at high
    buy_trades['potential_profit'] = buy_trades['High During Trade'] - buy_trades['Entry Price']
    buy_trades['potential_profit_pct'] = (buy_trades['potential_profit'] / buy_trades['Entry Price']) * 100

    # Actual profit
    buy_trades['actual_profit'] = buy_trades['Exit Price'] - buy_trades['Entry Price']
    buy_trades['actual_profit_pct'] = (buy_trades['actual_profit'] / buy_trades['Entry Price']) * 100

    # Profit capture efficiency
    buy_trades['profit_capture_pct'] = np.where(
        buy_trades['potential_profit'] > 0,
        (buy_trades['actual_profit'] / buy_trades['potential_profit']) * 100,
        np.nan
    )

    # Maximum adverse excursion (downside from entry)
    buy_trades['max_adverse'] = buy_trades['Entry Price'] - buy_trades['Low During Trade']
    buy_trades['max_adverse_pct'] = (buy_trades['max_adverse'] / buy_trades['Entry Price']) * 100

    # Statistics
    print(f"  Total BUY trades analyzed: {len(buy_trades):,}")
    print(f"  Average potential profit: ₹{buy_trades['potential_profit'].mean():.2f} ({buy_trades['potential_profit_pct'].mean():.2f}%)")
    print(f"  Average actual profit: ₹{buy_trades['actual_profit'].mean():.2f} ({buy_trades['actual_profit_pct'].mean():.2f}%)")
    print(f"  Average profit capture: {buy_trades['profit_capture_pct'].mean():.1f}%")
    print(f"  Average max adverse: ₹{buy_trades['max_adverse'].mean():.2f} ({buy_trades['max_adverse_pct'].mean():.2f}%)")

    # Profitable trades analysis
    profitable_buys = buy_trades[buy_trades['actual_profit'] > 0]
    if len(profitable_buys) > 0:
        print(f"\n  📊 PROFITABLE BUY TRADES ({len(profitable_buys):,}):")
        print(f"    Avg potential profit: ₹{profitable_buys['potential_profit'].mean():.2f}")
        print(f"    Avg actual profit: ₹{profitable_buys['actual_profit'].mean():.2f}")
        print(f"    Avg profit capture: {profitable_buys['profit_capture_pct'].mean():.1f}%")
        print(f"    Trades capturing >80% of peak: {(profitable_buys['profit_capture_pct'] > 80).sum():,} ({(profitable_buys['profit_capture_pct'] > 80).sum()/len(profitable_buys)*100:.1f}%)")
        print(f"    Trades capturing >60% of peak: {(profitable_buys['profit_capture_pct'] > 60).sum():,} ({(profitable_buys['profit_capture_pct'] > 60).sum()/len(profitable_buys)*100:.1f}%)")

    # Losing trades analysis
    losing_buys = buy_trades[buy_trades['actual_profit'] < 0]
    if len(losing_buys) > 0:
        print(f"\n  📉 LOSING BUY TRADES ({len(losing_buys):,}):")
        print(f"    Avg potential profit given up: ₹{losing_buys['potential_profit'].mean():.2f}")
        print(f"    Avg actual loss: ₹{losing_buys['actual_profit'].mean():.2f}")
        print(f"    Trades that were profitable at peak: {(losing_buys['potential_profit'] > 0).sum():,} ({(losing_buys['potential_profit'] > 0).sum()/len(losing_buys)*100:.1f}%)")

    return {
        'total_trades': len(buy_trades),
        'avg_potential_profit': buy_trades['potential_profit'].mean(),
        'avg_actual_profit': buy_trades['actual_profit'].mean(),
        'avg_profit_capture': buy_trades['profit_capture_pct'].mean(),
        'avg_max_adverse': buy_trades['max_adverse_pct'].mean(),
        'profitable_trades': len(profitable_buys),
        'losing_trades': len(losing_buys)
    }

def analyze_sell_intra_trade(sell_trades):
    """Detailed analysis of SELL trade intra-behavior"""
    sell_trades = sell_trades.copy()

    # Potential profit if exited at low (for sell trades, profit when price goes down)
    sell_trades['potential_profit'] = sell_trades['Entry Price'] - sell_trades['Low During Trade']
    sell_trades['potential_profit_pct'] = (sell_trades['potential_profit'] / sell_trades['Entry Price']) * 100

    # Actual profit
    sell_trades['actual_profit'] = sell_trades['Entry Price'] - sell_trades['Exit Price']
    sell_trades['actual_profit_pct'] = (sell_trades['actual_profit'] / sell_trades['Entry Price']) * 100

    # Profit capture efficiency
    sell_trades['profit_capture_pct'] = np.where(
        sell_trades['potential_profit'] > 0,
        (sell_trades['actual_profit'] / sell_trades['potential_profit']) * 100,
        np.nan
    )

    # Maximum adverse excursion (upside from entry)
    sell_trades['max_adverse'] = sell_trades['High During Trade'] - sell_trades['Entry Price']
    sell_trades['max_adverse_pct'] = (sell_trades['max_adverse'] / sell_trades['Entry Price']) * 100

    # Statistics
    print(f"  Total SELL trades analyzed: {len(sell_trades):,}")
    print(f"  Average potential profit: ₹{sell_trades['potential_profit'].mean():.2f} ({sell_trades['potential_profit_pct'].mean():.2f}%)")
    print(f"  Average actual profit: ₹{sell_trades['actual_profit'].mean():.2f} ({sell_trades['actual_profit_pct'].mean():.2f}%)")
    print(f"  Average profit capture: {sell_trades['profit_capture_pct'].mean():.1f}%")
    print(f"  Average max adverse: ₹{sell_trades['max_adverse'].mean():.2f} ({sell_trades['max_adverse_pct'].mean():.2f}%)")

    # Profitable trades analysis
    profitable_sells = sell_trades[sell_trades['actual_profit'] > 0]
    if len(profitable_sells) > 0:
        print(f"\n  📊 PROFITABLE SELL TRADES ({len(profitable_sells):,}):")
        print(f"    Avg potential profit: ₹{profitable_sells['potential_profit'].mean():.2f}")
        print(f"    Avg actual profit: ₹{profitable_sells['actual_profit'].mean():.2f}")
        print(f"    Avg profit capture: {profitable_sells['profit_capture_pct'].mean():.1f}%")
        print(f"    Trades capturing >80% of valley: {(profitable_sells['profit_capture_pct'] > 80).sum():,} ({(profitable_sells['profit_capture_pct'] > 80).sum()/len(profitable_sells)*100:.1f}%)")
        print(f"    Trades capturing >60% of valley: {(profitable_sells['profit_capture_pct'] > 60).sum():,} ({(profitable_sells['profit_capture_pct'] > 60).sum()/len(profitable_sells)*100:.1f}%)")

    # Losing trades analysis
    losing_sells = sell_trades[sell_trades['actual_profit'] < 0]
    if len(losing_sells) > 0:
        print(f"\n  📉 LOSING SELL TRADES ({len(losing_sells):,}):")
        print(f"    Avg potential profit given up: ₹{losing_sells['potential_profit'].mean():.2f}")
        print(f"    Avg actual loss: ₹{losing_sells['actual_profit'].mean():.2f}")
        print(f"    Trades that were profitable at valley: {(losing_sells['potential_profit'] > 0).sum():,} ({(losing_sells['potential_profit'] > 0).sum()/len(losing_sells)*100:.1f}%)")

    return {
        'total_trades': len(sell_trades),
        'avg_potential_profit': sell_trades['potential_profit'].mean(),
        'avg_actual_profit': sell_trades['actual_profit'].mean(),
        'avg_profit_capture': sell_trades['profit_capture_pct'].mean(),
        'avg_max_adverse': sell_trades['max_adverse_pct'].mean(),
        'profitable_trades': len(profitable_sells),
        'losing_trades': len(losing_sells)
    }

def timing_pattern_analysis(buy_trades, sell_trades):
    """Analyze timing patterns for Buy vs Sell trades"""
    print("\n" + "="*90)
    print("⏰ TIMING PATTERN ANALYSIS - BUY vs SELL")
    print("="*90)

    for trade_type, trades_df in [('BUY', buy_trades), ('SELL', sell_trades)]:
        print(f"\n📅 {trade_type} TRADE TIMING PATTERNS:")

        # Add time components
        trades_df = trades_df.copy()
        trades_df['entry_hour'] = trades_df['Entry Time'].dt.hour
        trades_df['entry_minute'] = trades_df['Entry Time'].dt.minute
        trades_df['day_of_week'] = trades_df['Entry Time'].dt.dayofweek

        # Hourly performance
        hourly_stats = trades_df.groupby('entry_hour').agg({
            'Profit (Currency)': ['sum', 'mean', 'count']
        }).round(2)
        hourly_stats.columns = ['Total_PnL', 'Avg_PnL', 'Trade_Count']

        print(f"  🕐 Top 5 Hours by Total P&L:")
        top_hours = hourly_stats.sort_values('Total_PnL', ascending=False).head(5)
        for hour, row in top_hours.iterrows():
            print(f"    {hour:02d}:00 - ₹{row['Total_PnL']:,.0f} ({row['Trade_Count']:,} trades, ₹{row['Avg_PnL']:.2f} avg)")

        # Day of week performance
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_stats = trades_df.groupby('day_of_week').agg({
            'Profit (Currency)': ['sum', 'mean', 'count']
        }).round(2)
        daily_stats.columns = ['Total_PnL', 'Avg_PnL', 'Trade_Count']
        daily_stats.index = [day_names[i] for i in daily_stats.index if i < len(day_names)]

        print(f"\n  📅 Day of Week Performance:")
        for day, row in daily_stats.iterrows():
            print(f"    {day}: ₹{row['Total_PnL']:,.0f} ({row['Trade_Count']:,} trades)")

def ticker_directional_bias_analysis(df: pd.DataFrame, min_trades: int = 100) -> pd.DataFrame:
    """Analyze which tickers favor Buy vs Sell trades"""
    print("\n" + "="*90)
    print("🏢 TICKER DIRECTIONAL BIAS ANALYSIS")
    print("="*90)

    # Calculate directional performance by ticker
    ticker_analysis = df.groupby(['ticker', 'Trade Type']).agg({
        'Profit (Currency)': ['sum', 'mean', 'count']
    }).round(2)

    ticker_analysis.columns = ['Total_PnL', 'Avg_PnL', 'Trade_Count']
    ticker_analysis = ticker_analysis.reset_index()

    # Pivot to compare Buy vs Sell by ticker
    buy_data = ticker_analysis[ticker_analysis['Trade Type'] == 'Buy'].set_index('ticker')
    sell_data = ticker_analysis[ticker_analysis['Trade Type'] == 'Sell'].set_index('ticker')

    # Merge and calculate bias
    bias_analysis = buy_data[['Total_PnL', 'Trade_Count']].join(
        sell_data[['Total_PnL', 'Trade_Count']],
        how='outer',
        lsuffix='_Buy',
        rsuffix='_Sell'
    ).fillna(0)

    bias_analysis['Total_Combined'] = bias_analysis['Total_PnL_Buy'] + bias_analysis['Total_PnL_Sell']
    bias_analysis['Buy_Contribution'] = bias_analysis['Total_PnL_Buy'] / bias_analysis['Total_Combined'] * 100
    bias_analysis['Sell_Contribution'] = bias_analysis['Total_PnL_Sell'] / bias_analysis['Total_Combined'] * 100
    bias_analysis['Total_Trades'] = bias_analysis['Trade_Count_Buy'] + bias_analysis['Trade_Count_Sell']

    # Filter tickers with meaningful trade count
    significant_tickers = bias_analysis[bias_analysis['Total_Trades'] >= min_trades].copy()

    print(f"📊 Analyzing {len(significant_tickers)} tickers with ≥{min_trades} trades")

    # Strong Buy bias tickers
    strong_buy_bias = significant_tickers[significant_tickers['Buy_Contribution'] > 70].sort_values('Total_Combined', ascending=False)
    print(f"\n🟢 TOP 10 BUY-BIASED TICKERS (>70% profits from Buy trades):")
    for i, (ticker, row) in enumerate(strong_buy_bias.head(10).iterrows(), 1):
        print(f"  {i:2d}. {ticker}: ₹{row['Total_Combined']:,.0f} total ({row['Buy_Contribution']:.1f}% from BUY)")

    # Strong Sell bias tickers
    strong_sell_bias = significant_tickers[significant_tickers['Sell_Contribution'] > 70].sort_values('Total_Combined', ascending=False)
    print(f"\n🔴 TOP 10 SELL-BIASED TICKERS (>70% profits from Sell trades):")
    for i, (ticker, row) in enumerate(strong_sell_bias.head(10).iterrows(), 1):
        print(f"  {i:2d}. {ticker}: ₹{row['Total_Combined']:,.0f} total ({row['Sell_Contribution']:.1f}% from SELL)")

    # Balanced performers
    balanced_tickers = significant_tickers[
        (significant_tickers['Buy_Contribution'] >= 40) &
        (significant_tickers['Buy_Contribution'] <= 60)
    ].sort_values('Total_Combined', ascending=False)

    print(f"\n⚖️ TOP 10 BALANCED TICKERS (40-60% split):")
    for i, (ticker, row) in enumerate(balanced_tickers.head(10).iterrows(), 1):
        print(f"  {i:2d}. {ticker}: ₹{row['Total_Combined']:,.0f} total (Buy: {row['Buy_Contribution']:.1f}%, Sell: {row['Sell_Contribution']:.1f}%)")

    return significant_tickers

def save_detailed_results(
    config: Dict[str, Any],
    metrics: Dict[str, Dict[str, float]],
    buy_analysis: Dict[str, float],
    sell_analysis: Dict[str, float],
    ticker_bias: pd.DataFrame,
    category: str = 'generic'
) -> None:
    """Save comprehensive analysis results to configured locations."""
    comparison_df = pd.DataFrame(metrics).T
    csv_path = Path(resolve_artifact_path(config, MODULE_NAME, 'directional_breakdown', category=category))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(csv_path)
    print(f"💾 Saved directional comparison CSV → {csv_path}")

    summary = {
        'analysis_date': datetime.now().isoformat(),
        'buy_vs_sell_metrics': metrics,
        'buy_intra_trade_analysis': buy_analysis,
        'sell_intra_trade_analysis': sell_analysis,
        'key_takeaways': {
            'sell_win_rate_advantage': metrics['Sell']['win_rate'] - metrics['Buy']['win_rate'],
            'sell_profit_factor_advantage': metrics['Sell']['profit_factor'] - metrics['Buy']['profit_factor'],
            'sell_efficiency_advantage': metrics['Sell']['avg_pnl'] - metrics['Buy']['avg_pnl'],
            'sell_drawdown_advantage': metrics['Buy']['avg_drawdown'] - metrics['Sell']['avg_drawdown'],
        },
    }
    json_path = csv_path.with_suffix('.json')
    json_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"💾 Saved JSON summary → {json_path}")

    markdown_path = Path(resolve_artifact_path(
        config,
        MODULE_NAME,
        'directional_summary',
        category=category,
        artifact_type='markdown'
    ))
    try:
        kpi_table = comparison_df.to_markdown()
    except ImportError:
        kpi_table = comparison_df.to_string()

    markdown_content = [
        "# Trade Type Analysis Summary",
        "",
        f"- **Timestamp**: {summary['analysis_date']}",
        f"- **Sell Win Rate Advantage**: {summary['key_takeaways']['sell_win_rate_advantage']:.2f}%",
        f"- **Sell Profit Factor Advantage**: {summary['key_takeaways']['sell_profit_factor_advantage']:.2f}",
        f"- **Sell Efficiency Advantage (₹)**: {summary['key_takeaways']['sell_efficiency_advantage']:.2f}",
        f"- **Sell Drawdown Advantage (%)**: {summary['key_takeaways']['sell_drawdown_advantage']:.2f}",
        "",
        "## KPI Table",
        kpi_table,
    ]
    markdown_path.write_text("\n".join(markdown_content))
    print(f"📝 Saved Markdown summary → {markdown_path}")

    bias_path = Path(resolve_artifact_path(
        config,
        MODULE_NAME,
        'ticker_bias',
        category=category,
        artifact_type='csv'
    ))
    ticker_bias.to_csv(bias_path)
    print(f"💾 Saved ticker bias breakdown → {bias_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Comprehensive Buy vs Sell Trade Analysis")
    parser.add_argument("--config", required=True, help="Path to analysis YAML config")
    parser.add_argument("--sample", type=int, help="Optional sample size for quick runs")
    args = parser.parse_args()

    print("🚀 Starting Comprehensive Buy vs Sell Analysis...")
    print("=" * 90)

    config = load_config(args.config)
    paths = resolve_paths(config)
    module_cfg = get_analysis_config(config, MODULE_NAME) or {}

    sample_size = args.sample or module_cfg.get('sample_size')
    trades_df = load_trades(config, paths, sample_size=sample_size)

    metrics_filter = module_cfg.get('metrics') if isinstance(module_cfg.get('metrics'), list) else None
    metrics, buy_trades, sell_trades = comprehensive_directional_analysis(trades_df, metrics_filter=metrics_filter)
    buy_analysis, sell_analysis = intra_trade_analysis(buy_trades, sell_trades)
    if module_cfg.get('include_consecutive_patterns', True):
        timing_pattern_analysis(buy_trades, sell_trades)
    min_trades_bias = module_cfg.get('min_trades_for_bias', 100)
    ticker_bias = ticker_directional_bias_analysis(trades_df, min_trades=min_trades_bias)

    save_detailed_results(config, metrics, buy_analysis, sell_analysis, ticker_bias)

    print("\n" + "=" * 90)
    print("✅ COMPREHENSIVE BUY vs SELL ANALYSIS COMPLETE!")
    print("=" * 90)
    print("🎯 Key Findings:")
    print(f"  • SELL trades have {metrics['Sell']['win_rate'] - metrics['Buy']['win_rate']:.1f}% higher win rate")
    print(f"  • SELL trades capture {sell_analysis['avg_profit_capture']:.1f}% of valley potential")
    print(f"  • BUY trades capture {buy_analysis['avg_profit_capture']:.1f}% of peak potential")
    print(f"  • SELL trades have {metrics['Buy']['avg_drawdown'] - metrics['Sell']['avg_drawdown']:.2f}% lower drawdown")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
