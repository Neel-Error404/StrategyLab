#!/usr/bin/env python3
"""
COMPREHENSIVE CASCADE vs ANTI-CASCADE ANALYSIS (Config-Driven)
==============================================================

Purpose: Foundation analysis for portfolio construction
- Analyze ALL trades to understand true performance differences
- Segregate into: All Trades, Cascading Trades, Anti-Cascading Trades
- Generate separate Top 50 performer lists for each category
- Compare performance differences to validate portfolio selection basis

Key Question: Are Top 50 performers from "All Trades" the same as
Top 50 performers from "Anti-Cascading Trades"?

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
from generic.modules.data_loader import load_trades, validate_trade_data


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Foundation Cascade vs Anti-Cascade Analysis')
    parser.add_argument('--config', required=True, help='Path to YAML configuration file')
    return parser.parse_args()


def load_and_tag_all_trades(trades_df):
    """
    Tag trades with cascade characteristics
    Returns DataFrame with cascade tags
    """

    print("🔄 COMPREHENSIVE CASCADE vs ANTI-CASCADE ANALYSIS")
    print("=" * 70)
    print("🎯 Objective: Compare Top 50 performers across trade categories")
    print("=" * 70)

    print("\n📊 STEP 1: TAGGING TRADES WITH CASCADE PATTERNS")
    print("=" * 50)

    print(f"✅ Loaded {len(trades_df):,} total trades")
    print(f"✅ Date range: {trades_df['Entry Time'].min().date()} to {trades_df['Entry Time'].max().date()}")
    print(f"✅ Unique tickers: {trades_df['ticker'].nunique()}")

    # Ensure datetime types
    trades_df['Entry Time'] = pd.to_datetime(trades_df['Entry Time'])
    trades_df['Exit Time'] = pd.to_datetime(trades_df['Exit Time'])
    trades_df['Entry Date'] = trades_df['Entry Time'].dt.date

    # Sort by ticker and time for cascade analysis
    trades_sorted = trades_df.sort_values(['ticker', 'Entry Time']).reset_index(drop=True)

    # Create previous trade reference columns
    trades_sorted['prev_ticker'] = trades_sorted['ticker'].shift(1)
    trades_sorted['prev_entry_date'] = trades_sorted['Entry Date'].shift(1)
    trades_sorted['prev_trade_type'] = trades_sorted['Trade Type'].shift(1)

    # Tag each trade with cascade characteristics
    print("🏷️  Tagging trades with cascade characteristics...")

    cascade_tags = []
    for _, trade in trades_sorted.iterrows():
        if pd.isna(trade['prev_ticker']):
            tag = 'FIRST_TRADE_OVERALL'
        elif trade['ticker'] != trade['prev_ticker']:
            tag = 'FIRST_TRADE_FOR_TICKER'
        elif trade['Entry Date'] != trade['prev_entry_date']:
            tag = 'FIRST_TRADE_OF_DAY'
        elif trade['Trade Type'] == trade['prev_trade_type']:
            tag = 'CONSECUTIVE_SAME_DIRECTION'  # CASCADING
        else:
            tag = 'CONSECUTIVE_OPPOSITE_DIRECTION'  # ANTI-CASCADING

        cascade_tags.append(tag)

    trades_sorted['cascade_tag'] = cascade_tags

    # Categorize trades into main categories
    trades_sorted['trade_category'] = trades_sorted['cascade_tag'].apply(lambda x:
        'CASCADING' if x == 'CONSECUTIVE_SAME_DIRECTION' else 'ANTI_CASCADING'
    )

    # Show tagging results
    tag_counts = trades_sorted['cascade_tag'].value_counts()
    print(f"\n📊 TRADE TAGGING RESULTS:")
    for tag, count in tag_counts.items():
        percentage = (count / len(trades_sorted)) * 100
        category = "🔄 CASCADING" if tag == 'CONSECUTIVE_SAME_DIRECTION' else "✅ ANTI-CASCADING"
        print(f"   {tag:30} | {count:8,} ({percentage:5.1f}%) | {category}")

    return trades_sorted


def calculate_ticker_performance_by_category(trades_df, category_filter=None, category_name="ALL"):
    """
    Calculate comprehensive ticker performance metrics for a specific trade category
    Returns DataFrame with performance metrics per ticker
    """

    print(f"\n📈 STEP 2{['A', 'B', 'C'][['ALL', 'CASCADING', 'ANTI_CASCADING'].index(category_name)]}: CALCULATING PERFORMANCE - {category_name} TRADES")
    print("=" * 60)

    # Filter trades by category
    if category_filter:
        filtered_trades = trades_df[trades_df['trade_category'] == category_filter].copy()
    else:
        filtered_trades = trades_df.copy()

    print(f"📊 Analyzing {len(filtered_trades):,} trades in {category_name} category")

    # Calculate ticker-wise performance metrics
    ticker_performance = {}

    for ticker in filtered_trades['ticker'].unique():
        ticker_trades = filtered_trades[filtered_trades['ticker'] == ticker].copy()

        if len(ticker_trades) < 10:  # Minimum trade threshold
            continue

        # PERCENTAGE-BASED METHODOLOGY (Industry Standard)
        ticker_trades['percentage_return'] = (
            (ticker_trades['Exit Price'] / ticker_trades['Entry Price'] - 1) * 100
        )

        # Basic metrics
        total_trades = len(ticker_trades)
        total_pnl = ticker_trades['Profit (Currency)'].sum()
        avg_return_pct = ticker_trades['percentage_return'].mean()

        # Risk metrics - PERCENTAGE-BASED
        winning_trades = ticker_trades[ticker_trades['percentage_return'] > 0]
        losing_trades = ticker_trades[ticker_trades['percentage_return'] < 0]

        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        # Profit factor from percentage returns
        profit_factor = abs(winning_trades['percentage_return'].sum() /
                           losing_trades['percentage_return'].sum()) if len(losing_trades) > 0 else float('inf')

        # Risk-reward ratio from percentage returns
        avg_win = winning_trades['percentage_return'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['percentage_return'].mean() if len(losing_trades) > 0 else 0
        risk_reward_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        # Sharpe-like ratio from percentage returns
        returns = ticker_trades['percentage_return']
        sharpe_like_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0

        # Store metrics
        ticker_performance[ticker] = {
            'ticker': ticker,
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'avg_return_pct': avg_return_pct,
            'win_rate': win_rate * 100,
            'profit_factor': profit_factor,
            'risk_reward_ratio': risk_reward_ratio,
            'sharpe_like_ratio': sharpe_like_ratio,
        }

    # Convert to DataFrame and rank
    performance_df = pd.DataFrame.from_dict(ticker_performance, orient='index')

    # Composite scoring (weighted average of key metrics)
    performance_df['composite_score'] = (
        performance_df['profit_factor'] * 0.4 +
        performance_df['win_rate'] * 0.3 / 100 +  # Normalize to 0-1
        performance_df['sharpe_like_ratio'] * 0.3
    )

    # Rank tickers
    performance_df = performance_df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    performance_df['rank'] = range(1, len(performance_df) + 1)

    print(f"✅ Calculated performance for {len(performance_df)} tickers")
    print(f"   Top 3: {', '.join(performance_df.head(3)['ticker'].tolist())}")

    return performance_df


def compare_top50_across_categories(all_performance, cascading_performance, anticascading_performance):
    """
    Compare Top 50 performers across different trade categories
    Returns comparison statistics
    """

    print(f"\n🔍 STEP 3: COMPARING TOP 50 PERFORMERS ACROSS CATEGORIES")
    print("=" * 60)

    # Extract Top 50 from each category
    top50_all = set(all_performance.head(50)['ticker'].tolist())
    top50_cascading = set(cascading_performance.head(50)['ticker'].tolist())
    top50_anticascading = set(anticascading_performance.head(50)['ticker'].tolist())

    # Calculate overlaps
    all_vs_cascade_overlap = len(top50_all & top50_cascading)
    all_vs_anti_overlap = len(top50_all & top50_anticascading)
    cascade_vs_anti_overlap = len(top50_cascading & top50_anticascading)

    # Unique tickers
    unique_to_anti = top50_anticascading - top50_all
    unique_to_all = top50_all - top50_anticascading

    print(f"📊 OVERLAP ANALYSIS:")
    print(f"   All vs Cascading: {all_vs_cascade_overlap}/50 ({all_vs_cascade_overlap/50*100:.1f}%)")
    print(f"   All vs Anti-Cascading: {all_vs_anti_overlap}/50 ({all_vs_anti_overlap/50*100:.1f}%)")
    print(f"   Cascading vs Anti-Cascading: {cascade_vs_anti_overlap}/50 ({cascade_vs_anti_overlap/50*100:.1f}%)")
    print(f"\n   Tickers unique to Anti-Cascading Top 50: {len(unique_to_anti)}")
    print(f"   Tickers unique to All Trades Top 50: {len(unique_to_all)}")

    return {
        'all_vs_cascade_overlap': all_vs_cascade_overlap,
        'current_vs_anti_overlap': all_vs_anti_overlap,
        'cascade_vs_anti_overlap': cascade_vs_anti_overlap,
        'unique_to_anti': list(unique_to_anti),
        'unique_to_current': list(unique_to_all),
    }


def save_comprehensive_results(config, paths, module_config, all_performance,
                               cascading_performance, anticascading_performance,
                               comparison_results):
    """
    Save all analysis results to config-specified output directory
    """

    print(f"\n💾 STEP 4: SAVING COMPREHENSIVE RESULTS")
    print("=" * 50)

    # Get output directory from config
    output_dir = Path(get_output_dir(config, 'ticker_ranking', category='portfolio'))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save performance rankings for each category
    all_perf_file = output_dir / "TOP50_ALL_TRADES.csv"
    cascade_perf_file = output_dir / "TOP50_CASCADING_TRADES.csv"
    anti_perf_file = output_dir / "TOP50_ANTICASCADING_TRADES.csv"

    all_performance.head(50).to_csv(all_perf_file, index=False)
    cascading_performance.head(50).to_csv(cascade_perf_file, index=False)
    anticascading_performance.head(50).to_csv(anti_perf_file, index=False)

    # Save full rankings
    all_performance.to_csv(output_dir / "all_tickers_performance_ALL.csv", index=False)
    cascading_performance.to_csv(output_dir / "all_tickers_performance_CASCADING.csv", index=False)
    anticascading_performance.to_csv(output_dir / "all_tickers_performance_ANTICASCADING.csv", index=False)

    # Save comparison summary
    summary_file = output_dir / "cascade_comparison_summary.md"
    with open(summary_file, 'w') as f:
        f.write("# CASCADE vs ANTI-CASCADE ANALYSIS SUMMARY\n\n")
        f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## OVERLAP STATISTICS\n\n")
        f.write(f"- All Trades vs Cascading Top 50: {comparison_results['all_vs_cascade_overlap']}/50 ({comparison_results['all_vs_cascade_overlap']/50*100:.1f}%)\n")
        f.write(f"- All Trades vs Anti-Cascading Top 50: {comparison_results['current_vs_anti_overlap']}/50 ({comparison_results['current_vs_anti_overlap']/50*100:.1f}%)\n")
        f.write(f"- Cascading vs Anti-Cascading Top 50: {comparison_results['cascade_vs_anti_overlap']}/50 ({comparison_results['cascade_vs_anti_overlap']/50*100:.1f}%)\n\n")

        f.write("## CRITICAL FINDINGS\n\n")
        f.write(f"- Current Top 50 vs Anti-Cascading Top 50 Overlap: {comparison_results['current_vs_anti_overlap']}/50 ({comparison_results['current_vs_anti_overlap']/50*100:.1f}%)\n")
        f.write(f"- Tickers unique to Anti-Cascading Top 50: {len(comparison_results['unique_to_anti'])}\n")
        f.write(f"- Current Top 50 tickers NOT in Anti-Cascading Top 50: {len(comparison_results['unique_to_current'])}\n\n")

        f.write("## CONCLUSION\n\n")
        if comparison_results['current_vs_anti_overlap'] < 40:
            f.write("❌ SIGNIFICANT DIFFERENCE: Current portfolio selection basis may be flawed!\n\n")
            f.write("✅ RECOMMENDATION: Use Anti-Cascading Top 50 for portfolio construction\n")
        else:
            f.write("✅ GOOD OVERLAP: Current selection basis is reasonably valid\n")

    print(f"✅ All performance rankings saved to: {output_dir}")
    print(f"✅ Top 50 lists saved for each category")
    print(f"✅ Comparison summary saved: {summary_file.name}")

    return summary_file


def main():
    """
    Execute the complete comprehensive cascade vs anti-cascade analysis
    """

    # Parse arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    paths = resolve_paths(config)
    module_config = get_module_spec(config, 'ticker_ranking', category='portfolio')

    print("🚀 STARTING COMPREHENSIVE CASCADE vs ANTI-CASCADE ANALYSIS")
    print("=" * 80)
    print(f"📁 Config: {args.config}")
    print(f"📊 Strategy: {config['run']['strategy']}")
    print(f"📅 Date Range: {config['run']['date_range']}")
    print("=" * 80)

    try:
        # Step 1: Load trades and tag with cascade patterns
        trades_df = load_trades(config, paths)
        validation = validate_trade_data(trades_df)

        if not validation['valid']:
            print(f"❌ Data validation failed: {validation['errors']}")
            return None

        all_trades_tagged = load_and_tag_all_trades(trades_df)

        # Step 2A: Calculate performance for ALL trades
        all_performance = calculate_ticker_performance_by_category(
            all_trades_tagged, category_filter=None, category_name="ALL"
        )

        # Step 2B: Calculate performance for CASCADING trades only
        cascading_performance = calculate_ticker_performance_by_category(
            all_trades_tagged, category_filter="CASCADING", category_name="CASCADING"
        )

        # Step 2C: Calculate performance for ANTI-CASCADING trades only
        anticascading_performance = calculate_ticker_performance_by_category(
            all_trades_tagged, category_filter="ANTI_CASCADING", category_name="ANTI_CASCADING"
        )

        # Step 3: Compare Top 50 performers across categories
        comparison_results = compare_top50_across_categories(
            all_performance, cascading_performance, anticascading_performance
        )

        # Step 4: Save comprehensive results
        summary_file = save_comprehensive_results(
            config, paths, module_config,
            all_performance, cascading_performance, anticascading_performance,
            comparison_results
        )

        print(f"\n🎯 CRITICAL QUESTION ANSWERED:")
        overlap_percentage = comparison_results['current_vs_anti_overlap'] / 50 * 100
        if overlap_percentage < 80:
            print(f"❌ Current Top 50 basis is FLAWED! Only {overlap_percentage:.1f}% overlap with Anti-Cascading Top 50")
            print(f"✅ MUST use Anti-Cascading Top 50 for portfolio construction")
        else:
            print(f"✅ Current Top 50 basis is VALID! {overlap_percentage:.1f}% overlap with Anti-Cascading Top 50")

        print(f"\n🏆 COMPREHENSIVE ANALYSIS COMPLETED!")
        print(f"📂 Results saved to: {Path(get_output_dir(config, 'ticker_ranking', category='portfolio'))}")

        return {
            'all_performance': all_performance,
            'cascading_performance': cascading_performance,
            'anticascading_performance': anticascading_performance,
            'comparison_results': comparison_results,
            'summary_file': str(summary_file)
        }

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
