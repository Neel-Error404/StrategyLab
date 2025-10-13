#!/usr/bin/env python3
"""
Cascade Analysis - Sequential Trade Pattern Identification
===========================================================

Identifies and analyzes cascading trade patterns (consecutive wins/losses,
revenge trading, momentum trades).

✅ GENERIC - Works with ANY strategy's trade data
❌ No strategy-specific logic required

**What is "Cascading"?**
Sequential trades on the same ticker where the previous trade outcome may
influence the next trade's psychology or timing.

**Patterns Detected**:
- Winning cascades: Trade entered after a win
- Losing cascades: Trade entered after a loss (revenge trading)
- Same-direction cascades: Consecutive Buy-Buy or Sell-Sell
- Opposite-direction cascades: Buy-Sell or Sell-Buy alternation
- Time gap analysis: How quickly does next trade occur?

**Usage**:
    python 03_cascade_analysis.py --config ../config.yaml

**Required Config**:
    run:
      run_id: "20251006_024924"
      strategy: "mse"
      date_range: "2022-01-01_to_2025-08-31"

    analysis:
      generic:
        modules:
          cascade_analysis:
            enabled: true
            config:
              min_time_gap_minutes: 30
              max_same_day_gap: 1440

**Outputs**:
    - output/cascade_tagged_trades.csv
    - reports/CASCADE_ANALYSIS_REPORT.md
    - output/cascade_statistics.json

Author: StrategyLab Team
Version: 2.0 - YAML Config Driven
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import argparse
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our reusable modules
from modules.config_loader import (
    get_analysis_config,
    load_config,
    resolve_artifact_path,
    resolve_paths,
)
from modules.data_loader import load_trades, validate_trade_data

def tag_trades(trades_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Tag each trade with cascade characteristics.

    Tags:
    - FIRST_TRADE_OVERALL: First trade in dataset
    - FIRST_TRADE_FOR_TICKER: First trade for this ticker
    - FIRST_TRADE_OF_DAY: First trade of the day for this ticker
    - CONSECUTIVE_SAME_DIRECTION: Same direction as previous trade
    - CONSECUTIVE_OPPOSITE_DIRECTION: Opposite direction from previous
    - WINNING_CASCADE: Trade after a winning trade
    - LOSING_CASCADE: Trade after a losing trade

    Args:
        trades_df: DataFrame with trade data
        config: Analysis configuration

    Returns:
        DataFrame with cascade tags added
    """
    print("\n🏷️  TAGGING TRADES WITH CASCADE PATTERNS")
    print("=" * 60)

    # Sort by ticker and entry time for sequential analysis
    trades = trades_df.copy().sort_values(['ticker', 'Entry Time']).reset_index(drop=True)

    # Add entry date column
    trades['Entry Date'] = trades['Entry Time'].dt.date

    # Create previous trade reference columns
    trades['prev_ticker'] = trades['ticker'].shift(1)
    trades['prev_entry_date'] = trades['Entry Date'].shift(1)
    trades['prev_trade_type'] = trades['Trade Type'].shift(1)
    trades['prev_exit_time'] = trades['Exit Time'].shift(1)
    trades['prev_profit'] = trades['Profit (Currency)'].shift(1)

    # Calculate time gap from previous trade (minutes)
    trades['time_gap_from_prev'] = np.where(
        trades['ticker'] == trades['prev_ticker'],
        (trades['Entry Time'] - trades['prev_exit_time']).dt.total_seconds() / 60,
        np.nan
    )

    # Tag each trade
    print(f"📊 Processing {len(trades):,} trades...")

    tags = []
    for _, trade in trades.iterrows():
        if pd.isna(trade['prev_ticker']):
            tag = 'FIRST_TRADE_OVERALL'
        elif trade['ticker'] != trade['prev_ticker']:
            tag = 'FIRST_TRADE_FOR_TICKER'
        elif trade['Entry Date'] != trade['prev_entry_date']:
            tag = 'FIRST_TRADE_OF_DAY'
        elif trade['Trade Type'] == trade['prev_trade_type']:
            # Further classify by previous outcome
            if trade['prev_profit'] > 0:
                tag = 'WINNING_CASCADE_SAME_DIR'
            else:
                tag = 'LOSING_CASCADE_SAME_DIR'
        else:  # Opposite direction
            if trade['prev_profit'] > 0:
                tag = 'WINNING_CASCADE_OPP_DIR'
            else:
                tag = 'LOSING_CASCADE_OPP_DIR'

        tags.append(tag)

    trades['cascade_tag'] = tags

    # Categorize time gaps
    trades['time_gap_category'] = categorize_time_gaps(trades['time_gap_from_prev'])

    # Additional flags
    trades['is_profitable'] = trades['Profit (Currency)'] > 0
    trades['is_cascade'] = ~trades['cascade_tag'].isin(['FIRST_TRADE_OVERALL', 'FIRST_TRADE_FOR_TICKER', 'FIRST_TRADE_OF_DAY'])
    trades['after_win'] = trades['prev_profit'] > 0
    trades['after_loss'] = trades['prev_profit'] < 0

    print(f"✅ Tagging complete!")

    # Show distribution
    print(f"\n📊 CASCADE TAG DISTRIBUTION:")
    print(trades['cascade_tag'].value_counts().to_string())

    return trades

def categorize_time_gaps(time_gaps: pd.Series) -> list:
    """Categorize time gaps into buckets."""
    categories = []

    for gap in time_gaps:
        if pd.isna(gap):
            categories.append('NO_GAP')
        elif gap < 5:
            categories.append('0-5_MIN')
        elif gap < 15:
            categories.append('5-15_MIN')
        elif gap < 30:
            categories.append('15-30_MIN')
        elif gap < 60:
            categories.append('30-60_MIN')
        elif gap < 120:
            categories.append('1-2_HOURS')
        elif gap < 240:
            categories.append('2-4_HOURS')
        else:
            categories.append('4+_HOURS')

    return categories

def analyze_cascade_performance(tagged_trades: pd.DataFrame) -> dict:
    """Analyze performance by cascade patterns."""

    print(f"\n📊 CASCADE PERFORMANCE ANALYSIS")
    print("=" * 60)

    # Overall statistics by tag
    tag_stats = tagged_trades.groupby('cascade_tag').agg({
        'Profit (Currency)': ['count', 'sum', 'mean', 'std'],
        'is_profitable': 'mean'
    }).round(3)

    tag_stats.columns = ['count', 'total_profit', 'avg_profit', 'profit_std', 'win_rate']
    tag_stats['win_rate'] = tag_stats['win_rate'] * 100

    print("\n📈 PERFORMANCE BY CASCADE TAG:")
    print(tag_stats.to_string())

    # Cascade vs Non-Cascade comparison
    cascade_trades = tagged_trades[tagged_trades['is_cascade']]
    non_cascade_trades = tagged_trades[~tagged_trades['is_cascade']]

    cascade_wr = cascade_trades['is_profitable'].mean() * 100
    non_cascade_wr = non_cascade_trades['is_profitable'].mean() * 100

    print(f"\n🔍 CASCADE vs NON-CASCADE:")
    print(f"   Cascade trades: {len(cascade_trades):,} ({cascade_wr:.1f}% WR)")
    print(f"   Non-cascade trades: {len(non_cascade_trades):,} ({non_cascade_wr:.1f}% WR)")
    print(f"   Difference: {cascade_wr - non_cascade_wr:+.1f}%")

    # After win vs After loss
    after_win_trades = tagged_trades[tagged_trades['after_win'] == True]
    after_loss_trades = tagged_trades[tagged_trades['after_loss'] == True]

    if len(after_win_trades) > 0:
        win_wr = after_win_trades['is_profitable'].mean() * 100
        print(f"\n🏆 TRADES AFTER WINNING:")
        print(f"   Count: {len(after_win_trades):,}")
        print(f"   Win Rate: {win_wr:.1f}%")
        print(f"   Avg Profit: ₹{after_win_trades['Profit (Currency)'].mean():.2f}")

    if len(after_loss_trades) > 0:
        loss_wr = after_loss_trades['is_profitable'].mean() * 100
        print(f"\n❌ TRADES AFTER LOSING:")
        print(f"   Count: {len(after_loss_trades):,}")
        print(f"   Win Rate: {loss_wr:.1f}%")
        print(f"   Avg Profit: ₹{after_loss_trades['Profit (Currency)'].mean():.2f}")

    # Time gap analysis for cascades
    print(f"\n⏰ TIME GAP ANALYSIS:")
    time_gap_stats = cascade_trades.groupby('time_gap_category').agg({
        'Profit (Currency)': ['count', 'mean'],
        'is_profitable': 'mean'
    }).round(3)

    time_gap_stats.columns = ['count', 'avg_profit', 'win_rate']
    time_gap_stats['win_rate'] = time_gap_stats['win_rate'] * 100
    print(time_gap_stats.to_string())

    # Compile results
    results = {
        'tag_statistics': tag_stats.to_dict(),
        'cascade_summary': {
            'cascade_trades': int(len(cascade_trades)),
            'cascade_win_rate': float(cascade_wr),
            'non_cascade_trades': int(len(non_cascade_trades)),
            'non_cascade_win_rate': float(non_cascade_wr),
            'difference': float(cascade_wr - non_cascade_wr)
        },
        'after_win_loss': {
            'after_win_count': int(len(after_win_trades)),
            'after_win_wr': float(win_wr) if len(after_win_trades) > 0 else None,
            'after_loss_count': int(len(after_loss_trades)),
            'after_loss_wr': float(loss_wr) if len(after_loss_trades) > 0 else None
        },
        'time_gap_stats': time_gap_stats.to_dict()
    }

    return results

def generate_report(config: dict, tagged_trades: pd.DataFrame, results: dict) -> Path:
    """Generate markdown report."""

    report_path = Path(resolve_artifact_path(config, 'cascade_analysis', 'cascade_insights', artifact_type='markdown'))
    run_cfg = config.get('run', {})

    lines = [
        "# Cascade Analysis Report",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Run ID**: {run_cfg.get('run_id', 'N/A')}",
        f"**Strategy**: {run_cfg.get('strategy', 'N/A')}",
    ]
    if run_cfg.get('date_range'):
        lines.append(f"**Date Range**: {run_cfg['date_range']}")
    lines.append("\n---\n")

    lines.append("## Executive Summary\n")
    cascade_summary = results['cascade_summary']
    cascade_pct = (cascade_summary['cascade_trades'] / len(tagged_trades) * 100) if len(tagged_trades) else 0
    lines.append(f"- **Total Trades**: {len(tagged_trades):,}")
    lines.append(f"- **Cascade Trades**: {cascade_summary['cascade_trades']:,} ({cascade_pct:.1f}%)")
    lines.append(f"- **Cascade Win Rate**: {cascade_summary['cascade_win_rate']:.1f}%")
    lines.append(f"- **Non-Cascade Win Rate**: {cascade_summary['non_cascade_win_rate']:.1f}%")
    lines.append(f"- **Performance Delta**: {cascade_summary['difference']:+.1f}%\n")

    lines.append("## Key Findings\n")
    delta = cascade_summary['difference']
    if delta < -2:
        lines.extend([
            "- ⚠️ **Cascade trades underperform** by more than 2%.",
            "- Consider cooldown periods after losses and throttling rapid re-entries.",
        ])
    elif delta > 2:
        lines.extend([
            "- ✅ **Cascade trades outperform**; consider momentum-friendly filters.",
            "- Portfolio construction can prioritize these patterns.",
        ])
    else:
        lines.extend([
            "- ➖ Cascade trades perform similarly to non-cascades.",
            "- Optimization can focus on other leverage points.",
        ])

    lines.append("\n## Performance by Cascade Tag\n")
    lines.append("| Tag | Count | Win Rate | Avg Profit | Total Profit |")
    lines.append("|-----|-------|----------|------------|-------------|")
    for tag, stats_count in results['tag_statistics']['count'].items():
        count = int(stats_count)
        wr = results['tag_statistics']['win_rate'].get(tag, 0)
        avg_profit = results['tag_statistics']['avg_profit'].get(tag, 0)
        total_profit = results['tag_statistics']['total_profit'].get(tag, 0)
        lines.append(f"| {tag} | {count:,} | {wr:.1f}% | ₹{avg_profit:.2f} | ₹{total_profit:,.2f} |")

    lines.append("\n## Recommendations\n")
    lines.append("- Adjust cascade handling based on performance delta and after-win/loss behaviour.")
    after_loss_wr = results['after_win_loss'].get('after_loss_wr')
    after_win_wr = results['after_win_loss'].get('after_win_wr')
    if after_loss_wr and after_loss_wr < 45:
        lines.append("- Consider cooldown after losses: cascade win rate <45%.")
    if after_win_wr and after_win_wr > 52:
        lines.append("- Consider momentum entries after wins (>52% win rate).")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    return report_path

def main():
    parser = argparse.ArgumentParser(
        description="Cascade Analysis - Sequential Trade Pattern Identification"
    )
    parser.add_argument(
        '--config',
        required=True,
        help="Path to YAML config file (e.g., ../config.yaml)"
    )
    parser.add_argument(
        '--sample',
        type=int,
        help="Sample N trades for faster testing"
    )

    args = parser.parse_args()

    print("="*60)
    print("CASCADE ANALYSIS - YAML Config Driven")
    print("="*60)

    # Load configuration
    config = load_config(args.config)
    paths = resolve_paths(config)

    # Get module-specific config
    module_config = get_analysis_config(config, 'cascade_analysis')

    # Load trade data
    trades_df = load_trades(config, paths, sample_size=args.sample)

    # Validate data
    print("\n🔍 Validating trade data...")
    validation = validate_trade_data(trades_df)
    if not validation['valid']:
        print(f"❌ Data validation failed: {validation['errors']}")
        return 1

    if validation['warnings']:
        print(f"⚠️  Warnings: {validation['warnings']}")

    # Tag trades with cascade patterns
    tagged_trades = tag_trades(trades_df, module_config)

    # Analyze cascade performance
    results = analyze_cascade_performance(tagged_trades)

    # Save tagged trades
    tagged_path = Path(resolve_artifact_path(config, 'cascade_analysis', 'cascade_tags', artifact_type='csv'))
    tagged_trades.to_csv(tagged_path, index=False)
    print(f"\n💾 Saved tagged trades to: {tagged_path}")

    # Save statistics JSON
    metrics_path = Path(resolve_artifact_path(config, 'cascade_analysis', 'cascade_metrics', artifact_type='json'))
    metrics_path.write_text(json.dumps(results, indent=2))
    print(f"💾 Saved statistics to: {metrics_path}")

    # Generate report
    report_path = generate_report(config, tagged_trades, results)

    print("\n" + "="*60)
    print("✅ CASCADE ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nOutputs:")
    print(f"  - Tagged Trades: {tagged_path}")
    print(f"  - Statistics: {metrics_path}")
    print(f"  - Report: {report_path}")

    return 0

if __name__ == "__main__":
    exit(main())
