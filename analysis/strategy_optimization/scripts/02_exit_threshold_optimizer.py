"""
Stage 2: Exit Threshold Optimization Script
============================================

Purpose:
--------
Find optimal exit threshold by testing alternatives on validation data.

Strategy:
---------
1. Load baseline trades from Stage 1 (8,186 trades, validation period)
2. Test thresholds: 50%, 55%, 60%, 65%, 70%, 75%, 80%, 85%, 90%, 95%
3. Test Buy and Sell trades separately (allow different optimal thresholds)
4. For each threshold, simulate what would have happened
5. Calculate metrics: WR, PF, Sharpe, return improvement
6. Find optimal threshold(s) based on success criteria
7. Generate comparison report and visualizations

This does NOT retrain the strategy - it replays existing trades with different exit rules.

Author: Strategy Optimization Pipeline
Date: 2025-10-04
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

# Add modules to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / 'modules'))

# Import local modules
from exit_simulator import (
    simulate_all_thresholds,
    calculate_threshold_metrics,
    find_optimal_threshold
)
from metrics_calculator import print_metrics_summary, compare_metrics

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_PATH = PROJECT_ROOT / 'config' / 'optimization_config.yaml'
BASE_DATA_DIR = PROJECT_ROOT / 'data' / 'base_data'
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
DOCS_DIR = PROJECT_ROOT / 'docs'

# Load baseline data from Stage 1
BASELINE_DATA_PATH = CHECKPOINT_DIR / 'stage1_baseline_data.csv'

# Thresholds to test (5% steps)
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_config():
    """Load optimization configuration"""
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)


def load_baseline_data():
    """Load baseline trades from Stage 1"""
    print("\n📂 Loading baseline data from Stage 1...")

    df = pd.read_csv(BASELINE_DATA_PATH)
    print(f"   ✓ Loaded {len(df):,} trades from validation period")

    # Convert timestamps
    df['Entry Time'] = pd.to_datetime(df['Entry Time'])
    df['Exit Time'] = pd.to_datetime(df['Exit Time'])

    # Summary
    buy_trades = (df['Trade Type'] == 'Buy').sum()
    sell_trades = (df['Trade Type'] == 'Sell').sum()

    print(f"   Buy trades: {buy_trades:,}")
    print(f"   Sell trades: {sell_trades:,}")

    return df


def plot_threshold_performance(all_metrics_df: pd.DataFrame,
                               trade_type: str,
                               save_path: str):
    """
    Plot threshold performance curves.

    Parameters:
    -----------
    all_metrics_df : pd.DataFrame
        Metrics for all thresholds
    trade_type : str
        'Buy', 'Sell', or 'Combined'
    save_path : str
        Path to save figure
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Win Rate
    ax = axes[0, 0]
    ax.plot(all_metrics_df['threshold'] * 100, all_metrics_df['win_rate_pct'],
            marker='o', linewidth=2, markersize=8, color='steelblue')
    ax.axhline(y=52, color='green', linestyle='--', label='Target: 52%')
    ax.axhline(y=all_metrics_df[all_metrics_df['threshold'] == 0.80]['win_rate_pct'].values[0],
               color='red', linestyle=':', label='Baseline (80%)')
    ax.set_xlabel('Exit Threshold (%)')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title(f'{trade_type} Trades: Win Rate vs Threshold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 2: Profit Factor
    ax = axes[0, 1]
    ax.plot(all_metrics_df['threshold'] * 100, all_metrics_df['profit_factor'],
            marker='o', linewidth=2, markersize=8, color='green')
    ax.axhline(y=1.25, color='green', linestyle='--', label='Target: 1.25')
    ax.axhline(y=all_metrics_df[all_metrics_df['threshold'] == 0.80]['profit_factor'].values[0],
               color='red', linestyle=':', label='Baseline (80%)')
    ax.set_xlabel('Exit Threshold (%)')
    ax.set_ylabel('Profit Factor')
    ax.set_title(f'{trade_type} Trades: Profit Factor vs Threshold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 3: Sharpe Ratio
    ax = axes[1, 0]
    ax.plot(all_metrics_df['threshold'] * 100, all_metrics_df['sharpe_ratio'],
            marker='o', linewidth=2, markersize=8, color='purple')
    ax.axhline(y=1.5, color='green', linestyle='--', label='Target: 1.5')
    ax.axhline(y=all_metrics_df[all_metrics_df['threshold'] == 0.80]['sharpe_ratio'].values[0],
               color='red', linestyle=':', label='Baseline (80%)')
    ax.set_xlabel('Exit Threshold (%)')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title(f'{trade_type} Trades: Sharpe Ratio vs Threshold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 4: Trade Duration
    ax = axes[1, 1]
    ax.plot(all_metrics_df['threshold'] * 100, all_metrics_df['avg_duration_minutes'],
            marker='o', linewidth=2, markersize=8, color='orange')
    ax.axhline(y=all_metrics_df[all_metrics_df['threshold'] == 0.80]['avg_duration_minutes'].values[0],
               color='red', linestyle=':', label='Baseline (80%)')
    ax.set_xlabel('Exit Threshold (%)')
    ax.set_ylabel('Avg Duration (minutes)')
    ax.set_title(f'{trade_type} Trades: Duration vs Threshold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {save_path}")
    plt.close()


def generate_optimization_report(buy_metrics: pd.DataFrame,
                                 sell_metrics: pd.DataFrame,
                                 optimal_buy: Dict,
                                 optimal_sell: Dict,
                                 output_path: str):
    """Generate Stage 2 optimization report"""

    baseline_buy = buy_metrics[buy_metrics['threshold'] == 0.80].iloc[0]
    baseline_sell = sell_metrics[sell_metrics['threshold'] == 0.80].iloc[0]

    report = f"""# Stage 2: Exit Threshold Optimization Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Method**: Simulation of alternative exit thresholds on validation data
**Thresholds Tested**: {len(THRESHOLDS)} values (50% to 95% in 5% steps)

---

## 📋 METHODOLOGY

We tested what would have happened if we had exited at different MACD threshold levels:

**Current Strategy (Baseline)**:
- Exit Buy trades when: 15min MACD drops to **80%** of peak
- Exit Sell trades when: 15min MACD rises to **80%** of valley

**Testing Approach**:
1. Loaded 8,186 trades from validation period (Stage 1 baseline)
2. For each trade, loaded full 5-minute bar data during the trade
3. Simulated exit at each threshold: 50%, 55%, ..., 95%
4. Calculated metrics for each threshold
5. Tested Buy and Sell trades separately

**Why This Works**:
- Uses actual historical data (no overfitting to new patterns)
- Replay-based simulation (deterministic)
- Tests single parameter in isolation (controlled experiment)

---

## 🎯 BUY TRADES OPTIMIZATION

### Baseline (80% Threshold)
| Metric | Value |
|--------|-------|
| **Total Trades** | {baseline_buy['total_trades']:,} |
| **Win Rate** | {baseline_buy['win_rate_pct']:.2f}% |
| **Profit Factor** | {baseline_buy['profit_factor']:.2f} |
| **Sharpe Ratio** | {baseline_buy['sharpe_ratio']:.2f} |
| **Avg Duration** | {baseline_buy['avg_duration_minutes']:.1f} minutes |
| **Total Return** | {baseline_buy['total_return_pct']:.2f}% |

### Optimal Threshold: {optimal_buy['threshold']:.0%}

| Metric | Value | vs Baseline | Status |
|--------|-------|-------------|--------|
| **Win Rate** | {optimal_buy['win_rate_pct']:.2f}% | {optimal_buy['win_rate_pct'] - baseline_buy['win_rate_pct']:+.2f}% | {'✅' if optimal_buy['win_rate_pct'] > baseline_buy['win_rate_pct'] else '❌'} |
| **Profit Factor** | {optimal_buy['profit_factor']:.2f} | {optimal_buy['profit_factor'] - baseline_buy['profit_factor']:+.2f} | {'✅' if optimal_buy['profit_factor'] > baseline_buy['profit_factor'] else '❌'} |
| **Sharpe Ratio** | {optimal_buy['sharpe_ratio']:.2f} | {optimal_buy['sharpe_ratio'] - baseline_buy['sharpe_ratio']:+.2f} | {'✅' if optimal_buy['sharpe_ratio'] > baseline_buy['sharpe_ratio'] else '❌'} |
| **Total Return** | {optimal_buy['total_return_pct']:.2f}% | {optimal_buy['total_return_pct'] - baseline_buy['total_return_pct']:+.2f}% | {'✅' if optimal_buy['total_return_pct'] > baseline_buy['total_return_pct'] else '❌'} |
| **Avg Duration** | {optimal_buy['avg_duration_minutes']:.1f} min | {optimal_buy['avg_duration_minutes'] - baseline_buy['avg_duration_minutes']:+.1f} min | - |

### Outcome Changes (Buy)
- **Losers → Winners**: {optimal_buy.get('losers_to_winners_count', 0):,} trades
- **Winners → Losers**: {optimal_buy.get('winners_to_losers_count', 0):,} trades
- **Net Improvement**: {optimal_buy.get('net_outcome_improvement', 0):,} trades

---

## 🎯 SELL TRADES OPTIMIZATION

### Baseline (80% Threshold)
| Metric | Value |
|--------|-------|
| **Total Trades** | {baseline_sell['total_trades']:,} |
| **Win Rate** | {baseline_sell['win_rate_pct']:.2f}% |
| **Profit Factor** | {baseline_sell['profit_factor']:.2f} |
| **Sharpe Ratio** | {baseline_sell['sharpe_ratio']:.2f} |
| **Avg Duration** | {baseline_sell['avg_duration_minutes']:.1f} minutes |
| **Total Return** | {baseline_sell['total_return_pct']:.2f}% |

### Optimal Threshold: {optimal_sell['threshold']:.0%}

| Metric | Value | vs Baseline | Status |
|--------|-------|-------------|--------|
| **Win Rate** | {optimal_sell['win_rate_pct']:.2f}% | {optimal_sell['win_rate_pct'] - baseline_sell['win_rate_pct']:+.2f}% | {'✅' if optimal_sell['win_rate_pct'] > baseline_sell['win_rate_pct'] else '❌'} |
| **Profit Factor** | {optimal_sell['profit_factor']:.2f} | {optimal_sell['profit_factor'] - baseline_sell['profit_factor']:+.2f} | {'✅' if optimal_sell['profit_factor'] > baseline_sell['profit_factor'] else '❌'} |
| **Sharpe Ratio** | {optimal_sell['sharpe_ratio']:.2f} | {optimal_sell['sharpe_ratio'] - baseline_sell['sharpe_ratio']:+.2f} | {'✅' if optimal_sell['sharpe_ratio'] > baseline_sell['sharpe_ratio'] else '❌'} |
| **Total Return** | {optimal_sell['total_return_pct']:.2f}% | {optimal_sell['total_return_pct'] - baseline_sell['total_return_pct']:+.2f}% | {'✅' if optimal_sell['total_return_pct'] > baseline_sell['total_return_pct'] else '❌'} |
| **Avg Duration** | {optimal_sell['avg_duration_minutes']:.1f} min | {optimal_sell['avg_duration_minutes'] - baseline_sell['avg_duration_minutes']:+.1f} min | - |

### Outcome Changes (Sell)
- **Losers → Winners**: {optimal_sell.get('losers_to_winners_count', 0):,} trades
- **Winners → Losers**: {optimal_sell.get('winners_to_losers_count', 0):,} trades
- **Net Improvement**: {optimal_sell.get('net_outcome_improvement', 0):,} trades

---

## 📊 KEY INSIGHTS

**Buy vs Sell Threshold Difference**:
- Buy optimal: {optimal_buy['threshold']:.0%}
- Sell optimal: {optimal_sell['threshold']:.0%}
- Difference: {abs(optimal_buy['threshold'] - optimal_sell['threshold']):.0%}

"""

    # Add interpretation
    if optimal_buy['threshold'] == optimal_sell['threshold']:
        report += "\n✅ **Same optimal threshold for both trade types** - simpler implementation\n"
    else:
        report += "\n⚠️ **Different optimal thresholds** - Buy and Sell trades benefit from separate exit rules\n"

    if optimal_buy['threshold'] < 0.80:
        report += f"\n💡 **Buy trades benefit from earlier exits** ({optimal_buy['threshold']:.0%} vs 80%) - more aggressive profit-taking\n"
    elif optimal_buy['threshold'] > 0.80:
        report += f"\n💡 **Buy trades benefit from later exits** ({optimal_buy['threshold']:.0%} vs 80%) - letting winners run\n"

    if optimal_sell['threshold'] < 0.80:
        report += f"\n💡 **Sell trades benefit from earlier exits** ({optimal_sell['threshold']:.0%} vs 80%) - more aggressive profit-taking\n"
    elif optimal_sell['threshold'] > 0.80:
        report += f"\n💡 **Sell trades benefit from later exits** ({optimal_sell['threshold']:.0%} vs 80%) - letting winners run\n"

    report += f"""

---

## 🚦 SUCCESS CRITERIA CHECK

**Target Metrics (from optimization_config.yaml)**:
- Win Rate ≥ 52%
- Profit Factor ≥ 1.25
- Sharpe Ratio ≥ 1.5

**Buy Trades**:
- Win Rate: {optimal_buy['win_rate_pct']:.2f}% {'✅' if optimal_buy['win_rate_pct'] >= 52 else '❌'}
- Profit Factor: {optimal_buy['profit_factor']:.2f} {'✅' if optimal_buy['profit_factor'] >= 1.25 else '❌'}
- Sharpe Ratio: {optimal_buy['sharpe_ratio']:.2f} {'✅' if optimal_buy['sharpe_ratio'] >= 1.5 else '❌'}

**Sell Trades**:
- Win Rate: {optimal_sell['win_rate_pct']:.2f}% {'✅' if optimal_sell['win_rate_pct'] >= 52 else '❌'}
- Profit Factor: {optimal_sell['profit_factor']:.2f} {'✅' if optimal_sell['profit_factor'] >= 1.25 else '❌'}
- Sharpe Ratio: {optimal_sell['sharpe_ratio']:.2f} {'✅' if optimal_sell['sharpe_ratio'] >= 1.5 else '❌'}

---

## 📁 OUTPUTS

**Checkpoint**: `checkpoints/stage2_optimized_thresholds.csv`
**Visualizations**: `docs/stage2_threshold_performance_*.png`
**Full Results**: `checkpoints/stage2_all_threshold_results.csv`

---

## 🚦 NEXT STEPS

**Decision Gate**: Does optimization meet success criteria?

1. **If YES (both Buy and Sell meet criteria)**:
   - ✅ Proceed to Stage 3: Walk-Forward Validation
   - Test threshold stability across different time windows
   - Ensure improvement is robust, not just lucky

2. **If PARTIAL (only one trade type meets criteria)**:
   - ⚠️ Review underperforming trade type
   - Consider if issue is data-specific or systematic
   - May proceed with caution (test different thresholds in Stage 3)

3. **If NO (neither meets criteria)**:
   - ❌ Exit threshold optimization insufficient
   - Skip to Stage 4: Entry Filter Optimization (H2)
   - OR reconsider threshold range (test 45%, 40%?)

**Current Recommendation**: [To be filled after analysis]

---

*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    # Write report
    with open(output_path, 'w') as f:
        f.write(report)

    print(f"\n   ✓ Report saved: {output_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run Stage 2: Exit Threshold Optimization"""

    print("="*70)
    print("STAGE 2: EXIT THRESHOLD OPTIMIZATION")
    print("="*70)

    # Load configuration
    config = load_config()
    success_criteria = config['success_criteria']['traditional']

    # Load baseline data
    baseline_data = load_baseline_data()

    # =========================================================================
    # OPTIMIZE BUY TRADES
    # =========================================================================

    print(f"\n{'='*70}")
    print("OPTIMIZING BUY TRADES")
    print("="*70)

    def progress_buy(current, total, threshold):
        if current % 500 == 0:
            print(f"      Progress: {current:,}/{total:,} ({current/total*100:.1f}%)")

    buy_simulations = simulate_all_thresholds(
        baseline_data,
        str(BASE_DATA_DIR),
        THRESHOLDS,
        trade_type_filter='Buy',
        progress_callback=progress_buy
    )

    # Calculate metrics for each threshold
    print(f"\n   Calculating metrics for each threshold...")
    buy_metrics_list = []
    for threshold in THRESHOLDS:
        metrics = calculate_threshold_metrics(buy_simulations, threshold)
        buy_metrics_list.append(metrics)

    buy_metrics_df = pd.DataFrame(buy_metrics_list)

    # Find optimal
    optimal_buy_threshold, optimal_buy_metrics = find_optimal_threshold(
        buy_metrics_list,
        success_criteria
    )

    print(f"\n   ✅ Optimal Buy Threshold: {optimal_buy_threshold:.0%}")
    print(f"      WR: {optimal_buy_metrics['win_rate_pct']:.2f}% | PF: {optimal_buy_metrics['profit_factor']:.2f} | Sharpe: {optimal_buy_metrics['sharpe_ratio']:.2f}")

    # =========================================================================
    # OPTIMIZE SELL TRADES
    # =========================================================================

    print(f"\n{'='*70}")
    print("OPTIMIZING SELL TRADES")
    print("="*70)

    def progress_sell(current, total, threshold):
        if current % 500 == 0:
            print(f"      Progress: {current:,}/{total:,} ({current/total*100:.1f}%)")

    sell_simulations = simulate_all_thresholds(
        baseline_data,
        str(BASE_DATA_DIR),
        THRESHOLDS,
        trade_type_filter='Sell',
        progress_callback=progress_sell
    )

    # Calculate metrics
    print(f"\n   Calculating metrics for each threshold...")
    sell_metrics_list = []
    for threshold in THRESHOLDS:
        metrics = calculate_threshold_metrics(sell_simulations, threshold)
        sell_metrics_list.append(metrics)

    sell_metrics_df = pd.DataFrame(sell_metrics_list)

    # Find optimal
    optimal_sell_threshold, optimal_sell_metrics = find_optimal_threshold(
        sell_metrics_list,
        success_criteria
    )

    print(f"\n   ✅ Optimal Sell Threshold: {optimal_sell_threshold:.0%}")
    print(f"      WR: {optimal_sell_metrics['win_rate_pct']:.2f}% | PF: {optimal_sell_metrics['profit_factor']:.2f} | Sharpe: {optimal_sell_metrics['sharpe_ratio']:.2f}")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================

    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print("="*70)

    # Save all simulation results
    all_simulations = pd.concat([buy_simulations, sell_simulations], ignore_index=True)
    all_simulations.to_csv(CHECKPOINT_DIR / 'stage2_all_simulations.csv', index=False)
    print(f"   ✓ Saved all simulations: {len(all_simulations):,} records")

    # Save optimal thresholds
    optimal_config = pd.DataFrame([
        {'trade_type': 'Buy', 'optimal_threshold': optimal_buy_threshold, **optimal_buy_metrics},
        {'trade_type': 'Sell', 'optimal_threshold': optimal_sell_threshold, **optimal_sell_metrics}
    ])
    optimal_config.to_csv(CHECKPOINT_DIR / 'stage2_optimal_thresholds.csv', index=False)
    print(f"   ✓ Saved optimal thresholds")

    # Save all metrics
    buy_metrics_df['trade_type'] = 'Buy'
    sell_metrics_df['trade_type'] = 'Sell'
    all_metrics = pd.concat([buy_metrics_df, sell_metrics_df], ignore_index=True)
    all_metrics.to_csv(CHECKPOINT_DIR / 'stage2_all_metrics.csv', index=False)
    print(f"   ✓ Saved all metrics")

    # =========================================================================
    # GENERATE VISUALIZATIONS
    # =========================================================================

    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    plot_threshold_performance(
        buy_metrics_df,
        'Buy',
        str(DOCS_DIR / 'stage2_threshold_performance_buy.png')
    )

    plot_threshold_performance(
        sell_metrics_df,
        'Sell',
        str(DOCS_DIR / 'stage2_threshold_performance_sell.png')
    )

    # =========================================================================
    # GENERATE REPORT
    # =========================================================================

    print(f"\n{'='*70}")
    print("GENERATING OPTIMIZATION REPORT")
    print("="*70)

    generate_optimization_report(
        buy_metrics_df,
        sell_metrics_df,
        optimal_buy_metrics,
        optimal_sell_metrics,
        str(DOCS_DIR / 'stage2_optimization_report.md')
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print(f"\n{'='*70}")
    print("✅ STAGE 2 COMPLETE")
    print("="*70)

    print(f"\n📊 Results:")
    print(f"   Buy Optimal: {optimal_buy_threshold:.0%} | WR: {optimal_buy_metrics['win_rate_pct']:.2f}% | PF: {optimal_buy_metrics['profit_factor']:.2f}")
    print(f"   Sell Optimal: {optimal_sell_threshold:.0%} | WR: {optimal_sell_metrics['win_rate_pct']:.2f}% | PF: {optimal_sell_metrics['profit_factor']:.2f}")

    print(f"\n📁 Outputs:")
    print(f"   Checkpoint: checkpoints/stage2_optimal_thresholds.csv")
    print(f"   Report: docs/stage2_optimization_report.md")
    print(f"   Visualizations: docs/stage2_threshold_performance_*.png")

    print(f"\n🚦 Next Steps:")
    print(f"   1. Review docs/stage2_optimization_report.md")
    print(f"   2. Check if success criteria met")
    print(f"   3. Update PHASE2_ANALYSIS_LOG.md")
    print(f"   4. Decide: Proceed to Stage 3 (Walk-Forward)?")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
