"""
Stage 3: Walk-Forward Validation Script
========================================

Purpose:
--------
Validate that the 95% threshold from Stage 2 is stable across different time periods.

Strategy:
---------
1. Load baseline data from Stage 1
2. Split validation period (2024 H1) into rolling monthly windows
3. For each window, test all thresholds and find optimal
4. Check if 95% is consistently optimal across windows
5. Calculate stability metrics (CV, consistency rate, optimality rate)
6. Make GO/NO-GO decision based on stability criteria

Success Criteria:
-----------------
- Coefficient of Variation (CV) < 10% for Win Rate and Profit Factor
- 95% threshold optimal in ≥70% of windows
- Consistency rate ≥70% (% windows meeting success criteria)

If PASS → Proceed to Stage 4 (Statistical Testing)
If FAIL → STOP (threshold is overfitted, not robust)

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
from walk_forward_validator import (
    walk_forward_validation,
    calculate_stability_metrics,
    print_stability_report,
    assess_stability
)

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_PATH = PROJECT_ROOT / 'config' / 'optimization_config.yaml'
BASE_DATA_DIR = PROJECT_ROOT / 'data' / 'base_data'
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
DOCS_DIR = PROJECT_ROOT / 'docs'

# Load baseline data from Stage 1
BASELINE_DATA_PATH = CHECKPOINT_DIR / 'stage1_baseline_data.csv'

# Load optimal thresholds from Stage 2
OPTIMAL_THRESHOLDS_PATH = CHECKPOINT_DIR / 'stage2_optimal_thresholds.csv'

# Test these thresholds on windows
THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]  # Reduced set for speed

# Window configuration
WINDOW_SIZE_DAYS = 30  # 1 month windows
STEP_SIZE_DAYS = 15    # 15-day step (50% overlap)

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
    df['Entry Time'] = pd.to_datetime(df['Entry Time'])
    df['Exit Time'] = pd.to_datetime(df['Exit Time'])
    print(f"   ✓ Loaded {len(df):,} trades")
    return df


def load_optimal_thresholds():
    """Load optimal thresholds from Stage 2"""
    print("\n📂 Loading optimal thresholds from Stage 2...")
    df = pd.read_csv(OPTIMAL_THRESHOLDS_PATH)
    buy_threshold = df[df['trade_type'] == 'Buy']['optimal_threshold'].values[0]
    sell_threshold = df[df['trade_type'] == 'Sell']['optimal_threshold'].values[0]
    print(f"   ✓ Buy optimal: {buy_threshold:.0%}")
    print(f"   ✓ Sell optimal: {sell_threshold:.0%}")
    return buy_threshold, sell_threshold


def plot_window_performance(walk_forward_results: pd.DataFrame,
                           target_threshold: float,
                           trade_type: str,
                           save_path: str):
    """
    Plot performance across windows for target threshold.
    """

    # Filter to target threshold
    threshold_data = walk_forward_results[
        walk_forward_results['threshold'] == target_threshold
    ].copy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Win Rate across windows
    ax = axes[0, 0]
    ax.plot(threshold_data['window_id'], threshold_data['win_rate_pct'],
            marker='o', linewidth=2, markersize=8, color='steelblue')
    ax.axhline(y=52, color='green', linestyle='--', label='Target: 52%')
    ax.axhline(y=threshold_data['win_rate_pct'].mean(), color='red', linestyle=':',
               label=f'Mean: {threshold_data["win_rate_pct"].mean():.2f}%')
    ax.set_xlabel('Window ID')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title(f'{trade_type} - Win Rate Stability Across Windows ({target_threshold:.0%})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Profit Factor across windows
    ax = axes[0, 1]
    ax.plot(threshold_data['window_id'], threshold_data['profit_factor'],
            marker='o', linewidth=2, markersize=8, color='green')
    ax.axhline(y=1.25, color='green', linestyle='--', label='Target: 1.25')
    ax.axhline(y=threshold_data['profit_factor'].mean(), color='red', linestyle=':',
               label=f'Mean: {threshold_data["profit_factor"].mean():.2f}')
    ax.set_xlabel('Window ID')
    ax.set_ylabel('Profit Factor')
    ax.set_title(f'{trade_type} - Profit Factor Stability')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Sharpe Ratio across windows
    ax = axes[1, 0]
    ax.plot(threshold_data['window_id'], threshold_data['sharpe_ratio'],
            marker='o', linewidth=2, markersize=8, color='purple')
    ax.axhline(y=1.5, color='green', linestyle='--', label='Target: 1.5')
    ax.axhline(y=threshold_data['sharpe_ratio'].mean(), color='red', linestyle=':',
               label=f'Mean: {threshold_data["sharpe_ratio"].mean():.2f}')
    ax.set_xlabel('Window ID')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title(f'{trade_type} - Sharpe Ratio Stability')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Optimal threshold distribution
    ax = axes[1, 1]

    # Find best threshold for each window
    best_thresholds = []
    for window_id in walk_forward_results['window_id'].unique():
        window_data = walk_forward_results[
            walk_forward_results['window_id'] == window_id
        ]
        if 'error' in window_data.columns:
            window_data = window_data[~window_data['error'].notna()]

        if len(window_data) > 0:
            best_idx = window_data['win_rate_pct'].idxmax()
            best_threshold = window_data.loc[best_idx, 'threshold']
            best_thresholds.append(best_threshold)

    # Plot histogram
    ax.hist(best_thresholds, bins=len(THRESHOLDS), edgecolor='black', alpha=0.7)
    ax.axvline(x=target_threshold, color='red', linestyle='--', linewidth=2,
               label=f'Stage 2 Optimal: {target_threshold:.0%}')
    ax.set_xlabel('Optimal Threshold')
    ax.set_ylabel('Number of Windows')
    ax.set_title(f'{trade_type} - Optimal Threshold Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {save_path}")
    plt.close()


def generate_walk_forward_report(buy_stability: Dict,
                                 sell_stability: Dict,
                                 buy_passes: bool,
                                 sell_passes: bool,
                                 buy_reason: str,
                                 sell_reason: str,
                                 output_path: str):
    """Generate Stage 3 walk-forward validation report"""

    report = f"""# Stage 3: Walk-Forward Validation Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Method**: Rolling window validation with {WINDOW_SIZE_DAYS}-day windows
**Purpose**: Verify 95% threshold stability across different time periods

---

## 📋 METHODOLOGY

**Walk-Forward Validation** tests if the optimal threshold from Stage 2 (95%) is:
- **Stable**: Performance doesn't vary wildly across different months
- **Robust**: Consistently optimal in different market conditions
- **Not overfitted**: Works on periods not used for optimization

**Approach:**
1. Split validation period (2024-01-01 to 2024-06-30) into rolling {WINDOW_SIZE_DAYS}-day windows
2. Step by {STEP_SIZE_DAYS} days between windows ({STEP_SIZE_DAYS/WINDOW_SIZE_DAYS*100:.0f}% overlap)
3. For each window, test all thresholds: {", ".join([f"{t:.0%}" for t in THRESHOLDS])}
4. Find optimal threshold for each window
5. Check if 95% is consistently optimal

**Success Criteria:**
- Coefficient of Variation (CV) < 10% for Win Rate and Profit Factor
- 95% threshold optimal in ≥70% of windows
- Consistency rate ≥70% (windows meeting success targets)

---

## 🎯 BUY TRADES STABILITY

### Overall Performance Across {buy_stability['total_windows']} Windows
| Metric | Value |
|--------|-------|
| **Avg Win Rate** | {buy_stability['avg_win_rate']:.2f}% |
| **Avg Profit Factor** | {buy_stability['avg_profit_factor']:.2f} |
| **Avg Sharpe Ratio** | {buy_stability['avg_sharpe_ratio']:.2f} |

### Stability Metrics (CV = Coefficient of Variation)
| Metric | CV | Target | Status |
|--------|----|---------| -------|
| **Win Rate** | {buy_stability['cv_win_rate_pct']:.2f}% | <10% | {'✅' if buy_stability['cv_win_rate_pct'] < 10 else '❌'} |
| **Profit Factor** | {buy_stability['cv_profit_factor_pct']:.2f}% | <10% | {'✅' if buy_stability['cv_profit_factor_pct'] < 10 else '❌'} |
| **Sharpe Ratio** | {buy_stability['cv_sharpe_ratio_pct']:.2f}% | <10% | {'✅' if buy_stability['cv_sharpe_ratio_pct'] < 10 else '⚠️'} |

**Interpretation:**
- CV < 5%: Very stable
- CV 5-10%: Stable (acceptable)
- CV > 10%: Unstable (overfitting risk)

### Consistency (Windows Meeting Success Criteria)
| Criteria | Windows | Rate | Status |
|----------|---------|------|--------|
| **Win Rate ≥52%** | {buy_stability['windows_meeting_wr_target']}/{buy_stability['total_windows']} | {buy_stability['windows_meeting_wr_target']/buy_stability['total_windows']*100:.1f}% | {'✅' if buy_stability['windows_meeting_wr_target']/buy_stability['total_windows'] >= 0.7 else '❌'} |
| **Profit Factor ≥1.25** | {buy_stability['windows_meeting_pf_target']}/{buy_stability['total_windows']} | {buy_stability['windows_meeting_pf_target']/buy_stability['total_windows']*100:.1f}% | {'✅' if buy_stability['windows_meeting_pf_target']/buy_stability['total_windows'] >= 0.7 else '❌'} |
| **Sharpe Ratio ≥1.5** | {buy_stability['windows_meeting_sharpe_target']}/{buy_stability['total_windows']} | {buy_stability['windows_meeting_sharpe_target']/buy_stability['total_windows']*100:.1f}% | {'✅' if buy_stability['windows_meeting_sharpe_target']/buy_stability['total_windows'] >= 0.7 else '❌'} |
| **Overall Consistency** | - | {buy_stability['consistency_rate_pct']:.1f}% | {'✅' if buy_stability['consistency_rate_pct'] >= 70 else '❌'} |

### Optimality (Is 95% Best Across Windows?)
- **Windows where 95% was optimal**: {buy_stability['windows_where_optimal']}/{buy_stability['total_windows']} ({buy_stability['optimal_rate_pct']:.1f}%)
- **Target**: ≥70% → {'✅ PASS' if buy_stability['optimal_rate_pct'] >= 70 else '❌ FAIL'}

### Win Rate Range
- **Min**: {buy_stability['min_win_rate']:.2f}%
- **Max**: {buy_stability['max_win_rate']:.2f}%
- **Range**: {buy_stability['win_rate_range']:.2f}%

---

## 🎯 SELL TRADES STABILITY

### Overall Performance Across {sell_stability['total_windows']} Windows
| Metric | Value |
|--------|-------|
| **Avg Win Rate** | {sell_stability['avg_win_rate']:.2f}% |
| **Avg Profit Factor** | {sell_stability['avg_profit_factor']:.2f} |
| **Avg Sharpe Ratio** | {sell_stability['avg_sharpe_ratio']:.2f} |

### Stability Metrics (CV = Coefficient of Variation)
| Metric | CV | Target | Status |
|--------|----|---------| -------|
| **Win Rate** | {sell_stability['cv_win_rate_pct']:.2f}% | <10% | {'✅' if sell_stability['cv_win_rate_pct'] < 10 else '❌'} |
| **Profit Factor** | {sell_stability['cv_profit_factor_pct']:.2f}% | <10% | {'✅' if sell_stability['cv_profit_factor_pct'] < 10 else '❌'} |
| **Sharpe Ratio** | {sell_stability['cv_sharpe_ratio_pct']:.2f}% | <10% | {'✅' if sell_stability['cv_sharpe_ratio_pct'] < 10 else '⚠️'} |

### Consistency (Windows Meeting Success Criteria)
| Criteria | Windows | Rate | Status |
|----------|---------|------|--------|
| **Win Rate ≥52%** | {sell_stability['windows_meeting_wr_target']}/{sell_stability['total_windows']} | {sell_stability['windows_meeting_wr_target']/sell_stability['total_windows']*100:.1f}% | {'✅' if sell_stability['windows_meeting_wr_target']/sell_stability['total_windows'] >= 0.7 else '❌'} |
| **Profit Factor ≥1.25** | {sell_stability['windows_meeting_pf_target']}/{sell_stability['total_windows']} | {sell_stability['windows_meeting_pf_target']/sell_stability['total_windows']*100:.1f}% | {'✅' if sell_stability['windows_meeting_pf_target']/sell_stability['total_windows'] >= 0.7 else '❌'} |
| **Sharpe Ratio ≥1.5** | {sell_stability['windows_meeting_sharpe_target']}/{sell_stability['total_windows']} | {sell_stability['windows_meeting_sharpe_target']/sell_stability['total_windows']*100:.1f}% | {'✅' if sell_stability['windows_meeting_sharpe_target']/sell_stability['total_windows'] >= 0.7 else '❌'} |
| **Overall Consistency** | - | {sell_stability['consistency_rate_pct']:.1f}% | {'✅' if sell_stability['consistency_rate_pct'] >= 70 else '❌'} |

### Optimality (Is 95% Best Across Windows?)
- **Windows where 95% was optimal**: {sell_stability['windows_where_optimal']}/{sell_stability['total_windows']} ({sell_stability['optimal_rate_pct']:.1f}%)
- **Target**: ≥70% → {'✅ PASS' if sell_stability['optimal_rate_pct'] >= 70 else '❌ FAIL'}

### Win Rate Range
- **Min**: {sell_stability['min_win_rate']:.2f}%
- **Max**: {sell_stability['max_win_rate']:.2f}%
- **Range**: {sell_stability['win_rate_range']:.2f}%

---

## 🚦 DECISION GATE

**Buy Trades**: {'✅ PASS' if buy_passes else '❌ FAIL'}
- Reason: {buy_reason}

**Sell Trades**: {'✅ PASS' if sell_passes else '❌ FAIL'}
- Reason: {sell_reason}

"""

    overall_pass = buy_passes and sell_passes

    if overall_pass:
        report += """
### ✅ RECOMMENDATION: PROCEED TO STAGE 4

**95% threshold is stable and robust across time windows.**

**Next Step**: Statistical Validation (Bootstrap Testing)
- Verify improvement over baseline is statistically significant (p < 0.05)
- Calculate confidence intervals
- Run: `python scripts/04_statistical_validation.py`
"""
    else:
        report += """
### ❌ RECOMMENDATION: STOP - THRESHOLD NOT STABLE

**95% threshold is NOT stable across time windows.**

**The optimal threshold from Stage 2 appears to be overfitted to the full validation period.**

**Options:**
1. **Re-run Stage 2 with different threshold range** (e.g., test 85%, 92%, 97%)
2. **Accept a more stable threshold** (check which threshold has best stability)
3. **Skip exit optimization** and proceed to Stage 5 (Entry Filter Optimization)

**Do NOT proceed to final testing** without addressing stability issues.
"""

    report += f"""

---

## 📁 OUTPUTS

**Checkpoint**: `checkpoints/stage3_walk_forward_results.csv`
**Visualizations**: `docs/stage3_window_stability_*.png`
**Stability Metrics**: `checkpoints/stage3_stability_metrics.csv`

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
    """Run Stage 3: Walk-Forward Validation"""

    print("="*70)
    print("STAGE 3: WALK-FORWARD VALIDATION")
    print("="*70)

    # Load configuration
    config = load_config()

    # Load data
    baseline_data = load_baseline_data()
    buy_optimal, sell_optimal = load_optimal_thresholds()

    # =========================================================================
    # WALK-FORWARD VALIDATION: BUY TRADES
    # =========================================================================

    print(f"\n{'='*70}")
    print("WALK-FORWARD VALIDATION: BUY TRADES")
    print("="*70)

    buy_wf_results = walk_forward_validation(
        baseline_data,
        THRESHOLDS,
        str(BASE_DATA_DIR),
        window_size_days=WINDOW_SIZE_DAYS,
        step_size_days=STEP_SIZE_DAYS,
        trade_type='Buy'
    )

    # Calculate stability metrics for optimal threshold
    buy_stability = calculate_stability_metrics(buy_wf_results, buy_optimal)
    print_stability_report(buy_stability)

    # Assess stability
    buy_passes, buy_reason = assess_stability(buy_stability)

    # =========================================================================
    # WALK-FORWARD VALIDATION: SELL TRADES
    # =========================================================================

    print(f"\n{'='*70}")
    print("WALK-FORWARD VALIDATION: SELL TRADES")
    print("="*70)

    sell_wf_results = walk_forward_validation(
        baseline_data,
        THRESHOLDS,
        str(BASE_DATA_DIR),
        window_size_days=WINDOW_SIZE_DAYS,
        step_size_days=STEP_SIZE_DAYS,
        trade_type='Sell'
    )

    # Calculate stability metrics
    sell_stability = calculate_stability_metrics(sell_wf_results, sell_optimal)
    print_stability_report(sell_stability)

    # Assess stability
    sell_passes, sell_reason = assess_stability(sell_stability)

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================

    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print("="*70)

    # Save walk-forward results
    all_wf_results = pd.concat([buy_wf_results, sell_wf_results], ignore_index=True)
    all_wf_results.to_csv(CHECKPOINT_DIR / 'stage3_walk_forward_results.csv', index=False)
    print(f"   ✓ Saved walk-forward results: {len(all_wf_results):,} records")

    # Save stability metrics
    stability_df = pd.DataFrame([
        {'trade_type': 'Buy', **buy_stability},
        {'trade_type': 'Sell', **sell_stability}
    ])
    stability_df.to_csv(CHECKPOINT_DIR / 'stage3_stability_metrics.csv', index=False)
    print(f"   ✓ Saved stability metrics")

    # =========================================================================
    # GENERATE VISUALIZATIONS
    # =========================================================================

    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    plot_window_performance(
        buy_wf_results,
        buy_optimal,
        'Buy',
        str(DOCS_DIR / 'stage3_window_stability_buy.png')
    )

    plot_window_performance(
        sell_wf_results,
        sell_optimal,
        'Sell',
        str(DOCS_DIR / 'stage3_window_stability_sell.png')
    )

    # =========================================================================
    # GENERATE REPORT
    # =========================================================================

    print(f"\n{'='*70}")
    print("GENERATING VALIDATION REPORT")
    print("="*70)

    generate_walk_forward_report(
        buy_stability,
        sell_stability,
        buy_passes,
        sell_passes,
        buy_reason,
        sell_reason,
        str(DOCS_DIR / 'stage3_walk_forward_report.md')
    )

    # =========================================================================
    # DECISION
    # =========================================================================

    print(f"\n{'='*70}")
    print("✅ STAGE 3 COMPLETE" if (buy_passes and sell_passes) else "⚠️ STAGE 3 COMPLETE - ISSUES DETECTED")
    print("="*70)

    overall_pass = buy_passes and sell_passes

    print(f"\n🎯 Results:")
    print(f"   Buy: {'✅ STABLE' if buy_passes else '❌ UNSTABLE'} - {buy_reason}")
    print(f"   Sell: {'✅ STABLE' if sell_passes else '❌ UNSTABLE'} - {sell_reason}")

    print(f"\n📁 Outputs:")
    print(f"   Report: docs/stage3_walk_forward_report.md")
    print(f"   Checkpoint: checkpoints/stage3_walk_forward_results.csv")
    print(f"   Visualizations: docs/stage3_window_stability_*.png")

    if overall_pass:
        print(f"\n🚦 Decision: ✅ PROCEED TO STAGE 4")
        print(f"   95% threshold is stable and robust")
        print(f"   Next: Statistical validation (bootstrap testing)")
    else:
        print(f"\n🚦 Decision: ⛔ STOP - STABILITY ISSUES")
        print(f"   95% threshold is NOT stable across windows")
        print(f"   Review: docs/stage3_walk_forward_report.md for details")

    return overall_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
