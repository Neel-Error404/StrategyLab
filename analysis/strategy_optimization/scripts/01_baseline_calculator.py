"""
Stage 1: Baseline Establishment Script
=======================================

Purpose:
--------
Calculate baseline metrics for MSE strategy on validation data (2024 H1).

Steps:
------
1. Load trade data for 24 tickers
2. Filter to validation period (2024-01-01 to 2024-06-30)
3. Enhance trades with base_data for MAE/MFE calculation
4. Calculate traditional metrics (WR, PF, Sharpe, etc.)
5. Calculate exit efficiency metrics
6. Generate visualizations
7. Create baseline report
8. Save checkpoint

Author: Strategy Optimization Pipeline
Date: 2025-10-04
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add modules to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / 'modules'))

# Import local modules
from mae_mfe_calculator import calculate_mae_mfe, get_mae_mfe_summary, print_mae_mfe_summary
from metrics_calculator import calculate_traditional_metrics, print_metrics_summary
from visualizer import create_baseline_visualizations

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_PATH = PROJECT_ROOT / 'config' / 'optimization_config.yaml'
BASE_DATA_DIR = PROJECT_ROOT / 'data' / 'base_data'
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
DOCS_DIR = PROJECT_ROOT / 'docs'

# Trade data location (from context)
TRADE_DATA_PATH = Path('/mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/StrategyLab-master/outputs/20250915_121714/mse_backtesting/2022-01-01_to_2025-08-31/data/all_trade_mereged.csv')

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_config():
    """Load optimization configuration"""
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)


def load_trade_data(config):
    """Load and filter trade data for 24 tickers and validation period"""

    print("\n📂 Loading trade data...")

    # Load all trades
    df = pd.read_csv(TRADE_DATA_PATH)
    print(f"   ✓ Loaded {len(df):,} total trades")

    # Standardize column names
    # Note: Keep 'ticker' lowercase for trade_enhancer compatibility
    # Add percentage_return as alias for Profit (%)
    df['percentage_return'] = df['Profit (%)']

    # Convert timestamps (removing timezone for comparison)
    df['Entry Time'] = pd.to_datetime(df['Entry Time']).dt.tz_localize(None)
    df['Exit Time'] = pd.to_datetime(df['Exit Time']).dt.tz_localize(None)

    # Filter to 24 configured tickers
    tickers = config['data']['tickers']
    df_filtered = df[df['ticker'].isin(tickers)].copy()
    print(f"   ✓ Filtered to {len(tickers)} configured tickers: {len(df_filtered):,} trades")

    # Filter to validation period
    val_start = pd.to_datetime(config['data']['date_splits']['validation_start'])
    val_end = pd.to_datetime(config['data']['date_splits']['validation_end'])

    df_val = df_filtered[
        (df_filtered['Entry Time'] >= val_start) &
        (df_filtered['Entry Time'] <= val_end)
    ].copy()

    print(f"   ✓ Filtered to validation period ({val_start.date()} to {val_end.date()}): {len(df_val):,} trades")

    # Summary by ticker
    print(f"\n   Trades per ticker:")
    ticker_counts = df_val['ticker'].value_counts().sort_index()
    for ticker, count in ticker_counts.items():
        print(f"      {ticker}: {count:,} trades")

    return df_val


def generate_baseline_report(metrics: dict,
                            mae_mfe_summary: dict,
                            enhanced_data: pd.DataFrame,
                            output_path: str):
    """Generate comprehensive baseline report in Markdown"""

    report = f"""# Stage 1: Baseline Establishment Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Period**: Validation Data (2024-01-01 to 2024-06-30)
**Tickers**: 24 tickers
**Total Trades Analyzed**: {len(enhanced_data):,}

---

## 📊 TRADITIONAL METRICS

### Trade Statistics
- **Total Trades**: {metrics['total_trades']:,}
- **Winning Trades**: {metrics['num_wins']:,} ({metrics['win_rate_pct']:.2f}%)
- **Losing Trades**: {metrics['num_losses']:,}
- **Breakeven Trades**: {metrics['num_breakeven']:,}

### Performance Metrics
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Win Rate** | {metrics['win_rate_pct']:.2f}% | ≥52% | {'✅' if metrics['win_rate_pct'] >= 52 else '❌'} |
| **Profit Factor** | {metrics['profit_factor']:.2f} | ≥1.25 | {'✅' if metrics['profit_factor'] >= 1.25 else '❌'} |
| **Average Win** | {metrics['avg_win_pct']:.2f}% | - | - |
| **Average Loss** | {metrics['avg_loss_pct']:.2f}% | - | - |
| **Risk-Reward Ratio** | {metrics['risk_reward_ratio']:.2f} | - | - |
| **Expectancy** | {metrics['expectancy_pct']:.2f}% per trade | - | - |

### Return Metrics
| Metric | Value |
|--------|-------|
| **Total Return** | {metrics['total_return_pct']:.2f}% |
| **Return on Capital** | {metrics['return_on_capital_pct']:.2f}% |
| **Final Equity** | ${metrics['final_equity']:,.2f} |

### Risk Metrics
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Max Drawdown** | {metrics['max_drawdown_pct']:.2f}% | ≤15% | {'✅' if metrics['max_drawdown_pct'] >= -15 else '❌'} |
| **Sharpe Ratio** | {metrics['sharpe_ratio']:.2f} | ≥1.5 | {'✅' if metrics['sharpe_ratio'] >= 1.5 else '❌'} |

### Duration Metrics
- **Average Trade Duration**: {metrics['avg_duration_hours']:.2f} hours
- **Median Trade Duration**: {metrics['median_duration_hours']:.2f} hours

### Streaks
- **Max Consecutive Wins**: {metrics['max_consecutive_wins']}
- **Max Consecutive Losses**: {metrics['max_consecutive_losses']}

---

## 🎯 MAE/MFE EXIT EFFICIENCY ANALYSIS

"""

    if 'error' not in mae_mfe_summary:
        report += f"""
### Data Coverage
- **Total Trades**: {mae_mfe_summary['total_trades']:,}
- **Valid MAE/MFE Data**: {mae_mfe_summary['valid_trades']:,} ({mae_mfe_summary['coverage_pct']:.1f}%)

### Maximum Favorable Excursion (MFE) - Best Price Available
| Metric | Value |
|--------|-------|
| **Average MFE** | {mae_mfe_summary['avg_MFE_pct']:.2f}% |
| **Median MFE** | {mae_mfe_summary['median_MFE_pct']:.2f}% |
| **Std Dev** | {mae_mfe_summary['std_MFE_pct']:.2f}% |

### Maximum Adverse Excursion (MAE) - Worst Drawdown
| Metric | Value |
|--------|-------|
| **Average MAE** | {mae_mfe_summary['avg_MAE_pct']:.2f}% |
| **Median MAE** | {mae_mfe_summary['median_MAE_pct']:.2f}% |
| **Std Dev** | {mae_mfe_summary['std_MAE_pct']:.2f}% |

### MFE Capture Ratio - % of Available Profit Captured
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Average Capture Ratio** | {mae_mfe_summary['avg_MFE_Capture_Ratio']:.2f}% | ≥70% | {'✅' if mae_mfe_summary['avg_MFE_Capture_Ratio'] >= 70 else '❌'} |
| **Median Capture Ratio** | {mae_mfe_summary['median_MFE_Capture_Ratio']:.2f}% | - | - |

### Exit Efficiency Score - Overall Exit Quality
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Average Efficiency Score** | {mae_mfe_summary['avg_Exit_Efficiency_Score']:.2f} | ≥55 | {'✅' if mae_mfe_summary['avg_Exit_Efficiency_Score'] >= 55 else '❌'} |
| **Median Efficiency Score** | {mae_mfe_summary['median_Exit_Efficiency_Score']:.2f} | - | - |

### Exit Efficiency Distribution
| Category | Count | Percentage |
|----------|-------|------------|
| **Excellent (>70)** | {mae_mfe_summary['efficiency_score_excellent']:,} | {(mae_mfe_summary['efficiency_score_excellent'] / mae_mfe_summary['valid_trades'] * 100):.1f}% |
| **Good (50-70)** | {mae_mfe_summary['efficiency_score_good']:,} | {(mae_mfe_summary['efficiency_score_good'] / mae_mfe_summary['valid_trades'] * 100):.1f}% |
| **Poor (30-50)** | {mae_mfe_summary['efficiency_score_poor']:,} | {(mae_mfe_summary['efficiency_score_poor'] / mae_mfe_summary['valid_trades'] * 100):.1f}% |
| **Terrible (<30)** | {mae_mfe_summary['efficiency_score_terrible']:,} | {(mae_mfe_summary['efficiency_score_terrible'] / mae_mfe_summary['valid_trades'] * 100):.1f}% |

### Potential Left on Table
| Metric | Value |
|--------|-------|
| **Average per Trade** | {mae_mfe_summary['avg_Potential_Left_on_Table_pct']:.2f}% |
| **Total Across All Trades** | {mae_mfe_summary['total_Potential_Left_pct']:.2f}% |

### MAE/MFE Ratio - Drawdown vs Profit Potential
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Average Ratio** | {mae_mfe_summary['avg_MAE_MFE_Ratio']:.2f} | ≤0.6 | {'✅' if mae_mfe_summary['avg_MAE_MFE_Ratio'] <= 0.6 else '❌'} |
| **Median Ratio** | {mae_mfe_summary['median_MAE_MFE_Ratio']:.2f} | - | - |
"""
    else:
        report += f"\n❌ **ERROR**: {mae_mfe_summary['error']}\n"

    # Overall assessment
    report += f"""
---

## 📋 BASELINE ASSESSMENT

### Success Criteria Check

**Traditional Metrics**:
- Win Rate ≥52%: {'✅ PASS' if metrics.get('win_rate_pct', 0) >= 52 else '❌ FAIL'}
- Profit Factor ≥1.25: {'✅ PASS' if metrics.get('profit_factor', 0) >= 1.25 else '❌ FAIL'}
- Max Drawdown ≤15%: {'✅ PASS' if metrics.get('max_drawdown_pct', -100) >= -15 else '❌ FAIL'}
- Sharpe Ratio ≥1.5: {'✅ PASS' if metrics.get('sharpe_ratio', 0) >= 1.5 else '❌ FAIL'}

"""

    if 'error' not in mae_mfe_summary:
        report += f"""**Exit Efficiency Metrics**:
- MFE Capture Ratio ≥70%: {'✅ PASS' if mae_mfe_summary.get('avg_MFE_Capture_Ratio', 0) >= 70 else '❌ FAIL'}
- Exit Efficiency Score ≥55: {'✅ PASS' if mae_mfe_summary.get('avg_Exit_Efficiency_Score', 0) >= 55 else '❌ FAIL'}
- MAE/MFE Ratio ≤0.6: {'✅ PASS' if mae_mfe_summary.get('avg_MAE_MFE_Ratio', 1) <= 0.6 else '❌ FAIL'}
"""

    report += f"""
### Key Insights

**Current State**:
- Win Rate: {metrics.get('win_rate_pct', 0):.2f}% (Target: 52%)
- Profit Factor: {metrics.get('profit_factor', 0):.2f} (Target: 1.25)
"""

    if 'error' not in mae_mfe_summary:
        report += f"""- MFE Capture Ratio: {mae_mfe_summary.get('avg_MFE_Capture_Ratio', 0):.2f}% (Target: 70%)
- Exit Efficiency Score: {mae_mfe_summary.get('avg_Exit_Efficiency_Score', 0):.2f} (Target: 55)
- Potential Left on Table: {mae_mfe_summary.get('avg_Potential_Left_on_Table_pct', 0):.2f}% per trade
"""

    report += f"""
### Recommendations

**Proceed to Stage 2 (Exit Threshold Optimization)?**
"""

    # Decision logic
    needs_exit_optimization = False
    if 'error' not in mae_mfe_summary:
        if mae_mfe_summary.get('avg_MFE_Capture_Ratio', 100) < 70:
            needs_exit_optimization = True
            report += f"\n✅ **YES** - MFE Capture Ratio is below target ({mae_mfe_summary['avg_MFE_Capture_Ratio']:.2f}% < 70%)"
        if mae_mfe_summary.get('avg_Exit_Efficiency_Score', 100) < 55:
            needs_exit_optimization = True
            report += f"\n✅ **YES** - Exit Efficiency Score is below target ({mae_mfe_summary['avg_Exit_Efficiency_Score']:.2f} < 55)"

    if not needs_exit_optimization:
        report += "\n⚠️ **REVIEW** - Exit efficiency metrics are acceptable, consider focusing on entry filters (H2)"

    report += f"""

---

## 📁 Outputs

**Checkpoint**: `checkpoints/stage1_baseline_data.csv`
**Visualizations**: `docs/` (5 charts generated)
**Raw Data**: Enhanced trade data with MAE/MFE columns saved

---

## 🚦 Next Steps

1. **Review this report** and visualizations
2. **Update PHASE2_ANALYSIS_LOG.md** with Stage 1 observations
3. **Make decision**: Proceed to Stage 2 (Exit Optimization)?
   - If YES → Run `python scripts/02_exit_threshold_optimizer.py`
   - If NO → Review strategy or proceed to Stage 3 (Entry Filters)

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
    """Run Stage 1: Baseline Establishment"""

    print("="*70)
    print("STAGE 1: BASELINE ESTABLISHMENT")
    print("="*70)

    # Load configuration
    config = load_config()
    print(f"\n✓ Configuration loaded")

    # Step 1: Load trade data
    trade_data = load_trade_data(config)

    if len(trade_data) == 0:
        print("\n❌ No trades found for validation period. Exiting.")
        return False

    # Step 2: Calculate MAE/MFE
    print(f"\n{'='*70}")
    print("CALCULATING MAE/MFE METRICS")
    print("="*70)

    def progress_callback(current, total):
        if current % 500 == 0:
            print(f"   Progress: {current:,}/{total:,} ({current/total*100:.1f}%)")

    enhanced_data = calculate_mae_mfe(
        trade_data,
        str(BASE_DATA_DIR),
        progress_callback=progress_callback
    )

    # Get MAE/MFE summary
    mae_mfe_summary = get_mae_mfe_summary(enhanced_data)
    print_mae_mfe_summary(mae_mfe_summary)

    # Step 3: Calculate traditional metrics
    print(f"\n{'='*70}")
    print("CALCULATING TRADITIONAL METRICS")
    print("="*70)

    metrics = calculate_traditional_metrics(enhanced_data)
    print_metrics_summary(metrics, title="BASELINE METRICS")

    # Step 4: Generate visualizations
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    create_baseline_visualizations(enhanced_data, str(DOCS_DIR), prefix="baseline")

    # Step 5: Save checkpoint
    print(f"\n{'='*70}")
    print("SAVING CHECKPOINT")
    print("="*70)

    checkpoint_path = CHECKPOINT_DIR / 'stage1_baseline_data.csv'
    enhanced_data.to_csv(checkpoint_path, index=False)
    print(f"   ✓ Checkpoint saved: {checkpoint_path}")
    print(f"   ✓ Saved {len(enhanced_data):,} trades with MAE/MFE data")

    # Step 6: Generate report
    print(f"\n{'='*70}")
    print("GENERATING BASELINE REPORT")
    print("="*70)

    report_path = DOCS_DIR / 'baseline_report.md'
    generate_baseline_report(metrics, mae_mfe_summary, enhanced_data, str(report_path))

    # Summary
    print(f"\n{'='*70}")
    print("✅ STAGE 1 COMPLETE")
    print("="*70)

    print(f"\n📊 Key Results:")
    print(f"   Win Rate: {metrics['win_rate_pct']:.2f}% (Target: ≥52%)")
    print(f"   Profit Factor: {metrics['profit_factor']:.2f} (Target: ≥1.25)")

    if 'error' not in mae_mfe_summary:
        print(f"   MFE Capture Ratio: {mae_mfe_summary['avg_MFE_Capture_Ratio']:.2f}% (Target: ≥70%)")
        print(f"   Exit Efficiency Score: {mae_mfe_summary['avg_Exit_Efficiency_Score']:.2f} (Target: ≥55)")

    print(f"\n📁 Outputs:")
    print(f"   Report: {report_path}")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Visualizations: {DOCS_DIR}/*.png (5 charts)")

    print(f"\n🚦 Next Steps:")
    print(f"   1. Review {report_path}")
    print(f"   2. Update PHASE2_ANALYSIS_LOG.md with observations")
    print(f"   3. Decide: Proceed to Stage 2?")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
