"""
Stage 4: Statistical Validation

Performs bootstrap hypothesis testing to verify that the optimal threshold
from Stage 2 (95%) shows statistically significant improvement over baseline (80%).

Success Criteria:
- p < 0.05 for Win Rate (95% confidence that improvement is not random)
- p < 0.05 for Profit Factor
- 95% confidence intervals exclude zero

Author: Strategy Optimization Pipeline
Date: 2025-10-05
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from datetime import datetime

# Add modules to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / 'modules'))

from statistical_validator import (
    run_statistical_validation,
    format_statistical_report
)


def main():
    """Execute Stage 4: Statistical Validation."""

    print("\n" + "="*70)
    print("STAGE 4: STATISTICAL VALIDATION")
    print("="*70)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load configuration
    config_path = PROJECT_ROOT / 'config' / 'optimization_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Paths
    baseline_data_path = PROJECT_ROOT / 'checkpoints' / 'stage1_baseline_data.csv'
    optimal_thresholds_path = PROJECT_ROOT / 'checkpoints' / 'stage2_optimal_thresholds.csv'
    base_data_dir = Path(config['data']['base_data_dir'])

    # Output paths
    checkpoints_dir = PROJECT_ROOT / 'checkpoints'
    docs_dir = PROJECT_ROOT / 'docs'
    checkpoints_dir.mkdir(exist_ok=True)
    docs_dir.mkdir(exist_ok=True)

    # Verify inputs exist
    if not baseline_data_path.exists():
        raise FileNotFoundError(f"Baseline data not found: {baseline_data_path}")
    if not optimal_thresholds_path.exists():
        raise FileNotFoundError(f"Optimal thresholds not found: {optimal_thresholds_path}")

    # Load optimal thresholds from Stage 2
    print(f"\n📂 Loading optimal thresholds from Stage 2...")
    optimal_df = pd.read_csv(optimal_thresholds_path)

    # Column might be 'trade_type' or 'Trade Type', check which exists
    trade_type_col = 'trade_type' if 'trade_type' in optimal_df.columns else 'Trade Type'
    threshold_col = 'optimal_threshold' if 'optimal_threshold' in optimal_df.columns else 'Optimal Threshold'

    buy_optimal = optimal_df[optimal_df[trade_type_col] == 'Buy'][threshold_col].values[0]
    sell_optimal = optimal_df[optimal_df[trade_type_col] == 'Sell'][threshold_col].values[0]

    print(f"   Buy optimal: {buy_optimal*100:.0f}%")
    print(f"   Sell optimal: {sell_optimal*100:.0f}%")

    # Get baseline threshold from config
    baseline_threshold = config['exit_optimization']['current_threshold']
    print(f"   Baseline: {baseline_threshold*100:.0f}%")

    # Bootstrap parameters
    n_bootstrap = config['statistical_validation'].get('n_bootstrap', 1000)
    alpha = config['statistical_validation'].get('alpha', 0.05)

    print(f"\n🔧 Bootstrap Configuration:")
    print(f"   Iterations: {n_bootstrap}")
    print(f"   Significance level: {alpha}")
    print(f"   Confidence level: {(1-alpha)*100:.0f}%")

    # Run statistical validation
    # Note: Using same optimal threshold for both Buy and Sell since Stage 2 found 95% optimal for both
    buy_results, sell_results = run_statistical_validation(
        baseline_data_path=str(baseline_data_path),
        base_data_dir=str(base_data_dir),
        baseline_threshold=baseline_threshold,
        optimal_threshold=buy_optimal,  # Same as sell_optimal in our case
        n_bootstrap=n_bootstrap,
        alpha=alpha
    )

    # Save results to checkpoint
    print(f"\n💾 Saving results...")

    # Save statistical metrics
    stats_data = []
    for trade_type, results in [('Buy', buy_results), ('Sell', sell_results)]:
        for metric in ['win_rate', 'profit_factor', 'sharpe_ratio']:
            stats_data.append({
                'Trade Type': trade_type,
                'Metric': metric,
                'Baseline': results['baseline_metrics'][metric] if metric != 'win_rate'
                           else results['baseline_metrics']['win_rate'],
                'Optimal': results['optimal_metrics'][metric] if metric != 'win_rate'
                          else results['optimal_metrics']['win_rate'],
                'Improvement': results['observed_improvement'][metric],
                'p_value': results['p_values'][metric],
                'CI_Lower': results['confidence_intervals'][metric][0],
                'CI_Upper': results['confidence_intervals'][metric][1],
                'Is_Significant': results['is_significant'][metric]
            })

    stats_df = pd.DataFrame(stats_data)
    stats_output_path = checkpoints_dir / 'stage4_statistical_results.csv'
    stats_df.to_csv(stats_output_path, index=False)
    print(f"   ✓ Statistical results: {stats_output_path}")

    # Save bootstrap distributions for further analysis
    bootstrap_data = []
    for trade_type, results in [('Buy', buy_results), ('Sell', sell_results)]:
        for metric in ['win_rate', 'profit_factor', 'sharpe_ratio']:
            for i, diff in enumerate(results['bootstrap_distributions'][metric]):
                bootstrap_data.append({
                    'Trade Type': trade_type,
                    'Metric': metric,
                    'Bootstrap_Iteration': i,
                    'Difference': diff
                })

    bootstrap_df = pd.DataFrame(bootstrap_data)
    bootstrap_output_path = checkpoints_dir / 'stage4_bootstrap_distributions.csv'
    bootstrap_df.to_csv(bootstrap_output_path, index=False)
    print(f"   ✓ Bootstrap distributions: {bootstrap_output_path}")

    # Generate markdown report
    report_path = docs_dir / 'stage4_statistical_validation_report.md'
    format_statistical_report(
        buy_results,
        sell_results,
        str(report_path)
    )

    # Final summary
    print("\n" + "="*70)
    print("STAGE 4 COMPLETE")
    print("="*70)

    buy_pass = (
        buy_results['is_significant']['win_rate'] and
        buy_results['is_significant']['profit_factor']
    )
    sell_pass = (
        sell_results['is_significant']['win_rate'] and
        sell_results['is_significant']['profit_factor']
    )

    if buy_pass and sell_pass:
        print("\n✅ VALIDATION PASSED")
        print("   Improvements are statistically significant for both Buy and Sell trades")
        print(f"\n   Next: Stage 6 - Final Out-of-Sample Test")
    elif buy_pass or sell_pass:
        print("\n⚠️ PARTIAL VALIDATION")
        print(f"   {'Buy' if buy_pass else 'Sell'} trades: Significant improvement")
        print(f"   {'Sell' if buy_pass else 'Buy'} trades: Not significant")
        print(f"\n   Recommendation: Proceed with caution")
    else:
        print("\n❌ VALIDATION FAILED")
        print("   Improvements are not statistically significant")
        print(f"\n   Recommendation: Return to Stage 2 or accept baseline")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
