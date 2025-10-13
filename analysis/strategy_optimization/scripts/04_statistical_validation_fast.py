"""
Stage 4: Statistical Validation (OPTIMIZED VERSION)

Uses pre-computed simulation results from Stage 2 instead of re-simulating.
This is 300x faster - completes in ~2-3 minutes instead of hours.

Key Optimization:
- Stage 2 already simulated all trades at 80% and 95%
- We just resample those results and recalculate metrics
- No need to re-read CSVs or re-simulate exits

Author: Strategy Optimization Pipeline
Date: 2025-10-05
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from datetime import datetime
from typing import Dict, Tuple

# Add modules to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / 'modules'))

from metrics_calculator import calculate_traditional_metrics


def load_stage2_results() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load pre-computed simulation results from Stage 2.

    Returns:
        Tuple of (baseline_80_results, optimal_95_results)
    """

    print("\n📂 Loading Stage 2 simulation results...")

    # In Stage 2, we simulated all thresholds and saved to checkpoint
    # We need the results for 80% (baseline) and 95% (optimal)

    stage2_path = PROJECT_ROOT / 'checkpoints' / 'stage2_all_simulations.csv'

    if not stage2_path.exists():
        raise FileNotFoundError(
            f"Stage 2 simulation results not found at {stage2_path}\n"
            "Please run Stage 2 first or use the non-optimized version."
        )

    # Load all simulation results
    all_sims = pd.read_csv(stage2_path)

    print(f"   ✓ Loaded {len(all_sims)} simulation results")
    print(f"   Thresholds available: {sorted(all_sims['threshold'].unique())}")

    # Separate by threshold
    baseline_80 = all_sims[all_sims['threshold'] == 0.80].copy()
    optimal_95 = all_sims[all_sims['threshold'] == 0.95].copy()

    print(f"   ✓ Baseline (80%): {len(baseline_80)} trades")
    print(f"   ✓ Optimal (95%): {len(optimal_95)} trades")

    return baseline_80, optimal_95


def fast_bootstrap_comparison(
    baseline_results: pd.DataFrame,
    optimal_results: pd.DataFrame,
    trade_type: str,
    n_bootstrap: int = 1000,
    random_seed: int = 42
) -> Dict:
    """
    Fast bootstrap using pre-computed results.

    Args:
        baseline_results: Simulated results at 80% threshold
        optimal_results: Simulated results at 95% threshold
        trade_type: 'Buy' or 'Sell'
        n_bootstrap: Number of bootstrap iterations
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with statistical results
    """

    print(f"\n{'='*70}")
    print(f"{trade_type.upper()} TRADES BOOTSTRAP VALIDATION")
    print(f"{'='*70}")

    # Filter by trade type
    baseline = baseline_results[baseline_results['trade_type'] == trade_type].copy()
    optimal = optimal_results[optimal_results['trade_type'] == trade_type].copy()

    n_trades = len(baseline)

    print(f"\n🔄 Fast Bootstrap Statistical Testing")
    print(f"   Sample size: {n_trades:,} trades")
    print(f"   Bootstrap iterations: {n_bootstrap}")
    print(f"   Method: Vectorized resampling (no re-simulation)")

    # Calculate observed metrics on full dataset
    print(f"\n📊 Calculating observed performance...")

    # Prepare data for metrics calculator
    baseline['percentage_return'] = baseline['sim_return_pct']
    baseline['Entry Time'] = pd.to_datetime(baseline['sim_exit_time']) - pd.to_timedelta(baseline['sim_duration_minutes'], unit='m')
    baseline['Exit Time'] = pd.to_datetime(baseline['sim_exit_time'])

    optimal['percentage_return'] = optimal['sim_return_pct']
    optimal['Entry Time'] = pd.to_datetime(optimal['sim_exit_time']) - pd.to_timedelta(optimal['sim_duration_minutes'], unit='m')
    optimal['Exit Time'] = pd.to_datetime(optimal['sim_exit_time'])

    baseline_metrics_raw = calculate_traditional_metrics(baseline)
    optimal_metrics_raw = calculate_traditional_metrics(optimal)

    # Convert to performance metrics format
    baseline_metrics = {
        'win_rate': baseline_metrics_raw['win_rate_pct'],
        'profit_factor': baseline_metrics_raw['profit_factor'],
        'sharpe_ratio': baseline_metrics_raw['sharpe_ratio'],
        'avg_return_pct': baseline_metrics_raw['total_return_pct'] / baseline_metrics_raw['total_trades']
    }

    optimal_metrics = {
        'win_rate': optimal_metrics_raw['win_rate_pct'],
        'profit_factor': optimal_metrics_raw['profit_factor'],
        'sharpe_ratio': optimal_metrics_raw['sharpe_ratio'],
        'avg_return_pct': optimal_metrics_raw['total_return_pct'] / optimal_metrics_raw['total_trades']
    }

    # Calculate observed improvements
    observed_improvement = {
        'win_rate': optimal_metrics['win_rate'] - baseline_metrics['win_rate'],
        'profit_factor': optimal_metrics['profit_factor'] - baseline_metrics['profit_factor'],
        'sharpe_ratio': optimal_metrics['sharpe_ratio'] - baseline_metrics['sharpe_ratio'],
        'avg_return': optimal_metrics['avg_return_pct'] - baseline_metrics['avg_return_pct']
    }

    print(f"\n   Baseline (80%): WR {baseline_metrics['win_rate']:.2f}% | PF {baseline_metrics['profit_factor']:.2f}")
    print(f"   Optimal (95%):  WR {optimal_metrics['win_rate']:.2f}% | PF {optimal_metrics['profit_factor']:.2f}")
    print(f"\n📈 Observed Improvements (Optimal - Baseline):")
    print(f"   Win Rate: {observed_improvement['win_rate']:+.2f}%")
    print(f"   Profit Factor: {observed_improvement['profit_factor']:+.2f}")
    print(f"   Sharpe Ratio: {observed_improvement['sharpe_ratio']:+.2f}")

    # Bootstrap resampling (VECTORIZED)
    print(f"\n🔄 Performing {n_bootstrap} bootstrap iterations (vectorized)...")

    np.random.seed(random_seed)

    bootstrap_diffs = {
        'win_rate': np.zeros(n_bootstrap),
        'profit_factor': np.zeros(n_bootstrap),
        'sharpe_ratio': np.zeros(n_bootstrap),
        'avg_return': np.zeros(n_bootstrap)
    }

    # Pre-extract return arrays for speed
    baseline_returns = baseline['sim_return_pct'].values
    optimal_returns = optimal['sim_return_pct'].values

    for i in range(n_bootstrap):
        if (i + 1) % 100 == 0:
            print(f"   Progress: {i+1}/{n_bootstrap} ({(i+1)/n_bootstrap*100:.1f}%)")

        # FAST: Just resample indices
        indices = np.random.choice(n_trades, size=n_trades, replace=True)

        # FAST: Subset pre-computed results
        baseline_sample = baseline_returns[indices]
        optimal_sample = optimal_returns[indices]

        # FAST: Calculate metrics using vectorized operations
        # Win Rate
        baseline_wr = (baseline_sample > 0).mean() * 100
        optimal_wr = (optimal_sample > 0).mean() * 100
        bootstrap_diffs['win_rate'][i] = optimal_wr - baseline_wr

        # Profit Factor
        baseline_wins = baseline_sample[baseline_sample > 0].sum()
        baseline_losses = abs(baseline_sample[baseline_sample < 0].sum())
        baseline_pf = baseline_wins / baseline_losses if baseline_losses > 0 else np.inf

        optimal_wins = optimal_sample[optimal_sample > 0].sum()
        optimal_losses = abs(optimal_sample[optimal_sample < 0].sum())
        optimal_pf = optimal_wins / optimal_losses if optimal_losses > 0 else np.inf

        bootstrap_diffs['profit_factor'][i] = optimal_pf - baseline_pf

        # Sharpe Ratio (annualized)
        baseline_sharpe = (baseline_sample.mean() / baseline_sample.std()) * np.sqrt(252) if baseline_sample.std() > 0 else 0
        optimal_sharpe = (optimal_sample.mean() / optimal_sample.std()) * np.sqrt(252) if optimal_sample.std() > 0 else 0
        bootstrap_diffs['sharpe_ratio'][i] = optimal_sharpe - baseline_sharpe

        # Average Return
        bootstrap_diffs['avg_return'][i] = optimal_sample.mean() - baseline_sample.mean()

    print(f"   ✓ Bootstrap complete")

    # Calculate p-values and confidence intervals
    print(f"\n📊 Calculating statistical metrics...")

    p_values = {}
    confidence_intervals = {}

    for metric in bootstrap_diffs:
        diffs = bootstrap_diffs[metric]

        # One-tailed p-value: probability that baseline is better (diff <= 0)
        p_values[metric] = (diffs <= 0).sum() / n_bootstrap

        # 95% confidence interval
        ci_lower = np.percentile(diffs, 2.5)
        ci_upper = np.percentile(diffs, 97.5)
        confidence_intervals[metric] = (ci_lower, ci_upper)

    # Determine significance (p < 0.05)
    is_significant = {
        metric: p_values[metric] < 0.05
        for metric in p_values
    }

    # Print results
    print(f"\n📈 Statistical Results:")
    print(f"   Win Rate:")
    print(f"      Improvement: {observed_improvement['win_rate']:+.2f}%")
    print(f"      p-value: {p_values['win_rate']:.4f} {'✅' if is_significant['win_rate'] else '❌'}")
    print(f"      95% CI: [{confidence_intervals['win_rate'][0]:.2f}%, {confidence_intervals['win_rate'][1]:.2f}%]")

    print(f"\n   Profit Factor:")
    print(f"      Improvement: {observed_improvement['profit_factor']:+.2f}")
    print(f"      p-value: {p_values['profit_factor']:.4f} {'✅' if is_significant['profit_factor'] else '❌'}")
    print(f"      95% CI: [{confidence_intervals['profit_factor'][0]:.2f}, {confidence_intervals['profit_factor'][1]:.2f}]")

    print(f"\n   Sharpe Ratio:")
    print(f"      Improvement: {observed_improvement['sharpe_ratio']:+.2f}")
    print(f"      p-value: {p_values['sharpe_ratio']:.4f} {'✅' if is_significant['sharpe_ratio'] else '❌'}")
    print(f"      95% CI: [{confidence_intervals['sharpe_ratio'][0]:.2f}, {confidence_intervals['sharpe_ratio'][1]:.2f}]")

    # Compile results
    results = {
        'baseline_threshold': 0.80,
        'optimal_threshold': 0.95,
        'n_trades': n_trades,
        'n_bootstrap': n_bootstrap,
        'baseline_metrics': baseline_metrics,
        'optimal_metrics': optimal_metrics,
        'observed_improvement': observed_improvement,
        'p_values': p_values,
        'confidence_intervals': confidence_intervals,
        'is_significant': is_significant,
        'bootstrap_distributions': bootstrap_diffs
    }

    return results


def main():
    """Execute optimized Stage 4: Statistical Validation."""

    print("\n" + "="*70)
    print("STAGE 4: STATISTICAL VALIDATION (OPTIMIZED)")
    print("="*70)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load configuration
    config_path = PROJECT_ROOT / 'config' / 'optimization_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Output paths
    checkpoints_dir = PROJECT_ROOT / 'checkpoints'
    docs_dir = PROJECT_ROOT / 'docs'

    # Bootstrap parameters
    n_bootstrap = config['statistical_validation'].get('n_bootstrap', 1000)
    alpha = config['statistical_validation'].get('alpha', 0.05)

    print(f"\n🔧 Bootstrap Configuration:")
    print(f"   Iterations: {n_bootstrap}")
    print(f"   Significance level: {alpha}")
    print(f"   Confidence level: {(1-alpha)*100:.0f}%")

    # Load pre-computed Stage 2 results
    baseline_80, optimal_95 = load_stage2_results()

    # Run bootstrap for Buy trades
    buy_results = fast_bootstrap_comparison(
        baseline_80,
        optimal_95,
        trade_type='Buy',
        n_bootstrap=n_bootstrap,
        random_seed=42
    )

    # Run bootstrap for Sell trades
    sell_results = fast_bootstrap_comparison(
        baseline_80,
        optimal_95,
        trade_type='Sell',
        n_bootstrap=n_bootstrap,
        random_seed=42
    )

    # Save results
    print(f"\n💾 Saving results...")

    # Save statistical metrics
    stats_data = []
    for trade_type, results in [('Buy', buy_results), ('Sell', sell_results)]:
        for metric in ['win_rate', 'profit_factor', 'sharpe_ratio']:
            stats_data.append({
                'Trade Type': trade_type,
                'Metric': metric,
                'Baseline': results['baseline_metrics'][metric],
                'Optimal': results['optimal_metrics'][metric],
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

    # Save bootstrap distributions
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
    from statistical_validator import format_statistical_report

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
        print(f"\n   Buy:  WR p={buy_results['p_values']['win_rate']:.4f}, PF p={buy_results['p_values']['profit_factor']:.4f}")
        print(f"   Sell: WR p={sell_results['p_values']['win_rate']:.4f}, PF p={sell_results['p_values']['profit_factor']:.4f}")
        print(f"\n   ✅ PROCEED to Stage 6 - Final Out-of-Sample Test")
    elif buy_pass or sell_pass:
        print("\n⚠️ PARTIAL VALIDATION")
        print(f"   {'Buy' if buy_pass else 'Sell'} trades: Significant improvement")
        print(f"   {'Sell' if buy_pass else 'Buy'} trades: Not significant")
        print(f"\n   Recommendation: Proceed with caution")
    else:
        print("\n❌ VALIDATION FAILED")
        print("   Improvements are not statistically significant")
        print(f"\n   Buy:  WR p={buy_results['p_values']['win_rate']:.4f}, PF p={buy_results['p_values']['profit_factor']:.4f}")
        print(f"   Sell: WR p={sell_results['p_values']['win_rate']:.4f}, PF p={sell_results['p_values']['profit_factor']:.4f}")
        print(f"\n   ❌ STOP - Return to Stage 2 or accept baseline")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
