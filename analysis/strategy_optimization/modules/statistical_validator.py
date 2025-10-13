"""
Statistical Validator Module

Performs bootstrap testing to verify that threshold optimization improvements
are statistically significant and not due to random chance.

Author: Strategy Optimization Pipeline
Date: 2025-10-05
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import sys

# Add modules to path
sys.path.append(str(Path(__file__).parent))
from metrics_calculator import calculate_traditional_metrics
from exit_simulator import simulate_all_thresholds


def simulate_threshold_on_dataset(
    trades_df: pd.DataFrame,
    base_data_dir: str,
    threshold: float,
    desc: str = None
) -> pd.DataFrame:
    """
    Simulate a single threshold on a dataset of trades.

    Wrapper around simulate_all_thresholds that tests only one threshold.

    Args:
        trades_df: Trade records
        base_data_dir: Directory containing base data files
        threshold: Single threshold value to test (e.g., 0.95)
        desc: Optional description for progress messages

    Returns:
        DataFrame with simulation results
    """
    # simulate_all_thresholds expects a list of thresholds
    results = simulate_all_thresholds(
        trades_df,
        base_data_dir,
        thresholds=[threshold],
        trade_type_filter=None,  # Already filtered before calling
        progress_callback=None
    )

    # Filter out errors if present
    if 'error' in results.columns:
        valid_results = results[~results['error'].notna()].copy()

        # If all results had errors, raise an exception
        if len(valid_results) == 0:
            error_counts = results['error'].value_counts()
            raise ValueError(f"All simulations failed. Error counts: {error_counts.to_dict()}")

        results = valid_results

    return results


def calculate_performance_metrics(results_df: pd.DataFrame) -> Dict:
    """
    Wrapper to convert traditional metrics to performance metrics format.

    Args:
        results_df: DataFrame with simulation results (from exit_simulator)

    Returns:
        Dictionary with metrics in the format expected by bootstrap_comparison
    """
    # Debug: Check what we received
    if len(results_df) == 0:
        return {'error': 'Empty results DataFrame'}

    # Prepare DataFrame for metrics calculator
    # The exit simulator returns 'sim_return_pct', 'sim_exit_time', 'actual_exit_time'
    # We need to map these to what calculate_traditional_metrics expects

    metrics_df = results_df.copy()

    # Debug: Print columns for first call
    #print(f"\n   [DEBUG] Results DataFrame columns: {list(metrics_df.columns[:15])}")
    #print(f"   [DEBUG] Number of rows: {len(metrics_df)}")

    # Map column names
    if 'sim_return_pct' in metrics_df.columns:
        metrics_df['percentage_return'] = metrics_df['sim_return_pct']
    elif 'percentage_return' not in metrics_df.columns:
        return {'error': f'No return column found. Available columns: {list(metrics_df.columns)}'}

    # Add Entry Time and Exit Time if missing
    if 'Entry Time' not in metrics_df.columns:
        if 'actual_exit_time' in metrics_df.columns and 'sim_duration_minutes' in metrics_df.columns:
            # Calculate entry time from exit time - duration
            metrics_df['Exit Time'] = pd.to_datetime(metrics_df['sim_exit_time'])
            metrics_df['Entry Time'] = metrics_df['Exit Time'] - pd.to_timedelta(metrics_df['sim_duration_minutes'], unit='m')
        else:
            # Use dummy timestamps
            metrics_df['Entry Time'] = pd.Timestamp('2024-01-01')
            metrics_df['Exit Time'] = pd.Timestamp('2024-01-02')

    if 'Exit Time' not in metrics_df.columns:
        if 'sim_exit_time' in metrics_df.columns:
            metrics_df['Exit Time'] = pd.to_datetime(metrics_df['sim_exit_time'])
        else:
            metrics_df['Exit Time'] = pd.Timestamp('2024-01-02')

    # Call the traditional metrics calculator
    trad_metrics = calculate_traditional_metrics(metrics_df)

    if 'error' in trad_metrics:
        return trad_metrics

    # Map to expected format
    return {
        'win_rate': trad_metrics['win_rate_pct'],
        'profit_factor': trad_metrics['profit_factor'],
        'sharpe_ratio': trad_metrics['sharpe_ratio'],
        'avg_return_pct': trad_metrics['total_return_pct'] / trad_metrics['total_trades'] if trad_metrics['total_trades'] > 0 else 0,
        'total_return_pct': trad_metrics['total_return_pct'],
        'num_trades': trad_metrics['total_trades']
    }


def bootstrap_comparison(
    trades_df: pd.DataFrame,
    base_data_dir: str,
    baseline_threshold: float,
    optimal_threshold: float,
    n_bootstrap: int = 1000,
    random_seed: int = 42
) -> Dict:
    """
    Perform bootstrap statistical testing to compare two thresholds.

    Bootstrap Logic:
    1. Resample trades with replacement N times
    2. For each sample, simulate both thresholds
    3. Calculate difference in metrics (optimal - baseline)
    4. Compute p-value: probability baseline is better (one-tailed test)

    Args:
        trades_df: Trade records with enhanced data
        base_data_dir: Directory containing base data files
        baseline_threshold: Current threshold (e.g., 80%)
        optimal_threshold: Proposed optimal threshold (e.g., 95%)
        n_bootstrap: Number of bootstrap samples
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with statistical results:
        - baseline_metrics: Performance at baseline threshold
        - optimal_metrics: Performance at optimal threshold
        - improvement: Difference (optimal - baseline)
        - p_values: Statistical significance for each metric
        - confidence_intervals: 95% CI for improvements
        - is_significant: Boolean flags for each metric
    """

    np.random.seed(random_seed)
    n_trades = len(trades_df)

    print(f"\n🔄 Bootstrap Statistical Testing")
    print(f"   Comparing: {baseline_threshold*100:.0f}% (baseline) vs {optimal_threshold*100:.0f}% (optimal)")
    print(f"   Sample size: {n_trades} trades")
    print(f"   Bootstrap iterations: {n_bootstrap}")

    # Calculate observed metrics on full dataset
    print(f"\n📊 Simulating baseline threshold ({baseline_threshold*100:.0f}%)...")
    baseline_results = simulate_threshold_on_dataset(
        trades_df.copy(),
        base_data_dir,
        baseline_threshold,
        desc=f"{baseline_threshold*100:.0f}% Baseline"
    )
    baseline_metrics = calculate_performance_metrics(baseline_results)

    # Check for errors in baseline calculation
    if 'error' in baseline_metrics:
        raise ValueError(f"Baseline metrics calculation failed: {baseline_metrics['error']}")

    print(f"\n📊 Simulating optimal threshold ({optimal_threshold*100:.0f}%)...")
    optimal_results = simulate_threshold_on_dataset(
        trades_df.copy(),
        base_data_dir,
        optimal_threshold,
        desc=f"{optimal_threshold*100:.0f}% Optimal"
    )
    optimal_metrics = calculate_performance_metrics(optimal_results)

    # Check for errors in optimal calculation
    if 'error' in optimal_metrics:
        raise ValueError(f"Optimal metrics calculation failed: {optimal_metrics['error']}")

    # Calculate observed improvements
    observed_improvement = {
        'win_rate': optimal_metrics['win_rate'] - baseline_metrics['win_rate'],
        'profit_factor': optimal_metrics['profit_factor'] - baseline_metrics['profit_factor'],
        'sharpe_ratio': optimal_metrics['sharpe_ratio'] - baseline_metrics['sharpe_ratio'],
        'avg_return': optimal_metrics['avg_return_pct'] - baseline_metrics['avg_return_pct']
    }

    print(f"\n📈 Observed Improvements (Optimal - Baseline):")
    print(f"   Win Rate: {observed_improvement['win_rate']:+.2f}%")
    print(f"   Profit Factor: {observed_improvement['profit_factor']:+.2f}")
    print(f"   Sharpe Ratio: {observed_improvement['sharpe_ratio']:+.2f}")

    # Bootstrap resampling
    print(f"\n🔄 Performing {n_bootstrap} bootstrap iterations...")

    bootstrap_diffs = {
        'win_rate': [],
        'profit_factor': [],
        'sharpe_ratio': [],
        'avg_return': []
    }

    for i in range(n_bootstrap):
        if (i + 1) % 100 == 0:
            print(f"   Progress: {i+1}/{n_bootstrap} ({(i+1)/n_bootstrap*100:.1f}%)")

        # Resample trades with replacement
        bootstrap_sample = trades_df.sample(n=n_trades, replace=True, random_state=random_seed + i)

        # Simulate both thresholds on this sample
        try:
            baseline_boot = simulate_threshold_on_dataset(
                bootstrap_sample.copy(),
                base_data_dir,
                baseline_threshold,
                desc=None  # Suppress progress for bootstrap
            )
            optimal_boot = simulate_threshold_on_dataset(
                bootstrap_sample.copy(),
                base_data_dir,
                optimal_threshold,
                desc=None
            )

            # Calculate metrics
            baseline_boot_metrics = calculate_performance_metrics(baseline_boot)
            optimal_boot_metrics = calculate_performance_metrics(optimal_boot)

            # Store differences
            bootstrap_diffs['win_rate'].append(
                optimal_boot_metrics['win_rate'] - baseline_boot_metrics['win_rate']
            )
            bootstrap_diffs['profit_factor'].append(
                optimal_boot_metrics['profit_factor'] - baseline_boot_metrics['profit_factor']
            )
            bootstrap_diffs['sharpe_ratio'].append(
                optimal_boot_metrics['sharpe_ratio'] - baseline_boot_metrics['sharpe_ratio']
            )
            bootstrap_diffs['avg_return'].append(
                optimal_boot_metrics['avg_return_pct'] - baseline_boot_metrics['avg_return_pct']
            )

        except Exception as e:
            # If simulation fails, record NaN and continue
            for metric in bootstrap_diffs:
                bootstrap_diffs[metric].append(np.nan)
            continue

    print(f"   ✓ Bootstrap complete")

    # Calculate p-values and confidence intervals
    p_values = {}
    confidence_intervals = {}

    for metric in bootstrap_diffs:
        diffs = np.array(bootstrap_diffs[metric])
        # Remove NaN values from failed simulations
        diffs = diffs[~np.isnan(diffs)]

        # One-tailed p-value: probability that baseline is better (diff <= 0)
        p_values[metric] = (diffs <= 0).sum() / len(diffs)

        # 95% confidence interval
        ci_lower = np.percentile(diffs, 2.5)
        ci_upper = np.percentile(diffs, 97.5)
        confidence_intervals[metric] = (ci_lower, ci_upper)

    # Determine significance (p < 0.05)
    is_significant = {
        metric: p_values[metric] < 0.05
        for metric in p_values
    }

    # Compile results
    results = {
        'baseline_threshold': baseline_threshold,
        'optimal_threshold': optimal_threshold,
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


def run_statistical_validation(
    baseline_data_path: str,
    base_data_dir: str,
    baseline_threshold: float,
    optimal_threshold: float,
    n_bootstrap: int = 1000,
    alpha: float = 0.05
) -> Tuple[Dict, Dict]:
    """
    Run complete statistical validation for Buy and Sell trades separately.

    Args:
        baseline_data_path: Path to Stage 1 baseline data CSV
        base_data_dir: Directory containing base data files
        baseline_threshold: Current threshold (e.g., 0.80)
        optimal_threshold: Proposed optimal threshold (e.g., 0.95)
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level (default 0.05 for 95% confidence)

    Returns:
        Tuple of (buy_results, sell_results) dictionaries
    """

    print("="*70)
    print("STAGE 4: STATISTICAL VALIDATION")
    print("="*70)

    # Load baseline data
    print(f"\n📂 Loading baseline data from Stage 1...")
    baseline_df = pd.read_csv(baseline_data_path)
    print(f"   ✓ Loaded {len(baseline_df)} trades")

    # Ensure datetime columns
    for col in ['Entry Time', 'Exit Time']:
        if col in baseline_df.columns:
            baseline_df[col] = pd.to_datetime(baseline_df[col]).dt.tz_localize(None)

    # Separate Buy and Sell trades
    buy_trades = baseline_df[baseline_df['Trade Type'] == 'Buy'].copy()
    sell_trades = baseline_df[baseline_df['Trade Type'] == 'Sell'].copy()

    print(f"   Buy trades: {len(buy_trades)}")
    print(f"   Sell trades: {len(sell_trades)}")

    # Run bootstrap for Buy trades
    print("\n" + "="*70)
    print("BUY TRADES STATISTICAL VALIDATION")
    print("="*70)

    buy_results = bootstrap_comparison(
        buy_trades,
        base_data_dir,
        baseline_threshold,
        optimal_threshold,
        n_bootstrap=n_bootstrap
    )

    # Run bootstrap for Sell trades
    print("\n" + "="*70)
    print("SELL TRADES STATISTICAL VALIDATION")
    print("="*70)

    sell_results = bootstrap_comparison(
        sell_trades,
        base_data_dir,
        baseline_threshold,
        optimal_threshold,
        n_bootstrap=n_bootstrap
    )

    # Print summary
    print("\n" + "="*70)
    print("STATISTICAL VALIDATION SUMMARY")
    print("="*70)

    print(f"\n🎯 BUY TRADES")
    print(f"   Baseline ({baseline_threshold*100:.0f}%): WR {buy_results['baseline_metrics']['win_rate']:.2f}% | "
          f"PF {buy_results['baseline_metrics']['profit_factor']:.2f}")
    print(f"   Optimal ({optimal_threshold*100:.0f}%): WR {buy_results['optimal_metrics']['win_rate']:.2f}% | "
          f"PF {buy_results['optimal_metrics']['profit_factor']:.2f}")
    print(f"\n   Improvements:")
    print(f"   - Win Rate: {buy_results['observed_improvement']['win_rate']:+.2f}% "
          f"(p={buy_results['p_values']['win_rate']:.4f}) {'✅' if buy_results['is_significant']['win_rate'] else '❌'}")
    print(f"   - Profit Factor: {buy_results['observed_improvement']['profit_factor']:+.2f} "
          f"(p={buy_results['p_values']['profit_factor']:.4f}) {'✅' if buy_results['is_significant']['profit_factor'] else '❌'}")
    print(f"   - Sharpe Ratio: {buy_results['observed_improvement']['sharpe_ratio']:+.2f} "
          f"(p={buy_results['p_values']['sharpe_ratio']:.4f}) {'✅' if buy_results['is_significant']['sharpe_ratio'] else '❌'}")

    print(f"\n🎯 SELL TRADES")
    print(f"   Baseline ({baseline_threshold*100:.0f}%): WR {sell_results['baseline_metrics']['win_rate']:.2f}% | "
          f"PF {sell_results['baseline_metrics']['profit_factor']:.2f}")
    print(f"   Optimal ({optimal_threshold*100:.0f}%): WR {sell_results['optimal_metrics']['win_rate']:.2f}% | "
          f"PF {sell_results['optimal_metrics']['profit_factor']:.2f}")
    print(f"\n   Improvements:")
    print(f"   - Win Rate: {sell_results['observed_improvement']['win_rate']:+.2f}% "
          f"(p={sell_results['p_values']['win_rate']:.4f}) {'✅' if sell_results['is_significant']['win_rate'] else '❌'}")
    print(f"   - Profit Factor: {sell_results['observed_improvement']['profit_factor']:+.2f} "
          f"(p={sell_results['p_values']['profit_factor']:.4f}) {'✅' if sell_results['is_significant']['profit_factor'] else '❌'}")
    print(f"   - Sharpe Ratio: {sell_results['observed_improvement']['sharpe_ratio']:+.2f} "
          f"(p={sell_results['p_values']['sharpe_ratio']:.4f}) {'✅' if sell_results['is_significant']['sharpe_ratio'] else '❌'}")

    # Decision gate
    print("\n" + "="*70)
    print("DECISION GATE")
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
        print("\n✅ PASS - Improvements are statistically significant")
        print(f"   Both Buy and Sell show p<{alpha} for Win Rate and Profit Factor")
        print(f"\n   ➡️ PROCEED to Stage 6 (Final Out-of-Sample Test)")
    elif buy_pass or sell_pass:
        print(f"\n⚠️ PARTIAL PASS - Mixed results")
        if buy_pass:
            print(f"   ✅ Buy trades: Significant improvement")
        else:
            print(f"   ❌ Buy trades: Not significant")
        if sell_pass:
            print(f"   ✅ Sell trades: Significant improvement")
        else:
            print(f"   ❌ Sell trades: Not significant")
        print(f"\n   Recommendation: Proceed with caution to Stage 6")
    else:
        print(f"\n❌ FAIL - Improvements not statistically significant")
        print(f"   p-values exceed {alpha} threshold")
        print(f"\n   ❌ STOP - Return to Stage 2 or accept baseline")

    return buy_results, sell_results


def format_statistical_report(
    buy_results: Dict,
    sell_results: Dict,
    output_path: str
) -> None:
    """
    Generate markdown report for statistical validation results.

    Args:
        buy_results: Bootstrap results for Buy trades
        sell_results: Bootstrap results for Sell trades
        output_path: Path to save report markdown file
    """

    from datetime import datetime

    report = f"""# Stage 4: Statistical Validation Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Method**: Bootstrap hypothesis testing with {buy_results['n_bootstrap']} resamples
**Purpose**: Verify that {buy_results['optimal_threshold']*100:.0f}% threshold improvement over {buy_results['baseline_threshold']*100:.0f}% is statistically significant

---

## 📋 METHODOLOGY

**Bootstrap Testing** verifies if observed improvements are:
- **Statistically significant**: Not due to random chance (p < 0.05)
- **Robust**: Improvement holds across resampled datasets
- **Reliable**: 95% confidence intervals exclude zero

**Approach:**
1. Simulate both thresholds on full validation period
2. Resample trades {buy_results['n_bootstrap']} times with replacement
3. Calculate improvement (optimal - baseline) for each sample
4. Compute p-value: probability that baseline is better
5. Success if p < 0.05 for Win Rate AND Profit Factor

**Null Hypothesis (H0)**: {buy_results['optimal_threshold']*100:.0f}% threshold is NOT better than {buy_results['baseline_threshold']*100:.0f}%
**Alternative (H1)**: {buy_results['optimal_threshold']*100:.0f}% threshold IS significantly better

---

## 🎯 BUY TRADES VALIDATION

### Performance Comparison
| Metric | Baseline ({buy_results['baseline_threshold']*100:.0f}%) | Optimal ({buy_results['optimal_threshold']*100:.0f}%) | Improvement | p-value | Significant |
|--------|----------|---------|-------------|---------|-------------|
| **Win Rate** | {buy_results['baseline_metrics']['win_rate']:.2f}% | {buy_results['optimal_metrics']['win_rate']:.2f}% | {buy_results['observed_improvement']['win_rate']:+.2f}% | {buy_results['p_values']['win_rate']:.4f} | {'✅' if buy_results['is_significant']['win_rate'] else '❌'} |
| **Profit Factor** | {buy_results['baseline_metrics']['profit_factor']:.2f} | {buy_results['optimal_metrics']['profit_factor']:.2f} | {buy_results['observed_improvement']['profit_factor']:+.2f} | {buy_results['p_values']['profit_factor']:.4f} | {'✅' if buy_results['is_significant']['profit_factor'] else '❌'} |
| **Sharpe Ratio** | {buy_results['baseline_metrics']['sharpe_ratio']:.2f} | {buy_results['optimal_metrics']['sharpe_ratio']:.2f} | {buy_results['observed_improvement']['sharpe_ratio']:+.2f} | {buy_results['p_values']['sharpe_ratio']:.4f} | {'✅' if buy_results['is_significant']['sharpe_ratio'] else '❌'} |

### 95% Confidence Intervals
| Metric | Lower Bound | Upper Bound | Excludes Zero |
|--------|-------------|-------------|---------------|
| **Win Rate** | {buy_results['confidence_intervals']['win_rate'][0]:.2f}% | {buy_results['confidence_intervals']['win_rate'][1]:.2f}% | {'✅' if buy_results['confidence_intervals']['win_rate'][0] > 0 else '❌'} |
| **Profit Factor** | {buy_results['confidence_intervals']['profit_factor'][0]:.2f} | {buy_results['confidence_intervals']['profit_factor'][1]:.2f} | {'✅' if buy_results['confidence_intervals']['profit_factor'][0] > 0 else '❌'} |
| **Sharpe Ratio** | {buy_results['confidence_intervals']['sharpe_ratio'][0]:.2f} | {buy_results['confidence_intervals']['sharpe_ratio'][1]:.2f} | {'✅' if buy_results['confidence_intervals']['sharpe_ratio'][0] > 0 else '❌'} |

**Interpretation**:
- p < 0.001: Very strong evidence
- p < 0.01: Strong evidence
- p < 0.05: Significant evidence
- p ≥ 0.05: Insufficient evidence

---

## 🎯 SELL TRADES VALIDATION

### Performance Comparison
| Metric | Baseline ({sell_results['baseline_threshold']*100:.0f}%) | Optimal ({sell_results['optimal_threshold']*100:.0f}%) | Improvement | p-value | Significant |
|--------|----------|---------|-------------|---------|-------------|
| **Win Rate** | {sell_results['baseline_metrics']['win_rate']:.2f}% | {sell_results['optimal_metrics']['win_rate']:.2f}% | {sell_results['observed_improvement']['win_rate']:+.2f}% | {sell_results['p_values']['win_rate']:.4f} | {'✅' if sell_results['is_significant']['win_rate'] else '❌'} |
| **Profit Factor** | {sell_results['baseline_metrics']['profit_factor']:.2f} | {sell_results['optimal_metrics']['profit_factor']:.2f} | {sell_results['observed_improvement']['profit_factor']:+.2f} | {sell_results['p_values']['profit_factor']:.4f} | {'✅' if sell_results['is_significant']['profit_factor'] else '❌'} |
| **Sharpe Ratio** | {sell_results['baseline_metrics']['sharpe_ratio']:.2f} | {sell_results['optimal_metrics']['sharpe_ratio']:.2f} | {sell_results['observed_improvement']['sharpe_ratio']:+.2f} | {sell_results['p_values']['sharpe_ratio']:.4f} | {'✅' if sell_results['is_significant']['sharpe_ratio'] else '❌'} |

### 95% Confidence Intervals
| Metric | Lower Bound | Upper Bound | Excludes Zero |
|--------|-------------|-------------|---------------|
| **Win Rate** | {sell_results['confidence_intervals']['win_rate'][0]:.2f}% | {sell_results['confidence_intervals']['win_rate'][1]:.2f}% | {'✅' if sell_results['confidence_intervals']['win_rate'][0] > 0 else '❌'} |
| **Profit Factor** | {sell_results['confidence_intervals']['profit_factor'][0]:.2f} | {sell_results['confidence_intervals']['profit_factor'][1]:.2f} | {'✅' if sell_results['confidence_intervals']['profit_factor'][0] > 0 else '❌'} |
| **Sharpe Ratio** | {sell_results['confidence_intervals']['sharpe_ratio'][0]:.2f} | {sell_results['confidence_intervals']['sharpe_ratio'][1]:.2f} | {'✅' if sell_results['confidence_intervals']['sharpe_ratio'][0] > 0 else '❌'} |

---

## 🚦 DECISION GATE

"""

    buy_pass = (
        buy_results['is_significant']['win_rate'] and
        buy_results['is_significant']['profit_factor']
    )
    sell_pass = (
        sell_results['is_significant']['win_rate'] and
        sell_results['is_significant']['profit_factor']
    )

    if buy_pass and sell_pass:
        report += f"""**Buy Trades**: ✅ PASS
- Win Rate improvement is significant (p={buy_results['p_values']['win_rate']:.4f} < 0.05)
- Profit Factor improvement is significant (p={buy_results['p_values']['profit_factor']:.4f} < 0.05)

**Sell Trades**: ✅ PASS
- Win Rate improvement is significant (p={sell_results['p_values']['win_rate']:.4f} < 0.05)
- Profit Factor improvement is significant (p={sell_results['p_values']['profit_factor']:.4f} < 0.05)


### ✅ RECOMMENDATION: PROCEED TO STAGE 6

**The {buy_results['optimal_threshold']*100:.0f}% threshold shows statistically significant improvements over {buy_results['baseline_threshold']*100:.0f}%.**

**Next Steps:**
1. **Skip Stage 5** (Entry Filter Optimization) - Exit optimization successful
2. **Proceed to Stage 6** (Final Out-of-Sample Test) - Test on unseen 2024 H2 data
3. **Require both conditions at Stage 6:**
   - {buy_results['optimal_threshold']*100:.0f}% outperforms {buy_results['baseline_threshold']*100:.0f}% on test data
   - Meet all success criteria (WR ≥52%, PF ≥1.25, Sharpe ≥1.5)
"""
    elif buy_pass or sell_pass:
        report += f"""**Buy Trades**: {'✅ PASS' if buy_pass else '❌ FAIL'}
{'- Win Rate and Profit Factor improvements are significant' if buy_pass else '- Improvements not statistically significant'}

**Sell Trades**: {'✅ PASS' if sell_pass else '❌ FAIL'}
{'- Win Rate and Profit Factor improvements are significant' if sell_pass else '- Improvements not statistically significant'}


### ⚠️ RECOMMENDATION: PROCEED WITH CAUTION

**Mixed results - some improvements significant, others not.**

**Options:**
1. **Proceed to Stage 6** with separate thresholds for Buy/Sell if one passed
2. **Re-run Stage 2** with different threshold range
3. **Accept baseline** and move to Stage 5 (Entry Filter Optimization)
"""
    else:
        report += f"""**Buy Trades**: ❌ FAIL
- Win Rate: p={buy_results['p_values']['win_rate']:.4f} ≥ 0.05
- Profit Factor: p={buy_results['p_values']['profit_factor']:.4f} ≥ 0.05

**Sell Trades**: ❌ FAIL
- Win Rate: p={sell_results['p_values']['win_rate']:.4f} ≥ 0.05
- Profit Factor: p={sell_results['p_values']['profit_factor']:.4f} ≥ 0.05


### ❌ RECOMMENDATION: STOP - NOT STATISTICALLY SIGNIFICANT

**The observed improvements are likely due to random chance.**

**Options:**
1. **Accept baseline threshold** ({buy_results['baseline_threshold']*100:.0f}%) and proceed to Stage 5
2. **Re-run Stage 2** with different threshold range (e.g., 85%-97% in 1% steps)
3. **Investigate regime-specific thresholds** (different thresholds for different market conditions)
"""

    report += f"""
---

## 📁 OUTPUTS

**Checkpoint**: `checkpoints/stage4_statistical_results.csv`
**Bootstrap Distributions**: `checkpoints/stage4_bootstrap_distributions.csv`

---

*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    # Write report
    with open(output_path, 'w') as f:
        f.write(report)

    print(f"\n✅ Report saved: {output_path}")
