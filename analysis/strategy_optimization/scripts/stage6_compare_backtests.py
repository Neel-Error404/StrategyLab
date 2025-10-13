#!/usr/bin/env python3
"""
Stage 6: Compare Baseline (0.80) vs Hypothesis (0.95) Backtests
================================================================

Compares two complete backtests on the test period (2024-07-01 to 2025-08-31)
to make final go/no-go decision on 0.95 threshold optimization.

Usage:
    python stage6_compare_backtests.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Paths
BASE_DIR = Path("/mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/StrategyLab-master")
BASELINE_DIR = BASE_DIR / "outputs/20251005_121223/mse/2022-01-01_to_2025-08-31/data/strategy_trades"
HYPOTHESIS_DIR = BASE_DIR / "outputs/20251005_193937/mse/2022-01-01_to_2025-08-31/data/strategy_trades"
OUTPUT_DIR = BASE_DIR / "analysis/strategy_optimization/checkpoints"

# Test period
TEST_START = pd.Timestamp("2024-07-01")
TEST_END = pd.Timestamp("2025-08-31")

# 14 tickers available in both baseline and hypothesis
TICKERS = [
    "RELIANCE", "TCS", "INFY", "HINDUNILVR", "ITC", "SBIN", "KOTAKBANK", "LT",
    "ASIANPAINT", "AXISBANK", "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO"
]

def load_all_trades(trades_dir, label):
    """Load and merge all strategy trades from directory"""
    all_trades = []

    for ticker in TICKERS:
        trade_file = trades_dir / f"{ticker}_StrategyTrades_2022-01-01_to_2025-08-31.csv"

        if not trade_file.exists():
            print(f"⚠️  Missing {label} trades for {ticker}")
            continue

        try:
            df = pd.read_csv(trade_file)
            df['ticker'] = ticker
            all_trades.append(df)
            print(f"✅ Loaded {label} - {ticker}: {len(df)} trades")
        except Exception as e:
            print(f"❌ Error loading {ticker}: {e}")

    if not all_trades:
        raise ValueError(f"No trades loaded for {label}")

    merged = pd.concat(all_trades, ignore_index=True)
    print(f"\n📊 Total {label} trades: {len(merged):,}")
    return merged

def filter_to_test_period(df):
    """Filter trades to test period (2024-07-01 to 2025-08-31)"""
    # Parse entry time and remove timezone info for comparison
    df['Entry Time'] = pd.to_datetime(df['Entry Time'])

    # Remove timezone for comparison (convert to naive datetime)
    if df['Entry Time'].dt.tz is not None:
        df['Entry Time'] = df['Entry Time'].dt.tz_localize(None)

    # Filter to test period
    test_trades = df[
        (df['Entry Time'] >= TEST_START) &
        (df['Entry Time'] <= TEST_END)
    ].copy()

    print(f"📅 Test period trades: {len(test_trades):,} (from {len(df):,} total)")
    return test_trades

def calculate_metrics(df, label):
    """Calculate comprehensive performance metrics"""

    # Split by trade type
    buy_trades = df[df['Trade Type'] == 'Buy'].copy()
    sell_trades = df[df['Trade Type'] == 'Sell'].copy()

    def calc_trade_metrics(trades, trade_type):
        if len(trades) == 0:
            return {}

        wins = trades[trades['Profit (%)'] > 0]
        losses = trades[trades['Profit (%)'] < 0]

        total_profit = wins['Profit (%)'].sum() if len(wins) > 0 else 0
        total_loss = abs(losses['Profit (%)'].sum()) if len(losses) > 0 else 0

        return {
            'trade_type': trade_type,
            'total_trades': len(trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': (len(wins) / len(trades) * 100) if len(trades) > 0 else 0,
            'profit_factor': (total_profit / total_loss) if total_loss > 0 else float('inf'),
            'avg_profit': trades['Profit (%)'].mean(),
            'avg_win': wins['Profit (%)'].mean() if len(wins) > 0 else 0,
            'avg_loss': losses['Profit (%)'].mean() if len(losses) > 0 else 0,
            'max_profit': trades['Profit (%)'].max(),
            'min_loss': trades['Profit (%)'].min(),
            'total_return': trades['Profit (%)'].sum(),
            'avg_duration': trades['Trade Duration (min)'].mean() if 'Trade Duration (min)' in trades.columns else 0
        }

    # Calculate for buy and sell separately
    buy_metrics = calc_trade_metrics(buy_trades, 'Buy')
    sell_metrics = calc_trade_metrics(sell_trades, 'Sell')

    # Combined metrics
    combined_metrics = calc_trade_metrics(df, 'Combined')

    # Calculate Sharpe Ratio (daily returns)
    if len(df) > 0:
        df_sorted = df.sort_values('Entry Time')
        df_sorted['date'] = df_sorted['Entry Time'].dt.date
        daily_returns = df_sorted.groupby('date')['Profit (%)'].sum()

        if len(daily_returns) > 1:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
        else:
            sharpe = 0
    else:
        sharpe = 0

    combined_metrics['sharpe_ratio'] = sharpe

    return {
        'label': label,
        'buy': buy_metrics,
        'sell': sell_metrics,
        'combined': combined_metrics
    }

def compare_metrics(baseline_metrics, hypothesis_metrics):
    """Compare and calculate improvements"""

    def calc_improvement(baseline, hypothesis, metric):
        """Calculate improvement percentage or absolute difference"""
        base_val = baseline.get(metric, 0)
        hyp_val = hypothesis.get(metric, 0)

        if base_val == 0:
            return hyp_val

        # For percentages and ratios, calculate absolute difference
        if metric in ['win_rate', 'avg_profit', 'avg_win', 'avg_loss', 'sharpe_ratio']:
            return hyp_val - base_val

        # For profit factor, calculate percentage improvement
        if metric == 'profit_factor':
            if base_val == float('inf') or hyp_val == float('inf'):
                return 0
            return ((hyp_val - base_val) / base_val) * 100

        return hyp_val - base_val

    comparison = {}

    for trade_type in ['buy', 'sell', 'combined']:
        base = baseline_metrics.get(trade_type, {})
        hyp = hypothesis_metrics.get(trade_type, {})

        comparison[trade_type] = {
            'baseline': base,
            'hypothesis': hyp,
            'improvements': {
                'win_rate': calc_improvement(base, hyp, 'win_rate'),
                'profit_factor': calc_improvement(base, hyp, 'profit_factor'),
                'sharpe_ratio': calc_improvement(base, hyp, 'sharpe_ratio'),
                'avg_profit': calc_improvement(base, hyp, 'avg_profit'),
                'total_return': calc_improvement(base, hyp, 'total_return')
            }
        }

    return comparison

def make_decision(comparison):
    """Make final go/no-go decision based on success criteria"""

    combined = comparison['combined']
    hyp = combined['hypothesis']
    improvements = combined['improvements']

    # Success criteria (from Phase 2 plan)
    criteria = {
        'win_rate_target': 52.0,  # ≥52%
        'profit_factor_target': 1.25,  # ≥1.25
        'sharpe_ratio_target': 1.5,  # ≥1.5
        'win_rate_beats_baseline': improvements['win_rate'] > 0,
        'profit_factor_beats_baseline': improvements['profit_factor'] > 0,
        'sharpe_beats_baseline': improvements['sharpe_ratio'] > 0
    }

    # Check if hypothesis meets all criteria
    checks = {
        'win_rate_meets_target': hyp.get('win_rate', 0) >= criteria['win_rate_target'],
        'profit_factor_meets_target': hyp.get('profit_factor', 0) >= criteria['profit_factor_target'],
        'sharpe_meets_target': hyp.get('sharpe_ratio', 0) >= criteria['sharpe_ratio_target'],
        'win_rate_beats_baseline': criteria['win_rate_beats_baseline'],
        'profit_factor_beats_baseline': criteria['profit_factor_beats_baseline'],
        'sharpe_beats_baseline': criteria['sharpe_beats_baseline']
    }

    # Final decision
    all_passed = all(checks.values())

    decision = {
        'recommendation': '✅ IMPLEMENT 0.95 THRESHOLD' if all_passed else '❌ KEEP BASELINE (0.80)',
        'criteria_checks': checks,
        'all_criteria_passed': all_passed,
        'summary': {
            'win_rate': f"{hyp.get('win_rate', 0):.2f}% (Target: ≥52%, Baseline: {combined['baseline'].get('win_rate', 0):.2f}%)",
            'profit_factor': f"{hyp.get('profit_factor', 0):.2f} (Target: ≥1.25, Baseline: {combined['baseline'].get('profit_factor', 0):.2f})",
            'sharpe_ratio': f"{hyp.get('sharpe_ratio', 0):.2f} (Target: ≥1.5, Baseline: {combined['baseline'].get('sharpe_ratio', 0):.2f})",
            'improvement_win_rate': f"{improvements['win_rate']:+.2f}%",
            'improvement_profit_factor': f"{improvements['profit_factor']:+.2f}%",
            'improvement_sharpe': f"{improvements['sharpe_ratio']:+.2f}"
        }
    }

    return decision

def main():
    print("=" * 80)
    print("STAGE 6: FINAL COMPARISON - BASELINE (0.80) vs HYPOTHESIS (0.95)")
    print("=" * 80)
    print(f"\nTest Period: {TEST_START.date()} to {TEST_END.date()}")
    print(f"Tickers: {len(TICKERS)} stocks\n")

    # Load trades
    print("\n📥 Loading Baseline (0.80) Trades...")
    baseline_all = load_all_trades(BASELINE_DIR, "Baseline")

    print("\n📥 Loading Hypothesis (0.95) Trades...")
    hypothesis_all = load_all_trades(HYPOTHESIS_DIR, "Hypothesis")

    # Filter to test period
    print("\n" + "=" * 80)
    print("FILTERING TO TEST PERIOD (OUT-OF-SAMPLE)")
    print("=" * 80)

    print("\n🔍 Baseline (0.80):")
    baseline_test = filter_to_test_period(baseline_all)

    print("\n🔍 Hypothesis (0.95):")
    hypothesis_test = filter_to_test_period(hypothesis_all)

    # Calculate metrics
    print("\n" + "=" * 80)
    print("CALCULATING METRICS")
    print("=" * 80)

    print("\n📊 Baseline (0.80) Metrics:")
    baseline_metrics = calculate_metrics(baseline_test, "Baseline (0.80)")

    print("\n📊 Hypothesis (0.95) Metrics:")
    hypothesis_metrics = calculate_metrics(hypothesis_test, "Hypothesis (0.95)")

    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON & DECISION")
    print("=" * 80)

    comparison = compare_metrics(baseline_metrics, hypothesis_metrics)
    decision = make_decision(comparison)

    # Print results
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    print(f"\n{decision['recommendation']}\n")

    print("Success Criteria Checks:")
    for criterion, passed in decision['criteria_checks'].items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {criterion}: {status}")

    print("\nPerformance Summary:")
    for key, value in decision['summary'].items():
        print(f"  {key}: {value}")

    # Save results
    output_file = OUTPUT_DIR / "stage6_final_comparison.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        'timestamp': datetime.now().isoformat(),
        'test_period': {'start': str(TEST_START.date()), 'end': str(TEST_END.date())},
        'baseline_metrics': baseline_metrics,
        'hypothesis_metrics': hypothesis_metrics,
        'comparison': comparison,
        'decision': decision
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")

    # Save summary CSV
    summary_csv = OUTPUT_DIR / "stage6_summary_comparison.csv"

    summary_data = []
    for trade_type in ['buy', 'sell', 'combined']:
        base = comparison[trade_type]['baseline']
        hyp = comparison[trade_type]['hypothesis']
        imp = comparison[trade_type]['improvements']

        summary_data.append({
            'Trade_Type': trade_type.capitalize(),
            'Baseline_WinRate': base.get('win_rate', 0),
            'Hypothesis_WinRate': hyp.get('win_rate', 0),
            'WinRate_Improvement': imp['win_rate'],
            'Baseline_ProfitFactor': base.get('profit_factor', 0),
            'Hypothesis_ProfitFactor': hyp.get('profit_factor', 0),
            'PF_Improvement_Pct': imp['profit_factor'],
            'Baseline_Sharpe': base.get('sharpe_ratio', 0),
            'Hypothesis_Sharpe': hyp.get('sharpe_ratio', 0),
            'Sharpe_Improvement': imp['sharpe_ratio'],
            'Baseline_TotalReturn': base.get('total_return', 0),
            'Hypothesis_TotalReturn': hyp.get('total_return', 0)
        })

    pd.DataFrame(summary_data).to_csv(summary_csv, index=False)
    print(f"💾 Summary CSV saved to: {summary_csv}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return decision['all_criteria_passed']

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
