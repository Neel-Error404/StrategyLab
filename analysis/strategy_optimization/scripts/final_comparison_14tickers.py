#!/usr/bin/env python3
"""
Stage 6 Final Comparison: Baseline (0.80) vs Hypothesis (0.95)
14 Common Tickers - Test Period (2024-07-01 to 2025-08-31)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Paths
BASE_DIR = Path("/mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/StrategyLab-master")
BASELINE_DIR = BASE_DIR / "outputs/20251005_121223/mse/2022-01-01_to_2025-08-31/data/strategy_trades"
HYPOTHESIS_DIR = BASE_DIR / "outputs/20251005_193937/mse/2022-01-01_to_2025-08-31/data/strategy_trades"
OUTPUT_DIR = BASE_DIR / "analysis/strategy_optimization/checkpoints"

# Test period (out-of-sample)
TEST_START = "2024-07-01"
TEST_END = "2025-08-31"

# 14 common tickers
TICKERS = [
    "ASIANPAINT", "AXISBANK", "HINDUNILVR", "INFY", "ITC", "KOTAKBANK", "LT",
    "MARUTI", "RELIANCE", "SBIN", "SUNPHARMA", "TCS", "TITAN", "ULTRACEMCO"
]

def load_trades(ticker, trades_dir):
    """Load trades for a single ticker"""
    file_path = trades_dir / f"{ticker}_StrategyTrades_2022-01-01_to_2025-08-31.csv"
    if not file_path.exists():
        return None

    df = pd.read_csv(file_path)
    df['ticker'] = ticker
    return df

def filter_test_period(df):
    """Filter to test period"""
    df['Entry Time'] = pd.to_datetime(df['Entry Time'])

    # Remove timezone if present
    if hasattr(df['Entry Time'].dtype, 'tz') and df['Entry Time'].dt.tz is not None:
        df['Entry Time'] = df['Entry Time'].dt.tz_localize(None)

    test_df = df[
        (df['Entry Time'] >= TEST_START) &
        (df['Entry Time'] <= TEST_END)
    ].copy()

    return test_df

def calculate_metrics(df):
    """Calculate performance metrics"""
    if len(df) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'avg_profit': 0,
            'total_return': 0
        }

    # Win rate
    wins = df[df['Profit (%)'] > 0]
    losses = df[df['Profit (%)'] < 0]
    win_rate = (len(wins) / len(df)) * 100

    # Profit factor
    total_profit = wins['Profit (%)'].sum() if len(wins) > 0 else 0
    total_loss = abs(losses['Profit (%)'].sum()) if len(losses) > 0 else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else 0

    # Sharpe ratio (annualized daily returns)
    df_sorted = df.sort_values('Entry Time').copy()
    df_sorted['date'] = df_sorted['Entry Time'].dt.date
    daily_returns = df_sorted.groupby('date')['Profit (%)'].sum()

    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0

    return {
        'total_trades': len(df),
        'winning_trades': len(wins),
        'losing_trades': len(losses),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe_ratio,
        'avg_profit': df['Profit (%)'].mean(),
        'avg_win': wins['Profit (%)'].mean() if len(wins) > 0 else 0,
        'avg_loss': losses['Profit (%)'].mean() if len(losses) > 0 else 0,
        'total_return': df['Profit (%)'].sum()
    }

def print_section(title):
    """Print section header"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def main():
    print_section("STAGE 6: FINAL COMPARISON - 14 COMMON TICKERS")
    print(f"\nTest Period: {TEST_START} to {TEST_END}")
    print(f"Tickers: {len(TICKERS)} stocks")
    print(f"Tickers: {', '.join(TICKERS)}")

    # Load all trades
    print_section("LOADING DATA")

    baseline_trades = []
    hypothesis_trades = []

    for ticker in TICKERS:
        # Baseline
        base_df = load_trades(ticker, BASELINE_DIR)
        if base_df is not None:
            baseline_trades.append(base_df)
            print(f"✅ Baseline {ticker}: {len(base_df)} trades")
        else:
            print(f"❌ Missing Baseline {ticker}")

        # Hypothesis
        hyp_df = load_trades(ticker, HYPOTHESIS_DIR)
        if hyp_df is not None:
            hypothesis_trades.append(hyp_df)
            print(f"✅ Hypothesis {ticker}: {len(hyp_df)} trades")
        else:
            print(f"❌ Missing Hypothesis {ticker}")

    # Merge all
    baseline_all = pd.concat(baseline_trades, ignore_index=True)
    hypothesis_all = pd.concat(hypothesis_trades, ignore_index=True)

    print(f"\n📊 Total Baseline trades: {len(baseline_all):,}")
    print(f"📊 Total Hypothesis trades: {len(hypothesis_all):,}")

    # Filter to test period
    print_section("FILTERING TO TEST PERIOD")

    baseline_test = filter_test_period(baseline_all)
    hypothesis_test = filter_test_period(hypothesis_all)

    print(f"\n📅 Baseline test trades: {len(baseline_test):,}")
    print(f"📅 Hypothesis test trades: {len(hypothesis_test):,}")

    # Calculate metrics
    print_section("PERFORMANCE METRICS")

    baseline_metrics = calculate_metrics(baseline_test)
    hypothesis_metrics = calculate_metrics(hypothesis_test)

    print("\n🔵 BASELINE (0.80 Threshold):")
    print(f"  Total Trades: {baseline_metrics['total_trades']:,}")
    print(f"  Win Rate: {baseline_metrics['win_rate']:.2f}%")
    print(f"  Profit Factor: {baseline_metrics['profit_factor']:.2f}")
    print(f"  Sharpe Ratio: {baseline_metrics['sharpe_ratio']:.2f}")
    print(f"  Avg Profit: {baseline_metrics['avg_profit']:.3f}%")
    print(f"  Total Return: {baseline_metrics['total_return']:.2f}%")

    print("\n🟢 HYPOTHESIS (0.95 Threshold):")
    print(f"  Total Trades: {hypothesis_metrics['total_trades']:,}")
    print(f"  Win Rate: {hypothesis_metrics['win_rate']:.2f}%")
    print(f"  Profit Factor: {hypothesis_metrics['profit_factor']:.2f}")
    print(f"  Sharpe Ratio: {hypothesis_metrics['sharpe_ratio']:.2f}")
    print(f"  Avg Profit: {hypothesis_metrics['avg_profit']:.3f}%")
    print(f"  Total Return: {hypothesis_metrics['total_return']:.2f}%")

    # Calculate improvements
    print_section("IMPROVEMENTS")

    wr_improvement = hypothesis_metrics['win_rate'] - baseline_metrics['win_rate']
    pf_improvement = hypothesis_metrics['profit_factor'] - baseline_metrics['profit_factor']
    sharpe_improvement = hypothesis_metrics['sharpe_ratio'] - baseline_metrics['sharpe_ratio']

    print(f"\n📈 Win Rate: {wr_improvement:+.2f}% ({baseline_metrics['win_rate']:.2f}% → {hypothesis_metrics['win_rate']:.2f}%)")
    print(f"📈 Profit Factor: {pf_improvement:+.2f} ({baseline_metrics['profit_factor']:.2f} → {hypothesis_metrics['profit_factor']:.2f})")
    print(f"📈 Sharpe Ratio: {sharpe_improvement:+.2f} ({baseline_metrics['sharpe_ratio']:.2f} → {hypothesis_metrics['sharpe_ratio']:.2f})")

    # Decision criteria
    print_section("SUCCESS CRITERIA CHECK")

    criteria = {
        'win_rate_meets_52': hypothesis_metrics['win_rate'] >= 52.0,
        'profit_factor_meets_1.25': hypothesis_metrics['profit_factor'] >= 1.25,
        'sharpe_meets_1.5': hypothesis_metrics['sharpe_ratio'] >= 1.5,
        'win_rate_beats_baseline': wr_improvement > 0,
        'profit_factor_beats_baseline': pf_improvement > 0,
        'sharpe_beats_baseline': sharpe_improvement > 0
    }

    print("\n✅ = PASS | ❌ = FAIL\n")

    status_wr_target = "✅" if criteria['win_rate_meets_52'] else "❌"
    print(f"{status_wr_target} Win Rate ≥ 52%: {hypothesis_metrics['win_rate']:.2f}%")

    status_pf_target = "✅" if criteria['profit_factor_meets_1.25'] else "❌"
    print(f"{status_pf_target} Profit Factor ≥ 1.25: {hypothesis_metrics['profit_factor']:.2f}")

    status_sharpe_target = "✅" if criteria['sharpe_meets_1.5'] else "❌"
    print(f"{status_sharpe_target} Sharpe Ratio ≥ 1.5: {hypothesis_metrics['sharpe_ratio']:.2f}")

    status_wr_beat = "✅" if criteria['win_rate_beats_baseline'] else "❌"
    print(f"{status_wr_beat} Win Rate beats baseline: {wr_improvement:+.2f}%")

    status_pf_beat = "✅" if criteria['profit_factor_beats_baseline'] else "❌"
    print(f"{status_pf_beat} Profit Factor beats baseline: {pf_improvement:+.2f}")

    status_sharpe_beat = "✅" if criteria['sharpe_beats_baseline'] else "❌"
    print(f"{status_sharpe_beat} Sharpe Ratio beats baseline: {sharpe_improvement:+.2f}")

    # Final decision
    all_pass = all(criteria.values())

    print_section("FINAL DECISION")

    if all_pass:
        print("\n✅ ✅ ✅ IMPLEMENT 0.95 THRESHOLD ✅ ✅ ✅")
        print("\nAll success criteria met on out-of-sample test data.")
        print("Recommendation: Update strategy to use exit_threshold = 0.95")
    else:
        print("\n❌ ❌ ❌ KEEP BASELINE (0.80) ❌ ❌ ❌")
        print("\nOne or more success criteria failed on test data.")
        print("Recommendation: Keep current exit_threshold = 0.80")

        # Show which failed
        failed = [k for k, v in criteria.items() if not v]
        print(f"\nFailed criteria: {', '.join(failed)}")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame([
        {
            'Metric': 'Total Trades',
            'Baseline': baseline_metrics['total_trades'],
            'Hypothesis': hypothesis_metrics['total_trades'],
            'Improvement': hypothesis_metrics['total_trades'] - baseline_metrics['total_trades']
        },
        {
            'Metric': 'Win Rate (%)',
            'Baseline': f"{baseline_metrics['win_rate']:.2f}",
            'Hypothesis': f"{hypothesis_metrics['win_rate']:.2f}",
            'Improvement': f"{wr_improvement:+.2f}"
        },
        {
            'Metric': 'Profit Factor',
            'Baseline': f"{baseline_metrics['profit_factor']:.2f}",
            'Hypothesis': f"{hypothesis_metrics['profit_factor']:.2f}",
            'Improvement': f"{pf_improvement:+.2f}"
        },
        {
            'Metric': 'Sharpe Ratio',
            'Baseline': f"{baseline_metrics['sharpe_ratio']:.2f}",
            'Hypothesis': f"{hypothesis_metrics['sharpe_ratio']:.2f}",
            'Improvement': f"{sharpe_improvement:+.2f}"
        },
        {
            'Metric': 'Total Return (%)',
            'Baseline': f"{baseline_metrics['total_return']:.2f}",
            'Hypothesis': f"{hypothesis_metrics['total_return']:.2f}",
            'Improvement': f"{hypothesis_metrics['total_return'] - baseline_metrics['total_return']:+.2f}"
        }
    ])

    output_csv = OUTPUT_DIR / "stage6_final_comparison_14tickers.csv"
    results_df.to_csv(output_csv, index=False)

    print(f"\n💾 Results saved to: {output_csv}")

    print("\n" + "=" * 80)

    return all_pass

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
