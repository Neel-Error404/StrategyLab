#!/usr/bin/env python3
"""
STAGE 6: FINAL OUT-OF-SAMPLE COMPARISON - ALL 24 TICKERS
Baseline (0.80) vs Hypothesis (0.95) - Complete Dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob

# ============================================================================
# CONFIGURATION
# ============================================================================

# Test period (out-of-sample - never used in optimization)
TEST_START = "2024-07-01"
TEST_END = "2025-08-31"

# All 24 tickers
TICKERS = [
    "ADANIPORTS", "ASIANPAINT", "AXISBANK", "HCLTECH", "HINDUNILVR", "INFY",
    "ITC", "JSWSTEEL", "KOTAKBANK", "LT", "MARUTI", "NESTLEIND", "NTPC",
    "ONGC", "POWERGRID", "RELIANCE", "SBIN", "SUNPHARMA", "TATASTEEL",
    "TCS", "TECHM", "TITAN", "ULTRACEMCO", "WIPRO"
]

# Paths
BASELINE_DIR = "outputs/20251006_024924/mse/2022-01-01_to_2025-08-31/data/strategy_trades"
HYPOTHESIS_DIR = "outputs/20251005_193937/mse/2022-01-01_to_2025-08-31/data/strategy_trades"
OUTPUT_FILE = "analysis/strategy_optimization/checkpoints/stage6_final_analysis_24tickers.csv"

# Success criteria
CRITERIA = {
    'win_rate_target': 52.0,
    'profit_factor_target': 1.25,
    'sharpe_ratio_target': 1.5
}

# ============================================================================
# FUNCTIONS
# ============================================================================

def load_trades(directory, ticker):
    """Load trades for a specific ticker"""
    pattern = f"{directory}/{ticker}_StrategyTrades_*.csv"
    files = glob.glob(pattern)

    if not files:
        return None

    df = pd.read_csv(files[0])
    df['Entry Time'] = pd.to_datetime(df['Entry Time'])

    # Remove timezone if present
    if df['Entry Time'].dt.tz is not None:
        df['Entry Time'] = df['Entry Time'].dt.tz_localize(None)

    return df

def filter_test_period(df, start_date, end_date):
    """Filter trades to test period"""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    mask = (df['Entry Time'] >= start) & (df['Entry Time'] <= end)
    return df[mask].copy()

def calculate_metrics(df):
    """Calculate performance metrics"""
    if len(df) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'avg_profit': 0,
            'total_return': 0,
            'max_drawdown': 0
        }

    # Win rate
    wins = df[df['Profit (%)'] > 0]
    losses = df[df['Profit (%)'] < 0]
    win_rate = (len(wins) / len(df)) * 100 if len(df) > 0 else 0

    # Profit factor
    total_profit = wins['Profit (%)'].sum() if len(wins) > 0 else 0
    total_loss = abs(losses['Profit (%)'].sum()) if len(losses) > 0 else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else 0

    # Sharpe ratio (annualized from daily returns)
    df_sorted = df.sort_values('Entry Time').copy()
    df_sorted['date'] = df_sorted['Entry Time'].dt.date
    daily_returns = df_sorted.groupby('date')['Profit (%)'].sum()

    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0

    # Other metrics
    avg_profit = df['Profit (%)'].mean()
    total_return = df['Profit (%)'].sum()

    # Max drawdown
    cumulative = df_sorted['Profit (%)'].cumsum()
    running_max = cumulative.expanding().max()
    drawdown = cumulative - running_max
    max_drawdown = drawdown.min()

    return {
        'total_trades': len(df),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe_ratio,
        'avg_profit': avg_profit,
        'total_return': total_return,
        'max_drawdown': max_drawdown
    }

def check_criteria(metrics, baseline_metrics):
    """Check if hypothesis meets all success criteria"""
    results = {
        'win_rate_meets_52': metrics['win_rate'] >= CRITERIA['win_rate_target'],
        'profit_factor_meets_1.25': metrics['profit_factor'] >= CRITERIA['profit_factor_target'],
        'sharpe_ratio_meets_1.5': metrics['sharpe_ratio'] >= CRITERIA['sharpe_ratio_target'],
        'win_rate_beats_baseline': metrics['win_rate'] > baseline_metrics['win_rate'],
        'profit_factor_beats_baseline': metrics['profit_factor'] > baseline_metrics['profit_factor'],
        'sharpe_ratio_beats_baseline': metrics['sharpe_ratio'] > baseline_metrics['sharpe_ratio']
    }

    all_pass = all(results.values())
    failed_criteria = [k for k, v in results.items() if not v]

    return results, all_pass, failed_criteria

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

print("=" * 80)
print("STAGE 6: FINAL OUT-OF-SAMPLE COMPARISON - ALL 24 TICKERS")
print("=" * 80)
print()
print(f"Test Period: {TEST_START} to {TEST_END}")
print(f"Tickers: {len(TICKERS)} stocks")
print(f"Tickers: {', '.join(TICKERS)}")
print()

# Load all data
print("=" * 80)
print("LOADING DATA")
print("=" * 80)

baseline_trades = []
hypothesis_trades = []

for ticker in TICKERS:
    # Baseline
    baseline_df = load_trades(BASELINE_DIR, ticker)
    if baseline_df is not None:
        print(f"✅ Baseline {ticker}: {len(baseline_df):,} trades")
        baseline_trades.append(baseline_df)
    else:
        print(f"❌ Baseline {ticker}: NOT FOUND")

    # Hypothesis
    hypothesis_df = load_trades(HYPOTHESIS_DIR, ticker)
    if hypothesis_df is not None:
        print(f"✅ Hypothesis {ticker}: {len(hypothesis_df):,} trades")
        hypothesis_trades.append(hypothesis_df)
    else:
        print(f"❌ Hypothesis {ticker}: NOT FOUND")

# Combine all tickers
baseline_all = pd.concat(baseline_trades, ignore_index=True)
hypothesis_all = pd.concat(hypothesis_trades, ignore_index=True)

print()
print(f"📊 Total Baseline trades: {len(baseline_all):,}")
print(f"📊 Total Hypothesis trades: {len(hypothesis_all):,}")
print()

# Filter to test period
print("=" * 80)
print("FILTERING TO TEST PERIOD (OUT-OF-SAMPLE)")
print("=" * 80)
print()

baseline_test = filter_test_period(baseline_all, TEST_START, TEST_END)
hypothesis_test = filter_test_period(hypothesis_all, TEST_START, TEST_END)

print(f"📅 Baseline test trades: {len(baseline_test):,}")
print(f"📅 Hypothesis test trades: {len(hypothesis_test):,}")
print()

# Calculate metrics
print("=" * 80)
print("PERFORMANCE METRICS")
print("=" * 80)
print()

baseline_metrics = calculate_metrics(baseline_test)
hypothesis_metrics = calculate_metrics(hypothesis_test)

print("🔵 BASELINE (0.80 Threshold):")
print(f"  Total Trades: {baseline_metrics['total_trades']:,}")
print(f"  Win Rate: {baseline_metrics['win_rate']:.2f}%")
print(f"  Profit Factor: {baseline_metrics['profit_factor']:.2f}")
print(f"  Sharpe Ratio: {baseline_metrics['sharpe_ratio']:.2f}")
print(f"  Avg Profit: {baseline_metrics['avg_profit']:.3f}%")
print(f"  Total Return: {baseline_metrics['total_return']:.2f}%")
print(f"  Max Drawdown: {baseline_metrics['max_drawdown']:.2f}%")
print()

print("🟢 HYPOTHESIS (0.95 Threshold):")
print(f"  Total Trades: {hypothesis_metrics['total_trades']:,}")
print(f"  Win Rate: {hypothesis_metrics['win_rate']:.2f}%")
print(f"  Profit Factor: {hypothesis_metrics['profit_factor']:.2f}")
print(f"  Sharpe Ratio: {hypothesis_metrics['sharpe_ratio']:.2f}")
print(f"  Avg Profit: {hypothesis_metrics['avg_profit']:.3f}%")
print(f"  Total Return: {hypothesis_metrics['total_return']:.2f}%")
print(f"  Max Drawdown: {hypothesis_metrics['max_drawdown']:.2f}%")
print()

# Improvements
print("=" * 80)
print("IMPROVEMENTS")
print("=" * 80)
print()

wr_delta = hypothesis_metrics['win_rate'] - baseline_metrics['win_rate']
pf_delta = hypothesis_metrics['profit_factor'] - baseline_metrics['profit_factor']
sr_delta = hypothesis_metrics['sharpe_ratio'] - baseline_metrics['sharpe_ratio']
ret_delta = hypothesis_metrics['total_return'] - baseline_metrics['total_return']

print(f"📈 Win Rate: {wr_delta:+.2f}% ({baseline_metrics['win_rate']:.2f}% → {hypothesis_metrics['win_rate']:.2f}%)")
print(f"📈 Profit Factor: {pf_delta:+.2f} ({baseline_metrics['profit_factor']:.2f} → {hypothesis_metrics['profit_factor']:.2f})")
print(f"📈 Sharpe Ratio: {sr_delta:+.2f} ({baseline_metrics['sharpe_ratio']:.2f} → {hypothesis_metrics['sharpe_ratio']:.2f})")
print(f"📈 Total Return: {ret_delta:+.2f}% ({baseline_metrics['total_return']:.2f}% → {hypothesis_metrics['total_return']:.2f}%)")
print()

# Check criteria
print("=" * 80)
print("SUCCESS CRITERIA CHECK")
print("=" * 80)
print()
print("✅ = PASS | ❌ = FAIL")
print()

criteria_results, all_pass, failed = check_criteria(hypothesis_metrics, baseline_metrics)

for criterion, result in criteria_results.items():
    status = "✅" if result else "❌"

    if criterion == 'win_rate_meets_52':
        print(f"{status} Win Rate ≥ {CRITERIA['win_rate_target']}%: {hypothesis_metrics['win_rate']:.2f}%")
    elif criterion == 'profit_factor_meets_1.25':
        print(f"{status} Profit Factor ≥ {CRITERIA['profit_factor_target']}: {hypothesis_metrics['profit_factor']:.2f}")
    elif criterion == 'sharpe_ratio_meets_1.5':
        print(f"{status} Sharpe Ratio ≥ {CRITERIA['sharpe_ratio_target']}: {hypothesis_metrics['sharpe_ratio']:.2f}")
    elif criterion == 'win_rate_beats_baseline':
        print(f"{status} Win Rate beats baseline: {wr_delta:+.2f}%")
    elif criterion == 'profit_factor_beats_baseline':
        print(f"{status} Profit Factor beats baseline: {pf_delta:+.2f}")
    elif criterion == 'sharpe_ratio_beats_baseline':
        print(f"{status} Sharpe Ratio beats baseline: {sr_delta:+.2f}")

print()

# Final decision
print("=" * 80)
print("FINAL DECISION")
print("=" * 80)
print()

if all_pass:
    print("✅ ✅ ✅ IMPLEMENT 0.95 THRESHOLD ✅ ✅ ✅")
    print()
    print("All success criteria met on out-of-sample test data.")
    print("Recommendation: Update exit_threshold = 0.95 in mse_strategy_backtesting.py")
else:
    print("❌ ❌ ❌ KEEP BASELINE (0.80) ❌ ❌ ❌")
    print()
    print("One or more success criteria failed on test data.")
    print("Recommendation: Keep current exit_threshold = 0.80")
    print()
    print(f"Failed criteria: {', '.join(failed)}")

print()

# Save results
output_path = Path(OUTPUT_FILE)
output_path.parent.mkdir(parents=True, exist_ok=True)

results_df = pd.DataFrame([
    {
        'threshold': '0.80 (Baseline)',
        'total_trades': baseline_metrics['total_trades'],
        'win_rate': baseline_metrics['win_rate'],
        'profit_factor': baseline_metrics['profit_factor'],
        'sharpe_ratio': baseline_metrics['sharpe_ratio'],
        'avg_profit': baseline_metrics['avg_profit'],
        'total_return': baseline_metrics['total_return'],
        'max_drawdown': baseline_metrics['max_drawdown']
    },
    {
        'threshold': '0.95 (Hypothesis)',
        'total_trades': hypothesis_metrics['total_trades'],
        'win_rate': hypothesis_metrics['win_rate'],
        'profit_factor': hypothesis_metrics['profit_factor'],
        'sharpe_ratio': hypothesis_metrics['sharpe_ratio'],
        'avg_profit': hypothesis_metrics['avg_profit'],
        'total_return': hypothesis_metrics['total_return'],
        'max_drawdown': hypothesis_metrics['max_drawdown']
    }
])

results_df.to_csv(output_path, index=False)
print(f"💾 Results saved to: {output_path.absolute()}")
print()
print("=" * 80)
