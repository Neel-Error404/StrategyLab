#!/usr/bin/env python3
"""
Quick feasibility check: How many combinations per portfolio size?
"""
import math

n_tickers = 28  # Affordable tickers from Script 1

print("=" * 80)
print("📊 COMPUTATIONAL FEASIBILITY ANALYSIS")
print("=" * 80)
print(f"\nAvailable tickers: {n_tickers}")
print(f"Basis: 5-ticker analysis completed 98,280 combinations in ~2 minutes\n")

portfolio_sizes = [3, 4, 5, 6, 7, 8, 10, 12]

print(f"{'Size':<6} {'Combinations':>15} {'Est. Time':>12} {'Feasibility':<20}")
print("-" * 80)

baseline_combinations = math.comb(28, 5)  # 98,280
baseline_time_minutes = 2

for size in portfolio_sizes:
    combinations = math.comb(n_tickers, size)
    ratio = combinations / baseline_combinations
    est_time = baseline_time_minutes * ratio

    if est_time < 5:
        feasibility = "⚡ Very Fast"
    elif est_time < 30:
        feasibility = "✅ Fast"
    elif est_time < 120:
        feasibility = "🔸 Medium (~1-2 hours)"
    elif est_time < 360:
        feasibility = "🔶 Slow (2-6 hours)"
    else:
        feasibility = "🔴 Very Slow (6+ hours)"

    time_str = f"{est_time:.1f} min" if est_time < 60 else f"{est_time/60:.1f} hrs"

    print(f"{size:<6} {combinations:>15,} {time_str:>12} {feasibility:<20}")

print("\n" + "=" * 80)
print("💡 RECOMMENDATION: Run portfolio sizes 4, 5, 6, 7, 8")
print("   Total estimated time: ~100 minutes (~1.5 hours)")
print("   Skip 10+ tickers (diminishing returns + over-diversification)")
print("=" * 80)
