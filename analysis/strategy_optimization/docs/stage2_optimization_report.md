# Stage 2: Exit Threshold Optimization Report

**Date**: 2025-10-04 17:14:01
**Method**: Simulation of alternative exit thresholds on validation data
**Thresholds Tested**: 10 values (50% to 95% in 5% steps)

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
| **Total Trades** | 4,076 |
| **Win Rate** | 50.96% |
| **Profit Factor** | 2.30 |
| **Sharpe Ratio** | 3.77 |
| **Avg Duration** | 57.0 minutes |
| **Total Return** | 535.60% |

### Optimal Threshold: 95%

| Metric | Value | vs Baseline | Status |
|--------|-------|-------------|--------|
| **Win Rate** | 53.75% | +2.80% | ✅ |
| **Profit Factor** | 2.65 | +0.34 | ✅ |
| **Sharpe Ratio** | 4.32 | +0.55 | ✅ |
| **Total Return** | 607.71% | +72.11% | ✅ |
| **Avg Duration** | 56.1 min | -0.9 min | - |

### Outcome Changes (Buy)
- **Losers → Winners**: 208 trades
- **Winners → Losers**: 44 trades
- **Net Improvement**: 164 trades

---

## 🎯 SELL TRADES OPTIMIZATION

### Baseline (80% Threshold)
| Metric | Value |
|--------|-------|
| **Total Trades** | 4,110 |
| **Win Rate** | 50.46% |
| **Profit Factor** | 2.28 |
| **Sharpe Ratio** | 3.16 |
| **Avg Duration** | 51.2 minutes |
| **Total Return** | 519.28% |

### Optimal Threshold: 95%

| Metric | Value | vs Baseline | Status |
|--------|-------|-------------|--------|
| **Win Rate** | 52.82% | +2.36% | ✅ |
| **Profit Factor** | 2.68 | +0.40 | ✅ |
| **Sharpe Ratio** | 3.68 | +0.52 | ✅ |
| **Total Return** | 598.54% | +79.27% | ✅ |
| **Avg Duration** | 50.4 min | -0.9 min | - |

### Outcome Changes (Sell)
- **Losers → Winners**: 213 trades
- **Winners → Losers**: 77 trades
- **Net Improvement**: 136 trades

---

## 📊 KEY INSIGHTS

**Buy vs Sell Threshold Difference**:
- Buy optimal: 95%
- Sell optimal: 95%
- Difference: 0%


✅ **Same optimal threshold for both trade types** - simpler implementation

💡 **Buy trades benefit from later exits** (95% vs 80%) - letting winners run

💡 **Sell trades benefit from later exits** (95% vs 80%) - letting winners run


---

## 🚦 SUCCESS CRITERIA CHECK

**Target Metrics (from optimization_config.yaml)**:
- Win Rate ≥ 52%
- Profit Factor ≥ 1.25
- Sharpe Ratio ≥ 1.5

**Buy Trades**:
- Win Rate: 53.75% ✅
- Profit Factor: 2.65 ✅
- Sharpe Ratio: 4.32 ✅

**Sell Trades**:
- Win Rate: 52.82% ✅
- Profit Factor: 2.68 ✅
- Sharpe Ratio: 3.68 ✅

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

*Report generated on 2025-10-04 17:14:01*
