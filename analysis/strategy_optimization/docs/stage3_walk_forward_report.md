# Stage 3: Walk-Forward Validation Report

**Date**: 2025-10-05 03:33:27
**Method**: Rolling window validation with 30-day windows
**Purpose**: Verify 95% threshold stability across different time periods

---

## 📋 METHODOLOGY

**Walk-Forward Validation** tests if the optimal threshold from Stage 2 (95%) is:
- **Stable**: Performance doesn't vary wildly across different months
- **Robust**: Consistently optimal in different market conditions
- **Not overfitted**: Works on periods not used for optimization

**Approach:**
1. Split validation period (2024-01-01 to 2024-06-30) into rolling 30-day windows
2. Step by 15 days between windows (50% overlap)
3. For each window, test all thresholds: 50%, 60%, 70%, 80%, 90%, 95%
4. Find optimal threshold for each window
5. Check if 95% is consistently optimal

**Success Criteria:**
- Coefficient of Variation (CV) < 10% for Win Rate and Profit Factor
- 95% threshold optimal in ≥70% of windows
- Consistency rate ≥70% (windows meeting success targets)

---

## 🎯 BUY TRADES STABILITY

### Overall Performance Across 12 Windows
| Metric | Value |
|--------|-------|
| **Avg Win Rate** | 53.78% |
| **Avg Profit Factor** | 2.67 |
| **Avg Sharpe Ratio** | 4.35 |

### Stability Metrics (CV = Coefficient of Variation)
| Metric | CV | Target | Status |
|--------|----|---------| -------|
| **Win Rate** | 2.28% | <10% | ✅ |
| **Profit Factor** | 14.70% | <10% | ❌ |
| **Sharpe Ratio** | 12.86% | <10% | ⚠️ |

**Interpretation:**
- CV < 5%: Very stable
- CV 5-10%: Stable (acceptable)
- CV > 10%: Unstable (overfitting risk)

### Consistency (Windows Meeting Success Criteria)
| Criteria | Windows | Rate | Status |
|----------|---------|------|--------|
| **Win Rate ≥52%** | 11/12 | 91.7% | ✅ |
| **Profit Factor ≥1.25** | 12/12 | 100.0% | ✅ |
| **Sharpe Ratio ≥1.5** | 12/12 | 100.0% | ✅ |
| **Overall Consistency** | - | 97.2% | ✅ |

### Optimality (Is 95% Best Across Windows?)
- **Windows where 95% was optimal**: 12/12 (100.0%)
- **Target**: ≥70% → ✅ PASS

### Win Rate Range
- **Min**: 51.30%
- **Max**: 55.54%
- **Range**: 4.23%

---

## 🎯 SELL TRADES STABILITY

### Overall Performance Across 12 Windows
| Metric | Value |
|--------|-------|
| **Avg Win Rate** | 52.81% |
| **Avg Profit Factor** | 2.65 |
| **Avg Sharpe Ratio** | 4.11 |

### Stability Metrics (CV = Coefficient of Variation)
| Metric | CV | Target | Status |
|--------|----|---------| -------|
| **Win Rate** | 5.19% | <10% | ✅ |
| **Profit Factor** | 16.17% | <10% | ❌ |
| **Sharpe Ratio** | 25.18% | <10% | ⚠️ |

### Consistency (Windows Meeting Success Criteria)
| Criteria | Windows | Rate | Status |
|----------|---------|------|--------|
| **Win Rate ≥52%** | 7/12 | 58.3% | ❌ |
| **Profit Factor ≥1.25** | 12/12 | 100.0% | ✅ |
| **Sharpe Ratio ≥1.5** | 12/12 | 100.0% | ✅ |
| **Overall Consistency** | - | 86.1% | ✅ |

### Optimality (Is 95% Best Across Windows?)
- **Windows where 95% was optimal**: 12/12 (100.0%)
- **Target**: ≥70% → ✅ PASS

### Win Rate Range
- **Min**: 49.13%
- **Max**: 57.52%
- **Range**: 8.39%

---

## 🚦 DECISION GATE

**Buy Trades**: ❌ FAIL
- Reason: Profit Factor CV too high (14.7% ≥ 10%)

**Sell Trades**: ❌ FAIL
- Reason: Profit Factor CV too high (16.2% ≥ 10%)


### ❌ RECOMMENDATION: STOP - THRESHOLD NOT STABLE

**95% threshold is NOT stable across time windows.**

**The optimal threshold from Stage 2 appears to be overfitted to the full validation period.**

**Options:**
1. **Re-run Stage 2 with different threshold range** (e.g., test 85%, 92%, 97%)
2. **Accept a more stable threshold** (check which threshold has best stability)
3. **Skip exit optimization** and proceed to Stage 5 (Entry Filter Optimization)

**Do NOT proceed to final testing** without addressing stability issues.


---

## 📁 OUTPUTS

**Checkpoint**: `checkpoints/stage3_walk_forward_results.csv`
**Visualizations**: `docs/stage3_window_stability_*.png`
**Stability Metrics**: `checkpoints/stage3_stability_metrics.csv`

---

*Report generated on 2025-10-05 03:33:27*
