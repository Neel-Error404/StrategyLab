# Stage 1: Baseline Establishment Report

**Date**: 2025-10-04 12:44:25
**Period**: Validation Data (2024-01-01 to 2024-06-30)
**Tickers**: 24 tickers
**Total Trades Analyzed**: 8,186

---

## 📊 TRADITIONAL METRICS

### Trade Statistics
- **Total Trades**: 8,186
- **Winning Trades**: 4,062 (49.62%)
- **Losing Trades**: 4,013
- **Breakeven Trades**: 111

### Performance Metrics
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Win Rate** | 49.62% | ≥52% | ❌ |
| **Profit Factor** | 1.87 | ≥1.25 | ✅ |
| **Average Win** | 0.46% | - | - |
| **Average Loss** | 0.25% | - | - |
| **Risk-Reward Ratio** | 1.85 | - | - |
| **Expectancy** | 0.10% per trade | - | - |

### Return Metrics
| Metric | Value |
|--------|-------|
| **Total Return** | 865.36% |
| **Return on Capital** | 491199.09% |
| **Final Equity** | $491,299,090.74 |

### Risk Metrics
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Max Drawdown** | -7.65% | ≤15% | ✅ |
| **Sharpe Ratio** | 2.75 | ≥1.5 | ✅ |

### Duration Metrics
- **Average Trade Duration**: 0.92 hours
- **Median Trade Duration**: 0.67 hours

### Streaks
- **Max Consecutive Wins**: 14
- **Max Consecutive Losses**: 11

---

## 🎯 MAE/MFE EXIT EFFICIENCY ANALYSIS


### Data Coverage
- **Total Trades**: 8,186
- **Valid MAE/MFE Data**: 8,186 (100.0%)

### Maximum Favorable Excursion (MFE) - Best Price Available
| Metric | Value |
|--------|-------|
| **Average MFE** | 0.53% |
| **Median MFE** | 0.33% |
| **Std Dev** | 0.71% |

### Maximum Adverse Excursion (MAE) - Worst Drawdown
| Metric | Value |
|--------|-------|
| **Average MAE** | 0.30% |
| **Median MAE** | 0.22% |
| **Std Dev** | 0.28% |

### MFE Capture Ratio - % of Available Profit Captured
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Average Capture Ratio** | -85.39% | ≥70% | ❌ |
| **Median Capture Ratio** | 0.00% | - | - |

### Exit Efficiency Score - Overall Exit Quality
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Average Efficiency Score** | -177.02 | ≥55 | ❌ |
| **Median Efficiency Score** | -30.12 | - | - |

### Exit Efficiency Distribution
| Category | Count | Percentage |
|----------|-------|------------|
| **Excellent (>70)** | 637 | 7.8% |
| **Good (50-70)** | 732 | 8.9% |
| **Poor (30-50)** | 792 | 9.7% |
| **Terrible (<30)** | 6,025 | 73.6% |

### Potential Left on Table
| Metric | Value |
|--------|-------|
| **Average per Trade** | 0.42% |
| **Total Across All Trades** | 3434.83% |

### MAE/MFE Ratio - Drawdown vs Profit Potential
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Average Ratio** | 1.83 | ≤0.6 | ❌ |
| **Median Ratio** | 0.65 | - | - |

---

## 📋 BASELINE ASSESSMENT

### Success Criteria Check

**Traditional Metrics**:
- Win Rate ≥52%: ❌ FAIL
- Profit Factor ≥1.25: ✅ PASS
- Max Drawdown ≤15%: ✅ PASS
- Sharpe Ratio ≥1.5: ✅ PASS

**Exit Efficiency Metrics**:
- MFE Capture Ratio ≥70%: ❌ FAIL
- Exit Efficiency Score ≥55: ❌ FAIL
- MAE/MFE Ratio ≤0.6: ❌ FAIL

### Key Insights

**Current State**:
- Win Rate: 49.62% (Target: 52%)
- Profit Factor: 1.87 (Target: 1.25)
- MFE Capture Ratio: -85.39% (Target: 70%)
- Exit Efficiency Score: -177.02 (Target: 55)
- Potential Left on Table: 0.42% per trade

### Recommendations

**Proceed to Stage 2 (Exit Threshold Optimization)?**

✅ **YES** - MFE Capture Ratio is below target (-85.39% < 70%)
✅ **YES** - Exit Efficiency Score is below target (-177.02 < 55)

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

*Report generated on 2025-10-04 12:44:25*
