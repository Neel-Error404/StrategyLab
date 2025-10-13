# Stage 6 Execution Guide - Quick Reference

**Status**: Backtests running at 0.80 and 0.95 thresholds
**Next**: Merge trades and analyze

---

## 📋 CURRENT SITUATION

You're running two backtests via command line:
1. **Baseline**: 0.80 threshold (2022-2025)
2. **Optimal**: 0.95 threshold (2022-2025)

**Data Periods**:
- Train: 2022-2023 (NOT used in comparison)
- Validation: 2024 H1 (used in Stages 1-4)
- **Test**: 2024 H2 - 2025 ← **THIS is what we compare**

---

## ✅ WHEN BACKTESTS COMPLETE

### **Step 1: Locate Trade Files**

Both backtests will produce:
```
outputs/[timestamp]/mse_backtesting/2022-01-01_to_2025-08-31/all_trade_merged.csv
```

**Find them**:
```bash
# Baseline run (0.80)
ls -lt outputs/*/mse_backtesting/2022-01-01_to_2025-08-31/all_trade_merged.csv | head -2

# Note the paths
BASELINE_PATH="/path/to/baseline/all_trade_merged.csv"
OPTIMAL_PATH="/path/to/optimal/all_trade_merged.csv"
```

---

### **Step 2: Run Analysis Script**

I'll create a script that:
1. Loads both trade files
2. Filters to TEST period (2024-07-01 to 2025-08-31)
3. Calculates all metrics
4. Compares performance
5. Makes GO/NO-GO decision

**Command**:
```bash
cd /mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/StrategyLab-master/analysis/strategy_optimization

python scripts/06_final_test_comparison.py \
  --baseline "/path/to/baseline/all_trade_merged.csv" \
  --optimal "/path/to/optimal/all_trade_merged.csv" \
  --test-start "2024-07-01" \
  --test-end "2025-08-31"
```

---

## 🎯 SUCCESS CRITERIA (BOTH Must Pass)

### **Criterion 1: Relative Performance**
95% threshold MUST outperform 80% on test data:
```
Optimal WR > Baseline WR  AND
Optimal PF > Baseline PF
```

### **Criterion 2: Absolute Performance**
95% threshold MUST meet ALL targets:
```
Win Rate ≥ 52%
Profit Factor ≥ 1.25
Sharpe Ratio ≥ 1.5
```

---

## 📊 EXPECTED RESULTS

Based on Stages 1-4:
```
BASELINE (0.80):
  Win Rate: ~50.5%
  Profit Factor: ~2.28
  Sharpe: ~3.5

OPTIMAL (0.95):
  Win Rate: ~53.0% (+2.5%)
  Profit Factor: ~2.68 (+0.40)
  Sharpe: ~4.0 (+0.5)
```

**Prediction**: Both criteria should PASS

---

## 🔴 IF CRITERIA FAIL

### **Scenario 1: Criterion 1 Fails** (95% worse than 80%)
**Meaning**: Validation data was unrepresentative
**Action**: REJECT optimization, keep 80% baseline
**Root Cause**: Overfitting to 2024 H1 market conditions

### **Scenario 2: Criterion 2 Fails** (Doesn't meet targets)
**Meaning**: Performance degraded on unseen data
**Action**: REJECT optimization, keep 80% baseline
**Root Cause**: Walk-forward validation warnings were valid

### **Scenario 3: Both Fail**
**Meaning**: Optimization completely failed
**Action**: REJECT, investigate why
**Options**:
- Analyze what changed between validation and test
- Check if test period has different market regime
- Consider regime-specific thresholds

---

## ✅ IF CRITERIA PASS

### **Immediate Actions**:
1. Generate final report
2. Update `PHASE2_ANALYSIS_LOG.md` with Stage 6 completion
3. Create implementation plan

### **Implementation Plan**:
```
1. Update Production Config:
   - Exit threshold: 0.80 → 0.95

2. Paper Trading (1 month):
   - Deploy with 0% capital (track signals only)
   - Monitor: WR, PF, trade count, parameter drift

3. Live Deployment (gradual):
   - Week 1: 10% allocation
   - Week 2: 25% allocation
   - Week 3: 50% allocation
   - Week 4: 100% allocation (if performance holds)

4. Monitoring Setup:
   - Daily: WR, PF, trade count
   - Weekly: Sharpe, max DD, parameter stability
   - Monthly: Full metrics review
   - Alert if WR drops below 50% for 3 days
```

---

## 📁 FILES TO PROVIDE IN NEW CONVERSATION

If context is lost, these files have everything:

**Primary Context**:
- `docs/STAGE6_CONTEXT_HANDOFF.md` ← **START HERE**
- `docs/STAGE6_EXECUTION_GUIDE.md` ← **THIS FILE**

**Supporting**:
- `docs/stage4_statistical_validation_report.md` (p < 0.0001 proof)
- `checkpoints/stage4_statistical_results.csv` (detailed results)
- `config/optimization_config.yaml` (all parameters)

**Summary Statement for New Conversation**:
```
We optimized MSE strategy exit threshold from 80% to 95%.
Completed Stages 1-4 (baseline, optimization, walk-forward, statistical validation).
Stage 4 showed p < 0.0001 (extremely significant).
Now at Stage 6: comparing performance on unseen 2024 H2 - 2025 test data.
Running two backtests (0.80 vs 0.95), will merge trades and analyze.
```

---

## 🛠️ TROUBLESHOOTING

### **Can't find trade files**
```bash
find outputs/ -name "all_trade_merged.csv" -newer [baseline_run_time]
```

### **Trades look wrong**
```bash
# Check row count
wc -l /path/to/all_trade_merged.csv

# Check date range
python -c "import pandas as pd; df = pd.read_csv('[path]'); print(df['Entry Time'].min(), df['Entry Time'].max())"

# Check thresholds used
grep -i "threshold\|exit" /path/to/config_or_log
```

### **Script not working**
```bash
# Run in debug mode
python -u scripts/06_final_test_comparison.py \
  --baseline "[path]" \
  --optimal "[path]" \
  --test-start "2024-07-01" \
  --test-end "2025-08-31" \
  --debug
```

---

**Created**: 2025-10-05
**Status**: Waiting for backtests to complete
**Next**: Run analysis script when both trades ready
