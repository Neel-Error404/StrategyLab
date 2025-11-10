# Strategy Optimization - MSE Exit & Entry Analysis

**Status**: Phase 2 In Progress
**Objective**: Improve MSE strategy from WR 48%→52%, PF 1.14→1.25
**Approach**: Hypothesis-driven analysis with MAE/MFE exit efficiency metrics

---

## 📁 Directory Structure

```
strategy_optimization/
├── scripts/              # Analysis scripts (executed in sequence)
│   ├── 00_data_loader.py
│   ├── 01_baseline_calculator.py
│   ├── 02_exit_threshold_optimizer.py
│   ├── 03_entry_filter_optimizer.py
│   ├── 04_walkforward_validator.py
│   ├── 05_statistical_tester.py
│   └── 06_final_verifier.py
│
├── modules/              # Reusable modules (copied locally, not imported)
│   ├── trade_enhancer.py         # Links trades to base_data
│   ├── mae_mfe_calculator.py     # Exit efficiency metrics
│   ├── exit_simulator.py         # Test different exit thresholds
│   ├── metrics_calculator.py     # Performance metrics
│   └── visualizer.py             # Chart generation
│
├── data/
│   └── base_data/ → symlink to outputs/.../base_data/
│
├── checkpoints/          # Revert points for each stage
│   ├── stage0_setup_complete/
│   ├── stage1_baseline_data.csv
│   ├── stage2_exit_optimization_results.csv
│   └── ...
│
├── docs/
│   ├── PHASE2_ANALYSIS_LOG.md    # Main analysis journal (CRITICAL)
│   ├── baseline_report.md
│   ├── exit_threshold_analysis_report.md
│   └── ...
│
├── logs/                 # Execution logs
│   ├── stage1_execution.log
│   ├── stage2_execution.log
│   └── ...
│
├── config/
│   └── optimization_config.yaml  # All parameters
│
└── README.md            # This file
```

---

## 📋 KEY DOCUMENTS

### 1. **PHASE2_ANALYSIS_LOG.md** (MOST IMPORTANT)
- Running journal of entire analysis
- Checkpoint/revert points for each stage
- Observations and insights
- Decision log (why we chose specific paths)

**Read this FIRST before doing anything**

### 2. **optimization_config.yaml**
- All parameters in one place
- Tickers, date splits, thresholds to test
- Success criteria

### 3. **Stage Reports** (Generated during execution)
- `baseline_report.md` - Current performance
- `exit_threshold_analysis_report.md` - H1 results
- `entry_filter_analysis_report.md` - H2 results
- `walkforward_validation_report.md` - Parameter stability
- `statistical_significance_report.md` - Bootstrap tests
- `final_verification_report.md` - Test data results
- `EXECUTIVE_SUMMARY.md` - Final Go/No-Go decision

---

## 🚀 EXECUTION WORKFLOW

### Stage-by-Stage Execution (DO NOT SKIP)

```bash
# Stage 0: Setup (verify everything is ready)
python scripts/00_setup_verification.py

# Review: Check docs/PHASE2_ANALYSIS_LOG.md Stage 0
# Decision: Proceed to Stage 1? [YES/NO]

# Stage 1: Baseline
python scripts/01_baseline_calculator.py

# Review: Read docs/baseline_report.md
# Review: Update PHASE2_ANALYSIS_LOG.md Stage 1 observations
# Decision: Baseline acceptable? Proceed to H1? [YES/NO]

# Stage 2: Exit Threshold Optimization (H1)
python scripts/02_exit_threshold_optimizer.py

# Review: Read docs/exit_threshold_analysis_report.md
# Review: Update PHASE2_ANALYSIS_LOG.md Stage 2 observations
# Decision: Improvement ≥5% PF? Accept H1? [YES/NO]

# Stage 3: Entry Filter Optimization (H2)
python scripts/03_entry_filter_optimizer.py

# Review: Read docs/entry_filter_analysis_report.md
# Review: Update PHASE2_ANALYSIS_LOG.md Stage 3 observations
# Decision: Improvement ≥10% WR? Accept H2? [YES/NO]

# Stage 4: Walk-Forward Validation
python scripts/04_walkforward_validator.py

# Review: Read docs/walkforward_validation_report.md
# Review: Update PHASE2_ANALYSIS_LOG.md Stage 4 observations
# Decision: Parameters stable? Performance consistent? [YES/NO]

# Stage 5: Statistical Testing
python scripts/05_statistical_tester.py

# Review: Read docs/statistical_significance_report.md
# Review: Update PHASE2_ANALYSIS_LOG.md Stage 5 observations
# Decision: p-value < 0.05? Statistically significant? [YES/NO]

# Stage 6: Final Test (TEST DATA - NEVER SEEN BEFORE)
python scripts/06_final_verifier.py

# Review: Read docs/final_verification_report.md
# Review: Read docs/EXECUTIVE_SUMMARY.md
# FINAL DECISION: ✅ DEPLOY or ❌ REJECT
```

---

## ⚠️ CRITICAL RULES

### 1. **Never Skip Stages**
- Each stage builds on previous
- Must review and approve before next stage

### 2. **Document Everything in PHASE2_ANALYSIS_LOG.md**
- Observations (what you noticed)
- Insights (patterns discovered)
- Decisions (why you chose this path)
- **This is your memory and revert point**

### 3. **Test Data is Sacred**
- Stage 6 uses test data (2024-07-01 to 2025-08-31)
- This data is **NEVER** used for optimization
- Only touched ONCE at the very end
- If test fails = optimization is rejected (overfitting)

### 4. **Checkpoint After Each Stage**
- Save results to `checkpoints/stageN_*.csv`
- Can revert if next stage reveals issues
- Git commit after each successful stage

### 5. **Manual Review Required**
- Scripts generate reports
- Human must review and approve
- Update PHASE2_ANALYSIS_LOG.md with observations
- Make explicit decision: [PROCEED / STOP / REVERT]

---

## 🎯 SUCCESS CRITERIA (Must Meet ALL on Test Data)

1. **Win Rate**: ≥ 52% (from 48%)
2. **Profit Factor**: ≥ 1.25 (from 1.14)
3. **MFE Capture Ratio**: ≥ 70% (from ~60%)
4. **Exit Efficiency Score**: ≥ 55 (from ~45)
5. **Potential Left on Table**: ≤ 1.0% per trade
6. **Sharpe Ratio**: ≥ 1.5
7. **Max Drawdown**: ≤ 15%
8. **Parameter Stability**: CV < 10%
9. **Trade Frequency**: ≥ 70% of baseline
10. **Statistical Significance**: p < 0.05

**If ANY criterion fails on test data → REJECT optimization**

---

## 🔄 HOW TO REVERT

If a stage fails or reveals issues:

```bash
# Identify which stage to revert to (from PHASE2_ANALYSIS_LOG.md)
# Example: Revert to baseline (before exit optimization)

git checkout stage1_baseline

# Or manually:
# 1. Delete subsequent checkpoint files
# 2. Restore stage1_baseline_data.csv as current state
# 3. Update PHASE2_ANALYSIS_LOG.md: Mark stages 2+ as "REVERTED"
# 4. Re-run from that stage with new approach
```

---

## 📊 CURRENT STATUS

**Stage**: 0 (Setup)
**Last Updated**: 2025-10-04
**Next Action**: Run `00_setup_verification.py` and review results

**Progress**:
- [ ] Stage 0: Setup & Preparation
- [ ] Stage 1: Baseline Establishment
- [ ] Stage 2: Exit Threshold Optimization (H1)
- [ ] Stage 3: Entry Filter Optimization (H2)
- [ ] Stage 4: Walk-Forward Validation
- [ ] Stage 5: Statistical Significance Testing
- [ ] Stage 6: Final Test Data Verification

---

## 💡 QUICK START

```bash
# 1. Read the analysis log (understand the approach)
cat docs/PHASE2_ANALYSIS_LOG.md

# 2. Verify setup
python scripts/00_setup_verification.py

# 3. Follow the workflow above, one stage at a time
# 4. Update PHASE2_ANALYSIS_LOG.md after EACH stage
# 5. Make explicit decisions before proceeding

# Remember: Slow, methodical, documented!
```

---

**Philosophy**: "Measure twice, cut once. Document everything. Trust nothing until validated on unseen data."
