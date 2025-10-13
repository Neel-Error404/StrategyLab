# Stage 6: Final Validation Execution Guide

## **Mission**: Test 95% threshold on REAL unseen data (2024 H2)

---

## **Step 1: Run Baseline Backtest (80% threshold)**

### 1.1 Modify MSE Strategy
```bash
# Edit file
nano src/strategies/mse_strategy_backtesting.py

# Find line 80 and ensure it says:
self.exit_threshold = 0.80  # Baseline threshold
```

### 1.2 Run Backtest
```bash
cd /mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/StrategyLab-master

python src/runners/unified_runner.py \
  --mode backtest \
  --template conservative \
  --dates 2022-01-01 \
  --end-date 2025-08-31 \
  --tickers RELIANCE,TCS,INFY,HINDUNILVR,ITC,SBIN,KOTAKBANK,LT,ASIANPAINT,AXISBANK,MARUTI,SUNPHARMA,TITAN,ULTRACEMCO,WIPRO,NESTLEIND,HCLTECH,POWERGRID,NTPC,ONGC,TATASTEEL,JSWSTEEL,ADANIPORTS,TECHM
```

### 1.3 Note Output Directory
```bash
# Example output directory:
outputs/20251005_153045/mse_backtesting/2022-01-01_to_2025-08-31/

# Save this path! You'll need it for Step 3
```

---

## **Step 2: Run Optimal Backtest (95% threshold)**

### 2.1 Modify MSE Strategy
```bash
# Edit file
nano src/strategies/mse_strategy_backtesting.py

# Change line 80 to:
self.exit_threshold = 0.95  # Optimal threshold
```

### 2.2 Run Backtest (Same Command)
```bash
python src/runners/unified_runner.py \
  --mode backtest \
  --template conservative \
  --dates 2022-01-01 \
  --end-date 2025-08-31 \
  --tickers RELIANCE,TCS,INFY,HINDUNILVR,ITC,SBIN,KOTAKBANK,LT,ASIANPAINT,AXISBANK,MARUTI,SUNPHARMA,TITAN,ULTRACEMCO,WIPRO,NESTLEIND,HCLTECH,POWERGRID,NTPC,ONGC,TATASTEEL,JSWSTEEL,ADANIPORTS,TECHM
```

### 2.3 Note Output Directory
```bash
# Example:
outputs/20251005_163012/mse_backtesting/2022-01-01_to_2025-08-31/

# Save this path too!
```

### 2.4 Restore Original Threshold
```bash
# Edit file back to original
nano src/strategies/mse_strategy_backtesting.py

# Change line 80 back to:
self.exit_threshold = 0.80  # Baseline (original)
```

---

## **Step 3: Analyze & Compare**

### 3.1 Run Analysis Script
```bash
cd /mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/StrategyLab-master/analysis/strategy_optimization

python scripts/stage6_analyze_real_backtests.py \
  --baseline-dir ../../outputs/20251005_153045/mse_backtesting/2022-01-01_to_2025-08-31 \
  --optimal-dir ../../outputs/20251005_163012/mse_backtesting/2022-01-01_to_2025-08-31 \
  --test-start 2024-07-01 \
  --test-end 2025-08-31
```

**Replace the timestamps with YOUR actual output directories!**

---

## **Expected Output**

The script will:
1. Load all `data/strategy_trades/*.csv` files from both directories
2. Filter trades to test period (2024-07-01+)
3. Calculate performance metrics for both thresholds
4. Compare and make final decision

**Decision Criteria:**
- ✅ **IMPLEMENT** if:
  - 95% beats 80% on all metrics (WR, PF, Sharpe)
  - AND 95% meets targets (WR≥52%, PF≥1.25, Sharpe≥1.5)

- ❌ **REJECT** if either condition fails

---

## **Files Created**

After analysis completes:
```
analysis/strategy_optimization/checkpoints/
└── stage6_final_decision.csv  (contains decision + all metrics)
```

---

## **Timeline**

- **Step 1 (Baseline backtest)**: ~30-45 minutes
- **Step 2 (Optimal backtest)**: ~30-45 minutes
- **Step 3 (Analysis)**: ~2-3 minutes

**Total**: ~1-1.5 hours

---

## **What to Do After**

### If **✅ IMPLEMENT** Decision:
1. Keep `self.exit_threshold = 0.95` in code
2. Deploy to paper trading
3. Monitor for 1 week
4. If paper trading confirms → production

### If **❌ REJECT** Decision:
1. Keep `self.exit_threshold = 0.80` (original)
2. Review Stage 6 results to understand why
3. Options:
   - Accept 80% as optimal
   - Re-run Stage 2 with different threshold range
   - Investigate regime-specific thresholds

---

## **Quick Reference**

**File to modify**: `src/strategies/mse_strategy_backtesting.py` (Line 80)

**Backtest command**:
```bash
python src/runners/unified_runner.py --mode backtest --template conservative --dates 2022-01-01 --end-date 2025-08-31 --tickers <24 tickers>
```

**Analysis command**:
```bash
python scripts/stage6_analyze_real_backtests.py --baseline-dir <path1> --optimal-dir <path2>
```

---

**Ready to execute? Start with Step 1!**
