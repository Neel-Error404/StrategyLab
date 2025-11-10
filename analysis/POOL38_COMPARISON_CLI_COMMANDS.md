# Pool38 Portfolio Comparison - CLI Commands
## Step-by-Step Execution Guide

**Date:** 2025-11-08
**Objective:** Compare portfolio_construction vs portfolio_experiments on same 38 tickers

---

## 🚀 QUICK START (Copy-Paste These Commands)

### **STEP 1: Portfolio Construction (Bug-Fixed) - 5 minutes**

```powershell
# Run from: D:\Balcony\Trading\unified_trading_setup\backtester
py analysis/run.py --config analysis/configs/mse_pool38_portfolio_base.yaml --targets portfolio
```

**Expected Output:**
- `analysis/output/mse_strategy_backtesting/pool38_base/portfolio/portfolio_optimizer/portfolio_performance_top50.csv`
- Sharpe ratios should be 16-18 range (FIXED!)

---

### **STEP 2: Portfolio Experiments - Greedy - 10 minutes**

```powershell
# Run from: D:\Balcony\Trading\unified_trading_setup\backtester\analysis\portfolio_experiments
py RUN_POOL38_COMPARISON.py --method greedy
```

**Expected Output:**
- `outputs/pool38_comparison/greedy/greedy_results.csv`
- `outputs/pool38_comparison/greedy/greedy_summary.md`

---

### **STEP 3: Portfolio Experiments - Bayesian - 15 minutes**

```powershell
# Run from: D:\Balcony\Trading\unified_trading_setup\backtester\analysis\portfolio_experiments
py RUN_POOL38_COMPARISON.py --method bayesian
```

**Expected Output:**
- `outputs/pool38_comparison/bayesian/bayesian_results.csv`
- `outputs/pool38_comparison/bayesian/bayesian_summary.md`

---

### **STEP 4: Portfolio Experiments - ACO-SA (Optional) - 20 minutes**

```powershell
# Run from: D:\Balcony\Trading\unified_trading_setup\backtester\analysis\portfolio_experiments
py RUN_POOL38_COMPARISON.py --method acosa
```

**Expected Output:**
- `outputs/pool38_comparison/acosa/acosa_results.csv`
- `outputs/pool38_comparison/acosa/acosa_summary.md`

---

### **STEP 5: Run All Methods (One Command) - 45 minutes**

```powershell
# Run from: D:\Balcony\Trading\unified_trading_setup\backtester\analysis\portfolio_experiments
py RUN_POOL38_COMPARISON.py --method all
```

**Expected Output:**
- All individual results plus
- `outputs/pool38_comparison/pool38_master_comparison.csv` (combined results)

---

## 📊 CHECK RESULTS

### **Portfolio Construction Results:**

```powershell
# View top 10 portfolios
Get-Content analysis\output\mse_strategy_backtesting\pool38_base\portfolio\portfolio_optimizer\portfolio_performance_top50.csv | Select-Object -First 11

# View PyPfOpt summary (6-ticker)
Get-Content analysis\output\mse_strategy_backtesting\pool38_base\portfolio\pypfopt_weights\pypfopt_summary_6ticker.md
```

### **Portfolio Experiments Results:**

```powershell
# View Greedy results
cd analysis\portfolio_experiments
Get-Content outputs\pool38_comparison\greedy\greedy_summary.md

# View Bayesian results
Get-Content outputs\pool38_comparison\bayesian\bayesian_summary.md

# View master comparison
Get-Content outputs\pool38_comparison\pool38_master_comparison.csv
```

---

## 🎯 EXPECTED RESULTS

**All methods should produce:**
- ✅ Sharpe ratios: **16-18 range** (after bug fixes)
- ✅ Annual returns: **45-55%**
- ✅ Max drawdowns: **15-30%**
- ✅ Similar top ticker selections

**If Sharpe ratios are still 1.3-1.4:**
- ❌ Bug fix didn't apply - check that you're running the fixed scripts
- ❌ Data file doesn't have 'Profit (%)' column

---

## 🔍 TROUBLESHOOTING

### **Error: Missing 'Profit (%)' column**

```python
# Check if column exists
import pandas as pd
df = pd.read_csv('analysis/output/mse_strategy_backtesting/pool38_base/data/pool38_trades_merged.csv', nrows=5)
print(df.columns.tolist())
```

### **Error: Import failed in RUN_POOL38_COMPARISON.py**

```powershell
# Check if methods exist
cd analysis\portfolio_experiments
dir methods\baseline\greedy_forward_selection.py
dir methods\statistical\bayesian_optimization.py
dir methods\metaheuristics\aco_sa_selection.py
```

### **Portfolio Construction takes 3 hours**

The ticker_ranking module is still enabled. Disable it:
```yaml
# In analysis/configs/mse_pool38_portfolio_base.yaml
ticker_ranking:
  enabled: false  # Change to false
```

---

## 📈 COMPARISON CHECKLIST

After all runs complete, compare:

- [ ] Best Sharpe ratio (5-ticker): Construction vs Greedy vs Bayesian vs ACO-SA
- [ ] Best Sharpe ratio (6-ticker): Construction vs Greedy vs Bayesian vs ACO-SA
- [ ] Best Sharpe ratio (7-ticker): Construction vs Greedy vs Bayesian vs ACO-SA
- [ ] Ticker overlap: Are top portfolios selecting similar tickers?
- [ ] Runtime efficiency: Which method is fastest?
- [ ] Consistency: Do all methods agree on top 10-15 tickers?

---

## 💡 NEXT STEPS

1. **Analyze results**: Which method found best portfolios?
2. **Validate tickers**: Do top portfolios make sense fundamentally?
3. **Walk-forward test**: Use portfolio_experiments walk-forward validator
4. **Production decision**: Choose method for live trading

---

## 📝 NOTES

**Portfolio Construction Strengths:**
- Exhaustive search (guaranteed optimal given constraints)
- Sector diversification built-in
- Multiple weighting schemes via PyPfOpt
- Fast for small-medium ticker pools

**Portfolio Experiments Strengths:**
- Smart algorithms (faster for large pools)
- Multiple methods to compare
- Walk-forward validation built-in
- Ensemble methods available

**Recommended:**
- Use **Portfolio Construction** for final portfolio selection (exhaustive)
- Use **Portfolio Experiments** for initial ticker screening and validation
