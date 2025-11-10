# Full Algorithmic Portfolio Selection
## Let Algorithms Decide Everything (No Human Pre-filtering)

**Philosophy:** Pure algorithmic approach - start with ALL 96 tickers, let algorithms choose the best.

---

## 🎯 THE RIGHT APPROACH

### **Input:**
- ✅ ALL 96 tickers from MSE strategy backtests
- ✅ Complete trade history (347,867 trades)
- ✅ NO pre-filtering by price or any other criteria

### **Selection:**
- ✅ Algorithms choose best tickers (Greedy, Bayesian, ACO-SA, PSO, etc.)
- ✅ Test multiple portfolio sizes (5, 6, 7, 10, 15, 20, 30, 50)
- ✅ NO human intervention in ticker selection

### **Weighting:**
- ✅ Test 17 different weighting schemes per portfolio
- ✅ Equal Weight, HRP, Risk Parity, Min Volatility, etc.
- ✅ Algorithm selects best weighting method

### **Walk-Forward:**
- ✅ Train on historical periods
- ✅ Test on out-of-sample periods
- ✅ Rebalancing studies built-in

---

## 🚀 EXECUTION COMMANDS

### **Method 1: Greedy Forward Selection** (~1-2 hours)

```powershell
cd analysis\portfolio_experiments

py RUN_NIFTY96_FULL_COMPARISON.py --method greedy --portfolio-sizes 5,6,7,10,15,20,30
```

**What it does:**
- Starts with 0 tickers
- Adds tickers one-by-one that maximize portfolio Sharpe
- Tests all 17 weighting schemes
- Returns best portfolio for each size

**Runtime:** ~1-2 hours for 7 sizes

---

### **Method 2: Bayesian Optimization** (~2-3 hours)

```powershell
py RUN_NIFTY96_FULL_COMPARISON.py --method bayesian --portfolio-sizes 5,6,7,10,15,20,30
```

**What it does:**
- Uses Gaussian Process to model Sharpe landscape
- Intelligently explores ticker combinations
- Balances exploration vs exploitation
- Often finds better portfolios than Greedy

**Runtime:** ~2-3 hours for 7 sizes

---

### **Method 3: ACO-SA Hybrid** (~3-4 hours)

```powershell
py RUN_NIFTY96_FULL_COMPARISON.py --method acosa --portfolio-sizes 5,6,7,10,15,20,30
```

**What it does:**
- Ant Colony Optimization + Simulated Annealing
- Good at escaping local optima
- Robust across different market conditions

**Runtime:** ~3-4 hours for 7 sizes

---

### **Method 4: Run ALL Methods** (~6-9 hours)

```powershell
py RUN_NIFTY96_FULL_COMPARISON.py --method all --portfolio-sizes 5,6,7,10,15,20,30,50
```

**What it does:**
- Runs all 3 methods sequentially
- Tests 8 portfolio sizes
- Creates master comparison across all methods
- Identifies best method for each size

**Runtime:** ~6-9 hours for complete analysis

---

## 📊 EXPECTED RESULTS

**Output Structure:**
```
analysis/portfolio_experiments/outputs/nifty96_full_comparison/
├── greedy/
│   ├── greedy_results.csv
│   └── greedy_summary.md
├── bayesian/
│   ├── bayesian_results.csv
│   └── bayesian_summary.md
├── acosa/
│   ├── acosa_results.csv
│   └── acosa_summary.md
└── nifty96_master_comparison.csv  ← Master results across all methods
```

**Expected Performance (based on previous runs):**
- **Sharpe Ratios:** 16-18 for best portfolios
- **Annual Returns:** 45-55%
- **Max Drawdowns:** 15-30%
- **Best Sizes:** Usually 20-30 tickers optimal (balance of diversification vs concentration)

---

## 🎯 RECOMMENDED WORKFLOW

### **Day 1: Quick Validation (3 hours)**

```powershell
# Test 3 key sizes with fastest method
py RUN_NIFTY96_FULL_COMPARISON.py --method greedy --portfolio-sizes 5,10,20
```

**Why:** Validates setup and gets initial results quickly

---

### **Day 2: Comprehensive Analysis (overnight run)**

```powershell
# Run all methods, all sizes
py RUN_NIFTY96_FULL_COMPARISON.py --method all --portfolio-sizes 5,6,7,10,15,20,30,50
```

**Why:** Complete comparison to identify best approach

---

### **Day 3: Walk-Forward & Rebalancing** (if needed)

Once you identify best method + size, run walk-forward validation:

```powershell
# Use existing portfolio_experiments walk-forward scripts
py src/runners/RUN_WALK_FORWARD.py --method bayesian --size 20
```

---

## 📈 INTERPRETATION GUIDE

### **Comparing Methods:**

| Metric | What It Tells You |
|--------|------------------|
| **Sharpe Ratio** | Risk-adjusted return (higher = better) |
| **Annual Return** | Absolute performance |
| **Max Drawdown** | Worst peak-to-trough decline (lower = better) |
| **Tickers Selected** | Which stocks algorithms prefer |
| **Best Weighting** | Which weighting scheme works best |

### **Optimal Portfolio Size:**

- **5-7 tickers:** High concentration, high risk/reward
- **10-15 tickers:** Good balance for most strategies
- **20-30 tickers:** Maximum diversification, lower drawdowns
- **50+ tickers:** Index-like behavior, lowest risk

---

## 🔄 ITERATIVE REFINEMENT

After initial runs:

1. **Analyze ticker overlap:** Which tickers appear across methods?
2. **Sector analysis:** Are selected tickers diversified?
3. **Time period stability:** Do results hold across sub-periods?
4. **Rebalancing impact:** Test monthly/quarterly rebalancing

---

## 🆚 COMPARISON WITH PORTFOLIO CONSTRUCTION

**Portfolio Construction (Your 38-ticker pre-filtered):**
- Human judgment: Pre-selected tickers ≤₹2,000
- Exhaustive enumeration of pre-filtered set
- Good for: Validation, sector constraints

**Portfolio Experiments (Pure Algorithmic):**
- Zero human judgment: Algorithms select from all 96
- Smart sampling of large search space
- Good for: Production, discovering unexpected winners

**Use Case:**
- **Primary:** Portfolio Experiments (algorithmic)
- **Secondary:** Portfolio Construction (validation)
- **Best:** Run both and compare results!

---

## ⚠️ IMPORTANT NOTES

1. **Runtime:** Be prepared for long runtimes (hours, not minutes)
2. **Memory:** Ensure sufficient RAM (8GB+ recommended)
3. **Data Quality:** Verify 'Profit (%)' column exists in trades
4. **Patience:** Algorithmic selection is computational intensive but worth it

---

## ✅ SUCCESS CRITERIA

After running experiments, you should be able to answer:

- ✅ Which portfolio size gives best risk-adjusted returns?
- ✅ Which selection method performs best?
- ✅ Which tickers are consistently selected across methods?
- ✅ Which weighting scheme works best?
- ✅ What's the expected Sharpe/Return/Drawdown?

---

**Ready to start? Run the Greedy method first for fastest results!**

```powershell
cd analysis\portfolio_experiments
py RUN_NIFTY96_FULL_COMPARISON.py --method greedy --portfolio-sizes 5,10,20
```
