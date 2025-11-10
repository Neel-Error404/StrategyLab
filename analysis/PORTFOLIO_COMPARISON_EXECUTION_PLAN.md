# Portfolio Comparison Execution Plan
## Comparing portfolio_construction vs portfolio_experiments on Same 38 Tickers

**Date:** 2025-11-08
**Objective:** Run both portfolio systems on identical 38-ticker pool and compare results

---

## 📋 COMMON INPUT

**38 Tickers (≤₹2,000 price constraint):**
```
AXISBANK, MAXHEALTH, INDHOTEL, POCL, NMDC, PNB, UBL, HERCULES, ITC, PSUBANK,
NTPC, PIDILITIND, DELHIVERY, JINDALSTEL, FEDERALBNK, SUNPHARMA, IPCALAB, ABFRL,
INFY, POWERGRID, IRFC, DABUR, PETRONET, VIMTALABS, TATASTEEL, MSUMI, RAMCOCEM,
GRAPHITE, TATAMOTORS, PFC, SBICARD, NESTLEIND, WIPRO, VGUARD, TATACHEM, TECHM,
SHYAMMETL, CIPLA
```

**Base Trades File:** `analysis/output/mse_strategy_backtesting/pool38_base/data/pool38_trades_merged.csv`
- Total Trades: 347,867 (all 96 tickers)
- Will be filtered to 38 tickers by both systems

**Ticker List File:** `analysis/output/mse_strategy_backtesting/20251107_154802_full/ticker_pool_selection/ticker_list_38.txt`

---

## 🔄 EXECUTION SEQUENCE

### **STEP 1: Portfolio Construction (Bug-Fixed Version)**

**What it does:**
- Takes 38 tickers as pre-filtered input (via config)
- Generates all valid portfolio combinations (5, 6, 7 tickers)
- Applies sector concentration (max 60%) and correlation (max 0.75) filters
- Ranks top 50 portfolios by Sharpe ratio
- Calculates PyPfOpt optimal weights (Equal Weight, Min Vol, etc.)
- Generates equity curves and visualizations

**Runtime:** ~5 minutes (skipping ticker_ranking)

**CLI Command:**
```powershell
py analysis/run.py --config analysis/configs/mse_pool38_portfolio_base.yaml --targets portfolio
```

**Outputs:**
```
analysis/output/mse_strategy_backtesting/pool38_base/portfolio/
├── anti_cascade_filter/
│   └── anti_cascading_trades_filtered.csv (38 tickers only)
├── sector_classification/
│   ├── sector_mapping.csv
│   └── correlation_matrix.csv
├── combination_generator/
│   ├── valid_combinations_5ticker.csv
│   ├── valid_combinations_6ticker.csv
│   └── valid_combinations_7ticker.csv
├── portfolio_optimizer/
│   ├── portfolio_performance_top50.csv  ← KEY RESULTS
│   └── portfolio_performance_all.csv
├── pypfopt_weights/
│   ├── optimal_weights_5ticker.csv
│   ├── optimal_weights_6ticker.csv
│   ├── optimal_weights_7ticker.csv
│   ├── pypfopt_summary_5ticker.md  ← KEY RESULTS
│   ├── pypfopt_summary_6ticker.md
│   └── pypfopt_summary_7ticker.md
└── equity_curves/
    ├── portfolio_summary_stats.csv  ← KEY RESULTS
    └── [various visualization PNGs]
```

---

### **STEP 2: Portfolio Experiments - Greedy Forward Selection**

**What it does:**
- Takes 38 tickers as input ticker pool
- Uses Greedy Forward Selection algorithm to build portfolios
- Tests portfolio sizes: 5, 6, 7 tickers (configurable)
- Calculates multiple weighting schemes (Equal Weight, HRP, Risk Parity, Min Vol, etc.)
- Returns best portfolio per size with optimal weighting method

**Runtime:** ~5-10 minutes

**CLI Command:**
```powershell
cd analysis/portfolio_experiments

py src/runners/RUN_SELECTION_02_GREEDY.py `
  --data-file ../output/mse_strategy_backtesting/pool38_base/data/pool38_trades_merged.csv `
  --tickers-file ../output/mse_strategy_backtesting/20251107_154802_full/ticker_pool_selection/ticker_list_38.txt `
  --portfolio-sizes 5,6,7 `
  --output-dir outputs/pool38_comparison/greedy
```

**Outputs:**
```
analysis/portfolio_experiments/outputs/pool38_comparison/greedy/
├── greedy_forward_results.csv  ← KEY RESULTS
├── greedy_forward_summary.md
└── greedy_forward_portfolios.json
```

---

### **STEP 3: Portfolio Experiments - Bayesian Optimization**

**What it does:**
- Uses Bayesian Optimization with Gaussian Process
- Intelligent exploration of portfolio space
- Often finds better Sharpe ratios than greedy methods
- Tests same portfolio sizes and weighting schemes

**Runtime:** ~10-15 minutes

**CLI Command:**
```powershell
cd analysis/portfolio_experiments

py src/runners/RUN_SELECTION_07_BAYESIAN.py `
  --data-file ../output/mse_strategy_backtesting/pool38_base/data/pool38_trades_merged.csv `
  --tickers-file ../output/mse_strategy_backtesting/20251107_154802_full/ticker_pool_selection/ticker_list_38.txt `
  --portfolio-sizes 5,6,7 `
  --output-dir outputs/pool38_comparison/bayesian
```

**Outputs:**
```
analysis/portfolio_experiments/outputs/pool38_comparison/bayesian/
├── bayesian_results.csv  ← KEY RESULTS
├── bayesian_summary.md
└── bayesian_portfolios.json
```

---

### **STEP 4: Portfolio Experiments - ACO-SA (Hybrid Metaheuristic)**

**What it does:**
- Combines Ant Colony Optimization + Simulated Annealing
- Good at escaping local optima
- Often competitive with Bayesian for complex portfolios

**Runtime:** ~15-20 minutes

**CLI Command:**
```powershell
cd analysis/portfolio_experiments

py src/runners/RUN_SELECTION_05_ACOSA.py `
  --data-file ../output/mse_strategy_backtesting/pool38_base/data/pool38_trades_merged.csv `
  --tickers-file ../output/mse_strategy_backtesting/20251107_154802_full/ticker_pool_selection/ticker_list_38.txt `
  --portfolio-sizes 5,6,7 `
  --output-dir outputs/pool38_comparison/acosa
```

**Outputs:**
```
analysis/portfolio_experiments/outputs/pool38_comparison/acosa/
├── acosa_results.csv  ← KEY RESULTS
├── acosa_summary.md
└── acosa_portfolios.json
```

---

## 📊 COMPARISON METRICS

After all runs complete, compare:

| Metric | Portfolio Construction | Greedy | Bayesian | ACO-SA |
|--------|------------------------|--------|----------|--------|
| **Best Sharpe (5-ticker)** | ? | ? | ? | ? |
| **Best Sharpe (6-ticker)** | ? | ? | ? | ? |
| **Best Sharpe (7-ticker)** | ? | ? | ? | ? |
| **Best Overall Sharpe** | ? | ? | ? | ? |
| **Annual Return (%)** | ? | ? | ? | ? |
| **Max Drawdown (%)** | ? | ? | ? | ? |
| **Total Runtime** | 5 min | 10 min | 15 min | 20 min |
| **# Portfolios Evaluated** | ~Thousands | ~Hundreds | ~50-100 | ~Hundreds |

---

## ⚡ QUICK START (Recommended Execution Order)

**For fastest comparison, run these in sequence:**

```powershell
# 1. Portfolio Construction (5 min)
py analysis/run.py --config analysis/configs/mse_pool38_portfolio_base.yaml --targets portfolio

# 2. Wait for completion, then check results:
cat analysis/output/mse_strategy_backtesting/pool38_base/portfolio/portfolio_optimizer/portfolio_performance_top50.csv | Select-Object -First 5

# 3. Greedy Forward (10 min)
cd analysis/portfolio_experiments
py src/runners/RUN_SELECTION_02_GREEDY.py --data-file ../output/mse_strategy_backtesting/pool38_base/data/pool38_trades_merged.csv --tickers-file ../output/mse_strategy_backtesting/20251107_154802_full/ticker_pool_selection/ticker_list_38.txt --portfolio-sizes 5,6,7 --output-dir outputs/pool38_comparison/greedy

# 4. Bayesian (15 min)
py src/runners/RUN_SELECTION_07_BAYESIAN.py --data-file ../output/mse_strategy_backtesting/pool38_base/data/pool38_trades_merged.csv --tickers-file ../output/mse_strategy_backtesting/20251107_154802_full/ticker_pool_selection/ticker_list_38.txt --portfolio-sizes 5,6,7 --output-dir outputs/pool38_comparison/bayesian
```

**Total Time: ~30 minutes for all 3 methods**

---

## 🎯 EXPECTED OUTCOMES

**Portfolio Construction:**
- ✅ Exhaustive enumeration → finds ALL valid combinations
- ✅ Guarantees best portfolio given constraints
- ✅ Slower for large ticker pools (but fast for 38)
- ✅ Multiple weighting schemes via PyPfOpt

**Portfolio Experiments (Algorithms):**
- ✅ Smart sampling → faster for large spaces
- ✅ May miss some combinations but finds high-quality solutions
- ✅ Multiple algorithmic approaches to compare
- ✅ Ensemble methods available

**Likely Result:**
- Portfolio Construction should find slightly better portfolios (exhaustive search)
- Bayesian/ACO-SA should get very close (95-99% of optimal)
- All should have Sharpe ratios in 16-18 range (after bug fixes)

---

## 🚨 TROUBLESHOOTING

**If portfolio_experiments scripts don't accept the arguments:**

Check if they need modification to accept custom data/ticker files. Let me know and I'll adapt them.

**If runtime is too long:**

Skip ACO-SA (slowest) and just compare Construction vs Bayesian (fastest smart algorithm).

**If Sharpe ratios still look wrong:**

Double-check that portfolio_experiments scripts are using 'Profit (%)' column correctly (we fixed portfolio_construction but not portfolio_experiments).

---

## 📝 NEXT STEPS AFTER COMPARISON

1. **Analyze which method performed best**
2. **Check if tickers in top portfolios are similar across methods**
3. **Validate walk-forward if needed** (portfolio_experiments has this built-in)
4. **Generate final recommendations** for production use

---

**Ready to execute? Start with Portfolio Construction (Step 1) and let me know the results!**
