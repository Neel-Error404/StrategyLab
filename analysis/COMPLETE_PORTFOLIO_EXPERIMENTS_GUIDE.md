# Complete Portfolio Experiments Guide
## Run ALL Selection Methods on Full 96-Ticker Universe

**Your existing portfolio_experiments framework has ALL methods ready!**

---

## 🎯 AVAILABLE SELECTION METHODS (8 Total)

### **1. Random Selection** (Baseline)
- Random ticker sampling
- Performance benchmark

### **2. Greedy Forward Selection**
- Iterative best-ticker addition
- Fast and effective

### **3. k-DPP Sampling** (Drug Discovery)
- Determinantal Point Process
- Maximizes diversity

### **4. SPARROW Clustering** (Drug Discovery)
- Hierarchical clustering
- Quality + diversity balance

### **5. ACO-SA Hybrid** (Metaheuristics)
- Ant Colony Optimization + Simulated Annealing
- Escapes local optima

### **6. PSO** (Swarm Intelligence)
- Particle Swarm Optimization
- Bio-inspired search

### **7. Bayesian Optimization** (Statistical)
- Gaussian Process
- Intelligent exploration

### **8. Three-Stage Funnel** (Materials Science)
- Multi-stage filtering
- Quality gates approach

---

## 🚀 RECOMMENDED EXECUTION SEQUENCE

### **Option 1: Run Individual Methods (Parallel if possible)**

```powershell
cd analysis\portfolio_experiments

# Method 1: Greedy (FASTEST - 30-60 min)
py src\runners\RUN_SELECTION_02_GREEDY.py

# Method 2: SPARROW (60-90 min)
py src\runners\RUN_SELECTION_04_SPARROW.py

# Method 3: ACO-SA (90-120 min)
py src\runners\RUN_SELECTION_05_ACOSA.py

# Method 4: PSO (60-90 min)
py src\runners\RUN_SELECTION_06_PSO.py

# Method 5: Bayesian (90-120 min)
py src\runners\RUN_SELECTION_07_BAYESIAN.py
```

**Total Time (Sequential):** ~5-8 hours
**Total Time (If run in parallel on different terminals):** ~2 hours

---

### **Option 2: Use Universal Pipeline (ALL METHODS)**

```powershell
cd analysis\portfolio_experiments

# Check if config exists
cat config\portfolio_pipeline.yaml

# Run complete pipeline (ALL 8 methods + ALL 17 weightings + rebalancing)
py src\runners\RUN_UNIVERSAL_PIPELINE.py --config config\portfolio_pipeline.yaml
```

**What it does:**
- Runs ALL 8 selection methods
- Tests multiple portfolio sizes
- Applies all 17 weighting schemes
- Tests 8 rebalancing frequencies
- Walk-forward validation
- Complete comparison report

**Total Time:** 6-12 hours (comprehensive!)

---

### **Option 3: Use Comprehensive Pipeline**

```powershell
cd analysis\portfolio_experiments

py src\runners\RUN_COMPREHENSIVE_PIPELINE.py
```

**What it does:**
- Phase 1: Position sizing on top 30 tickers
- Phase 2: Scale to full dataset with selection methods
- Phase 3: Generate comparison report

**Total Time:** 30-45 minutes (focused!)

---

## 📊 EXPECTED OUTPUTS

### **Individual Method Outputs:**

```
analysis/portfolio_experiments/results/
├── selection_02_greedy/
│   ├── greedy_results.csv
│   ├── greedy_portfolios.json
│   └── greedy_summary.md
├── selection_04_sparrow/
│   ├── sparrow_results.csv
│   └── sparrow_summary.md
├── selection_05_acosa/
│   ├── acosa_results.csv
│   └── acosa_summary.md
├── selection_06_pso/
│   ├── pso_results.csv
│   └── pso_summary.md
└── selection_07_bayesian/
    ├── bayesian_results.csv
    └── bayesian_summary.md
```

### **Universal Pipeline Outputs:**

```
analysis/portfolio_experiments/outputs/universal_pipeline/
├── selection_results/
│   ├── random_results.csv
│   ├── greedy_results.csv
│   ├── kdpp_results.csv
│   ├── sparrow_results.csv
│   ├── acosa_results.csv
│   ├── pso_results.csv
│   ├── bayesian_results.csv
│   └── funnel_results.csv
├── position_sizing_results/
│   ├── equal_weight.csv
│   ├── risk_parity.csv
│   ├── hrp_*.csv (multiple variants)
│   └── ... (17 total)
├── rebalancing_results/
│   └── [results by frequency]
└── master_comparison.csv  ← FINAL RESULTS
```

---

## 🎯 QUICK START (Fastest Results)

### **For Quick Validation (1 hour):**

```powershell
cd analysis\portfolio_experiments
py src\runners\RUN_SELECTION_02_GREEDY.py
```

This gives you immediate results with Greedy Forward selection.

---

### **For Comprehensive Analysis (Overnight):**

```powershell
cd analysis\portfolio_experiments
py src\runners\RUN_UNIVERSAL_PIPELINE.py --config config\portfolio_pipeline.yaml
```

Wake up to complete results across ALL methods!

---

### **For Fast Multi-Method Comparison (3-4 hours):**

Run these in **separate terminal windows** (parallel execution):

**Terminal 1:**
```powershell
py src\runners\RUN_SELECTION_02_GREEDY.py
```

**Terminal 2:**
```powershell
py src\runners\RUN_SELECTION_04_SPARROW.py
```

**Terminal 3:**
```powershell
py src\runners\RUN_SELECTION_06_PSO.py
```

**Terminal 4:**
```powershell
py src\runners\RUN_SELECTION_07_BAYESIAN.py
```

---

## 📋 CONFIGURATION CUSTOMIZATION

If you want to modify portfolio sizes, edit the scripts or check for config files:

```powershell
# Check for configuration files
ls analysis\portfolio_experiments\config\
```

Most scripts have hardcoded parameters, but you can edit them:

```python
# In RUN_SELECTION_02_GREEDY.py (for example)
PORTFOLIO_SIZES = [5, 10, 15, 20, 30, 50]  # Edit this line
```

---

## 🔍 COMPARISON FRAMEWORK

After running methods, compare:

| Method | Sharpe | Return | Max DD | Tickers | Weighting | Runtime |
|--------|--------|--------|--------|---------|-----------|---------|
| Random | ? | ? | ? | ? | ? | ? |
| Greedy | ? | ? | ? | ? | ? | ? |
| k-DPP | ? | ? | ? | ? | ? | ? |
| SPARROW | ? | ? | ? | ? | ? | ? |
| ACO-SA | ? | ? | ? | ? | ? | ? |
| PSO | ? | ? | ? | ? | ? | ? |
| Bayesian | ? | ? | ? | ? | ? | ? |
| Funnel | ? | ? | ? | ? | ? | ? |

---

## 💡 RECOMMENDED WORKFLOW

### **Day 1: Quick Methods (Parallel)**
Run Greedy + SPARROW + PSO in parallel (3-4 hours total)

### **Day 2: Advanced Methods**
Run ACO-SA + Bayesian overnight (8 hours)

### **Day 3: Analysis**
Compare all results, identify best method + size

### **Day 4: Walk-Forward Validation**
Validate winner with walk-forward testing

---

## ✅ SUCCESS CRITERIA

After completion, you should know:

- ✅ Which selection method finds best portfolios?
- ✅ Which portfolio size is optimal?
- ✅ Which tickers are consistently selected?
- ✅ Which weighting scheme works best?
- ✅ What are expected returns/drawdowns?
- ✅ How stable are results across methods?

---

## 🚀 YOUR FIRST COMMAND

```powershell
cd analysis\portfolio_experiments
py src\runners\RUN_SELECTION_02_GREEDY.py
```

Start with Greedy to get fast baseline results!

---

## 📞 ALTERNATIVE: Check Existing Results

You might already have results! Check:

```powershell
ls analysis\portfolio_experiments\outputs\
ls analysis\portfolio_experiments\results\
cat analysis\portfolio_experiments\outputs\MASTER_COMPARISON.csv
```

Your `MASTER_COMPARISON.csv` (from your paste) shows you already ran some methods with excellent results (Sharpe 16-18)!
