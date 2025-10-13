# Portfolio Construction System - Complete End-to-End Map

**Last Updated**: 2025-10-04
**Status**: Phase 1 Complete (Directory cleanup, PyPortfolioOpt integration)

---

## 📊 SYSTEM OVERVIEW

**Purpose**: Identify optimal stock portfolios from 1.15M historical trades using percentage-based returns, sector diversification, and correlation filtering.

**Key Achievement**: Identified **6-7 ticker portfolios** as optimal (Sharpe 1.76-1.81)

**Architecture**: Sequential pipeline with 6 scripts (00→05) + Master Optimizer

---

## 🔄 COMPLETE DATA FLOW

### INPUT (Script 0)
- **Source**: `all_trade_merged.csv` (1.15M trades, 2022-2025)
- **Columns Used**:
  - `ticker` - Stock symbol
  - `Entry Price`, `Exit Price` - Trade execution prices
  - `Entry Time`, `Exit Time` - Trade timestamps
  - `Trade Type` - Buy/Sell direction
  - `trade_category` - Cascade classification

### SCRIPT 0: Foundation Analysis
**File**: `scripts/00_foundation_cascade_vs_anticascade_analysis.py`

**Purpose**: Rank tickers by performance using percentage returns (not absolute currency)

**Filters & Logic**:
```python
# 1. Split trades by cascade type
cascading_trades = df[df['trade_category'] == 'CONSECUTIVE_SAME_DIRECTION']
anti_cascading_trades = df[df['trade_category'] == 'CONSECUTIVE_OPPOSITE_DIRECTION']

# 2. Calculate percentage returns (KEY CHANGE from absolute)
trades['percentage_return'] = ((Exit_Price / Entry_Price - 1) * 100)

# 3. Calculate metrics PER TICKER
for ticker:
    winning_trades = trades[percentage_return > 0]
    losing_trades = trades[percentage_return <= 0]

    profit_factor = abs(winning_trades.sum() / losing_trades.sum())
    win_rate = (winning_trades.count() / total_trades) * 100
    avg_return_pct = trades['percentage_return'].mean()

    # No Sharpe ratio at individual level anymore (removed)

# 4. Weighted scoring (HARDCODED WEIGHTS)
weights = {
    'profit_factor_score': 0.40,    # 40% weight
    'win_rate_score': 0.30,         # 30% weight
    'avg_return_pct_score': 0.30    # 30% weight
}

# 5. Top 50 selection
top_50 = tickers.nlargest(50, 'composite_score')
```

**Hardcoded Parameters**:
- ✅ `TOP_N = 50` - Number of top performers
- ✅ Scoring weights (PF: 0.40, WR: 0.30, AvgRet: 0.30)
- ✅ Trade category filtering (anti-cascading only for final selection)

**Outputs**:
- `data/foundation/TOP50_ANTICASCADING_TRADES.csv` (50 tickers)
- `data/foundation/top50_ANTICASCADING_TRADES_performance.csv` (metrics)

**Metrics Example** (Top 5):
```
Ticker      PF      WR      Avg Ret%   Score
KOTAKBANK   1.23    52.1%   0.089%     0.845
AXISBANK    1.21    51.8%   0.087%     0.842
HCG         1.19    50.3%   0.091%     0.838
```

---

### SCRIPT 1: Affordability Filter
**File**: `scripts/01_corrected_anti_cascading_subset.py`

**Purpose**: Filter to affordable tickers (< ₹2,000 entry price) for retail traders

**Filter Logic**:
```python
# HARDCODED THRESHOLD
PRICE_THRESHOLD = 2000  # Rupees

affordable_tickers = top_50[top_50['Entry Price'].mean() < PRICE_THRESHOLD]
```

**Hardcoded Parameters**:
- ✅ `PRICE_THRESHOLD = 2000` - Max entry price

**Outputs**:
- `data/filtered/CORRECTED_affordable_tickers_metadata.csv` (**28 tickers**)
- `data/filtered/CORRECTED_anti_cascading_trades_under2k.csv` (39,221 trades)

**Reduction**: 50 tickers → **28 affordable tickers**

---

### SCRIPT 2: Sector Classification & Correlation
**File**: `scripts/02_corrected_sector_classification_correlation.py`

**Purpose**: Map tickers to sectors, calculate pairwise correlations for diversification

**Logic**:
```python
# 1. Manual sector mapping (HARDCODED)
sector_mapping = {
    'KOTAKBANK': 'Banking & Financial Services',
    'AXISBANK': 'Banking & Financial Services',
    'HCG': 'Pharmaceuticals & Healthcare',
    'KAJARIACER': 'Chemicals & Materials',
    # ... 28 tickers total
}

# 2. Calculate daily returns per ticker
for ticker:
    daily_returns = trades.groupby('Entry Date')['percentage_return'].mean()

# 3. Create returns matrix & correlation
returns_matrix = pd.DataFrame({ticker: daily_returns for ticker in tickers})
correlation_matrix = returns_matrix.corr()  # Pearson correlation
```

**Hardcoded Parameters**:
- ✅ Sector mapping (28 ticker→sector pairs) - **UPDATE NEEDED if Top 50 changes**
- ✅ Correlation method: Pearson (could be Spearman/Kendall)

**Outputs**:
- `data/classified/CORRECTED_sector_mapping.csv` (28 tickers → 9 sectors)
- `data/classified/CORRECTED_correlation_matrix.csv` (28×28 matrix)
- `data/classified/CORRECTED_daily_returns_data.csv` (time series)

**Sector Distribution**:
```
Banking & Financial Services: 6 tickers
Pharmaceuticals & Healthcare: 4 tickers
IT Services & Software: 3 tickers
... (9 sectors total)
```

---

### SCRIPT 3: Combination Generation
**File**: `scripts/03_corrected_intelligent_combination_generation.py`

**Purpose**: Generate all valid N-ticker portfolios with diversification filters

**Filter Logic**:
```python
# HARDCODED FILTERS
MAX_SECTOR_CONCENTRATION = 0.60  # Max 60% from one sector
MAX_CORRELATION = 0.70           # Max 0.70 pairwise correlation

# Generate all combinations
for combination in itertools.combinations(28_tickers, N):

    # Filter 1: Sector diversification
    sector_counts = count_sectors(combination)
    max_sector_pct = max(sector_counts) / N
    if max_sector_pct > 0.60:
        reject()

    # Filter 2: Correlation check
    for ticker1, ticker2 in itertools.combinations(combination, 2):
        correlation = corr_matrix[ticker1][ticker2]
        if abs(correlation) > 0.70:
            reject()

    # Accept if passes both filters
    valid_combinations.append(combination)
```

**Hardcoded Parameters**:
- ✅ `MAX_SECTOR_CONCENTRATION = 0.60` (60% limit)
- ✅ `MAX_CORRELATION = 0.70` (correlation threshold)
- ✅ `N_TICKERS` - Portfolio size (run separately for each N)

**NO ADDITIONAL FILTERING** - All 28 tickers already pre-selected as Top 50 performers

**Outputs** (per portfolio size):
- `data/combinations/PERCENTAGE_valid_combinations_4ticker.csv` (30,135 combos)
- `data/combinations/PERCENTAGE_valid_combinations_5ticker.csv` (97,712 combos)
- `data/combinations/PERCENTAGE_valid_combinations_6ticker.csv` (389,224 combos)
- `data/combinations/PERCENTAGE_valid_combinations_7ticker.csv` (1,184,040 combos)
- `data/combinations/PERCENTAGE_valid_combinations_8ticker.csv` (3,108,105 combos)

---

### SCRIPT 4: Portfolio Performance Calculation
**File**: `scripts/04_portfolio_optimization_engine.py`

**Purpose**: Calculate portfolio-level Sharpe ratio using equal-weight allocation

**Logic**:
```python
# For each portfolio combination:
for combination in valid_combinations:

    # 1. Get all trades for tickers in portfolio
    portfolio_trades = trades[trades['ticker'].isin(combination)]

    # 2. Calculate PORTFOLIO metrics (equal weight 1/N)
    portfolio_returns = portfolio_trades['percentage_return'].mean()
    portfolio_volatility = portfolio_trades['percentage_return'].std()

    # 3. Sharpe Ratio (HARDCODED RISK-FREE RATE)
    risk_free_rate = 0.0  # 0% assumed
    sharpe_ratio = (portfolio_returns - risk_free_rate) / portfolio_volatility

    # 4. Additional metrics
    profit_factor = abs(wins.sum() / losses.sum())
    win_rate = (wins.count() / total_trades) * 100
    max_drawdown = calculate_drawdown(cumulative_returns)

# 5. Rank by Sharpe, select Top 50
top_50_portfolios = all_portfolios.nlargest(50, 'sharpe_ratio')
```

**Hardcoded Parameters**:
- ✅ `RISK_FREE_RATE = 0.0` (should be ~6-7% for India)
- ✅ `N_PORTFOLIOS_TO_TEST = 10000` - Sample 10K random portfolios (for speed)
- ✅ `TOP_N_RESULTS = 50` - Keep top 50 per size
- ✅ **Equal-weight allocation** (1/N) - No optimization yet

**Outputs** (per portfolio size):
- `data/results/portfolio_performance_4ticker_ALL.csv` (all 10K tested)
- `data/results/portfolio_performance_4ticker_TOP50.csv` (top 50)
- Same for 5,6,7,8 ticker sizes

**Best Results**:
```
6-ticker: Sharpe 1.811, Avg Return 0.078%, Max DD -0.52% ✅ OPTIMAL
7-ticker: Sharpe 1.811, Avg Return 0.077%, Max DD -0.52%
```

---

### SCRIPT 5: PyPortfolioOpt Weight Optimization
**File**: `scripts/05_pypfopt_optimal_weights.py` (**NEW - Phase 1**)

**Purpose**: Try optimal weight allocation (replacing equal-weight 1/N)

**Logic**:
```python
# For each Top 50 portfolio:
for portfolio in top_50_portfolios:

    # 1. Load trade data for tickers
    trades = get_trades(portfolio.tickers)

    # 2. Create daily returns matrix (resampled from trades)
    returns_df = trades.groupby(['Entry Date', 'ticker'])['percentage_return'].mean()
    returns_matrix = returns_df.pivot(columns='ticker', values='percentage_return')

    # 3. Calculate expected returns & covariance
    mu = expected_returns.mean_historical_return(returns_matrix, frequency=252)
    S = risk_models.sample_cov(returns_matrix, frequency=252)

    # 4. Optimization attempts:

    # Method 1: Equal Weight (baseline) ✅ WORKS
    weights = {ticker: 1/N for ticker in tickers}

    # Method 2: Max Sharpe Ratio ❌ FAILED (non-convex covariance)
    try:
        ef = EfficientFrontier(mu, S)
        ef.add_objective(objective_functions.L2_reg, gamma=0.1)
        weights = ef.max_sharpe()
    except:
        fail()

    # Method 3: Min Volatility ❌ FAILED (non-convex covariance)
    try:
        ef = EfficientFrontier(mu, S)
        weights = ef.min_volatility()
    except:
        fail()
```

**Why Advanced Optimizations Failed**:
- Trade-level data → resampled to daily → sparse time series
- Sparse data → singular/near-singular covariance matrices
- Non-convex optimization problem → solver failure

**Hardcoded Parameters**:
- ✅ `FREQUENCY = 252` - Trading days per year (annualization)
- ✅ `L2_REG_GAMMA = 0.1` - Regularization strength (for Max Sharpe)
- ✅ `TARGET_VOLATILITY = 0.15` (15%) for Efficient Risk method

**Outputs**:
- `data/results/optimal_weights_6ticker.csv` (50 portfolios × weight allocations)
- `data/results/optimal_weights_7ticker.csv`

**Result**: Equal-weight remains best option (Sharpe 1.76 avg)

---

### MASTER OPTIMIZER
**File**: `scripts/master_portfolio_optimizer.py`

**Purpose**: Orchestrate Scripts 3-4 across multiple portfolio sizes in one run

**Logic**:
```python
# HARDCODED PORTFOLIO SIZES
portfolio_sizes = [4, 5, 6, 7, 8]

for size in portfolio_sizes:
    # Run Script 3: Generate combinations
    generate_combinations(size)

    # Run Script 4: Calculate performance
    calculate_performance(size)

# Generate comparison report
comparison_df = compare_all_sizes()
```

**Hardcoded Parameters**:
- ✅ `PORTFOLIO_SIZES = [4,5,6,7,8]` - Which sizes to test

**Outputs**:
- `data/results/portfolio_size_comparison_report.csv`

**Finding**: 6-7 tickers optimal (best Sharpe, good diversification)

---

## 🎯 ALL HARDCODED PARAMETERS (Configurable Opportunities)

### Script 0 - Foundation
| Parameter | Current Value | Purpose | Should Change? |
|-----------|--------------|---------|----------------|
| `TOP_N` | 50 | Top performers to select | ✅ Yes (try 30, 75, 100) |
| `SCORING_WEIGHTS` | PF:0.4, WR:0.3, Ret:0.3 | Composite score formula | ✅ Yes (experiment) |
| Trade category | Anti-cascading only | Final selection filter | ⚠️ Maybe (try both) |

### Script 1 - Affordability
| Parameter | Current Value | Purpose | Should Change? |
|-----------|--------------|---------|----------------|
| `PRICE_THRESHOLD` | ₹2,000 | Max entry price | ✅ Yes (₹1,500/₹3,000/₹5,000) |

### Script 2 - Sector Mapping
| Parameter | Current Value | Purpose | Should Change? |
|-----------|--------------|---------|----------------|
| `SECTOR_MAPPING` | 28 hardcoded pairs | Ticker→Sector | ⚠️ Must update if Top 50 changes |
| Correlation method | Pearson | Correlation calc | ⚠️ Maybe (try Spearman) |

### Script 3 - Combination Filters
| Parameter | Current Value | Purpose | Should Change? |
|-----------|--------------|---------|----------------|
| `MAX_SECTOR_CONCENTRATION` | 0.60 (60%) | Diversification limit | ✅ Yes (try 0.50, 0.70) |
| `MAX_CORRELATION` | 0.70 | Correlation threshold | ✅ Yes (try 0.60, 0.80) |

### Script 4 - Performance
| Parameter | Current Value | Purpose | Should Change? |
|-----------|--------------|---------|----------------|
| `RISK_FREE_RATE` | 0.0% | Sharpe calculation | ✅ YES! (India ~6-7%) |
| `N_PORTFOLIOS_TEST` | 10,000 | Random sample size | ⚠️ Maybe (if compute allows) |
| `TOP_N_RESULTS` | 50 | Keep top N | ⚠️ Maybe (10, 100) |
| Weight allocation | Equal (1/N) | Portfolio weighting | ⚠️ Already tried optimization |

### Script 5 - PyPortfolioOpt
| Parameter | Current Value | Purpose | Should Change? |
|-----------|--------------|---------|----------------|
| `FREQUENCY` | 252 days | Annualization factor | ⚠️ No (standard) |
| `L2_REG_GAMMA` | 0.1 | Regularization | ⚠️ Maybe (if trying again) |

---

## 📁 COMPLETE OUTPUT INVENTORY

After running **full pipeline** (Scripts 0→5 + Master Optimizer), you'll have:

### Foundation Data
```
data/foundation/
├── TOP50_ALL_TRADES.csv                          # 50 tickers (all trades)
├── TOP50_ANTICASCADING_TRADES.csv                # 50 tickers (anti-cascading)
├── TOP50_CASCADING_TRADES.csv                    # 50 tickers (cascading)
├── top50_ALL_TRADES_performance.csv              # Metrics
├── top50_ANTICASCADING_TRADES_performance.csv
└── cascade_vs_anticascade_comparison_summary.txt
```

### Filtered Data
```
data/filtered/
├── CORRECTED_affordable_tickers_metadata.csv     # 28 tickers
├── CORRECTED_anti_cascading_trades_under2k.csv   # 39,221 trades
└── CORRECTED_anti_cascading_dataset_summary.txt
```

### Classified Data
```
data/classified/
├── CORRECTED_sector_mapping.csv                  # 28 ticker→sector
├── CORRECTED_correlation_matrix.csv              # 28×28 matrix
├── CORRECTED_daily_returns_data.csv              # Time series
└── CORRECTED_sector_correlation_summary.txt
```

### Combinations (5 files)
```
data/combinations/
├── PERCENTAGE_valid_combinations_4ticker.csv     # 30,135 portfolios
├── PERCENTAGE_valid_combinations_5ticker.csv     # 97,712
├── PERCENTAGE_valid_combinations_6ticker.csv     # 389,224
├── PERCENTAGE_valid_combinations_7ticker.csv     # 1,184,040
├── PERCENTAGE_valid_combinations_8ticker.csv     # 3,108,105
└── PERCENTAGE_combination_generation_summary_5ticker.txt
```

### Results (14 files)
```
data/results/
# Portfolio Performance (Script 4)
├── portfolio_performance_4ticker_ALL.csv         # 10K tested
├── portfolio_performance_4ticker_TOP50.csv       # Top 50
├── portfolio_performance_5ticker_ALL.csv
├── portfolio_performance_5ticker_TOP50.csv
├── portfolio_performance_6ticker_ALL.csv
├── portfolio_performance_6ticker_TOP50.csv       ✅ OPTIMAL
├── portfolio_performance_7ticker_ALL.csv
├── portfolio_performance_7ticker_TOP50.csv
├── portfolio_performance_8ticker_ALL.csv
├── portfolio_performance_8ticker_TOP50.csv
├── portfolio_size_comparison_report.csv          # Master summary

# Optimal Weights (Script 5)
├── optimal_weights_6ticker.csv                   # PyPortfolioOpt results
└── optimal_weights_7ticker.csv
```

### Logs
```
logs/
├── master_optimizer.log                          # Full run log
├── script1_output.log
├── script2_output.log
├── script3_output.log
└── script4_output.log
```

---

## 🔧 NEW FEATURES ADDED (vs Original)

### ✅ Percentage Returns (vs Absolute Currency)
**Change**: Entire pipeline now uses percentage returns instead of absolute profit
**Why**: Capital efficiency matters - small capital + high % > large capital + low %
**Impact**: Only 8% overlap between old/new Top 50 lists

### ✅ Portfolio Size Optimization
**New**: Master optimizer tests 4,5,6,7,8 ticker portfolios systematically
**Finding**: 6-7 tickers optimal (Sharpe 1.81, best consistency)

### ✅ PyPortfolioOpt Integration (Script 5)
**New**: Professional weight optimization using Markowitz mean-variance
**Result**: Equal-weight remains best due to data sparsity

### ✅ Organized Directory Structure
**New**: Clean separation (scripts/, data/{foundation,filtered,classified,combinations,results}/)
**Benefit**: Easy to navigate, understand data flow

### ✅ Removed Premature Filtering (Script 3)
**Change**: No longer filters by PF≥0.95, Accuracy≥42% after Top 50 selection
**Why**: Double-filtering was redundant (already selected as Top 50)
**Impact**: 8 valid tickers → 28 valid tickers

---

## 🎨 VISUALIZATIONS TO CREATE (Option 1 - Next)

### 1. Equity Curves (Primary)
**Script**: `scripts/06_equity_curve_generator.py`
**Charts**:
- Cumulative return curves (Top 10 portfolios vs benchmark)
- Drawdown analysis (underwater equity curves)
- Monthly return heatmaps
- Rolling Sharpe ratio

### 2. Portfolio Comparison
**Charts**:
- 6-ticker vs 7-ticker performance comparison
- Sector allocation pie charts (Top 5 portfolios)
- Correlation heatmaps (Top portfolio constituents)

### 3. Risk Analysis
**Charts**:
- Return vs Volatility scatter (efficient frontier approximation)
- Max drawdown distribution
- Win rate vs Profit factor scatter (by portfolio)

**Output Location**: `data/results/visualizations/`

---

## 🚀 COMPLETE RUN SEQUENCE

To reproduce entire pipeline from scratch:

```bash
# Step 1: Foundation (Top 50 selection)
python scripts/00_foundation_cascade_vs_anticascade_analysis.py
# Output: 50 tickers

# Step 2: Affordability filter
python scripts/01_corrected_anti_cascading_subset.py
# Output: 28 affordable tickers

# Step 3: Sector classification
python scripts/02_corrected_sector_classification_correlation.py
# Output: Sector mapping, correlation matrix

# Step 4: Generate combinations + Calculate performance (all sizes)
python scripts/master_portfolio_optimizer.py
# Output: 4-8 ticker portfolio results, size comparison

# Step 5: Optimize weights
python scripts/05_pypfopt_optimal_weights.py
# Output: optimal_weights_6ticker.csv, optimal_weights_7ticker.csv

# Step 6: Generate visualizations (TO BE CREATED)
python scripts/06_equity_curve_generator.py
# Output: Equity curves, drawdown charts, heatmaps
```

**Total Runtime**: ~10-15 minutes (depending on portfolio sizes tested)

---

## 📊 KEY INSIGHTS SUMMARY

### What Works Well ✅
- **Percentage returns**: Reveals true capital efficiency
- **Diversification filters**: 60% sector limit, 0.70 correlation prevents over-concentration
- **6-7 ticker portfolios**: Sweet spot (Sharpe 1.76-1.81, good diversification without dilution)
- **Equal-weight allocation**: Performs as well as optimization attempts

### What Needs Improvement ⚠️
- **Risk-free rate = 0%**: Should be ~6-7% for India (impacts Sharpe calculation)
- **Sparse data for optimization**: Trade-level → daily resampling creates gaps
- **Hardcoded sector mapping**: Breaks if Top 50 changes (needs dynamic mapping or API)
- **No transaction costs**: Real portfolios need cost modeling

### Quick Wins 🎯
1. Update `RISK_FREE_RATE = 0.065` (6.5% India benchmark)
2. Make scoring weights configurable (currently 0.4/0.3/0.3)
3. Test price thresholds (₹1,500 / ₹3,000 instead of ₹2,000)
4. Try different correlation thresholds (0.60 / 0.80 instead of 0.70)

---

## 🔮 PHASE 2 PREVIEW (After Visualizations)

Once equity curves are done, Phase 2 focuses on **MSE Strategy Optimization**:

### MSE Strategy Current State
- **Entry**: ALL 4 indicators bullish (5min MACD>signal, 15min MACD>signal, 5min EMA9>EMA20, 15min EMA9>EMA20)
- **Exit**: 15min MACD histogram at **80% of peak/valley**
- **Performance**: WR 48%, PF 1.14 (marginal)

### Phase 2 Optimization Targets
1. **Exit threshold**: Test 50-95% instead of 80% (expected optimal: 85%)
2. **Entry filters**: Add MACD strength >0.3, EMA spread >0.5%
3. **Timeframe analysis**: Why 5min + 15min? Test other combinations
4. **Walk-forward validation**: Out-of-sample testing

**Data Source**: `outputs/.../data/base_data/*.parquet` (5min granularity with all indicators)

---

## 📝 CONFIGURATION FILE NEEDED (Future Enhancement)

Currently everything is hardcoded. Should create `config.yaml`:

```yaml
# Portfolio Construction Configuration

foundation:
  top_n: 50
  scoring_weights:
    profit_factor: 0.40
    win_rate: 0.30
    avg_return_pct: 0.30
  trade_category: "anti_cascading"  # or "all", "cascading"

affordability:
  price_threshold: 2000  # Rupees

sector_classification:
  correlation_method: "pearson"  # or "spearman", "kendall"

combination_generation:
  max_sector_concentration: 0.60
  max_correlation: 0.70

portfolio_performance:
  risk_free_rate: 0.065  # 6.5% India benchmark
  n_portfolios_test: 10000
  top_n_results: 50

optimization:
  portfolio_sizes: [4, 5, 6, 7, 8]
  weight_method: "equal"  # or "max_sharpe", "min_vol"

visualization:
  top_n_portfolios: 10
  benchmark_ticker: "NIFTY50"
```

This would make the pipeline **fully configurable** without code changes.

---

**NEXT IMMEDIATE TASK**: Create Script 6 for equity curve visualizations, then we'll have complete Phase 1 validation before Phase 2 strategy optimization.
