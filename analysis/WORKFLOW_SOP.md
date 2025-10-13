# Complete Analysis Workflow - Standard Operating Procedure (SOP)
## From Backtest to Validation & Optimization

**Version**: 2.0 - Config-Driven Workflow
**Last Updated**: October 2025
**Purpose**: Step-by-step guide for analyzing ANY trading strategy from backtest to production

---

## 📋 **Table of Contents**

1. [Overview & Philosophy](#overview--philosophy)
2. [Prerequisites](#prerequisites)
3. [Stage 1: Run Backtest](#stage-1-run-backtest)
4. [Stage 2: Merge Trade Files](#stage-2-merge-trade-files)
5. [Stage 3: Generic Analysis](#stage-3-generic-analysis)
6. [Stage 4: Portfolio Construction](#stage-4-portfolio-construction)
7. [Stage 5: Strategy Optimization](#stage-5-strategy-optimization-optional)
8. [Stage 6: Validation & Sign-Off](#stage-6-validation--sign-off)
9. [Terminal Commands Quick Reference](#terminal-commands-quick-reference)
10. [Troubleshooting](#troubleshooting)

---

## **Overview & Philosophy**

### **The Complete Journey**

```
┌─────────────────┐
│  1. BACKTEST    │  ← Generate trade data
│  unified_runner │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  2. MERGE       │  ← Consolidate ticker files
│  merge_trades   │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  3. ANALYZE     │  ← Generic insights
│  generic/       │     (works with ANY strategy)
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  4. PORTFOLIO   │  ← Optimal combinations
│  construction/  │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  5. OPTIMIZE    │  ← Strategy-specific tuning
│  strategy_      │     (MSE: exit thresholds, etc.)
│  specific/      │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  6. VALIDATE    │  ← Test data verification
│  Sign-off       │     (never seen before!)
└─────────────────┘
```

### **Key Principles**

1. **Config-Driven**: One YAML file controls everything
2. **Modular**: Each stage is independent
3. **Reusable**: Same workflow for ANY strategy
4. **Reproducible**: Documented decisions at each step
5. **Safe**: Test data remains untouched until final validation

---

## **Prerequisites**

### **1. Environment Setup**

```powershell
# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Verify Python and packages
python --version
pip list | grep pandas
```

### **2. Required Files**

- ✅ `config/unified_config.py` - Backtester configuration
- ✅ `config/templates/*.yaml` - Risk templates
- ✅ `src/strategies/mse_strategy_backtesting.py` - Your strategy
- ✅ `utils/merge_trades.py` - Trade merger (just created!)
- ✅ `analysis/config_template.yaml` - Analysis config template

### **3. Data Requirements**

Ensure you have market data in:
```
data/pools/YYYY-MM-DD_to_YYYY-MM-DD/
├── 1minute/
├── 5minute/
└── 15minute/
```

---

## **Stage 1: Run Backtest**

### **Purpose**
Generate trade data for analysis.

### **Command**

```powershell
# Basic backtest (auto-discovers tickers from data pools)
python src/runners/unified_runner.py `
    --mode backtest `
    --strategy mse `
    --date-ranges 2022-01-01_to_2025-08-31

# With specific tickers
python src/runners/unified_runner.py `
    --mode backtest `
    --strategy mse `
    --date-ranges 2022-01-01_to_2025-08-31 `
    --tickers RELIANCE TCS INFY HDFC

# With template (risk management)
python src/runners/unified_runner.py `
    --mode backtest `
    --template conservative `
    --strategy mse `
    --date-ranges 2022-01-01_to_2025-08-31
```

### **Output Structure**

```
outputs/
└── 20251006_024924/              ← RUN_ID (timestamp)
    └── mse/                      ← STRATEGY
        └── 2022-01-01_to_2025-08-31/  ← DATE_RANGE
            ├── data/
            │   ├── strategy_trades/   ← Raw strategy signals
            │   │   ├── RELIANCE_StrategyTrades_*.csv
            │   │   ├── TCS_StrategyTrades_*.csv
            │   │   └── ...
            │   ├── risk_approved_trades/  ← After risk management
            │   │   ├── RELIANCE_RiskApprovedTrades_*.csv
            │   │   └── ...
            │   └── base_data/         ← Indicator-level data
            │       ├── RELIANCE_Base_*.csv
            │       └── ...
            ├── tickers/              ← Per-ticker metrics
            ├── portfolio/            ← Portfolio-level analysis
            └── visualizations/       ← Charts
```

### **Key Information to Record**

After backtest completes, note:
- ✅ **RUN_ID**: `20251006_024924` (from outputs/ folder name)
- ✅ **STRATEGY**: `mse`
- ✅ **DATE_RANGE**: `2022-01-01_to_2025-08-31`
- ✅ **Trade Source**: `strategy_trades` or `risk_approved_trades`

**You need these values for the next step!**

---

## **Stage 2: Merge Trade Files**

### **Purpose**
Combine individual ticker trade CSVs into a single consolidated file for analysis.

### **Why Merge?**

**Problem**: Backtest creates 30-50 separate CSV files (one per ticker)
**Solution**: Merge into ONE file that analysis scripts can process

**Benefits**:
- Single file = faster analysis
- Cross-ticker patterns (cascading trades)
- Portfolio-level insights
- Easier to share/archive

### **Steps**

#### **2.1: Create Analysis Config**

```powershell
# Navigate to analysis directory
cd analysis

# Copy template
cp config_template.yaml config.yaml

# Edit config.yaml
```

**Fill in your backtest details in `config.yaml`**:

```yaml
run:
  run_id: "20251006_024924"          # ← FROM BACKTEST OUTPUT
  strategy: "mse"                     # ← YOUR STRATEGY
  date_range: "2022-01-01_to_2025-08-31"  # ← YOUR DATE RANGE
  trade_source: "strategy_trades"     # ← OR "risk_approved_trades"

output:
  merged_filename: "all_trades_merged.csv"
```

#### **2.2: Run Merge Script**

```powershell
# From analysis/ directory
python ../utils/merge_trades.py --config config.yaml
```

**Expected Output**:

```
✅ Loaded config from: config.yaml
Found 30 files to merge

Sample files:
  - RELIANCE_StrategyTrades_2022-01-01_to_2025-08-31.csv
  - TCS_StrategyTrades_2022-01-01_to_2025-08-31.csv
  ...

Reading files...
  Processed 30/30 files...

Merging dataframes...
Sorting by entry time...
Saving to: outputs/.../data/all_trades_merged.csv

============================================================
MERGE SUMMARY
============================================================
✅ Files processed: 30
✅ Total trades: 45,234
📅 Date range: 2022-01-03 to 2025-08-29
🎯 Unique tickers: 30
📁 Output file: outputs/.../all_trades_merged.csv
💾 File size: 12.45 MB

📊 Trade Type Distribution:
   Buy: 22,145 (48.9%)
   Sell: 23,089 (51.1%)

🏆 Top 10 Tickers by Trade Count:
   RELIANCE: 2,145
   TCS: 1,987
   INFY: 1,876
   ...

💰 Profitability Summary:
   Total P&L: ₹1,234,567.89
   Winning Trades: 21,234 (46.9%)
   Losing Trades: 24,000 (53.1%)

✅ Merge completed successfully!
============================================================
```

#### **2.3: Verify Merged File**

```powershell
# Check file exists
ls outputs/20251006_024924/mse/2022-01-01_to_2025-08-31/data/all_trades_merged.csv

# View first 10 rows
head -10 outputs/.../all_trades_merged.csv
```

**Merged File Columns**:
- `ticker`: Stock symbol
- `Entry Time`, `Exit Time`: Timestamps
- `Entry Price`, `Exit Price`: Prices
- `Profit (Currency)`: P&L in ₹
- `Profit (%)`: Percentage return
- `Trade Type`: Buy/Sell
- `Trade Duration (min)`: Holding period
- ... and more

---

## **Stage 3: Generic Analysis**

### **Purpose**
Run strategy-agnostic analysis to understand trade patterns, performance, and risks.

### **Available Modules**

#### **3.1: Basic EDA** (Exploratory Data Analysis)

**What it does**: Overall statistics, win rate, profit factor, duration analysis

**Run** (once scripts are migrated to generic/):
```powershell
python generic/scripts/01_basic_eda.py --config config.yaml
```

**Outputs**:
- Total trades, win rate, profit factor
- P&L distribution
- Trade duration statistics
- Time-of-day patterns

#### **3.2: Trade Type Analysis** (Buy vs Sell)

**What it does**: Compare Buy vs Sell performance

**Run**:
```powershell
python generic/scripts/02_trade_type_analysis.py --config config.yaml
```

**Insights**:
- Which direction performs better?
- Risk/reward differences
- Duration patterns
- Sector preferences by direction

#### **3.3: Cascade Analysis** 🌟 **IMPORTANT**

**What it does**: Identify sequential trade patterns (winning/losing streaks)

**Why it matters**:
- Detect "revenge trading" (quick re-entry after loss)
- Find "momentum trades" (consecutive wins)
- Understand trade clustering
- **Portfolio Construction Input**: Filter "cascading" trades

**Run**:
```powershell
python generic/scripts/03_cascade_analysis.py --config config.yaml
```

**Outputs**:
- Cascade tags for each trade:
  - `FIRST_TRADE_OF_DAY`
  - `CONSECUTIVE_SAME_DIRECTION`
  - `CONSECUTIVE_OPPOSITE_DIRECTION`
  - `WINNING_CASCADE` (after win)
  - `LOSING_CASCADE` (after loss)
- Performance by cascade type
- Time gaps between trades

**Example Findings**:
- "Trades after a loss have 35% win rate (vs 48% overall)"
- "Trades 30-60 min after a win have 55% win rate"
- "First trade of day: 52% win rate"

#### **3.4: Ticker Ranking**

**What it does**: Rank tickers by performance metrics

**Run**:
```powershell
python generic/scripts/04_ticker_ranking.py --config config.yaml
```

**Outputs**:
- Top 50 performers (by weighted score)
- Ranking criteria:
  - Profit Factor (40%)
  - Win Rate (30%)
  - Avg Return (30%)
- Filter for portfolio construction

#### **3.5: Exit Timing Analysis**

**What it does**: Analyze optimal holding periods

**Run**:
```powershell
python generic/scripts/05_exit_timing.py --config config.yaml
```

**Insights**:
- Best/worst exit times
- Duration vs profitability
- Peak capture analysis

#### **3.6: Stop Loss Simulation**

**What it does**: Test different stop loss thresholds (0.5%, 1%, 1.5%, 2%, 2.5%)

**Run**:
```powershell
python generic/scripts/06_stop_loss_sim.py --config config.yaml
```

**Outputs**:
- Trades saved/killed by each threshold
- Net P&L impact
- Optimal stop loss level

---

## **Stage 4: Portfolio Construction**

### **Purpose**
Identify optimal ticker combinations for diversified portfolios.

### **Workflow**

#### **4.1: Top Performer Selection**

**Run**:
```powershell
cd analysis/portfolio_construction
python scripts/00_foundation.py --config ../config.yaml
```

**What it does**:
- Rank all tickers by weighted score
- Filter for:
  - Anti-cascading trades (use cascade tags from Stage 3!)
  - Minimum trade count (>20 trades)
  - Positive profit factor (>1.0)
- Output: `Top50_performers.csv`

#### **4.2: Affordability Filter**

**Run**:
```powershell
python scripts/01_affordability.py --config ../config.yaml
```

**What it does**:
- Filter tickers with Entry Price < ₹2,000
- Output: `affordable_tickers.csv` (~28 tickers)

#### **4.3: Sector Classification**

**Run**:
```powershell
python scripts/02_sector_classification.py --config ../config.yaml
```

**What it does**:
- Map tickers to sectors (Finance, IT, Energy, etc.)
- Calculate pairwise correlations
- Output:
  - `sector_mapping.csv`
  - `correlation_matrix.csv`

#### **4.4: Portfolio Combinations**

**Run**:
```powershell
python scripts/03_combinations.py --config ../config.yaml
```

**What it does**:
- Generate all valid N-ticker portfolios (N=4,5,6,7,8)
- Apply filters:
  - Max 60% sector concentration
  - Max 0.7 pairwise correlation
- Output: `valid_combinations_N_ticker.csv`

#### **4.5: Portfolio Optimization**

**Run**:
```powershell
python scripts/04_optimization.py --config ../config.yaml
```

**What it does**:
- Test 10,000 random portfolios per size
- Calculate portfolio-level Sharpe ratio
- Rank by: Sharpe, return, drawdown
- Output: `portfolio_performance_N_ticker.csv`

**Key Finding** (from previous runs):
- **6-7 ticker portfolios optimal** (Sharpe 1.81)
- Best balance of diversification vs complexity

#### **4.6: Master Optimizer** (Run All)

**Run**:
```powershell
python scripts/master_optimizer.py --config ../config.yaml
```

**What it does**:
- Runs scripts 3-5 for all portfolio sizes
- Generates comparison report
- Outputs: `portfolio_size_comparison_report.csv`

---

## **Stage 5: Strategy Optimization** (Optional - MSE Specific)

### **Purpose**
Optimize strategy-specific parameters (exit thresholds, entry filters).

### **When to Run**

**Use this if**:
- Strategy is performing < target (e.g., 48% WR < 52% target)
- Want to fine-tune entry/exit logic
- Have strategy-specific parameters to optimize

**Skip this if**:
- Strategy already meets goals
- Just want portfolio insights
- This is a NEW strategy (validate first!)

### **MSE Optimization Example**

#### **Problem Statement**
- Current MSE: 48% WR, PF 1.14
- Target: 52% WR, PF 1.25
- Hypothesis: 80% MACD exit threshold is suboptimal

#### **Workflow**

**Navigate to MSE optimization**:
```powershell
cd analysis/strategy_specific/mse
```

**Read the guide**:
```powershell
cat README.md
```

**Stage-by-stage execution** (6 stages):

```powershell
# Stage 0: Setup verification
python scripts/00_setup_verification.py

# Review: Check docs/PHASE2_ANALYSIS_LOG.md
# Decision: Proceed? [YES/NO]

# Stage 1: Baseline
python scripts/01_baseline_calculator.py

# Review: Read docs/baseline_report.md
# Update: PHASE2_ANALYSIS_LOG.md with observations
# Decision: Baseline acceptable? [YES/NO]

# Stage 2: Exit threshold optimization (H1)
python scripts/02_exit_threshold_optimizer.py

# Review: Read docs/exit_threshold_analysis_report.md
# Decision: Improvement ≥5% PF? Accept H1? [YES/NO]

# Stage 3: Entry filter optimization (H2)
python scripts/03_entry_filter_optimizer.py

# Review: Read docs/entry_filter_analysis_report.md
# Decision: Improvement ≥10% WR? Accept H2? [YES/NO]

# Stage 4: Walk-forward validation
python scripts/04_walkforward_validator.py

# Review: Parameters stable across time windows?
# Decision: CV < 10%? [YES/NO]

# Stage 5: Statistical testing
python scripts/05_statistical_tester.py

# Review: p-value < 0.05?
# Decision: Statistically significant? [YES/NO]

# Stage 6: FINAL TEST DATA VERIFICATION
# ⚠️ CRITICAL: This data has NEVER been seen before!
python scripts/06_final_verifier.py

# Review: Read docs/EXECUTIVE_SUMMARY.md
# FINAL DECISION: ✅ DEPLOY or ❌ REJECT
```

**Philosophy**:
> "Measure twice, cut once. Document everything. Trust nothing until validated on unseen data."

**Key Rules**:
- Never skip stages
- Update PHASE2_ANALYSIS_LOG.md after EACH stage
- Test data is sacred (Stage 6 only, one time)
- If Stage 6 fails → entire optimization is rejected (overfitting)

---

## **Stage 6: Validation & Sign-Off**

### **Purpose**
Final verification before production deployment.

### **Validation Checklist**

#### **Data Quality**
- [ ] No look-ahead bias in strategy
- [ ] Timestamps are sequential
- [ ] No duplicate trades
- [ ] Entry/exit prices realistic
- [ ] No missing critical data

#### **Performance Metrics**
- [ ] Win rate > target (e.g., 48% → 52%)
- [ ] Profit factor > target (e.g., 1.14 → 1.25)
- [ ] Max drawdown < acceptable (e.g., < 15%)
- [ ] Sharpe ratio > threshold (e.g., > 1.5)
- [ ] Trade frequency adequate (not too sparse)

#### **Portfolio Construction**
- [ ] Top 50 tickers identified
- [ ] Sector diversification validated
- [ ] Correlations acceptable (< 0.7)
- [ ] Optimal portfolio size determined (6-7 tickers)
- [ ] Anti-cascading filter applied

#### **Strategy Optimization** (if performed)
- [ ] Baseline established
- [ ] Hypothesis 1 (H1) validated
- [ ] Hypothesis 2 (H2) validated
- [ ] Walk-forward validation passed
- [ ] Statistical significance confirmed (p < 0.05)
- [ ] **Test data verification passed** (Stage 6)

#### **Documentation**
- [ ] All analysis scripts documented
- [ ] Config files saved
- [ ] Decision log updated
- [ ] Reports generated
- [ ] Results shared with stakeholders

### **Sign-Off**

**Decision Matrix**:

| Criterion | Status | Notes |
|-----------|--------|-------|
| Data Quality | ✅/❌ | |
| Performance | ✅/❌ | |
| Portfolio | ✅/❌ | |
| Optimization | ✅/❌ | (if performed) |
| Documentation | ✅/❌ | |

**Final Decision**:
- ✅ **DEPLOY**: All criteria met
- ⚠️ **CONDITIONAL**: Minor issues, deploy with monitoring
- ❌ **REJECT**: Critical issues, return to optimization

---

## **Terminal Commands Quick Reference**

### **Complete Workflow (Copy-Paste)**

```powershell
# ============================================
# COMPLETE ANALYSIS WORKFLOW
# ============================================

# Step 1: Run Backtest
python src/runners/unified_runner.py `
    --mode backtest `
    --strategy mse `
    --date-ranges 2022-01-01_to_2025-08-31

# Note the RUN_ID, STRATEGY, DATE_RANGE from output

# Step 2: Setup Analysis Config
cd analysis
cp config_template.yaml config.yaml
# Edit config.yaml with your RUN_ID, STRATEGY, DATE_RANGE

# Step 3: Merge Trades
python ../utils/merge_trades.py --config config.yaml

# Step 4: Generic Analysis
python generic/scripts/01_basic_eda.py --config config.yaml
python generic/scripts/02_trade_type_analysis.py --config config.yaml
python generic/scripts/03_cascade_analysis.py --config config.yaml
python generic/scripts/04_ticker_ranking.py --config config.yaml

# Step 5: Portfolio Construction
cd portfolio_construction
python scripts/master_optimizer.py --config ../config.yaml

# Step 6: Strategy Optimization (Optional)
cd ../strategy_specific/mse
python scripts/00_setup_verification.py
# Follow stage-by-stage workflow...

# Step 7: Review & Sign-Off
# Check all reports in analysis/reports/
# Make GO/NO-GO decision
```

---

## **Troubleshooting**

### **Common Issues**

#### **Issue 1: Merge script finds no files**

```
❌ No CSV files found matching pattern!
```

**Solution**:
- Verify `run_id`, `strategy`, `date_range` in `config.yaml`
- Check if backtest actually completed
- Verify trade_source is correct (`strategy_trades` vs `risk_approved_trades`)

**Debug**:
```powershell
# Check if directory exists
ls outputs/20251006_024924/mse/2022-01-01_to_2025-08-31/data/strategy_trades/
```

#### **Issue 2: Analysis script errors on hardcoded paths**

```
FileNotFoundError: No such file or directory: '/mnt/batch/...'
```

**Solution**:
- Script hasn't been updated to use YAML config yet
- Check `analysis/DIRECTORY_STRUCTURE.md` for migration status
- Use legacy scripts in `mse_analysis/` temporarily

#### **Issue 3: 0% Win Rate After Backtest**

```
"winning_trades": 0,
"losing_trades": 2591,
"win_rate_pct": 0.0
```

**Solution**:
- Check strategy logic (entry/exit conditions)
- Verify data quality (missing OHLCV data?)
- Compare strategy_trades vs risk_approved_trades
- Check warmup period (first 525 minutes may be invalid)

**Debug**:
```powershell
# Check first 20 trades
head -20 outputs/.../strategy_trades/RELIANCE_StrategyTrades_*.csv
```

#### **Issue 4: Config not found**

```
❌ Config file not found: config.yaml
```

**Solution**:
- Ensure you're in `analysis/` directory
- Copy template: `cp config_template.yaml config.yaml`
- Use correct path: `--config analysis/config.yaml` if running from root

---

## **Summary**

### **Key Takeaways**

1. **Config-Driven**: One YAML controls everything
2. **Modular**: Each stage independent, can skip/repeat
3. **Reusable**: Same workflow for ANY strategy
4. **Safe**: Test data untouched until final validation
5. **Documented**: Decision log at every step

### **What Makes This System Unique**

- **Three-file output** (strategy, risk, base_data)
- **Cascade analysis** (sequential patterns)
- **Indicator-level debugging** (integration module)
- **Portfolio construction pipeline** (institutional-grade)
- **Hypothesis-driven optimization** (not just parameter sweeping)

### **Next Steps**

1. ✅ Run your first backtest
2. ✅ Merge trades using config
3. ✅ Run generic analysis
4. ✅ Build optimal portfolios
5. ⏳ Optimize strategy (if needed)
6. ✅ Validate and deploy

---

**Questions?** Check:
- `analysis/ANALYSIS_PROTOCOL.md` - Analysis methodology
- `analysis/DIRECTORY_STRUCTURE.md` - File organization
- `CLAUDE.md` - System architecture
- `README.md` - Quick start guide

**Ready to start? Run Stage 1!**

```powershell
python src/runners/unified_runner.py --mode backtest --strategy mse --date-ranges 2022-01-01_to_2025-08-31
```

🚀 **Happy Trading!**
