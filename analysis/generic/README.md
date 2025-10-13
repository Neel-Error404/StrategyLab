# Generic Analysis Scripts
## Strategy-Agnostic Trade Analysis

**Purpose**: This directory contains analysis scripts that work with **ANY** strategy's trade data, not just MSE.

---

## 📋 **Available Scripts**

| Script | Description | Status |
|--------|-------------|--------|
| `01_basic_eda.py` | Overall statistics, win rate, profit factor | ✅ Ready |
| `02_trade_type_analysis.py` | Directional (buy vs sell) performance deep dive | ✅ Ready |
| `03_cascade_analysis.py` | Sequential trade pattern tagging & analysis | ✅ Ready |
| `04_stop_loss_simulation.py` | Percentage stop-loss sweeps & recommendations | ✅ Ready |
| `05_ticker_ranking.py` | Weighted ranking + filter pipeline for top tickers | ✅ Ready |
| `06_risk_adjusted_patterns.py` | Risk-adjusted motif analytics & exposure heatmaps | ✅ Ready |
| `07_top50_vs_overall.py` | Compare Top-50 portfolio vs entire trade universe | ✅ Ready |
| `08_top50_pattern_breakdown.py` | Drill down into Top-50 constituent behavior | ✅ Ready |
| `09_validation_check.py` | Runbook of guard-rail checks before portfolio build | ✅ Ready |

---

## 🚀 **Quick Start**

### **1. Setup Configuration**

```powershell
# Copy template
cd analysis
cp config_template.yaml config.yaml

# Edit config.yaml with your backtest details
# run_id: "20251006_024924"
# strategy: "mse"
# date_range: "2022-01-01_to_2025-08-31"
```

### **2. Merge Trade Files**

```powershell
python ../utils/merge_trades.py --config config.yaml
```

### **3. Run Analysis Scripts**

```powershell
# Navigate to generic scripts
cd generic/scripts

# Basic EDA
python 01_basic_eda.py --config ../../config.yaml

# Directional performance
python 02_trade_type_analysis.py --config ../../config.yaml

# Cascade Analysis (momentum vs mean-reversion patterns)
python 03_cascade_analysis.py --config ../../config.yaml

# Stop-loss sweep (thresholds configured in YAML)
python 04_stop_loss_simulation.py --config ../../config.yaml

# Weighted ticker ranking for portfolio prep
python 05_ticker_ranking.py --config ../../config.yaml

# Validation guard-rails before downstream workflows
python 09_validation_check.py --config ../../config.yaml
```

---

## ✅ Quality Gates & Execution Modes

The generic suite behaves like a regression harness: shared loaders enforce consistent inputs, the YAML schema captures run metadata, and the orchestrator in `analysis/run.py` can replay entire batteries of checks for any strategy without editing code.【F:analysis/generic/modules/config_loader.py†L1-L120】【F:analysis/generic/modules/data_loader.py†L17-L115】【F:analysis/run.py†L1-L188】

| Checkpoint | Primary scripts | What it validates | Adapting to another strategy |
|------------|-----------------|-------------------|------------------------------|
| **Data integrity** | `09_validation_check.py`, `modules.data_loader.validate_trade_data` | Confirms required columns, timestamp ordering, and profit consistency before downstream analytics.【F:analysis/generic/scripts/09_validation_check.py†L40-L160】【F:analysis/generic/modules/data_loader.py†L200-L259】 | Point the config to a different `run_id` and merged CSV; the validator will surface schema mismatches instantly. |
| **Directional bias** | `02_trade_type_analysis.py` | Quantifies buy vs sell profitability, drawdowns, and efficiency deltas so hedging rules can be tuned per strategy.【F:analysis/generic/scripts/02_trade_type_analysis.py†L40-L160】 | Adjust `analysis.generic.modules.trade_type_analysis.config.metrics` or the optional `sample_size` in YAML to focus on the most relevant KPIs. |
| **Risk protection** | `04_stop_loss_simulation.py` | Sweeps stop-loss thresholds, reports P&L deltas, and highlights saved losses to test capital protection policies.【F:analysis/generic/scripts/04_stop_loss_simulation.py†L47-L160】 | Override `analysis.generic.modules.stop_loss_simulation.config.thresholds` in the config to match the volatility profile of the new strategy. |
| **Ticker selection** | `05_ticker_ranking.py`, `07_top50_vs_overall.py`, `08_top50_pattern_breakdown.py` | Scores instruments across profitability, drawdown, consistency, and pattern behaviour to create a portfolio-ready whitelist.【F:analysis/generic/scripts/05_ticker_ranking.py†L60-L180】【F:analysis/generic/scripts/07_top50_vs_overall.py†L1-L110】【F:analysis/generic/scripts/08_top50_pattern_breakdown.py†L1-L120】 | Update ranking weights in YAML or feed custom ticker universes; downstream scripts automatically consume the generated CSVs. |
| **Pattern & regime** | `03_cascade_analysis.py`, `06_risk_adjusted_patterns.py` | Tags cascades vs anti-cascades and measures Sharpe/drawdown per motif to detect behavioural edge or drag.【F:analysis/generic/scripts/03_cascade_analysis.py†L1-L80】【F:analysis/generic/scripts/06_risk_adjusted_patterns.py†L1-L84】 | Enable or disable specific categories via `analysis.generic.modules.cascade_analysis` toggles for different holding styles. |

Run everything hands-free with:

```bash
python analysis/run.py --config analysis/config.yaml --targets generic
```

The runner respects module dependencies (for example, ticker ranking before Top-50 comparisons) and writes consolidated logs per execution.【F:analysis/run.py†L41-L188】

---

## 📊 **Script Details**

### **01_basic_eda.py**

**What it does**:
- Overall statistics (total trades, win rate, profit factor)
- Trade distribution (Buy vs Sell)
- Ticker-level performance summary
- Duration and timing analysis

**Why run it**:
- First gate that confirms merged trades are numerically sane before deeper studies—win rate, profit factor, and duration checks quickly catch corrupt inputs or broken strategies.【F:analysis/generic/scripts/01_basic_eda.py†L55-L110】

**Outputs**:
- `output/basic_eda/basic_eda_statistics.json`
- `output/basic_eda/ticker_performance.csv`
- `reports/BASIC_EDA_REPORT.md`

**Example**:
```powershell
python 01_basic_eda.py --config ../../config.yaml

# With sampling for quick test
python 01_basic_eda.py --config ../../config.yaml --sample 10000
```

---

### **02_trade_type_analysis.py**

**What it does**:
- Deep comparison of Buy vs Sell trades across profitability, drawdown, efficiency, and cadence metrics.
- Optional sampling + metric filters to focus on specific KPIs defined in `analysis/config.yaml`.
- Generates ticker-level directional bias tables to support hedging and pair selection.

**Why run it**:
- Surfaces whether the strategy needs hedging or direction-specific throttles by contrasting win rates, profit factors, and drawdowns between buys and sells.【F:analysis/generic/scripts/02_trade_type_analysis.py†L45-L144】

**Outputs**:
- `directional_summary.csv`
- `directional_summary.json`
- `directional_summary.md`
- `ticker_bias.csv`

**Example**:
```powershell
python 02_trade_type_analysis.py --config ../../config.yaml
python 02_trade_type_analysis.py --config ../../config.yaml --sample 5000
```

---

### **03_cascade_analysis.py**

**What it does**:
- Tags each trade with cascade patterns:
  - First trade of day
  - Consecutive same-direction (Buy→Buy, Sell→Sell)
  - Consecutive opposite-direction (Buy→Sell, Sell→Buy)
  - After winning trade
  - After losing trade
- Analyzes performance by pattern
- Time gap analysis (how quickly next trade occurs)

**Why it matters**:
- Identifies "revenge trading" (quick re-entry after loss)
- Finds momentum trades (consecutive wins)
- **Portfolio construction input**: Filter anti-cascading trades

**Outputs**:
- `output/cascade_analysis/cascade_tagged_trades.csv` (all trades with tags)
- `output/cascade_analysis/cascade_statistics.json`
- `reports/CASCADE_ANALYSIS_REPORT.md`

**Example**:
```powershell
python 03_cascade_analysis.py --config ../../config.yaml
```

**Key Insight**: If "trades after a loss" have <45% win rate, consider blocking them!

---

### **04_stop_loss_simulation.py**

**What it does**:
- Sweeps configurable stop-loss thresholds, highlights optimal protection levels, and surfaces performance deltas.
- Supports scenario comparisons (base vs optimal) and prints narrative recommendations.

**Outputs**:
- `stop_loss_scenarios.csv`
- `stop_loss_summary.json`
- Console summary with optimal threshold + P&L impact.

**Why run it**:
- Quantifies whether additional risk controls help or hurt by comparing baseline vs simulated P&L and win rates for each threshold sweep.【F:analysis/generic/scripts/04_stop_loss_simulation.py†L47-L160】

**Example**:
```powershell
python 04_stop_loss_simulation.py --config ../../config.yaml
```

---

### **05_ticker_ranking.py**

**What it does**:
- Computes composite scores blending profitability, risk, efficiency, and consistency measures.
- Applies liquidity + minimum trade filters before generating ranked tables and JSON summaries.
- Produces tiered segments (S/A/B...) that integrate directly with portfolio construction scripts.

**Outputs**:
- `ticker_scores.csv`
- `top_performers.csv`
- `bottom_performers.csv`
- `ticker_analysis_summary.json`

**Why run it**:
- Produces a ranked whitelist that portfolio scripts and options replay can consume, balancing profitability, drawdown, consistency, and efficiency metrics.【F:analysis/generic/scripts/05_ticker_ranking.py†L60-L180】

**Example**:
```powershell
python 05_ticker_ranking.py --config ../../config.yaml
```

---

### **09_validation_check.py**

**What it does**:
- Runs guard-rail checks (required columns, win-rate sanity, consecutive trade sampling) before handing data to portfolio engines.
- Generates a Markdown dossier with sample trades to accelerate manual QA.

**Outputs**:
- `validation_report.md`
- Console warnings for missing data or suspicious metrics.

**Why run it**:
- Final gate before portfolio construction—emits Markdown dossiers with sample trades, column coverage, and consecutive-trade reviews so reviewers can sign off on data quality.【F:analysis/generic/scripts/09_validation_check.py†L40-L160】

**Example**:
```powershell
python 09_validation_check.py --config ../../config.yaml
```

---

### **Additional Modules**

- `06_risk_adjusted_patterns.py`: Heatmaps risk-adjusted motifs (volatility buckets, holding-period cohorts).
- `07_top50_vs_overall.py`: Benchmarks the curated Top-50 ticker set against the full trade universe.
- `08_top50_pattern_breakdown.py`: Dissects Top-50 combinations for cascade, sector, and regime behaviour.

## 🔧 **Requirements**

### **Data Requirements**

Your merged trades CSV must have these columns:
- `ticker`: Stock symbol
- `Entry Time`: Entry timestamp
- `Exit Time`: Exit timestamp
- `Entry Price`: Entry price
- `Exit Price`: Exit price
- `Profit (Currency)`: P&L in currency
- `Trade Type`: "Buy" or "Sell"
- `Trade Duration (min)`: Holding period

### **Python Requirements**

```
pandas >= 2.0.0
numpy >= 1.24.0
pyyaml >= 6.0
```

---

## 📁 **Directory Structure**

```
generic/
├── README.md              # This file
├── scripts/               # Analysis scripts
│   ├── 01_basic_eda.py
│   ├── 03_cascade_analysis.py
│   └── ...
├── modules/               # Reusable modules
│   ├── __init__.py
│   ├── config_loader.py   # Load YAML config
│   └── data_loader.py     # Load trade/base data
├── output/                # Generated files (gitignored)
│   ├── basic_eda/
│   ├── cascade_analysis/
│   └── ...
└── reports/               # Markdown reports (gitignored)
    ├── BASIC_EDA_REPORT.md
    ├── CASCADE_ANALYSIS_REPORT.md
    └── ...
```

---

## 🧩 **Modules**

### **config_loader.py**

Load and validate YAML configuration:

```python
from modules.config_loader import load_config, resolve_paths

config = load_config('../../config.yaml')
paths = resolve_paths(config)

# Access paths
merged_file = paths['merged_trades_file']
base_data_dir = paths['base_data_dir']
```

### **data_loader.py**

Load trade and base data:

```python
from modules.data_loader import load_trades, validate_trade_data

# Load trades
trades_df = load_trades(config, paths)

# Validate data quality
validation = validate_trade_data(trades_df)
if not validation['valid']:
    print(f"Errors: {validation['errors']}")
```

---

## ✅ **What Makes These Scripts Generic?**

**They only need**:
- Trade CSV with entry/exit times, prices, P&L
- No knowledge of strategy logic
- No strategy-specific indicators

**They work with**:
- MSE strategy ✅
- Bollinger Bands strategy ✅
- SMA Crossover strategy ✅
- ANY future strategy ✅

**vs Strategy-Specific Scripts**:
- Located in `analysis/strategy_specific/mse/`
- Require MSE-specific logic (MACD thresholds, EMA spreads)
- Optimize MSE parameters only

---

## 🔄 **Workflow**

```
1. Run Backtest
   ↓
2. Merge Trades (utils/merge_trades.py)
   ↓
3. Run Generic Analysis
   ├─→ 01_basic_eda.py
   ├─→ 03_cascade_analysis.py
   ├─→ 04_ticker_ranking.py
   └─→ ... more scripts
   ↓
4. Review Reports (reports/)
   ↓
5. Portfolio Construction (../portfolio_construction/)
   ↓
6. Strategy Optimization (../strategy_specific/mse/) [optional]
```

---

## 🎯 **Best Practices**

### **1. Always Use Config File**
```powershell
# ✅ Good
python 01_basic_eda.py --config ../../config.yaml

# ❌ Bad
# (Editing hardcoded paths in script)
```

### **2. Sample for Quick Testing**
```powershell
# Test with 10,000 trades first
python 03_cascade_analysis.py --config ../../config.yaml --sample 10000

# Then run full analysis
python 03_cascade_analysis.py --config ../../config.yaml
```

### **3. Check Output Directories**
```powershell
# Results are saved to:
ls output/basic_eda/
ls output/cascade_analysis/
ls reports/
```

### **4. Version Control Your Config**
```powershell
# Save your config with descriptive name
cp config.yaml configs/mse_2022-2025_baseline.yaml
```

---

## 🐛 **Troubleshooting**

### **Issue: "Merged trades file not found"**

**Solution**:
```powershell
# Run merge script first
cd ..
python ../utils/merge_trades.py --config config.yaml
cd generic/scripts
```

### **Issue: "Config file not found"**

**Solution**:
```powershell
# Verify path is correct (relative to scripts/)
python 01_basic_eda.py --config ../../config.yaml

# Or use absolute path
python 01_basic_eda.py --config /full/path/to/config.yaml
```

### **Issue: "Module not found"**

**Solution**:
```powershell
# Run from scripts/ directory, not root
cd analysis/generic/scripts
python 01_basic_eda.py --config ../../config.yaml
```

---

## 📚 **Additional Resources**

- **Complete Workflow SOP**: `analysis/WORKFLOW_SOP.md`
- **Directory Structure**: `analysis/DIRECTORY_STRUCTURE.md`
- **Config Template**: `analysis/config_template.yaml`
- **Analysis Protocol**: `analysis/ANALYSIS_PROTOCOL.md`

---

## 🚧 **Migration Status**

**Phase 1 (Complete)**: ✅
- Directory structure created
- Config loader module
- Data loader module
- 01_basic_eda.py migrated
- 03_cascade_analysis.py migrated

**Phase 2 (In Progress)**: ⏳
- 02_trade_type_analysis.py
- 04_ticker_ranking.py
- 05_exit_timing.py
- 06_stop_loss_sim.py

**Phase 3 (Planned)**:
- Visualization module
- Metrics calculator module
- Report generator module

---

## 💡 **Tips**

1. **Run basic EDA first** - Gives you overall picture
2. **Cascade analysis next** - Identifies patterns for portfolio construction
3. **Ticker ranking** - Selects top performers
4. **Then portfolio construction** - Builds optimal combinations

**Happy Analyzing!** 📊
