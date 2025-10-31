# Analysis Framework Quick Start

5-minute guide to running the StrategyLab Analysis Framework

**Time to First Results**: 5-10 minutes
**Prerequisites**: Completed backtest with trade CSV files
**Output**: Performance metrics, portfolio recommendations, visualizations

---

## 🚀 Quick Start (3 Steps)

### Step 1: Run a Backtest (if you haven't already)

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Run a simple backtest to generate trade data
python src/runners/unified_runner.py \
    --mode backtest \
    --template conservative \
    --date-ranges 2024-01-01_to_2024-03-31 \
    --tickers RELIANCE TCS INFY \
    --strategies mse
```

**Output**: Trade CSV files in `outputs/{timestamp}/mse/`

---

### Step 2: Run Basic Analysis

```bash
# Navigate to analysis directory
cd analysis/generic/scripts

# Run basic EDA (win rate, profit factor, Sharpe)
python 01_basic_eda.py --trades-path ../../../outputs/{your_timestamp}/mse/*_StrategyTrades_*.csv

# Run ticker ranking (identify best performers)
python 05_ticker_ranking.py --trades-path ../../../outputs/{your_timestamp}/mse/*_StrategyTrades_*.csv
```

**Output**:
- `analysis/output/mse/{run_id}/generic/basic_eda/` - Performance metrics per ticker
- `analysis/output/mse/{run_id}/generic/ticker_ranking/` - Top 50 ranked tickers

---

### Step 3: Build Optimal Portfolio

```bash
# Navigate to portfolio construction
cd ../../portfolio_construction/scripts

# Run complete portfolio optimization pipeline
python 00_ticker_ranking.py
python 01_anti_cascade_filter.py
python 02_sector_classification.py
python 03_combination_generator.py
python 04_portfolio_optimizer.py
python 05_pypfopt_weights.py
python 06_equity_curves.py
```

**Output**:
- Best portfolio recommendation (5-10 tickers)
- Sharpe ratio, returns, drawdown
- Sector diversification analysis
- Equity curves and visualizations

**Example Result**:
```
Best Portfolio: AXISBANK, HCLTECH, INFY, SUNPHARMA, KOTAKBANK
Sharpe Ratio: 0.826
Annual Return: 3.37%
Max Drawdown: -4.88%
Diversification: Banking (40%), IT (40%), Pharma (20%)
```

---

## 📊 What You Get

### Generic Analysis (9 Scripts)

| Script | What It Does | Key Output | Time |
|--------|--------------|------------|------|
| **01_basic_eda** | Foundation statistics | Win rate, Sharpe, profit factor per ticker | 30s |
| **02_trade_type_analysis** | Long vs short bias | Directional performance analysis | 30s |
| **03_cascade_analysis** | Behavioral patterns | Cascade vs first-trade metrics | 1min |
| **04_stop_loss_simulation** | Optimize stop loss | Optimal SL threshold recommendation | 2min |
| **05_ticker_ranking** | Quality scoring | Top 50 tickers list | 1min |
| **06_risk_adjusted_patterns** | Risk-normalized perf | Pattern Sharpe ratios | 1min |
| **07_top50_vs_overall** | Validate selection | Statistical significance test | 30s |
| **08_top50_pattern_breakdown** | Winner profiling | Top 50 pattern prevalence | 30s |
| **09_validation_check** | Data integrity | Data quality score | 30s |

**Total Time**: ~8 minutes for complete generic analysis

---

### Portfolio Construction (7 Scripts)

| Script | What It Does | Key Output | Time |
|--------|--------------|------------|------|
| **00_ticker_ranking** | Comprehensive ranking | Top 50 (ALL, CASCADING, ANTI-CASCADING) | 1min |
| **01_anti_cascade_filter** | Remove bias | Filtered trade list | 1min |
| **02_sector_classification** | Diversification | Correlation matrix, sector map | 2min |
| **03_combination_generator** | Optimization space | 5,000+ valid combinations | 3min |
| **04_portfolio_optimizer** | Equal-weight eval | Top 50 portfolios by Sharpe | 5min |
| **05_pypfopt_weights** | Markowitz optimization | Optimal weights | 2min |
| **06_equity_curves** | Visual validation | Equity curves, drawdown charts | 2min |

**Total Time**: ~16 minutes for complete portfolio construction

---

## 🎯 Most Common Use Cases

### Use Case 1: "Which tickers work best for my strategy?"

**Solution**: Run ticker ranking

```bash
cd analysis/generic/scripts
python 05_ticker_ranking.py --trades-path {your_trades_path}
```

**Output**: `ticker_ranking_summary.csv` with top 50 tickers ranked by composite score

**Time**: 1 minute

---

### Use Case 2: "Build me a diversified portfolio"

**Solution**: Run full portfolio construction pipeline

```bash
cd analysis/portfolio_construction/scripts

# Run all scripts in sequence
for script in 00_*.py 01_*.py 02_*.py 03_*.py 04_*.py 05_*.py 06_*.py; do
    python $script
done
```

**Output**: `portfolio_optimization_summary.md` with best portfolio recommendation

**Time**: 15-20 minutes

---

### Use Case 3: "Optimize my stop loss threshold"

**Solution**: Run stop loss simulation

```bash
cd analysis/generic/scripts
python 04_stop_loss_simulation.py --trades-path {your_trades_path}
```

**Output**: Optimal SL threshold (e.g., "Use 2.5% stop loss for maximum Sharpe")

**Time**: 2 minutes

---

### Use Case 4: "Detect trading biases in my strategy"

**Solution**: Run cascade analysis

```bash
cd analysis/generic/scripts
python 03_cascade_analysis.py --trades-path {your_trades_path}
```

**Output**: Cascade vs first-trade performance comparison

**Time**: 1 minute

---

## 📁 Output Structure

After running analysis, outputs are organized as:

```
analysis/output/
└── {strategy}/              # e.g., mse
    └── {run_id}/            # e.g., 20251030_123456
        ├── generic/         # Generic analysis outputs
        │   ├── basic_eda/
        │   │   ├── summary_stats.csv
        │   │   └── ticker_performance.csv
        │   ├── ticker_ranking/
        │   │   ├── ticker_ranking_summary.csv
        │   │   └── top_50_tickers.csv
        │   └── ...
        └── portfolio/       # Portfolio construction outputs
            ├── ticker_ranking/
            ├── sector_classification/
            │   ├── correlation_matrix.csv
            │   └── sector_mapping.csv
            ├── portfolio_optimizer/
            │   ├── portfolio_optimization_summary.md  ⭐ READ THIS
            │   └── top_50_portfolios.csv
            └── equity_curves/
                └── best_portfolio_equity_curve.png
```

**Key File**: `portfolio_optimization_summary.md` - Your final portfolio recommendation

---

## ⚡ Express Mode (Single Command)

Want to run everything in one go?

### Option 1: Generic Analysis Only

```bash
cd analysis
python run_generic_analysis.py --trades-path ../outputs/{timestamp}/mse/
```

### Option 2: Full Pipeline (Generic + Portfolio)

```bash
cd analysis
python run_full_pipeline.py --trades-path ../outputs/{timestamp}/mse/
```

**Time**: 20-25 minutes for complete pipeline

---

## 🔧 Configuration

### Quick Config Template

Create `analysis/my_config.yaml`:

```yaml
# Input
trades_path: "../outputs/20251030_123456/mse/*_StrategyTrades_*.csv"
strategy: "mse"

# Generic Analysis Settings
min_trades: 10           # Minimum trades per ticker
sharpe_threshold: 0.5    # Minimum Sharpe to consider

# Portfolio Construction Settings
portfolio_size: 5        # Target portfolio size (5-10 tickers)
max_correlation: 0.7     # Max correlation between tickers
sector_limits:
  max_per_sector: 0.4    # Max 40% in any sector
  min_sectors: 3         # Minimum 3 sectors

# Output
output_dir: "analysis/output/mse/custom_run"
save_plots: true
```

**Usage**:
```bash
python 01_basic_eda.py --config my_config.yaml
```

---

## 📚 Next Steps

After getting your first results:

1. **Understand the Methodology**: Read `METHODOLOGY.md` (82 pages)
   - Why each analysis matters
   - Statistical foundations
   - Assumptions and limitations

2. **Review Results**: Read `PHASE1_RESULTS_SUMMARY.md` (35 pages)
   - Example results from Phase 1
   - Best portfolio identified (Sharpe 0.826)
   - Production readiness assessment

3. **Execute Full Workflow**: Follow `WORKFLOW_SOP.md` (60 pages)
   - Stage-by-stage execution guide
   - Troubleshooting tips
   - Production deployment checklist

4. **Deep Dive**: Read `DOCUMENTATION_INDEX.md`
   - Complete navigation guide
   - Document relationships
   - Learning paths

---

## 🐛 Troubleshooting

### "No trades found"

**Issue**: Script can't find trade CSV files

**Solution**:
```bash
# Check your trades path
ls outputs/{timestamp}/mse/*_StrategyTrades_*.csv

# Use absolute path or correct relative path
python 01_basic_eda.py --trades-path /full/path/to/trades
```

---

### "Not enough tickers for portfolio construction"

**Issue**: < 5 tickers meet minimum criteria

**Solution**:
```bash
# Lower minimum trade threshold
python 05_ticker_ranking.py --min-trades 5  # Default is 10

# Or use more tickers in backtest
python src/runners/unified_runner.py ... --tickers RELIANCE TCS INFY HDFCBANK ICICIBANK ...
```

---

### "Correlation matrix error"

**Issue**: Need at least 2 tickers for correlation

**Solution**:
- Run backtest with 5+ tickers
- Ensure tickers have overlapping date ranges

---

### "Memory error on large dataset"

**Issue**: Too many trades to process at once

**Solution**:
```bash
# Process in chunks
python 01_basic_eda.py --trades-path {path} --chunk-size 10000
```

---

## 💡 Pro Tips

### Tip 1: Start Small
Run analysis on 3-5 tickers first (5 min), then scale to 20-50 tickers (30 min)

### Tip 2: Use Wildcards
```bash
# Process all tickers from a backtest run
python 05_ticker_ranking.py --trades-path "../outputs/*/mse/*_StrategyTrades_*.csv"
```

### Tip 3: Chain Scripts
```bash
# Run generic analysis pipeline
cd analysis/generic/scripts
python 01_basic_eda.py && \
python 05_ticker_ranking.py && \
python 03_cascade_analysis.py
```

### Tip 4: Save Best Portfolios
Output includes `top_50_portfolios.csv` - test top 3-5 portfolios in live conditions

### Tip 5: Iterate on Results
- Start with Phase 1: Portfolio construction (done)
- Move to Phase 2: Strategy optimization (in progress)
- Optimize entry/exit thresholds based on portfolio results

---

## 📞 Getting Help

**Can't find your outputs?**
- Check `analysis/output/{strategy}/{run_id}/`
- Run with `--verbose` flag for detailed logs

**Analysis taking too long?**
- Run only essential scripts first (01, 05, 04)
- Use `--sample` flag to process subset of data

**Results seem wrong?**
- Verify trade CSV format (see `WORKFLOW_SOP.md`)
- Check for data quality issues (`09_validation_check.py`)
- Review assumptions in `METHODOLOGY.md`

**Want to understand the math?**
- Read `METHODOLOGY.md` - Statistical foundations section
- Check `DOCUMENTATION_INDEX.md` - Glossary

---

## 🎯 Summary

**5-Minute Workflow**:
1. Run backtest → Generate trades
2. Run `01_basic_eda.py` → Get performance metrics
3. Run `05_ticker_ranking.py` → Identify best tickers
4. Review outputs in `analysis/output/`

**20-Minute Full Pipeline**:
1. Run backtest with 10+ tickers
2. Execute all 9 generic scripts
3. Execute all 7 portfolio scripts
4. Review `portfolio_optimization_summary.md`

**Phase 1 Achievement**:
- **Best Portfolio**: AXISBANK, HCLTECH, INFY, SUNPHARMA, KOTAKBANK
- **Sharpe**: 0.826 (excellent)
- **Return**: 3.37% annually
- **Drawdown**: -4.88% (minimal)

---

**Ready to dive deeper?** → Start with `DOCUMENTATION_INDEX.md`

**Want complete methodology?** → Read `METHODOLOGY.md`

**Need step-by-step guide?** → Follow `WORKFLOW_SOP.md`

---

**Last Updated**: October 30, 2025
**Status**: Phase 1 Complete, Phase 2 In Progress
**Next**: Strategy optimization (exit thresholds, entry signals)
