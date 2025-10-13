# Portfolio Construction Pipeline

**Status**: Phase 1 - Day 1 Complete (Directory cleaned and organized)

## Overview

This pipeline optimizes portfolio construction using percentage-based returns and multi-size portfolio analysis. The system identifies optimal ticker combinations through systematic filtering, sector diversification, and correlation analysis.

## Directory Structure

```
portfolio_construction/
├── scripts/              # Core pipeline scripts (run in sequence 00→04)
├── data/
│   ├── foundation/      # Script 0: Top 50 performer lists
│   ├── filtered/        # Script 1: Affordable tickers (<₹2,000)
│   ├── classified/      # Script 2: Sector mapping & correlations
│   ├── combinations/    # Script 3: Valid portfolio combinations
│   └── results/         # Script 4: Portfolio performance metrics
├── backups/             # Version history (_v1 backups from absolute→percentage conversion)
├── docs/                # Implementation roadmap & analysis reports
├── logs/                # Execution logs from pipeline runs
└── utils/               # Helper scripts (comparison, feasibility analysis)
```

## Pipeline Flow

### Script 0: Foundation Analysis
**File**: `scripts/00_foundation_cascade_vs_anticascade_analysis.py`
**Input**: Raw trade data (1.15M trades)
**Output**: `data/foundation/TOP50_ANTICASCADING_TRADES.csv`
**Logic**:
- Calculate percentage returns (not absolute currency)
- Rank by weighted score: PF (40%) + WR (30%) + Avg Return (30%)
- Filter anti-cascading trades only

### Script 1: Affordability Filter
**File**: `scripts/01_corrected_anti_cascading_subset.py`
**Input**: `data/foundation/TOP50_ANTICASCADING_TRADES.csv`
**Output**: `data/filtered/CORRECTED_affordable_tickers_metadata.csv` (28 tickers)
**Logic**: Keep tickers with Entry Price < ₹2,000

### Script 2: Sector Classification
**File**: `scripts/02_corrected_sector_classification_correlation.py`
**Input**: `data/filtered/CORRECTED_anti_cascading_trades_under2k.csv`
**Output**: `data/classified/CORRECTED_sector_mapping.csv`, `CORRECTED_correlation_matrix.csv`
**Logic**: Map tickers to sectors, calculate pairwise correlations

### Script 3: Combination Generation
**File**: `scripts/03_corrected_intelligent_combination_generation.py`
**Input**: `data/classified/CORRECTED_sector_mapping.csv`, `CORRECTED_correlation_matrix.csv`
**Output**: `data/combinations/PERCENTAGE_valid_combinations_{N}ticker.csv`
**Logic**:
- Generate all N-ticker combinations (N=4,5,6,7,8)
- Filter: Max 60% sector concentration, Max 0.7 correlation
- **No individual ticker filtering** (already Top 50)

### Script 4: Portfolio Optimization
**File**: `scripts/04_portfolio_optimization_engine.py`
**Input**: `data/combinations/PERCENTAGE_valid_combinations_{N}ticker.csv`
**Output**: `data/results/portfolio_performance_{N}ticker_TOP50.csv`
**Logic**:
- Calculate portfolio-level Sharpe ratio
- Test 10,000 random portfolios per size
- Rank by Sharpe, return, drawdown

### Master Optimizer
**File**: `scripts/master_portfolio_optimizer.py`
**Purpose**: Orchestrate Scripts 3-4 across multiple portfolio sizes
**Output**: `data/results/portfolio_size_comparison_report.csv`
**Result**: **6-7 tickers optimal** (Sharpe 1.81, best consistency)

## Key Findings

### Performance Metrics (Percentage-Based Returns)
- **Best 5-ticker**: Sharpe 1.790, Avg Return 0.084%, Max DD -0.58%
- **Best 6-ticker**: Sharpe 1.811, Avg Return 0.078%, Max DD -0.52% ✅ **OPTIMAL**
- **Best 7-ticker**: Sharpe 1.811, Avg Return 0.077%, Max DD -0.52%

### Top 50 Tickers (Percentage Returns vs Absolute)
- **Only 8% overlap** between absolute and percentage-based Top 50
- Percentage returns reveal **capital efficiency**, not just profit size
- Example: Small capital + high % return > Large capital + low % return

### Critical Fix: Absolute → Percentage Conversion
**Problem**: Script 0 used absolute currency (showing PF 1.8-2.3) while Script 4 used percentage returns (showing PF 0.7-0.99), causing all portfolios to appear NEGATIVE.
**Solution**: Converted entire pipeline to percentage returns. All portfolios now show POSITIVE Sharpe ratios (1.78-1.81).

## Current Status

**Phase 1 - Portfolio Construction Finalization**
- ✅ Day 1: Directory cleaned and organized
- ⏳ Day 1 (Next): Install PyPortfolioOpt, implement optimal weight allocation
- Days 2-3: Complete PyPortfolioOpt integration, compare equal-weight vs optimized
- Days 4-5: Generate equity curves for top portfolios
- Day 6: Documentation

**Phase 2 - MSE Strategy Fine-Tuning** (Weeks 2-6)
- Analyze MSE strategy entry/exit logic using base data files
- Optimize 80% MACD histogram exit threshold → expected 85% optimal
- Test entry filters: MACD strength >0.3, EMA spread >0.5%
- Walk-forward validation

## Dependencies

See `requirements.txt` for full list. Key libraries:
- pandas, numpy (data processing)
- PyPortfolioOpt (optimal weight allocation) - **To be installed**
- matplotlib (equity curve visualization)

## Usage

### Run Full Pipeline
```bash
python scripts/00_foundation_cascade_vs_anticascade_analysis.py
python scripts/01_corrected_anti_cascading_subset.py
python scripts/02_corrected_sector_classification_correlation.py
python scripts/master_portfolio_optimizer.py  # Runs Scripts 3-4 for all sizes
```

### Run Single Portfolio Size
```bash
python scripts/03_corrected_intelligent_combination_generation.py  # Edit N_TICKERS variable
python scripts/04_portfolio_optimization_engine.py
```

## Notes

- **Always use percentage returns** for scale-independent comparison
- **No additional filtering in Script 3** - tickers already pre-selected as Top 50
- **Sharpe calculation**: (Portfolio Return - Risk-free Rate) / Portfolio Volatility
- **Diversification filters**: Max 60% sector concentration, Max 0.7 pairwise correlation
- **Backup files**: All _v1.py files are backups from absolute→percentage conversion

## References

- Implementation Roadmap: `docs/IMPLEMENTATION_ROADMAP.md`
- Portfolio Analysis Report: `docs/PORTFOLIO_ANALYSIS_REPORT.md`
- Strategy Analysis: `docs/STRATEGY_VS_PORTFOLIO_ANALYSIS.md`
