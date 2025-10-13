# Analysis Directory Structure
## Post-Reorganization Architecture

```
analysis/
├── config_template.yaml           # Template for new analysis runs
├── config.yaml                    # Your current analysis config (gitignored)
├── config_with_paths.yaml         # Auto-generated after merge (gitignored)
├── DIRECTORY_STRUCTURE.md         # This file
├── ANALYSIS_PROTOCOL.md           # Existing - Analysis methodology
├── WORKFLOW_SOP.md                # NEW - Complete end-to-end workflow
│
├── generic/                       # ✅ GENERIC - Works with ANY strategy
│   ├── README.md                  # How to use generic analysis
│   ├── scripts/
│   │   ├── 01_basic_eda.py        # Basic statistics
│   │   ├── 02_trade_type_analysis.py  # Buy vs Sell
│   │   ├── 03_cascade_analysis.py     # Sequential patterns
│   │   ├── 04_ticker_ranking.py       # Performance ranking
│   │   ├── 05_exit_timing.py          # Holding period optimization
│   │   └── 06_stop_loss_sim.py        # Stop loss simulation
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── config_loader.py       # Load YAML config
│   │   ├── data_loader.py         # Load trades from config
│   │   ├── cascade_tagger.py      # Cascade identification logic
│   │   ├── metrics_calculator.py  # Generic metrics
│   │   └── visualizer.py          # Generic charts
│   ├── output/                    # Generated analysis files (gitignored)
│   └── reports/                   # Generated reports (gitignored)
│
├── portfolio_construction/        # ✅ GENERIC - Works with ANY strategy
│   ├── README.md                  # Existing
│   ├── scripts/
│   │   ├── 00_foundation.py       # Top performer selection
│   │   ├── 01_affordability.py    # Price filtering
│   │   ├── 02_sector_classification.py  # Diversification
│   │   ├── 03_combinations.py     # Generate portfolios
│   │   ├── 04_optimization.py     # Sharpe optimization
│   │   ├── 05_pypfopt_weights.py  # Optimal weights
│   │   ├── 06_equity_curve.py     # Backtest portfolios
│   │   └── master_optimizer.py    # Full pipeline
│   ├── utils/
│   │   ├── compute_feasibility.py # Existing utility
│   │   └── sector_mapper.py       # Sector classification helper
│   ├── data/                      # Portfolio analysis data
│   │   ├── foundation/
│   │   ├── filtered/
│   │   ├── classified/
│   │   ├── combinations/
│   │   └── results/
│   ├── docs/                      # Documentation
│   └── logs/                      # Execution logs
│
├── strategy_specific/             # 🎯 STRATEGY-SPECIFIC optimizations
│   ├── mse/                       # MSE strategy optimization
│   │   ├── README.md              # MSE optimization guide
│   │   ├── STAGE6_EXECUTION_GUIDE.md
│   │   ├── scripts/
│   │   │   ├── 00_setup_verification.py
│   │   │   ├── 01_baseline_calculator.py
│   │   │   ├── 02_exit_threshold_optimizer.py
│   │   │   ├── 03_walk_forward_validation.py
│   │   │   ├── 04_statistical_validation.py
│   │   │   └── ...
│   │   ├── modules/
│   │   │   ├── mae_mfe_calculator.py   # ✅ Actually generic!
│   │   │   ├── exit_simulator.py       # ✅ Actually generic!
│   │   │   ├── trade_enhancer.py       # ✅ Generic (moved to integration/)
│   │   │   ├── metrics_calculator.py   # ✅ Generic
│   │   │   └── visualizer.py           # ✅ Generic
│   │   ├── checkpoints/
│   │   ├── config/
│   │   ├── data/
│   │   ├── docs/
│   │   └── logs/
│   │
│   └── [future_strategy]/         # Template for other strategies
│       └── (same structure as mse/)
│
├── integration/                   # ✅ GENERIC - Indicator-level enhancement
│   ├── README.md                  # Existing - already perfect!
│   ├── __init__.py                # Clean API
│   └── core/
│       └── trade_enhancer.py      # Links trades to base_data
│
├── comparative_analysis/          # ✅ GENERIC - Compare strategies/tickers
│   ├── scripts/
│   │   ├── 01_high_performers_vs_universe.py
│   │   ├── 02_quick_comparative.py
│   │   └── 03_volume_weighted.py
│   └── reports/
│
└── mse_analysis/                  # 🔄 LEGACY - To be migrated
    ├── README.md
    ├── scripts/                   # These will move to generic/ or strategy_specific/
    │   ├── 01_basic_eda.py        → Move to generic/scripts/
    │   ├── 02_trade_type_analysis.py → Move to generic/scripts/
    │   ├── 03_stop_loss_simulation.py → Move to generic/scripts/
    │   ├── 04_exit_timing_analysis.py → Move to generic/scripts/
    │   ├── 05_ticker_ranking.py       → Move to generic/scripts/
    │   ├── 07_macd_exit_threshold_optimization.py → Move to strategy_specific/mse/
    │   ├── 08_validation_check_analysis.py
    │   ├── 10_cascade_tagging_analysis.py → Move to generic/scripts/
    │   ├── 13_corrected_cascade_analysis.py → Merge with 10
    │   ├── 14_risk_adjusted_pattern_analysis.py → Move to generic/scripts/
    │   ├── 15_comprehensive_top50_vs_overall_analysis.py → Move to comparative_analysis/
    │   ├── 16_top50_detailed_pattern_breakdown.py → Move to portfolio_construction/
    │   ├── 17_top50_optimal_portfolio_design.py → Move to portfolio_construction/
    │   ├── 18_portfolio_data_kitchen_prep.py → Move to portfolio_construction/
    │   ├── 19_corrected_portfolio_prep.py → Move to portfolio_construction/
    │   └── 20_portfolio_combinations_testing.py → Move to portfolio_construction/
    ├── comparative_analysis/      → Already moved above
    ├── output/
    └── reports/
```

---

## **File Classification**

### ✅ **GENERIC** (Strategy-Agnostic)
**Works with ANY trade CSV - only needs columns:**
- `ticker`, `Entry Time`, `Exit Time`, `Profit (Currency)`, `Trade Type`

**Scripts:**
- `01_basic_eda.py`
- `02_trade_type_analysis.py`
- `03_stop_loss_simulation.py` (simulates, doesn't need strategy logic)
- `04_exit_timing_analysis.py`
- `05_ticker_ranking.py`
- `10_cascade_tagging_analysis.py`
- `13_corrected_cascade_analysis.py`
- `14_risk_adjusted_pattern_analysis.py`
- All portfolio_construction scripts
- All comparative_analysis scripts

**Why Generic?**
- Only analyze trade outcomes (entry/exit/profit)
- Don't care about HOW the strategy generated signals
- Can be used for: MSE, Bollinger Bands, SMA, any future strategy

---

### 🎯 **STRATEGY-SPECIFIC** (MSE Only)
**Requires MSE logic or MSE-specific indicators (MACD, EMA):**

**Scripts:**
- `07_macd_exit_threshold_optimization.py` (tests 80% → 85% MACD exit)
- Strategy optimization scripts in `strategy_optimization/`

**Why Specific?**
- Optimize MSE parameters (MACD thresholds, EMA spreads)
- Require knowledge of MSE entry/exit rules
- Cannot be used for other strategies without modification

---

## **Migration Plan**

### **Phase 1: Create New Structure**
```bash
mkdir -p analysis/generic/{scripts,modules,output,reports}
mkdir -p analysis/strategy_specific/mse
mkdir -p analysis/comparative_analysis/scripts
```

### **Phase 2: Move Generic Scripts**
```bash
# Move to generic/scripts/
cp analysis/mse_analysis/scripts/01_basic_eda.py analysis/generic/scripts/
cp analysis/mse_analysis/scripts/02_trade_type_analysis.py analysis/generic/scripts/
cp analysis/mse_analysis/scripts/10_cascade_tagging_analysis.py analysis/generic/scripts/03_cascade_analysis.py
cp analysis/mse_analysis/scripts/05_ticker_ranking.py analysis/generic/scripts/04_ticker_ranking.py
# ... etc
```

### **Phase 3: Update Scripts to Use YAML Config**
For each script, replace hardcoded paths with:

```python
import yaml
import argparse

def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    # Use config paths
    merged_trades = config['paths']['merged_trades_file']
    base_data_dir = config['paths']['base_data_dir']

    # ... rest of analysis
```

### **Phase 4: Move Strategy-Specific**
```bash
mv analysis/strategy_optimization analysis/strategy_specific/mse/
```

### **Phase 5: Test Workflow**
1. Run backtest
2. Merge trades using config
3. Run generic analysis
4. Run portfolio construction
5. Verify all outputs

### **Phase 6: Deprecate mse_analysis/**
Once all scripts are migrated and tested:
```bash
mv analysis/mse_analysis analysis/_LEGACY_mse_analysis
# Keep for reference but don't use
```

---

## **Benefits of New Structure**

1. **Clear Separation**: Generic vs Strategy-specific
2. **Reusable**: Run same analysis on any strategy
3. **Config-Driven**: No more hardcoded paths
4. **Scalable**: Easy to add new strategies
5. **Maintainable**: Each category has clear purpose
6. **Documented**: README in each directory explains usage

---

## **Next Steps**

1. ✅ Cleanup complete (Phase 1 done)
2. ✅ Enhanced merge_trades.py created
3. ✅ YAML config template created
4. ⏳ Create directory structure (run Phase 1 commands)
5. ⏳ Migrate & update scripts (Phase 2-3)
6. ⏳ Create WORKFLOW_SOP.md
7. ⏳ Test complete workflow

---

**Status**: Directory structure designed, ready for implementation.
**Next Action**: Create folders and start migrating scripts with YAML config support.
