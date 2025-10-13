# Options Backtesting Module

**Status**: Planning & Development Phase
**Version**: 0.1.0 (Pre-MVP)
**Last Updated**: 2025-10-08

---

## Overview

This module extends the existing equity backtesting system to support **options trading strategies**. The primary goal is to validate whether equity trading signals can be effectively executed using options contracts, and to compare performance, risk, and capital efficiency.

### Core Capabilities

- **Replay Engine**: Re-execute equity trades using options contracts
- **Dual Pricing**: Support both synthetic (Black-Scholes) and actual historical option prices
- **Strike/Expiry Selection**: Configurable logic for contract selection
- **Position Tracking**: Bar-by-bar tracking with Greeks calculation
- **Validation Framework**: Measure synthetic pricing accuracy vs market reality
- **Comparison Analytics**: Equity vs Options performance reports

---

## Quick Start

### Prerequisites

1. **Existing Equity Backtest Results**:
   - `trades.csv` - Equity trade ledger
   - `base_data.csv` - Bar-by-bar underlying price data

2. **Upstox API Access** (for actual historical data):
   - Active Upstox Plus subscription
   - API credentials configured in `config/config.py`

3. **Python Dependencies**:
   ```bash
   pip install pandas numpy scipy pyyaml tqdm
   ```

### Phase 1: Data Validation (Current Phase)

**Objective**: Measure synthetic pricing accuracy

```bash
# 1. Configure validation parameters
# Edit: src/core/options/validation/validation_config.yaml

# 2. Run validation (script to be implemented)
python src/core/options/validation/validation_runner.py

# 3. Review results
cat src/core/options/data/validation_results/pricing_validation_summary.csv
```

**Expected Output**:
- Pricing error metrics by ticker, model, moneyness, DTE
- Recommendation on which pricing mode to use (synthetic/actual/hybrid)

### Phase 2: MVP Replay (Next Phase)

**Objective**: Run first options backtest on small dataset

```bash
# 1. Configure options parameters
# Edit: src/core/options/config/options_config.yaml

# 2. Run replay engine (script to be implemented)
python src/core/options/replay/replay_runner.py \
  --equity-trades outputs/20240101_120000/trades.csv \
  --base-data outputs/20240101_120000/base_data.csv \
  --ticker RELIANCE \
  --output-dir outputs/options_mvp

# 3. Review outputs
ls outputs/options_mvp/
# - options_trades.csv
# - options_base_data.csv
# - options_metrics.json
```

---

## Directory Structure

```
src/core/options/
├── README.md                       # This file
├── options_engine.py               # EXISTING: Black-Scholes pricing + Greeks
│
├── planning/
│   ├── implementation_plan.md      # Comprehensive planning document
│   └── decisions.md                # Architecture decision records (TBD)
│
├── validation/
│   ├── __init__.py                 # (To be created)
│   ├── validation_config.yaml      # Validation parameters
│   ├── pricing_validator.py        # Synthetic vs actual comparison (TBD)
│   ├── data_fetcher.py             # Fetch historical options from Upstox (TBD)
│   └── validation_runner.py        # Main validation orchestrator (TBD)
│
├── pricing/
│   ├── __init__.py                 # (To be created)
│   ├── synthetic_engine.py         # Black-Scholes implementations (TBD)
│   ├── actual_engine.py            # Historical data pricing (TBD)
│   ├── hybrid_engine.py            # Combined approach (TBD)
│   └── volatility_models.py        # Historical vol, Parkinson, EWMA (TBD)
│
├── replay/
│   ├── __init__.py                 # (To be created)
│   ├── trade_mapper.py             # Equity → option contract mapping (TBD)
│   ├── position_tracker.py         # Bar-by-bar position tracking (TBD)
│   ├── metrics_calculator.py       # Performance metrics (TBD)
│   └── replay_runner.py            # Main orchestrator (TBD)
│
├── data/
│   ├── cache/                      # Cached options data (parquet)
│   ├── validation_results/         # Validation reports
│   ├── schemas.py                  # Data schemas (TBD)
│   └── lot_sizes.csv               # Ticker → lot size mapping
│
└── config/
    └── options_config.yaml         # Options backtesting configuration
```

**Legend**:
- ✅ **EXISTING**: Already implemented
- 📄 **Created**: File exists (config/docs)
- ⏳ **TBD**: To be developed in upcoming phases

---

## Key Concepts

### 1. Pricing Modes

**Synthetic Mode**:
- Uses Black-Scholes model with historical volatility
- Fast, works for any time period
- Approximate (may not reflect market reality)
- Best for: Initial strategy validation

**Actual Mode**:
- Uses real historical option OHLC from Upstox
- Accurate, reflects true market conditions
- Limited to data availability (~6 months)
- Best for: Final validation before live trading

**Hybrid Mode** (Recommended):
- Uses actual data when available
- Falls back to synthetic for gaps
- Balances accuracy and coverage
- Best for: Production backtesting

### 2. Strike Selection Methods

**ATM (At-The-Money)**:
- Strike closest to underlying price
- Moderate delta (~0.50), balanced theta/gamma
- Most liquid, tightest spreads

**Delta-Based**:
- Select strike with specific delta (e.g., 0.30)
- Allows OTM/ITM targeting
- Delta changes with underlying movement

**Moneyness**:
- Strike as % of underlying (e.g., 105% = 5% OTM call)
- Fixed distance from price
- Simple, deterministic

**Premium %**:
- Option costs specific % of underlying (e.g., 2%)
- Capital-based selection
- Varies by volatility regime

### 3. Lot Sizing Strategies

**Fixed Lots**:
- Always trade N lots (e.g., 1 lot)
- Simplest, clean comparisons
- Capital deployed varies by option premium

**Capital Matching**:
- Deploy same ₹ as equity trade
- Lots = equity_capital / (option_premium × lot_size)
- May result in fractional lots (round to integer)

**Delta Matching**:
- Match notional underlying exposure
- Accounts for delta < 1.0
- Most accurate risk replication

### 4. Greeks

**Delta**: Sensitivity to underlying price change (0 to 1 for calls)
**Gamma**: Rate of delta change
**Theta**: Time decay (typically negative for long positions)
**Vega**: Sensitivity to volatility changes
**Rho**: Sensitivity to interest rate (usually negligible for short-dated)

---

## Configuration

### Primary Config: `config/options_config.yaml`

**Key Parameters**:

```yaml
pricing:
  mode: "hybrid"  # synthetic | actual | hybrid

strike_selection:
  method: "atm"  # atm | delta | moneyness | premium_pct

expiry_selection:
  method: "nearest_weekly"  # nearest_weekly | nearest_monthly | fixed_dte

lot_sizing:
  method: "fixed"  # fixed | capital_match | delta_match
  fixed:
    lots_per_trade: 1

position_management:
  entry:
    min_dte_to_enter: 3
  exit:
    follow_equity_signal: true
    force_close_before_expiry:
      enabled: true
      hours_before: 24
```

See file for full documentation of all parameters.

### Validation Config: `validation/validation_config.yaml`

**Key Parameters**:

```yaml
validation:
  tickers:
    - "NIFTY"
    - "BANKNIFTY"
    - "RELIANCE"
    - "TCS"
    - "INFY"
  time_period:
    months: 6

synthetic_models:
  bs_hist_20d:
    enabled: true
  bs_calibrated_iv:
    enabled: true

decision_criteria:
  thresholds:
    good:
      atm_median_error: 0.10
```

---

## Data Schemas

### Options Trade Ledger (`options_trades.csv`)

| Column | Type | Description |
|--------|------|-------------|
| trade_id | int | Unique trade identifier |
| underlying | str | Ticker symbol (e.g., 'RELIANCE') |
| option_type | str | 'CE' (call) or 'PE' (put) |
| strike | float | Strike price |
| expiry | datetime | Expiry date |
| entry_time | datetime | Entry timestamp |
| exit_time | datetime | Exit timestamp |
| entry_price | float | Option premium at entry |
| exit_price | float | Option premium at exit |
| lots | int | Number of lots traded |
| lot_size | int | Contracts per lot |
| quantity | int | Total contracts (lots × lot_size) |
| entry_cost | float | Total capital deployed (₹) |
| exit_value | float | Total exit value (₹) |
| realized_pnl | float | Profit/Loss (₹) |
| realized_pnl_pct | float | P&L % |
| hold_time_hours | float | Duration of position |
| dte_at_entry | float | Days to expiry at entry |
| dte_at_exit | float | Days to expiry at exit |
| pricing_mode | str | 'synthetic' | 'actual' | 'hybrid' |
| exit_reason | str | Why position closed |

### Position Lifecycle (`options_base_data.csv`)

| Column | Type | Description |
|--------|------|-------------|
| trade_id | int | Links to options_trades.csv |
| timestamp | datetime | Bar timestamp |
| underlying_price | float | Underlying asset price |
| option_price | float | Option premium |
| position_value | float | Mark-to-market value (₹) |
| unrealized_pnl | float | Unrealized P&L (₹) |
| unrealized_pnl_pct | float | Unrealized P&L % |
| dte_remaining | float | Days to expiry remaining |
| delta | float | Option delta |
| gamma | float | Option gamma |
| theta | float | Option theta |
| vega | float | Option vega |
| pricing_mode | str | Which pricing mode used |

### Metrics Output (`options_metrics.json`)

```json
{
  "summary": {
    "total_trades": 1523,
    "win_rate": 0.645,
    "total_pnl": 2458300,
    "sharpe_ratio": 1.87,
    "max_drawdown_pct": -18.4,
    ...
  },
  "by_ticker": { ... },
  "options_specific": {
    "theta_capture_total": 125600,
    "gamma_pnl_total": 342100,
    ...
  }
}
```

---

## Implementation Phases

### Phase 0: Preparation ✅
- ✅ Directory structure
- ✅ Planning documentation
- ✅ Configuration templates
- ✅ Lot size reference data

### Phase 1: Data Validation ⏳ (Current)
**Timeline**: Weeks 2-3

**Tasks**:
1. Implement data fetcher (Upstox API)
2. Implement 4 synthetic pricing models
3. Run validation experiment (6 months, 5 tickers)
4. Generate validation report
5. **Decision**: Choose pricing mode

**Deliverables**:
- `pricing_validation_summary.csv`
- `validation_metrics.json`
- Recommendation document

### Phase 2: MVP Replay Engine ⏳
**Timeline**: Weeks 4-5

**Tasks**:
1. Implement Trade Mapper
2. Implement Position Tracker
3. Implement Metrics Calculator
4. Test on 1 ticker, 1 month
5. Debug and validate

**Deliverables**:
- Working replay engine
- Test results on small dataset

### Phase 3: Actual Data Integration ⏳
**Timeline**: Weeks 6-7

**Tasks**:
1. Add actual pricing mode
2. Implement hybrid mode
3. Re-run MVP with all pricing modes
4. Compare synthetic vs actual results
5. **Decision**: Hybrid viable?

### Phase 4: Full Backtest ⏳
**Timeline**: Weeks 8-10

**Tasks**:
1. Scale to all 5 tickers, 6 months
2. Add configuration support
3. Run sensitivity analysis
4. Generate comprehensive reports

**Deliverables**:
- Full backtest results
- Equity vs Options comparison
- Sensitivity analysis
- Recommendations for live trading

---

## Open Questions

Critical decisions to be made during implementation:

1. **Capital Allocation**: How to match equity position sizes to option lots?
2. **Exit Logic**: Follow equity exits only, or add options-specific risk management?
3. **Option Type for SHORT**: Buy puts vs sell calls vs spreads?
4. **Data Fetching**: Pre-fetch all or on-demand?
5. **Fractional Lots**: How to handle rounding?
6. **Fill Prices**: Open/close/mid/VWAP?
7. **Liquidity Filters**: When to skip illiquid contracts?
8. **Volatility Regimes**: Adjust strategy based on vol?
9. **Roll Logic**: Handle expiries or let positions close?
10. **Greeks Frequency**: Calculate every bar or optimize?

See `planning/implementation_plan.md` for detailed discussion and current decisions.

---

## Expected Outputs

### Validation Phase
- **File**: `data/validation_results/pricing_validation_summary.csv`
- **Purpose**: Determine synthetic pricing accuracy
- **Columns**: Ticker, Model, Moneyness, DTE_Range, Error_Metrics

### Backtest Phase
- **File**: `outputs/{run_id}/options_trades.csv` - Trade ledger
- **File**: `outputs/{run_id}/options_base_data.csv` - Position lifecycle
- **File**: `outputs/{run_id}/options_metrics.json` - Performance stats
- **File**: `outputs/{run_id}/comparison_equity_vs_options.csv` - Side-by-side
- **File**: `outputs/{run_id}/sensitivity_analysis.csv` - Parameter impact

---

## Testing & Validation

### Unit Tests (To Be Implemented)
- Black-Scholes calculations (exact values)
- Greeks calculations (known test cases)
- Strike selection logic (edge cases)
- Lot size rounding (fractional handling)

### Integration Tests
- End-to-end replay on small dataset
- Validation of P&L calculations
- Expiry handling (positions auto-close)
- Data quality checks

### Validation Checks
- ✅ No negative option prices
- ✅ No positions held past expiry
- ✅ P&L balances (unrealized → realized)
- ✅ Greeks within bounds (delta 0-1, etc.)
- ✅ No lookahead bias

---

## Dependencies

### External APIs
- **Upstox API**: Historical options data
  - Endpoints: `/v2/expired-instruments/*`
  - Auth: OAuth2 (managed in main config)
  - Rate limits: 5 req/sec, 100 req/min

### Internal Dependencies
- **Equity Backtester**: Provides `trades.csv`, `base_data.csv`
- **Data Provider Layer**: Reuse existing Upstox integration
- **Config System**: Extend existing YAML-based config

### Python Libraries
- `pandas`, `numpy`: Data processing
- `scipy`: Black-Scholes (normal distribution)
- `pyyaml`: Config parsing
- `tqdm`: Progress bars
- `matplotlib`/`seaborn`: Visualization (Phase 4)

---

## Performance Considerations

### Data Volume
- **Validation**: ~300K data points (5 tickers × 6 months × 20 strikes × 125 days)
- **Full Backtest**: ~1M data points (1,500 trades × bar-by-bar tracking)

### Optimization Strategies
- ✅ Parquet format (compressed, columnar)
- ✅ Vectorized pandas operations (no Python loops)
- ✅ Caching (vol calculations, Greeks)
- ⏳ Parallel processing (multi-ticker)
- ⏳ Chunked processing (avoid OOM)

### Memory Management
- Process tickers sequentially (for large datasets)
- Clear intermediate DataFrames
- Use generators for large file reads

---

## Contributing

### Code Style
- **Files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions**: `snake_case()`
- **Type hints**: All public functions
- **Docstrings**: Google style

### Development Workflow
1. Create feature branch: `git checkout -b feature/pricing-engine`
2. Implement with tests
3. Run validation checks
4. Update documentation
5. Submit for review

### Documentation
- Update `README.md` when adding features
- Document architectural decisions in `planning/decisions.md`
- Add inline comments for complex logic
- Keep `implementation_plan.md` in sync

---

## Troubleshooting

### Common Issues

**Issue**: Upstox API rate limit exceeded
**Solution**: Reduce `requests_per_second` in `validation_config.yaml`, enable caching

**Issue**: Options data not available for ticker/date
**Solution**: Check Upstox Plus subscription, verify ticker has listed options, try different date range

**Issue**: Synthetic prices wildly different from actual
**Solution**: Review volatility model choice, check for data quality issues in underlying, consider using calibrated IV model

**Issue**: Positions held past expiry
**Solution**: Verify `force_close_before_expiry` is enabled, check expiry date format, ensure DTE calculation is correct

**Issue**: P&L doesn't balance (unrealized ≠ realized at exit)
**Solution**: Check for missing bars in `base_data.csv`, verify option pricing at exit timestamp, review rounding errors

---

## Resources

### Internal Documentation
- `planning/implementation_plan.md` - Comprehensive planning doc
- `planning/decisions.md` - Architecture decisions (TBD)
- `config/options_config.yaml` - Full config reference
- `validation/validation_config.yaml` - Validation parameters

### External References
- [Upstox API Docs](https://upstox.com/developer/api-documentation/)
- [Black-Scholes Model](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model)
- [Options Greeks Explained](https://www.investopedia.com/trading/using-the-greeks-to-understand-options/)
- [Parkinson Volatility](https://en.wikipedia.org/wiki/Volatility_(finance)#Parkinson_volatility)

---

## License

Part of StrategyLab proprietary codebase. All rights reserved.

---

## Changelog

### v0.1.0 (2025-10-08)
- Initial planning and setup
- Directory structure created
- Configuration templates defined
- Implementation plan documented

---

## Contact

For questions or issues related to options backtesting:
- Review `planning/implementation_plan.md` first
- Check open questions in this README
- Consult with team lead

---

**Last Updated**: 2025-10-08
**Next Milestone**: Phase 1 (Data Validation) completion
