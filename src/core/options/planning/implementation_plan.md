# Options Backtesting Implementation Plan

**Version**: 1.0
**Date**: 2025-10-08
**Status**: Planning Phase
**Owner**: StrategyLab Team

---

## Executive Summary

This document outlines the end-to-end implementation plan for adding options backtesting capabilities to the existing equity trading system. The primary goal is to **validate if equity trading signals can be effectively executed using options instruments**, comparing performance, risk, and capital efficiency.

### Core Objectives

1. **Validate Synthetic Pricing**: Determine if Black-Scholes-based synthetic option pricing is accurate enough for backtesting, or if actual historical options data is required
2. **Build Replay Infrastructure**: Create a decoupled engine that takes equity trades and "replays" them using options contracts
3. **Compare Performance**: Generate empirical evidence showing when/where options execution outperforms equity execution
4. **Prepare for Live Trading**: Establish a foundation that can later be extended to live options trading

### Guiding Principles

- **Validation First**: Don't build strategy logic until pricing foundation is validated
- **Decoupled Design**: Options engine consumes equity outputs without modifying equity code
- **Start Simple, Add Complexity**: Begin with fixed parameters (1 lot, ATM, weekly expiry), expand incrementally
- **Empirical Evidence**: Every decision backed by data (synthetic vs actual pricing errors, P&L comparisons)
- **Production-Ready**: All code should be reproducible, auditable, and extensible to live trading

---

## Architecture Overview

### System Layers

```
┌─────────────────────────────────────────────────────────┐
│  EXISTING EQUITY SYSTEM (UNCHANGED)                     │
│  ├─ Strategy signals on underlying data                  │
│  ├─ Backtester execution                                 │
│  └─ Outputs: trades.csv + base_data.csv                  │
└────────────────┬────────────────────────────────────────┘
                 │
                 │ Input: Equity trade signals
                 ▼
┌─────────────────────────────────────────────────────────┐
│  OPTIONS REPLAY ENGINE (NEW)                            │
│                                                         │
│  ┌────────────────────────────────────────────────┐     │
│  │ 1. TRADE MAPPER                                │     │
│  │    - Reads equity trades from trades.csv       │     │
│  │    - Maps to option contracts:                 │     │
│  │      * Strike selection (ATM, delta, moneyness)│     │
│  │      * Expiry selection (weekly, monthly, DTE) │     │
│  │      * Lot size calculation                    │     │
│  │    - Validates contract availability           │     │
│  └────────────────────────────────────────────────┘     │
│                                                         │
│  ┌────────────────────────────────────────────────┐     │
│  │ 2. PRICING ENGINE (DUAL MODE)                  │     │
│  │                                                │     │
│  │    MODE A: SYNTHETIC                           │     │
│  │    - Black-Scholes (multiple vol models)       │     │
│  │    - Historical vol (20-day, 5-day, Parkinson) │     │
│  │    - Calibrated IV (fitted to ATM actuals)     │     │
│  │                                                │     │
│  │    MODE B: ACTUAL                              │     │
│  │    - Historical option OHLC from Upstox        │     │
│  │    - Cached locally for performance            │     │
│  │    - Liquidity filters (OI, spread)            │     │
│  │                                                │     │
│  │    MODE C: HYBRID                              │     │
│  │    - Prefer actual data when available         │     │
│  │    - Fall back to synthetic for gaps           │     │
│  │    - Track which mode used per trade           │     │
│  └────────────────────────────────────────────────┘     │
│                                                         │
│  ┌────────────────────────────────────────────────┐     │
│  │ 3. POSITION TRACKER                            │     │
│  │   - Tracks options bar-by-bar from base_data   │     │
│  │   - Calculates position value at each timestamp│     │
│  │   - Handles:                                   │     │
│  │      * Time decay (theta)                      │     │
│  │      * Delta changes (underlying moves)        │     │
│  │      * Expiry events (forced close)            │     │
│  │      * Early exits (stop-loss, take-profit)    │     │
│  │   - Computes Greeks (delta, gamma, theta, vega)│     │
│  └────────────────────────────────────────────────┘     │
│                                                         │
│  ┌────────────────────────────────────────────────┐     │
│  │ 4. METRICS CALCULATOR                          │     │
│  │    - Equity-equivalent metrics:                │     │
│  │      * Total P&L, P&L%, win rate, Sharpe       │     │
│  │      * Max drawdown, avg hold time             │     │
│  │    - Options-specific metrics:                 │     │
│  │      * Theta capture, gamma P&L                │     │
│  │      * IV change contribution                  │     │
│  │      * Expiry rate, roll count                 │     │
│  │    - Comparison analytics:                     │     │
│  │      * Equity vs Options performance           │     │
│  │      * Capital efficiency (ROI)                │     │
│  └────────────────────────────────────────────────┘     │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼ Outputs
         ┌──────────────────────────────┐
         │ options_trades.csv           │  Entry/exit for each option
         │ options_base_data.csv        │  Bar-by-bar position values
         │ options_metrics.json         │  Performance statistics
         │ comparison_report.csv        │  Equity vs Options
         │ pricing_validation.csv       │  Synthetic vs Actual errors
         │ position_lifecycle.csv       │  Greeks, time decay tracking
         └──────────────────────────────┘
```

### Directory Structure

```
src/core/options/
├── planning/
│   ├── implementation_plan.md       # This document
│   └── decisions.md                 # Architecture decision records
│
├── validation/
│   ├── __init__.py
│   ├── pricing_validator.py         # Synthetic vs actual comparison
│   ├── data_fetcher.py              # Fetch historical options from Upstox
│   ├── validation_runner.py         # Run validation experiments
│   └── validation_config.yaml       # Validation parameters
│
├── pricing/
│   ├── __init__.py
│   ├── synthetic_engine.py          # Black-Scholes implementations
│   ├── actual_engine.py             # Historical data pricing
│   ├── hybrid_engine.py             # Combined approach
│   └── volatility_models.py         # Historical vol, Parkinson, EWMA
│
├── replay/
│   ├── __init__.py
│   ├── trade_mapper.py              # Equity → option contract mapping
│   ├── position_tracker.py          # Bar-by-bar position tracking
│   ├── metrics_calculator.py        # Performance metrics
│   └── replay_runner.py             # Main orchestrator
│
├── data/
│   ├── cache/                       # Cached options data (parquet)
│   ├── validation_results/          # Validation reports
│   ├── schemas.py                   # Data schemas
│   └── lot_sizes.csv                # Ticker → lot size mapping
│
├── config/
│   └── options_config.yaml          # Options-specific configuration
│
└── options_engine.py                # EXISTING (Black-Scholes, Greeks)
```

---

## Implementation Phases

### Phase 0: Preparation (Week 1)

**Objective**: Set up workspace, tooling, and data infrastructure

**Tasks**:
1. ✅ Create directory structure
2. Create `lot_sizes.csv` with ticker → lot size mapping
3. Set up Upstox API integration for historical options data
4. Define data schemas for options contracts
5. Create configuration templates

**Deliverables**:
- Working directory structure
- Lot size reference data
- Data schema definitions
- Configuration files

**Success Criteria**:
- Can fetch at least 1 week of options data for 1 ticker
- Data is stored in consistent schema (parquet format)

---

### Phase 1: Data Validation (Weeks 2-3)

**Objective**: Empirically measure synthetic pricing accuracy vs actual market prices

#### 1.1 Data Collection

**Tickers** (5 liquid underlyings):
- **NIFTY** (index, high liquidity)
- **BANKNIFTY** (index, high liquidity)
- **RELIANCE** (equity, large cap)
- **TCS** (equity, IT sector)
- **INFY** (equity, IT sector)

**Time Period**:
- **6 months** (maximum available from Upstox expired contracts API)
- Specific period: TBD based on data availability (e.g., Apr 2024 - Oct 2024)

**Data to Fetch**:
For each ticker:
- All weekly and monthly expiries in the period
- Strikes: ±20% around ATM (typically 15-25 strikes per expiry)
- Timeframe: Daily OHLC (sufficient for validation)
- Additional fields: Open Interest, Volume
- Underlying OHLC (for vol calculations)

**Estimated volume**:
- 5 tickers × 6 months × 4 weekly expiries/month × 20 strikes = ~2,400 option contracts
- Daily data: ~125 trading days × 2,400 contracts = ~300,000 data points

#### 1.2 Synthetic Pricing Models to Test

**Model 1: BS + 20-day Historical Vol** (current implementation)
```python
returns = underlying.pct_change()
vol = returns.rolling(20).std() * sqrt(252)
price = black_scholes(S, K, T, r=0.06, sigma=vol)
```

**Model 2: BS + 5-day Realized Vol** (short-term regime)
```python
vol = returns.rolling(5).std() * sqrt(252)
```

**Model 3: BS + Parkinson Volatility** (uses high-low range)
```python
vol = sqrt(1/(4*n*ln(2)) * sum((ln(High/Low))^2)) * sqrt(252)
```

**Model 4: BS + Calibrated IV** (fit to ATM, apply skew)
```python
# Back out IV from ATM options, apply empirical skew adjustment
iv_atm = implied_vol_from_market(ATM_option_price)
skew_factor = 1.0 + skew_coefficient * (K/S - 1)  # Linear skew approximation
iv = iv_atm * skew_factor
price = black_scholes(S, K, T, r, sigma=iv)
```

#### 1.3 Validation Methodology

For each model and each option contract:

**Error Metrics**:
```python
actual_price = historical_option_mid_price  # (bid + ask) / 2
synthetic_price = model.calculate_price(S, K, T, vol)

absolute_error = abs(synthetic_price - actual_price)
percentage_error = absolute_error / actual_price * 100
```

**Segmentation** (analyze errors by):
- **Moneyness bins**:
  - Deep ITM: S/K > 1.10 (calls) or < 0.90 (puts)
  - ITM: 1.05 < S/K < 1.10
  - ATM: 0.95 < S/K < 1.05
  - OTM: 0.90 < S/K < 0.95
  - Deep OTM: S/K < 0.90

- **DTE bins**:
  - Very short: 1-7 days
  - Short: 8-30 days
  - Medium: 31-60 days
  - Long: 60+ days

- **Volatility regime**:
  - Low vol: realized vol < 15%
  - Medium vol: 15% < vol < 25%
  - High vol: vol > 25%

**Liquidity Filtering**:
Only include contracts with:
- Open Interest > 100
- Bid-Ask Spread < 10% of mid price
- Volume > 10 contracts/day

#### 1.4 Validation Outputs

**Primary Report**: `pricing_validation_summary.csv`
```csv
Ticker,Model,Moneyness,DTE_Range,Vol_Regime,Sample_Size,Mean_Error_Pct,Median_Error_Pct,Std_Error,P95_Error_Pct
NIFTY,BS_20d,ATM,8-30,Medium,1250,8.2,6.5,4.3,18.3
NIFTY,BS_5d,ATM,8-30,Medium,1250,12.1,9.8,6.8,28.5
NIFTY,BS_Parkinson,ATM,8-30,Medium,1250,7.8,6.1,4.1,16.9
NIFTY,BS_CalibratedIV,ATM,8-30,Medium,1250,5.2,4.1,3.2,12.8
...
```

**Detail Report**: `pricing_validation_detail.csv`
- Row per option contract per timestamp
- Columns: ticker, strike, expiry, timestamp, actual_price, model1_price, model1_error, model2_price, ...

**Visualizations**:
- Error distribution histograms (by model, moneyness, DTE)
- Error heatmaps (moneyness × DTE)
- Time series of errors (track model drift over 6 months)

#### 1.5 Decision Criteria

**Acceptable Error Thresholds**:
- **ATM options, 8-30 DTE**: Median error < 10%
- **OTM options, 8-30 DTE**: Median error < 20%
- **All options**: 95th percentile error < 30%

**Decision Tree**:
```
IF best_model median_error(ATM, 8-30 DTE) < 10%:
    → Hybrid mode viable (use synthetic for missing data)
    → Document which model to use (likely Model 4: Calibrated IV)

ELIF best_model median_error < 15%:
    → Synthetic acceptable for directional strategies only
    → Use actual data for final validation

ELSE:
    → Synthetic not reliable
    → Must use actual data exclusively
    → Limit backtest period to data availability (6 months)
```

**Expected Outcome**:
Based on empirical evidence from similar studies, Model 4 (Calibrated IV) should achieve 5-8% median error for ATM options, making hybrid mode viable.

---

### Phase 2: MVP Replay Engine (Weeks 4-5)

**Objective**: Build minimal working replay engine with fixed parameters

#### 2.1 Fixed Parameters (Simplicity)

- **Lot size**: Fixed at 1 lot per trade
- **Strike selection**: ATM only (nearest strike to underlying price)
- **Expiry selection**: Nearest weekly expiry
- **Option type**:
  - LONG equity signal → Buy Call (CE)
  - SHORT equity signal → Buy Put (PE)
- **Pricing mode**: Synthetic only (using best model from Phase 1)
- **Exit logic**: Follow equity exit signal exactly (no early exits)

#### 2.2 Components to Build

**A. Trade Mapper** (`replay/trade_mapper.py`)

**Input**: Equity `trades.csv`
```csv
ticker,entry_time,exit_time,side,entry_price,exit_price,quantity,pnl
RELIANCE,2024-01-15 09:30,2024-01-15 14:45,LONG,2850.25,2868.50,100,1825.0
```

**Logic**:
```python
def map_equity_trade_to_option(equity_trade, config):
    # 1. Determine option type
    option_type = 'CE' if equity_trade['side'] == 'LONG' else 'PE'

    # 2. Select expiry
    expiry = get_nearest_weekly_expiry(equity_trade['entry_time'])

    # 3. Check if enough DTE (minimum 3 days)
    dte = (expiry - equity_trade['entry_time']).days
    if dte < 3:
        return None  # Skip trade, insufficient time

    # 4. Select strike (ATM)
    underlying_price = equity_trade['entry_price']
    strike = round_to_nearest_strike(underlying_price, ticker)

    # 5. Get lot size
    lot_size = get_lot_size(equity_trade['ticker'], expiry)

    # 6. Calculate entry price (synthetic)
    T = dte / 365.0
    vol = get_historical_vol(equity_trade['ticker'], equity_trade['entry_time'])
    option_entry_price = black_scholes(
        S=underlying_price, K=strike, T=T, r=0.06,
        sigma=vol, option_type=option_type
    )

    # 7. Create option position
    return {
        'underlying': equity_trade['ticker'],
        'option_type': option_type,
        'strike': strike,
        'expiry': expiry,
        'entry_time': equity_trade['entry_time'],
        'exit_time': equity_trade['exit_time'],
        'entry_price': option_entry_price,
        'lots': 1,
        'lot_size': lot_size,
        'quantity': lot_size,
        'entry_cost': option_entry_price * lot_size,
        'pricing_mode': 'synthetic'
    }
```

**Output**: `options_trades_mapped.csv` (one row per option trade)

**B. Position Tracker** (`replay/position_tracker.py`)

**Input**:
- Mapped option trades
- Equity `base_data.csv` (for underlying price movement)

**Logic**:
```python
def track_option_position(option_trade, base_data):
    # Filter base_data for this position's timeframe
    position_bars = base_data[
        (base_data['timestamp'] >= option_trade['entry_time']) &
        (base_data['timestamp'] <= option_trade['exit_time'])
    ]

    position_history = []

    for timestamp, bar in position_bars.iterrows():
        underlying_price = bar['close']

        # Calculate DTE remaining
        dte_remaining = (option_trade['expiry'] - timestamp).total_seconds() / (365 * 24 * 3600)

        # Handle expiry
        if dte_remaining <= 0:
            # Option expired, intrinsic value only
            if option_trade['option_type'] == 'CE':
                option_price = max(underlying_price - option_trade['strike'], 0)
            else:  # PE
                option_price = max(option_trade['strike'] - underlying_price, 0)

            # Force close position
            close_early = True
        else:
            # Calculate synthetic price
            vol = get_historical_vol(option_trade['underlying'], timestamp)
            option_price = black_scholes(
                S=underlying_price,
                K=option_trade['strike'],
                T=dte_remaining,
                r=0.06,
                sigma=vol,
                option_type=option_trade['option_type']
            )
            close_early = False

        # Calculate position metrics
        position_value = option_price * option_trade['quantity']
        unrealized_pnl = (option_price - option_trade['entry_price']) * option_trade['quantity']
        unrealized_pnl_pct = unrealized_pnl / option_trade['entry_cost']

        # Calculate Greeks
        greeks = calculate_greeks(
            S=underlying_price, K=option_trade['strike'],
            T=dte_remaining, r=0.06, sigma=vol,
            option_type=option_trade['option_type']
        )

        position_history.append({
            'timestamp': timestamp,
            'underlying_price': underlying_price,
            'option_price': option_price,
            'position_value': position_value,
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_pct': unrealized_pnl_pct,
            'dte_remaining': dte_remaining,
            'delta': greeks['delta'],
            'theta': greeks['theta'],
            'gamma': greeks['gamma'],
            'vega': greeks['vega']
        })

        if close_early:
            break

    # Calculate final P&L
    final_price = position_history[-1]['option_price']
    realized_pnl = (final_price - option_trade['entry_price']) * option_trade['quantity']
    realized_pnl_pct = realized_pnl / option_trade['entry_cost']

    return position_history, realized_pnl, realized_pnl_pct
```

**Output**: `options_base_data.csv` (bar-by-bar position values)

**C. Metrics Calculator** (`replay/metrics_calculator.py`)

**Metrics to Calculate**:

**Equity-Equivalent Metrics**:
- Total P&L (absolute ₹)
- P&L % (return on capital deployed)
- Win Rate (% profitable trades)
- Sharpe Ratio (annualized)
- Max Drawdown (%)
- Average Hold Time (hours)
- Number of Trades

**Options-Specific Metrics**:
- Theta Capture (total time decay captured, ₹)
- Gamma P&L (P&L attributed to delta changes, ₹)
- Vega P&L (P&L from vol changes, ₹)
- Expiry Rate (% of trades that reached expiry)
- Average DTE at Entry
- Average DTE at Exit
- Capital Efficiency (P&L / max capital deployed)

**Output**: `options_metrics.json`

#### 2.3 Testing Strategy

**Test Dataset**:
- 1 ticker (RELIANCE)
- 1 month of trades (e.g., January 2024)
- Expected: ~15-30 equity trades → 15-30 option replays

**Validation Checks**:
1. All equity trades successfully mapped (or documented why skipped)
2. No negative option prices
3. No positions held past expiry
4. P&L calculations balance (sum of unrealized = realized at exit)
5. Greeks are within reasonable bounds (delta 0-1, etc.)

**Success Criteria**:
- MVP runs end-to-end without errors
- Produces all 3 output files (trades, base_data, metrics)
- Results are manually spot-checked and make intuitive sense

---

### Phase 3: Validation with Actual Data (Weeks 6-7)

**Objective**: Compare MVP synthetic results with actual historical option prices

#### 3.1 Enhance Pricing Engine

**Add Actual Data Mode** (`pricing/actual_engine.py`)

**Logic**:
```python
def get_actual_option_price(ticker, strike, expiry, option_type, timestamp):
    # Query cached historical options data
    option_data = load_option_chain_data(ticker, expiry)

    # Filter for specific contract
    contract = option_data[
        (option_data['strike'] == strike) &
        (option_data['option_type'] == option_type) &
        (option_data['timestamp'] == timestamp)
    ]

    if contract.empty:
        return None  # Data not available

    # Apply liquidity filters
    if contract['open_interest'].values[0] < 100:
        return None  # Illiquid

    spread_pct = (contract['ask'] - contract['bid']) / contract['mid']
    if spread_pct > 0.10:  # >10% spread
        return None  # Too wide

    # Use mid price for fill assumption
    return contract['mid'].values[0]
```

**Add Hybrid Mode** (`pricing/hybrid_engine.py`)

```python
def get_hybrid_price(ticker, strike, expiry, option_type, timestamp, underlying_price, vol):
    # Try actual first
    actual_price = get_actual_option_price(ticker, strike, expiry, option_type, timestamp)

    if actual_price is not None:
        return actual_price, 'actual'

    # Fall back to synthetic
    synthetic_price = black_scholes(...)
    return synthetic_price, 'synthetic'
```

#### 3.2 Re-run MVP with All 3 Pricing Modes

For the same test dataset (RELIANCE, 1 month):

1. **Run with Synthetic pricing** (already done in Phase 2)
2. **Run with Actual pricing** (requires fetched data)
3. **Run with Hybrid pricing**

#### 3.3 Comparison Analysis

**Per-Trade Comparison**: `pricing_mode_comparison.csv`
```csv
trade_id,entry_time,strike,expiry,synthetic_entry,actual_entry,entry_diff_pct,synthetic_exit,actual_exit,exit_diff_pct,synthetic_pnl,actual_pnl,pnl_diff_pct
1,2024-01-15 09:30,2850,2024-01-18,45.2,48.3,6.4,62.1,58.9,-5.4,8450,5300,-37.3
...
```

**Aggregate Comparison**:
```
Metric                | Synthetic | Actual   | Hybrid   | Difference
----------------------|-----------|----------|----------|------------
Total P&L (₹)         | 125,430   | 98,200   | 110,500  | -21.7%
Win Rate              | 67%       | 62%      | 64%      | -5 pp
Sharpe Ratio          | 1.85      | 1.52     | 1.68     | -17.8%
Max Drawdown          | -12.3%    | -18.7%   | -15.2%   | +52.0%
Avg Hold Time (hrs)   | 5.2       | 5.2      | 5.2      | 0%
```

**Insights to Extract**:
- Which pricing mode is most conservative? (likely Actual)
- Are synthetic results systematically biased (over/under-estimate P&L)?
- How often does hybrid mode fall back to synthetic? (data availability %)

#### 3.4 Decision Point

**Question**: Is the difference between synthetic and actual acceptable?

**If Total P&L difference < 20%**:
- Hybrid mode is viable
- Use hybrid for full backtest (Phase 4)
- Document the bias (e.g., "synthetic overstates by ~15%")

**If Total P&L difference > 30%**:
- Synthetic is not reliable
- Use actual-only mode
- Accept shorter backtest period (limited to data availability)

---

### Phase 4: Full Backtest with Enhancements (Weeks 8-10)

**Objective**: Run full-scale backtest on all 5 tickers, all available data, with configurable parameters

#### 4.1 Expand Dataset

**Tickers**: All 5 (NIFTY, BANKNIFTY, RELIANCE, TCS, INFY)

**Time Period**:
- **Actual data mode**: 6 months (Apr-Oct 2024, or whatever is available)
- **Hybrid mode**: Up to 3 years (if underlying data exists, use synthetic for missing options data)

**Expected Scale**:
- 5 tickers × 6 months × ~50 trades/ticker = ~1,500 option replays

#### 4.2 Add Configuration Options

**Enhance `config/options_config.yaml`**:

```yaml
options:
  pricing_mode: "hybrid"  # synthetic | actual | hybrid

  # Strike selection
  strike_selection:
    method: "atm"  # atm | delta | moneyness | premium_pct
    # If method = delta:
    target_delta: 0.30
    # If method = moneyness:
    target_moneyness: 1.05  # 5% OTM call
    # If method = premium_pct:
    target_premium_pct: 0.02  # Option costs 2% of underlying

  # Expiry selection
  expiry_selection:
    method: "nearest_weekly"  # nearest_weekly | nearest_monthly | fixed_dte
    # If method = fixed_dte:
    target_dte: 7  # Always 7 days to expiry

  # Lot sizing
  lot_sizing:
    method: "fixed"  # fixed | capital_match | delta_match
    # If method = fixed:
    fixed_lots: 1
    # If method = capital_match:
    equity_capital_to_match: "from_equity_trade"  # Use equity trade size
    # If method = delta_match:
    match_delta_exposure: true

  # Risk management
  risk:
    min_dte_to_enter: 3  # Don't enter if <3 days to expiry
    force_close_before_expiry_hours: 24  # Close 1 day before expiry
    enable_stop_loss: false  # Disabled for MVP (follow equity exit)
    enable_take_profit: false

  # Liquidity filters
  liquidity:
    min_open_interest: 100
    max_spread_pct: 0.10  # 10% max bid-ask spread

  # Synthetic pricing (if using synthetic/hybrid)
  synthetic:
    volatility_model: "historical_20d"  # historical_20d | historical_5d | parkinson | calibrated_iv
    risk_free_rate: 0.06
    vol_floor: 0.10  # Minimum 10% vol
    vol_cap: 1.50    # Maximum 150% vol

  # Actual pricing (if using actual/hybrid)
  actual:
    fill_assumption: "mid"  # mid | open | close | vwap
    cache_dir: "src/core/options/data/cache"
```

#### 4.3 Implement Configurable Logic

Update Trade Mapper and Position Tracker to respect config parameters:

- Strike selection becomes pluggable (factory pattern)
- Expiry selection becomes pluggable
- Lot sizing becomes pluggable
- Early exit logic (when enabled) checks stop-loss/take-profit

#### 4.4 Run Sensitivity Analysis

**Vary key parameters** to understand impact:

**Experiment 1: Strike Selection**
- ATM vs 30-delta vs 5% OTM
- Expected: ATM has highest theta decay, OTM has highest leverage

**Experiment 2: Expiry Selection**
- Weekly vs Monthly
- Expected: Weekly has higher theta but more roll frequency

**Experiment 3: Volatility Model** (for synthetic/hybrid)
- 20-day vs 5-day vs Parkinson
- Expected: 5-day adapts faster to regime changes

**Output**: `sensitivity_analysis.csv`
```csv
parameter,value,total_pnl,win_rate,sharpe,max_dd
strike_method,atm,125000,65,1.85,-12.3
strike_method,delta_30,148000,62,1.92,-15.8
strike_method,otm_5pct,172000,58,1.78,-22.1
expiry_method,weekly,125000,65,1.85,-12.3
expiry_method,monthly,98000,68,1.72,-10.5
...
```

#### 4.5 Full Backtest Outputs

**Primary Outputs**:
1. `options_trades.csv` - All option trades
2. `options_base_data.csv` - Bar-by-bar position tracking
3. `options_metrics.json` - Aggregate performance
4. `comparison_equity_vs_options.csv` - Side-by-side comparison
5. `sensitivity_analysis.csv` - Parameter impact
6. `data_quality_report.csv` - Track synthetic vs actual usage % in hybrid mode

**Visualizations**:
- Equity curve comparison (equity vs options)
- Drawdown comparison
- P&L distribution (histogram)
- Win rate by ticker
- Greeks evolution (example positions)

---

## Data Requirements

### Underlying Data (Already Available)

- **Source**: Existing equity backtester outputs
- **Files**: `trades.csv`, `base_data.csv`
- **Timeframe**: Multiple years of history (2020-2025)
- **Tickers**: All tickers in current system

### Options Data (New)

#### Historical Options Contracts

**API**: Upstox Expired Instruments API

**Endpoints**:
1. `GET /v2/expired-instruments/expiries` - Get expiry dates for a ticker
2. `GET /v2/expired-instruments/option/contract` - Get all strikes for an expiry
3. `GET /v2/expired-instruments/historical-candle` - Get OHLC for a specific option

**Data Points per Contract**:
- Strike price
- Expiry date
- Option type (CE/PE)
- Timestamp
- OHLC (Open, High, Low, Close)
- Volume
- Open Interest
- Lot size

**Storage Format**: Parquet (columnar, compressed)

**Schema**:
```python
{
    'ticker': str,          # Underlying (e.g., 'RELIANCE')
    'strike': float,        # Strike price
    'expiry': datetime,     # Expiry date
    'option_type': str,     # 'CE' or 'PE'
    'timestamp': datetime,  # Bar timestamp
    'open': float,          # Option open price
    'high': float,
    'low': float,
    'close': float,
    'volume': int,
    'open_interest': int,
    'bid': float,           # If available
    'ask': float,           # If available
    'mid': float,           # (bid + ask) / 2
    'lot_size': int
}
```

#### Lot Size Reference Data

**File**: `src/core/options/data/lot_sizes.csv`

**Schema**:
```csv
ticker,lot_size,exchange,last_updated
NIFTY,50,NSE,2024-01-01
BANKNIFTY,15,NSE,2024-01-01
RELIANCE,505,NSE,2024-01-01
TCS,150,NSE,2024-01-01
INFY,300,NSE,2024-01-01
```

**Note**: Lot sizes can change; maintain version history

---

## Expected Outputs

### 1. Validation Phase Outputs

**File**: `src/core/options/data/validation_results/pricing_validation_summary.csv`

**Purpose**: Determine synthetic pricing accuracy

**Columns**:
- Ticker, Model, Moneyness, DTE_Range, Vol_Regime
- Sample_Size, Mean_Error_Pct, Median_Error_Pct, Std_Error, P95_Error_Pct

**Usage**: Decide which pricing mode and model to use

---

### 2. Options Trade Ledger

**File**: `outputs/{run_id}/options_trades.csv`

**Schema**:
```csv
trade_id,underlying,option_type,strike,expiry,entry_time,exit_time,entry_price,exit_price,lots,lot_size,quantity,entry_cost,exit_value,realized_pnl,realized_pnl_pct,hold_time_hours,dte_at_entry,dte_at_exit,pricing_mode,exit_reason
1,RELIANCE,CE,2850,2024-01-18,2024-01-15 09:30,2024-01-15 14:45,45.2,62.1,1,505,505,22826,31370.5,8544.5,37.4,5.25,3.0,2.78,hybrid,equity_signal
2,TCS,PE,3450,2024-01-12,2024-01-08 10:15,2024-01-11 15:30,58.3,42.1,1,150,150,8745,6315,-2430,-27.8,77.25,4.0,1.02,actual,equity_signal
...
```

**Key Fields**:
- `pricing_mode`: Track if synthetic, actual, or hybrid was used
- `exit_reason`: equity_signal | expiry | stop_loss | take_profit (for later enhancements)

---

### 3. Position Lifecycle Tracking

**File**: `outputs/{run_id}/options_base_data.csv`

**Schema**:
```csv
trade_id,timestamp,underlying_price,option_price,position_value,unrealized_pnl,unrealized_pnl_pct,dte_remaining,delta,gamma,theta,vega,pricing_mode
1,2024-01-15 09:30,2850.25,45.2,22826,0,0,3.0,0.52,0.018,-12.3,85.2,hybrid
1,2024-01-15 09:35,2853.10,47.8,24139,1313,5.8,2.998,0.54,0.019,-12.1,84.8,hybrid
1,2024-01-15 09:40,2854.50,48.9,24695.5,1869.5,8.2,2.995,0.55,0.019,-12.0,84.5,hybrid
...
```

**Purpose**:
- Intraday P&L tracking
- Greeks evolution
- Theta decay visualization
- Max favorable/adverse excursion

---

### 4. Performance Metrics

**File**: `outputs/{run_id}/options_metrics.json`

**Structure**:
```json
{
  "summary": {
    "total_trades": 1523,
    "winning_trades": 982,
    "losing_trades": 541,
    "win_rate": 0.645,
    "total_pnl": 2458300,
    "total_pnl_pct": 42.8,
    "sharpe_ratio": 1.87,
    "max_drawdown_pct": -18.4,
    "avg_hold_time_hours": 12.3,
    "avg_dte_at_entry": 6.8,
    "total_capital_deployed": 5742000
  },
  "by_ticker": {
    "RELIANCE": {
      "trades": 305,
      "pnl": 512300,
      "win_rate": 0.68,
      "sharpe": 2.05
    },
    ...
  },
  "options_specific": {
    "theta_capture_total": 125600,
    "gamma_pnl_total": 342100,
    "vega_pnl_total": -18900,
    "expiry_rate": 0.12,
    "avg_entry_delta": 0.48,
    "avg_exit_delta": 0.62
  },
  "data_quality": {
    "actual_data_usage_pct": 78.3,
    "synthetic_fallback_pct": 21.7,
    "skipped_trades_pct": 2.1,
    "skipped_reasons": {
      "insufficient_dte": 15,
      "illiquid": 8,
      "wide_spread": 9
    }
  }
}
```

---

### 5. Comparison Report

**File**: `outputs/{run_id}/comparison_equity_vs_options.csv`

**Schema**:
```csv
ticker,strategy,time_period,equity_trades,options_trades,equity_pnl,options_pnl,equity_pnl_pct,options_pnl_pct,equity_sharpe,options_sharpe,equity_max_dd,options_max_dd,equity_win_rate,options_win_rate,winner,pnl_improvement_pct
RELIANCE,mse,2024-04 to 2024-10,45,43,125300,512300,8.3,42.1,1.52,2.05,-8.2,-15.3,0.62,0.68,options,309
TCS,mse,2024-04 to 2024-10,38,36,98200,-42100,6.5,-3.2,1.38,-0.45,-6.5,-22.8,0.58,0.47,equity,-143
...
```

**Purpose**:
- Identify which underlyings benefit from options execution
- Understand risk-reward tradeoff (higher returns but higher drawdowns)
- Guide live trading decisions (use options for RELIANCE, equity for TCS)

---

### 6. Sensitivity Analysis

**File**: `outputs/{run_id}/sensitivity_analysis.csv`

**Schema**:
```csv
parameter,value,total_pnl,win_rate,sharpe,max_dd,avg_hold_time,capital_deployed
strike_method,atm,2458300,64.5,1.87,-18.4,12.3,5742000
strike_method,delta_30,2815600,61.2,1.95,-22.1,12.8,6120000
strike_method,otm_5pct,3124800,58.7,1.72,-28.3,13.1,6890000
expiry_method,weekly,2458300,64.5,1.87,-18.4,12.3,5742000
expiry_method,monthly,2012500,67.8,1.68,-14.2,18.6,8320000
vol_model,hist_20d,2458300,64.5,1.87,-18.4,12.3,5742000
vol_model,hist_5d,2621700,63.1,1.91,-19.8,12.1,5742000
vol_model,parkinson,2503400,64.8,1.89,-17.9,12.4,5742000
vol_model,calibrated_iv,2687500,65.2,1.93,-16.8,12.2,5742000
```

**Purpose**: Guide configuration choices for live trading

---

## Open Questions & Decision Points

### 1. Capital Allocation Strategy

**Question**: How do we match equity position size to options lot size?

**Options**:
- A. Fixed lots (always 1 lot, regardless of equity size)
- B. Capital matching (deploy same ₹ in options as equity)
- C. Delta-adjusted matching (match notional exposure via delta)

**Current Decision**: **Option A for MVP** (simplicity), evaluate B and C in Phase 4

**Impacts**:
- Capital deployed
- Risk per trade
- Comparability to equity results

---

### 2. Exit Trigger Logic

**Question**: When do we close an option position?

**Options**:
- A. Always follow equity exit signal (pure replay)
- B. Independent options exits (stop-loss, take-profit, time-based)
- C. Hybrid (whichever comes first: equity signal OR options risk limit)

**Current Decision**: **Option A for MVP**, add Option C in Phase 4

**Rationale**: MVP tests "if I traded options instead of equity with same signals, what happens?" (apples-to-apples)

**Future Enhancement**: Add options-specific risk management (e.g., close if down 50%, regardless of equity signal)

---

### 3. Option Type for SHORT Signals

**Question**: For bearish equity signals (SHORT), which option strategy?

**Options**:
- A. Buy puts (directional, limited risk)
- B. Sell calls (premium collection, unlimited risk)
- C. Put spreads (defined risk, lower cost)

**Current Decision**: **Option A (buy puts)** for MVP

**Rationale**:
- Keeps risk profile similar to equity (defined risk)
- Simpler logic (no margin requirements, assignment risk)
- Later can add B/C as alternative strategies

---

### 4. Data Fetching Strategy

**Question**: Pre-fetch all data upfront, or fetch on-demand during backtest?

**Options**:
- A. Pre-fetch all 6 months upfront, cache locally
- B. Fetch on-demand per trade (lazy loading)

**Current Decision**: **Option A (pre-fetch)**

**Rationale**:
- Slower initial setup (~1-2 hours to fetch all data)
- But faster backtests (no API calls mid-run)
- Reproducible (same data for every run)
- Can work offline after initial fetch

**Implementation**: Create `validation/data_fetcher.py` that bulk downloads and caches

---

### 5. Fractional Lot Handling

**Question**: If capital matching requires 2.3 lots, what do we trade?

**Options**:
- A. Round down (2 lots) - conservative
- B. Round up (3 lots) - aggressive
- C. Round to nearest (2 lots)
- D. Skip trade (insufficient capital)

**Current Decision**: **Option C (round to nearest)**, with minimum 1 lot

**Track**: Log the fractional amount for analysis (understand bias)

---

### 6. Fill Price Assumption

**Question**: What price do we assume for entry/exit fills?

**Options**:
- A. Open price of next bar (realistic, no lookahead)
- B. Close price of signal bar (lookahead bias)
- C. Mid price (bid+ask)/2 (theoretical, assumes no spread)
- D. VWAP of next bar (most realistic but data-intensive)

**Current Decision**: **Option A (open of next bar)** for synthetic mode, **Option C (mid price)** for actual mode

**Rationale**:
- Synthetic doesn't have bid/ask, so mid is only option
- Actual data has bid/ask, use mid as proxy for executable price
- Document this assumption (real spreads will eat into P&L)

---

### 7. Liquidity Filters

**Question**: When do we skip a trade due to poor liquidity?

**Thresholds**:
- Open Interest < 100 contracts → Skip
- Bid-Ask Spread > 10% of mid → Skip
- Volume < 10 contracts/day → Skip (for actual mode)

**Current Decision**: **Enable all 3 filters** for actual mode, **N/A for synthetic mode**

**Track**: Log skipped trades to measure impact on trade count

---

### 8. Volatility Regime Handling

**Question**: Do we adjust strategy based on vol regime (low/medium/high)?

**Options**:
- A. Ignore regime (same logic always)
- B. Adjust strike selection (go further OTM in high vol)
- C. Adjust expiry (go shorter DTE in high vol)
- D. Skip trading in extreme vol (>50% or <10%)

**Current Decision**: **Option A (ignore)** for MVP

**Future Enhancement**: Phase 5 could add regime-aware logic

---

### 9. Roll Logic (Future)

**Question**: If equity signal persists but option is about to expire, do we roll?

**Options**:
- A. Close and stop tracking (current plan)
- B. Roll to next expiry (more complex, incurs roll cost)

**Current Decision**: **Option A** (no rolling in MVP)

**Rationale**: Rolling adds complexity (spread costs, execution risk). First validate that base strategy works.

**Future Enhancement**: Phase 5+ could add intelligent roll logic

---

### 10. Greeks Calculation Frequency

**Question**: How often do we calculate Greeks during position hold?

**Options**:
- A. Every bar (comprehensive but slow)
- B. Only at entry/exit (fast but incomplete)
- C. Hourly (compromise)

**Current Decision**: **Option A (every bar)** for MVP with small dataset, **Option C** for full-scale backtest

**Rationale**: Greeks evolution is insightful for analysis; compute intensively early, optimize later

---

## Success Criteria

### Phase 1 (Validation)

✅ **Success**:
- Best synthetic model achieves <10% median error on ATM options (8-30 DTE)
- Have 6 months of actual options data for 5 tickers
- Validation report published with clear recommendation

❌ **Failure**:
- Cannot fetch sufficient actual data (API issues, data gaps >30%)
- All synthetic models have >25% median error → Cannot use synthetic

**Mitigation**: If validation fails, pivot to actual-only mode, accept shorter backtest period

---

### Phase 2 (MVP)

✅ **Success**:
- MVP runs end-to-end on 1 ticker, 1 month
- Produces all output files (trades, base_data, metrics)
- Manual spot-checks confirm P&L calculations are correct
- No crashes, negative prices, or positions held past expiry

❌ **Failure**:
- Logic errors in P&L calculation
- Performance is abysmal (crashes on small dataset)

---

### Phase 3 (Actual Data Comparison)

✅ **Success**:
- Synthetic vs Actual P&L difference is <30%
- Understand bias direction (synthetic over/under-estimates)
- Hybrid mode works (falls back gracefully)

❌ **Failure**:
- Synthetic and actual P&L differ by >50% → Synthetic unusable

**Mitigation**: Use actual-only for final validation

---

### Phase 4 (Full Backtest)

✅ **Success**:
- Complete backtest on 5 tickers, 6 months
- Generate comparison report showing equity vs options performance
- Identify at least 2 tickers where options outperform
- Sensitivity analysis reveals optimal parameter choices

❌ **Failure**:
- Options underperform equity on ALL tickers (suggests strategy doesn't work with options)
- Results are non-reproducible (different runs give different results)

**Mitigation**: If options consistently underperform, may indicate:
- Theta decay eats into slow signals (need faster signals)
- Lot size mismatch (need better capital matching)
- Config is suboptimal (need more sensitivity experiments)

---

## Technology Stack

### Core Libraries

**Data Processing**:
- `pandas` - DataFrames for OHLC, trades, positions
- `numpy` - Numerical calculations
- `pyarrow` / `fastparquet` - Parquet file I/O

**Options Pricing**:
- `scipy.stats` - Normal distribution (for Black-Scholes)
- `numpy` - Mathematical functions
- Custom implementations (already in `options_engine.py`)

**Data Fetching**:
- `requests` / `httpx` - Upstox API calls
- `upstox-python` SDK (if available)
- `tenacity` - Retry logic for API failures

**Visualization** (Phase 4):
- `matplotlib` / `seaborn` - Static plots
- `plotly` - Interactive charts (optional)

**Configuration**:
- `pyyaml` - YAML config parsing
- `pydantic` - Config validation (optional)

**Utilities**:
- `tqdm` - Progress bars for long-running tasks
- `loguru` / `logging` - Structured logging

### File Formats

- **Configuration**: YAML (human-readable)
- **Data Storage**: Parquet (compressed, columnar)
- **Outputs**: CSV (equity compatibility), JSON (metrics)
- **Logs**: Structured text logs

---

## Risks & Mitigation

### Risk 1: Upstox API Limitations

**Risk**: API rate limits, downtime, or data gaps prevent fetching historical options data

**Mitigation**:
- Implement retry logic with exponential backoff
- Cache all fetched data locally (never re-fetch)
- Graceful degradation (fall back to synthetic if actual data unavailable)
- Pre-fetch all data upfront (don't depend on API during backtest runs)

---

### Risk 2: Synthetic Pricing Inaccuracy

**Risk**: Synthetic models are too inaccurate, invalidating backtest results

**Mitigation**:
- Phase 1 validation explicitly measures this (fail-fast if error >threshold)
- Use hybrid mode (actual where available, synthetic as fallback)
- Document bias clearly in reports ("synthetic results are indicative only")
- For final validation before live trading, use actual-only mode

---

### Risk 3: Data Volume & Performance

**Risk**: Processing 1,500 trades with bar-by-bar tracking may be slow (1M+ data points)

**Mitigation**:
- Use vectorized pandas operations (avoid Python loops)
- Parquet for fast I/O (columnar format)
- Cache intermediate results (vol calculations, Greeks)
- Parallel processing for independent tickers (if needed)
- Start with small dataset (Phase 2), optimize before scaling (Phase 4)

---

### Risk 4: Configuration Complexity

**Risk**: Too many parameters lead to overfitting or confusion

**Mitigation**:
- Start with fixed defaults (MVP)
- Add configurability incrementally (Phase 4)
- Sensitivity analysis to understand parameter impact
- Document recommended configs based on empirical results

---

### Risk 5: Misalignment with Live Trading

**Risk**: Backtest shows great results but live trading fails (different execution reality)

**Mitigation**:
- Use conservative assumptions (mid price fills, liquidity filters)
- Track data quality (% synthetic vs actual usage)
- Document all assumptions clearly
- Phase 5 (future) would implement paper trading to validate live execution
- For now, treat backtest as **signal quality test**, not execution promise

---

### Risk 6: Lot Size Capital Mismatch

**Risk**: Fixed lot sizes create capital allocation problems (can't replicate equity position sizes)

**Mitigation**:
- Start with fixed 1-lot (accept mismatch, measure impact)
- Phase 4 adds capital-matching and delta-matching strategies
- Comparison report tracks "capital efficiency" metric separately from absolute P&L

---

## Next Steps (Immediate)

### Week 1: Foundation

1. ✅ Create directory structure
2. Create `lot_sizes.csv` with 5 tickers
3. Set up Upstox API credentials and test connection
4. Create data schemas (`data/schemas.py`)
5. Create base configuration (`config/options_config.yaml`)

### Week 2-3: Data Validation

1. Implement `validation/data_fetcher.py`
2. Fetch 6 months of options data for 5 tickers
3. Implement 4 synthetic pricing models
4. Run validation comparison
5. Generate validation report
6. **Decision**: Choose pricing mode for MVP

### Week 4-5: MVP

1. Implement `replay/trade_mapper.py`
2. Implement `replay/position_tracker.py`
3. Implement `replay/metrics_calculator.py`
4. Test on 1 ticker, 1 month
5. Debug and validate results

### Week 6-7: Actual Data Integration

1. Enhance pricing engine with actual mode
2. Implement hybrid mode
3. Re-run MVP with all 3 modes
4. Generate comparison report
5. **Decision**: Proceed with hybrid or actual-only for Phase 4

### Week 8-10: Full Backtest

1. Expand to all 5 tickers, 6 months
2. Add configuration support
3. Run sensitivity analysis
4. Generate final reports and visualizations
5. **Deliverable**: Comprehensive options backtest analysis

---

## Appendix A: File Outputs Reference

| File | Purpose | Schema | Phase |
|------|---------|--------|-------|
| `pricing_validation_summary.csv` | Synthetic vs actual error metrics | ticker, model, moneyness, error_pct | 1 |
| `options_trades.csv` | Option trade ledger | trade_id, strike, expiry, entry/exit, pnl | 2-4 |
| `options_base_data.csv` | Bar-by-bar position tracking | trade_id, timestamp, option_price, greeks | 2-4 |
| `options_metrics.json` | Performance statistics | total_pnl, sharpe, greeks summary | 2-4 |
| `comparison_equity_vs_options.csv` | Equity vs options comparison | ticker, equity_pnl, options_pnl | 3-4 |
| `sensitivity_analysis.csv` | Parameter impact analysis | parameter, value, pnl, sharpe | 4 |
| `pricing_mode_comparison.csv` | Synthetic vs actual per-trade | trade_id, synthetic_pnl, actual_pnl | 3 |

---

## Appendix B: Code Conventions

**File Naming**:
- Snake_case for Python modules: `trade_mapper.py`
- PascalCase for classes: `TradeMapper`, `SyntheticPricingEngine`
- lowercase for packages: `validation`, `pricing`

**Documentation**:
- Docstrings for all public functions (Google style)
- Type hints for function signatures
- README.md in each major directory

**Logging**:
- Use `logging` module (or `loguru`)
- Log levels: DEBUG (verbose), INFO (progress), WARNING (degraded mode), ERROR (failures)
- Log to both file and console

**Testing**:
- Unit tests for pricing models (exact Black-Scholes values)
- Integration tests for replay engine (end-to-end on small dataset)
- Validation tests (spot-check known trades)

**Version Control**:
- Commit frequently with clear messages
- Tag releases (v1.0-validation, v1.0-mvp, etc.)

---

## Appendix C: Glossary

| Term | Definition |
|------|------------|
| **ATM** | At-The-Money: Strike price equals underlying price |
| **DTE** | Days To Expiry: Time remaining until option expiration |
| **Greeks** | Option sensitivity metrics (delta, gamma, theta, vega, rho) |
| **IV** | Implied Volatility: Market's expectation of future volatility |
| **OI** | Open Interest: Number of outstanding option contracts |
| **Moneyness** | Ratio of strike to underlying price (S/K) |
| **Synthetic** | Calculated via model (Black-Scholes) vs actual market price |
| **Hybrid Mode** | Use actual data where available, synthetic for gaps |
| **Replay** | Re-executing equity trades using options contracts |
| **Theta Decay** | Time decay of option value |
| **Delta Exposure** | Notional underlying exposure (delta × lot_size × underlying_price) |

---

**Document Status**: DRAFT for review and iteration

**Next Review**: After Phase 1 completion (pricing validation results)

**Maintained By**: StrategyLab Team

**Last Updated**: 2025-10-08
