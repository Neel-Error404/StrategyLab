# Phase 4 Full Backtest: Multi-Ticker Production Validation

**Objective**: Execute full-scale production backtest across 5 tickers and 6 months of data to validate options strategy performance vs equity baseline

**Estimated Time**: 4-8 hours
**Priority**: HIGH - Final validation before production readiness
**Success Criteria**: Full backtest completes with actionable insights on equity vs options performance

---

## CONTEXT: System Overview

### What You're Working On

You are executing **Phase 4 (Full Backtest & Sensitivity Analysis)** of an options backtesting system. Phase 3 MVP has been completed and validated on single-ticker minimal data. Phase 4 scales to production-level testing with multiple tickers, full time periods, and parameter optimization.

**System Architecture** (Production Scale):
```
┌─────────────────────────────────────────────────────────────────────┐
│ INPUT: EQUITY SIGNALS                                               │
│ - Source: Equity backtester trades.csv (hypothetical or real)      │
│ - Period: 6 months (2025-04-01 to 2025-10-08)                      │
│ - Tickers: 5 liquid instruments                                     │
│   * RELIANCE (Large cap equity)                                     │
│   * TCS (IT sector equity)                                          │
│   * INFY (IT sector equity)                                         │
│   * NIFTY (Index)                                                   │
│   * BANKNIFTY (Index)                                               │
└────────────┬────────────────────────────────────────────────────────┘

> ℹ️ **Data availability note (current repo snapshot)**  
> Equity reference data and pre-generated trades are present for `RELIANCE`, `TCS`, and `INFY`.  
> Options data already exists for `NIFTY` and `BANKNIFTY`, but equity trades for those indices need to be created before they can join the production run.  
> The prompts below call out where to switch from the 3-ticker baseline to the 5-ticker target once index equity trades are ready.
             │
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 4 FULL BACKTEST ENGINE                                        │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 1. MULTI-TICKER PROCESSING                                  │   │
│  │    - Parallel execution (ThreadPoolExecutor)                │   │
│  │    - Per-ticker isolation (independent P&L)                 │   │
│  │    - Cross-ticker aggregation                               │   │
│  │    - Error handling & resilience                            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 2. EXTENDED TIME PERIOD                                      │   │
│  │    - 6 months of continuous data                             │   │
│  │    - ~26 weekly expiries                                     │   │
│  │    - ~500-1000 equity trades → ~400-900 option trades        │   │
│  │    - Realistic market conditions (volatility regimes)        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 3. CONFIGURABLE PARAMETERS (Future Phase)                    │   │
│  │    - Strike selection: ATM / delta-based / moneyness         │   │
│  │    - Expiry selection: weekly / monthly / fixed DTE          │   │
│  │    - Lot sizing: fixed / capital match / delta match         │   │
│  │    - Risk management: stops / take-profits                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 4. COMPREHENSIVE ANALYTICS                                   │   │
│  │    - Equity vs Options comparison (per ticker)               │   │
│  │    - Performance attribution (theta, gamma, vega)            │   │
│  │    - Risk metrics (Sharpe, Sortino, Max DD)                  │   │
│  │    - Data quality tracking (actual vs synthetic usage)       │   │
│  └─────────────────────────────────────────────────────────────┘   │
└────────────┬────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ OUTPUTS (Production Grade)                                          │
│ - options_trades.csv             (~400-900 rows)                    │
│ - options_positions.csv          (~50,000-100,000 position updates) │
│ - options_metrics.json           (aggregated + per-ticker)          │
│ - comparison_equity_vs_options.csv (side-by-side analysis)          │
│ - run_manifest.json              (audit trail with SHA256)          │
│ - logs.jsonl                     (structured execution logs)        │
│ - performance_dashboard.html     (interactive visualizations)       │
└─────────────────────────────────────────────────────────────────────┘
```

### Current Implementation Status

**Phase 1** (Data Validation): ✅ **COMPLETE**
- 6 months of option data fetched for 5 tickers
- Data quality validated (OHLC, OI, volume)
- Storage: Parquet format with metadata

**Phase 2** (Pricing Validation): ✅ **COMPLETE**
- Synthetic pricing accuracy measured
- Best model: BS + Calibrated IV (5.6% error)
- Decision: Hybrid mode (actual preferred, synthetic fallback)

**Phase 3** (MVP Replay Engine): ✅ **COMPLETE**
- Single-ticker test passed (RELIANCE, ~4 weeks)
- All validation criteria met
- P&L accuracy: ₹0.00 error
- Greeks validated, no positions past expiry
- Output files generated correctly

**Phase 4** (Full Backtest): 🔵 **READY TO START** ← **YOU ARE HERE**
- Infrastructure ready (parallel processing, multi-timeframe fallback)
- Configuration system in place
- Analytics engines built
- **Needs**: Full execution + validation on production scale

---

## PROBLEM: Current Situation

### The Challenge

Phase 3 MVP validated the **technical correctness** of the replay engine on a small dataset:
- ✅ 1 ticker (RELIANCE)
- ✅ ~4 weeks of data (May 2025)
- ✅ 10 test trades
- ✅ All technical validations passed

However, **production readiness** requires validation at scale:
- ❓ Does it work across **5 different tickers** (equities + indices)?
- ❓ Can it handle **6 months** of continuous data?
- ❓ Are the results **statistically significant** (enough trades)?
- ❓ How does options performance **compare to equity baseline**?
- ❓ Which tickers **benefit from options execution**?
- ❓ What are the **optimal configuration parameters**?

### Why This Matters

**Business Question**: *"Should we trade options instead of equity for these signals?"*

To answer this, we need:
1. **Sufficient Sample Size**: 400-900 option trades (vs 10 in MVP)
2. **Cross-Ticker Validation**: Performance varies by underlying characteristics
3. **Realistic Market Conditions**: 6 months covers multiple volatility regimes
4. **Apples-to-Apples Comparison**: Same signals, different execution instruments
5. **Risk-Adjusted Returns**: Not just P&L, but Sharpe, drawdown, capital efficiency

### What Could Go Wrong

**Potential Issues**:
1. **Performance Degradation**: Slow execution on large dataset (50K+ position updates)
2. **Data Quality Issues**: Missing options data for some tickers/periods
3. **Memory Constraints**: Loading 6 months of bar-by-bar data
4. **Inconsistent Results**: Different tickers behave differently
5. **Misleading Metrics**: Small sample bias in single-ticker test doesn't generalize

**Example Scenarios**:
- ✅ MVP: RELIANCE 10 trades, 60% win rate, Sharpe 10.93 (overstated due to small sample)
- ❓ Full: 500 trades across 5 tickers, true win rate may be 52%, Sharpe 1.2 (realistic)

---

## TASK: Your Mission

### Primary Objective

**Execute a full-scale production backtest** across 5 tickers and 6 months to:
1. Validate system scalability and robustness
2. Generate equity vs options comparison report
3. Identify which tickers benefit from options execution
4. Establish baseline performance metrics for production

### Success Criteria (From Implementation Plan)

You will know Phase 4 is complete when:

> ✅ **Baseline vs target**  
> - Baseline (available today): 3 equity tickers (`RELIANCE`, `TCS`, `INFY`) run end-to-end.  
> - Target (production scope): expand to 5 ticker set once equity trades for `NIFTY` and `BANKNIFTY` are generated.

#### 1. Technical Execution ✅
- [ ] Backtest runs end-to-end on all 5 tickers without crashes *(start with 3 equity tickers, expand to 5 when index equity trades are available)*
- [ ] Processes 6 months of data (2025-04-01 to 2025-10-08)
- [ ] Generates 400-900 option trades (realistic volume)
- [ ] All output files created with correct schemas
- [ ] No data integrity issues (negative prices, positions past expiry)

#### 2. Performance Validation ✅
- [ ] At least 2 tickers show **positive options performance** vs equity
- [ ] Win rates are realistic (45-65% range, not inflated)
- [ ] Sharpe ratios are achievable (0.5-2.5 range for options)
- [ ] Drawdowns are acceptable (<30% for options strategies)
- [ ] P&L is reproducible (same inputs → same outputs)

#### 3. Analytical Insights ✅
- [ ] Clear equity vs options comparison report generated
- [ ] Per-ticker performance breakdown available
- [ ] Identified which tickers benefit from options
- [ ] Understood failure modes (which tickers underperform)
- [ ] Data quality metrics tracked (actual vs synthetic usage %)

#### 4. Production Readiness ✅
- [ ] Execution time is acceptable (<30 minutes for full backtest)
- [ ] Memory usage is reasonable (<16GB)
- [ ] Logs are structured and debuggable
- [ ] Results can be explained to stakeholders
- [ ] System can be re-run with different parameters

### Acceptance Test

**Command**:
```bash
python test_phase3_integration.py --mode multi
```

> ℹ️ The sample output below reflects the 5-ticker target state.  
> Running with the current 3-ticker dataset will report fewer total trades and omit NIFTY/BANKNIFTY until equity trades for those indices are generated.

**Expected Output**:
```
================================================================================
PHASE 4 FULL BACKTEST - MULTI-TICKER INTEGRATION TEST
================================================================================

Configuration:
   Tickers: RELIANCE, TCS, INFY, NIFTY, BANKNIFTY
   Period: 2025-04-01 to 2025-10-08 (6 months)
   Mode: Actual pricing (hybrid fallback)
   Parallel: Enabled (5 workers)

Ticker Processing:
   ✅ RELIANCE: 245 trades processed, 12 skipped (₹1,245,300 P&L, Sharpe 1.45)
   ✅ TCS: 189 trades processed, 8 skipped (₹-125,400 P&L, Sharpe 0.82)
   ✅ INFY: 201 trades processed, 6 skipped (₹842,100 P&L, Sharpe 1.68)
   ✅ NIFTY: 156 trades processed, 4 skipped (₹523,200 P&L, Sharpe 1.22)
   ✅ BANKNIFTY: 132 trades processed, 5 skipped (₹-89,500 P&L, Sharpe 0.65)

Aggregated Results:
   Total trades: 923
   Total P&L: ₹2,395,700
   Overall win rate: 56.2%
   Overall Sharpe: 1.36
   Max drawdown: -18.4%
   Execution time: 18m 32s

Equity vs Options Comparison:
   Winners (Options > Equity): RELIANCE (+45%), INFY (+28%)
   Neutral: NIFTY (-5%)
   Losers (Options < Equity): TCS (-32%), BANKNIFTY (-18%)

Data Quality:
   Actual pricing used: 87.3%
   Synthetic fallback: 12.7%
   Skipped trades: 3.8%

✅ ✅ ✅  PHASE 4 INTEGRATION TEST PASSED  ✅ ✅ ✅
================================================================================
```

---

## IMPLEMENTATION GUIDE

### Step 1: Pre-Flight Checks

Before running the full backtest, validate prerequisites:

**1.1 Data Availability Check**
```bash
# Target vs available tickers (adjust when index equity trades are ready)
TICKERS_TARGET=(RELIANCE TCS INFY NIFTY BANKNIFTY)
TICKERS_AVAILABLE=(RELIANCE TCS INFY)  # update to TICKERS_TARGET once NIFTY/BANKNIFTY equity trades exist

# Check equity data exists for available tickers
for ticker in "${TICKERS_AVAILABLE[@]}"; do
    echo "Checking $ticker..."
    ls -lh data/pools/2022-01-01_to_2025-08-31/$ticker/15m.parquet
done

# Check options data exists for target tickers (NIFTY/BANKNIFTY options already present)
for ticker in "${TICKERS_TARGET[@]}"; do
    echo "Checking $ticker options..."
    ls -d data/pools/options/2025-04-01_to_2025-10-08/$ticker/
    ls data/pools/options/2025-04-01_to_2025-10-08/$ticker/1day/ | wc -l
done
```

**Expected Output**:
```
Checking RELIANCE...
-rw-r--r-- 1 user user 125M Sep 13 15:50 data/pools/2022-01-01_to_2025-08-31/RELIANCE/15m.parquet

... (similar for TCS, INFY; repeat for NIFTY/BANKNIFTY once equity trades exist)

Checking RELIANCE options...
data/pools/options/2025-04-01_to_2025-10-08/RELIANCE/
26  (number of expiry files)
```

**1.2 Configuration Validation**
```bash
# Validate options config
python -c "
from src.core.options.config.config import OptionsConfig
config = OptionsConfig.from_yaml('src/core/options/config/options_config.yaml')
print(f'Config loaded: {config.replay.pricing_mode}')
print(f'Tickers configured: {config.replay.tickers or \"auto-discover\"}')
print(f'Parallel enabled: {config.replay.parallel_processing}')
"
```

**1.3 System Resources Check**
```bash
# Check available memory
free -h | grep Mem

# Check available disk space
df -h | grep -E "(Filesystem|/mnt)"

# Check CPU cores
nproc
```

**Minimum Requirements**:
- Memory: 8GB available (16GB recommended)
- Disk: 10GB free space
- CPU: 4+ cores for parallel processing

### Step 2: Create Synthetic Equity Trades (If Needed)

If you don't have actual equity trades, create synthetic test trades:

**2.1 Generate Synthetic Trade Signal**
```bash
# Create synthetic trades covering 6 months
python -c "
import pandas as pd
from pathlib import Path

# Generate ~100 trades per ticker over 6 months
# Start with available equity tickers; append NIFTY/BANKNIFTY once equity trades exist
tickers = ['RELIANCE', 'TCS', 'INFY']  # + ['NIFTY', 'BANKNIFTY'] when ready
start_date = pd.Timestamp('2025-04-01', tz='Asia/Kolkata')
end_date = pd.Timestamp('2025-10-08', tz='Asia/Kolkata')

trades = []
for ticker in tickers:
    # Generate 100 random trades per ticker
    dates = pd.date_range(start_date, end_date, periods=100)
    for i, entry_time in enumerate(dates):
        # Random hold time: 4-48 hours
        hold_hours = pd.Timedelta(hours=4 + (i % 44))
        exit_time = entry_time + hold_hours

        # Random side (60% LONG, 40% SHORT)
        side = 'LONG' if i % 5 < 3 else 'SHORT'

        trades.append({
            'ticker': ticker,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'side': side,
            'entry_price': 1000 + (i * 10),  # Dummy prices
            'exit_price': 1020 + (i * 10),
            'quantity': 100,
            'pnl': 2000,
            'strategy': 'test_strategy'
        })

df = pd.DataFrame(trades)
output_path = Path('outputs/phase4_backtest/equity_trades.csv')
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)
print(f'Created {len(df)} synthetic trades → {output_path}')
"
```

### Step 3: Execute Full Backtest

**3.1 Run Multi-Ticker Test**
```bash
# Baseline (current dataset: RELIANCE, TCS, INFY)
python run_phase4_backtest.py

# Direct integration test (enable once all 5 tickers have equity trades)
python test_phase3_integration.py --mode multi

# Or with explicit configuration
# Append ,NIFTY,BANKNIFTY to --tickers once equity trades exist
python test_phase3_integration.py \
    --mode multi \
    --tickers RELIANCE,TCS,INFY \
    --start-date 2025-04-01 \
    --end-date 2025-10-08 \
    --parallel
```

**3.2 Monitor Execution**
```bash
# In another terminal, monitor progress
tail -f outputs/phase4_backtest/options_replay/*/logs.jsonl | grep -E "(ticker_start|ticker_complete|error)"

# Monitor memory usage
watch -n 5 'free -h | grep Mem'
```

**3.3 Handle Errors**

If execution fails, check:
```bash
# Check last error in logs
tail -100 outputs/phase4_backtest/options_replay/*/logs.jsonl | grep -i error

# Check which ticker failed
ls -lt outputs/phase4_backtest/options_replay/*/

# Re-run with debug logging
python test_phase3_integration.py --mode multi --log-level DEBUG
```

### Step 4: Validate Outputs

**4.1 Check File Generation**
```bash
# List all output files
ls -lh outputs/phase4_backtest/options_replay/*/

# Verify file sizes are reasonable
du -sh outputs/phase4_backtest/options_replay/*/options_*.csv
```

**Expected Files**:
- `options_trades.csv` (400-900 rows, ~200-400 KB)
- `options_positions.csv` (50,000-100,000 rows, ~10-20 MB)
- `options_metrics.json` (~5-10 KB)
- `comparison_equity_vs_options.csv` (5-10 rows, one per ticker)
- `run_manifest.json` (~3 KB)
- `logs.jsonl` (~500 KB - 2 MB)

**4.2 Quick Data Quality Checks**
```python
import pandas as pd
import json
from pathlib import Path

# Find latest run
run_dir = sorted(Path('outputs/phase4_backtest/options_replay/').glob('*'))[-1]

# Load trades
trades = pd.read_csv(run_dir / 'options_trades.csv')
print(f"Total trades: {len(trades)}")
print(f"Tickers: {trades['ticker'].unique()}")
print(f"Date range: {trades['entry_time'].min()} to {trades['exit_time'].max()}")
print(f"Total P&L: ₹{trades['realized_pnl'].sum():,.2f}")

# Load metrics
with open(run_dir / 'options_metrics.json') as f:
    metrics = json.load(f)

print(f"\nOverall Metrics:")
print(f"  Win Rate: {metrics['summary']['win_rate_pct']:.1f}%")
print(f"  Sharpe: {metrics['summary']['sharpe_ratio']:.2f}")
print(f"  Max DD: {metrics['summary']['max_drawdown_pct']:.1f}%")

print(f"\nPer-Ticker Results:")
for ticker, data in metrics['per_ticker'].items():
    print(f"  {ticker}: {data['trades']} trades, ₹{data['pnl']:,.0f} P&L, {data['win_rate_pct']:.1f}% WR")
```

### Step 5: Generate Comparison Report

**5.1 Equity vs Options Analysis**
```python
import pandas as pd
import numpy as np

# Load options results
trades_opt = pd.read_csv(run_dir / 'options_trades.csv')

# Load or create equity baseline (if available)
# For now, simulate equity baseline
trades_opt['equity_pnl_estimated'] = trades_opt['realized_pnl'] * 0.85  # Assume options 15% better

# Per-ticker comparison
comparison = []
for ticker in trades_opt['ticker'].unique():
    ticker_data = trades_opt[trades_opt['ticker'] == ticker]

    opt_pnl = ticker_data['realized_pnl'].sum()
    opt_win_rate = (ticker_data['realized_pnl'] > 0).mean() * 100
    opt_trades = len(ticker_data)

    comparison.append({
        'ticker': ticker,
        'options_trades': opt_trades,
        'options_pnl': opt_pnl,
        'options_win_rate': opt_win_rate,
        'verdict': 'Winner' if opt_pnl > 0 else 'Loser'
    })

comparison_df = pd.DataFrame(comparison)
comparison_df = comparison_df.sort_values('options_pnl', ascending=False)

print("\n" + "="*80)
print("EQUITY VS OPTIONS COMPARISON")
print("="*80)
print(comparison_df.to_string(index=False))
print("="*80)

# Save comparison
comparison_df.to_csv(run_dir / 'comparison_equity_vs_options.csv', index=False)
```

### Step 6: Statistical Validation

**6.1 Check Sample Size Adequacy**
```python
# Calculate confidence intervals
from scipy import stats

trades = pd.read_csv(run_dir / 'options_trades.csv')

# Overall
returns = trades['return_pct'] / 100
mean_return = returns.mean()
std_return = returns.std()
n = len(returns)

# 95% confidence interval
ci_95 = stats.t.interval(0.95, n-1, mean_return, std_return / np.sqrt(n))

print(f"\nStatistical Validation:")
print(f"  Sample size: {n} trades")
print(f"  Mean return: {mean_return*100:.2f}%")
print(f"  95% CI: [{ci_95[0]*100:.2f}%, {ci_95[1]*100:.2f}%]")
print(f"  Adequate sample: {'✅ YES' if n >= 100 else '❌ NO (need 100+ trades)'}")

# Per-ticker sample sizes
print(f"\nPer-Ticker Sample Sizes:")
for ticker, count in trades['ticker'].value_counts().items():
    status = '✅' if count >= 50 else '⚠️'
    print(f"  {status} {ticker}: {count} trades")
```

**6.2 Sharpe Ratio Validation**
```python
# Validate Sharpe is not inflated by small sample
from scipy.stats import skew, kurtosis

returns = trades['return_pct'] / 100

sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
sharpe_std_error = np.sqrt((1 + 0.5 * sharpe_ratio**2) / n)

print(f"\nSharpe Ratio Analysis:")
print(f"  Sharpe: {sharpe_ratio:.2f}")
print(f"  Standard Error: ±{sharpe_std_error:.2f}")
print(f"  95% CI: [{sharpe_ratio - 1.96*sharpe_std_error:.2f}, {sharpe_ratio + 1.96*sharpe_std_error:.2f}]")
print(f"  Realistic: {'✅ YES' if 0.5 <= sharpe_ratio <= 3.0 else '❌ NO (likely overstated)'}")

# Check for skewness (options have negative skew)
print(f"\nReturn Distribution:")
print(f"  Skew: {skew(returns):.2f} (negative skew expected for long options)")
print(f"  Kurtosis: {kurtosis(returns):.2f} (fat tails expected)")
```

---

## VALIDATION: Self-Check Criteria

After the backtest completes, validate ALL of these criteria:

### Validation Checklist

#### 1. Execution Completeness ✅
```bash
# Check all tickers processed
python -c "
import pandas as pd
from pathlib import Path

run_dir = sorted(Path('outputs/phase4_backtest/options_replay/').glob('*'))[-1]
trades = pd.read_csv(run_dir / 'options_trades.csv')

expected_tickers = {'RELIANCE', 'TCS', 'INFY', 'NIFTY', 'BANKNIFTY'}
actual_tickers = set(trades['ticker'].unique())

missing = expected_tickers - actual_tickers
if not missing:
    print('✅ All 5 tickers processed')
else:
    print(f'❌ Missing tickers: {missing}')

print(f'   Tickers found: {actual_tickers}')
"
```

#### 2. Trade Volume Adequacy ✅
```python
# Minimum 400 total trades, 50+ per ticker
trades = pd.read_csv(run_dir / 'options_trades.csv')

total_trades = len(trades)
per_ticker = trades['ticker'].value_counts()

print(f"Trade Volume Check:")
print(f"  Total: {total_trades} ({'✅' if total_trades >= 400 else '❌'} >= 400)")
print(f"  Per-Ticker:")
for ticker, count in per_ticker.items():
    status = '✅' if count >= 50 else '⚠️'
    print(f"    {status} {ticker}: {count} ({'PASS' if count >= 50 else 'LOW'})")
```

#### 3. Data Quality Metrics ✅
```python
# Check pricing fallback rates
with open(run_dir / 'options_metrics.json') as f:
    metrics = json.load(f)

actual_rate = metrics['diagnostics']['actual_entry_count'] / metrics['summary']['total_trades'] * 100
fallback_rate = 100 - actual_rate

print(f"\nData Quality:")
print(f"  Actual pricing used: {actual_rate:.1f}% ({'✅' if actual_rate >= 70 else '⚠️'} target: >70%)")
print(f"  Synthetic fallback: {fallback_rate:.1f}% ({'✅' if fallback_rate <= 30 else '⚠️'} target: <30%)")
print(f"  Skipped trades: {metrics.get('skipped_trades_pct', 0):.1f}% ({'✅' if metrics.get('skipped_trades_pct', 0) <= 10 else '⚠️'} target: <10%)")
```

#### 4. Performance Realism ✅
```python
# Check metrics are in realistic ranges
summary = metrics['summary']

checks = [
    ('Win Rate', summary['win_rate_pct'], 45, 65, '%'),
    ('Sharpe Ratio', summary['sharpe_ratio'], 0.5, 2.5, ''),
    ('Max Drawdown', abs(summary['max_drawdown_pct']), 5, 30, '%'),
    ('Avg Hold Hours', summary.get('average_hold_hours', 0), 2, 72, 'h'),
]

print(f"\nPerformance Realism Check:")
for name, value, min_val, max_val, unit in checks:
    status = '✅' if min_val <= value <= max_val else '⚠️'
    print(f"  {status} {name}: {value:.1f}{unit} (expected: {min_val}-{max_val}{unit})")
```

#### 5. No Data Integrity Issues ✅
```python
# Check for common data issues
trades = pd.read_csv(run_dir / 'options_trades.csv')

issues = []

# Negative prices
neg_prices = trades[(trades['entry_price'] < 0) | (trades['exit_price'] < 0)]
if len(neg_prices) > 0:
    issues.append(f"❌ {len(neg_prices)} trades with negative prices")
else:
    print("✅ No negative option prices")

# Positions past expiry
trades['exit_time'] = pd.to_datetime(trades['exit_time'])
trades['expiry'] = pd.to_datetime(trades['expiry'])
past_expiry = trades[trades['exit_time'] > trades['expiry']]
if len(past_expiry) > 0:
    issues.append(f"❌ {len(past_expiry)} positions held past expiry")
else:
    print("✅ No positions held past expiry")

# NaN values
nan_count = trades.isnull().sum().sum()
if nan_count > 0:
    issues.append(f"❌ {nan_count} NaN values in trades")
else:
    print("✅ No NaN values in trades data")

if issues:
    print("\n⚠️  Data Integrity Issues Found:")
    for issue in issues:
        print(f"  {issue}")
```

#### 6. Cross-Ticker Consistency ✅
```python
# Verify similar trade counts across tickers (within 2x range)
per_ticker_counts = trades['ticker'].value_counts()
min_count = per_ticker_counts.min()
max_count = per_ticker_counts.max()
ratio = max_count / min_count

print(f"\nCross-Ticker Consistency:")
print(f"  Min trades: {min_count}")
print(f"  Max trades: {max_count}")
print(f"  Ratio: {ratio:.2f}x ({'✅' if ratio <= 3.0 else '⚠️'} expect < 3.0x)")
```

#### 7. Execution Performance ✅
```bash
# Check execution time from manifest
python -c "
import json
from pathlib import Path

run_dir = sorted(Path('outputs/phase4_backtest/options_replay/').glob('*'))[-1]
with open(run_dir / 'run_manifest.json') as f:
    manifest = json.load(f)

exec_time_seconds = manifest.get('execution_time_seconds', 0)
exec_time_minutes = exec_time_seconds / 60

print(f'Execution Performance:')
print(f'  Time: {exec_time_minutes:.1f} minutes ({'✅' if exec_time_minutes <= 30 else '⚠️'} target: <30 min)')
"
```

---

## SUCCESS CRITERIA: How to Know You're Done

### Phase 4 is COMPLETE when:

#### Technical Success (Must Pass ALL) ✅
1. ✅ Backtest executes on all 5 tickers without crashes
2. ✅ Processes 400+ total option trades
3. ✅ Generates all output files with correct schemas
4. ✅ No data integrity issues (negative prices, expiry violations)
5. ✅ Execution completes in <30 minutes
6. ✅ Memory usage stays under 16GB

#### Analytical Success (Must Pass 4 of 6) ✅
1. ✅ At least 2 tickers show positive options performance
2. ✅ Win rates are in realistic range (45-65%)
3. ✅ Sharpe ratios are achievable (0.5-2.5)
4. ✅ Per-ticker sample sizes adequate (50+ trades each)
5. ✅ Actual pricing used >70% of time
6. ✅ Comparison report identifies winners/losers

#### Business Value (Must Answer 3 of 4) ✅
1. ✅ Can identify which tickers benefit from options execution
2. ✅ Can explain why some tickers outperform/underperform
3. ✅ Can quantify capital efficiency (options vs equity)
4. ✅ Can recommend next steps (production rollout or parameter tuning)

### Completion Document

Create: `src/core/options/PHASE4_COMPLETE.md`

```markdown
# Phase 4 Full Backtest - COMPLETION REPORT

**Date**: 2025-10-10
**Status**: ✅ COMPLETE
**Test Mode**: Multi-ticker production validation
**Run ID**: [run_id from manifest]

---

## Execution Summary

**Configuration**:
- Tickers: RELIANCE, TCS, INFY, NIFTY, BANKNIFTY (5)
- Period: 2025-04-01 to 2025-10-08 (6 months, 26 weeks)
- Pricing Mode: Hybrid (actual preferred, synthetic fallback)
- Parallel Processing: Enabled (5 workers)
- Execution Time: [XX] minutes

**Trade Volume**:
- Total Trades: [XXX] option trades
- Per-Ticker Breakdown:
  - RELIANCE: [XX] trades
  - TCS: [XX] trades
  - INFY: [XX] trades
  - NIFTY: [XX] trades
  - BANKNIFTY: [XX] trades

---

## Performance Results

### Aggregated Metrics

- **Total P&L**: ₹[X,XXX,XXX]
- **Overall Win Rate**: [XX]%
- **Overall Sharpe Ratio**: [X.XX]
- **Max Drawdown**: -[XX]%
- **Average Hold Time**: [XX] hours
- **Capital Deployed**: ₹[X,XXX,XXX]

### Per-Ticker Performance

| Ticker | Trades | P&L (₹) | Win Rate | Sharpe | Max DD | Verdict |
|--------|--------|---------|----------|--------|--------|---------|
| RELIANCE | XXX | XXX,XXX | XX% | X.XX | -XX% | ✅ Winner |
| TCS | XXX | -XX,XXX | XX% | X.XX | -XX% | ❌ Loser |
| INFY | XXX | XXX,XXX | XX% | X.XX | -XX% | ✅ Winner |
| NIFTY | XXX | XXX,XXX | XX% | X.XX | -XX% | ⚖️ Neutral |
| BANKNIFTY | XXX | -XX,XXX | XX% | X.XX | -XX% | ❌ Loser |

---

## Equity vs Options Comparison

### Winners (Options Outperform)
1. **RELIANCE**: Options P&L +[XX]% vs equity baseline
   - Why: [Reason - e.g., high theta capture, strong directional moves]
2. **INFY**: Options P&L +[XX]% vs equity baseline
   - Why: [Reason]

### Losers (Equity Outperforms)
1. **TCS**: Options P&L -[XX]% vs equity baseline
   - Why: [Reason - e.g., theta decay faster than signal, choppy markets]
2. **BANKNIFTY**: Options P&L -[XX]% vs equity baseline
   - Why: [Reason]

### Neutral
- **NIFTY**: Approximately equal performance (±[X]%)

---

## Data Quality

**Pricing Quality**:
- Actual prices used: [XX]%
- Synthetic fallback: [XX]%
- Skipped trades: [XX]%

**Reasons for Skipped Trades**:
- Insufficient DTE: [XX] trades
- Illiquid options (OI < 100): [XX] trades
- Wide spreads (>10%): [XX] trades

---

## Statistical Validation

**Sample Size Adequacy**:
- Total trades: [XXX] ✅ Adequate (target: 400+)
- Min per-ticker: [XX] trades ({'✅' if XX >= 50 else '⚠️'} Adequate per ticker)

**Confidence Intervals** (95%):
- Mean return: [X.XX]% ± [X.XX]%
- Sharpe ratio: [X.XX] ± [X.XX]

**Distribution Characteristics**:
- Return skewness: [X.XX] (negative skew typical for long options)
- Return kurtosis: [X.XX] (fat tails indicate outlier trades)

---

## Key Insights

### What Worked
1. [Insight 1 - e.g., Options excel on RELIANCE due to high liquidity + strong trends]
2. [Insight 2]
3. [Insight 3]

### What Didn't Work
1. [Insight 1 - e.g., TCS options suffered from theta decay in choppy markets]
2. [Insight 2]

### Surprises
1. [Unexpected finding]

---

## Recommendations

### For Production Rollout
- ✅ **Recommended tickers**: RELIANCE, INFY (options show clear advantage)
- ⚠️ **Caution tickers**: NIFTY (marginal benefit)
- ❌ **Avoid tickers**: TCS, BANKNIFTY (stick to equity execution)

### For Parameter Tuning (Future Phase 5)
1. Test ATM vs OTM strike selection on RELIANCE (may improve returns)
2. Test monthly vs weekly expiries on TCS (reduce theta drag)
3. Implement stop-loss on BANKNIFTY (limit drawdowns)

### For Data Quality
- Priority: Improve options data coverage for [ticker] (XX% synthetic usage)
- Consider: Expanding backtest period to 12 months for more statistical power

---

## Next Steps

✅ **Phase 4 Complete** - Full backtest validated
🎯 **Next Phase**: Phase 5 (optional) - Parameter optimization & sensitivity analysis
📊 **Deliverable**: Present comparison report to stakeholders
🚀 **Production**: Ready to implement live on RELIANCE & INFY (with paper trading validation)

---

## Validation Checklist

- [x] All 5 tickers processed successfully
- [x] 400+ total trades generated
- [x] No data integrity issues
- [x] Performance metrics in realistic ranges
- [x] At least 2 tickers show positive options performance
- [x] Statistical sample sizes adequate
- [x] Execution time acceptable
- [x] Comparison report generated
- [x] Insights documented
- [x] Recommendations provided

---

**Completion Timestamp**: [YYYY-MM-DD HH:MM:SS]
**Phase 4 Status**: ✅ **PRODUCTION READY**
```

---

## TROUBLESHOOTING

### Issue 1: "Ticker processing failed"

**Error**:
```
ERROR: Failed to process ticker TCS: FileNotFoundError: Option data not found
```

**Diagnosis**:
```bash
# Check which ticker failed
grep "error" outputs/phase4_backtest/options_replay/*/logs.jsonl | grep ticker

# Check option data availability
ls data/pools/options/2025-04-01_to_2025-10-08/TCS/1day/ | wc -l
```

**Fix**:
- If specific expiries missing: Adjust test period or skip that ticker
- If data format issue: Verify parquet files are readable
- Re-run with `--tickers` excluding failed ticker to complete others

### Issue 2: "Out of memory"

**Error**:
```
MemoryError: Unable to allocate array
```

**Diagnosis**:
```bash
# Check memory usage during execution
ps aux | grep python | awk '{print $6/1024 " MB"}'
```

**Fix**:
1. Reduce parallel workers:
   ```python
   config.replay.max_workers = 2  # Instead of 5
   ```
2. Process tickers sequentially:
   ```bash
   python test_phase3_integration.py --mode multi --no-parallel
   ```
3. Free up memory:
   ```bash
   # Clear caches
   sync; echo 3 > /proc/sys/vm/drop_caches
   ```

### Issue 3: "Execution too slow"

**Symptom**:
```
Execution time: 65 minutes (exceeds 30-minute target)
```

**Diagnosis**:
```bash
# Check if parallel processing enabled
grep "parallel" outputs/phase4_backtest/options_replay/*/run_manifest.json

# Check CPU usage
top -n 1 | grep python
```

**Fix**:
1. Enable parallel processing if disabled
2. Reduce position tracking granularity (if Phase 4+ feature exists):
   ```python
   config.replay.greeks_calculation_frequency = 'hourly'  # Instead of 'every_bar'
   ```
3. Profile slow sections:
   ```bash
   python -m cProfile -o profile.stats test_phase3_integration.py --mode multi
   ```

### Issue 4: "Low trade counts"

**Symptom**:
```
Total trades: 85 (expected: 400+)
```

**Diagnosis**:
```bash
# Check skipped trade reasons
cat outputs/phase4_backtest/options_replay/*/logs.jsonl | grep "trade_skipped" | jq .reason | sort | uniq -c
```

**Common Reasons**:
- Insufficient equity trades as input (need ~500 equity trades for ~400 option trades)
- Strict DTE filters (min_dte_to_enter = 3)
- Liquidity filters rejecting illiquid options

**Fix**:
1. Generate more synthetic equity trades (Step 2)
2. Relax DTE filter:
   ```yaml
   # options_config.yaml
   min_dte_to_enter: 1  # Was 3
   ```
3. Relax liquidity filters:
   ```yaml
   min_open_interest: 50  # Was 100
   max_spread_pct: 0.15  # Was 0.10
   ```

### Issue 5: "Unrealistic Sharpe ratio"

**Symptom**:
```
Overall Sharpe: 8.52 (suspiciously high, expected: 0.5-2.5)
```

**Diagnosis**:
```python
# Check if sample size too small
trades = pd.read_csv('outputs/.../options_trades.csv')
print(f"Trade count: {len(trades)}")

# Check if all trades positive (too good to be true)
print(f"Win rate: {(trades['realized_pnl'] > 0).mean() * 100:.1f}%")
```

**Root Causes**:
1. Small sample size (<100 trades) → inflated metrics
2. Cherry-picked time period (only bull market)
3. Lookahead bias in signals
4. Synthetic prices too optimistic

**Fix**:
- Increase sample size (more trades/longer period)
- Include bearish periods in backtest
- Validate equity signals don't have lookahead
- Compare actual vs synthetic pricing usage

---

## IMPLEMENTATION WORKFLOW

### Recommended Steps (In Order)

**Step 1** (15 minutes): **Pre-Flight Checks**
- Validate data availability for all 5 tickers
- Check system resources (memory, disk)
- Verify configuration is correct

**Step 2** (30 minutes): **Generate Test Trades** (if needed)
- Create synthetic equity trades covering 6 months
- ~100 trades per ticker = 500 total
- Save to `outputs/phase4_backtest/equity_trades.csv`

**Step 3** (20-30 minutes): **Execute Backtest**
- Run `python test_phase3_integration.py --mode multi`
- Monitor progress in logs
- Wait for completion

**Step 4** (30 minutes): **Validate Outputs**
- Run all 7 validation checks
- Verify trade counts, data quality, performance metrics
- Check for data integrity issues

**Step 5** (45 minutes): **Analyze Results**
- Generate comparison report
- Calculate per-ticker performance
- Statistical validation (confidence intervals)
- Identify winners/losers

**Step 6** (30 minutes): **Document Completion**
- Create `PHASE4_COMPLETE.md`
- Fill in all metrics and insights
- Add recommendations for production

**Step 7** (30 minutes): **Stakeholder Report** (optional)
- Prepare presentation slides
- Highlight key findings (which tickers benefit)
- Quantify improvement (options vs equity)

**Total Time**: ~3-4 hours (optimistic), ~6-8 hours (realistic with debugging)

---

## CONTEXT FOR SELF-VALIDATION

### What "Success" Looks Like

You'll know Phase 4 is successful when:

1. **Test Completes Clean**:
   ```
   ✅ ✅ ✅  PHASE 4 INTEGRATION TEST PASSED  ✅ ✅ ✅
   ```

2. **Trade Volumes Make Sense**:
   - Total: 400-900 trades ✅
   - Per-ticker: 50-200 trades ✅
   - Skipped rate: <10% ✅

3. **Performance is Realistic**:
   - Win rate: 45-65% (not 80-90%)
   - Sharpe: 0.5-2.5 (not 5.0+)
   - Max DD: 10-30% (not 2%)

4. **Clear Winners/Losers Identified**:
   - At least 2 tickers show options advantage
   - At least 1 ticker shows equity advantage
   - Can explain why each behaves differently

5. **Data Quality is High**:
   - Actual pricing: >70%
   - No integrity issues
   - Execution time: <30 minutes

6. **You Can Answer Business Questions**:
   - "Should we trade options on RELIANCE?" → YES (if winner)
   - "What's the expected Sharpe?" → X.XX (from backtest)
   - "What's the risk?" → Max DD: -XX%, capital: ₹XXX

### What "Failure" Looks Like

Test is NOT successful if:

❌ **Execution Failures**:
- Crashes mid-run
- Only processes 1-2 tickers
- Takes >60 minutes

❌ **Data Issues**:
- <200 total trades (insufficient sample)
- Negative prices or expiry violations
- >30% synthetic fallback rate

❌ **Unrealistic Results**:
- Win rate >75% (too good to be true)
- Sharpe >4.0 (inflated by small sample)
- All 5 tickers show same behavior (lack of diversification)

❌ **No Actionable Insights**:
- Can't identify winners/losers
- Can't explain performance differences
- No clear recommendation for production

---

## EXPECTED TIMELINE

### Optimistic (First-Try Success): 3-4 hours
- ✅ Pre-flight checks: 15 min
- ✅ Generate trades: 30 min
- ✅ Execute backtest: 20 min
- ✅ Validate outputs: 30 min
- ✅ Analyze results: 45 min
- ✅ Document completion: 30 min
- ✅ Review & publish: 30 min

### Realistic (1-2 Issues): 6-8 hours
- ✅ Pre-flight checks: 15 min
- ✅ Generate trades: 30 min
- ⚠️ Execute backtest: 30 min (first try fails)
- ⚠️ Debug issue: 60 min
- ✅ Re-run backtest: 20 min
- ✅ Validate outputs: 30 min
- ⚠️ Fix data issue: 30 min
- ✅ Analyze results: 60 min
- ✅ Document completion: 45 min
- ✅ Review & iterate: 45 min

### Pessimistic (Multiple Issues): 12-16 hours
- Multiple execution failures
- Memory/performance tuning required
- Data quality issues requiring re-fetch
- Statistical analysis reveals issues
- Multiple documentation iterations

---

## FINAL CHECKLIST

Before marking Phase 4 complete, ensure:

### Technical Validation
- [ ] Backtest executed on all 5 tickers without crashes
- [ ] Processed 400+ total option trades
- [ ] Generated all output files with correct schemas
- [ ] No data integrity issues (negative prices, expiry violations, NaNs)
- [ ] Execution time <30 minutes
- [ ] Memory usage <16GB

### Analytical Validation
- [ ] At least 2 tickers show positive options performance
- [ ] Win rates in realistic range (45-65%)
- [ ] Sharpe ratios achievable (0.5-2.5)
- [ ] Per-ticker sample sizes adequate (50+ each)
- [ ] Actual pricing used >70% of time
- [ ] Statistical confidence intervals calculated

### Business Validation
- [ ] Identified which tickers benefit from options
- [ ] Explained why some outperform/underperform
- [ ] Quantified capital efficiency vs equity
- [ ] Provided production rollout recommendations
- [ ] Comparison report generated and reviewed
- [ ] Key insights documented

### Documentation
- [ ] `PHASE4_COMPLETE.md` created with all metrics
- [ ] Stakeholder report prepared (if required)
- [ ] Next steps clearly defined
- [ ] Code committed to version control (if applicable)

---

## ACCEPTANCE CRITERIA (From Implementation Plan)

This task fulfills **Phase 4 Success Criteria** from `implementation_plan.md`:

| Criterion | Status | How Validated |
|-----------|--------|---------------|
| Complete backtest on 5 tickers, 6 months | ☐ | All tickers processed, date range correct |
| Generate comparison report | ☐ | `comparison_equity_vs_options.csv` exists |
| Identify at least 2 tickers where options outperform | ☐ | Comparison shows clear winners |
| Sensitivity analysis reveals optimal parameters | ☐ | (Optional for Phase 4, required for Phase 5) |
| Results are reproducible | ☐ | Same inputs → same outputs validated |

**Completion Definition**: First 4 criteria must pass. Sensitivity analysis optional (can be Phase 5).

---

## ADDITIONAL RESOURCES

### Files to Reference

1. **Implementation Plan**: `src/core/options/planning/implementation_plan.md`
   - Phase 4 specification (lines 630-759)
   - Success criteria
   - Expected outputs

2. **Phase 3 Completion**: `src/core/options/PHASE3_COMPLETE.md`
   - MVP results for comparison
   - Baseline performance metrics

3. **Decisions Log**: `src/core/options/planning/decisions.md`
   - ADR-001: Decoupled replay architecture
   - ADR-002: Hybrid pricing mode
   - ADR-003: Fixed 1-lot strategy

4. **Integration Test Script**: `test_phase3_integration.py`
   - `--mode multi` implementation
   - Multi-ticker orchestration logic

### Key Code Locations

- Replay engine: `src/core/options/replay/engine.py`
- Multi-ticker orchestration: `test_phase3_integration.py` (lines for multi mode)
- Metrics calculation: `src/core/options/replay/metrics.py`
- Comparison analytics: (to be generated by analysis scripts)
- Config: `src/core/options/config/options_config.yaml`

### Data Locations

- Equity data: `data/pools/2022-01-01_to_2025-08-31/{TICKER}/15m.parquet`
- Options data: `data/pools/options/2025-04-01_to_2025-10-08/{TICKER}/1day/`
- Test outputs: `outputs/phase4_backtest/options_replay/`

---

## SUMMARY

**What**: Execute full-scale production backtest on 5 tickers over 6 months

**Why**: Validate system at scale and identify which tickers benefit from options execution

**How**:
1. Pre-flight checks (data availability, system resources)
2. Generate synthetic trades (if needed, ~500 trades)
3. Execute multi-ticker backtest (`--mode multi`)
4. Validate outputs (7-point checklist)
5. Analyze results (equity vs options comparison)
6. Document completion (`PHASE4_COMPLETE.md`)

**Success Metrics**:
- ✅ 400+ trades processed across 5 tickers
- ✅ At least 2 tickers show options advantage
- ✅ Performance metrics in realistic ranges (Sharpe 0.5-2.5, WR 45-65%)
- ✅ Execution time <30 minutes
- ✅ Clear recommendations for production rollout

**Time**: 3-4 hours (optimistic), 6-8 hours (realistic with debugging)

**Outcome**: **PRODUCTION-READY SYSTEM** with data-driven recommendations ✅

---

**Execute this Phase 4 backtest to validate your options strategy at scale. Good luck!** 🚀
