# ARCHITECTURAL ANALYSIS: Backtest vs Live Parity
**Critical Investigation of Strategy Unification & Bias Prevention**

Date: October 15, 2025  
Status: CRITICAL ISSUES FOUND   
Author: Phase 6 Parity Validation Review

---

##  EXECUTIVE SUMMARY

**CRITICAL FINDING:** The current architecture has **TWO COMPLETELY SEPARATE strategy implementations**:
1. **Backtester**: `mse_strategy_backtesting.py` (768 lines) - Sophisticated, bias-free
2. **Live Trading**: `mse_strategy.py` (876 lines) - Different implementation

**This violates the fundamental principle of strategy unification and creates parity risks.**

---

##  QUESTIONS INVESTIGATED

### Q1: Who Defines Warmup Period - Strategy or System?

**Answer:** **BOTH** (and that's a problem!)

**Backtester:**
```python
# mse_strategy_backtesting.py line 55-60
self.warmup_periods = {
    '5m': 175,   # 35 periods * 5 minutes
    '15m': 525   # 35 periods * 15 minutes (CRITICAL)
}
```
 **Strategy explicitly defines 525-minute warmup**

**Live Trading:**
```python
# mse_strategy.py line 100-108
self._requirements = StrategyRequirements(
    timeframes=["5min", "15min"],
    minimum_candles={
        "5min": 40,
        "15min": 40
    }
    # NO warmup_minutes defined!
)
```
 **Strategy does NOT define warmup_minutes**  
 **Defaults to 0, then falls back to system default of 60 minutes**

**PARITY VIOLATION:**
- Backtester: 525 minutes warmup (strategy-defined)
- Live: 60 minutes warmup (system default fallback)
- **Risk:** MACD indicators not fully stable in live trading

**FIX REQUIRED:**
```python
# mse_strategy.py line 100
self._requirements = StrategyRequirements(
    timeframes=["5min", "15min"],
    minimum_candles={"5min": 40, "15min": 40},
    warmup_minutes=525  # ADD THIS LINE
)
```

---

### Q2: Is There a Unified Strategy Implementation?

**Answer:** **NO - TWO SEPARATE IMPLEMENTATIONS **

| Aspect | Backtester | Live Trading | Match? |
|--------|-----------|--------------|--------|
| **File** | `mse_strategy_backtesting.py` | `mse_strategy.py` |  Different |
| **Lines of Code** | 768 | 876 |  Different size |
| **Base Class** | `StrategyBase` | `StrategyInterface` |  Different base |
| **Warmup Definition** | Explicit (525min) | Missing (defaults to 060) |  Mismatch |
| **Look-Ahead Prevention** | `.shift(1)` everywhere | Unknown |  Needs verification |
| **Two-Bar Rule** | Explicit pending flags | Unknown |  Needs verification |
| **Entry Logic** | 4-indicator system | Unknown |  Needs verification |
| **Exit Logic** | 80% peak/valley | Unknown |  Needs verification |

**CRITICAL ISSUE:** These are NOT the same strategy running in different environments.  
They are **two different codebases** that are supposed to produce identical signals.

---

### Q3: Does Backtester Use Two-Bar Execution Rule?

**Answer:** **YES  - Properly Implemented**

**Implementation Details:**

```python
# mse_strategy_backtesting.py line 59-66
# Trade state variables
self.in_buy_trade = False
self.in_sell_trade = False
self.buy_entry_pending = False   # Two-bar rule flag
self.sell_entry_pending = False  # Two-bar rule flag
self.buy_exit_pending = False
self.sell_exit_pending = False
```

**Execution Flow:**

**Bar N (Signal Detection):**
```python
# Line 230-242: Use .shift(1) to detect signal on PREVIOUS bar data
buy_signal_raw = (
    (merged_df['5m_macd_line'].shift(1) > merged_df['5m_signal_line'].shift(1)) &  # Previous bar
    (merged_df['5m_ema_9'].shift(1) > merged_df['5m_ema_20'].shift(1)) &
    # ... all 4 indicators use .shift(1)
)

# Set pending flag
self.buy_entry_pending = True  # Will execute NEXT bar
```

**Bar N+1 (Execution):**
```python
# Line 428-438: Execute at OPEN of next bar
if self.buy_entry_pending and not self.in_buy_trade:
    entry_price = row['open']  # NEXT bar's open price
    self.in_buy_trade = True
    self.buy_entry_pending = False
    # Record trade entry
```

**Verification:**
-  Signal detected on previous bar data (`.shift(1)`)
-  Pending flag set (not executed immediately)
-  Execution at next bar's open price
-  Realistic slippage modeled

**Does Live Trading Use Two-Bar Rule?**
 **UNKNOWN - Needs Verification**

---

### Q4: Is There Look-Ahead Bias Prevention?

**Answer:** **YES in Backtester  - Unknown in Live **

**Backtester Implementation:**

**Every single indicator comparison uses `.shift(1)`:**

```python
# Line 232-242: Entry signal generation
buy_signal_raw = (
    (merged_df['5m_macd_line'].shift(1) > merged_df['5m_signal_line'].shift(1)) &  #  Previous bar
    (merged_df['5m_ema_9'].shift(1) > merged_df['5m_ema_20'].shift(1)) &          #  Previous bar
    (merged_df['15m_macd_line'].shift(1) > merged_df['15m_signal_line'].shift(1)) & #  Previous bar
    (merged_df['15m_ema_9'].shift(1) > merged_df['15m_ema_20'].shift(1))          #  Previous bar
)
```

**Exit signal uses previous bar MACD histogram:**

```python
# Line 342-348: Exit logic
# Use previous bar MACD histogram (bias prevention)
prev_hist = merged_df['5m_macd_hist'].shift(1).iloc[i]
threshold_hist = self.buy_max_hist * self.exit_threshold

if prev_hist <= threshold_hist:
    self.buy_exit_pending = True  # Signal exit on next bar
```

**Warmup Period Enforcement:**

```python
# Line 406-413: Skip first 525 minutes
warmup_min = max(self.warmup_periods.values())  # 525 minutes
cutoff_ts = df['timestamp'].min() + pd.Timedelta(minutes=warmup_min)

for i, (_, row) in enumerate(df.iterrows()):
    if row['timestamp'] < cutoff_ts:
        continue  # Skip warmup period
```

**Comprehensive Bias Prevention:**
-  All indicator comparisons use `.shift(1)`
-  Entry price is NEXT bar's open (not current bar's close)
-  Exit price is NEXT bar's open (not current bar where signal triggered)
-  525-minute warmup skipped (no trades during indicator stabilization)
-  Peak tracking initialized AFTER entry completion
-  No future data leakage

**Does Live Trading Prevent Look-Ahead Bias?**
 **UNKNOWN - Needs Verification**

Live trading operates in real-time, so look-ahead bias is naturally prevented by time itself. However, the signal generation logic must match the backtester's `.shift(1)` approach for parity.

---

### Q5: How Are Indicators Calculated in Both Environments?

**Answer:** **NEEDS INVESTIGATION **

**Backtester Indicator Calculation:**

```python
# mse_strategy_backtesting.py line 141-180
def prepare_data(self, data, ticker, pull_date):
    # Calculate MACD for 5min timeframe
    df_5m['5m_macd_line'], df_5m['5m_signal_line'], df_5m['5m_macd_hist'] = self._calculate_macd(
        df_5m['close'], 
        fast=12, slow=26, signal=9
    )
    
    # Calculate EMA for 5min timeframe
    df_5m['5m_ema_9'] = df_5m['close'].ewm(span=9, adjust=False).mean()
    df_5m['5m_ema_20'] = df_5m['close'].ewm(span=20, adjust=False).mean()
    
    # Same for 15min timeframe
    # ... (lines 160-180)
```

**Live Trading Indicator Calculation:**
 **LOCATION UNKNOWN - Needs Investigation**

**Critical Questions:**
1. Where does live trading calculate MACD and EMA?
2. Does it use the same parameters (12, 26, 9)?
3. Does it use the same pandas methods (`.ewm()`)?
4. How are 15-minute indicators derived from 5-minute data?
5. Are indicators recalculated on every bar or cached?

**PARITY RISK:**
If live trading uses different indicator calculation methods (e.g., different library, different rounding), signals will diverge even if logic is identical.

**RECOMMENDATION:**
Extract indicator calculation into a **shared utility module** used by both backtester and live trading.

---

### Q6: How Does a Single Strategy Serve Both Backtester and Live?

**Answer:** **IT DOESN'T - THEY ARE SEPARATE IMPLEMENTATIONS **

**Current Reality:**

```
backtester/
 src/strategies/
     mse_strategy_backtesting.py    # 768 lines, bias-free

trading_system_clean/live_module/
 src/strategies/
     mse_strategy.py                # 876 lines, different base class
```

**These are NOT the same strategy.**

**Ideal Architecture (NOT CURRENT):**

```
shared_strategies/
 mse_strategy.py                    # Single implementation

backtester/
 adapters/
     strategy_adapter.py            # Adapter for backtesting env

live_trading/
 adapters/
     strategy_adapter.py            # Adapter for live env
```

**The Problem:**

Currently, the "unified strategy" is a **myth**. We have:
- Two separate strategy files
- Two separate base classes
- Two separate sets of parameters
- Two separate implementations of the same logic

**This creates maintenance burden and parity risk:**
- Bug fix in backtester  must manually fix in live
- Parameter change in one  must manually sync to other
- Different developers may modify each independently
- No guarantee signals will match

---

##  CRITICAL FINDINGS SUMMARY

###  Issues Found:

1. **Warmup Period Mismatch**
   - Backtester: 525 minutes (strategy-defined)
   - Live: 60 minutes (system default)
   - **Impact:** MACD not stable in live, signals diverge

2. **No Strategy Unification**
   - Two completely separate implementations
   - Different base classes
   - Different file locations
   - **Impact:** Code duplication, maintenance burden

3. **Unknown Live Trading Behavior**
   - Look-ahead prevention unknown
   - Two-bar rule unknown
   - Indicator calculation unknown
   - **Impact:** Cannot guarantee parity

4. **No Shared Indicator Library**
   - Each implementation calculates indicators separately
   - No guarantee of identical calculations
   - **Impact:** Potential signal divergence

###  Things Done Right:

1. **Backtester Bias Prevention**
   - Comprehensive `.shift(1)` usage
   - Two-bar execution rule
   - 525-minute warmup
   - Entry at next bar's open

2. **Backtester Documentation**
   - Clear comments explaining bias prevention
   - Explicit warmup periods
   - Well-structured code

3. **Backtester Testing**
   - Can be validated against historical data
   - Reproducible results

---

##  RECOMMENDED FIXES

### Priority 1: Fix Warmup Period in Live Trading (URGENT)

**File:** `trading_system_clean/live_module/src/strategies/mse_strategy.py`

```python
# Line 100-108: CURRENT
self._requirements = StrategyRequirements(
    timeframes=["5min", "15min"],
    minimum_candles={"5min": 40, "15min": 40}
)

# CHANGE TO:
self._requirements = StrategyRequirements(
    timeframes=["5min", "15min"],
    minimum_candles={"5min": 40, "15min": 40},
    warmup_minutes=525  # CRITICAL: Must match backtester for MACD stability
)
```

### Priority 2: Verify Live Trading Signal Generation

**Investigate:**
1. Does live MSE strategy use `.shift(1)` or equivalent?
2. Does it implement two-bar execution?
3. Are indicators calculated identically?

**File to Check:** `trading_system_clean/live_module/src/strategies/mse_strategy.py`  
**Lines to Review:** Entry signal generation logic

### Priority 3: Create Shared Indicator Library

**New Module:** `shared/indicators.py`

```python
def calculate_macd(close_prices, fast=12, slow=26, signal=9):
    """
    Calculate MACD with consistent methodology for backtest and live.
    
    MUST be used by both environments to ensure parity.
    """
    # Single implementation used everywhere
    pass

def calculate_ema(prices, span):
    """Calculate EMA with consistent methodology."""
    pass
```

### Priority 4: Strategy Unification (Long-term)

**Option A: Adapter Pattern**
- Keep single strategy logic
- Create environment-specific adapters
- Strategy calls adapter methods for data access

**Option B: Shared Base Class**
- Create `UnifiedStrategyBase`
- Both backtester and live inherit from it
- Enforce identical signal generation

**Option C: Single Strategy File** (RECOMMENDED)
- Move to shared location
- Environment detection at runtime
- Both systems import same file

---

##  PARITY VALIDATION CHECKLIST

Use this checklist to verify backtest/live parity:

### Warmup Period:
- [ ] Backtester uses 525 minutes  VERIFIED
- [ ] Live uses 525 minutes  NEEDS FIX
- [ ] Strategy defines warmup (not system default)  NEEDS FIX

### Look-Ahead Bias Prevention:
- [ ] Backtester uses `.shift(1)`  VERIFIED
- [ ] Live uses `.shift(1)` or equivalent  NEEDS VERIFICATION
- [ ] Entry price is NEXT bar's open  VERIFIED (backtester)
- [ ] Entry price is NEXT bar's open  NEEDS VERIFICATION (live)

### Two-Bar Execution:
- [ ] Backtester implements pending flags  VERIFIED
- [ ] Live implements pending flags  NEEDS VERIFICATION
- [ ] Signal  Pending  Execute flow  VERIFIED (backtester)
- [ ] Signal  Pending  Execute flow  NEEDS VERIFICATION (live)

### Indicator Calculation:
- [ ] MACD parameters identical (12, 26, 9)  NEEDS VERIFICATION
- [ ] EMA parameters identical (9, 20)  NEEDS VERIFICATION
- [ ] Calculation method identical  NEEDS VERIFICATION
- [ ] Same library/functions used  NO (separate implementations)

### Strategy Parameters:
- [ ] Exit threshold identical (0.80)  NEEDS VERIFICATION
- [ ] Entry cooldown identical  NEEDS VERIFICATION
- [ ] Position size rules identical  NEEDS VERIFICATION

### Code Structure:
- [ ] Same base class  NO (StrategyBase vs StrategyInterface)
- [ ] Same file location  NO (separate files)
- [ ] Shared indicator library  NO
- [ ] Unified strategy implementation  NO

---

##  NEXT STEPS FOR PHASE 7

Before proceeding to Phase 7 (Documentation), we MUST:

1. **URGENT:** Fix warmup period in live MSE strategy (Priority 1)
2. **CRITICAL:** Verify live trading signal generation logic (Priority 2)
3. **IMPORTANT:** Document current parity gaps (this document)
4. **RECOMMENDED:** Create shared indicator library (Priority 3)
5. **LONG-TERM:** Plan strategy unification (Priority 4)

**Phase 7 Can Proceed IF:**
- Warmup period is fixed
- Live signal generation is verified to match backtester
- Parity gaps are documented and accepted as technical debt

**Phase 7 Should WAIT IF:**
- Significant parity violations are found
- Live trading cannot be verified
- Risk of signal divergence is high

---

##  CONCLUSION

**Current Status:**  **PARITY AT RISK**

The system has:
-  Excellent backtester with bias prevention
-  Separate live trading implementation
-  Warmup period mismatch
-  Unknown live trading behavior

**Recommendation:** **Fix Priority 1 and 2 before Phase 7.**

The architectural ideal of "one strategy, two environments" is **NOT currently achieved**. This is technical debt that should be addressed for production readiness.

---

**Document Version:** 1.0  
**Date:** October 15, 2025  
**Status:** Pending User Decision on Phase 7 Proceeding
