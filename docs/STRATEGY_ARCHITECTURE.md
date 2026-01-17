# Strategy Architecture Summary

> **Purpose:** Quick reference for agents. Read this FIRST before reading any strategy files.
> This saves tokens by providing structure overview without reading full source code.

## Quick Reference

### File Locations (Windows Paths)
```
D:\Balcony\Trading\unified_trading_setup\backtester\
├── src\strategies\
│   ├── support\
│   │   ├── strategy_base.py    # Base class - inherit from this
│   │   ├── indicator_registry.py # Indicator calculations
│   │   ├── exit_manager.py     # Exit logic handling
│   │   └── register_strategies.py # Strategy registration
│   ├── strategy_mac.py         # Example: Multi-timeframe MACD+EMA
│   ├── strategy_template.py    # Template for new strategies
│   └── [your_strategy].py      # New strategies go here
├── config\templates\
│   ├── strategy_mac.yaml       # Example YAML config
│   └── [your_strategy].yaml    # New configs go here
└── docs\
    └── STRATEGY_ARCHITECTURE.md # This file
```

## StrategyBase Interface (REQUIRED)

Every strategy MUST inherit from `StrategyBase` and implement:

```python
from src.strategies.support.strategy_base import StrategyBase

class YourStrategy(StrategyBase):
    def __init__(self, name: str, parameters: dict = None, config=None):
        super().__init__(name, parameters, config)
        # Set required_timeframes = ["5m"] or ["5m", "15m"]
        # Set warmup_periods = {"5m": 50}

    def prepare_data(self, data, ticker, pull_date) -> dict:
        """
        Input: Dict of DataFrames by timeframe {"5m": df, "15m": df}
        Output: Same dict with indicators attached

        Must do:
        1. Apply indicators via self._apply_configured_indicators(data)
        2. Drop warmup bars: df.iloc[warmup_bars:]
        3. Return dict of prepared DataFrames
        """
        pass

    def generate_signals(self, data) -> pd.DataFrame:
        """
        Input: Dict of prepared DataFrames
        Output: Single DataFrame with signal columns

        Must have columns:
        - entry_signal_buy: bool (True = buy signal)
        - entry_signal_sell: bool (True = sell signal)
        - exit_signal_buy: bool (True = exit long)
        - exit_signal_sell: bool (True = exit short)

        CRITICAL: Use .shift(1) to prevent look-ahead bias!
        Signals generated at bar N execute at bar N+1.
        """
        pass
```

## YAML Config Structure (REQUIRED)

```yaml
strategy:
  name: strategy_your_name       # Must match Python class registration
  description: "Brief description"
  risk_profile: moderate         # conservative|moderate|aggressive
  timeframes:
    entry: [5m]                  # Timeframes for entry signals
    exit: [5m]                   # Timeframes for exit signals
  parameters:
    entry_timeframe: 5m
    # Your custom parameters here
    intraday_cutoff: "15:15"     # No entries after this time
    min_volume: 0                # Minimum volume filter

  indicators:
    entry:
      - name: my_sma
        type: sma                # See Indicator Types below
        timeframe: 5m
        params:
          period: 20

# Standard sections (copy as-is for new strategies)
risk:
  enabled: false
  bypass_mode: true

transaction:
  enabled: false

validation:
  enabled: true
  lookahead_bias_check: true

output:
  save_trades: true
  save_signals: true
  save_metrics: true
```

## Available Indicator Types

| Type | Parameters | Description |
|------|------------|-------------|
| `sma` | period | Simple Moving Average |
| `ema` | period | Exponential Moving Average |
| `rsi` | period | Relative Strength Index |
| `macd_val_X_Y_Z` | - | MACD line (fast=X, slow=Y, signal=Z) |
| `macd_signal_X_Y_Z` | - | MACD signal line |
| `macd_histogram_X_Y_Z` | - | MACD histogram |
| `atr` | period | Average True Range |
| `bollinger_upper` | period, std | Upper Bollinger Band |
| `bollinger_lower` | period, std | Lower Bollinger Band |

## Critical Rules

### 1. No Look-Ahead Bias
```python
# WRONG - uses current bar data to generate current bar signal
df["signal"] = df["fast_sma"] > df["slow_sma"]

# CORRECT - uses previous bar data (signal on bar N executes on bar N+1)
df["signal"] = df["fast_sma"].shift(1) > df["slow_sma"].shift(1)
```

### 2. Warmup Period
- Must be >= longest indicator period
- Typically: longest_period + 10 bars buffer
- Example: 50-period SMA → warmup = 60 bars

### 3. Session Cutoff (Indian Markets)
- Last entry: 15:15 IST
- Market close: 15:30 IST
- Square off all positions before 15:30

### 4. Registration
Add to `register_strategies.py`:
```python
from src.strategies.strategy_your_name import YourStrategy
STRATEGY_REGISTRY["strategy_your_name"] = YourStrategy
```

## Example: Simple SMA Crossover

### Minimal YAML
```yaml
strategy:
  name: strategy_sma_simple
  timeframes:
    entry: [5m]
  parameters:
    fast_period: 20
    slow_period: 50
  indicators:
    entry:
      - name: fast_sma
        type: sma
        timeframe: 5m
        params:
          period: 20
      - name: slow_sma
        type: sma
        timeframe: 5m
        params:
          period: 50
```

### Minimal Python
```python
class StrategySMASimple(StrategyBase):
    def __init__(self, name, parameters=None, config=None):
        super().__init__(name, parameters or {}, config)
        self.fast = self.parameters.get("fast_period", 20)
        self.slow = self.parameters.get("slow_period", 50)
        self.required_timeframes = ["5m"]
        self.warmup_periods = {"5m": self.slow + 10}

    def prepare_data(self, data, ticker, pull_date):
        frames = self._apply_configured_indicators(data)
        for tf, df in frames.items():
            warmup = self.warmup_periods.get(tf, 0)
            frames[tf] = df.iloc[warmup:].reset_index(drop=True)
        return frames

    def generate_signals(self, data):
        df = data["5m"].copy()
        fast_col = f"5m_sma_{self.fast}"
        slow_col = f"5m_sma_{self.slow}"

        # SHIFT(1) prevents look-ahead bias!
        df["entry_signal_buy"] = (
            (df[fast_col].shift(1) > df[slow_col].shift(1)) &
            (df[fast_col].shift(2) <= df[slow_col].shift(2))  # crossover
        )
        df["entry_signal_sell"] = (
            (df[fast_col].shift(1) < df[slow_col].shift(1)) &
            (df[fast_col].shift(2) >= df[slow_col].shift(2))
        )
        df["exit_signal_buy"] = df["entry_signal_sell"]
        df["exit_signal_sell"] = df["entry_signal_buy"]
        return df
```

## When to Read Full Files

| If you need... | Read this file |
|----------------|----------------|
| Full StrategyBase methods | `strategy_base.py` |
| Complex multi-timeframe logic | `strategy_mac.py` |
| Indicator calculation details | `indicator_registry.py` |
| YAML parsing logic | `strategy_factory.py` |

**For simple strategies:** This summary should be sufficient. Only read full files if you need specific implementation details not covered here.
