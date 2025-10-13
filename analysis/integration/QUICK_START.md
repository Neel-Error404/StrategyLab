# Trade Enhancement Integration - Quick Start

## For Analysis Script Authors

### Simple API - User Provides Paths

```python
from analysis.integration import enhance_trades

# User specifies exactly where their data is
enhanced_data = enhance_trades(
    trade_file="path/to/your/trades.csv",
    base_data_dir="path/to/your/base_data"
)
```

**No auto-detection, no guessing - user knows where their data is!**

## Required User Inputs

### 1. Trade File Path
```python
trade_file = "/path/to/all_trade_merged.csv"
```

**Must contain columns:**
- `ticker` - Stock symbol
- `Entry Time` - Trade entry timestamp
- `Exit Time` - Trade exit timestamp
- `Profit (Currency)` - Trade P&L
- `Trade Type` - BUY/SELL

### 2. Base Data Directory Path
```python
base_data_dir = "/path/to/outputs/.../data/base_data"
```

**Must contain files:**
- `TICKER1_Base_*.csv`
- `TICKER2_Base_*.csv`
- etc.

## Complete Example

```python
from analysis.integration import enhance_trades

def analyze_with_enhancement():
    # USER PROVIDES EXACT PATHS
    trade_file = "/mnt/batch/.../outputs/.../all_trade_merged.csv"
    base_data_dir = "/mnt/batch/.../outputs/.../data/base_data"

    # ONE FUNCTION CALL
    enhanced = enhance_trades(trade_file, base_data_dir)

    # NOW ANALYZE WITH 20+ NEW COLUMNS
    return enhanced.groupby('ticker').agg({
        'Profit (Currency)': 'mean',           # Original
        'macd_change': 'mean',                 # NEW: MACD discipline
        'trade_duration_minutes': 'mean',      # NEW: Time efficiency
        'entry_time_alignment_seconds': 'std'  # NEW: Execution consistency
    })
```

## Optional Parameters

### Sample Large Datasets
```python
# For quick analysis of large datasets
enhanced = enhance_trades(
    trade_file="trades.csv",
    base_data_dir="base_data/",
    sample_size=1000  # Process only 1000 random trades
)
```

## Function Signature

```python
def enhance_trades(
    trade_file: str,        # REQUIRED: Path to trade CSV
    base_data_dir: str,     # REQUIRED: Path to base data directory
    sample_size: int = None # OPTIONAL: Limit number of trades
) -> pd.DataFrame           # Returns: Enhanced DataFrame
```

## What You Get

**Original trade data PLUS 20+ new columns:**

- `entry_open`, `entry_close` - OHLC at trade points
- `macd_change` - MACD evolution during trade
- `trade_duration_minutes` - How long trade lasted
- `entry_time_alignment_seconds` - Timing precision
- `entry_5m_macd`, `exit_5m_macd` - MACD values
- `entry_5m_ema21`, `exit_5m_ema21` - EMA values
- Plus indicator signals and volume data

## Error Handling

```python
try:
    enhanced = enhance_trades(trade_file, base_data_dir)
except FileNotFoundError as e:
    print(f"File not found: {e}")
    # User provides correct paths
except ValueError as e:
    print(f"Data format error: {e}")
    # Check trade file column names
```

## Integration Pattern for Existing Scripts

```python
# BEFORE: Your existing analysis
def your_analysis(trade_file):
    trades = pd.read_csv(trade_file)
    return trades.groupby('ticker')['Profit (Currency)'].mean()

# AFTER: Enhanced analysis (add 3 lines)
from analysis.integration import enhance_trades

def your_enhanced_analysis(trade_file, base_data_dir):
    enhanced = enhance_trades(trade_file, base_data_dir)  # ← ADD
    return enhanced.groupby('ticker').agg({               # ← ENHANCE
        'Profit (Currency)': 'mean',
        'macd_change': 'mean',                            # ← NEW INSIGHTS
        'trade_duration_minutes': 'mean'                  # ← NEW INSIGHTS
    })
```

**Bottom Line:** User provides 2 paths, gets 3x more insights!