# Integration Module - Function Flow & Dependencies

## Call Chain: What Calls What

```
USER CODE
    ↓
enhance_trades(trade_data)
    ↓
[__init__.py] enhance_trades() wrapper
    ↓
get_base_data_dir() → Auto-finds base data location
    ↓
[core/trade_enhancer.py] _enhance_trades() → Core logic
    ↓
For each ticker:
    _load_ticker_base_data() → Loads CSV files
    _enhance_single_trade() → Maps trade to base data
    _align_timestamp() → Finds matching 5-minute interval
    ↓
Returns enhanced DataFrame
```

## Detailed Function Breakdown

### Level 1: User Interface (`__init__.py`)

```python
def enhance_trades(trade_data, base_data_dir=None, sample_size=None):
    # 1. Auto-detect base data if not provided
    if base_data_dir is None:
        base_data_dir = get_base_data_dir()  # ← CALLS get_base_data_dir()

    # 2. Sample data if requested
    if sample_size:
        trade_data = trade_data.sample(n=sample_size)

    # 3. Call core enhancement
    return _enhance_trades(trade_data, base_data_dir)  # ← CALLS core function
```

### Level 2: Base Data Discovery (`__init__.py`)

```python
def get_base_data_dir(search_from=None):
    # 1. Set search starting point
    search_from = Path(__file__).parent.parent.parent  # StrategyLab-master/

    # 2. Try common patterns
    patterns = [
        "outputs/*/mse_backtesting/*/data/base_data",    # ← MSE output structure
        "outputs/*/*/data/base_data",
        "data/base_data",
        "**/base_data"
    ]

    # 3. Find first match
    for pattern in patterns:
        matches = list(search_from.glob(pattern))  # ← FILE SYSTEM SEARCH
        if matches:
            return str(max(matches, key=lambda p: p.stat().st_mtime))  # Latest

    raise FileNotFoundError("No base data directory found")
```

### Level 3: Core Enhancement Logic (`core/trade_enhancer.py`)

```python
def _enhance_trades(trade_data, base_data_dir):
    # 1. Validate input data
    _validate_trade_data(trade_data)  # ← CALLS validator

    # 2. Process each ticker
    tickers = trade_data['ticker'].unique()
    enhanced_records = []

    for ticker in tickers:
        # 3. Load base data for ticker
        base_data = _load_ticker_base_data(ticker, base_data_dir)  # ← CALLS loader

        # 4. Enhance each trade
        ticker_trades = trade_data[trade_data['ticker'] == ticker]
        for _, trade in ticker_trades.iterrows():
            enhanced_trade = _enhance_single_trade(trade, base_data)  # ← CALLS enhancer
            enhanced_records.append(enhanced_trade)

    return pd.DataFrame(enhanced_records)
```

### Level 4: Core Helper Functions (`core/trade_enhancer.py`)

```python
def _load_ticker_base_data(ticker, base_data_dir):
    # 1. Find base data file
    base_files = list(Path(base_data_dir).glob(f"{ticker}_Base_*.csv"))  # ← FILE SEARCH

    # 2. Load and prepare
    base_data = pd.read_csv(base_files[0])  # ← CSV READ
    base_data['timestamp'] = pd.to_datetime(base_data['timestamp'])
    return base_data.sort_values('timestamp')

def _enhance_single_trade(trade, base_data):
    # 1. Get trade times
    entry_time = trade['Entry Time']
    exit_time = trade['Exit Time']

    # 2. Find matching base data records
    entry_idx = _align_timestamp(entry_time, base_data)  # ← CALLS timestamp aligner
    exit_idx = _align_timestamp(exit_time, base_data)

    # 3. Extract base data context
    entry_data = base_data.iloc[entry_idx]
    exit_data = base_data.iloc[exit_idx]

    # 4. Build enhanced record
    enhanced = trade.to_dict()
    enhanced.update({
        'entry_close': entry_data['close'],
        'exit_close': exit_data['close'],
        'macd_change': exit_data['5m_macd'] - entry_data['5m_macd'],
        # ... 20+ more fields
    })
    return enhanced

def _align_timestamp(trade_time, base_data):
    # Find base data record with timestamp <= trade_time
    before_mask = base_data['timestamp'] <= trade_time  # ← TIMESTAMP COMPARISON
    return before_mask[before_mask].index[-1]
```

## Base Requirements - What Must Exist

### 1. File System Structure
```
StrategyLab-master/
└── outputs/
    └── [date_folder]/
        └── mse_backtesting/
            └── [date_range]/
                └── data/
                    └── base_data/           # ← MUST EXIST
                        ├── TICKER1_Base_*.csv
                        ├── TICKER2_Base_*.csv
                        └── ...
```

### 2. Trade Data Format (Input)
```python
# DataFrame must have these columns:
trade_data = pd.DataFrame({
    'ticker': ['RELIANCE', 'TCS', ...],           # ← REQUIRED
    'Entry Time': ['2024-01-01 09:30:00', ...],  # ← REQUIRED (parseable datetime)
    'Exit Time': ['2024-01-01 10:30:00', ...],   # ← REQUIRED (parseable datetime)
    'Profit (Currency)': [100, -50, ...],        # ← REQUIRED (numeric)
    'Trade Type': ['BUY', 'SELL', ...],          # ← REQUIRED
    # Other columns preserved as-is
})
```

### 3. Base Data CSV Format (Auto-loaded)
```python
# Each TICKER_Base_*.csv must have:
base_data_columns = [
    'timestamp',      # ← REQUIRED (datetime, 5-minute intervals)
    'open', 'high', 'low', 'close', 'volume',  # ← REQUIRED (OHLCV)
    '5m_macd', '15m_macd',     # ← OPTIONAL (indicators added if present)
    '5m_ema21', '5m_ema50',    # ← OPTIONAL (indicators added if present)
    '*_signal',                # ← OPTIONAL (signals added if present)
    # Any other columns are automatically detected and added
]
```

## Dependencies Flow

```
USER INPUT (trade_data DataFrame)
    ↓
FILE SYSTEM (base data directory with CSV files)
    ↓
PANDAS (CSV loading, timestamp parsing, data joining)
    ↓
NUMPY (timestamp alignment, calculations)
    ↓
PATHLIB (file system navigation)
    ↓
ENHANCED DATAFRAME (original + 20+ new columns)
```

## Error Points - Where It Can Fail

1. **File System**: No base data directory found
2. **CSV Files**: Missing ticker base data files
3. **Data Format**: Trade data missing required columns
4. **Timestamps**: Invalid datetime formats
5. **Memory**: Large datasets (1M+ trades) without sampling

## Minimal Working Example

```python
# What the user needs:
import pandas as pd
from analysis.integration import enhance_trades

# 1. Trade data with required columns
trades = pd.DataFrame({
    'ticker': ['RELIANCE'],
    'Entry Time': ['2024-01-01 09:30:00'],
    'Exit Time': ['2024-01-01 10:30:00'],
    'Profit (Currency)': [100],
    'Trade Type': ['BUY']
})

# 2. Base data files must exist:
# outputs/.../base_data/RELIANCE_Base_2024-01-01_to_2024-12-31.csv

# 3. One function call
enhanced = enhance_trades(trades)

# Result: trades DataFrame + 20+ new columns
```

The system is designed to **fail gracefully** - if base data isn't found, it provides clear error messages about what's missing and where it looked.