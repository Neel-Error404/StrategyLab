# Options Validation - Phase 1

**Status**: Ready for Testing
**Goal**: Fetch 6 months of RELIANCE options data and validate synthetic pricing

---

## Configuration

The data fetcher uses `validation_config.yaml` for all default settings. You can:

1. **Edit the config file** to change defaults (tickers, strike range, OI filters, etc.)
2. **Use CLI arguments** to override config values
3. **Specify a custom config** with `--config /path/to/config.yaml`

**Key config settings**:
- `validation.tickers`: List of tickers to fetch (default: NIFTY, BANKNIFTY, RELIANCE, TCS, INFY)
- `validation.timeframe`: Data timeframe (default: 1day)
- `validation.strike_range.percentage_range`: Strike filtering (default: ±20%)
- `validation.filters.min_open_interest`: Minimum OI filter (default: 100)
- `api.rate_limit.requests_per_second`: API rate limit (default: 5 req/sec)

See `validation_config.yaml` for all available settings.

---

## Quick Start

### Step 1: Test Data Fetch (RELIANCE, 1 Expiry)

Fetch just 1 expiry to test the pipeline:

```bash
cd /mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/StrategyLab-master

python src/core/options/validation/data_fetcher.py \
  --ticker RELIANCE \
  --timeframe 1day \
  --max-expiries 1 \
  --log-level INFO
```

**Expected Output**:
```
Fetching options data for RELIANCE
Found X expiries
[1/1] Processing expiry: 2025-XX-XX
Reference price: 2850.0 (date: 2025-04-XX)
Fetched Y rows (Z strikes) from 2025-04-01 to 2025-XX-XX
Saved data to data/pools/options/2025-04-01_to_2025-10-08/RELIANCE/1day/expiry_2025-XX-XX.parquet
```

**Check Results**:
```bash
# List fetched files
ls -lh data/pools/options/2025-04-01_to_2025-10-08/RELIANCE/1day/

# Inspect parquet file
python -c "
import pandas as pd
df = pd.read_parquet('data/pools/options/2025-04-01_to_2025-10-08/RELIANCE/1day/expiry_2025-XX-XX.parquet')
print(f'Rows: {len(df)}')
print(f'Strikes: {df[\"strike\"].nunique()}')
print(f'Date range: {df[\"timestamp\"].min()} to {df[\"timestamp\"].max()}')
print(df.head(10))
"
```

---

### Step 2: Fetch RELIANCE Full Dataset (6 Months)

Once Step 1 works, fetch all expiries:

```bash
python src/core/options/validation/data_fetcher.py \
  --ticker RELIANCE \
  --timeframe 1day \
  --log-level INFO
```

**This will**:
- Fetch all expiries from April 2025 to October 2025 (~25 expiries)
- Filter strikes to ±20% around price
- Filter for OI > 100
- Save ~15-20 parquet files (~200 KB each)
- Total: ~3-5 MB

**Time**: ~5-10 minutes (rate-limited to 5 req/sec)

---

### Step 3: Fetch All Validation Tickers

```bash
python src/core/options/validation/data_fetcher.py \
  --all \
  --timeframe 1day \
  --log-level INFO
```

**This will fetch**:
- NIFTY, BANKNIFTY, RELIANCE, TCS, INFY
- ~25 expiries each × 5 tickers = ~125 files
- Total: ~15-25 MB

**Time**: ~30-45 minutes

---

## Data Structure

After fetching, your directory will look like:

```
data/pools/options/2025-04-01_to_2025-10-08/
└── RELIANCE/
    ├── 1day/
    │   ├── expiry_2025-04-24.parquet   (~200 KB, ~1.5K rows)
    │   ├── expiry_2025-05-29.parquet
    │   ├── expiry_2025-06-26.parquet
    │   └── ... (~25 files)
    └── metadata/
        ├── expiry_2025-04-24.json
        └── ...
```

**Each parquet file contains**:
- Columns: timestamp, strike, option_type (CE/PE), open, high, low, close, volume, open_interest, ticker, expiry, lot_size
- Rows: ~15 strikes × 2 types (CE/PE) × ~40-50 trading days = ~1,500 rows

---

## Querying Data

### Load Single Expiry

```python
from src.core.options.validation.data_storage import OptionsDataStorage
from datetime import date

storage = OptionsDataStorage()

df = storage.load_expiry_data(
    ticker='RELIANCE',
    expiry=date(2025, 5, 29),
    timeframe='1day',
    date_range='2025-04-01_to_2025-10-08'
)

print(df.head())
```

### Load All Expiries

```python
df_all = storage.load_all_expiries(
    ticker='RELIANCE',
    timeframe='1day',
    date_range='2025-04-01_to_2025-10-08'
)

print(f"Total rows: {len(df_all)}")
print(f"Expiries: {df_all['expiry'].nunique()}")
print(f"Strikes: {df_all['strike'].nunique()}")
```

### Filter ATM Options

```python
# Get ATM call options for a specific date
atm_calls = df_all[
    (df_all['timestamp'].dt.date == date(2025, 4, 15)) &
    (df_all['strike'] == 2850) &
    (df_all['option_type'] == 'CE')
]

print(atm_calls[['timestamp', 'strike', 'close', 'open_interest']])
```

---

## Troubleshooting

### Error: "No Upstox access token found"

**Solution**: Authenticate with Upstox first
```bash
# Use existing equity data fetcher to authenticate
python src/core/etl/data_fetcher.py --mode fetch --provider upstox

# Then retry options fetch
```

### Error: "Equity data not found for RELIANCE"

**Solution**: Make sure equity data exists at `data/pools/2022-01-01_to_2025-08-31/RELIANCE/1day.parquet`

If not, fetch it first:
```bash
python src/core/etl/data_fetcher.py \
  --mode fetch \
  --provider upstox \
  --tickers RELIANCE \
  --timeframe 1day \
  --days 365
```

### Error: "Rate limit exceeded"

The fetcher is already rate-limited to 5 req/sec. If you still hit limits:
- Reduce `requests_per_second` in `upstox_options_api.py` (line 38)
- Or wait a few minutes and retry

### Empty Data for Some Expiries

This is normal! Some reasons:
- Expired long ago (may not have data available)
- No strikes met the OI > 100 filter
- Options weren't liquid for that expiry

---

## Next Steps (After Data is Fetched)

1. **Implement Synthetic Pricing Models** (`pricing/synthetic_engine.py`)
   - Black-Scholes + 20-day vol
   - Black-Scholes + 5-day vol
   - Parkinson volatility
   - Calibrated IV

2. **Build Pricing Validator** (`pricing_validator.py`)
   - Compare synthetic vs actual prices
   - Generate error metrics by moneyness, DTE, vol regime

3. **Run Validation Experiment**
   - Measure which model is most accurate
   - Decide if hybrid mode is viable

---

## File Reference

**Core Files**:
- `upstox_options_api.py` - Upstox API wrapper (fetch expiries, contracts, OHLC)
- `data_storage.py` - Save/load parquet & metadata
- `data_fetcher.py` - Main orchestrator (CLI entry point)

**Supporting Files**:
- `../data/schemas.py` - Data classes, validation, path helpers
- `../data/lot_sizes.csv` - Ticker → lot size mapping

**Configuration**:
- Tickers: VALIDATION_TICKERS in `schemas.py`
- Date range: `2025-04-01_to_2025-10-08`
- Strike filter: ±20% around price
- OI filter: > 100

---

## Quick Commands Reference

```bash
# Test with 1 expiry
python src/core/options/validation/data_fetcher.py --ticker RELIANCE --max-expiries 1

# Fetch RELIANCE full
python src/core/options/validation/data_fetcher.py --ticker RELIANCE

# Fetch all tickers
python src/core/options/validation/data_fetcher.py --all

# Check what was fetched
ls -lh data/pools/options/2025-04-01_to_2025-10-08/RELIANCE/1day/

# View data
python -c "import pandas as pd; df = pd.read_parquet('data/pools/options/2025-04-01_to_2025-10-08/RELIANCE/1day/expiry_2025-05-29.parquet'); print(df.info()); print(df.head())"
```

---

**Status**: Ready for testing! Start with `--max-expiries 1` to validate the pipeline works.
