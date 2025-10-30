# Setup Guide

Installation steps for the StrategyLab equities backtester.

---

## Quick Start

```bash
git clone <repository-url>
cd backtester
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/runners/unified_runner.py --mode validate --dates 2024-01-01
```

---

## Environment Setup

### venv (recommended)
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate    # macOS/Linux
pip install -r requirements.txt
```

### conda
```bash
conda create -n strategylab python=3.10
conda activate strategylab
pip install -r requirements.txt
```

---

## Repository Structure

```
backtester/
├── src/
├── config/
├── docs/
├── data/
└── outputs/
```

Pools are generated under `data/pools/<date_range>/<timeframe>/` as parquet files.

---

## Broker Credentials

Create `.env` and populate keys:

```bash
ZERODHA_API_KEY=...
ZERODHA_API_SECRET=...
UPSTOX_CLIENT_ID=...
UPSTOX_CLIENT_SECRET=...
```

`config/config_loader.py` reads these values automatically.

---

## Fetching Data

```bash
python src/core/etl/data_fetcher.py --mode fetch --tickers RELIANCE TCS --timeframe 1minute --days 5
```

Inspect pools before backtests:

```bash
python src/core/etl/pool_inspector.py --pool-path data/pools/2024-01-01_to_2024-06-30
```

---

## Running Backtests

```bash
python src/runners/unified_runner.py --mode backtest \
  --template conservative \
  --date-ranges 2024-01-01_to_2024-01-31 \
  --tickers RELIANCE TCS
```

Available templates: `minimal`, `conservative`, `aggressive`, `portfolio_diversified`.

---

## Incremental Updates

```bash
python src/runners/unified_runner.py --mode update \
  --pool-path data/pools/2024-01-01_to_2024-06-30 \
  --extend-to 2024-09-30 \
  --dry-run
```

Workflow:
1. Inspect pool.
2. Calculate gaps (`src/core/etl/gap_calculator.py`).
3. Run update (dry run, then `--yes`).

---

## Verification

```bash
python src/runners/unified_runner.py --mode validate --dates 2024-01-01
.venv\Scripts\python.exe -m pytest tests/test_backtest_live_parity.py tests/test_precision_validation.py -q
```

---

## Troubleshooting

- Missing tickers: ensure data pools exist and filenames are correct.
- Authentication failures: re-check `.env` or refresh tokens.
- Large downloads: limit `--days` or split date ranges.
- Encoding errors: ensure files remain UTF-8.

See `docs/TROUBLESHOOTING.md` for more scenarios.
