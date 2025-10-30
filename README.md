# StrategyLab Backtester (Equities V2)

A production-ready, modular backtesting system for equities strategies with broker integration, incremental data management, and parity/precision validation tooling.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Brokers](https://img.shields.io/badge/Brokers-Zerodha%20%7C%20Upstox%20%7C%20Binance-orange.svg)](docs/BROKER_SETUP.md)

---

## Quick Start (AI Assistant Prompt)

Use the prompt below with your preferred LLM for guided setup:

```
I'm setting up an algorithmic trading backtester. Please help me configure it based on my requirements.

SYSTEM INFO:
- Repository: https://github.com/yourusername/StrategyLab (Fork before use)
- Language: Python 3.10+
- Supported Brokers: Zerodha Kite API, Upstox API, Binance API
- Architecture: Modular, production-ready with real-time data

MY REQUIREMENTS:
[Describe trading style, risk tolerance, preferred broker, strategies of interest]

AVAILABLE DOCUMENTATION:
- Setup Guide: docs/SETUP_GUIDE.md (installation, dependencies, environment)
- Broker Setup: docs/BROKER_SETUP.md (API keys, authentication, data fetching)
- Strategy Guide: docs/STRATEGY_GUIDE.md (custom strategy development)
- Template Guide: docs/TEMPLATE_GUIDE.md (risk templates, YAML configuration)
- CLI Reference: docs/CLI_REFERENCE.md (all command-line options)
- Output Guide: docs/OUTPUT_GUIDE.md (understanding results, visualisations)

CONFIGURATION TEMPLATES:
- minimal.yaml: Ultra-safe learning (5% max position)
- conservative.yaml: Low-risk trading (15% max position)
- aggressive.yaml: High-risk trading (20% max position)
- portfolio_diversified.yaml: Multi-ticker portfolio

Please provide step-by-step setup instructions, recommend appropriate templates, and suggest CLI commands based on my requirements.
```

---

## Recent Updates (V2 Equities Release)

- Incremental parquet updates via `--mode update`, avoiding full re-fetches.
- Parity and precision validation modules align live vs. backtest signals.
- UTF-8 sanitised codebase with proprietary options stack removed for OSS release.
- Environment-aware YAML config loader with `.env` support.

---

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate        # macOS/Linux
pip install -r requirements.txt
```

Optional: copy `.env.example` to `.env` (create the file if it does not exist) and populate broker credentials.

---

## Command Overview

| Mode | Purpose | Example |
| --- | --- | --- |
| `validate` | Run data and bias validation | `python src/runners/unified_runner.py --mode validate --dates 2024-01-03` |
| `backtest` | Full workflow (backtest + analysis + viz) | `python src/runners/unified_runner.py --mode backtest --date-ranges 2024-01-01_to_2024-01-15 --tickers RELIANCE TCS` |
| `analyze` | Backtest + analysis only | `python src/runners/unified_runner.py --mode analyze --date-ranges 2024-01-01_to_2024-01-15` |
| `visualize` | Backtest + visualisation only | `python src/runners/unified_runner.py --mode visualize --date-ranges 2024-01-01_to_2024-01-15` |
| `fetch` | Pull fresh market data | `python src/runners/unified_runner.py --mode fetch --date-ranges 2024-01-01_to_2024-01-05 --tickers RELIANCE` |
| `update` | Incrementally extend an existing pool | `python src/runners/unified_runner.py --mode update --pool-path data/pools/2024-01-01_to_2024-06-30 --dry-run` |
| `replay` | Run stored manifest through replay engine | `python src/runners/unified_runner.py --mode replay --manifest manifest.json` |
| `optimize` | Strategy parameter search (WIP) | `python src/runners/unified_runner.py --mode optimize --strategy mse` |

The `update` mode shares logic with `src/core/etl/data_fetcher.py`:

```bash
python src/core/etl/data_fetcher.py --mode update --pool-path data/pools/2024-01-01_to_2024-06-30 --extend-to 2024-08-31 --yes
```

Key flags: `--dry-run`, `--validate-only`, `--no-backup` (use with caution).

---

## 🪙 Cryptocurrency Examples

**StrategyLab supports 35+ cryptocurrencies through Binance (no API key required for backtesting):**

```bash
# Fetch Bitcoin data (last 90 days)
python src/runners/unified_runner.py --mode fetch --tickers BTC ETH --timeframes 1h --days 90

# Backtest crypto portfolio (24/7 trading)
python src/runners/unified_runner.py --mode backtest --template aggressive --date-ranges 2024-01-01_to_2024-12-31 --tickers BTCUSDT ETHUSDT

# Multi-crypto analysis
python src/runners/unified_runner.py --mode analyze --date-ranges 2024-Q1 --tickers BTC ETH BNB SOL
```

**Supported Cryptocurrencies**: BTC, ETH, XRP, BNB, SOL, DOGE, ADA, AVAX, SHIB, TRX, UNI, LINK, AAVE, and 20+ more
📖 **Full Documentation**: [docs/BROKER_SETUP.md - Binance Section](docs/BROKER_SETUP.md#-binance-cryptocurrency)

---

## Validation Tooling

| Module | Description |
| --- | --- |
| `src/core/validation/config_parity_validator.py` | Ensures critical config parity between live and backtest |
| `src/core/validation/signal_parity_validator.py` | Compares signal streams and generates parity reports |
| `src/core/validation/precision_validator.py` | Enforces price/quantity precision and PnL rounding |

Run targeted suites:

```bash
.venv\Scripts\python.exe -m pytest tests/test_backtest_live_parity.py tests/test_precision_validation.py -q
```

---

## Data Pools & Incremental Updates

1. Inspect pool: `python src/core/etl/pool_inspector.py --pool-path data/pools/2024-01-01_to_2024-06-30`
2. Calculate gaps: `python src/core/etl/gap_calculator.py --pool-path ...`
3. Fetch/update with `data_fetcher.py` or runner `--mode update`.

Small sample metadata lives in `data/indian_equities_master.csv`. Large historical data is intentionally excluded from the repository.

---

## Configuration Loader

`config/config_loader.py` loads YAML files with environment substitution:

```python
from config.config_loader import ConfigLoader
config = ConfigLoader.load_yaml('config/templates/conservative.yaml')
```

Supported syntax: `${UPSTOX_CLIENT_ID}` or `${UPSTOX_CLIENT_ID:demo}` (default fallback).

---

## Release Checklist (V2)

- [x] Options infrastructure and large datasets removed
- [x] Incremental parquet update workflow documented
- [x] Parity/precision pytest suites green (`59 passed`)
- [x] README/notes updated for equities-only release

See `docs/strategylab_v2_phase0_audit.md` for the full decision history.

---

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
