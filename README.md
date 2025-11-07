# StrategyLab Backtester (Equities V2)

A production-ready, modular backtesting system for equities strategies with broker integration, incremental data management, and parity/precision validation tooling.

### Included Open-Source Strategies
- `open_source_baseline`: trend + momentum hybrid tuned for reproducible demos (default)
- `sma_crossover`: introductory moving-average crossover example
- `bollinger_bands`: volatility-channel strategy template

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Brokers](https://img.shields.io/badge/Brokers-Zerodha%20%7C%20Upstox%20%7C%20Binance-orange.svg)](docs/BROKER_SETUP.md)

---

## 🚀 New Users - Start Here!

**First time setting up? Follow these steps:**

1. **📖 Read the Quick Start**: [QUICKSTART.md](QUICKSTART.md) - Get from zero to your first backtest in 15 minutes
2. **⚙️ Run automated setup**: `python setup.py` - One command to install everything
3. **✅ Verify installation**: `python scripts/verify_setup.py` - Check that everything works
4. **🎯 Run first backtest**: `python scripts/quickstart.py` - Interactive guided backtest

**Having issues?** See [docs/ERROR_REFERENCE.md](docs/ERROR_REFERENCE.md) for common problems and solutions.

**For LLMs/AI Assistants**: The [QUICKSTART.md](QUICKSTART.md) provides a complete, linear setup path. Start there for guiding users through installation and first use.

---

## 🤖 AI-Assisted Setup (Copy-Paste Prompt for Claude/ChatGPT/Cursor)

**New users**: Copy the prompt below and paste it into your AI coding assistant (Claude Code, Cursor, ChatGPT, etc.). The AI will guide you through the complete setup process.

```markdown
I've just cloned the StrategyLab Backtester repository and need help setting it up from scratch to run my first backtest.

CONTEXT:
- Repository: https://github.com/Neel-Error404/StrategyLab
- I'm in directory: backtester/
- Operating System: [Windows/Linux/Mac]
- Python installed: [Yes/No/Don't know]
- Broker API: [Have Upstox/Have Zerodha/Will register Binance (free)/Need help choosing]

NOTE: ALL brokers require API keys (even free ones like Binance for crypto).
Binance is easiest - free, 5-minute registration, no subscription fees.

COMPLETE WORKFLOW NEEDED:
1. **Environment Setup**: Python 3.9+ check, virtual environment, dependencies
2. **Configuration**: Broker API credentials setup (.env file)
3. **Verification**: Validate installation, check strategies registered
4. **Data Fetching**: Download market data (critical step - explain how!)
5. **First Backtest**: Run a simple backtest to verify everything works
6. **Results**: Understand what the output means

CRITICAL REQUIREMENTS:
- Show me EXACT commands to run (PowerShell for Windows, bash for Linux/Mac)
- Explain WHEN and WHY to run each command
- Handle the DATA FETCHING step explicitly (many users miss this!)
- Tell me what to do if something fails
- Verify each step before moving to the next

AVAILABLE DOCUMENTATION (read these as needed):
- QUICKSTART.md - Complete 5-step setup guide
- docs/BROKER_SETUP.md - API credential setup
- docs/ERROR_REFERENCE.md - Common errors and solutions
- docs/SETUP_GUIDE.md - Detailed installation guide
- docs/STRATEGY_GUIDE.md - Creating custom strategies
- docs/CLI_REFERENCE.md - All CLI commands

KEY FEATURES:
- Automated setup: `python setup.py`
- Installation verification: `python scripts/verify_setup.py`
- Interactive first backtest: `python scripts/quickstart.py`
- Helper commands: `--list-strategies`, `--check-data`, `--verify-config`

START BY:
1. Reading QUICKSTART.md to understand the 5-step process
2. Checking if I have Python 3.9+ installed
3. Walking me through setup.py execution
4. Helping me configure broker API (critical!)
5. Guiding me through data fetching (MOST COMMON ISSUE!)
6. Running first backtest and explaining results

Please start by checking my Python version and then guide me through each step with clear explanations of what's happening and why.
```

**After setup, you can ask your AI:**
- "How do I fetch data for RELIANCE stock?"
- "Explain the difference between conservative and aggressive templates"
- "How do I create a custom strategy?"
- "What do the backtest results mean?"
- "How do I run multiple tickers in parallel?"

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
| `optimize` | Strategy parameter search (WIP) | `python src/runners/unified_runner.py --mode optimize --strategies open_source_baseline` |

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
