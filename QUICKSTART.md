# 🚀 Quick Start Guide - StrategyLab Backtester

**Get from zero to your first successful backtest in under 15 minutes.**

## Prerequisites Check

Before starting, verify you have:

- **Python 3.9 or higher** - Check with:
  ```powershell
  python --version
  ```
  If less than 3.9, download from [python.org](https://www.python.org/downloads/)

- **Git** (for cloning the repository)
  ```powershell
  git --version
  ```

- **Broker API Access** - You'll need ONE of:
  - **Upstox API** (recommended for Indian equities - free tier available)
  - **Zerodha Kite API** (Indian equities - paid subscription)
  - **Binance API** (crypto only - FREE, just requires registration)

  **Getting Started**: See [docs/BROKER_SETUP.md](docs/BROKER_SETUP.md) for step-by-step setup guides.

  **Easiest Option**: Binance for crypto (free, 5-minute setup, no subscription fees)

## Step 1: Clone and Setup (2 minutes)

```powershell
# Clone the repository
git clone <your-repo-url>
cd backtester

# Run automated setup (creates venv, installs dependencies, configures environment)
python setup.py
```

The setup script will:
- ✅ Create a virtual environment (`.venv`)
- ✅ Install all required Python packages
- ✅ Create your `.env` configuration file from template
- ✅ Validate your Python environment
- ✅ Display next steps

**Troubleshooting**: If `setup.py` fails, see [docs/ERROR_REFERENCE.md](docs/ERROR_REFERENCE.md) for common issues.

## Step 2: Configure Broker API (5 minutes)

Edit the `.env` file created by setup.py with your broker credentials:

```bash
# For Upstox (recommended for beginners)
UPSTOX_CLIENT_ID=your_client_id_here
UPSTOX_CLIENT_SECRET=your_client_secret_here
UPSTOX_REDIRECT_URI=https://127.0.0.1:5000/

# OR for Zerodha
ZERODHA_API_KEY=your_api_key_here
ZERODHA_API_SECRET=your_api_secret_here
```

**Need credentials?** Follow the detailed setup guides:
- [Upstox API Setup](docs/BROKER_SETUP.md#upstox-setup)
- [Zerodha Kite API Setup](docs/BROKER_SETUP.md#zerodha-setup)

## Step 3: Verify Installation (2 minutes)

Activate your virtual environment and run the verification script:

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
# OR
source .venv/bin/activate      # Linux/Mac

# Run verification
python scripts/verify_setup.py
```

Expected output:
```
✅ Python 3.9+ detected (3.11.0)
✅ Virtual environment active
✅ All dependencies installed (45/45)
✅ Strategies registered (4 found):
   - open_source_baseline
   - sma_crossover
   - bollinger_bands
   - mse_strategy_backtesting
✅ Configuration templates valid (5/5)
✅ Broker API connection successful
✅ Data fetching operational

🎉 Setup verification complete! You're ready to backtest.
```

**If you see ❌ errors**: The script will provide specific troubleshooting links. Common issues are documented in [docs/ERROR_REFERENCE.md](docs/ERROR_REFERENCE.md).

## Step 4: Fetch Market Data (3 minutes)

**IMPORTANT**: The backtester needs historical market data to work. You must fetch data before running backtests.

### Check if you have data:

```powershell
# Check data availability for a ticker
python src/runners/unified_runner.py --check-data RELIANCE
```

### If no data found, fetch it:

```powershell
# Option A: Interactive data fetching (AI-assisted, recommended)
python src/runners/unified_runner.py --mode fetch

# The system will ask you:
# - Which broker to use (Upstox/Zerodha)
# - Which tickers to fetch (e.g., RELIANCE, TCS)
# - Time period (e.g., last 90 days)
# - Timeframes (1minute, 5minute, 15minute, etc.)
```

**OR**

```powershell
# Option B: Direct command (faster if you know what you want)
python src/runners/unified_runner.py --mode fetch \
  --tickers RELIANCE TCS INFY \
  --date-ranges 2024-01-01_to_2024-03-31

# This will fetch:
# - Tickers: RELIANCE, TCS, INFY
# - Date range: Jan 1 to Mar 31, 2024
# - Default timeframes: 1min, 5min, 15min
# - Default broker: From your .env file (Upstox or Zerodha)
```

**What happens during fetch:**
- ✅ Connects to your broker API
- ✅ Downloads OHLCV data (Open, High, Low, Close, Volume)
- ✅ Saves data to `data/pools/[date_range]/`
- ✅ Validates data quality
- ⏱️ Takes 1-3 minutes depending on data range

**Common fetch scenarios:**

```powershell
# Fetch last 30 days for single ticker
python src/runners/unified_runner.py --mode fetch --tickers RELIANCE

# Fetch specific date range for multiple tickers
python src/runners/unified_runner.py --mode fetch \
  --tickers RELIANCE TCS HDFCBANK \
  --date-ranges 2024-01-01_to_2024-06-30

# Fetch crypto data (Binance - FREE API, registration required)
# Note: Binance API is free but you need to register and get API keys
# See docs/BROKER_SETUP.md for Binance setup (takes 5 minutes)
python src/runners/unified_runner.py --mode fetch \
  --tickers BTCUSDT ETHUSDT \
  --date-ranges 2024-01-01_to_2024-03-31
```

**Verify data was fetched:**

```powershell
# Check what data pools exist
dir data\pools\  # Windows
ls data/pools/   # Linux/Mac

# Verify specific ticker
python src/runners/unified_runner.py --check-data RELIANCE
```

**Troubleshooting data fetch:**
- **"API authentication failed"** → Check .env file has correct credentials
- **"No data returned"** → Check ticker symbol is correct (RELIANCE not reliance)
- **"Rate limit exceeded"** → Wait 1 minute and try again (broker API limits)
- See [docs/ERROR_REFERENCE.md](docs/ERROR_REFERENCE.md) for more help

---

## Step 5: Run Your First Backtest (5 minutes)

### Option A: Interactive Mode (Recommended for First Time)

```powershell
python scripts/quickstart.py
```

This interactive script will guide you through:
- Selecting a strategy (start with `open_source_baseline`)
- Choosing a ticker (try `RELIANCE` or `TCS`)
- Setting date range (last 30 days is good for testing)
- Picking a risk template (use `conservative` to start)

### Option B: Direct Command

```powershell
# Basic backtest: Open Source Baseline strategy on RELIANCE for January 2024
python src/runners/unified_runner.py \
  --mode backtest \
  --strategies open_source_baseline \
  --template conservative \
  --date-ranges 2024-01-01_to_2024-01-31 \
  --tickers RELIANCE
```

**Command breakdown**:
- `--mode backtest`: Run backtesting mode
- `--strategies open_source_baseline`: Use the baseline strategy (best for learning)
- `--template conservative`: Use conservative risk management (5% max position)
- `--date-ranges 2024-01-01_to_2024-01-31`: Test period (1 month)
- `--tickers RELIANCE`: Stock to backtest (Reliance Industries)

### Understanding Results

After the backtest completes, you'll find results in:
```
outputs/
└── YYYYMMDD_HHMMSS/           # Timestamped run folder
    ├── metrics/
    │   └── performance_metrics.csv    # Key performance indicators
    ├── trades/
    │   └── trades.csv                 # All trade details
    └── visualizations/
        ├── equity_curve.png           # Portfolio value over time
        └── drawdown.png               # Risk visualization
```

**Key metrics to check**:
- **Total Return**: Overall profit/loss percentage
- **Sharpe Ratio**: Risk-adjusted return (higher is better)
- **Max Drawdown**: Largest peak-to-trough decline (lower is better)
- **Win Rate**: Percentage of profitable trades

For detailed interpretation, see [docs/OUTPUT_GUIDE.md](docs/OUTPUT_GUIDE.md).

## 🎯 What's Next?

### Fetch More Data
```powershell
# Add more tickers to your data pool
python src/runners/unified_runner.py --mode fetch \
  --tickers WIPRO TATAMOTORS BAJFINANCE \
  --date-ranges 2024-01-01_to_2024-06-30

# Update existing data pool with latest data
python src/runners/unified_runner.py --mode update \
  --pool-path data/pools/2024-01-01_to_2024-06-30 \
  --extend-to 2024-12-31
```

### Learn the System
- **Understand architecture**: [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)
- **Explore strategies**: [docs/STRATEGY_GUIDE.md](docs/STRATEGY_GUIDE.md)
- **Learn risk templates**: [docs/TEMPLATE_GUIDE.md](docs/TEMPLATE_GUIDE.md)

### Test Different Scenarios
```powershell
# Test multiple tickers
python src/runners/unified_runner.py --mode backtest --strategies open_source_baseline --template conservative --date-ranges 2024-01-01_to_2024-03-31 --tickers RELIANCE TCS INFY

# Try aggressive risk template
python src/runners/unified_runner.py --mode backtest --strategies open_source_baseline --template aggressive --date-ranges 2024-01-01_to_2024-01-31 --tickers RELIANCE

# Test a different strategy
python src/runners/unified_runner.py --mode backtest --strategies sma_crossover --template conservative --date-ranges 2024-01-01_to_2024-01-31 --tickers RELIANCE
```

### Fetch More Data
```powershell
# Interactive data fetching (AI-assisted)
python src/runners/unified_runner.py --mode fetch

# Or use the data fetcher directly
python src/core/etl/data_fetcher.py
```

### Develop Your Own Strategy
1. Read [docs/STRATEGY_GUIDE.md](docs/STRATEGY_GUIDE.md)
2. Create your strategy class in `src/strategies/`
3. Register it in `src/strategies/register_strategies.py`
4. Test it: `python src/runners/unified_runner.py --mode backtest --strategies your_strategy ...`

### Advanced Features
- **Parallel execution**: Add `--parallel` flag for multi-core processing
- **Analysis mode**: Generate detailed analytics with `--mode analyze`
- **Visualization**: Create charts with `--mode visualize`
- **Validation**: Verify system integrity with `--mode validate`

## 🆘 Getting Help

### Common Issues
- **Import errors**: Dependencies not installed → Run `pip install -r requirements.txt`
- **No data found**: Data not fetched → Run `python src/runners/unified_runner.py --mode fetch`
- **Strategy not found**: Registration issue → Check `src/strategies/register_strategies.py`
- **API errors**: Broker credentials → Verify `.env` file configuration

### Documentation
- **Detailed setup**: [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md)
- **Error reference**: [docs/ERROR_REFERENCE.md](docs/ERROR_REFERENCE.md)
- **Troubleshooting**: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
- **CLI reference**: [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md)

### Verify Your Setup Anytime
```powershell
# List available strategies
python src/runners/unified_runner.py --list-strategies

# Verify configuration template
python src/runners/unified_runner.py --verify-config --template conservative

# Check data availability
python src/runners/unified_runner.py --check-data RELIANCE
```

## 📚 Learning Path

**Week 1**: Run example backtests, understand results
**Week 2**: Modify risk templates, test different parameters
**Week 3**: Create your first custom strategy
**Week 4**: Optimize and validate your strategy

---

**🎉 Congratulations!** You've successfully set up the StrategyLab backtester. Start experimenting with different strategies, tickers, and risk parameters to learn what works best.

**Questions?** Check the comprehensive documentation in the `docs/` folder or create an issue on GitHub.

---

*Last updated: 2025-11-07*
