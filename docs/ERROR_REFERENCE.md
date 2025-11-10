# Error Reference Guide

**Quick error lookup for common issues in StrategyLab Backtester**

This document catalogues the most common errors users encounter and provides step-by-step solutions. Errors are organized by category for easy lookup.

---

## Table of Contents

1. [Setup & Installation Errors](#setup--installation-errors)
2. [Data Fetching Errors](#data-fetching-errors)
3. [Backtest Execution Errors](#backtest-execution-errors)
4. [Strategy Registration Errors](#strategy-registration-errors)
5. [Configuration Errors](#configuration-errors)
6. [Broker API Errors](#broker-api-errors)
7. [Environment & Dependency Errors](#environment--dependency-errors)

---

## Setup & Installation Errors

### Error: `Python version 3.X is not supported`

**Symptom**:
```
✗ Python 3.8.0 (requires 3.9+)
Please upgrade Python from https://www.python.org/downloads/
```

**Cause**: Python version is below 3.9

**Solution**:
1. Download Python 3.9+ from [python.org](https://www.python.org/downloads/)
2. Install and verify: `python --version`
3. Re-run `python setup.py`

---

### Error: `ModuleNotFoundError: No module named 'X'`

**Symptom**:
```
ModuleNotFoundError: No module named 'pandas'
ModuleNotFoundError: No module named 'numpy'
ModuleNotFoundError: No module named 'yaml'
```

**Cause**: Dependencies not installed or virtual environment not activated

**Solution**:
```powershell
# Windows
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Linux/Mac
source .venv/bin/activate
pip install -r requirements.txt
```

**Prevention**: Always activate venv before running commands

---

### Error: `FileNotFoundError: [Errno 2] No such file or directory: 'requirements.txt'`

**Symptom**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'requirements.txt'
```

**Cause**: Running commands from wrong directory

**Solution**:
```powershell
# Navigate to backtester directory
cd path/to/backtester

# Verify you're in the right place
ls requirements.txt  # Should exist

# Then run setup
python setup.py
```

---

## Data Fetching Errors

### Error: `No data found for ticker X`

**Symptom**:
```
✗ No data found for RELIANCE
Data pools searched:
  (no pools found)

Action needed:
  Fetch data: python src/runners/unified_runner.py --mode fetch --tickers RELIANCE
```

**Cause**: Data has not been fetched yet (MOST COMMON ERROR!)

**Solution**:
```powershell
# Fetch data for the ticker
python src/runners/unified_runner.py --mode fetch --tickers RELIANCE

# Or use interactive mode
python src/runners/unified_runner.py --mode fetch

# Verify data was fetched
python src/runners/unified_runner.py --check-data RELIANCE
```

**Why this happens**: The backtester needs historical market data to work. It does NOT come with pre-loaded data.

---

### Error: `API authentication failed`

**Symptom**:
```
✗ Upstox API authentication failed
Error: Invalid client ID or secret

✗ Binance API authentication failed
Error: Invalid API key
```

**Cause**: Broker API credentials not configured or incorrect

**Solution**:
1. Check `.env` file exists: `ls .env`
2. Open `.env` and verify credentials:
   ```bash
   # For Upstox
   UPSTOX_CLIENT_ID=your_actual_client_id  # NOT "your_client_id_here"
   UPSTOX_CLIENT_SECRET=your_actual_secret

   # For Binance (crypto - FREE but requires registration)
   BINANCE_API_KEY=your_binance_api_key
   BINANCE_API_SECRET=your_binance_secret
   ```
3. Get credentials from:
   - **Binance** (easiest - FREE): https://www.binance.com/en/my/settings/api-management
   - **Upstox**: https://developer.upstox.com/
   - **Zerodha**: https://kite.trade/
4. Save `.env` and retry fetch

**Important**: ALL brokers require API keys, even Binance for crypto. Binance is free (no subscription), but you still need to register and generate API keys.

**Detailed guide**: [docs/BROKER_SETUP.md](BROKER_SETUP.md)

---

### Error: `Rate limit exceeded`

**Symptom**:
```
✗ Error fetching data: Rate limit exceeded (429)
Please wait 60 seconds and try again
```

**Cause**: Broker API rate limits (too many requests)

**Solution**:
```powershell
# Wait 1-2 minutes
timeout /t 60  # Windows
sleep 60       # Linux/Mac

# Then retry
python src/runners/unified_runner.py --mode fetch --tickers RELIANCE
```

**Prevention**: Don't fetch data too frequently (once per day is usually enough)

---

### Error: `No data returned from broker for ticker X`

**Symptom**:
```
⚠ No data returned for RELIANCE
Possible reasons:
  - Ticker symbol incorrect
  - Data not available for date range
  - Market was closed
```

**Cause**: Invalid ticker or date range

**Solution**:
1. **Check ticker symbol**:
   - Must be uppercase: `RELIANCE` not `reliance`
   - Must be exact NSE/BSE symbol
   - For crypto: `BTCUSDT` not `BTC`

2. **Check date range**:
   ```powershell
   # Don't fetch future dates or very old data
   python src/runners/unified_runner.py --mode fetch \
     --tickers RELIANCE \
     --date-ranges 2024-01-01_to_2024-03-31  # Valid range
   ```

3. **Check market hours**:
   - Indian equities: Mon-Fri 9:15 AM - 3:30 PM IST
   - Crypto: 24/7 (any time)

---

## Backtest Execution Errors

### Error: `ValueError: No data found for RELIANCE`

**Symptom**:
```
ValueError: No data found for RELIANCE in date range 2024-01-01 to 2024-01-31
```

**Cause**: Data not fetched OR wrong date range

**Solution**:
```powershell
# 1. Check if data exists
python src/runners/unified_runner.py --check-data RELIANCE

# 2. If no data, fetch it
python src/runners/unified_runner.py --mode fetch \
  --tickers RELIANCE \
  --date-ranges 2024-01-01_to_2024-01-31

# 3. Verify data pools
dir data\pools\  # Should show date range folders

# 4. Retry backtest
python src/runners/unified_runner.py --mode backtest \
  --strategies open_source_baseline \
  --template conservative \
  --date-ranges 2024-01-01_to_2024-01-31 \
  --tickers RELIANCE
```

---

### Error: `FileNotFoundError: data/pools/...`

**Symptom**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/pools/2024-01-01_to_2024-01-31/...'
```

**Cause**: Data pool directory doesn't exist

**Solution**:
```powershell
# Fetch data (creates the pool automatically)
python src/runners/unified_runner.py --mode fetch \
  --tickers RELIANCE \
  --date-ranges 2024-01-01_to_2024-01-31
```

---

### Error: `Strategy 'X' not found`

**Symptom**:
```
ValueError: Strategy 'my_custom_strategy' not registered
Available strategies: open_source_baseline, sma_crossover, bollinger_bands
```

**Cause**: Strategy not registered in `register_strategies.py`

**Solution**: See [Strategy Registration Errors](#strategy-registration-errors) section below

---

## Strategy Registration Errors

### Error: `Strategy 'X' not registered`

**Symptom**:
```
ValueError: Strategy 'my_strategy' not registered
Available strategies: ['open_source_baseline', 'sma_crossover', 'bollinger_bands']
```

**Cause**: Strategy class exists but not registered

**Solution**:
1. **Check available strategies**:
   ```powershell
   python src/runners/unified_runner.py --list-strategies
   ```

2. **If your strategy is missing**, register it:
   - Open `src/strategies/register_strategies.py`
   - Add registration:
     ```python
     from strategies.my_strategy import MyStrategy

     def register_all_strategies():
         StrategyFactory.register_strategy('open_source_baseline', OpenSourceBaselineStrategy)
         StrategyFactory.register_strategy('my_strategy', MyStrategy)  # ADD THIS
     ```

3. **Verify registration**:
   ```powershell
   python src/runners/unified_runner.py --list-strategies
   # Should now show 'my_strategy'
   ```

**Detailed guide**: [docs/STRATEGY_GUIDE.md](STRATEGY_GUIDE.md)

---

### Error: `ImportError: cannot import name 'MyStrategy'`

**Symptom**:
```
ImportError: cannot import name 'MyStrategy' from 'strategies.my_strategy'
```

**Cause**: Strategy file doesn't exist or class name mismatch

**Solution**:
1. **Verify file exists**: `ls src/strategies/my_strategy.py`
2. **Check class name in file**:
   ```python
   # File: src/strategies/my_strategy.py
   class MyStrategy(BaseStrategy):  # Must match import
       ...
   ```
3. **Verify import path** in `register_strategies.py`:
   ```python
   from strategies.my_strategy import MyStrategy  # Must match filename
   ```

---

## Configuration Errors

### Error: `YAML syntax error`

**Symptom**:
```
yaml.scanner.ScannerError: while scanning a simple key
  in "config/templates/my_template.yaml", line 5, column 1
```

**Cause**: Invalid YAML syntax (usually indentation)

**Solution**:
1. **Verify YAML syntax**:
   ```powershell
   python src/runners/unified_runner.py --verify-config --template my_template
   ```

2. **Common YAML issues**:
   - Use spaces (NOT tabs) for indentation
   - Colons must have space after: `key: value` not `key:value`
   - Lists need consistent indentation

3. **Use a YAML validator**: [yamllint.com](http://www.yamllint.com/)

**Valid YAML example**:
```yaml
risk:
  max_position_size: 0.15  # Space after colon
  stop_loss_pct: 0.02      # Consistent indentation
```

---

### Error: `Template 'X' not found`

**Symptom**:
```
✗ Template not found: my_template
Available templates:
  - conservative
  - aggressive
  - minimal
```

**Cause**: Template file doesn't exist

**Solution**:
1. **Check available templates**:
   ```powershell
   dir config\templates\  # Windows
   ls config/templates/   # Linux/Mac
   ```

2. **Create template** (copy existing):
   ```powershell
   cp config/templates/conservative.yaml config/templates/my_template.yaml
   # Edit my_template.yaml as needed
   ```

3. **Verify template**:
   ```powershell
   python src/runners/unified_runner.py --describe-template my_template
   ```

---

## Broker API Errors

### Error: `Invalid redirect URI`

**Symptom**:
```
✗ Upstox API Error: Invalid redirect URI
```

**Cause**: Redirect URI in `.env` doesn't match broker app settings

**Solution**:
1. **Check `.env` file**:
   ```bash
   UPSTOX_REDIRECT_URI=https://127.0.0.1:5000/
   ```

2. **Login to broker developer portal**:
   - Upstox: https://developer.upstox.com/
   - Zerodha: https://developers.kite.trade/

3. **Verify redirect URI matches** in app settings

4. **Common valid URIs**:
   - `https://127.0.0.1:5000/`
   - `http://localhost:5000/`

---

### Error: `Token expired`

**Symptom**:
```
✗ API Error: Access token expired
```

**Cause**: Authentication token expired (broker APIs expire after 24 hours)

**Solution**:
```powershell
# Re-authenticate (will open browser)
python src/core/etl/data_fetcher.py

# System will guide you through re-authentication
```

**Note**: Most broker APIs require daily re-authentication for security

---

## Environment & Dependency Errors

### Error: `'python' is not recognized`

**Symptom**:
```
'python' is not recognized as an internal or external command
```

**Cause**: Python not installed or not in PATH

**Solution**:
1. **Install Python 3.9+** from [python.org](https://www.python.org/downloads/)
2. **During installation**: Check "Add Python to PATH"
3. **Restart terminal** and verify: `python --version`

---

### Error: `Permission denied` when creating venv

**Symptom**:
```
PermissionError: [Errno 13] Permission denied: '.venv/Scripts/python.exe'
```

**Cause**: Windows execution policy or antivirus

**Solution**:
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Retry setup
python setup.py
```

---

### Error: `.env file not found`

**Symptom**:
```
⚠ .env file not found - you may not be able to fetch data
```

**Cause**: `.env` file not created

**Solution**:
```powershell
# Create from template
cp .env.example .env  # Linux/Mac
copy .env.example .env  # Windows

# Edit .env with your broker credentials
notepad .env  # Windows
nano .env     # Linux
```

---

## Quick Diagnostic Commands

**Run these to diagnose common issues**:

```powershell
# Check Python version
python --version

# Check virtual environment
python -c "import sys; print('venv' if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else 'system')"

# List registered strategies
python src/runners/unified_runner.py --list-strategies

# Check data availability
python src/runners/unified_runner.py --check-data RELIANCE

# Verify configuration template
python src/runners/unified_runner.py --verify-config --template conservative

# Run full system verification
python scripts/verify_setup.py
```

---

## Still Having Issues?

1. **Run verification script**: `python scripts/verify_setup.py`
2. **Check QUICKSTART.md**: Step-by-step setup guide
3. **Review logs**: Check `logs/` directory for detailed error messages
4. **Create GitHub Issue**: https://github.com/Neel-Error404/StrategyLab/issues

Include in your issue:
- Error message (full traceback)
- Command you ran
- Output of `python scripts/verify_setup.py`
- Operating system and Python version

---

*Last updated: 2025-01-07*
