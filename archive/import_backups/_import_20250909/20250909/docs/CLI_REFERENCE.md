# ⚙️ CLI Reference Guide

**Complete command-line interface documentation for the algorithmic trading backtester.**

## 🎯 **New Simplified Requirements**

**MINIMUM REQUIREMENTS:**
- **All modes**: Only `--date-ranges` required (tickers auto-discovered)
- **Fetch mode**: Can run with zero arguments (interactive mode)

**AUTO-DISCOVERY:**
- Tickers automatically found from data pools when not specified
- Explicit tickers override auto-discovery
- Clear error messages when no data found

---

## 🚀 **Quick Commands**

```bash
# ⚡ MINIMAL USAGE - Auto-discover all tickers
python src/runners/unified_runner.py --mode backtest --date-ranges 2024-01-01_to_2024-01-31
python src/runners/unified_runner.py --mode validate --date-ranges 2024-01-01_to_2024-01-31
python src/runners/unified_runner.py --mode analyze --date-ranges 2024-01-01_to_2024-12-31
python src/runners/unified_runner.py --mode visualize --date-ranges 2024-01-01_to_2024-12-31

# 🎯 INTERACTIVE FETCH - Zero arguments
python src/runners/unified_runner.py --mode fetch

# 🎯 SPECIFIC TICKERS - Override auto-discovery  
python src/runners/unified_runner.py --mode backtest --date-ranges 2024-01-01_to_2024-01-31 --tickers RELIANCE TCS

# 📊 EXPLICIT FETCH - With parameters
python src/runners/unified_runner.py --mode fetch --date-ranges 2024-01-01_to_2024-01-31 --tickers RELIANCE TCS
```

---

## 📖 **Command Structure**

```bash
python src/runners/unified_runner.py [MODE] [CONFIG] [DATA] [OPTIONS]
```

### **🎯 Core Arguments**

#### **Mode Selection** (Required)
```bash
--mode {validate,backtest,analyze,visualize,fetch}
```

- `validate`: Check data quality and system health
- `backtest`: Run strategy backtesting (full workflow)
- `analyze`: Generate analysis from existing results
- `visualize`: Generate visualizations only
- `fetch`: Download market data from broker APIs

#### **Configuration Options** (Choose one)
```bash
# Pre-built templates
--template {minimal,conservative,aggressive,options,portfolio_diversified}

# Custom configuration file
--config path/to/config.yaml
```

#### **Data Selection** 
```bash
# Date ranges (REQUIRED for all modes except zero-argument fetch)
--date-ranges 2024-01-01_to_2024-12-31 2024-06-01_to_2024-06-30

# Individual dates (alternative to date ranges)
--dates 2024-01-01 2024-01-02 2024-01-03

# Tickers (OPTIONAL - auto-discovered if not provided)
--tickers RELIANCE TCS INFY ZOMATO

# Strategy selection (defaults to 'sma_crossover')
--strategies sma_crossover bollinger_bands custom_strategy
```

**📌 IMPORTANT:** 
- **Tickers are now OPTIONAL** - system auto-discovers from data pools
- When no tickers specified: all available tickers in date range are used
- When tickers specified: only those tickers are processed
- Auto-discovery searches `data/pools/{date_range}/1minute/*.csv` files

---

## 🎯 **Template Usage**

### **Minimal Template** (Ultra-Safe Learning)
```bash
# With auto-discovery
python src/runners/unified_runner.py \
  --mode backtest \
  --template minimal \
  --date-ranges 2024-01-01_to_2024-01-31

# With specific ticker
python src/runners/unified_runner.py \
  --mode backtest \
  --template minimal \
  --date-ranges 2024-01-01_to_2024-01-31 \
  --tickers RELIANCE
```
- **Risk**: 5% max position
- **Use Case**: Learning, testing new strategies
- **Features**: Single-threaded, detailed logging

### **Conservative Template** (Stable Income)
```bash
# Auto-discover all tickers
python src/runners/unified_runner.py \
  --mode backtest \
  --template conservative \
  --date-ranges 2024-01-01_to_2024-03-31

# Specific tickers
python src/runners/unified_runner.py \
  --mode backtest \
  --template conservative \
  --date-ranges 2024-01-01_to_2024-12-31 \
  --tickers RELIANCE TCS INFY
```
- **Risk**: 15% max position  
- **Use Case**: Steady returns, risk-averse traders
- **Features**: Multi-threading, moderate risk controls

### **Aggressive Template** (Growth Focused)
```bash
python src/runners/unified_runner.py \
  --mode backtest \
  --template aggressive \
  --date-ranges 2024-01-01_to_2024-12-31 \
  --tickers RELIANCE TCS INFY ZOMATO ADANIENT \
  --parallel
```
- **Risk**: 20% max position
- **Use Case**: Higher returns, growth-focused
- **Features**: Parallel processing, advanced risk management

### **Options Template** (Options Strategies)
```bash
python src/runners/unified_runner.py \
  --mode backtest \
  --template options \
  --dates 2024-01-01 \
  --tickers NIFTY BANKNIFTY
```
- **Risk**: 15% max position
- **Use Case**: Options trading strategies  
- **Features**: Greeks calculation, volatility modeling

### **Portfolio Diversified Template** (Multi-Asset)
```bash
python src/runners/unified_runner.py \
  --mode backtest \
  --template portfolio_diversified \
  --date-ranges 2024-01-01_to_2024-12-31 \
  --tickers RELIANCE TCS INFY ZOMATO ADANIENT TATAMOTORS
```
- **Risk**: 15% max position, portfolio-level controls
- **Use Case**: Diversified portfolio management
- **Features**: Portfolio optimization, correlation analysis

---

## 🔧 **Advanced Options**

### **Execution Control**
```bash
--parallel                    # Enable parallel processing
--max-workers 4              # Number of CPU cores to use  
--no-validation              # Skip data validation
--skip-visualization         # Skip chart generation
```

### **Data Management**
```bash
--data-dir "custom/data/path"        # Override data directory
--output-dir "custom/output/path"    # Override output directory  
--cache-enabled                      # Enable data caching
--force-data-refresh                 # Force fresh data download
```

### **Strategy Parameters** 
```bash
# JSON string for strategy parameters
--strategy-params '{"lookback": 20, "threshold": 0.02, "stop_loss": 0.03}'

# Override individual parameters
--lookback 25
--threshold 0.015
--stop-loss 0.05
```

### **Logging & Debug**
```bash
--log-level {DEBUG,INFO,WARNING,ERROR}    # Set logging verbosity
--log-file "custom_log.log"               # Log to specific file
--verbose                                 # Detailed console output
--quiet                                   # Minimal console output  
--profile                                 # Enable performance profiling
```

---

## 📊 **Data Commands**

### **Data Validation**
```bash
# Check specific dates
python src/runners/unified_runner.py --mode validate --dates 2024-01-01 2024-01-02

# Validate date ranges
python src/runners/unified_runner.py --mode validate --date-ranges 2024-01-01_to_2024-12-31

# Check specific tickers
python src/runners/unified_runner.py --mode validate --dates 2024-01-01 --tickers RELIANCE TCS
```

### **Data Fetching** (with broker setup)
```bash
# Fetch data for specific date range
python src/runners/unified_runner.py --mode fetch --date-ranges 2024-01-01_to_2024-01-31 --tickers RELIANCE TCS

# Fetch data for multiple date ranges
python src/runners/unified_runner.py --mode fetch --date-ranges 2024-01-01_to_2024-01-15 2024-02-01_to_2024-02-15 --tickers RELIANCE
```

---

## 🎮 **Complete Examples**

### **1. Quick Test Run**
```bash
python src/runners/unified_runner.py \
  --mode backtest \
  --template minimal \
  --dates 2024-01-01 \
  --tickers RELIANCE \
  --log-level INFO
```

### **2. Production Backtest**  
```bash
python src/runners/unified_runner.py \
  --mode backtest \
  --template conservative \
  --date-ranges 2024-01-01_to_2024-12-31 \
  --tickers RELIANCE TCS INFY \
  --parallel \
  --max-workers 4 \
  --cache-enabled
```

### **3. Custom Strategy with Parameters**
```bash
python src/runners/unified_runner.py \
  --mode backtest \
  --config config/my_custom.yaml \
  --strategies custom_ma \
  --strategy-params '{"fast_ma": 10, "slow_ma": 30}' \
  --dates 2024-01-01 2024-01-02 \
  --tickers RELIANCE TCS
```

### **4. Analysis Only** (Reuse Existing Data)
```bash
python src/runners/unified_runner.py \
  --mode analyze \
  --date-ranges 2024-01-01_to_2024-12-31 \
  --tickers RELIANCE \
  --output-dir "outputs/custom_analysis"
```

### **5. Data Quality Check**
```bash
python src/runners/unified_runner.py \
  --mode validate \
  --dates 2024-01-01 2024-01-02 2024-01-03 \
  --tickers RELIANCE TCS INFY ZOMATO \
  --log-level DEBUG
```

---

## 🛠️ **Configuration File Usage**

### **Create Custom Config**
```bash
# Copy template
cp config/templates/conservative.yaml my_config.yaml

# Edit parameters
# ... modify YAML file ...

# Use custom config
python src/runners/unified_runner.py \
  --mode backtest \
  --config my_config.yaml \
  --dates 2024-01-01
```

### **YAML Configuration Structure**
```yaml
# my_config.yaml
strategy:
  name: "mse"
  risk_profile: "custom"
  
risk:
  max_position_size: 0.12      # 12% max position
  stop_loss_pct: 0.04          # 4% stop loss
  
data:
  data_pool_dir: "data/pools"
  default_timeframe: "1minute"
  
validation:
  enabled: true
  strict_mode: false
```

---

## 🔍 **Troubleshooting Commands**

### **System Health Check**
```bash
python src/runners/unified_runner.py --mode validate --dates 2024-01-01
```

### **Verbose Debugging**
```bash
python src/runners/unified_runner.py \
  --mode backtest \
  --template minimal \
  --dates 2024-01-01 \
  --log-level DEBUG \
  --verbose
```

### **Performance Profiling**
```bash
python src/runners/unified_runner.py \
  --mode backtest \
  --template conservative \
  --dates 2024-01-01 \
  --profile \
  --log-level INFO
```

---

## ⚡ **Performance Tips**

1. **Use parallel processing**: `--parallel --max-workers 4`
2. **Enable caching**: `--cache-enabled`  
3. **Limit date ranges**: Start with small date ranges for testing
4. **Use appropriate templates**: `minimal` for testing, `conservative` for production
5. **Validate data first**: Always run `--mode validate` before large backtests

---

## 🎯 **Common Workflows**

### **New Strategy Development**
```bash
# 1. Quick validation
python src/runners/unified_runner.py --mode validate --dates 2024-01-01

# 2. Minimal test  
python src/runners/unified_runner.py --mode backtest --template minimal --dates 2024-01-01 --tickers RELIANCE

# 3. Extended test
python src/runners/unified_runner.py --mode backtest --template conservative --date-ranges 2024-01-01_to_2024-03-31 --tickers RELIANCE TCS

# 4. Production run
python src/runners/unified_runner.py --mode backtest --template conservative --date-ranges 2024-01-01_to_2024-12-31 --tickers RELIANCE TCS INFY --parallel
```

### **Risk Analysis Workflow**
```bash
# 1. Conservative baseline
python src/runners/unified_runner.py --mode backtest --template conservative --date-ranges 2024-01-01_to_2024-12-31 --tickers RELIANCE

# 2. Aggressive comparison  
python src/runners/unified_runner.py --mode backtest --template aggressive --date-ranges 2024-01-01_to_2024-12-31 --tickers RELIANCE

# 3. Analysis comparison
python src/runners/unified_runner.py --mode analyze --date-ranges 2024-01-01_to_2024-12-31 --tickers RELIANCE
```

---

## 📋 **Exit Codes**

- `0`: Success
- `1`: General error
- `2`: Configuration error  
- `3`: Data validation error
- `4`: Strategy execution error
- `5`: Analysis generation error

---

## 🤝 **Getting Help**

```bash
# Built-in help
python src/runners/unified_runner.py --help

# Check available templates
ls config/templates/

# Check available strategies  
python -c "from src.strategies.strategy_factory import get_available_strategies; print(get_available_strategies())"
```

For more help:
- **Setup Issues**: See `docs/SETUP_GUIDE.md`
- **Broker Connection**: See `docs/BROKER_SETUP.md`  
- **Understanding Results**: See `docs/OUTPUT_GUIDE.md`
- **Custom Strategies**: See `docs/STRATEGY_GUIDE.md`
- **Risk Configuration**: See `docs/TEMPLATE_GUIDE.md`

---

## 📋 **Minimum Requirements by Mode**

| **Mode** | **Required Arguments** | **Optional Arguments** | **Auto-Discovery** | **Notes** |
|----------|----------------------|----------------------|-------------------|-----------|
| **`validate`** | `--date-ranges` | `--tickers` | ✅ Auto-discovers tickers from data pools | Most flexible mode |
| **`fetch`** | `--date-ranges` + `--tickers` | `--strategies` | ❌ | Requires broker credentials |
| **`backtest`** | `--date-ranges` + `--tickers` | `--strategies` | ❌ | Defaults to `sma_crossover` strategy |
| **`analyze`** | `--date-ranges` + `--tickers` | `--strategies` | ❌ | Runs internal backtest + analysis |
| **`visualize`** | `--date-ranges` + `--tickers` | `--strategies` | ❌ | Runs internal backtest + visualization |

### **💡 Key Points:**
- **All modes require `--date-ranges`** - No mode works without date information
- **Validate mode is special** - Only mode that can auto-discover tickers from existing data
- **Strategy defaults** - If not specified, `sma_crossover` strategy is used for backtest/analyze/visualize
- **Clear error messages** - Missing requirements show actionable error messages

### **❌ Common Error Messages:**
```bash
# Missing date ranges
Error: At least one date or date range must be specified using --dates or --date-ranges

# Missing tickers (for fetch/backtest/analyze/visualize)
Error: Tickers must be specified for this mode
```
