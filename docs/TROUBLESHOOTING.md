# 🔧 Troubleshooting Guide

**Common issues and solutions for the algorithmic trading backtester.**

---

## 🚨 **Quick Fixes**

### **🔥 Most Common Issues**

#### **1. "Module not found" or Import Errors**
```bash
# Problem: Python can't find modules
# Solution: Check Python path and installation
cd backtester
python -c "import sys; print(sys.path)"
pip install -r requirements.txt
```

#### **2. "No data found" Errors**
```bash
# Problem: Missing data files
# Solution: Check data directory structure
ls -la data/pools/
python src/runners/unified_runner.py --mode validate --dates 2024-01-01
```

#### **3. "Strategy not registered" Error**
```bash
# Problem: Strategy not found
# Solution: Check strategy registration
python -c "from src.strategies.strategy_factory import get_available_strategies; print(get_available_strategies())"
```

#### **4. Configuration Errors**
```bash
# Problem: Invalid YAML configuration
# Solution: Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config/templates/conservative.yaml'))"
```

---

## 🛠️ **Installation Issues**

### **Python Environment Problems**

#### **Wrong Python Version**
```bash
# Check Python version (need 3.9+)
python --version

# Install correct version
# Windows: Download from python.org
# Linux: sudo apt install python3.9
# Mac: brew install python@3.9
```

#### **Virtual Environment Issues**
```bash
# Create fresh virtual environment
python -m venv trading_env
source trading_env/bin/activate  # Linux/Mac
trading_env\Scripts\activate     # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### **Package Installation Failures**
```bash
# Clear pip cache
pip cache purge

# Install with verbose output
pip install -r requirements.txt -v

# Install packages individually
pip install pandas numpy scipy matplotlib

# For Windows compilation issues
pip install --only-binary=all -r requirements.txt
```

### **Directory Permission Issues**

#### **Permission Denied Errors**
```bash
# Linux/Mac: Fix permissions
sudo chown -R $USER:$USER backtester/
chmod -R 755 backtester/

# Windows: Run as administrator or check folder permissions
```

#### **Path Not Found Errors**
```bash
# Check current directory
pwd
ls -la

# Navigate to correct directory
cd path/to/backtester

# Verify file structure
ls -la src/ config/ docs/
```

---

## 📊 **Data Issues**

### **Data Loading Problems**

#### **"Data directory not found"**
```bash
# Create data directory structure
mkdir -p data/pools
mkdir -p outputs

# Check configuration
grep -r "data_pool_dir" config/
```

#### **"Insufficient data" Errors**
```bash
# Check data availability
ls -la data/pools/
python utils/data_checker.py --date 2024-01-01

# Use sample data for testing
python utils/create_sample_data.py
```

#### **Data Format Issues**
```bash
# Validate data format
python -c "
import pandas as pd
data = pd.read_csv('data/pools/sample.csv')
print(data.columns)
print(data.head())
"
```

### **Broker API Issues**

#### **API Key Problems**
```bash
# Check API key configuration
ls -la config/access_tokens/

# Test API connection
python utils/test_broker_connection.py --broker upstox
python utils/test_broker_connection.py --broker zerodha
```

#### **Rate Limiting**
```bash
# Add delays between API calls
# Edit config/config.py:
# 'REQUEST_DELAY': 1.0  # 1 second delay
```

#### **Authentication Failures**
```bash
# Regenerate access tokens
rm config/access_tokens/upstox/*
rm config/access_tokens/zerodha/*

# Follow broker setup guide
# See docs/BROKER_SETUP.md
```

---

## ⚙️ **Configuration Issues**

### **YAML Configuration Problems**

#### **Invalid YAML Syntax**
```bash
# Validate YAML files
python -c "
import yaml
try:
    with open('config/templates/conservative.yaml') as f:
        config = yaml.safe_load(f)
    print('YAML is valid')
except yaml.YAMLError as e:
    print(f'YAML error: {e}')
"
```

#### **Missing Required Parameters**
```bash
# Check required parameters
python src/runners/unified_runner.py --mode validate --config my_config.yaml
```

#### **Invalid Parameter Values**
```bash
# Common parameter issues:
# - max_position_size > 1.0 (should be <= 1.0)
# - negative stop_loss_pct (should be positive)
# - invalid risk_profile name

# Validate ranges
python -c "
config = {'risk': {'max_position_size': 0.15}}
assert 0 < config['risk']['max_position_size'] <= 1.0
print('Parameters valid')
"
```

### **Template Loading Issues**

#### **Template Not Found**
```bash
# List available templates
ls -la config/templates/

# Use correct template name
python src/runners/unified_runner.py --template conservative
# NOT: python src/runners/unified_runner.py --template Conservative
```

#### **Template Inheritance Problems**
```bash
# Check template structure
python -c "
import yaml
with open('config/templates/conservative.yaml') as f:
    template = yaml.safe_load(f)
print(yaml.dump(template, indent=2))
"
```

---

## 🎯 **Strategy Issues**

### **Strategy Registration Problems**

#### **Strategy Not Found**
```bash
# Check available strategies
python -c "
from src.strategies.strategy_factory import get_available_strategies
print('Available strategies:', get_available_strategies())
"

# Register missing strategy
# Edit src/strategies/register_strategies.py
# Add: factory.register('my_strategy', MyStrategy)
```

#### **Strategy Import Errors**
```bash
# Test strategy import individually
python -c "
from src.strategies.strategy_mse import MSEStrategy
print('Strategy imported successfully')
"

# Check for syntax errors in strategy file
python -m py_compile src/strategies/strategy_mse.py
```

### **Strategy Execution Issues**

#### **Signal Generation Errors**
```bash
# Enable debug logging
python src/runners/unified_runner.py \
  --mode backtest \
  --template minimal \
  --dates 2024-01-01 \
  --tickers RELIANCE \
  --log-level DEBUG
```

#### **Performance Issues**
```bash
# Reduce data size for testing
python src/runners/unified_runner.py \
  --mode backtest \
  --template minimal \
  --dates 2024-01-01 \
  --tickers RELIANCE

# Use single-threaded execution
python src/runners/unified_runner.py \
  --mode backtest \
  --template minimal \
  --no-parallel
```

---

## 🖥️ **Execution Issues**

### **Memory Problems**

#### **Out of Memory Errors**
```bash
# Reduce batch size
# Edit config/unified_config.py:
# batch_size: 500  # Reduce from 1000

# Use chunked processing
python src/runners/unified_runner.py \
  --mode backtest \
  --template conservative \
  --max-workers 1 \
  --chunk-size 100
```

#### **Memory Leaks**
```bash
# Enable memory monitoring
python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"

# Use garbage collection
python -c "
import gc
gc.collect()
print('Garbage collection completed')
"
```

### **Performance Issues**

#### **Slow Execution**
```bash
# Use parallel processing
python src/runners/unified_runner.py --parallel --max-workers 4

# Enable caching
python src/runners/unified_runner.py --cache-enabled

# Profile performance
python -m cProfile -o profile.out src/runners/unified_runner.py --mode backtest --template minimal
```

#### **Timeout Issues**
```bash
# Increase timeout values
# Edit config/config.py:
# 'REQUEST_TIMEOUT': 60  # Increase from 30 seconds

# Reduce data size
python src/runners/unified_runner.py --dates 2024-01-01  # Single day
```

---

## 📈 **Output Issues**

### **Missing Results**

#### **No Output Files Generated**
```bash
# Check output directory permissions
ls -la outputs/
mkdir -p outputs
chmod 755 outputs/

# Enable output saving
# Ensure in config:
# output:
#   save_trades: true
#   save_metrics: true
```

#### **Incomplete Results**
```bash
# Check for execution errors
tail -n 50 logs/backtester.log

# Run with verbose logging
python src/runners/unified_runner.py \
  --mode backtest \
  --template minimal \
  --verbose \
  --log-level DEBUG
```

### **Visualization Problems**

#### **Plot Generation Failures**
```bash
# Check matplotlib backend
python -c "
import matplotlib
print('Backend:', matplotlib.get_backend())
matplotlib.use('Agg')  # Use non-interactive backend
"

# Install required packages
pip install matplotlib seaborn plotly

# Skip visualization for testing
python src/runners/unified_runner.py --skip-visualization
```

#### **Chart Display Issues**
```bash
# Save plots to files instead of displaying
# Edit config:
# output:
#   save_plots: true
#   show_plots: false

# Check available display
echo $DISPLAY  # Linux
# If empty, plots can't be displayed
```

---

## 🔍 **Debugging Tools**

### **System Health Check**

```bash
# Complete system validation
python src/runners/unified_runner.py --mode validate --dates 2024-01-01

# Check Python environment
python -c "
import sys
print('Python version:', sys.version)
print('Python path:', sys.path)
import pandas, numpy, matplotlib
print('Key packages imported successfully')
"

# Check file permissions
ls -la src/ config/ data/ outputs/
```

### **Verbose Debugging**

```bash
# Maximum verbosity
python src/runners/unified_runner.py \
  --mode backtest \
  --template minimal \
  --dates 2024-01-01 \
  --tickers RELIANCE \
  --log-level DEBUG \
  --verbose \
  --profile

# Step-by-step execution
python -u src/runners/unified_runner.py \
  --mode backtest \
  --template minimal \
  --dates 2024-01-01 \
  --log-level DEBUG 2>&1 | tee debug_output.log
```

### **Component Testing**

```bash
# Test individual components
python -c "from config.unified_config import BacktestConfig; print('Config OK')"
python -c "from src.strategies.strategy_factory import StrategyFactory; print('Strategy Factory OK')"
python -c "from src.runners.unified_runner import UnifiedBacktesterRunner; print('Runner OK')"

# Test data loading
python -c "
import pandas as pd
data = pd.read_csv('data/pools/sample.csv')
print(f'Data shape: {data.shape}')
print(f'Columns: {data.columns.tolist()}')
"
```

---

## 🆘 **Emergency Recovery**

### **Complete Reset**

```bash
# Clean installation
rm -rf trading_env/  # Remove virtual environment
rm -rf outputs/*     # Clear outputs
rm -rf cache/*       # Clear cache
rm -rf __pycache__/  # Clear Python cache

# Fresh setup
python -m venv trading_env
source trading_env/bin/activate
pip install -r requirements.txt

# Verify installation
python src/runners/unified_runner.py --mode validate --dates 2024-01-01
```

### **Backup Recovery**

```bash
# Backup important configurations
cp -r config/ config_backup/
cp -r src/strategies/ strategies_backup/

# Restore from backup
cp -r config_backup/ config/
cp -r strategies_backup/ src/strategies/
```

---

## 📊 **Error Code Reference**

| Exit Code | Meaning | Solution |
|-----------|---------|----------|
| 0 | Success | No action needed |
| 1 | General error | Check logs for details |
| 2 | Configuration error | Validate YAML config |
| 3 | Data validation error | Check data availability |
| 4 | Strategy execution error | Debug strategy code |
| 5 | Analysis generation error | Check output permissions |

---

## 🔧 **Platform-Specific Issues**

### **Windows Issues**

#### **Path Separator Problems**
```bash
# Use forward slashes or raw strings
data_dir = "data/pools"  # Correct
# NOT: data_dir = "data\pools"

# Or use pathlib
from pathlib import Path
data_dir = Path("data") / "pools"
```

#### **PowerShell Execution Policy**
```powershell
# Enable script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Run Python scripts
python src/runners/unified_runner.py --mode validate --dates 2024-01-01
```

### **Linux/Mac Issues**

#### **Permission Issues**
```bash
# Fix file permissions
chmod +x src/runners/unified_runner.py
chmod -R 755 backtester/

# User permissions
sudo chown -R $USER:$USER backtester/
```

#### **Display Issues (Linux)**
```bash
# Set display for GUI applications
export DISPLAY=:0

# Use X11 forwarding for SSH
ssh -X user@server
```

---

## 🤖 **AI-Assisted Debugging**

### **Debugging Prompt**

Use this prompt with any AI model when you encounter issues:

```
I'm having an issue with the algorithmic trading backtester system.

ERROR DETAILS:
- Error message: [copy exact error message]
- Command used: [copy exact command]
- Operating system: [Windows/Linux/Mac]
- Python version: [version number]

CONTEXT:
- What I was trying to do: [describe goal]
- When the error occurs: [during installation/configuration/execution]
- What I've already tried: [list attempted solutions]

SYSTEM INFO:
- Repository: algorithmic trading backtester
- Framework: Python-based modular system
- Key components: unified_runner.py, strategy system, YAML config

LOGS/OUTPUT:
[paste relevant log output or error traceback]

Please provide:
1. Explanation of what's causing the error
2. Step-by-step solution
3. Commands to test the fix
4. Prevention tips for the future
```

---

## 📞 **Getting Additional Help**

### **Documentation Resources**
- **Setup Guide**: `docs/SETUP_GUIDE.md`
- **CLI Reference**: `docs/CLI_REFERENCE.md`
- **Strategy Guide**: `docs/STRATEGY_GUIDE.md`
- **Template Guide**: `docs/TEMPLATE_GUIDE.md`
- **Output Guide**: `docs/OUTPUT_GUIDE.md`

### **Self-Help Commands**
```bash
# Built-in help
python src/runners/unified_runner.py --help

# System validation
python src/runners/unified_runner.py --mode validate --dates 2024-01-01

# Check available options
ls config/templates/
python -c "from src.strategies.strategy_factory import get_available_strategies; print(get_available_strategies())"
```

### **Log Analysis**
```bash
# Check recent logs
tail -n 100 logs/backtester.log

# Search for specific errors
grep -i "error" logs/backtester.log
grep -i "failed" logs/backtester.log

# Monitor real-time logs
tail -f logs/backtester.log
```

---

## ✅ **Prevention Checklist**

### **Before Running Backtests**
- [ ] Virtual environment activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Configuration validated
- [ ] Data directory structure correct
- [ ] Templates accessible
- [ ] Strategies registered

### **Regular Maintenance**
- [ ] Update dependencies monthly
- [ ] Clear cache periodically (`rm -rf cache/*`)
- [ ] Backup configurations
- [ ] Monitor disk space in `outputs/`
- [ ] Review log files for warnings

---

**When in doubt, start simple and build up complexity gradually! 🚀**
