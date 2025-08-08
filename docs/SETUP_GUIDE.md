# 📖 Setup Guide

Complete installation and environment setup for the algorithmic trading backtester.

---

## 🎯 **Quick Start (5 minutes)**

```bash
# 1. Install Python 3.9+ and Git
# 2. Clone repository
git clone <repository-url>
cd backtester

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python src/runners/unified_runner.py --mode validate --dates 2024-01-01

# 5. Run sample backtest
python src/runners/unified_runner.py --mode backtest --template conservative --dates 2024-01-01 --tickers RELIANCE
```

**✅ If no errors, you're ready to trade!**

---

## 🔧 **Detailed Installation**

### **Prerequisites**

```bash
# Required
Python 3.9+ 
Git 2.0+

# Recommended
Virtual environment manager (venv, conda, poetry)
```

### **Step 1: Environment Setup**

#### **Option A: Using venv (Recommended)**
```bash
# Create virtual environment
python -m venv trading_env

# Activate (Windows)
trading_env\Scripts\activate

# Activate (Linux/Mac)
source trading_env/bin/activate
```

#### **Option B: Using conda**
```bash
# Create environment
conda create -n trading python=3.9
conda activate trading
```

### **Step 2: Clone Repository**

```bash
git clone <repository-url>
cd backtester

# Verify structure
ls -la  # Should show src/, config/, docs/, etc.
```

### **Step 3: Install Dependencies**

```bash
# Core dependencies
pip install -r requirements.txt

# Verify key packages
python -c "import pandas, numpy, matplotlib, kiteconnect; print('✅ Core packages installed')"
```

### **Step 4: Directory Structure**

The system auto-creates these directories when needed:
```
backtester/
├── data/pools/          # Market data (auto-created)
├── outputs/             # Results (auto-created)
├── .runtime/logs/       # Log files (auto-created)
└── config/access_tokens/ # API tokens (manual setup)
```

---

## 📊 **Data Management**

### **Data Sources**

1. **Broker APIs** (Recommended)
   - **Zerodha Kite**: Real NSE data
   - **Upstox**: Real NSE data
   - Setup: See `docs/BROKER_SETUP.md`

2. **Sample Data** (Testing)
   ```bash
   # Generate sample data
   python utils/create_sample_data.py
   ```

### **Data Fetching**

```bash
# Fetch live data (requires broker setup)
python src/core/etl/data_fetcher.py

# Interactive mode
python src/core/etl/data_fetcher.py
# Follow prompts to select provider, timeframes, dates
```

### **Data Structure**

```
data/pools/
├── 2024-01-01_to_2024-01-31/
│   ├── 1minute/
│   │   ├── RELIANCE_1minute.csv
│   │   └── TCS_1minute.csv
│   ├── 5minute/
│   └── 1day/
└── 2024-02-01_to_2024-02-28/
```

---

## ⚙️ **Configuration**

### **Template System**

Choose a pre-built template:

```bash
# Conservative (15% max position)
python src/runners/unified_runner.py --template conservative

# Aggressive (20% max position)
python src/runners/unified_runner.py --template aggressive

# Learning mode (5% max position)
python src/runners/unified_runner.py --template minimal
```

### **Custom Configuration**

```yaml
# my_config.yaml
strategy:
  name: "mse"
  risk_profile: "custom"
  initial_capital: 100000

risk:
  max_position_size: 0.10  # 10%
  max_daily_loss: 0.02     # 2%
  stop_loss_pct: 0.03      # 3%

data:
  data_pool_dir: "data/pools"
  default_timeframe: "1minute"
```

Use with:
```bash
python src/runners/unified_runner.py --config my_config.yaml
```

---

## 🧪 **Verification Tests**

### **Test 1: System Health**
```bash
python src/runners/unified_runner.py --mode validate --dates 2024-01-01
```

**Expected Output:**
```
✅ Configuration loaded
✅ Strategy registration successful
✅ Data validation passed
✅ System ready
```

### **Test 2: Sample Backtest**
```bash
python src/runners/unified_runner.py \
  --mode backtest \
  --template conservative \
  --dates 2024-01-01 \
  --tickers RELIANCE
```

**Expected Output:**
```
✅ Backtest completed
✅ Analysis generated
✅ Visualizations created
📊 Results: outputs/YYYYMMDD_HHMMSS/
```

### **Test 3: Template Loading**
```bash
# Test all templates
for template in minimal conservative aggressive options; do
  echo "Testing $template..."
  python src/runners/unified_runner.py --template $template --mode validate --dates 2024-01-01
done
```

---

## 🛠️ **Environment Variables (Optional)**

Create `.env` file for enhanced security:

```bash
# .env
ZERODHA_API_KEY=your_api_key
ZERODHA_API_SECRET=your_secret
UPSTOX_CLIENT_ID=your_client_id
UPSTOX_CLIENT_SECRET=your_secret

# System settings
LOG_LEVEL=INFO
DATA_PROVIDER=upstox
TRADING_ENV=development
```

---

## 🐍 **Python Environment Details**

### **Supported Versions**
- **Python 3.9**: ✅ Recommended
- **Python 3.10**: ✅ Fully supported
- **Python 3.11**: ✅ Tested
- **Python 3.12**: ⚠️ Some dependencies may need updates

### **Key Dependencies**

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥2.0.0 | Data manipulation |
| numpy | ≥1.24.0 | Numerical computing |
| matplotlib | ≥3.7.0 | Visualization |
| kiteconnect | ≥4.1.0 | Zerodha API |
| PyYAML | ≥6.0 | Configuration |
| requests | ≥2.28.0 | HTTP client |

### **Optional Dependencies**

```bash
# Advanced analytics
pip install scikit-learn statsmodels

# Database support
pip install SQLAlchemy psycopg2-binary

# Development tools
pip install pytest black flake8
```

---

## 🔍 **Troubleshooting**

### **Common Issues**

#### **ImportError: No module named 'kiteconnect'**
```bash
pip install kiteconnect
# or
pip install -r requirements.txt --force-reinstall
```

#### **Permission denied creating directories**
```bash
# Check permissions
ls -la 

# Fix permissions (Linux/Mac)
chmod 755 .
mkdir -p data/pools outputs

# Windows: Run as administrator
```

#### **Python version issues**
```bash
# Check version
python --version

# Use specific version
python3.9 -m pip install -r requirements.txt
```

### **Getting Help**

1. **Check logs**: `.runtime/logs/`
2. **Validation**: `python src/runners/unified_runner.py --mode validate`
3. **Documentation**: `docs/TROUBLESHOOTING.md`
4. **AI Assistant**: Use the prompt from `README.md`

---

## 🚀 **Next Steps**

After successful setup:

1. **🔑 Broker Setup**: `docs/BROKER_SETUP.md` (for live data)
2. **📋 CLI Reference**: `docs/CLI_REFERENCE.md` (all commands)
3. **🎯 Strategy Development**: `docs/STRATEGY_GUIDE.md`
4. **📊 Understanding Results**: `docs/OUTPUT_GUIDE.md`

---

## 📋 **Installation Checklist**

- [ ] Python 3.9+ installed
- [ ] Virtual environment created and activated
- [ ] Repository cloned
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] System validation passed
- [ ] Sample backtest completed
- [ ] Templates loaded successfully
- [ ] (Optional) Broker API configured

**✅ Ready to start trading!**
