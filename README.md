# 🚀 Algorithmic Trading Backtester

A **production-ready, modular backtesting system** for algorithmic trading strategies with real broker integration, comprehensive analysis, and AI-assisted configuration.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Brokers](https://img.shields.io/badge/Brokers-Zerodha%20%7C%20Upstox%20%7C%20Binance-orange.svg)](docs/BROKER_SETUP.md)

---

## ✨ **Quick Start with AI Assistant**

**🤖 Use this prompt with any AI model (ChatGPT, Claude, Gemini) for personalized setup assistance:**

```
I'm setting up an algorithmic trading backtester. Please help me configure it based on my requirements.

SYSTEM INFO:
- Repository: https://github.com/yourusername/backtester
- Language: Python 3.9+
- Supported Brokers: Zerodha Kite API, Upstox API, Binance API
- Architecture: Modular, production-ready with real-time data

MY REQUIREMENTS:
[Describe your trading style, risk tolerance, preferred broker, strategies of interest]

AVAILABLE DOCUMENTATION:
- Setup Guide: docs/SETUP_GUIDE.md (Installation, dependencies, environment)
- Broker Setup: docs/BROKER_SETUP.md (API keys, authentication, data fetching)
- Strategy Guide: docs/STRATEGY_GUIDE.md (Custom strategy development)
- Template Guide: docs/TEMPLATE_GUIDE.md (Risk templates, YAML configuration)
- CLI Reference: docs/CLI_REFERENCE.md (All command-line options)
- Output Guide: docs/OUTPUT_GUIDE.md (Understanding results, visualizations)

CONFIGURATION TEMPLATES:
- minimal.yaml: Ultra-safe learning (5% max position)
- conservative.yaml: Low-risk trading (15% max position)
- aggressive.yaml: High-risk trading (20% max position)
- options.yaml: Options trading strategies
- portfolio_diversified.yaml: Multi-ticker portfolio

Please provide step-by-step setup instructions, recommend appropriate templates, and suggest CLI commands based on my requirements.
```

---

## 🆕 **Recent Updates & Improvements**

### **System Optimization (Latest)**
- **Cleaned Architecture**: Removed redundant files and streamlined codebase
- **Dual Configuration System**: 
  - `config/config.py` - Broker connections and data provider settings
  - `config/unified_config.py` - Strategy parameters and risk management
- **Enhanced Data Providers**: Added Binance support for cryptocurrency data
- **Improved Documentation**: Comprehensive guides and troubleshooting

### **Configuration Architecture**
The system uses a **dual configuration approach** for optimal separation of concerns:

```
📋 config/config.py          # Broker API credentials, data connections
📋 config/unified_config.py   # Trading strategies, risk parameters
📁 config/templates/          # Pre-built risk management templates
```

This design allows independent management of:
- **Infrastructure** (brokers, authentication, data sources)
- **Trading Logic** (strategies, risk rules, portfolio settings)

---

## 🎯 **Core Features**

### **📊 Trading System**
- **Real Broker Integration**: Zerodha Kite, Upstox, & Binance APIs
- **Multi-Timeframe Support**: 1min to monthly data
- **Live Data Fetching**: Real-time and historical data
- **Strategy Framework**: Modular, extensible strategy system

### **🛡️ Risk Management**
- **Portfolio-Level Controls**: Position sizing, drawdown limits
- **Trade-Level Protection**: Stop-loss, take-profit, trailing stops
- **Risk Templates**: Pre-configured risk profiles
- **Real-Time Monitoring**: Live risk assessment

### **📈 Analysis & Visualization**
- **Comprehensive Reports**: Performance metrics, trade analysis
- **Interactive Charts**: Price action, signals, portfolio performance
- **Statistical Analysis**: Sharpe ratio, maximum drawdown, win rate
- **Export Capabilities**: CSV, JSON, PNG formats

### **⚡ Performance**
- **Parallel Processing**: Multi-core execution
- **Efficient Data Handling**: Optimized for large datasets
- **Caching System**: Intelligent data caching
- **Modular Architecture**: Clean, maintainable codebase

---

## 🚀 **30-Second Setup**

```bash
# 1. Clone and Install
git clone <repository-url>
cd backtester
pip install -r requirements.txt

# 2. Run with sample data
python src/runners/unified_runner.py --mode backtest --date-ranges 2025-06-06_to_2025-06-07 --tickers RELIANCE

# 3. Set up broker (optional, for live data)
# See docs/BROKER_SETUP.md for API key setup
```

---

## 📚 **Documentation Guide**

### **🎯 For Different User Types**

| **User Type** | **Start Here** | **Key Documents** |
|---------------|----------------|-------------------|
| **First-time user** | `docs/SETUP_GUIDE.md` | Setup → Broker → CLI Reference |
| **Strategy developer** | `docs/STRATEGY_GUIDE.md` | Strategy → Template → Output |
| **Risk manager** | `docs/TEMPLATE_GUIDE.md` | Template → CLI Reference → Output |
| **Data analyst** | `docs/OUTPUT_GUIDE.md` | Output → CLI Reference → Setup |

### **📋 Essential Documentation**

- **📖 [Setup Guide](docs/SETUP_GUIDE.md)**: Installation, dependencies, environment setup
- **🔑 [Broker Setup](docs/BROKER_SETUP.md)**: API keys, authentication, data fetching
- **⚙️ [CLI Reference](docs/CLI_REFERENCE.md)**: Complete command-line interface guide
- **🎯 [Strategy Guide](docs/STRATEGY_GUIDE.md)**: Custom strategy development
- **📊 [Template Guide](docs/TEMPLATE_GUIDE.md)**: Risk templates and YAML configuration
- **📈 [Output Guide](docs/OUTPUT_GUIDE.md)**: Understanding results and visualizations
- **🔧 [Troubleshooting](docs/TROUBLESHOOTING.md)**: Common issues and solutions

---

## 🎮 **CLI Examples**

```bash
# ⚡ MINIMAL USAGE - Auto-discover tickers from data pools
python src/runners/unified_runner.py --mode backtest --date-ranges 2024-01-01_to_2024-12-31
python src/runners/unified_runner.py --mode analyze --date-ranges 2024-01-01_to_2024-12-31
python src/runners/unified_runner.py --mode visualize --date-ranges 2024-01-01_to_2024-12-31
python src/runners/unified_runner.py --mode validate --date-ranges 2024-01-01_to_2024-12-31

# 🎯 INTERACTIVE FETCH - No arguments needed
python src/runners/unified_runner.py --mode fetch

# 🎯 SPECIFIC TICKERS - Override auto-discovery
python src/runners/unified_runner.py --mode backtest --date-ranges 2024-01-01_to_2024-12-31 --tickers RELIANCE TCS

# 🚀 FULL CONTROL - All advanced features available
python src/runners/unified_runner.py \
  --mode backtest \
  --template aggressive \
  --date-ranges 2024-01-01_to_2024-12-31 \
  --tickers RELIANCE TCS INFY \
  --strategies sma_crossover bollinger_bands \
  --parallel \
  --max-workers 4

# 📊 EXPLICIT FETCH - With specific parameters
python src/runners/unified_runner.py \
  --mode fetch \
  --date-ranges 2024-01-01_to_2024-01-31 \
  --tickers RELIANCE TCS

# 🔧 CUSTOM CONFIGURATION
python src/runners/unified_runner.py \
  --mode backtest \
  --config my_custom_config.yaml \
  --date-ranges 2024-01-01_to_2024-12-31 \
  --parallel
```

---

## 🏗️ **Project Structure** (Optimized & Clean)

```
backtester/
├── 📁 src/                     # Core modular system
│   ├── strategies/             # Trading strategies (MSE, SMA, Bollinger, etc.)
│   ├── core/                   # Analysis, risk, ETL, data processing
│   │   ├── etl/                # Data fetching and provider management
│   │   ├── risk/               # Risk management engine
│   │   ├── analysis/           # Performance analysis and visualization
│   │   └── output/             # Three-file output system
│   └── runners/                # Execution engines and CLI handlers
├── 📁 config/                  # Dual configuration system
│   ├── config.py               # Broker connections & data providers
│   ├── unified_config.py       # Strategy & risk parameters
│   ├── templates/              # Pre-built risk templates
│   └── access_tokens/          # Broker API credentials (user-created)
├── 📁 docs/                    # Comprehensive documentation
├── 📁 data/pools/              # Market data storage (auto-created)
├── 📁 outputs/                 # Results and reports (auto-created)
├── 📋 CLAUDE.md                # Claude Code integration guide
├── 📋 comprehensive_data_puller.py  # Historical data utility (1996-2025)
├── 📋 mse_data_puller.py       # MSE strategy data utility
└── 📋 requirements.txt         # Python dependencies
```

### **Key Architecture Improvements**
- **Clean Modular Design**: Separated ETL, risk, analysis, and output systems
- **Dual Configuration**: Infrastructure vs trading logic separation
- **Specialized Data Pullers**: Different tools for different data requirements
- **Removed Legacy Files**: Cleaned ~14 empty/redundant files for optimal maintainability

---

## 🤝 **Getting Help**

### **AI-Powered Assistance**
Use the prompt at the top with any AI model for personalized help.

### **Documentation Flow**
1. **Setup Issues**: `docs/SETUP_GUIDE.md` → `docs/TROUBLESHOOTING.md`
2. **Broker Problems**: `docs/BROKER_SETUP.md` → AI prompt with broker details
3. **Strategy Questions**: `docs/STRATEGY_GUIDE.md` → AI prompt with strategy requirements
4. **Configuration Help**: `docs/TEMPLATE_GUIDE.md` → AI prompt with risk preferences

### **Common Use Cases**
- **"I want to test a strategy"**: Use `docs/CLI_REFERENCE.md` + AI prompt
- **"I need to connect my broker"**: Use `docs/BROKER_SETUP.md`
- **"I don't understand the results"**: Use `docs/OUTPUT_GUIDE.md`
- **"I want custom risk settings"**: Use `docs/TEMPLATE_GUIDE.md` + AI prompt

---

## 🎯 **Templates Overview**

| Template | Risk Level | Max Position | Use Case |
|----------|------------|-------------|----------|
| `minimal` | Ultra-safe | 5% | Learning, testing |
| `conservative` | Low | 15% | Stable income |
| `aggressive` | High | 20% | Growth focused |
| `options` | Moderate | 15% | Options strategies |
| `portfolio_diversified` | Balanced | 15% | Multi-asset |

---

## 🔧 **Production Features**

- **🔄 Live Data Integration**: Real-time market data
- **🛡️ Risk Management**: Multi-level protection
- **📊 Performance Analytics**: Comprehensive metrics
- **⚡ Parallel Processing**: High-performance execution
- **💾 Data Persistence**: Reliable storage
- **📈 Visualization**: Professional charts
- **🔍 Monitoring**: System health checks

---

## 📄 **License**

MIT License - See `LICENSE` file for details.

---

## 🚀 **Ready to Start?**

1. **🔥 Try the 30-second setup above**
2. **🤖 Use the AI prompt for personalized help**
3. **📚 Browse the documentation**
4. **💬 Join our community for support**

**Happy Trading! 📈**
