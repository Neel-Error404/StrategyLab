# 🚀 Strategy Lab - Trading Backtester

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
- Repository: https://github.com/yourusername/StrategyLab  (→ Fork before use)
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
- **Derivatives Pipeline**: Phase 1–4 options validation and replay flows are production-tested with hybrid pricing, Upstox ingestion, and full multi-ticker coverage.【F:src/core/options/PHASE4_COMPLETE.md†L1-L24】【F:src/core/options/validation/README.md†L1-L67】
- **Analysis Migration**: Nine legacy analytics have been ported to the config-driven framework so trade diagnostics, ticker ranking, and validation checks run without hardcoded paths.【F:analysis/generic/scripts/02_trade_type_analysis.py†L1-L41】【F:analysis/generic/scripts/09_validation_check.py†L1-L39】
- **Enhanced Data Providers**: Binance crypto feed and the Indian equities master pipeline combine broker APIs with YFinance fallbacks for reference data and discovery.【F:src/core/etl/data_provider/binance_provider.py†L21-L92】【F:src/data_tools/indian_equities_master/discovery.py†L7-L33】
- **Parquet-First Storage**: Unified loader/fetcher modules read and write ticker-first parquet layouts with automatic CSV fallback when needed.【F:src/core/etl/loader.py†L23-L74】【F:src/core/etl/data_fetcher.py†L93-L197】

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

### **🧮 Options & Derivatives**
- **Replay Engine**: Converts equity trade ledgers into option executions with lifecycle tracking and portfolio-aware risk checks.【F:src/core/options/replay/engine.py†L1-L120】【F:src/core/options/replay/trade_mapper.py†L1-L74】
- **Hybrid Pricing**: Combines actual Upstox OHLC chains with Black-Scholes synthetic fills, including automatic fallbacks and validation reports.【F:src/core/options/validation/pricing_validator.py†L257-L335】【F:src/core/options/replay/pricing.py†L150-L238】
- **Operational Reports**: Phase completion manifests document throughput, P&L, and data quality for Phase 3 MVP and Phase 4 production tests.【F:src/core/options/PHASE3_COMPLETE.md†L1-L32】【F:src/core/options/PHASE4_COMPLETE.md†L1-L35】

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
- **Config-Driven Analytics**: Trade type analysis, stop-loss simulation, and validation sweeps run through reusable modules with YAML-controlled inputs.【F:analysis/generic/scripts/04_stop_loss_simulation.py†L1-L44】【F:analysis/generic/modules/config_loader.py†L300-L373】

### **⚡ Performance**
- **Parallel Processing**: Multi-core execution
- **Efficient Data Handling**: Optimized for large datasets
- **Caching System**: Intelligent data caching
- **Modular Architecture**: Clean, maintainable codebase

### **₿ Crypto Coverage**
- **BTC & Top 30 Tokens**: Binance provider ships symbol metadata and fetch routines for Bitcoin pairs and the highest-liquidity USDT listings.【F:src/core/etl/data_provider/binance_provider.py†L21-L122】【F:config/binance_instruments.csv†L13-L70】
- **Unified Config Templates**: Crypto-friendly defaults control trade file formats, leverage, and data cadence via YAML templates.【F:config/templates/aggressive.yaml†L79-L92】【F:config/unified_config.py†L118-L156】

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
  --parallel --max-workers 6 --skip-validation
```

---

## Utilities

- `src/scripts/compare_broker_vs_strategy.py`
  - Compares broker order CSVs to strategy trades from a run directory with entry/exit tolerances; writes audit JSONs.
  - Example:
    - `python -m src.scripts.compare_broker_vs_strategy --orders broker_orders.csv --run-dir outputs/<RUN>/<STRAT>/<RANGE> --start 2025-08-26 --end 2025-09-04 --entry-tol 20 --exit-tol 20 --exit-enforce true`

- `src/scripts/monitor_progress.py`
  - Monitors a `historical_data/progress.json` file for long-running data jobs; prints live progress.

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

StrategyLab is released under the MIT License – see the [LICENSE](./LICENSE) file for details.

---

## 🚀 **Ready to Start?**

1. **🔥 Try the 5-minute setup above**
2. **🤖 Use the AI prompt for personalized help**
3. **📚 Browse the documentation**
4. **💬 Join our community for support**

**Happy Trading! 📈**
