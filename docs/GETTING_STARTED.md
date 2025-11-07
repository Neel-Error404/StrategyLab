# Getting Started with StrategyLab Backtester

**A comprehensive guide to understanding and using the StrategyLab backtesting system.**

This guide provides a detailed introduction to the architecture, concepts, and workflows of StrategyLab. For a quick 15-minute setup, see [QUICKSTART.md](../QUICKSTART.md).

---

## Table of Contents

1. [What is StrategyLab?](#what-is-strategylab)
2. [System Architecture](#system-architecture)
3. [Core Concepts](#core-concepts)
4. [Complete Workflow](#complete-workflow)
5. [Understanding Data Flow](#understanding-data-flow)
6. [Working with Strategies](#working-with-strategies)
7. [Risk Management](#risk-management)
8. [Results & Analysis](#results--analysis)
9. [Next Steps](#next-steps)

---

## What is StrategyLab?

StrategyLab is a **production-ready backtesting framework** for algorithmic trading strategies. It enables you to:

- 📊 **Test trading strategies** on historical market data
- 📈 **Analyze performance** with comprehensive metrics
- 🎯 **Optimize parameters** to improve strategy returns
- ⚡ **Validate parity** between backtest and live trading
- 🔄 **Iterate quickly** with modular architecture

### Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Broker Support** | Zerodha, Upstox, Binance (crypto), Flattrade |
| **Strategy Framework** | Modular, reusable strategy components |
| **Risk Management** | Built-in position sizing, stop losses, circuit breakers |
| **Data Management** | Incremental updates, validation, quality checks |
| **Analysis Engine** | Portfolio metrics, visualizations, P&L tracking |
| **Parallel Processing** | Multi-core execution for faster backtests |

### What You Can Build

- **Trend-following systems** (moving averages, breakouts)
- **Mean reversion strategies** (Bollinger Bands, RSI)
- **Multi-timeframe strategies** (combine 1min, 5min, 15min signals)
- **Portfolio strategies** (diversified multi-ticker systems)
- **Options strategies** (via separate live module)

---

## System Architecture

StrategyLab follows a **modular, production-grade architecture** designed for clarity and extensibility.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    UNIFIED RUNNER (CLI)                     │
│  Entry point for all operations (backtest/fetch/analyze)   │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
    ┌────▼─────┐                   ┌────▼─────┐
    │ WORKFLOW │                   │   DATA   │
    │ MANAGER  │                   │   ETL    │
    └────┬─────┘                   └────┬─────┘
         │                               │
    ┌────▼──────────────┐         ┌─────▼────────┐
    │  TASK EXECUTOR    │         │  BROKERS API │
    │ (Parallel/Serial) │         │  PROVIDERS   │
    └────┬──────────────┘         └──────────────┘
         │
    ┌────▼────────────────────────────────┐
    │         STRATEGY ENGINE             │
    │  - Signal Generation                │
    │  - Position Management              │
    │  - Risk Controls                    │
    └────┬────────────────────────────────┘
         │
    ┌────▼────────────────────────────────┐
    │      ANALYSIS & VISUALIZATION       │
    │  - Performance Metrics              │
    │  - P&L Calculation                  │
    │  - Charts & Reports                 │
    └─────────────────────────────────────┘
```

### Directory Structure

```
backtester/
├── src/
│   ├── runners/          # CLI, workflow orchestration
│   ├── strategies/       # Strategy implementations
│   ├── core/
│   │   ├── etl/         # Data fetching, loading
│   │   ├── risk/        # Risk management
│   │   └── output/      # Results generation
│   └── analysis/         # Performance analytics
├── config/
│   ├── templates/        # Risk management templates
│   └── access_tokens/    # Broker API tokens (gitignored)
├── data/
│   └── pools/           # Historical market data
├── outputs/             # Backtest results
├── scripts/             # Helper scripts
├── tests/               # Test suite
└── docs/                # Documentation
```

---

## Core Concepts

### 1. **Data Pools**

**Data pools** are organized collections of historical market data.

**Structure**:
```
data/pools/
└── 2024-01-01_to_2024-03-31/    # Date range
    ├── RELIANCE/
    │   ├── 1minute.parquet       # 1-minute candles
    │   ├── 5minute.parquet       # 5-minute candles
    │   └── 15minute.parquet      # 15-minute candles
    ├── TCS/
    │   └── 1minute.parquet
    └── metadata.json             # Pool metadata
```

**Why pools?**
- **Organized**: Data grouped by date range
- **Reusable**: Same data for multiple backtests
- **Incremental**: Extend pools without re-fetching
- **Validated**: Quality checks on fetch

### 2. **Strategies**

**Strategies** define the trading logic (when to buy/sell).

**Components**:
```python
class MyStrategy(BaseStrategy):
    def initialize(self):
        # Setup indicators, parameters
        pass

    def generate_signals(self, data):
        # Analyze data, return BUY/SELL/HOLD
        pass

    def manage_positions(self, positions):
        # Handle existing positions (exits, stops)
        pass
```

**Built-in Strategies**:
- `open_source_baseline` - Trend + momentum hybrid
- `sma_crossover` - Moving average crossover
- `bollinger_bands` - Volatility channel strategy

**Creating Custom Strategies**: See [STRATEGY_GUIDE.md](STRATEGY_GUIDE.md)

### 3. **Risk Templates**

**Risk templates** control position sizing, stops, and portfolio limits.

**Example (conservative.yaml)**:
```yaml
risk:
  max_position_size: 0.15      # 15% max per trade
  stop_loss_pct: 0.02          # 2% stop loss
  portfolio_risk: 0.30         # 30% total exposure
  max_trades_per_day: 5        # Limit overtrading
```

**Available Templates**:
- **minimal** - Ultra-safe (5% position) for learning
- **conservative** - Low-risk (15% position) for stable returns
- **aggressive** - High-risk (20% position) for growth

**See**: [TEMPLATE_GUIDE.md](TEMPLATE_GUIDE.md)

### 4. **Modes**

StrategyLab operates in different **modes** for different tasks:

| Mode | Purpose | Example |
|------|---------|---------|
| `fetch` | Download market data | `--mode fetch --tickers RELIANCE` |
| `validate` | Check data quality | `--mode validate --dates 2024-01-01` |
| `backtest` | Run full backtest | `--mode backtest --strategies baseline` |
| `analyze` | Generate analytics | `--mode analyze --date-ranges ...` |
| `visualize` | Create charts | `--mode visualize --date-ranges ...` |
| `update` | Extend data pools | `--mode update --pool-path ...` |

---

## Complete Workflow

### Step-by-Step: From Installation to Results

#### **Phase 1: Environment Setup**

```powershell
# 1. Clone repository
git clone https://github.com/Neel-Error404/StrategyLab.git
cd StrategyLab/backtester

# 2. Automated setup
python setup.py

# 3. Configure API credentials
notepad .env  # Add your broker API keys

# 4. Verify installation
python scripts/verify_setup.py
```

**Duration**: ~5 minutes
**Output**: Configured environment, verified installation

---

#### **Phase 2: Data Acquisition**

```powershell
# Check current data
python src/runners/unified_runner.py --check-data RELIANCE

# Fetch historical data
python src/runners/unified_runner.py --mode fetch \
  --tickers RELIANCE TCS INFY \
  --date-ranges 2024-01-01_to_2024-03-31

# Verify data fetched
dir data\pools\
```

**Duration**: 2-5 minutes (depends on date range)
**Output**: Data pools in `data/pools/[date_range]/`

---

#### **Phase 3: Strategy Development & Testing**

```powershell
# Option A: Use built-in strategy
python src/runners/unified_runner.py --mode backtest \
  --strategies open_source_baseline \
  --template conservative \
  --date-ranges 2024-01-01_to_2024-03-31 \
  --tickers RELIANCE

# Option B: Interactive mode
python scripts/quickstart.py
```

**Duration**: 1-10 minutes (depends on data volume)
**Output**: Backtest results in `outputs/[timestamp]/`

---

#### **Phase 4: Analysis & Interpretation**

```powershell
# Generate detailed analytics
python src/runners/unified_runner.py --mode analyze \
  --date-ranges 2024-01-01_to_2024-03-31

# Create visualizations
python src/runners/unified_runner.py --mode visualize \
  --date-ranges 2024-01-01_to_2024-03-31
```

**Output**:
- `outputs/[timestamp]/metrics/` - Performance metrics CSV
- `outputs/[timestamp]/trades/` - All trades CSV
- `outputs/[timestamp]/visualizations/` - Charts (equity curve, drawdown, etc.)

---

#### **Phase 5: Optimization & Iteration**

```powershell
# Test different strategies
python src/runners/unified_runner.py --mode backtest \
  --strategies sma_crossover bollinger_bands \
  --template aggressive \
  --date-ranges 2024-01-01_to_2024-03-31 \
  --tickers RELIANCE TCS

# Test different templates
python src/runners/unified_runner.py --mode backtest \
  --strategies open_source_baseline \
  --template conservative  # vs aggressive
  --date-ranges 2024-01-01_to_2024-03-31 \
  --tickers RELIANCE
```

**Goal**: Find best strategy-template-ticker combination

---

## Understanding Data Flow

### How Data Moves Through the System

```
1. FETCH
   Broker API → Raw OHLCV → Validation → Parquet Files
   └─ data/pools/[date_range]/[TICKER]/[timeframe].parquet

2. LOAD
   Parquet Files → Pandas DataFrame → Technical Indicators
   └─ In-memory data structure

3. BACKTEST
   Data + Strategy Logic → Signals (BUY/SELL) → Simulated Trades
   └─ Trade execution with realistic fills

4. RISK
   Trades + Risk Rules → Position Sizing → Stop Losses → Portfolio Limits
   └─ Risk-adjusted position sizes

5. ANALYZE
   Trades + Prices → Performance Metrics → P&L Calculation
   └─ outputs/[timestamp]/metrics/

6. VISUALIZE
   Metrics + Trades → Charts → Reports
   └─ outputs/[timestamp]/visualizations/
```

### Data Quality Checks

At each stage, StrategyLab validates:

- **Completeness**: No missing bars in trading hours
- **Consistency**: OHLC relationships (High ≥ Open, Close, Low)
- **Timeliness**: Data within expected time bounds
- **Accuracy**: Cross-validation against multiple sources (when available)

---

## Working with Strategies

### Strategy Lifecycle

1. **Create** - Write strategy class inheriting from `BaseStrategy`
2. **Register** - Add to `src/strategies/register_strategies.py`
3. **Test** - Run backtest on sample data
4. **Optimize** - Tune parameters
5. **Validate** - Test on out-of-sample data
6. **Deploy** - (Via separate live module)

### Strategy Example: Simple Moving Average Crossover

```python
from strategies.base_strategy import BaseStrategy
import pandas as pd

class SMAcrossoverStrategy(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.short_window = 10
        self.long_window = 30

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Calculate moving averages
        data['SMA_short'] = data['close'].rolling(self.short_window).mean()
        data['SMA_long'] = data['close'].rolling(self.long_window).mean()

        # Generate signals
        signals = pd.Series('HOLD', index=data.index)
        signals[data['SMA_short'] > data['SMA_long']] = 'BUY'
        signals[data['SMA_short'] < data['SMA_long']] = 'SELL'

        return signals
```

**See [STRATEGY_GUIDE.md](STRATEGY_GUIDE.md) for complete guide**

---

## Risk Management

### Built-in Risk Controls

StrategyLab implements **multiple layers of risk protection**:

#### 1. **Position-Level Risk**

- **Max Position Size**: Limits capital per trade
- **Stop Loss**: Automatic exit on loss threshold
- **Take Profit**: Lock in gains at target

#### 2. **Portfolio-Level Risk**

- **Portfolio Risk Limit**: Max total exposure
- **Max Open Positions**: Limit concurrent trades
- **Sector Limits**: Avoid over-concentration

#### 3. **Operational Risk**

- **Max Trades Per Day**: Prevent overtrading
- **Circuit Breakers**: Halt on excessive losses
- **Drawdown Limits**: Stop trading after threshold

### Risk Template Example

```yaml
risk:
  # Position sizing
  max_position_size: 0.15          # 15% of portfolio per trade
  min_position_size: 0.05          # 5% minimum

  # Stop losses
  stop_loss_pct: 0.02              # 2% stop loss
  trailing_stop_pct: 0.015         # 1.5% trailing stop
  take_profit_pct: 0.04            # 4% take profit

  # Portfolio limits
  portfolio_risk: 0.30             # 30% max total exposure
  max_open_positions: 5            # Max 5 concurrent trades
  max_trades_per_day: 10           # Daily trade limit

  # Drawdown protection
  max_daily_drawdown: 0.05         # 5% max daily loss
  max_total_drawdown: 0.15         # 15% max total loss
```

**Learn more**: [TEMPLATE_GUIDE.md](TEMPLATE_GUIDE.md)

---

## Results & Analysis

### Understanding Backtest Output

After running a backtest, you'll find results in `outputs/[timestamp]/`:

```
outputs/20250107_143022/
├── metadata.json                   # Run configuration
├── metrics/
│   └── performance_metrics.csv     # Key statistics
├── trades/
│   └── trades.csv                  # All trade details
└── visualizations/
    ├── equity_curve.png            # Portfolio value over time
    ├── drawdown.png                # Drawdown chart
    ├── monthly_returns.png         # Monthly returns heatmap
    └── trade_distribution.png      # Win/loss distribution
```

### Key Performance Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| **Total Return** | Overall profit/loss % | > 15% annual |
| **Sharpe Ratio** | Risk-adjusted return | > 1.5 |
| **Max Drawdown** | Largest peak-to-trough decline | < 20% |
| **Win Rate** | % of profitable trades | > 50% |
| **Profit Factor** | Gross profit / Gross loss | > 1.5 |
| **Avg Win / Avg Loss** | Risk-reward ratio | > 1.5 |

**Detailed interpretation**: [OUTPUT_GUIDE.md](OUTPUT_GUIDE.md)

---

## Next Steps

### Learning Path

**Week 1: Basics**
- ✅ Complete QUICKSTART.md
- ✅ Run sample backtest with `open_source_baseline`
- ✅ Understand results in OUTPUT_GUIDE.md

**Week 2: Exploration**
- 📊 Test different strategies (`sma_crossover`, `bollinger_bands`)
- 📈 Try different risk templates (`conservative` vs `aggressive`)
- 🎯 Test on multiple tickers

**Week 3: Customization**
- 🔧 Create your first custom strategy (STRATEGY_GUIDE.md)
- ⚙️ Create custom risk template (TEMPLATE_GUIDE.md)
- 🧪 Run parameter optimization

**Week 4: Advanced**
- 🔬 Multi-timeframe strategies
- 📊 Portfolio-level strategies
- ⚡ Live trading preparation (separate module)

### Recommended Reading Order

1. ✅ **QUICKSTART.md** - Get up and running (15 min)
2. 📖 **GETTING_STARTED.md** (this doc) - Understand architecture
3. 🔧 **SETUP_GUIDE.md** - Detailed installation reference
4. 💰 **BROKER_SETUP.md** - Configure API access
5. 🎯 **STRATEGY_GUIDE.md** - Create custom strategies
6. ⚖️ **TEMPLATE_GUIDE.md** - Manage risk templates
7. 📊 **OUTPUT_GUIDE.md** - Interpret results
8. 🖥️ **CLI_REFERENCE.md** - Master all commands
9. 🐛 **ERROR_REFERENCE.md** - Troubleshoot issues

### Community & Support

- **Documentation**: All guides in `docs/`
- **GitHub Issues**: Report bugs, request features
- **Examples**: `examples/` directory (coming soon)
- **Discussions**: GitHub Discussions for questions

---

## Appendix: Common Workflows

### Workflow 1: Testing a New Strategy Idea

```powershell
# 1. Create strategy file
# src/strategies/my_new_strategy.py

# 2. Register strategy
# Edit src/strategies/register_strategies.py

# 3. Verify registration
python src/runners/unified_runner.py --list-strategies

# 4. Test on sample data
python src/runners/unified_runner.py --mode backtest \
  --strategies my_new_strategy \
  --template conservative \
  --date-ranges 2024-01-01_to_2024-01-31 \
  --tickers RELIANCE

# 5. Analyze results
# Review outputs/[timestamp]/
```

### Workflow 2: Expanding Data Coverage

```powershell
# 1. Check current data
dir data\pools\

# 2. Fetch new tickers
python src/runners/unified_runner.py --mode fetch \
  --tickers HDFCBANK ICICIBANK KOTAKBANK \
  --date-ranges 2024-01-01_to_2024-06-30

# 3. Extend existing pool
python src/runners/unified_runner.py --mode update \
  --pool-path data/pools/2024-01-01_to_2024-06-30 \
  --extend-to 2024-12-31

# 4. Validate data quality
python src/runners/unified_runner.py --mode validate \
  --dates 2024-01-01
```

### Workflow 3: Multi-Strategy Comparison

```powershell
# Run all strategies in parallel
python src/runners/unified_runner.py --mode backtest \
  --strategies open_source_baseline sma_crossover bollinger_bands \
  --template conservative \
  --date-ranges 2024-01-01_to_2024-06-30 \
  --tickers RELIANCE TCS \
  --parallel

# Compare results
python src/analysis/compare_strategies.py outputs/[timestamp]/
```

---

**🎉 You're now ready to master StrategyLab!**

Start with QUICKSTART.md, then experiment with different strategies and templates. Happy backtesting!

---

*Last updated: 2025-01-07*
