# StrategyLab Architecture

Comprehensive system architecture documentation for the StrategyLab Backtesting System

**Version**: 2.0 (Equities Release)
**Last Updated**: October 30, 2025

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architectural Principles](#architectural-principles)
3. [System Components](#system-components)
4. [Data Flow](#data-flow)
5. [Module Details](#module-details)
6. [Design Patterns](#design-patterns)
7. [Configuration Architecture](#configuration-architecture)
8. [Execution Flow](#execution-flow)
9. [Extension Points](#extension-points)
10. [Performance Considerations](#performance-considerations)

---

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│  CLI (unified_runner.py) + Interactive Modes + Templates       │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      ORCHESTRATION LAYER                        │
│  Workflow Manager + Mode Handlers + Task Executor              │
└────────┬────────────────────┬─────────────────┬────────────────┘
         │                    │                 │
┌────────▼────────┐  ┌───────▼────────┐  ┌────▼──────────────┐
│  CORE ENGINE    │  │  ANALYSIS      │  │  VALIDATION       │
│  - Strategies   │  │  - Generic     │  │  - Parity         │
│  - Execution    │  │  - Portfolio   │  │  - Precision      │
│  - Risk Mgmt    │  │  - Optimization│  │  - Bias Detection │
└────────┬────────┘  └───────┬────────┘  └────┬──────────────┘
         │                    │                 │
┌────────▼────────────────────▼─────────────────▼────────────────┐
│                         DATA LAYER                              │
│  ETL + Data Providers + Token Manager + Data Validation        │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      EXTERNAL SYSTEMS                           │
│  Upstox API + Zerodha Kite API + Binance API + File System    │
└─────────────────────────────────────────────────────────────────┘
```

### Component Summary

| Layer | Components | Purpose |
|-------|-----------|---------|
| **UI** | CLI, Interactive Modes | User interaction and command processing |
| **Orchestration** | Workflow Manager, Mode Handlers | Coordinate execution across components |
| **Core Engine** | Strategies, Execution, Risk | Trading logic and backtest execution |
| **Analysis** | Generic, Portfolio, Optimization | Performance analysis and optimization |
| **Validation** | Parity, Precision, Bias | Quality assurance and compliance |
| **Data** | ETL, Providers, Token Manager | Data acquisition and management |
| **External** | Broker APIs, File System | External integrations |

---

## Architectural Principles

### 1. Modular Design
**Principle**: Clean separation of concerns with well-defined interfaces

**Implementation**:
- Each module has a single responsibility
- Interfaces define contracts between modules
- Dependencies flow in one direction (no circular dependencies)

**Example**:
```
Strategy (abstract) → Concrete Strategy Implementation
    ↓                           ↓
Strategy Factory → Strategy Executor → Risk Manager → Output Manager
```

### 2. Pluggable Components
**Principle**: Easy extension without modifying core code

**Implementation**:
- Factory patterns for strategies and data providers
- Registration systems for strategies
- Configuration-driven behavior

**Example**:
```python
# Adding a new strategy
class MyStrategy(StrategyBase):
    def generate_signals(self, data):
        # Implementation
        pass

# Register in src/strategies/register_strategies.py
STRATEGIES['my_strategy'] = MyStrategy
```

### 3. Configuration Over Code
**Principle**: Behavior controlled by configuration, not code changes

**Implementation**:
- YAML-based configuration templates
- Environment variable substitution
- Dual configuration system (infrastructure + trading)

**Example**:
```yaml
# config/templates/custom.yaml
risk_management:
  max_position_size: ${MAX_POSITION_SIZE:15}
  max_drawdown_percent: 20
```

### 4. Fail-Fast Validation
**Principle**: Detect errors early before expensive computation

**Implementation**:
- Data validation gates before execution
- Configuration validation on load
- Strategy validation before backtest
- Parity validation before live trading

**Example**:
```
Load Config → Validate Config → Load Data → Validate Data → Execute Strategy
     ↓             ↓              ↓             ↓               ↓
   PASS         PASS           PASS          PASS           Execute
   FAIL → Exit  FAIL → Exit    FAIL → Exit   FAIL → Exit
```

### 5. Reproducibility First
**Principle**: Same input always produces same output

**Implementation**:
- Deterministic execution (no random without seed)
- Immutable data pools
- Complete environment capture
- Version-locked dependencies

### 6. Observability Built-In
**Principle**: System behavior is always visible

**Implementation**:
- Structured logging throughout
- Complete audit trails
- Output manifests
- Progress indicators

---

## System Components

### 1. Entry Points

#### `src/runners/unified_runner.py`
**Purpose**: Main CLI entry point

**Responsibilities**:
- Parse command-line arguments
- Select execution mode
- Initialize orchestrator
- Handle interrupts (Ctrl+C)

**Key Functions**:
```python
main() -> int:
    - Parse CLI args
    - Load configuration
    - Initialize workflow
    - Execute mode handler
    - Return exit code
```

**Modes Supported**:
- `backtest`: Full workflow
- `analyze`: Analysis only
- `visualize`: Visualization only
- `validate`: Data validation
- `fetch`: Data retrieval
- `update`: Incremental update
- `replay`: Manifest replay
- `optimize`: Parameter search (WIP)

### 2. Orchestration Layer

#### `src/runners/workflow/orchestrator.py`
**Purpose**: High-level workflow coordination

**Responsibilities**:
- Initialize workflow components
- Coordinate mode execution
- Handle errors and cleanup

#### `src/runners/workflow/mode_handlers.py`
**Purpose**: Mode-specific execution logic

**Responsibilities**:
- Implement mode-specific workflows
- Coordinate between components
- Manage mode-specific configuration

**Mode Handlers**:
- `BacktestModeHandler`: Full backtest workflow
- `AnalyzeModeHandler`: Analysis-only workflow
- `VisualizeModeHandler`: Visualization-only workflow
- `ValidateModeHandler`: Validation workflow
- `FetchModeHandler`: Data fetching workflow
- `UpdateModeHandler`: Incremental update workflow

#### `src/runners/task_executor.py`
**Purpose**: Parallel/sequential task execution

**Responsibilities**:
- Execute tasks in parallel or sequential
- Handle multiprocessing on Windows
- Manage worker pools
- Aggregate results

### 3. Core Engine

#### Strategy System

**`src/strategies/strategy_base.py`**
- Abstract base class for all strategies
- Defines strategy interface
- Provides common utilities

**`src/strategies/strategy_factory.py`**
- Factory for creating strategy instances
- Strategy validation
- Timeframe validation

**`src/strategies/register_strategies.py`**
- Strategy registration system
- Maps strategy names to classes

**Built-in Strategies**:
- `strategy_mse.py`: Mean Squared Error strategy
- `strategy_sma.py`: Simple Moving Average
- `strategy_sma_crossover.py`: SMA crossover
- `strategy_bollinger_bands.py`: Bollinger Bands

#### Execution Engine

**`src/runners/workflow/execution_engine.py`**
- Core backtest execution logic
- Bar-by-bar simulation
- Signal generation and execution
- Position management

**Key Functions**:
```python
execute_backtest(strategy, data, config):
    - Initialize positions
    - For each bar:
        - Generate signals
        - Apply risk rules
        - Execute trades
        - Update positions
    - Return results
```

#### Risk Management

**`src/core/risk/risk_manager.py`**
- Position sizing
- Drawdown monitoring
- Risk-adjusted trade approval
- Portfolio-level risk controls

**Risk Checks**:
- Max position size (% of capital)
- Max portfolio exposure
- Drawdown limits
- Per-ticker exposure limits

#### Transaction Costs

**`src/core/costs/transaction_models.py`**
- Slippage modeling
- Brokerage fees
- Exchange fees
- Transaction cost analysis

### 4. Analysis Framework

#### Generic Analysis

**Location**: `analysis/generic/scripts/`

**Scripts**:
1. `01_basic_eda.py`: Win rate, profit factor, Sharpe
2. `02_trade_type_analysis.py`: Long vs short performance
3. `03_cascade_analysis.py`: Trade pattern detection
4. `04_stop_loss_simulation.py`: Optimal SL threshold
5. `05_ticker_ranking.py`: Quality scoring
6. `06_risk_adjusted_patterns.py`: Risk-normalized metrics
7. `07_top50_vs_overall.py`: Selection validation
8. `08_top50_pattern_breakdown.py`: Winner profiling
9. `09_validation_check.py`: Data integrity audit

#### Portfolio Construction

**Location**: `analysis/portfolio_construction/scripts/`

**Scripts**:
1. `00_ticker_ranking.py`: Comprehensive ranking
2. `01_anti_cascade_filter.py`: Bias removal
3. `02_sector_classification.py`: Diversification
4. `03_combination_generator.py`: Optimization space
5. `04_portfolio_optimizer.py`: Equal-weight evaluation
6. `05_pypfopt_weights.py`: Markowitz optimization
7. `06_equity_curves.py`: Visual validation

#### Analysis Modules

**Location**: `analysis/generic/modules/`

**Modules**:
- `config_loader.py`: Configuration utilities
- `data_loader.py`: Data loading utilities
- `metrics_calculator.py`: Performance metrics
- `visualizer.py`: Chart generation

#### Analysis Orchestration System

**`analysis/run.py`** (445 lines)
**Purpose**: Main orchestrator for all analysis workflows

**Architecture**:
```python
Orchestrator (run.py)
    │
    ├── Config Loader (YAML-based configuration)
    │   ├── Load run metadata (run_id, strategy, date_range)
    │   ├── Load data source paths
    │   └── Load module registry (generic, portfolio, optimization)
    │
    ├── Trade Merger (Multi-ticker aggregation)
    │   ├── Auto-detect trade source (strategy_trades vs risk_approved)
    │   ├── Merge per-ticker CSVs into unified dataset
    │   └── Output: all_trades_merged.csv
    │
    ├── Module Executor (22+ registered modules)
    │   ├── Generic Analysis (9 modules)
    │   ├── Portfolio Construction (7 modules)
    │   └── Strategy Optimization (6+ modules)
    │
    └── Output Router
        ├── CSV artifacts → analysis/output/{strategy}/{run_id}/
        ├── JSON reports → analysis/reports/{strategy}/{run_id}/
        └── Run logs → analysis/run_logs/{timestamp}.log
```

**YAML Configuration System**:
```yaml
# analysis/configs/example_config.yaml
run:
  run_id: "20251024_095707"
  strategy: "mse"
  trade_source: "strategy_trades"

data_sources:
  strategy_trades_dir: "outputs/{run_id}/mse/{date_range}/data/strategy_trades"
  base_data_dir: "outputs/{run_id}/mse/{date_range}/data/base_data"

analysis:
  generic:
    enabled: true
    modules:
      basic_eda:
        enabled: true
        config:
          include_ticker_breakdown: true
  portfolio:
    enabled: true
    modules:
      portfolio_optimizer:
        enabled: true
        config:
          top_n: 50
```

**Key Features**:
- **Module Registry**: Dynamic module discovery and execution
- **Trade Merging**: Automatic multi-ticker CSV aggregation
- **Config Validation**: YAML schema validation on load
- **Output Routing**: Organized directory structure for artifacts
- **Run Logging**: Complete audit trail with timestamps
- **Documentation**: Comprehensive guide in `analysis/README.md`

**Strategy Optimization Suite**:
**Location**: `analysis/strategy_optimization/scripts/`

**Scripts** (6+):
1. Exit threshold optimization (50-95% MACD range)
2. Entry signal analysis (MACD strength filters)
3. Combined optimization (joint entry + exit)
4. Parameter grid search
5. Walk-forward analysis
6. Out-of-sample validation

### 5. Validation Framework

#### Config Parity Validator

**`src/core/validation/config_parity_validator.py`**

**Purpose**: Ensure live and backtest configs match

**Validations**:
- Critical parameter comparison
- Environment variable checks
- Configuration drift detection
- Pre-trade validation

**Example Usage**:
```python
validator = ConfigParityValidator(backtest_config, live_config)
parity_report = validator.validate()
if not parity_report.is_valid:
    raise ConfigParityError(parity_report.issues)
```

#### Signal Parity Validator

**`src/core/validation/signal_parity_validator.py`**

**Purpose**: Compare live and backtest signals

**Validations**:
- Signal-by-signal comparison
- Timestamp synchronization
- Divergence detection
- Parity score calculation

**Example Usage**:
```python
validator = SignalParityValidator()
parity_report = validator.compare(backtest_signals, live_signals)
print(f"Parity Score: {parity_report.parity_score:.2%}")
```

#### Precision Validator

**`src/core/validation/precision_validator.py`**

**Purpose**: Enforce exchange precision rules

**Validations**:
- Price precision (tick size)
- Quantity precision (lot size)
- PnL rounding
- Order validation

**Example Usage**:
```python
validator = PrecisionValidator(exchange='NSE')
is_valid = validator.validate_order(price=1234.567, quantity=10)
```

#### Bias Detector

**`src/core/validation/bias_detector.py`**

**Purpose**: Detect data biases

**Checks**:
- Look-ahead bias (using future data)
- Survivorship bias (missing delisted tickers)
- Data snooping (overfitting)

### 6. Data Layer

#### ETL System

**`src/core/etl/data_fetcher.py`**
- Main data fetching orchestrator
- Chunked fetching for large ranges
- Retry logic with exponential backoff
- Progress tracking

**`src/core/etl/loader.py`**
- Data loading from disk
- Multi-timeframe loading
- Data merging and alignment

**`src/core/etl/data_integrity.py`**
- Data quality validation
- Gap detection
- Outlier detection

#### Data Provider System

**`src/core/etl/data_provider/provider_factory.py`**
- Provider factory pattern
- Auto-discovery of available providers
- Provider validation

**`src/core/etl/data_provider/base_provider.py`**
- Abstract provider interface
- Common provider utilities

**Concrete Providers**:
- `upstox_provider.py`: Upstox API integration
- `zerodha_provider.py`: Zerodha Kite API integration
- `binance_provider.py`: Binance API integration

#### Token Manager

**`src/core/etl/token_manager.py`**
- Centralized token management
- Auto-refresh for expired tokens
- Multi-broker token handling
- Secure token storage

**Token Flow**:
```
Request Data → Check Token → Valid? → Fetch Data
                     ↓          ↓
                  Invalid   Expired → Refresh → Fetch Data
```

### 7. Output System

#### Output Orchestrator

**`src/core/output/enhanced_output_orchestrator.py`**
- Comprehensive output generation
- Multi-file output coordination
- Output manifest generation

#### Output Manager

**`src/core/output/output_manager.py`**
- Centralized output coordination
- File organization
- Output validation

#### Three-File System

**`src/core/output/three_file_system.py`**
- Compact summary format
- Three files: config, metrics, risk
- Quick results review

**Output Structure**:
```
outputs/
├── {timestamp}/
│   ├── {strategy}/
│   │   ├── {ticker}_Base_{daterange}.csv         # OHLCV + signals
│   │   ├── {ticker}_StrategyTrades_{daterange}.csv
│   │   ├── {ticker}_RiskApprovedTrades_{daterange}.csv
│   │   ├── {ticker}_Analysis_{daterange}.json
│   │   ├── performance_summary.png
│   │   ├── trade_distribution.png
│   │   └── trade_timeline.png
│   ├── portfolio/
│   │   ├── portfolio_Analysis_{daterange}.json
│   │   ├── portfolio_master_dashboard.png
│   │   └── ...
│   └── output_manifest.json
```

### 8. Visualization System

**`src/core/analysis/visualization.py`**
- Individual ticker visualizations
- Performance summary charts
- Trade distribution histograms
- Trade timeline charts

**`src/core/analysis/portfolio_visualization.py`**
- Portfolio-level visualizations
- Multi-ticker comparison
- Risk dashboards
- Signal analysis charts

**Chart Types**:
- Performance Summary: Equity curve, metrics, trade stats
- Trade Distribution: Win/loss histogram
- Trade Timeline: Chronological trade view
- Educational Insights: Risk management, psychology
- Portfolio Dashboard: Multi-ticker overview
- Risk Dashboard: Risk metrics visualization
- Signal Analysis: Entry/exit signal quality

---

## Data Flow

### 1. Backtest Mode Data Flow

```
User Command
    ↓
CLI Parser → Validate Args
    ↓
Load Config (YAML + env vars)
    ↓
Initialize Orchestrator
    ↓
Fetch/Load Data → Validate Data Quality
    ↓
Initialize Strategy → Validate Strategy
    ↓
Execute Backtest (bar-by-bar)
    ↓
Generate Trades
    ↓
Risk Manager → Approve/Reject Trades
    ↓
Record Approved Trades
    ↓
Calculate Metrics
    ↓
Generate Analysis
    ↓
Create Visualizations
    ↓
Save Outputs (CSV, JSON, PNG)
    ↓
Return Results
```

### 2. Fetch Mode Data Flow

```
User Command
    ↓
CLI Parser → Parse Date Range, Tickers
    ↓
Initialize Token Manager
    ↓
Discover Available Providers
    ↓
Select Provider (or prompt user)
    ↓
Validate Date Range
    ↓
Chunk Date Range (smart chunking)
    ↓
For each chunk:
    ↓
    Fetch Data from Provider API
    ↓
    Retry on Failure (exponential backoff)
    ↓
    Validate Data Quality
    ↓
    Save to Data Pool
    ↓
Merge Chunks
    ↓
Final Validation
    ↓
Return Success/Failure
```

### 3. Analysis Mode Data Flow

```
Load Backtest Results (trades CSV)
    ↓
Load Configuration
    ↓
Initialize Analysis Modules
    ↓
Run Generic Analysis:
    ↓
    - Basic EDA
    ↓
    - Trade Type Analysis
    ↓
    - Cascade Analysis
    ↓
    - Risk-Adjusted Patterns
    ↓
    - Ticker Ranking
    ↓
Run Portfolio Analysis:
    ↓
    - Sector Classification
    ↓
    - Correlation Matrix
    ↓
    - Portfolio Optimization
    ↓
    - Equity Curves
    ↓
Generate Reports (JSON, CSV)
    ↓
Create Visualizations (PNG)
    ↓
Save Analysis Results
    ↓
Return Analysis Summary
```

---

## Design Patterns

### 1. Factory Pattern

**Used In**: Strategy creation, data provider selection

**Implementation**:
```python
class StrategyFactory:
    @staticmethod
    def create(strategy_name, config):
        strategy_class = STRATEGIES.get(strategy_name)
        if not strategy_class:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        return strategy_class(config)
```

**Benefits**:
- Decouples strategy creation from usage
- Easy to add new strategies
- Centralized strategy validation

### 2. Template Method Pattern

**Used In**: Strategy base class

**Implementation**:
```python
class StrategyBase(ABC):
    def execute(self, data):
        self.validate_data(data)
        self.prepare_indicators(data)
        signals = self.generate_signals(data)
        return self.post_process(signals)

    @abstractmethod
    def generate_signals(self, data):
        pass  # Subclasses implement
```

**Benefits**:
- Common workflow in base class
- Strategy-specific logic in subclasses
- Easy to enforce common behavior

### 3. Dependency Injection

**Used In**: Component initialization

**Implementation**:
```python
class BacktestEngine:
    def __init__(self, strategy, risk_manager, output_manager):
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.output_manager = output_manager
```

**Benefits**:
- Loose coupling
- Easy testing (mock dependencies)
- Flexible configuration

### 4. Observer Pattern

**Used In**: Progress tracking, logging

**Implementation**:
```python
class ProgressTracker:
    def __init__(self):
        self.observers = []

    def attach(self, observer):
        self.observers.append(observer)

    def notify(self, progress):
        for observer in self.observers:
            observer.update(progress)
```

**Benefits**:
- Decoupled progress reporting
- Multiple observers (console, file, etc.)
- Easy to add new observers

### 5. Strategy Pattern

**Used In**: Risk management, output generation

**Implementation**:
```python
class RiskStrategy(ABC):
    @abstractmethod
    def evaluate(self, trade):
        pass

class PositionSizeRiskStrategy(RiskStrategy):
    def evaluate(self, trade):
        # Check position size
        pass
```

**Benefits**:
- Flexible risk rules
- Easy to combine strategies
- Runtime strategy selection

---

## Configuration Architecture

### Dual Configuration System

#### Infrastructure Config (`config/config.py`)
**Purpose**: System-level configuration

**Contents**:
- Broker API credentials
- Data provider settings
- Token management
- Logging configuration
- Directory structure

**Example**:
```python
BROKERS = {
    'upstox': {
        'client_id': os.getenv('UPSTOX_CLIENT_ID'),
        'client_secret': os.getenv('UPSTOX_CLIENT_SECRET'),
    },
    'zerodha': {
        'api_key': os.getenv('ZERODHA_API_KEY'),
    }
}
```

#### Trading Config (`config/unified_config.py`)
**Purpose**: Trading-specific configuration

**Contents**:
- Strategy parameters
- Risk management settings
- Position sizing
- Transaction costs
- Validation rules

**Example**:
```python
RISK_MANAGEMENT = {
    'max_position_size_percent': 15,
    'max_drawdown_percent': 20,
    'max_portfolio_exposure': 100,
}
```

### Template System

**Location**: `config/templates/`

**Templates**:
- `minimal.yaml`: Ultra-safe (5% position)
- `conservative.yaml`: Low-risk (15% position)
- `aggressive.yaml`: High-risk (20% position)
- `portfolio_diversified.yaml`: Multi-ticker
- `debug.yaml`: Development/debugging

**Template Structure**:
```yaml
strategy:
  name: mse
  params:
    macd_fast: 12
    macd_slow: 26

risk_management:
  max_position_size: 15
  max_drawdown_percent: 20

output:
  save_trades: true
  save_visualizations: true
```

### Config Loader

**`config/config_loader.py`**

**Features**:
- YAML parsing
- Environment variable substitution
- Schema validation
- Default values

**Usage**:
```python
from config.config_loader import ConfigLoader

config = ConfigLoader.load_yaml('config/templates/conservative.yaml')
# ${VAR} replaced with env var value
# ${VAR:default} replaced with default if VAR not set
```

---

## Execution Flow

### Backtest Execution Flow

```
1. INITIALIZATION
   ├─ Parse CLI args
   ├─ Load configuration (template + overrides)
   ├─ Validate configuration
   └─ Initialize components

2. DATA ACQUISITION
   ├─ Load data from pool (or fetch if missing)
   ├─ Validate data quality
   ├─ Merge multi-timeframe data
   └─ Apply warmup period

3. STRATEGY INITIALIZATION
   ├─ Create strategy instance
   ├─ Validate strategy config
   ├─ Prepare indicators
   └─ Set initial state

4. BAR-BY-BAR EXECUTION
   For each bar in data:
   ├─ Update strategy state
   ├─ Generate signals (long/short/flat)
   ├─ Apply two-bar rule (signal → pending → execute)
   ├─ Check EOD (flatten positions at 15:15)
   └─ Record signals

5. RISK MANAGEMENT
   For each signal:
   ├─ Check position size limits
   ├─ Check portfolio exposure
   ├─ Check drawdown limits
   ├─ Approve or reject trade
   └─ Record decision

6. TRADE EXECUTION
   For approved trades:
   ├─ Execute at next bar open (realistic)
   ├─ Apply slippage and fees
   ├─ Update positions
   └─ Record trade

7. METRICS CALCULATION
   ├─ Calculate returns
   ├─ Calculate Sharpe ratio
   ├─ Calculate profit factor
   ├─ Calculate max drawdown
   └─ Calculate all metrics

8. OUTPUT GENERATION
   ├─ Save trade logs (CSV, JSON)
   ├─ Save analysis (JSON)
   ├─ Generate visualizations (PNG)
   ├─ Save config snapshot
   └─ Create manifest

9. CLEANUP
   └─ Close files, return results
```

---

## Extension Points

### 1. Adding a New Strategy

**Steps**:
1. Create strategy class in `src/strategies/`
2. Inherit from `StrategyBase`
3. Implement `generate_signals()` method
4. Register in `src/strategies/register_strategies.py`
5. Add tests (optional but recommended)

**Example**:
```python
# src/strategies/strategy_rsi.py
from src.strategies.strategy_base import StrategyBase

class RSIStrategy(StrategyBase):
    def __init__(self, config):
        super().__init__(config)
        self.rsi_period = config.get('rsi_period', 14)
        self.oversold = config.get('oversold', 30)
        self.overbought = config.get('overbought', 70)

    def generate_signals(self, data):
        # Calculate RSI
        rsi = self.calculate_rsi(data, self.rsi_period)

        # Generate signals
        signals = pd.Series(0, index=data.index)
        signals[rsi < self.oversold] = 1  # Buy
        signals[rsi > self.overbought] = -1  # Sell

        return signals

# src/strategies/register_strategies.py
from src.strategies.strategy_rsi import RSIStrategy

STRATEGIES = {
    'mse': MSEStrategy,
    'sma': SMAStrategy,
    'rsi': RSIStrategy,  # Add new strategy
}
```

### 2. Adding a New Data Provider

**Steps**:
1. Create provider class in `src/core/etl/data_provider/`
2. Inherit from `BaseProvider`
3. Implement `fetch_data()` method
4. Register in `provider_factory.py`
5. Add credentials to `config/config.py`

**Example**:
```python
# src/core/etl/data_provider/fyers_provider.py
from src.core.etl.data_provider.base_provider import BaseProvider

class FyersProvider(BaseProvider):
    def __init__(self, credentials):
        self.app_id = credentials['app_id']
        self.access_token = credentials['access_token']

    def fetch_data(self, symbol, start_date, end_date, timeframe):
        # Implement Fyers API call
        response = requests.get(...)
        return self.parse_response(response)

# src/core/etl/data_provider/provider_factory.py
from src.core.etl.data_provider.fyers_provider import FyersProvider

PROVIDERS = {
    'upstox': UpstoxProvider,
    'zerodha': ZerodhaProvider,
    'fyers': FyersProvider,  # Add new provider
}
```

### 3. Adding a New Analysis Module

**Steps**:
1. Create script in `analysis/generic/scripts/` or `analysis/portfolio_construction/scripts/`
2. Follow module template
3. Use `config_loader` and `data_loader` utilities
4. Add to documentation

**Example**:
```python
# analysis/generic/scripts/10_my_analysis.py
from analysis.generic.modules.config_loader import load_config
from analysis.generic.modules.data_loader import load_trades

def analyze(trades_df, config):
    # Your analysis logic
    results = {
        'metric1': ...,
        'metric2': ...,
    }
    return results

if __name__ == '__main__':
    config = load_config()
    trades = load_trades(config['trades_path'])
    results = analyze(trades, config)
    print(results)
```

### 4. Adding a New Validation Check

**Steps**:
1. Create validator class in `src/core/validation/`
2. Implement validation logic
3. Add to validation pipeline
4. Add tests

**Example**:
```python
# src/core/validation/order_validator.py
class OrderValidator:
    def validate(self, order):
        issues = []

        # Check order size
        if order.quantity <= 0:
            issues.append("Invalid quantity")

        # Check price
        if order.price <= 0:
            issues.append("Invalid price")

        return ValidationResult(is_valid=len(issues)==0, issues=issues)
```

---

## Performance Considerations

### 1. Data Loading Optimization

**Strategy**: Lazy loading with caching

**Implementation**:
```python
class DataLoader:
    def __init__(self):
        self._cache = {}

    def load(self, ticker, date_range):
        cache_key = f"{ticker}_{date_range}"
        if cache_key not in self._cache:
            self._cache[cache_key] = self._load_from_disk(ticker, date_range)
        return self._cache[cache_key]
```

**Benefits**:
- Avoid repeated disk reads
- Memory efficient (load on demand)
- Fast repeated access

### 2. Parallel Processing

**Strategy**: Multi-core execution for independent tasks

**Implementation**:
```python
from concurrent.futures import ProcessPoolExecutor

def backtest_ticker(ticker, config):
    # Backtest single ticker
    pass

def backtest_portfolio(tickers, config):
    with ProcessPoolExecutor(max_workers=config['max_workers']) as executor:
        results = list(executor.map(backtest_ticker, tickers))
    return results
```

**Speedup**: Linear with number of cores (for independent tickers)

### 3. Vectorized Operations

**Strategy**: Use pandas vectorized operations instead of loops

**Bad**:
```python
for i in range(len(df)):
    df.loc[i, 'rsi'] = calculate_rsi(df.loc[:i])
```

**Good**:
```python
df['rsi'] = df['close'].rolling(14).apply(rsi_func)
```

**Speedup**: 10-100x faster

### 4. Memory Management

**Strategy**: Process data in chunks for large datasets

**Implementation**:
```python
def process_large_dataset(file_path):
    chunk_size = 10000
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        process_chunk(chunk)
```

**Benefits**:
- Handles datasets larger than RAM
- Predictable memory usage

### 5. Incremental Updates

**Strategy**: Only fetch/process new data

**Implementation**:
```python
def update_pool(pool_path, extend_to_date):
    existing_data = load_pool(pool_path)
    last_date = existing_data['date'].max()

    if extend_to_date > last_date:
        new_data = fetch_data(last_date + 1, extend_to_date)
        updated_data = pd.concat([existing_data, new_data])
        save_pool(pool_path, updated_data)
```

**Speedup**: Proportional to data size reduction

---

## Security Considerations

### 1. Credential Management

**Best Practices**:
- Store credentials in environment variables or `.env` file
- Never commit credentials to git
- Use `.gitignore` for sensitive files
- Rotate tokens regularly

**Implementation**:
```python
# .env file
UPSTOX_CLIENT_ID=your_id_here
UPSTOX_CLIENT_SECRET=your_secret_here

# Python code
from dotenv import load_dotenv
load_dotenv()

client_id = os.getenv('UPSTOX_CLIENT_ID')
```

### 2. Token Security

**Best Practices**:
- Store tokens in secure directory (`config/access_tokens/`)
- Set file permissions (read-only for owner)
- Expire tokens after reasonable time
- Refresh tokens automatically

### 3. Data Validation

**Best Practices**:
- Validate all external data
- Sanitize user inputs
- Check data types and ranges
- Reject malformed data

---

## Testing Strategy

### 1. Unit Tests

**Scope**: Individual functions and classes

**Location**: `tests/unit/`

**Example**:
```python
def test_sharpe_ratio_calculation():
    returns = pd.Series([0.01, 0.02, -0.01, 0.03])
    sharpe = calculate_sharpe_ratio(returns)
    assert sharpe > 0
```

### 2. Integration Tests

**Scope**: Component interactions

**Location**: `tests/integration/`

**Example**:
```python
def test_backtest_pipeline():
    config = load_config('test_config.yaml')
    data = load_test_data()
    strategy = create_strategy(config)
    results = execute_backtest(strategy, data)
    assert len(results.trades) > 0
```

### 3. Validation Tests

**Scope**: Parity and precision validation

**Location**: `tests/`

**Files**:
- `test_backtest_live_parity.py`
- `test_precision_validation.py`
- `indian_equities_master/test_pipeline.py`

### 4. End-to-End Tests

**Scope**: Complete workflows

**Example**:
```python
def test_full_backtest_workflow():
    result = run_command([
        'python', 'src/runners/unified_runner.py',
        '--mode', 'backtest',
        '--template', 'minimal',
        '--dates', '2024-01-01',
        '--tickers', 'RELIANCE'
    ])
    assert result.returncode == 0
    assert os.path.exists('outputs/...')
```

---

## Deployment Architecture

### Development Environment
```
Windows PowerShell + Python 3.10+ + VSCode
├─ Virtual environment (.venv)
├─ Local data pools
├─ Local broker credentials
└─ Development configs
```

### Testing Environment
```
CI/CD (GitHub Actions)
├─ Automated testing
├─ Linting and formatting
├─ Coverage reports
└─ Integration tests
```

### Production Environment (Future)
```
Cloud Deployment (AWS/Azure/GCP)
├─ Scalable compute (for parallel backtests)
├─ Data storage (S3/Blob/GCS)
├─ Secret management (Secrets Manager)
├─ Monitoring and alerts
└─ Live trading infrastructure
```

---

## Future Architecture Plans

### Phase 2: Strategy Optimization
- Parameter grid search engine
- Walk-forward optimization
- Out-of-sample validation
- Multi-objective optimization

### Phase 3: Live Trading
- WebSocket integration for real-time data
- Order management system
- Position reconciliation
- Live risk monitoring
- Paper trading mode

### Phase 4: Web Interface
- React/Vue.js frontend
- REST API backend
- Real-time dashboard
- Strategy marketplace
- User management

### Phase 5: Machine Learning
- ML-based signal generation
- Reinforcement learning for optimization
- Anomaly detection
- Predictive analytics

---

**Document Maintained By**: StrategyLab Architecture Team
**Last Updated**: October 30, 2025
**Next Review**: After Phase 2 completion
