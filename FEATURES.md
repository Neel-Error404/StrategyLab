# StrategyLab Features

Comprehensive feature list for the StrategyLab Backtesting System

**Current Version**: V2 (Equities Release - October 2025)

---

## Core Backtesting Engine

### Execution Modes
- **Backtest Mode**: Full workflow with execution, analysis, and visualization
- **Analysis Mode**: Backtest + analysis only (no visualization)
- **Visualize Mode**: Backtest + visualization only (no deep analysis)
- **Validate Mode**: Data quality and bias validation
- **Fetch Mode**: Market data retrieval from brokers
- **Update Mode**: Incremental data pool updates
- **Replay Mode**: Stored manifest replay engine
- **Optimize Mode**: Strategy parameter search (WIP)

### Strategy System
- **Pluggable Architecture**: Easy strategy registration and discovery
- **Multi-Timeframe Support**: Strategies can use multiple timeframes (e.g., 5m + 15m)
- **Look-Ahead Protection**: Built-in bias prevention with `.shift(1)` enforcement
- **Warmup Periods**: Configurable warmup for indicator stability
- **Strategy Factory**: Automated strategy instantiation with validation

#### Built-in Strategies
- **MSE (Mean Squared Error)**: Production-grade multi-timeframe strategy
  - 4-indicator entry system (5m + 15m MACD + EMA alignment)
  - 80% peak/valley exits based on MACD extremes
  - EOD risk management (automatic closure at 15:15 IST)
  - Single position enforcement per direction per day
  - 525-minute warmup for MACD stability
- **SMA (Simple Moving Average)**: Classic trend-following strategy
- **SMA Crossover**: Dual moving average crossover system
- **Bollinger Bands**: Mean reversion strategy
- **Custom Strategies**: Template-based development framework

### Risk Management
- **Position Sizing**: Configurable limits (5%-20% of capital)
- **Drawdown Protection**: Portfolio-level risk monitoring
- **Trade Validation**: Real-time risk assessment per trade
- **Stop Loss Simulation**: Optimize SL thresholds via analysis
- **Circuit Breakers**: Configurable max positions and exposure caps
- **EOD Flattening**: Automatic position closure at market close

---

## Data Management

### Data Providers
- **Upstox API**: Real-time and historical Indian equities data
- **Zerodha Kite API**: NSE/BSE market data
- **Binance API**: Cryptocurrency historical data (no auth required)
- **Provider Factory**: Auto-detection of available authenticated providers
- **Fallback Mechanism**: Automatic provider switching on failures

### Data Fetching
- **Interactive Mode**: User-guided data fetching with prompts
- **Programmatic Mode**: CLI-based data retrieval with date ranges
- **Chunked Fetching**: Handles large date ranges with API rate limits
- **Smart Chunk Sizing**: Automatic chunk sizing based on timeframe (1-day chunks for minute data)
- **Retry Logic**: Exponential backoff for failed requests
- **Progress Tracking**: Real-time progress for long-running operations

### Data Organization
- **Date-Range Pools**: Organized by date ranges (YYYY-MM-DD_to_YYYY-MM-DD)
- **Timeframe Folders**: Separate folders for 1minute, 5minute, 1day, etc.
- **Parquet Support**: Efficient columnar storage for large datasets
- **CSV Support**: Legacy format for compatibility
- **Ticker-First Structure**: Optimized for ticker-based analysis

### Data Quality
- **Price Consistency Validation**: Detects OHLC inconsistencies
- **Volume Outlier Detection**: Flags unusual volume patterns
- **Gap Detection**: Identifies missing data periods
- **Timestamp Validation**: Ensures monotonic timestamps
- **Duplicate Detection**: Removes duplicate records
- **Null Value Checks**: Zero-tolerance for missing OHLC values
- **Smart Thresholds**: Statistical thresholds (0.1% of data) for reasonable validation

### Incremental Updates
- **Pool Inspector**: Analyze existing data pools
- **Gap Calculator**: Identify missing data periods
- **Incremental Fetch**: Extend pools without re-downloading
- **Dry-Run Mode**: Preview updates before execution
- **Validation-Only Mode**: Validate without fetching
- **Backup System**: Automatic backups before updates

---

## Validation Framework (V2)

### Parity Validation
- **Config Parity Validator**: Ensures live and backtest configs match
  - Critical parameter validation
  - Environment variable checks
  - Configuration drift detection
- **Signal Parity Validator**: Compares live vs backtest signal streams
  - Signal-by-signal comparison
  - Divergence detection and reporting
  - Timestamp synchronization validation
- **Precision Validator**: Enforces price/quantity precision rules
  - Exchange-specific precision rules
  - PnL rounding validation
  - Order quantity precision checks

### Bias Detection
- **Look-Ahead Bias Detection**: Identifies future data leakage
- **Survivorship Bias Checks**: Validates historical data completeness
- **Data Snooping Detection**: Monitors for overfitting patterns

### Test Suites
- **test_backtest_live_parity.py**: Validates backtest-live signal alignment
- **test_precision_validation.py**: Tests price/quantity precision enforcement
- **indian_equities_master/test_pipeline.py**: End-to-end pipeline testing
- **59 Passing Tests**: Comprehensive test coverage

---

## Analysis Framework (V2)

### Generic Analysis Suite (Strategy-Agnostic)
1. **Basic EDA**: Foundation statistics (win rate, profit factor, Sharpe per ticker)
2. **Trade Type Analysis**: Directional bias detection (long vs short performance)
3. **Cascade Analysis**: Behavioral pattern detection (cascade vs first-trade metrics)
4. **Stop Loss Simulation**: Risk management optimization (optimal SL threshold)
5. **Ticker Ranking**: Quality scoring system (top 50 tickers list)
6. **Risk-Adjusted Patterns**: Risk-normalized performance (pattern Sharpe ratios)
7. **Top50 vs Overall**: Selection validation (statistical significance test)
8. **Top50 Pattern Breakdown**: Winner profiling (top 50 pattern prevalence)
9. **Validation Check**: Data integrity audit (data quality score)

### Portfolio Construction Suite
1. **Ticker Ranking** (00): Comprehensive ranking (ALL, CASCADING, ANTI-CASCADING)
2. **Anti-Cascade Filter** (01): Behavioral bias removal
3. **Sector Classification** (02): Diversification framework with correlation matrix
4. **Combination Generator** (03): Constrained optimization space generation
5. **Portfolio Optimizer** (04): Equal-weight evaluation ranked by Sharpe
6. **PyPortfolioOpt Weights** (05): Markowitz optimization for optimal weights
7. **Equity Curves** (06): Visual validation with drawdown charts

### Strategy Optimization (Phase 2)
- **Exit Threshold Optimization**: Test 50-95% MACD threshold range
- **Entry Signal Analysis**: MACD strength and EMA spread filters
- **Combined Optimization**: Joint entry + exit optimization
- **Parameter Grid Search**: Systematic parameter space exploration
- **Walk-Forward Analysis**: Out-of-sample validation

### Statistical Methods
- **Sharpe Ratio**: Risk-adjusted return calculation
- **Profit Factor**: Gross profit / gross loss analysis
- **Win Rate Analysis**: Success rate statistics
- **Max Drawdown**: Peak-to-trough decline measurement
- **Correlation Analysis**: Co-movement between tickers
- **Z-Score Normalization**: Standard deviation-based ranking
- **ANOVA Testing**: Statistical significance validation
- **Herfindahl Index**: Portfolio concentration metrics

---

## Configuration System

### Template System
- **minimal.yaml**: Ultra-safe learning (5% max position)
- **conservative.yaml**: Low-risk trading (15% max position)
- **aggressive.yaml**: High-risk trading (20% max position)
- **portfolio_diversified.yaml**: Multi-ticker portfolio strategy
- **debug.yaml**: Development and debugging configuration
- **Custom Templates**: User-defined YAML configurations

### Configuration Features
- **Environment Variable Substitution**: `${VAR}` or `${VAR:default}` syntax
- **YAML Validation**: Schema validation on load
- **Config Loader**: Centralized configuration management
- **Dual Config System**:
  - **config/config.py**: Infrastructure (broker credentials, data providers)
  - **config/unified_config.py**: Trading (strategy params, risk management)
- **Template Auto-Discovery**: Automatic template detection and loading

---

## Broker Integration

### Authentication
- **Token Manager**: Centralized token management for all brokers
- **Auto-Refresh**: Automatic token renewal on expiry
- **Multi-Broker Support**: Simultaneous connections to multiple brokers
- **Secure Storage**: Tokens stored in `config/access_tokens/`

### Upstox Integration
- **OAuth Flow**: Interactive authentication with CLIENT_ID and CLIENT_SECRET
- **Historical Data**: Intraday and daily data retrieval
- **Instrument Master**: Complete NSE/BSE instrument database
- **API Rate Limiting**: Automatic rate limit handling

### Zerodha Integration
- **Kite Connect API**: API_KEY and API_SECRET authentication
- **Historical Data**: Multi-timeframe data support
- **Instrument Dump**: Full instrument database download
- **WebSocket Support**: Real-time data streaming (ready for live trading)

### Binance Integration
- **Public API**: No authentication required for historical data
- **Cryptocurrency Support**: Full crypto market coverage
- **Historical Data**: Comprehensive historical OHLCV data

---

## Output & Reporting

### Output Modes
- **Three-File System**: Compact summary format (config, metrics, risk)
- **Enhanced Output Orchestrator**: Comprehensive multi-file output
- **Optimized Output System**: Performance-optimized reporting
- **Output Manager**: Centralized output coordination

### Report Types
- **Trade Reports**: CSV and JSON trade logs (Strategy, RiskApproved)
- **Analysis Reports**: JSON performance metrics
- **Portfolio Reports**: Multi-ticker aggregated metrics
- **Bias Reports**: Look-ahead and survivorship bias detection
- **Risk Reports**: Risk-adjusted metrics and exposure analysis
- **Config Snapshots**: YAML configuration captures

### Visualizations
- **Performance Summary**: Comprehensive performance charts
- **Trade Distribution**: Win/loss distribution histograms
- **Trade Timeline**: Chronological trade visualization
- **Educational Insights**: Risk management and trading psychology charts
- **Portfolio Master Dashboard**: Multi-ticker overview
- **Performance Dashboard**: Strategy comparison charts
- **Risk Dashboard**: Risk metrics visualization
- **Signal Analysis**: Entry/exit signal quality charts
- **Trade Analysis**: Trade pattern analysis
- **Equity Curves**: Cumulative return curves with drawdown overlays
- **Comparison Dashboard**: Multi-strategy comparison

---

## CLI & User Experience

### Command-Line Interface
- **Mode-Based Execution**: `--mode` flag for different workflows
- **Date Range Support**: Flexible date range specification
- **Ticker Selection**: Multi-ticker support with auto-discovery
- **Template Selection**: `--template` flag for risk profiles
- **Strategy Selection**: `--strategies` flag for strategy choice
- **Parallel Processing**: `--parallel` flag for concurrent execution
- **Worker Limits**: `--max-workers` for resource management
- **Dry-Run Mode**: Preview operations without execution
- **Verbose Logging**: `--verbose` flag for detailed logs

### User Assistance
- **AI-First Documentation**: LLM-ready prompts in README
- **Interactive Modes**: Guided setup and data fetching
- **Error Guidance**: Clear error messages with troubleshooting hints
- **Progress Indicators**: Real-time progress for long operations
- **Validation Feedback**: Immediate feedback on data quality issues

---

## Architecture & Performance

### Modular Architecture
- **Clean Separation of Concerns**: Core, runners, strategies, analysis
- **Pluggable Components**: Easy extension and customization
- **Factory Patterns**: Strategy, provider, and output factories
- **Dependency Injection**: Configurable component composition

### Performance Optimizations
- **Parallel Processing**: Multi-core execution for backtests
- **Caching System**: Repeated data access optimization
- **Memory Management**: Efficient handling of large datasets
- **Lazy Loading**: On-demand data loading
- **Incremental Updates**: Avoid full re-processing

### Workflow Orchestration
- **Mode Handlers**: Specialized handlers for each execution mode
- **Task Executor**: Parallel/sequential task execution
- **Component Isolation**: Independent testing of workflow components
- **Execution Engine**: Core backtesting logic
- **Orchestrator**: High-level workflow coordination

---

## Documentation

### User Guides
- **README.md**: Quick start and overview
- **SETUP_GUIDE.md**: Installation and environment setup
- **BROKER_SETUP.md**: Broker API configuration
- **STRATEGY_GUIDE.md**: Custom strategy development
- **TEMPLATE_GUIDE.md**: Risk template configuration
- **CLI_REFERENCE.md**: Complete CLI documentation
- **OUTPUT_GUIDE.md**: Understanding results and visualizations
- **TROUBLESHOOTING.md**: Common issues and solutions

### Technical Documentation
- **CLAUDE.md**: AI assistant instructions and system architecture
- **CHANGELOG.md**: Comprehensive version history (210 lines)
- **RELEASE_NOTES.md**: Release highlights and features
- **DATA_VALIDATION_CRITERIA.md**: Validation rules and thresholds
- **SIGNAL_HANDLING_AND_VALIDATION_FIXES.md**: Technical fix documentation
- **strategylab_v2_phase0_audit.md**: V2 release decision history

### Analysis Documentation
- **METHODOLOGY.md** (82 pages): Backend logic and statistical foundations
- **WORKFLOW_SOP.md** (60 pages): Step-by-step execution guide
- **IMPLEMENTATION_STATUS.md** (42 pages): Technical tracking
- **DOCUMENTATION_INDEX.md**: Navigation guide for analysis docs
- **MSE_STRATEGY_GUIDE.md**: MSE strategy deep dive
- **ANALYSIS_PROTOCOL.md**: Analysis execution protocol

### Total Documentation
- **~220 pages** of comprehensive documentation
- **~65,000 words** across all docs
- **~5,000 lines** of code comments
- **~2,000 lines** of inline docstrings

---

## Logging & Observability

### Logging System
- **Structured Logging**: JSON-formatted logs for parsing
- **Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Component-Specific Logs**: Separate logs per module
- **Timestamped Entries**: Precise timing for all events
- **Run Logs**: Complete audit trail in `analysis/run_logs/`

### Debugging Tools
- **Verbose Mode**: Detailed execution traces
- **Dry-Run Mode**: Preview without execution
- **Validation Reports**: Comprehensive data quality reports
- **Error Stack Traces**: Full error context for debugging
- **Output Manifests**: Complete output file inventory

---

## Indian Equities Master Pipeline (V2)

### Pipeline Features
- **Master CSV Management**: `data/indian_equities_master.csv`
- **Sector Classification**: Automatic sector mapping
- **Metadata Enrichment**: Company info, sector, industry
- **Quality Scoring**: Data quality assessment per ticker
- **Incremental Updates**: Add new tickers without re-processing
- **Gap Detection**: Identify missing data periods
- **Validation Integration**: End-to-end quality checks

---

## Production Readiness

### Reliability
- **Error Handling**: Comprehensive exception handling
- **Retry Logic**: Automatic retry with exponential backoff
- **Fallback Mechanisms**: Provider and data source fallbacks
- **Validation Gates**: Data quality gates before execution
- **Circuit Breakers**: Automatic halts on critical errors

### Reproducibility
- **Deterministic Pipelines**: Consistent results across runs
- **Pinned Dependencies**: requirements.txt with version locks
- **Seed Control**: Random seed management for reproducibility
- **Immutable Datasets**: Data pool integrity preservation
- **Environment Capture**: Complete environment documentation

### Testing
- **Unit Tests**: Component-level testing
- **Integration Tests**: End-to-end pipeline testing
- **Parity Tests**: Live vs backtest validation
- **Precision Tests**: Exchange precision compliance
- **CI/CD Ready**: GitHub Actions workflow templates

---

## Platform Support

### Operating Systems
- **Windows**: Primary platform (PowerShell-first)
- **macOS**: Fully supported
- **Linux**: Fully supported

### Python Versions
- **Python 3.10+**: Required
- **Python 3.13**: Fully tested

### Dependencies
- **pandas**: Core data manipulation
- **numpy**: Numerical computing
- **matplotlib**: Visualization
- **seaborn**: Advanced plotting
- **PyPortfolioOpt**: Portfolio optimization
- **pytest**: Testing framework
- **requests**: HTTP client for broker APIs

---

## Future Roadmap

### Planned Features (Phase 2)
- **ML-Based Validation**: Intelligent anomaly detection
- **Real-Time Processing**: Live trading integration
- **Advanced Risk Management**: Portfolio-level risk controls
- **Cloud Deployment**: Scalable cloud-based backtesting
- **Web Interface**: Interactive web dashboard
- **Natural Language Interface**: LLM-powered command interface
- **Strategy Marketplace**: Community strategy sharing
- **Paper Trading**: Virtual trading environment

### Continuous Improvements
- **Performance Optimization**: Faster execution
- **Enhanced Visualizations**: More chart types
- **Additional Strategies**: More built-in strategies
- **Extended Broker Support**: More broker integrations
- **Advanced Analytics**: Machine learning integration

---

## Version History

### V2 (Equities Release - October 2025)
- Validation framework (parity, precision, bias)
- Test suites (59 passing tests)
- Advanced analysis framework
- Portfolio construction system
- Indian equities master pipeline
- Incremental parquet updates
- Comprehensive documentation (~220 pages)

### V1 (Initial Release - June 2025)
- Core backtesting engine
- Broker integrations (Upstox, Zerodha, Binance)
- Basic strategies (MSE, SMA, Bollinger Bands)
- Configuration templates
- Basic documentation
- Multi-mode execution

---

**Total Feature Count**: 200+ features across 8 major categories

**Lines of Code**: ~15,000+ lines of production Python code

**Test Coverage**: 59 passing tests

**Documentation**: ~220 pages

**Supported Assets**: Equities (NSE/BSE), Cryptocurrencies

**Supported Brokers**: 3 (Upstox, Zerodha, Binance)

**Strategies**: 4 built-in + custom template

**Analysis Scripts**: 16 (9 generic + 7 portfolio)

**Configuration Templates**: 5

**Validation Modules**: 4

**Output Formats**: CSV, JSON, PNG, YAML

---

*Built with ❤️ for traders who believe in data-driven decisions*
