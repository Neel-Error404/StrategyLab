# StrategyLab Releases

Complete version history and release notes for the StrategyLab Backtesting System

---

## Version 2.0 (Equities Release) - October 2025

**Release Date**: October 29, 2025
**Codename**: "Equities v2"
**Git Tag**: `release/strategylab-v2`
**Commit**: `1955cc1` (Merge PR #11)

### Overview
Major release focusing on production-ready equities trading with comprehensive validation, testing, and analysis frameworks. This release removes options infrastructure for a clean open-source equities-only system.

### 🚀 Major Features

#### 1. Validation Framework (NEW)
Complete validation system ensuring backtest-live parity and data precision:

**Config Parity Validator** (`src/core/validation/config_parity_validator.py`)
- Ensures critical config parameters match between live and backtest
- Environment variable validation
- Configuration drift detection
- Pre-trade validation gates

**Signal Parity Validator** (`src/core/validation/signal_parity_validator.py`)
- Signal-by-signal comparison between live and backtest
- Divergence detection and reporting
- Timestamp synchronization validation
- Parity score calculation

**Precision Validator** (`src/core/validation/precision_validator.py`)
- Exchange-specific price/quantity precision enforcement
- PnL rounding validation
- Order quantity precision checks
- Compliance with broker rules

**Enhanced Bias Detector** (`src/core/validation/bias_detector.py`)
- Look-ahead bias detection
- Survivorship bias checks
- Data snooping monitoring

#### 2. Test Suites (NEW)
Comprehensive test coverage with 59 passing tests:

- `tests/test_backtest_live_parity.py`: Validates backtest-live signal alignment
- `tests/test_precision_validation.py`: Tests price/quantity precision enforcement
- `tests/indian_equities_master/test_pipeline.py`: End-to-end pipeline testing
- `tests/conftest.py`: Test configuration and fixtures

**Test Results**:
```
59 passed in X.XXs
Coverage: Validation framework, data pipeline, strategy execution
```

#### 3. Advanced Analysis Framework (NEW)
Complete portfolio construction and strategy optimization system:

**Generic Analysis Suite** (9 scripts):
- `01_basic_eda.py`: Foundation statistics
- `02_trade_type_analysis.py`: Directional bias detection
- `03_cascade_analysis.py`: Behavioral pattern detection
- `04_stop_loss_simulation.py`: Risk management optimization
- `05_ticker_ranking.py`: Quality scoring system
- `06_risk_adjusted_patterns.py`: Risk-normalized performance
- `07_top50_vs_overall.py`: Selection validation
- `08_top50_pattern_breakdown.py`: Winner profiling
- `09_validation_check.py`: Data integrity audit

**Portfolio Construction Suite** (7 scripts):
- `00_ticker_ranking.py`: Comprehensive ranking
- `01_anti_cascade_filter.py`: Behavioral bias removal
- `02_sector_classification.py`: Diversification framework
- `03_combination_generator.py`: Constrained optimization
- `04_portfolio_optimizer.py`: Equal-weight evaluation
- `05_pypfopt_weights.py`: Markowitz optimization
- `06_equity_curves.py`: Visual validation

**Analysis Documentation** (~220 pages):
- `METHODOLOGY.md` (82 pages): Statistical foundations and backend logic
- `WORKFLOW_SOP.md` (60 pages): Step-by-step execution guide
- `IMPLEMENTATION_STATUS.md` (42 pages): Technical tracking
- `DOCUMENTATION_INDEX.md`: Navigation guide

#### 4. Incremental Data Updates (NEW)
Efficient data management without full re-fetches:

**New Tools**:
- `src/core/etl/pool_inspector.py`: Analyze existing data pools
- `src/core/etl/gap_calculator.py`: Identify missing data periods
- `--mode update`: Incremental pool extension

**Features**:
- Parquet-based incremental updates
- Dry-run mode for preview
- Validation-only mode
- Automatic backups before updates
- Gap-aware fetching

#### 5. Indian Equities Master Pipeline (NEW)
Production-grade data pipeline for Indian equities:

**Components**:
- `data/indian_equities_master.csv`: Master ticker database
- Sector classification and metadata
- Quality scoring per ticker
- Incremental ticker additions
- End-to-end validation

**PR #3**: Indian Equities Master Pipeline
- Commit: `588988c`
- Complete NSE/BSE ticker management
- Sector and industry mapping

#### 6. Configuration Enhancements (ENHANCED)
Environment-aware configuration system:

**New Config Loader** (`config/config_loader.py`):
- YAML loading with environment substitution
- `${VAR}` or `${VAR:default}` syntax
- `.env` file support
- Schema validation

**Template Updates**:
- `aggressive.yaml`: Enhanced with V2 features
- `conservative.yaml`: Updated risk parameters
- `minimal.yaml`: Refined for learning
- `debug.yaml`: New debugging template
- `portfolio_diversified.yaml`: Multi-ticker optimization

**Removed**:
- `options.yaml`: Removed for OSS equities-only release

### 📊 Phase Integration

This release integrates multiple development phases:

**Phase 4: Backtest System** (PR #7)
- Commit: `8212cec`
- Complete backtesting engine
- Multi-timeframe support
- Intraday session restrictions

**Phase 5: Analysis Framework** (PR #5)
- Commit: `5b8fd29`
- Advanced analysis modules
- Portfolio construction
- Statistical validation

**Phase 6: Options Infrastructure** (PR #6)
- Commit: `9f4d440`
- **Note**: Added then removed for V2 OSS release
- Options code moved to private repository

### 🔧 Technical Improvements

#### Multi-Timeframe Architecture
- Complete 5m + 15m strategy support
- Enhanced MSE strategy with production features
- Two-bar execution rule (signal → pending → execute)
- 525-minute warmup for MACD stability
- Proper look-ahead protection with `.shift(1)`

#### Data Management
- Ticker-first parquet structure
- Auto-discovery of available tickers
- Multi-timeframe loading with fallback
- Legacy CSV support removed
- UTF-8 sanitized codebase

#### Strategy Engine
- Strategy factory with timeframe validation
- Mandatory warmup periods
- EOD handling at 15:15 IST
- Cascade trade prevention
- Timestamp-based warmup

### 📝 Documentation Updates

**New Documentation**:
- `FEATURES.md`: Comprehensive feature list (200+ features)
- `RELEASES.md`: This version history document
- `docs/CHANGELOG.md`: Detailed change log (210 lines)
- `docs/DATA_VALIDATION_CRITERIA.md`: Validation rules
- `docs/SIGNAL_HANDLING_AND_VALIDATION_FIXES.md`: Technical fixes
- `docs/strategylab_v2_phase0_audit.md`: Release decision history

**Updated Documentation**:
- `README.md`: V2 features and quick start
- `docs/BROKER_SETUP.md`: Enhanced setup instructions
- `docs/CLI_REFERENCE.md`: New modes and flags
- `docs/OUTPUT_GUIDE.md`: V2 output formats
- `docs/STRATEGY_GUIDE.md`: Multi-timeframe strategies
- `docs/TEMPLATE_GUIDE.md`: Updated templates
- `docs/TROUBLESHOOTING.md`: V2 common issues

### 🐛 Critical Fixes

**Signal Handling**:
- Fixed Ctrl+C (SIGINT) now properly terminates instead of defaulting to SMA crossover
- Fixed multiprocessing worker interruption on Windows
- Removed hardcoded `sma_crossover` default
- Fixed Windows multiprocessing pickle serialization error

**Data Validation**:
- Fixed overly strict validation blocking real market data
- Changed single-row price inconsistencies to warnings
- Statistical thresholds (0.1% of data) for reasonable validation
- Fixed strategy re-registration spam in multiprocessing

### ⚡ Breaking Changes

1. **Strategies Required**: `--strategies` parameter now mandatory for backtest mode
2. **No Default Fallback**: System no longer defaults to SMA crossover on errors
3. **Data Structure**: Prefers ticker-first parquet over timeframe-first CSV
4. **Validation Behavior**: Minor anomalies generate warnings instead of blocking
5. **Options Removed**: Options infrastructure removed for OSS equities release

### 📈 Performance Metrics

**Before V2 vs After V2**:

| Metric | V1 | V2 | Improvement |
|--------|----|----|-------------|
| Successful Backtests | 0% (validation blocked) | 96.2% (25/26 tickers) | +96.2% |
| Ctrl+C Behavior | SMA fallback bug | Immediate exit | ✅ Fixed |
| Strategy Registration | Per task (spam) | Per worker | ~90% log reduction |
| Data Validation | Zero tolerance | Smart thresholds | Real market data works |
| Trade Generation | Blocked | 1,351 trades on RELIANCE | ✅ Working |
| Test Coverage | 0 tests | 59 passing tests | ✅ Complete |
| Documentation | ~50 pages | ~220 pages | +340% |

### 🏆 Phase 1 Results

**Best Portfolio Identified**:
- **Tickers**: AXISBANK, HCLTECH, INFY, SUNPHARMA, KOTAKBANK
- **Sharpe Ratio**: 0.826 (excellent)
- **Annual Return**: 3.37%
- **Volatility**: 4.08% (very low)
- **Max Drawdown**: -4.88% (minimal)
- **Period**: 2022-01-01 to 2025-08-31 (3.66 years)

**Sector Diversification**:
- Banking: 40%
- IT: 40%
- Pharma: 20%

### 🎯 Known Limitations

1. **Equities Only**: Options infrastructure removed for V2
2. **Indian Markets Focus**: Primarily NSE/BSE data
3. **Strategy Optimization**: Phase 2 in progress
4. **Live Trading**: Not yet implemented (coming in Phase 3)
5. **Web Interface**: CLI only (web UI planned)

### 📦 What's Included

**Supported Assets**: Indian equities (NSE/BSE), cryptocurrencies
**Supported Brokers**: Upstox, Zerodha, Binance
**Built-in Strategies**: 4 (MSE, SMA, SMA Crossover, Bollinger Bands)
**Templates**: 5 (minimal, conservative, aggressive, portfolio_diversified, debug)
**Analysis Scripts**: 16 (9 generic + 7 portfolio)
**Test Suites**: 3 (59 tests total)
**Documentation**: ~220 pages across 25+ documents

### 🔮 Next Steps (Phase 2)

**Strategy Optimization**:
- Exit threshold optimization (50-95% MACD range)
- Entry signal analysis (MACD strength, EMA spread filters)
- Combined optimization (entry + exit)
- Expected: Sharpe 0.83 → 1.5-2.0 (+80-140%)

**Timeline**: 4-6 weeks

### 📥 Upgrade Path from V1

1. **Backup existing data**: `cp -r data data.v1.backup`
2. **Pull latest code**: `git pull origin main`
3. **Install dependencies**: `pip install -r requirements.txt`
4. **Run tests**: `python -m pytest tests/ -v`
5. **Update configs**: Review new templates in `config/templates/`
6. **Run validation**: `python src/runners/unified_runner.py --mode validate --dates 2024-01-03`

**Migration Notes**:
- Update strategy calls to use `--strategies` flag
- Review validation thresholds in configs
- Update to parquet-based data pools if using CSV
- Remove any options-related configurations

### 🙏 Contributors

- **Lead Developer**: Development Team
- **Analysis Framework**: Quantitative Research Team
- **Validation System**: Quality Assurance Team
- **Documentation**: Technical Writing Team
- **Testing**: QA and Testing Team

---

## Version 1.0 (Initial Public Release) - September 2025

**Release Date**: September 12, 2025
**Codename**: "Foundation"
**Git Tag**: `v1.0.0`
**Commit**: `fcb1806`

### Overview
First stable release establishing the core backtesting framework with broker integration, risk management, and comprehensive documentation.

### 🚀 Major Features

#### 1. Core Backtesting Engine
- Multi-mode execution (backtest, analyze, visualize, validate, fetch)
- Parallel processing support
- Clean modular architecture
- Pluggable strategy system

#### 2. Broker Integration
**Upstox**:
- OAuth authentication flow
- Historical data fetching (intraday and daily)
- Instrument master database
- Token management

**Zerodha Kite**:
- API key authentication
- Multi-timeframe data support
- Complete instrument dump
- Rate limit handling

**Binance**:
- Public API (no auth for historical)
- Cryptocurrency data
- Full OHLCV historical data

#### 3. Strategy System
**Built-in Strategies**:
- **MSE (Mean Squared Error)**: Multi-indicator strategy
- **SMA (Simple Moving Average)**: Trend following
- **SMA Crossover**: Dual moving average system
- **Bollinger Bands**: Mean reversion

**Strategy Features**:
- Pluggable architecture
- Easy registration via `register_strategies.py`
- Strategy factory pattern
- Template for custom strategies (`strategy_template.py`)

#### 4. Risk Management
- Position sizing controls (5%-20%)
- Drawdown protection
- Trade validation
- Portfolio-level risk monitoring
- Transaction cost modeling

#### 5. Configuration System
**Templates**:
- `minimal.yaml`: 5% max position (learning)
- `conservative.yaml`: 15% max position (low risk)
- `aggressive.yaml`: 20% max position (high risk)
- `options.yaml`: Options trading (V1 only)
- `portfolio_diversified.yaml`: Multi-ticker portfolio

**Dual Config System**:
- `config/config.py`: Infrastructure configuration
- `config/unified_config.py`: Trading configuration

#### 6. Data Management
- Date-range organized pools
- Timeframe-based folder structure
- CSV format support
- Multi-timeframe loading
- Data validation and quality checks

#### 7. Output & Reporting
**Output Modes**:
- Three-file system (compact)
- Enhanced output orchestrator (comprehensive)
- Output manager (centralized)

**Reports**:
- Trade logs (CSV, JSON)
- Analysis reports
- Performance metrics
- Bias reports
- Risk reports
- Config snapshots

#### 8. Visualizations
- Performance summary charts
- Trade distribution histograms
- Trade timeline chronological view
- Educational insights charts
- Portfolio master dashboard
- Risk dashboard
- Signal analysis charts

#### 9. CLI Interface
- Mode-based execution
- Date range support
- Ticker selection with auto-discovery
- Template selection
- Strategy selection
- Parallel processing flags
- Verbose logging

#### 10. Documentation
**User Guides**:
- `README.md`: Quick start
- `docs/SETUP_GUIDE.md`: Installation
- `docs/BROKER_SETUP.md`: Broker configuration
- `docs/STRATEGY_GUIDE.md`: Custom strategies
- `docs/TEMPLATE_GUIDE.md`: Risk templates
- `docs/CLI_REFERENCE.md`: CLI documentation
- `docs/OUTPUT_GUIDE.md`: Understanding results
- `docs/TROUBLESHOOTING.md`: Common issues

**Technical Docs**:
- `CLAUDE.md`: System architecture
- `RELEASE_NOTES.md`: Release highlights
- `GITHUB_REPO_DESCRIPTION.md`: Repository overview

### 📊 Initial Capabilities

**Data Coverage**:
- 6 months historical data included
- 6 tickers (RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, ITC)
- 1-minute to daily timeframes
- ~850,000 data rows included

**Performance**:
- Parallel processing on multi-core systems
- Memory-efficient data handling
- ~22 seconds for full workflow on RELIANCE (67K records)

### 🐛 Known Issues (Fixed in V2)

1. **Signal Handling**: Ctrl+C defaults to SMA crossover (fixed in V2)
2. **Data Validation**: Too strict, blocks real market data (fixed in V2)
3. **Multiprocessing**: Serialization errors on Windows (fixed in V2)
4. **Strategy Registration**: Spam logs in parallel mode (fixed in V2)
5. **Default Fallback**: Unwanted SMA crossover execution (fixed in V2)

### 🎯 Limitations

1. Single-timeframe strategy support only
2. CSV-based data structure
3. No multi-timeframe strategies
4. No validation framework
5. No test suites
6. No portfolio construction tools

### 📦 What's Included

**Initial Release Artifacts**:
- ~100 Python files
- ~10,000 lines of code
- 7 comprehensive documentation files
- 5 configuration templates
- Sample data for 6 tickers
- GitHub Actions CI/CD workflows

### 🔮 Roadmap (Completed in V2)

- ✅ Multi-timeframe architecture
- ✅ Enhanced data validation
- ✅ Validation framework
- ✅ Test suites
- ✅ Portfolio construction
- ✅ Strategy optimization pipeline
- ✅ Incremental data updates

---

## Version 0.05 (Pre-Release) - June 2025

**Release Date**: June 19, 2025
**Codename**: "Laboratory"
**Git Tag**: `v0.05`
**Commit**: `2c69e5b` (Initial commit)

### Overview
Initial public release establishing core functionality and documentation. Experimental release for early adopters and testing.

### 🚀 Features

#### Core Features
- Basic backtesting framework
- Broker integration (Zerodha, Upstox)
- Simple strategies (MSE, SMA)
- Risk management templates
- Basic analysis and visualization

#### Documentation
- README with AI-assisted setup prompts
- Basic user guides
- Broker setup instructions

#### Limitations
- Experimental release
- Limited testing
- Basic features only
- No validation framework

### 📦 Initial Commit
**260 files changed**:
- 853,667 insertions
- Complete project structure
- Sample data included
- Documentation framework

---

## Version History Summary

| Version | Date | Codename | Major Features | Lines of Code | Documentation | Tests |
|---------|------|----------|----------------|---------------|---------------|-------|
| V2.0 | Oct 2025 | Equities v2 | Validation, Testing, Analysis, Portfolio | ~15,000 | ~220 pages | 59 |
| V1.0 | Sep 2025 | Foundation | Core Engine, Brokers, Strategies, Risk | ~10,000 | ~50 pages | 0 |
| V0.05 | Jun 2025 | Laboratory | Initial Release, Basic Features | ~8,000 | ~20 pages | 0 |

---

## Feature Evolution

### Core Engine
- **V0.05**: Basic backtesting
- **V1.0**: Multi-mode execution, parallel processing
- **V2.0**: Validation framework, incremental updates

### Strategies
- **V0.05**: MSE, SMA (basic)
- **V1.0**: MSE, SMA, SMA Crossover, Bollinger Bands
- **V2.0**: Production-grade MSE with multi-timeframe

### Data Management
- **V0.05**: Simple CSV files
- **V1.0**: Organized pools, timeframe folders
- **V2.0**: Parquet support, incremental updates, master pipeline

### Analysis
- **V0.05**: Basic metrics
- **V1.0**: Comprehensive metrics, visualizations
- **V2.0**: Advanced analysis (16 scripts), portfolio construction, strategy optimization

### Validation
- **V0.05**: None
- **V1.0**: Basic data quality checks
- **V2.0**: Complete validation framework (parity, precision, bias)

### Testing
- **V0.05**: None
- **V1.0**: None
- **V2.0**: 59 passing tests across 3 test suites

### Documentation
- **V0.05**: ~20 pages (basic)
- **V1.0**: ~50 pages (comprehensive user guides)
- **V2.0**: ~220 pages (complete technical and analysis docs)

---

## Upgrade Paths

### V0.05 → V1.0
1. Pull latest code
2. Install new dependencies
3. Update configs to use unified system
4. Reorganize data into pool structure

### V1.0 → V2.0
1. Backup existing data
2. Pull latest code
3. Install new dependencies (pytest, etc.)
4. Update configs (remove options, add debug)
5. Migrate to parquet data pools
6. Add `--strategies` flag to backtest commands
7. Run validation suite

---

## Support

### V2.0 (Current)
- ✅ **Active Support**: Full support with updates
- ✅ **Bug Fixes**: Critical and minor bug fixes
- ✅ **Feature Updates**: New features added
- ✅ **Documentation**: Maintained and updated

### V1.0 (Previous)
- ⚠️ **Maintenance Mode**: Critical bug fixes only
- ⚠️ **No New Features**: Upgrade to V2 for new features
- ⚠️ **Limited Support**: Community support only

### V0.05 (Deprecated)
- ❌ **No Support**: Deprecated, upgrade to V2
- ❌ **No Bug Fixes**: Not maintained
- ❌ **Security**: May have unpatched vulnerabilities

---

## License

All versions released under **MIT License**. See [LICENSE](LICENSE) for details.

---

## Feedback & Contributions

- **Issues**: Report via [GitHub Issues](https://github.com/Neel-Error404/StrategyLab/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Neel-Error404/StrategyLab/discussions)
- **Pull Requests**: Welcome! See `CONTRIBUTING.md` (if available)
- **Documentation**: Corrections and improvements welcome

---

**Document Maintained By**: StrategyLab Development Team
**Last Updated**: October 30, 2025
**Next Update**: After Phase 2 completion (Strategy Optimization)
