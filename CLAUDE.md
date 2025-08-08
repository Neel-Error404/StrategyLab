# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Environment Setup (Windows PowerShell)
```powershell
# Install dependencies
pip install -r requirements.txt

# Set up virtual environment (recommended)
python -m venv trading_env
trading_env\Scripts\activate
pip install -r requirements.txt
```

### Core Development Commands
```powershell
# Main backtesting runner
python src/runners/unified_runner.py --mode backtest --template conservative --dates 2024-01-01 --tickers RELIANCE

# Interactive data fetching (no arguments needed)
python src/runners/unified_runner.py --mode fetch

# Validate system setup and data
python src/runners/unified_runner.py --mode validate --dates 2024-01-01

# Run analysis on existing backtest results
python src/runners/unified_runner.py --mode analyze --date-ranges 2024-01-01_to_2024-12-31

# Generate visualizations
python src/runners/unified_runner.py --mode visualize --date-ranges 2024-01-01_to_2024-12-31
```

### Data Fetching Commands
```powershell
# Interactive data fetching
python src/core/etl/data_fetcher.py

# Programmatic data fetching
python -c "from src.core.etl.data_fetcher import main; main(provider='upstox', timeframe='1m,5m', days=7)"
```

### Template Testing
```powershell
# Test all available templates
foreach ($template in @("minimal","conservative","aggressive","options")) {
    python src/runners/unified_runner.py --template $template --mode validate --dates 2024-01-01
}
```

## System Architecture

### Recent Cleanup & Optimization
**Latest Update**: System has been optimized and cleaned:
- **Removed 14 empty/redundant files**: test files, unused utilities, legacy components
- **Preserved essential architecture**: All functional components intact
- **Dual configuration confirmed**: Both `config.py` and `unified_config.py` are essential
- **Clean file structure**: Streamlined for optimal maintainability

### Core Architecture Pattern
This is a **modular backtesting system** with a clean separation of concerns:

1. **Entry Points**: `src/runners/unified_runner.py` - Main CLI interface with mode-based execution
2. **Configuration Layer**: `config/` - YAML-based templates and unified configuration management
3. **ETL System**: `src/core/etl/` - Multi-broker data fetching with provider abstraction
4. **Strategy System**: `src/strategies/` - Pluggable trading strategies with registration
5. **Execution Engine**: `src/runners/workflow/` - Modular workflow orchestration
6. **Analysis & Visualization**: Separate engines for metrics and chart generation

### Key Architectural Patterns

#### Data Provider Factory Pattern
The system uses a factory pattern for broker integration:
- **Factory**: `src/core/etl/data_provider/provider_factory.py`
- **Providers**: Upstox (`upstox_provider.py`), Zerodha, Binance (`binance_provider.py`)
- **Token Management**: Centralized in `src/core/etl/token_manager.py`
- **Auto-detection**: Automatically finds providers with valid authentication tokens
- **Configuration**: Broker settings in `config/config.py`

#### Dual Configuration System
The system uses two configuration files for optimal separation of concerns:

**Infrastructure Configuration** (`config/config.py`):
- Broker API credentials (Upstox, Zerodha, Binance)
- Data provider settings and timeframe mappings
- Authentication and token management
- System logging and directory structure

**Trading Configuration** (`config/unified_config.py`):
- Strategy parameters and risk management
- Portfolio settings and position sizing
- Transaction costs and options trading
- Validation and output configurations

**Risk Templates** (`config/templates/`):
- `minimal.yaml` - Ultra-safe (5% max position)
- `conservative.yaml` - Low risk (15% max position) 
- `aggressive.yaml` - High risk (20% max position)
- `options.yaml` - Options trading strategies
- Custom templates supported via `--config` flag

#### Modular Workflow System
Located in `src/runners/workflow/`, the system separates:
- **Mode Handlers**: Different execution modes (backtest, analyze, visualize, fetch, validate)
- **Task Execution**: Parallel/sequential processing with validation
- **Component Isolation**: Each workflow component can be tested independently

#### Data Organization
```
data/pools/
├── YYYY-MM-DD_to_YYYY-MM-DD/    # Date range folders
│   ├── 1minute/                 # Timeframe folders
│   ├── 5minute/
│   └── 1day/
outputs/
├── YYYYMMDD_HHMMSS/            # Timestamped runs
│   ├── {strategy}/             # Strategy-specific results
│   └── {date_range}/           # Date range results
```

## Development Workflow

### Adding New Strategies
1. Create strategy class in `src/strategies/` inheriting from base strategy
2. Register in `src/strategies/register_strategies.py`
3. Test with: `python src/runners/unified_runner.py --mode validate`

### Adding New Data Providers
1. Implement `DataProvider` interface in `src/core/etl/data_provider/`
2. Register in `provider_factory.py`
3. Add configuration to `config/config.py`

### Configuration Management
- **Unified Config**: `config/unified_config.py` - Single source of truth
- **Template System**: Create custom YAML configs based on existing templates
- **Environment Variables**: Support for `.env` files for broker credentials

## Critical Implementation Details

### Token Management (src/core/etl/token_manager.py)
- Handles broker API authentication tokens
- Auto-refresh capability for expired tokens
- Multi-broker support with fallback mechanisms
- Token validation before data fetching operations

### Data Fetching Strategy (src/core/etl/data_fetcher.py)
- Chunked fetching for large date ranges (handles API rate limits)
- Smart chunk sizing based on timeframe (1-day chunks for minute data)
- Retry logic with exponential backoff
- Progress tracking for long-running operations

### Risk Management Integration
Risk parameters are deeply integrated throughout the system:
- Position sizing in strategy execution
- Portfolio-level drawdown monitoring
- Transaction cost modeling
- Multi-timeframe risk controls

### Auto-Discovery Features
- **Ticker Discovery**: Automatically finds available tickers in data pools
- **Provider Discovery**: Detects available authenticated data providers
- **Template Loading**: Discovers and validates configuration templates

## Testing and Validation

### System Health Checks
```powershell
# Validate entire system
python src/runners/unified_runner.py --mode validate --dates 2024-01-01

# Test specific components
python -m pytest tests/ -v  # If test suite exists
```

### Data Quality Validation
The system includes built-in data quality checks:
- Look-ahead bias detection
- Survivorship bias checks
- Data completeness validation
- Outlier detection and handling

## Environment Considerations

### Windows PowerShell Environment
- Always use PowerShell commands first (as per user's global CLAUDE.md instructions)
- UTF-8 initialization may be needed for emoji output
- Virtual environment activation: `trading_env\Scripts\activate`

### Broker API Requirements
- **Upstox**: Requires CLIENT_ID, CLIENT_SECRET, and interactive authentication
- **Zerodha**: Requires API_KEY, API_SECRET
- **Binance**: No authentication for historical data
- Tokens stored in `config/access_tokens/` directory

### Performance Optimization
- Parallel processing available via `--parallel` flag
- Caching system for repeated data access
- Memory management for large datasets
- Configurable worker limits via `--max-workers`

## Mode-Specific Behaviors

### Fetch Mode
- Interactive mode when no arguments provided
- Programmatic mode with date ranges and tickers
- Multi-provider support with automatic fallback

### Backtest Mode
- Full workflow including execution, analysis, and visualization
- Template-based risk management
- Auto-discovery of tickers from data pools

### Analysis Mode
- Portfolio-level performance metrics
- Risk decomposition analysis
- Comparative analysis across strategies

### Visualize Mode
- Comprehensive chart generation
- Interactive plots for strategy performance
- Export capabilities (PNG, HTML)

This architecture enables rapid development while maintaining production-ready reliability and comprehensive risk management capabilities.