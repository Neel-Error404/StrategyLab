# StrategyLab Test Suite

Comprehensive test documentation for the StrategyLab Backtesting System

**Test Framework**: pytest
**Coverage**: Validation framework, data pipeline, strategy execution
**Status**: 59 passing tests

---

## Quick Start

### Running All Tests

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run quietly (summary only)
python -m pytest tests/ -q
```

### Running Specific Test Suites

```bash
# Backtest-Live Parity Tests
python -m pytest tests/test_backtest_live_parity.py -v

# Precision Validation Tests
python -m pytest tests/test_precision_validation.py -v

# Indian Equities Master Pipeline Tests
python -m pytest tests/indian_equities_master/test_pipeline.py -v
```

### Running Specific Test Classes or Functions

```bash
# Run a specific test class
python -m pytest tests/test_precision_validation.py::TestPriceValidation -v

# Run a specific test function
python -m pytest tests/test_backtest_live_parity.py::TestWarmupPeriodParity::test_warmup_periods_match_525_minutes -v
```

---

## Test Suite Structure

### 1. **Backtest-Live Parity Tests** (`test_backtest_live_parity.py`)

**Purpose**: Ensure backtest and live trading environments produce identical signals

**Test Coverage**:
- ✅ Warmup period validation (525-minute MSE warmup)
- ✅ Configuration parameter parity
- ✅ Timestamp synchronization
- ✅ Signal generation parity
- ✅ Indicator value alignment
- ✅ Order generation consistency
- ✅ Position tracking parity
- ✅ PnL calculation accuracy

**Key Test Classes**:
- `TestWarmupPeriodParity`: Validates warmup configuration
- `TestConfigurationParity`: Critical parameter matching
- `TestSignalGenerationParity`: Signal stream comparison
- `TestIndicatorParity`: MACD, EMA alignment validation
- `TestOrderParity`: Order generation consistency
- `TestPositionParity`: Position tracking validation
- `TestPnLParity`: PnL calculation accuracy

**Example Usage**:
```bash
# Run all parity tests
python -m pytest tests/test_backtest_live_parity.py -v

# Run only signal generation tests
python -m pytest tests/test_backtest_live_parity.py::TestSignalGenerationParity -v
```

**Critical for**: Live trading deployment

---

### 2. **Precision Validation Tests** (`test_precision_validation.py`)

**Purpose**: Enforce exchange-specific precision rules for prices, quantities, and PnL

**Test Coverage**:
- ✅ Price validation and rounding (4 decimals)
- ✅ Quantity validation (integer lots)
- ✅ PnL calculation precision
- ✅ Order value calculation
- ✅ Edge cases (large numbers, small numbers, negatives)
- ✅ Floating-point arithmetic precision
- ✅ Compliance checking

**Key Test Classes**:
- `TestPriceValidation`: Price rounding to 4 decimals
- `TestQuantityValidation`: Lot size enforcement
- `TestPnLValidation`: PnL precision checks
- `TestOrderValueValidation`: Order value calculation
- `TestEdgeCases`: Boundary and edge case handling
- `TestFloatingPointPrecision`: Floating-point accuracy
- `TestComplianceChecking`: Exchange rule compliance

**Example Usage**:
```bash
# Run all precision tests
python -m pytest tests/test_precision_validation.py -v

# Run only price validation tests
python -m pytest tests/test_precision_validation.py::TestPriceValidation -v
```

**Critical for**: Exchange compliance, order rejection prevention

---

### 3. **Indian Equities Master Pipeline Tests** (`indian_equities_master/test_pipeline.py`)

**Purpose**: End-to-end testing of the Indian equities data pipeline

**Test Coverage**:
- ✅ Data fetching from NSE/BSE
- ✅ Sector classification
- ✅ Metadata enrichment
- ✅ Quality scoring
- ✅ Incremental updates
- ✅ Gap detection
- ✅ Validation integration

**Example Usage**:
```bash
# Run pipeline tests
python -m pytest tests/indian_equities_master/test_pipeline.py -v
```

**Critical for**: Data quality assurance

---

## Test Configuration

### `conftest.py`

**Purpose**: Shared fixtures and configuration for all tests

**Fixtures Provided**:
- Test data generators
- Mock broker API responses
- Temporary file system setup
- Configuration templates
- Strategy instances

**Usage in Tests**:
```python
def test_example(tmpdir, sample_data):
    # tmpdir: temporary directory fixture
    # sample_data: sample OHLCV data fixture
    pass
```

---

## Writing New Tests

### Test File Naming Convention

```
test_<module_name>.py          # For unit tests
test_<feature>_integration.py  # For integration tests
test_<component>_e2e.py         # For end-to-end tests
```

### Test Function Naming Convention

```python
def test_<function>_<expected_behavior>():
    """Test that <function> <expected_behavior>"""
    pass
```

**Examples**:
- `test_sharpe_ratio_positive_returns()`: Test that Sharpe ratio is positive for positive returns
- `test_price_rounds_to_4_decimals()`: Test that prices are rounded to 4 decimal places
- `test_warmup_periods_match_525_minutes()`: Test that both backtest and live use 525-minute warmup

### Test Structure (AAA Pattern)

```python
def test_example():
    # ARRANGE: Set up test data and environment
    data = create_sample_data()
    strategy = MSEStrategy(config)

    # ACT: Execute the function under test
    result = strategy.generate_signals(data)

    # ASSERT: Verify expected behavior
    assert result is not None
    assert len(result) == len(data)
    assert result['signal'].isin([-1, 0, 1]).all()
```

### Using Fixtures

```python
import pytest

@pytest.fixture
def sample_config():
    """Fixture providing a sample configuration"""
    return {
        'strategy': 'mse',
        'max_position_size': 15,
        'warmup_minutes': 525
    }

def test_with_fixture(sample_config):
    """Test using the sample_config fixture"""
    assert sample_config['strategy'] == 'mse'
```

### Parameterized Tests

```python
import pytest

@pytest.mark.parametrize("price,expected", [
    (123.456789, 123.4568),
    (100.00001, 100.0000),
    (999.99999, 1000.0000),
])
def test_price_rounding(price, expected):
    """Test price rounding with multiple inputs"""
    result = validate_price(price)
    assert result == Decimal(str(expected))
```

---

## Test Categories

### Unit Tests
**Purpose**: Test individual functions in isolation

**Characteristics**:
- Fast execution (< 1 second per test)
- No external dependencies (mocked)
- Single responsibility
- High coverage of edge cases

**Example**:
```python
def test_calculate_sharpe_ratio():
    returns = pd.Series([0.01, 0.02, -0.01, 0.03])
    sharpe = calculate_sharpe_ratio(returns)
    assert sharpe > 0
```

### Integration Tests
**Purpose**: Test interactions between components

**Characteristics**:
- Moderate execution time (1-10 seconds per test)
- Real dependencies (actual file I/O, database, etc.)
- Tests component integration
- Validates workflows

**Example**:
```python
def test_backtest_workflow_integration():
    config = load_config('test_config.yaml')
    data = load_data('RELIANCE', '2024-01-01', '2024-01-31')
    strategy = create_strategy(config)
    results = execute_backtest(strategy, data)
    assert len(results.trades) > 0
```

### End-to-End Tests
**Purpose**: Test complete workflows from CLI to output

**Characteristics**:
- Slow execution (10+ seconds per test)
- Full system integration
- Tests user workflows
- Validates outputs

**Example**:
```python
def test_full_backtest_e2e():
    result = subprocess.run([
        'python', 'src/runners/unified_runner.py',
        '--mode', 'backtest',
        '--template', 'minimal',
        '--dates', '2024-01-01',
        '--tickers', 'RELIANCE'
    ], capture_output=True)
    assert result.returncode == 0
    assert os.path.exists('outputs/')
```

---

## Test Markers

### Custom Markers

```python
@pytest.mark.slow
def test_large_dataset():
    """Mark tests that take > 10 seconds"""
    pass

@pytest.mark.integration
def test_database_connection():
    """Mark integration tests"""
    pass

@pytest.mark.e2e
def test_full_workflow():
    """Mark end-to-end tests"""
    pass

@pytest.mark.skip(reason="WIP - not implemented yet")
def test_future_feature():
    """Skip tests temporarily"""
    pass

@pytest.mark.xfail(reason="Known issue #123")
def test_known_bug():
    """Mark expected failures"""
    pass
```

### Running Tests by Marker

```bash
# Run only fast tests (exclude slow)
python -m pytest -m "not slow" -v

# Run only integration tests
python -m pytest -m integration -v

# Run only e2e tests
python -m pytest -m e2e -v
```

---

## Test Data

### Sample Data Location

```
tests/
├── fixtures/           # Test data fixtures
│   ├── sample_ohlcv.csv
│   ├── sample_trades.csv
│   └── sample_configs/
├── mocks/              # Mock objects
│   ├── mock_broker_api.py
│   └── mock_data_provider.py
└── conftest.py         # Shared fixtures
```

### Creating Test Data

```python
import pandas as pd
import numpy as np

def create_sample_ohlcv(n_bars=100):
    """Create sample OHLCV data for testing"""
    dates = pd.date_range('2024-01-01', periods=n_bars, freq='5min')

    # Generate realistic price data
    close = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)
    high = close + np.random.rand(n_bars) * 2
    low = close - np.random.rand(n_bars) * 2
    open_price = close.shift(1).fillna(close[0])
    volume = np.random.randint(1000, 10000, n_bars)

    return pd.DataFrame({
        'timestamp': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
```

---

## Continuous Integration

### GitHub Actions

Tests run automatically on:
- Every push to any branch
- Every pull request
- Scheduled nightly builds

**Workflow**: `.github/workflows/ci.yml`

**Test Commands in CI**:
```yaml
- name: Run tests
  run: |
    python -m pytest tests/ -v --cov=src --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

### Pre-commit Hooks

**Install pre-commit hooks**:
```bash
pip install pre-commit
pre-commit install
```

**Runs before each commit**:
- Linting (ruff, black)
- Type checking (mypy)
- Fast tests (< 5 seconds)

---

## Test Coverage

### Current Coverage

**Overall**: 59 passing tests

**By Module**:
- `src/core/validation/`: 100% (all validators tested)
- `src/core/etl/`: 85% (data fetching, loading)
- `src/strategies/`: 70% (strategy execution)
- `src/core/risk/`: 65% (risk management)
- `src/runners/`: 50% (orchestration)

### Generating Coverage Reports

```bash
# Generate HTML coverage report
python -m pytest tests/ --cov=src --cov-report=html

# View report
open htmlcov/index.html  # macOS
start htmlcov/index.html  # Windows
```

### Coverage Goals

- **Critical modules** (validation, risk): 100%
- **Core logic** (strategies, execution): 90%+
- **Utilities** (helpers, utils): 80%+
- **Overall project**: 85%+

---

## Debugging Tests

### Running Tests with Debugging

```bash
# Print stdout/stderr during tests
python -m pytest tests/ -v -s

# Stop on first failure
python -m pytest tests/ -x

# Drop into debugger on failure
python -m pytest tests/ --pdb

# Verbose output
python -m pytest tests/ -vv
```

### Using pdb in Tests

```python
def test_with_debugging():
    data = create_sample_data()

    import pdb; pdb.set_trace()  # Debugger breakpoint

    result = process_data(data)
    assert result is not None
```

### Logging in Tests

```python
import logging

def test_with_logging(caplog):
    """Test with log capture"""
    with caplog.at_level(logging.DEBUG):
        result = some_function()

    assert "Expected log message" in caplog.text
```

---

## Best Practices

### ✅ DO

- Write tests for all new features
- Use descriptive test names
- Test edge cases and error conditions
- Keep tests independent (no shared state)
- Mock external dependencies (APIs, databases)
- Use fixtures for common setup
- Parameterize tests for multiple inputs
- Add docstrings to test functions
- Run tests before committing
- Maintain high test coverage (85%+)

### ❌ DON'T

- Write tests that depend on external services (without mocking)
- Share state between tests
- Use hardcoded paths (use `tmpdir` fixture)
- Skip tests without good reason
- Ignore failing tests
- Write overly complex tests
- Test implementation details (test behavior)
- Duplicate test logic

---

## Common Test Patterns

### Testing Exceptions

```python
import pytest

def test_raises_error_on_invalid_input():
    """Test that function raises ValueError for invalid input"""
    with pytest.raises(ValueError, match="Invalid input"):
        validate_data(invalid_data)
```

### Testing with Temporary Files

```python
def test_file_writing(tmpdir):
    """Test file writing using tmpdir fixture"""
    file_path = tmpdir.join("test_output.csv")

    # Write data
    df.to_csv(file_path)

    # Read back and verify
    result = pd.read_csv(file_path)
    assert len(result) == len(df)
```

### Testing Async Functions

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    """Test asynchronous function"""
    result = await fetch_data_async()
    assert result is not None
```

### Mocking External APIs

```python
from unittest.mock import Mock, patch

@patch('src.core.etl.data_provider.upstox_provider.requests.get')
def test_api_call(mock_get):
    """Test API call with mocked response"""
    mock_get.return_value.json.return_value = {'data': [...]}

    provider = UpstoxProvider(credentials)
    result = provider.fetch_data('RELIANCE', '2024-01-01', '2024-01-31')

    assert mock_get.called
    assert len(result) > 0
```

---

## Test Maintenance

### Regular Tasks

1. **Weekly**: Review failed tests, update snapshots
2. **Monthly**: Check test coverage, add missing tests
3. **Quarterly**: Refactor slow tests, update fixtures
4. **Per Release**: Run full test suite, update test docs

### Updating Tests

When modifying code:
1. Run affected tests first
2. Update tests to match new behavior
3. Add new tests for new functionality
4. Verify all tests pass
5. Update test documentation if needed

---

## FAQ

### Q: How do I run tests faster?

**A**: Use pytest-xdist for parallel execution:
```bash
pip install pytest-xdist
python -m pytest tests/ -n auto
```

### Q: How do I debug a failing test?

**A**: Use `--pdb` flag to drop into debugger on failure:
```bash
python -m pytest tests/test_example.py::test_function --pdb
```

### Q: How do I skip slow tests during development?

**A**: Mark slow tests with `@pytest.mark.slow` and run:
```bash
python -m pytest -m "not slow"
```

### Q: How do I test only changed code?

**A**: Use pytest-testmon:
```bash
pip install pytest-testmon
python -m pytest --testmon
```

### Q: How do I generate a test report?

**A**: Use pytest-html:
```bash
pip install pytest-html
python -m pytest tests/ --html=report.html
```

---

## Additional Resources

### Documentation
- **pytest documentation**: https://docs.pytest.org/
- **Coverage.py**: https://coverage.readthedocs.io/
- **Python unittest**: https://docs.python.org/3/library/unittest.html

### Related Files
- **CI/CD**: `.github/workflows/ci.yml`
- **Pre-commit**: `.pre-commit-config.yaml`
- **Test fixtures**: `tests/conftest.py`
- **Coverage config**: `.coveragerc` or `pyproject.toml`

### Internal Documentation
- **Validation Framework**: `src/core/validation/README.md` (if exists)
- **Strategy Testing**: `docs/STRATEGY_GUIDE.md`
- **Architecture**: `ARCHITECTURE.md`

---

## Contact

For questions about tests:
- Check existing test files for examples
- Review this README
- Consult `ARCHITECTURE.md` for system design
- Ask in team discussions or open an issue

---

**Last Updated**: October 30, 2025
**Maintainer**: StrategyLab Testing Team
**Next Review**: After Phase 2 completion
