# Contributing to StrategyLab

Thank you for your interest in contributing to StrategyLab! This document provides guidelines and instructions for contributing to the project.

**Table of Contents**:
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of experience level, background, or identity.

### Expected Behavior

- ✅ Be respectful and considerate in communication
- ✅ Welcome newcomers and help them get started
- ✅ Focus on constructive feedback
- ✅ Accept criticism gracefully
- ✅ Prioritize what's best for the community

### Unacceptable Behavior

- ❌ Harassment, discrimination, or trolling
- ❌ Publishing others' private information
- ❌ Spam or promotional content
- ❌ Disruptive or inappropriate communication

**Enforcement**: Violations may result in temporary or permanent ban from the project.

---

## Getting Started

### Prerequisites

- **Python**: 3.10 or higher
- **Git**: For version control
- **Virtual Environment**: Recommended for isolated development
- **Operating System**: Windows, macOS, or Linux

### Development Setup

```bash
# 1. Fork the repository on GitHub
#    Click "Fork" at https://github.com/Neel-Error404/StrategyLab

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/StrategyLab.git
cd StrategyLab

# 3. Add upstream remote
git remote add upstream https://github.com/Neel-Error404/StrategyLab.git

# 4. Create virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate

# macOS/Linux:
source .venv/bin/activate

# 5. Install dependencies
pip install -r requirements.txt

# 6. Install development dependencies
pip install pytest pytest-cov black ruff mypy pre-commit

# 7. Set up pre-commit hooks
pre-commit install

# 8. Verify installation
python -m pytest tests/ -v
```

### Environment Configuration

Create `.env` file for local development:

```bash
# Copy example (if exists)
cp .env.example .env

# Or create manually
cat > .env << EOF
# Broker API Credentials (optional for development)
UPSTOX_CLIENT_ID=your_client_id_here
UPSTOX_CLIENT_SECRET=your_secret_here
ZERODHA_API_KEY=your_api_key_here
ZERODHA_API_SECRET=your_secret_here
EOF
```

**Important**: Never commit `.env` files or API credentials!

---

## Development Workflow

### 1. Create a Feature Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-123-description
```

**Branch Naming Convention**:
- `feature/` - New features (e.g., `feature/add-rsi-strategy`)
- `fix/` - Bug fixes (e.g., `fix/validation-error`)
- `docs/` - Documentation only (e.g., `docs/update-readme`)
- `refactor/` - Code refactoring (e.g., `refactor/cleanup-etl`)
- `test/` - Test additions (e.g., `test/add-strategy-tests`)

### 2. Make Your Changes

```bash
# Make changes to code
# Write tests for new functionality
# Update documentation

# Check status
git status

# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add RSI strategy with configurable periods"
```

**Commit Message Format**:
```
<type>: <short description>

<optional detailed description>

<optional footer: references, breaking changes>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples**:
```bash
git commit -m "feat: add RSI strategy with 14-period default"
git commit -m "fix: resolve precision validation error for small quantities"
git commit -m "docs: update CONTRIBUTING.md with PR guidelines"
git commit -m "test: add unit tests for signal parity validator"
```

### 3. Keep Your Branch Updated

```bash
# Fetch latest changes
git fetch upstream

# Rebase on main
git rebase upstream/main

# Or merge (if rebase is problematic)
git merge upstream/main

# Resolve conflicts if any
# Then push to your fork
git push origin feature/your-feature-name
```

### 4. Run Tests Locally

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_precision_validation.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Verify coverage report
open htmlcov/index.html  # macOS
start htmlcov/index.html  # Windows
```

### 5. Check Code Quality

```bash
# Format code with Black
black src/ tests/

# Lint with Ruff
ruff check src/ tests/

# Type check with MyPy
mypy src/

# Run all pre-commit checks
pre-commit run --all-files
```

### 6. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name

# Go to GitHub and create Pull Request
# https://github.com/YOUR_USERNAME/StrategyLab/pulls
```

---

## Code Standards

### Python Style Guide

We follow **PEP 8** with some modifications:

- **Line length**: 100 characters (not 79)
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Double quotes for strings
- **Imports**: Organized (stdlib, third-party, local)

### Code Formatting

We use **Black** for automatic formatting:

```bash
# Format all code
black src/ tests/

# Check without modifying
black src/ tests/ --check
```

**Black Configuration** (in `pyproject.toml`):
```toml
[tool.black]
line-length = 100
target-version = ['py310']
include = '\.pyi?$'
```

### Linting

We use **Ruff** for fast linting:

```bash
# Lint all code
ruff check src/ tests/

# Auto-fix issues
ruff check src/ tests/ --fix
```

**Ruff Configuration** (in `pyproject.toml`):
```toml
[tool.ruff]
line-length = 100
select = ["E", "F", "W", "I", "N"]
ignore = ["E501"]  # Line too long (handled by Black)
```

### Type Hints

We encourage type hints for better code clarity:

```python
# Good
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio from returns series."""
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std()

# Avoid (no type hints)
def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std()
```

**Type Checking**:
```bash
mypy src/
```

### Docstrings

Use **Google-style** docstrings:

```python
def validate_price(price: float) -> Decimal:
    """Validate and round price to exchange precision.

    Args:
        price: Raw price value to validate

    Returns:
        Decimal: Rounded price to 4 decimal places

    Raises:
        PrecisionError: If price is negative or exceeds limits

    Example:
        >>> validate_price(123.456789)
        Decimal('123.4568')
    """
    if price < MIN_PRICE or price > MAX_PRICE:
        raise PrecisionError(f"Price {price} out of range")

    return round_to_precision(price, PRICE_DECIMALS)
```

---

## Testing Guidelines

### Test Requirements

- **All new features** must include tests
- **Bug fixes** should include regression tests
- **Coverage** should not decrease (aim for 85%+)
- **Tests must pass** before PR approval

### Writing Tests

**Location**: `tests/`

**Naming Convention**:
```
test_<module_name>.py          # Unit tests
test_<feature>_integration.py  # Integration tests
test_<component>_e2e.py         # End-to-end tests
```

**Test Structure** (AAA Pattern):
```python
def test_sharpe_ratio_positive_returns():
    """Test that Sharpe ratio is positive for positive returns."""
    # ARRANGE
    returns = pd.Series([0.01, 0.02, 0.01, 0.03])

    # ACT
    sharpe = calculate_sharpe_ratio(returns)

    # ASSERT
    assert sharpe > 0
    assert isinstance(sharpe, float)
```

**Using Fixtures**:
```python
import pytest

@pytest.fixture
def sample_ohlcv_data():
    """Fixture providing sample OHLCV data."""
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='5min'),
        'open': 100 + np.random.randn(100),
        'high': 102 + np.random.randn(100),
        'low': 98 + np.random.randn(100),
        'close': 100 + np.random.randn(100),
        'volume': 1000 + np.random.randint(-100, 100, 100)
    })

def test_strategy_with_sample_data(sample_ohlcv_data):
    """Test strategy with fixture data."""
    strategy = MSEStrategy(config)
    signals = strategy.generate_signals(sample_ohlcv_data)
    assert len(signals) == len(sample_ohlcv_data)
```

**Running Tests**:
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_precision_validation.py::test_price_rounds_to_4_decimals -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing
```

See [`tests/README.md`](tests/README.md) for complete testing guide.

---

## Documentation

### Documentation Requirements

- **Code changes** require corresponding doc updates
- **New features** need usage examples
- **API changes** must update API documentation
- **README** should reflect major changes

### Documentation Structure

```
docs/
├── SETUP_GUIDE.md          # Installation and setup
├── BROKER_SETUP.md         # Broker API configuration
├── STRATEGY_GUIDE.md       # Custom strategy development
├── TEMPLATE_GUIDE.md       # Configuration templates
├── CLI_REFERENCE.md        # Command-line interface
├── OUTPUT_GUIDE.md         # Understanding results
├── TROUBLESHOOTING.md      # Common issues
└── CHANGELOG.md            # Version history

Root level:
├── README.md               # Project overview
├── FEATURES.md             # Feature catalog
├── RELEASES.md             # Version history
├── ARCHITECTURE.md         # System architecture
├── CONTRIBUTING.md         # This file
└── LICENSE                 # MIT License
```

### Writing Documentation

**Markdown Style**:
- Use headers hierarchically (`#`, `##`, `###`)
- Include code examples with syntax highlighting
- Add table of contents for long documents
- Use tables for comparison/reference
- Include links to related documentation

**Code Examples**:
````markdown
```python
# Example: Creating a custom strategy
from src.strategies.strategy_base import StrategyBase

class MyStrategy(StrategyBase):
    def generate_signals(self, data):
        # Your logic here
        return signals
```
````

**Updating Changelog**:

Add entries to `docs/CHANGELOG.md` for significant changes:

```markdown
## [Unreleased]

### Added
- RSI strategy with configurable periods
- Support for custom indicator functions

### Fixed
- Precision validation error for small quantities

### Changed
- Updated default warmup period to 600 minutes
```

---

## Pull Request Process

### Before Submitting

**Checklist**:
- ✅ Code follows style guidelines (Black, Ruff)
- ✅ All tests pass (`pytest tests/ -v`)
- ✅ New tests added for new functionality
- ✅ Documentation updated
- ✅ Commit messages follow convention
- ✅ Branch is up-to-date with main
- ✅ No merge conflicts

### PR Title and Description

**Title Format**:
```
<type>: <short description>
```

**Example**:
```
feat: add RSI strategy with configurable periods
```

**Description Template**:
```markdown
## Description
Brief description of changes and motivation.

## Changes Made
- Added RSI strategy class in src/strategies/strategy_rsi.py
- Registered strategy in register_strategies.py
- Added unit tests in tests/test_strategy_rsi.py
- Updated STRATEGY_GUIDE.md with RSI example

## Testing
- [ ] All existing tests pass
- [ ] New tests added
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes (or documented if yes)

## Related Issues
Closes #123
```

### PR Review Process

1. **Automated Checks**: CI/CD runs tests, linting, type checking
2. **Code Review**: Maintainer reviews code quality and design
3. **Feedback**: Address review comments, update PR
4. **Approval**: Maintainer approves PR
5. **Merge**: PR is merged into main branch

**Review Timeline**:
- **Initial response**: Within 48 hours
- **Full review**: Within 1 week
- **Merge**: After approval and passing checks

### Responding to Feedback

```bash
# Make changes based on feedback
# Stage and commit
git add .
git commit -m "refactor: address PR feedback - improve error handling"

# Push to update PR
git push origin feature/your-feature-name
```

---

## Issue Guidelines

### Reporting Bugs

**Use the bug report template**:

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce:
1. Run command '...'
2. See error '...'

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened.

**Environment**
- OS: [Windows/macOS/Linux]
- Python version: [e.g., 3.10]
- StrategyLab version: [e.g., v2.0]

**Additional context**
Error messages, logs, screenshots.
```

### Requesting Features

**Use the feature request template**:

```markdown
**Is your feature request related to a problem?**
Describe the problem.

**Describe the solution you'd like**
Clear description of desired functionality.

**Describe alternatives you've considered**
Other solutions you've thought about.

**Additional context**
Use cases, examples, mockups.
```

### Issue Labels

| Label | Description |
|-------|-------------|
| `bug` | Something isn't working |
| `enhancement` | New feature request |
| `documentation` | Documentation improvements |
| `good first issue` | Good for newcomers |
| `help wanted` | Extra attention needed |
| `question` | Further information requested |
| `wontfix` | Will not be worked on |
| `duplicate` | Duplicate of existing issue |

---

## Community

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, ideas, general discussion
- **Pull Requests**: Code contributions

### Getting Help

**For questions**:
1. Check existing documentation
2. Search GitHub Issues
3. Ask in GitHub Discussions
4. Open a new issue with `question` label

**For bugs**:
1. Verify it's a bug (check docs, search issues)
2. Create minimal reproducible example
3. Open bug report with all details

### Recognition

Contributors are recognized:
- Listed in `CONTRIBUTORS.md` (if exists)
- Mentioned in release notes
- GitHub contribution graph
- Acknowledgment in relevant documentation

---

## Development Resources

### Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for:
- System overview
- Component details
- Design patterns
- Extension points

### Code Examples

**Adding a new strategy**:
```python
# src/strategies/strategy_rsi.py
from src.strategies.strategy_base import StrategyBase
import pandas as pd

class RSIStrategy(StrategyBase):
    """RSI-based mean reversion strategy."""

    def __init__(self, config):
        super().__init__(config)
        self.period = config.get('rsi_period', 14)
        self.oversold = config.get('oversold', 30)
        self.overbought = config.get('overbought', 70)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate RSI-based trading signals."""
        rsi = self.calculate_rsi(data['close'], self.period)

        signals = pd.Series(0, index=data.index)
        signals[rsi < self.oversold] = 1    # Buy
        signals[rsi > self.overbought] = -1  # Sell

        return signals

    def calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
```

**Register the strategy**:
```python
# src/strategies/register_strategies.py
from src.strategies.strategy_rsi import RSIStrategy

STRATEGIES = {
    'mse': MSEStrategy,
    'sma': SMAStrategy,
    'rsi': RSIStrategy,  # Add new strategy
}
```

**Add tests**:
```python
# tests/test_strategy_rsi.py
import pytest
import pandas as pd
from src.strategies.strategy_rsi import RSIStrategy

def test_rsi_oversold_signal():
    """Test that RSI generates buy signal when oversold."""
    config = {'rsi_period': 14, 'oversold': 30}
    strategy = RSIStrategy(config)

    # Create data with oversold RSI
    data = create_oversold_data()

    signals = strategy.generate_signals(data)
    assert signals.iloc[-1] == 1  # Buy signal
```

---

## License

By contributing to StrategyLab, you agree that your contributions will be licensed under the MIT License.

See [`LICENSE`](LICENSE) for full license text.

---

## Questions?

- **Documentation**: Check [README.md](README.md), [ARCHITECTURE.md](ARCHITECTURE.md), [`docs/`](docs/)
- **Issues**: Search or create GitHub issue
- **Discussions**: Ask in GitHub Discussions

---

**Thank you for contributing to StrategyLab!** 🎉

Every contribution, no matter how small, helps make StrategyLab better for everyone.

---

**Last Updated**: October 30, 2025
**Maintainer**: StrategyLab Development Team
