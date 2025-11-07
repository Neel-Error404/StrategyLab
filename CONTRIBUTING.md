# Contributing to StrategyLab

Thank you for your interest in contributing to StrategyLab! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of experience level, background, or identity.

### Expected Behavior

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment, trolling, or discriminatory comments
- Personal or political attacks
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

---

## Getting Started

### Ways to Contribute

1. **Report Bugs** - Found a bug? Open an issue
2. **Suggest Features** - Have an idea? We'd love to hear it
3. **Improve Documentation** - Clarity improvements always welcome
4. **Write Code** - Fix bugs or implement features
5. **Add Strategies** - Share your trading strategies
6. **Add Broker Support** - Integrate new data providers

### Good First Issues

Look for issues labeled `good-first-issue` - these are beginner-friendly and a great way to start contributing.

---

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- Virtual environment tool (venv)
- Broker API access (for testing)

### Fork and Clone

```bash
# 1. Fork the repository on GitHub
# Click "Fork" button at https://github.com/Neel-Error404/StrategyLab

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/StrategyLab.git
cd StrategyLab/backtester

# 3. Add upstream remote
git remote add upstream https://github.com/Neel-Error404/StrategyLab.git
```

### Set Up Development Environment

```powershell
# 1. Run automated setup
python setup.py

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate      # Linux/Mac

# 3. Install development dependencies
pip install -r requirements-dev.txt  # If exists

# 4. Configure environment
cp .env.example .env
# Edit .env with your test API credentials

# 5. Verify setup
python scripts/verify_setup.py
```

### Development Workflow

```bash
# 1. Create a feature branch
git checkout -b feature/your-feature-name

# 2. Make your changes
# ... code, test, document ...

# 3. Run tests
pytest tests/ -v

# 4. Commit changes
git add .
git commit -m "feat: add your feature description"

# 5. Push to your fork
git push origin feature/your-feature-name

# 6. Open Pull Request on GitHub
```

---

## How to Contribute

### Reporting Bugs

**Before submitting a bug report:**
1. Check existing issues to avoid duplicates
2. Run `python scripts/verify_setup.py` to rule out setup issues
3. Check [docs/ERROR_REFERENCE.md](docs/ERROR_REFERENCE.md) for known issues

**When reporting a bug, include:**
- Clear, descriptive title
- Steps to reproduce
- Expected vs actual behavior
- Full error traceback
- Environment details:
  ```bash
  python --version
  pip list
  ```
- Minimal code example (if applicable)

**Bug Report Template:**
```markdown
## Description
Brief description of the bug

## Steps to Reproduce
1. Run command X
2. With configuration Y
3. See error Z

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: Windows 10 / Ubuntu 22.04 / macOS 13
- Python: 3.11.0
- StrategyLab version: v2.2

## Additional Context
- Logs: (attach or paste)
- Screenshots: (if applicable)
```

### Suggesting Features

**Feature Request Template:**
```markdown
## Feature Description
Clear description of the proposed feature

## Use Case
Why is this feature needed? What problem does it solve?

## Proposed Solution
How would you implement this?

## Alternatives Considered
What other approaches did you consider?

## Additional Context
- Mockups/diagrams (if applicable)
- Related issues/PRs
```

### Adding a New Strategy

```python
# 1. Create strategy file
# src/strategies/my_new_strategy.py

from strategies.base_strategy import BaseStrategy
import pandas as pd

class MyNewStrategy(BaseStrategy):
    """
    Brief description of your strategy.

    This strategy implements [explain core logic].

    Parameters:
        param1 (int): Description of parameter 1
        param2 (float): Description of parameter 2
    """

    def __init__(self, config):
        super().__init__(config)
        self.param1 = config.get('param1', 10)
        self.param2 = config.get('param2', 0.5)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on data.

        Args:
            data: OHLCV DataFrame with columns [open, high, low, close, volume]

        Returns:
            Series of signals: 'BUY', 'SELL', or 'HOLD'
        """
        # Your logic here
        signals = pd.Series('HOLD', index=data.index)

        # Example: Simple logic
        data['indicator'] = data['close'].rolling(self.param1).mean()
        signals[data['close'] > data['indicator']] = 'BUY'
        signals[data['close'] < data['indicator']] = 'SELL'

        return signals

# 2. Register strategy
# src/strategies/register_strategies.py

from strategies.my_new_strategy import MyNewStrategy

def register_all_strategies():
    StrategyFactory.register_strategy('open_source_baseline', OpenSourceBaselineStrategy)
    StrategyFactory.register_strategy('my_new_strategy', MyNewStrategy)  # Add this

# 3. Add tests
# tests/test_my_new_strategy.py

import pytest
from strategies.my_new_strategy import MyNewStrategy

def test_strategy_initialization():
    config = {'param1': 20}
    strategy = MyNewStrategy(config)
    assert strategy.param1 == 20

def test_signal_generation():
    # Test signal logic
    pass

# 4. Document strategy
# Update docs/STRATEGY_GUIDE.md with your strategy details
```

### Adding Broker Support

To add a new broker/data provider:

```python
# 1. Create provider class
# src/core/etl/data_provider/my_broker_provider.py

from src.core.etl.data_provider.base_provider import BaseDataProvider

class MyBrokerProvider(BaseDataProvider):
    """
    Data provider for MyBroker API.

    API Documentation: [link]
    Rate Limits: [specify]
    """

    def __init__(self, api_key, api_secret):
        super().__init__()
        self.api_key = api_key
        self.api_secret = api_secret

    def fetch_historical_data(self, ticker, start_date, end_date, timeframe):
        """
        Fetch historical OHLCV data.

        Args:
            ticker: Stock/crypto symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Candle size (1minute, 5minute, etc.)

        Returns:
            DataFrame with columns: [timestamp, open, high, low, close, volume]
        """
        # Implement API calls
        pass

    def validate_connection(self):
        """Test API connectivity"""
        pass

# 2. Register provider
# src/core/etl/data_fetcher.py

from src.core.etl.data_provider.my_broker_provider import MyBrokerProvider

PROVIDERS = {
    'upstox': UpstoxProvider,
    'zerodha': ZerodhaProvider,
    'mybroker': MyBrokerProvider,  # Add this
}

# 3. Update documentation
# docs/BROKER_SETUP.md - Add setup instructions
# .env.example - Add API key fields
```

---

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

```python
# Line length: 100 characters (not 79)
# Imports: Organize in groups (stdlib, third-party, local)

# Good
import os
import sys

import pandas as pd
import numpy as np

from src.core.etl import data_fetcher
from src.strategies import BaseStrategy

# Type hints encouraged
def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate percentage returns."""
    return prices.pct_change()

# Docstrings: Google style
def my_function(param1, param2):
    """
    Brief description.

    Args:
        param1 (int): Description of param1
        param2 (str): Description of param2

    Returns:
        bool: Description of return value

    Raises:
        ValueError: When param1 < 0
    """
    pass
```

### Code Quality Tools

```bash
# Format code (optional but recommended)
black src/ tests/

# Check style
flake8 src/ tests/

# Type checking (if using mypy)
mypy src/
```

### Commit Message Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(strategies): add RSI strategy implementation

fix(data_fetcher): handle rate limit errors correctly

docs(README): update installation instructions

test(strategies): add unit tests for SMA crossover
```

---

## Testing Requirements

### Test Organization

```
tests/
├── unit/              # Unit tests (fast, isolated)
├── integration/       # Integration tests
├── strategies/        # Strategy-specific tests
└── fixtures/          # Test data and mocks
```

### Writing Tests

```python
import pytest
import pandas as pd
from datetime import datetime

def test_strategy_signal_generation():
    """Test strategy generates valid signals"""
    # Arrange
    strategy = MyStrategy(config={})
    data = create_sample_data()  # Helper function

    # Act
    signals = strategy.generate_signals(data)

    # Assert
    assert isinstance(signals, pd.Series)
    assert set(signals.unique()).issubset({'BUY', 'SELL', 'HOLD'})
    assert len(signals) == len(data)

@pytest.fixture
def sample_data():
    """Fixture providing sample OHLCV data"""
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
        'open': [100 + i for i in range(100)],
        'high': [101 + i for i in range(100)],
        'low': [99 + i for i in range(100)],
        'close': [100.5 + i for i in range(100)],
        'volume': [1000 for _ in range(100)]
    })
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_my_feature.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run fast tests only
pytest -m "not slow"
```

### Test Requirements

- All new features must include tests
- Bug fixes should include regression tests
- Aim for >80% code coverage on new code
- Tests must pass before PR is merged

---

## Documentation

### Documentation Types

1. **Code Comments** - Explain complex logic
2. **Docstrings** - Document functions/classes
3. **README Updates** - For user-facing changes
4. **Guide Updates** - Update relevant docs/ files
5. **Changelog** - Add entry for notable changes

### Updating Documentation

```markdown
# If you change CLI behavior, update:
- docs/CLI_REFERENCE.md
- QUICKSTART.md (if affects setup)

# If you add a feature, update:
- README.md (feature list)
- docs/GETTING_STARTED.md (if architectural)
- CHANGELOG.md (version notes)

# If you add a strategy, update:
- docs/STRATEGY_GUIDE.md
- Register in src/strategies/register_strategies.py
```

### Documentation Style

- Use clear, concise language
- Include code examples
- Add screenshots/diagrams when helpful
- Link to related documentation
- Keep examples up-to-date with code

---

## Pull Request Process

### Before Submitting

- [ ] Tests pass locally (`pytest`)
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] Commit messages follow convention
- [ ] Branch is up-to-date with `master`
- [ ] No merge conflicts

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing performed

## Checklist
- [ ] Code follows project style guide
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests pass locally

## Screenshots (if applicable)

## Related Issues
Closes #issue_number
```

### Review Process

1. **Automated Checks** - CI/CD runs tests
2. **Code Review** - Maintainer reviews code
3. **Feedback** - Address review comments
4. **Approval** - At least 1 maintainer approval required
5. **Merge** - Maintainer merges PR

### After Merge

- Delete your feature branch
- Update your fork:
  ```bash
  git checkout master
  git pull upstream master
  git push origin master
  ```

---

## Community

### Communication Channels

- **GitHub Issues** - Bug reports, feature requests
- **GitHub Discussions** - Questions, ideas, community chat
- **Pull Requests** - Code contributions

### Getting Help

**For Users:**
- Check [docs/](docs/) folder
- Search existing GitHub issues
- Read [docs/ERROR_REFERENCE.md](docs/ERROR_REFERENCE.md)
- Open a GitHub Discussion

**For Contributors:**
- Comment on the issue you're working on
- Ask questions in PR comments
- Request review when ready

### Recognition

Contributors are recognized in:
- GitHub contributor graph
- Release notes (for significant contributions)
- CHANGELOG.md credits

---

## Release Process (Maintainers Only)

1. Update version in `__init__.py`
2. Update CHANGELOG.md
3. Create release tag
4. Build and publish (if applicable)
5. Announce release

---

## Questions?

If you have questions about contributing:

1. Check this guide
2. Search existing issues/discussions
3. Open a GitHub Discussion
4. Tag a maintainer in your question

---

Thank you for contributing to StrategyLab! 🚀

Every contribution, no matter how small, helps make algorithmic trading more accessible to everyone.

---

*Last updated: 2025-01-07*
