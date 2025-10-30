# Strategy Guide

How to build and extend equities strategies within StrategyLab V2.

---

## Lifecycle

1. Configuration loaded (template or custom YAML).
2. Market data fetched/updated into parquet pools.
3. Strategies run via `src/runners/unified_runner.py`.
4. Results saved under `outputs/<timestamp>/`.

---

## Creating a Strategy

```bash
cp src/strategies/strategy_template.py src/strategies/strategy_my_alpha.py
```

Implement the signal logic:

```python
from src.strategies.strategy_base import StrategyBase

class MyAlphaStrategy(StrategyBase):
    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.name = "my_alpha"

    def generate_signals(self, data, current_time, ticker):
        df = data[ticker]
        if df is None or df.empty:
            return None
        latest = df.iloc[-1]
        sma_fast = df.close.rolling(10).mean().iloc[-1]
        if latest.close > sma_fast:
            return {"action": "BUY", "confidence": 0.6}
        return None
```

Register it:

```python
from .strategy_my_alpha import MyAlphaStrategy

STRATEGY_REGISTRY = {
    # ...
    "my_alpha": MyAlphaStrategy,
}
```

Run:
```bash
python src/runners/unified_runner.py --mode backtest --strategies my_alpha --date-ranges 2024-03-01_to_2024-03-31
```

---

## Data & Indicators

- Incoming data is a dict of DataFrames keyed by ticker.
- Cache computed indicators to avoid duplication.
- Precompute common factors via helpers in `src/core/analysis/` if needed.

---

## Risk & Costs

- Risk manager and transaction costs configured via YAML.
- Post-signal risk enforcement occurs in `TaskExecutor`.
- Tune parameters like `max_drawdown`, `stop_loss_pct`, `commission_pct` in templates.

---

## Validation

Ensure backtest and live parity before deployment:

```bash
python src/runners/unified_runner.py --mode validate --dates 2024-01-01
.venv\Scripts\python.exe -m pytest tests/test_backtest_live_parity.py -q
```

---

## Best Practices

1. Keep strategy logic pure (no direct I/O).
2. Expose parameters via `parameters` dict for experimentation.
3. Add unit tests under `tests/` to cover edge cases.
4. Run validation suites before tagging releases.

For architectural context, review `docs/strategylab_v2_phase0_audit.md`.
