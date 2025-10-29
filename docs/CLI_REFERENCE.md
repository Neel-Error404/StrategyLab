# CLI Reference

This guide explains the available command-line options for the StrategyLab equities backtester.

---

## Core Command

```bash
python src/runners/unified_runner.py [--mode MODE] [arguments]
```

Use `--help` for quick usage:

```bash
python src/runners/unified_runner.py --help
```

---

## Modes

| Mode      | Description                                                     |
|-----------|-----------------------------------------------------------------|
| `validate`| Run data quality checks and bias detection only.                 |
| `backtest`| Execute full workflow: backtest + analysis + visualisation.      |
| `analyze` | Run backtest and post-trade analysis (no visual output).         |
| `visualize` | Produce plots/report output (runs backtest if needed).        |
| `fetch`   | Download fresh market data using configured providers.          |
| `update`  | Incrementally extend an existing parquet pool.                  |
| `replay`  | Replay a stored manifest via adapters (advanced use).           |
| `optimize`| Experimental parameter search (requires strategy support).      |

Example:

```bash
python src/runners/unified_runner.py --mode backtest --date-ranges 2024-01-01_to_2024-01-31
```

---

## Configuration Sources

```bash
--template {minimal,conservative,aggressive,portfolio_diversified}
--config path/to/custom_config.yaml
```

Templates live in `config/templates/`. Custom YAML files benefit from environment substitution (see `config/config_loader.py`).

---

## Data Selection Arguments

| Argument        | Purpose                                             |
|-----------------|-----------------------------------------------------|
| `--date-ranges` | Required for most modes. Format `YYYY-MM-DD_to_YYYY-MM-DD`. Accept multiple ranges. |
| `--dates`       | Alternative to `--date-ranges`; provide individual dates. |
| `--tickers`     | Optional override for pool discovery (e.g., `--tickers RELIANCE TCS`). |
| `--strategies`  | Strategy list (defaults to template strategy).        |

When tickers are omitted, the runner auto-discovers symbols from the pool on disk.

---

## Execution Flags

| Argument           | Description                                                |
|--------------------|------------------------------------------------------------|
| `--parallel`       | Enable multiprocessing where supported.                    |
| `--max-workers N`  | Override parallel worker count.                            |
| `--skip-visualization` | Skip plot generation for faster runs.                 |
| `--manifest PATH`  | Required for `--mode replay`; points to a saved manifest.   |

---

## Update Mode

```bash
python src/runners/unified_runner.py --mode update \
  --pool-path data/pools/2024-01-01_to_2024-06-30 \
  --extend-to 2024-08-31 \
  --yes
```

Key flags:
- `--dry-run` – show planned actions without writing files.
- `--validate-only` – run pool integrity checks only.
- `--no-backup` – skip backup creation (use with caution).
- `--yes` – bypass confirmation prompts.

Equivalent low-level command:
```bash
python src/core/etl/data_fetcher.py --mode update --pool-path <path> --extend-to <date>
```

---

## Fetch Mode Examples

```bash
python src/runners/unified_runner.py --mode fetch --date-ranges 2024-01-01_to_2024-01-07 --tickers RELIANCE TCS
```

---

## Template Overview

| Template                | Profile       | Notes                                      |
|-------------------------|---------------|--------------------------------------------|
| `minimal`               | Learning      | 5% max position, single-thread, verbose logs. |
| `conservative`          | Low risk      | 15% max position, risk controls enabled.    |
| `aggressive`            | High risk     | 20% max position, multi-threaded.           |
| `portfolio_diversified` | Multi-ticker  | Balanced allocations across diversified basket. |

---

## Validation Utilities

Run the parity and precision suites before tagging a release:

```bash
.venv\Scripts\python.exe -m pytest tests/test_backtest_live_parity.py tests/test_precision_validation.py -q
```

---

## Troubleshooting Tips

1. Use `--log-level DEBUG` in your config to inspect data flow.
2. Audit pools with `src/core/etl/pool_inspector.py` before running updates.
3. Ensure `.env` variables are loaded by `config/config_loader.py` if authentication fails.

Refer to README and strategylab_v2_phase0_audit for more context.
