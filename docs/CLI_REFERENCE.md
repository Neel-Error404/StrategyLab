# CLI Reference

This guide explains the available command-line options for the StrategyLab equities backtester.

---

## Core Command

```bash
python src/runners/unified_runner.py [--mode MODE] [arguments]
```

All modes share the same executable. Use `--help` for quick usage:

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

Choose one of the following:

```bash
--template {minimal,conservative,aggressive,portfolio_diversified}
--config path/to/custom_config.yaml
```

Templates live in `config/templates/`. Custom YAML files can be loaded after environment variable substitution (see `config/config_loader.py`).

---

## Data Selection Arguments

| Argument        | Purpose                                             |
|-----------------|-----------------------------------------------------|
| `--date-ranges` | Required for most modes. Format `YYYY-MM-DD_to_YYYY-MM-DD`. Accept multiple ranges. |
| `--dates`       | Alternative to `--date-ranges`; provide individual dates. |
| `--tickers`     | Optional. Overrides pool discovery. Example: `--tickers RELIANCE TCS`. |
| `--strategies`  | Strategy list (defaults to template strategy). Example: `--strategies sma_crossover bollinger_bands`. |

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

Incremental updates keep historical pools current without refetching the full range.

```bash
python src/runners/unified_runner.py --mode update \
  --pool-path data/pools/2024-01-01_to_2024-06-30 \
  --extend-to 2024-08-31 \
  --yes
```

Key flags:

- `--dry-run` – show planned actions without writing files.
- `--validate-only` – run pool integrity checks only.
- `--no-backup` – skip automatic backup (use with caution).
- `--yes` – skip confirmation prompts (useful in CI).

These flags mirror the lower-level interface in `src/core/etl/data_fetcher.py`.

---

## Fetch Mode Examples

```bash
# Interactive fetch (prompts for details)
python src/runners/unified_runner.py --mode fetch

# Explicit parameters
python src/runners/unified_runner.py --mode fetch \
  --date-ranges 2024-01-01_to_2024-01-07 \
  --tickers RELIANCE TCS
```

---

## Template Overview

| Template                | Profile       | Notes                                           |
|-------------------------|---------------|-------------------------------------------------|
| `minimal`               | Learning      | 5% max position, single-thread, verbose logs.   |
| `conservative`          | Low risk      | 15% max position, risk controls enabled.        |
| `aggressive`            | High risk     | 20% max position, multi-threaded.               |
| `portfolio_diversified` | Multi-ticker  | Balanced allocations across diversified basket. |

Use `--template NAME` with any mode to apply the preset.

---

## Validation Utilities

Dedicated modules under `src/core/validation/` provide post-run checks. Run the parity suites as part of release validation:

```bash
.venv\Scripts\python.exe -m pytest tests/test_backtest_live_parity.py tests/test_precision_validation.py -q
```

---

## Troubleshooting Tips

1. Use `--log-level DEBUG` (via template or config) to inspect data issues.
2. Run `python src/core/etl/pool_inspector.py --pool-path <path>` to audit pools before updates.
3. Check environment variables loaded by `config/config_loader.py` when credentials are missing.

---

For deeper guidance see the README, TEMPLATE_GUIDE, and strategylab_v2_phase0_audit.
