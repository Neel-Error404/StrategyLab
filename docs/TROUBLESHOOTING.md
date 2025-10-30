# Troubleshooting Guide

Common issues and resolutions for the StrategyLab equities backtester.

---

## Quick Checklist

1. Activate environment: `.venv\Scripts\activate`.
2. Run validation: `python src/runners/unified_runner.py --mode validate --dates 2024-01-01`.
3. Inspect logs under `.runtime/logs/` or `outputs/<run>/`.

---

## Frequent Issues

### Module Not Found
- Install dependencies: `pip install -r requirements.txt`.
- Confirm you are in `backtester/` directory.
- Activate correct Python interpreter.

### No Data Found
- list `data/pools/` to ensure parquet files exist.
- run `src/core/etl/pool_inspector.py --pool-path ...`.
- specify tickers explicitly if pools are sparse.

### Strategy Not Registered
- add strategy to `STRATEGY_REGISTRY`.
- ensure module import path is correct.

### Authentication Errors
- populate `.env` with broker keys.
- refresh tokens per `docs/BROKER_SETUP.md`.

### Update Workflow Fails
- run with `--dry-run` first.
- inspect gaps (`gap_calculator.py`).
- check for conflicting backups (`*.bak`).

### Null Byte / Encoding Errors
- convert files to UTF-8:
  ```bash
  python - <<'PY'
  from pathlib import Path
  for path in Path('src').rglob('*.py'):
      data = path.read_bytes()
      if b'\x00' in data:
          path.write_text(data.decode('utf-16'), encoding='utf-8')
  PY
  ```

---

## Helpful Commands

```bash
python src/core/etl/pool_inspector.py --pool-path data/pools/2024-01-01_to_2024-06-30
python src/core/etl/gap_calculator.py --pool-path data/pools/2024-01-01_to_2024-06-30
python src/core/etl/data_fetcher.py --mode update --pool-path ... --dry-run
```

---

## Reporting Issues

Include:
- Python version and OS.
- Command executed with full output.
- Validation/pytest results.
- Description of recent changes (e.g., ran update, modified template).

Refer to `docs/strategylab_v2_phase0_audit.md` for project history.
