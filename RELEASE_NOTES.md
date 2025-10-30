# StrategyLab Equities V2 Release Notes

## Highlights
- Incremental parquet updates with data validation (`--mode update`).
- Config parity and precision validator modules with accompanying pytest suites.
- Sanitised OSS footprint: options stack and large data pools removed; UTF-8 codebase.
- Environment-aware YAML loader for broker credentials and templates.

## Tested
- `.venv\Scripts\python.exe -m pytest tests/test_backtest_live_parity.py tests/test_precision_validation.py -q`
- Python 3.10 virtual environment on Windows.

## Known Limitations
- Options trading infrastructure intentionally excluded from this release.
- Sample datasets limited to metadata (`data/indian_equities_master.csv`). Supply your own historical candles.
- `optimize` mode remains experimental.

## Upgrade Notes
1. Pull latest `release/strategylab-v2` branch.
2. Install dependencies in a fresh venv: `pip install -r requirements.txt`.
3. Populate `.env` with broker credentials or export environment variables.
4. Use the new update workflow to extend existing pools instead of full re-fetches.

See `docs/strategylab_v2_phase0_audit.md` and `docs/TASKS.md` for full change history.




