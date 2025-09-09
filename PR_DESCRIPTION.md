Title: Enhance CLI controls; add analysis/scripts; harden .gitignore; cleanup

Summary:
- Adds CLI enhancements for execution control: `--max-workers`, `--skip-validation`.
- Includes public-safe utilities under `src/analysis/**` and `src/scripts/**`.
- Hardens `.gitignore` (imports/archives, private strategies) and removes tracked bytecode.
- Keeps private strategies and data tools out of the public repo.

Motivation:
- Improve ergonomics and repo hygiene without changing private/live systems.

Changes by module:
- CLI/Orchestration:
  - Update `src/runners/cli_handler.py` to support `--max-workers` and `--skip-validation`.
- Strategy:
  - No new public strategies added. `register_strategies.py` remains compatible.
- Analysis & Scripts:
  - Add `src/analysis/**` (performance and research) and `src/scripts/**` (comparison, monitoring).
- Docs:
  - No private docs added.
- Repo hygiene:
  - Harden `.gitignore` to exclude imports/archives and private strategy files.
  - Remove tracked `__pycache__/` artifacts.

Security:
- No secrets in code. Token files and artifacts remain ignored by `.gitignore`.

Testing/Validation:
- Syntax compile: `py -m py_compile` on modified/new modules (passed).
- Examples:
  - Validate: `py src/runners/unified_runner.py --mode validate --date-ranges 2024-12-12_to_2025-06-09`
  - Backtest: `py src/runners/unified_runner.py --mode backtest --date-ranges 2024-12-12_to_2025-06-09 --strategies sma_crossover --parallel --max-workers 4 --skip-validation`

Follow-ups:
- Keep private strategies and data tools in a separate private repo.
