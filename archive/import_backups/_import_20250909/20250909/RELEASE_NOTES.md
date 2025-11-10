# Strategy Lab - Release Notes

## v0.06 (September 2025) — Code + Config Alignment ✅

Focused on source-of-truth modules that generate trades (code and config). No analysis/reporting changes are listed here by design.

### Runners (src/runners)
- Unified entrypoint: `src/runners/unified_runner.py` orchestrates modes `backtest|analyze|visualize|validate|fetch`.
- CLI separation: `src/runners/cli_handler.py` and `src/runners/cli/argument_parser.py` handle args and validation.
- Auto-discovery: When `--tickers` not provided, discovers from `data/pools/{date_range}/1minute/*.csv`.
- Output structure: `{timestamp}/{strategy}/{date_range}` via `src/runners/utils/naming.py#create_monolith_directory_structure`.

### Strategies (src/strategies)
- Registry: `src/strategies/register_strategies.py` registers core strategies.
- Implementations: `strategy_mse.py`, `strategy_sma_crossover.py`, `strategy_bollinger_bands.py` (imported in `__init__.py`).

### Configuration (config)
- Core config: `config/unified_config.py` dataclasses used across runners.
- Templates: YAML templates under `config/templates/` (minimal, conservative, aggressive, options, portfolio_diversified).
- CSVs: `complete.csv`, `ticker_config.csv`, `zerodha_instruments.csv` included for code-driven configuration.

### Requirements & Compatibility
- Python 3.10+; recommend virtualenv (`python -m venv .venv && source .venv/bin/activate`).
- Curated backups exclude `config/access_tokens/**` and generated data.

### Notes
- No analysis or reporting changes included in this release section.
- Documentation updated to align with current code paths and Python version.

## v0.05 (June 2025) — Initial Public Release 🧪

Summary of initial capabilities and architecture (superseded by v0.06 for code/config alignment).
