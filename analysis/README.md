# StrategyLab Analysis Toolkit

Curated analysis helpers for the V2 equities release. The goal is to keep the
open-source footprint focused on workflows that pair with the parquet data
pipeline and parity/precision validation suites.

## What's Included
- `generic/` – lightweight modules for quick exploratory data analysis and
  equity-level diagnostics using parquet pools.
- `configs/` – starter YAML configs that mirror the unified runner template
  structure (`docs/TEMPLATE_GUIDE.md`) so the same parameters can drive ad-hoc
  analysis runs.
- `config_template.yaml` – reference schema for building bespoke analysis
  manifests.
- `ANALYSIS_PROTOCOL.md` / `METHODOLOGY.md` – background on the validation
  checkpoints we retain from the internal toolkit, pruned to equities-ready
  guidance.

## Usage
```bash
python -m analysis.generic.scripts.01_basic_eda \
  --pool-path data/pools/2024-01-01_to_2024-06-30 \
  --ticker RELIANCE \
  --output-dir outputs/eda
```

The scripts expect parquet pools generated via `--mode update` or
`src/core/etl/data_fetcher.py` and reuse the validation utilities already
shipped in `src/core/validation/`. Keep runs scoped to a local workspace –
outputs are intentionally ignored by `.gitignore`.

## What's Removed
Legacy portfolio construction, learning curricula, and options-specific analysis
have been archived privately. If you need to extend the public toolkit, start
with `generic/` and add new modules under clear directory names so we can keep
the release footprint lean.
