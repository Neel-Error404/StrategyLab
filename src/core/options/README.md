# Options Backtesting Module

**Status**: Phase 4 replay complete (production hybrid pricing)
**Version**: 0.4.0
**Last Updated**: 2025-10-11

---

## Overview
The options stack extends the equity backtester with a fully validated replay engine, hybrid pricing that blends actual Upstox chains with synthetic Black–Scholes fills, and reporting manifests that prove out multi-ticker production runs. Phase 3 and Phase 4 completion reports document the throughput, data quality, and profitability delivered by the replay engine across RELIANCE, TCS, INFY, NIFTY, and BANKNIFTY contracts.

Key capabilities include:
- Automated data ingestion from Upstox with cached parquet storage, metadata tracking, and WiFinance parity fallbacks where the daily equity feed is still pending.
- Pricing infrastructure that calibrates implied volatility, compares synthetic vs actual fills, and recommends the correct mode (synthetic, actual, or hybrid) before replaying trades.
- A replay engine that maps equity signals into option trades, enforces risk rules, and produces CSV/JSON artifacts alongside structured logs.

---

## Feature Highlights

### 1. Data & Validation Pipeline
- `validation/data_fetcher.py` fetches historical option chains via Upstox, writes deterministic parquet layouts, and produces JSON summaries of coverage.
- `validation/pricing_validator.py` benchmarks synthetic, actual, and hybrid pricing accuracy to generate production recommendations.
- `validation/data_storage.py` and `validation/README.md` capture the on-disk schema, troubleshooting flows, and storage guarantees.

### 2. Pricing & Greeks
- `options_engine.py` implements the Black–Scholes engine with full Greeks for synthetic fills.
- `replay/pricing.py` orchestrates hybrid pricing, selecting actual OHLC data where available and falling back to synthetic outputs otherwise.
- `pricing/synthetic_engine.py` and `pricing/volatility_models.py` package volatility calibration and reusable pricing helpers.

### 3. Replay & Risk Management
- `replay/engine.py` drives the end-to-end replay, coordinating data loading, trade mapping, pricing, and artifact emission.
- `replay/data_loader.py` discovers parquet/CSV sources across ticker-first layouts, builds option data stores, and aligns equity + option timeframes.
- `replay/risk.py` enforces position sizing, exposure caps, and per-trade guardrails before outputting option executions.
- `replay/metrics.py` aggregates trade-level metrics, Sharpe/Drawdown stats, and pricing diagnostics for reports.

### 4. Reporting & Manifests
- Phase completion documents (`PHASE3_COMPLETE.md`, `PHASE4_COMPLETE.md`) summarise KPIs, fallback ratios, and timeline traces.
- `VALIDATION_REPORT.md`, `PHASE1_STATUS.md`, and `PHASE2_READINESS_ASSESSMENT.md` log historical validation decisions, data dependencies, and open risks.

---

## Quick Start Workflow

1. **Fetch a validation dataset**
   ```bash
   python src/core/options/validation/data_fetcher.py \
     --ticker RELIANCE \
     --timeframe 1day \
     --max-expiries 1 \
     --log-level INFO
   ```
   This writes parquet chains under `data/pools/options/<date_range>/<ticker>/<timeframe>/` and prints a coverage summary.

2. **Run pricing validation**
   ```bash
   python - <<'PY'
   from src.core.options.validation.pricing_validator import run_pricing_validation
   result = run_pricing_validation()
   print(result["summary"]["recommendation"])
   PY
   ```
   The validator emits aggregated CSV/JSON reports in `src/core/options/data/validation_results/` and recommends the correct pricing mode for replay.

3. **Execute the replay engine**
   ```bash
   python run_phase4_backtest.py
   ```
   The Phase 4 runner loads sample equity trades, launches the replay engine across five tickers, and saves metrics, trades, positions, manifests, and structured logs under `outputs/phase4_backtest/options_replay/`.

---

## Directory Structure

```
src/core/options/
├── PHASE1_STATUS.md / PHASE2_READINESS_ASSESSMENT.md / PHASE3_COMPLETE.md / PHASE4_COMPLETE.md
│   └── Status reports, validation outcomes, and production sign-off
├── README.md                      # (this file)
├── options_engine.py              # Black–Scholes pricing + Greeks
├── config/
│   └── options_config.yaml        # Master configuration for replay runs
├── validation/
│   ├── README.md                  # Operational runbook for data + validation
│   ├── config_loader.py           # Loads validation YAMLs and CLI overrides
│   ├── data_fetcher.py            # Upstox fetcher with caching + summaries
│   ├── data_storage.py            # Parquet persistence and metadata helpers
│   ├── pricing_validator.py       # Hybrid vs actual vs synthetic evaluation
│   └── upstox_options_api.py      # API bindings and rate-limit helpers
├── pricing/
│   ├── synthetic_engine.py        # Synthetic pricing utilities
│   └── volatility_models.py       # Volatility estimators (historical, EWMA)
├── replay/
│   ├── config.py                  # Dataclasses + YAML loader for replay config
│   ├── data_loader.py             # Discovers equity + option data stores
│   ├── engine.py                  # Core orchestrator (multiprocessing aware)
│   ├── metrics.py                 # Portfolio-level reporting helpers
│   ├── pricing.py                 # Hybrid pricing wrapper used during replay
│   ├── risk.py                    # Position sizing + guardrails
│   └── trade_mapper.py            # Equity-to-option mapping logic
├── data/
│   ├── lot_sizes.csv              # Contract lot metadata
│   ├── schemas.py                 # Schema helpers for saved artefacts
│   └── validation_results/        # Pricing validation outputs (CSV/Parquet)
└── planning/ / config/ etc.       # Historical planning docs and templates
```

---

## Operational Notes & Caveats
- **Equity reference feed**: WiFinance daily reference prices are still pending; manual overrides remain in place until the upstream feed is restored (see `PHASE1_STATUS.md`).
- **Data footprint**: Ensure the equity parquet pool (`data/pools/2022-01-01_to_2025-08-31/<ticker>/`) is available before running replay or validation jobs.
- **Hybrid pricing**: `options_config.yaml` defaults to `mode: "hybrid"`; adjust to `synthetic` or `actual` only if validation recommends it.
- **Testing**: `test_phase3_integration.py` smoke-tests the replay engine against staged parquet fixtures, while `run_phase4_backtest.py` performs the full multi-ticker validation run.

---

## Next Steps
- Wire `run_phase4_backtest.py` into CI once the data pool is mirrored in non-production environments.
- Expand coverage beyond the initial five tickers by reusing the validation fetcher and updating `options_config.yaml` weightings.
- Integrate options analytics with the generic analysis framework so equity and option trade diagnostics share the same reporting surface.
