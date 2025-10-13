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

## Hypothesis Lifecycle & Trade Simulation

1. **Frame the options hypothesis**
   - Start with a validated equity thesis (e.g., "trend-following longs should be hedged with near-term calls").
   - Express the replay assumptions in `config/options_config.yaml`—inputs, strike selection, risk guardrails, and logging—so the engine knows which trades, data pools, and pricing mode to use.【F:src/core/options/config/options_config.yaml†L1-L120】【F:src/core/options/replay/config.py†L20-L118】

2. **Acquire and verify the option surface**
   - Run the validation fetcher to build an options parquet lake and inspect the JSON summary for missing expiries or strikes.【F:src/core/options/validation/data_fetcher.py†L32-L190】
   - Execute `pricing_validator.py` to calibrate implied volatility, compare synthetic vs actual fills, and document the recommended pricing mode before running any replay experiments.【F:src/core/options/validation/pricing_validator.py†L63-L348】

3. **Map equity trades to option contracts**
   - The replay engine loads merged equity trades, aligns them with the underlying OHLCV cache, and chooses expiries/strikes that satisfy the configured DTE and ATM requirements.【F:src/core/options/replay/data_loader.py†L51-L258】【F:src/core/options/replay/trade_mapper.py†L1-L74】
   - Metadata describing strike source, expiry type, and pricing context is captured alongside each mapped trade, ensuring downstream attribution.

4. **Synthesize option executions**
   - Hybrid pricing evaluates each contract path-by-path, blending actual Upstox bars with synthetic Black–Scholes fills when intraday data is missing, and records every pricing decision in structured logs.【F:src/core/options/replay/pricing.py†L52-L238】【F:src/core/options/replay/engine.py†L1-L188】
   - The risk manager enforces allocation, concurrency, and kill-switch rules before confirming or rejecting entries, so hypotheses are judged under real capital constraints.【F:src/core/options/replay/risk.py†L22-L192】

5. **Generate trade analytics and manifests**
   - Replay metrics aggregate realized P&L, Greeks exposure, fallback ratios, and pricing variance into CSV/JSON outputs for inspection or import into the analysis framework.【F:src/core/options/replay/metrics.py†L19-L162】
   - Completion manifests (`PHASE3_COMPLETE.md`, `PHASE4_COMPLETE.md`) link each run to validation evidence, allowing reviewers to trace assumptions back to data quality checks.

6. **Feed results into portfolio or risk review**
   - Use `analysis/generic` modules (e.g., cascade analysis, stop-loss sweeps) on the replay output to compare option-adjusted outcomes against the base equity performance.
   - Iterate on the hypothesis by editing the YAML config and repeating the cycle; structured logs and run directories provide reproducible artefacts for each revision.

---

## Quality Gates & Tests

- **Unit safeguards**: `tests/options/replay/test_risk_manager.py` validates allocation and kill-switch behaviour so oversized trades are blocked before execution.【F:tests/options/replay/test_risk_manager.py†L1-L36】
- **Data store fallbacks**: `tests/options/replay/test_phase4_behaviour.py::test_option_data_store_intraday_fallback` proves the loader falls back to daily bars when minute data is absent, ensuring continuity across sparse expiries.【F:tests/options/replay/test_phase4_behaviour.py†L32-L89】
- **Replay ordering**: `tests/options/replay/test_phase4_behaviour.py::test_replay_engine_multi_ticker_ordering` exercises the engine’s scheduling logic so mixed ticker portfolios process deterministically before pricing.【F:tests/options/replay/test_phase4_behaviour.py†L91-L170】
- **Integration smoke test**: `test_phase3_integration.py` and `run_phase4_backtest.py` execute the entire replay stack against staged parquet pools, generating manifests and logs that mirror production validation.【F:test_phase3_integration.py†L1-L210】【F:run_phase4_backtest.py†L1-L153】

These checks make the replay deterministic, reproducible, and regression-friendly when expanding coverage to new instruments or strategies.

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
- Restore the WiFinance equity reference feed (see `PHASE1_STATUS.md`) and remove temporary overrides once parity reports confirm accuracy.【F:src/core/options/PHASE1_STATUS.md†L34-L57】
