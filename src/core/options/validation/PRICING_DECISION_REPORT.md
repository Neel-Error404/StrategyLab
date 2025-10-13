# Pricing Validation Decision Report – RELIANCE (Phase 2)

**Run date**: 2025-10-09  
**Validator**: `python -m src.core.options.validation.pricing_validator` (single-ticker run: RELIANCE)

---

## 1. Context
- **Objective**: Decide the production pricing mode (synthetic vs actual vs hybrid) based on Phase 2 validation criteria in `planning/implementation_plan.md` and `validation_config.yaml`.
- **Data windows**:
  - Underlying: `data/pools/2022-01-01_to_2025-08-31/RELIANCE/15m.parquet` → resampled to daily OHLC.
  - Options: `data/pools/options/2025-04-01_to_2025-10-08/RELIANCE/1day/*.parquet` (11 expiries, partially truncated after 2025-08-31 when equity coverage ends).
- **Synthetic models evaluated**: BS + historical vol (20d, 5d), BS + Parkinson, BS + calibrated IV (ATM fit + linear skew).
- **Config alignment**: All model parameters and segmentation bins sourced from `validation_config.yaml`; volatility floors/caps from `options_config.yaml`.

---

## 2. Aggregate Model Performance

| Model | MAE (₹) | RMSE (₹) | MAPE (%) | Median Abs % Error (%) | Bias (% of price) | ATM Median Abs % (%) | Sample (rows) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **BS + Calibrated IV** | **2.95** | **8.24** | **36.94** | **11.28** | **-18.67** | **5.61** | 23,314 |
| BS + 20-Day Historical Vol | 4.43 | 10.94 | 41.12 | 25.22 | -28.19 | 12.86 | 23,314 |
| BS + Parkinson Volatility | 4.60 | 11.13 | 43.77 | 32.15 | -40.62 | 16.27 | 23,314 |
| BS + 5-Day Historical Vol | 6.00 | 12.16 | 52.07 | 43.46 | -27.79 | 28.27 | 23,314 |

**Segmentation (Calibrated IV)**
- **Moneyness**: ATM/ITM median abs error ≤ 5.6% / 2.6%; Deep OTM contracts remain highly unstable (≈99%) due to penny-priced options.
- **DTE**: 31–60 day bucket median ≈8.1%; <8 DTE ≈38.6% (daily sampling underestimates time decay).
- **Volatility regimes**: Medium-vol bucket (<25% σ) dominates coverage with ≈9.7% median abs error; high-vol (>25% σ) swings to ≈30%.

---

## 3. Threshold Check (Validation Plan)
| Threshold band | ATM median abs error | Observed (Calibrated IV) | Outcome |
| --- | --- | --- | --- |
| Excellent | < 5% | 5.61% | ❌ borderline miss |
| Good | < 10% | 5.61% | ✅ pass |
| Acceptable | < 15% | 5.61% | — |
| Poor | < 25% | 5.61% | — |

The calibrated-IV model clears the “good” bar and is the only contender within 10%. Other variants exceed 12%+ ATM error and exhibit stronger systematic underpricing.

---

## 4. Recommendation
**Adopt the Hybrid pricing mode** with `bs_calibrated_iv` as the synthetic backbone and cached actual prices as the first preference when available.

Guidance:
1. **ATM/ITM trades** – synthetic prices within 6% median error → safe to backfill gaps.
2. **Near-expiry (<8 DTE)** – favour actual prices; if absent, alert replay engine about elevated decay risk.
3. **Deep OTM legs** – enforce a price floor (e.g., ₹5) or require actual quotes to avoid 90%+ relative error.
4. **Volatility spikes (>25% σ)** – monitor; synthetic bias widens to ~30%. Consider recalibrating more frequently or widening skew parameter when realised vol jumps.

Decision payload recorded in `validation_metrics.json` (`evaluations[*].recommendation`).

---

## 5. Outstanding Tasks
- Fetch/run pricing validator for NIFTY, BANKNIFTY, TCS, INFY once option datasets are available.
- Backfill underlying equity data past 2025-08-31 to re-test October expiries without truncation.
- Integrate validator into CI to refresh metrics whenever new pools are ingested; persist run metadata (model parameters, git SHA, data ranges).
- At replay-engine integration, log the chosen pricing mode per trade for auditability.

---

## 6. Artefacts
- Row-level comparison: `src/core/options/data/validation_results/pricing_validation_rows.parquet`
- Summary tables: `.../pricing_validation_summary.csv`, `.../pricing_validation_detail.csv`, `.../model_level_summary.csv`
- Plots: `.../plots/` (error distribution, heatmaps, boxplots, bias scatter)
- Decision JSON: `.../validation_metrics.json`

**Decision**: Proceed with **Hybrid** pricing; promote `bs_calibrated_iv` as the default synthetic engine, constrained by the guardrails above.
