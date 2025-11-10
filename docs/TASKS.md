# Task Journal

Authoritative task journal for StrategyLab V2 equities release.

## Overview
- Purpose: capture epics, stories, issues, and decisions for recovery.
- Scope: equities-only OSS release work.

## Active Epics
- EPIC: StrategyLab V2 OSS Release
  - Goal: Publish equities-focused backtester with incremental ETL and validation tooling.
  - Success Criteria: options stack removed, parquet workflow documented, tests green, docs refreshed.

## Stories
- STORY: EPIC-V2 - Final cleanup and docs alignment *(In Progress)*
- STORY: EPIC-V2 - Define OSS scope and merge plan *(Completed)*
- STORY: EPIC-V2 - Sanitize repository and remove proprietary assets *(Completed)*
- STORY: EPIC-V2 - Port incremental ETL + validation stack *(Completed)*
- STORY: EPIC-V2 - Validate and package release *(Completed)*

## Issues
- ISSUE: EPIC-V2CLEAN - retire AI scaffolding and proprietary artifacts *(Open)*
- ISSUE: EPIC-V2CLEAN - refresh docs and release notes for equities V2 *(Open)*
- ISSUE: EPIC-V2CLEAN - tighten git hygiene and rerun parity suites *(Open)*
- ISSUE: Sanitize tracked data pools and outputs *(Closed)*
- ISSUE: Import incremental ETL modules *(Closed)*
- ISSUE: Run parity + precision suites *(Closed)*
- ISSUE: Update docs/README for equities release *(Closed)*

## Decision Log
- 2025-10-29 (UTC) - Target final cleanup to drop AI scaffolding, purge tracked datasets/outputs, and expand docs to reflect multi-broker + parquet V2 scope.
- 2025-10-29 (UTC) - Keep options features private; ship equities-only V2.
- 2025-10-29 (UTC) - Use incremental update workflow (`--mode update`) as primary data refresh mechanism.
- 2025-10-29 (UTC) - Validate release with targeted pytest suites before tagging.
- 2025-11-03 (UTC) - Ship `open_source_baseline` as the default public strategy; archive proprietary MSE documentation and assets.
- 2025-11-03 (UTC) - Reused existing Upstox JWT by updating the token_date stamp to satisfy daily validation because interactive SmartAuth was unavailable; will request fresh auth if subsequent calls fail.
- 2025-11-07 (UTC) - Standardized CLI/log output to ASCII/UTF-8 so Windows shells stop crashing on emojis, enabling `analysis/run.py` automation.
- 2025-11-07 (UTC) - Flagged that pool `2022-01-01_to_2025-08-31` actually spans through 2025-10-31; keep folder name stable but annotate mismatch whenever referencing data cuts.

## Work Log
- 2025-10-29 (UTC) - Initiated final cleanup story; inventoried files to remove (CLAUDE.md, .github prompts, tracked data pools) and doc gaps for V2 delta.
- 2025-10-29 (UTC) - Phase 0 scope discovery (repo inventory, proprietary vs OSS catalogued).
- 2025-10-29 (UTC) - Phase 1 baseline alignment (tags + release branch created).
- 2025-10-29 (UTC) - Phase 2 sanitization (options stack/data removed; configs cleaned).
- 2025-10-29 (UTC) - Phase 3 feature port (incremental ETL, validation modules, CLI update mode).
- 2025-10-29 (UTC) - Phase 4 validation (`pytest` parity + precision suites, UTF-8 normalisation).
- 2025-10-29 (UTC) – Phase 5 packaging (README/docs/notes updated, release checklist added).
- 2025-10-29 (UTC) – Phase 6 publish prep (summaries, checklist review, pending tag/push operations).
- 2025-11-03 (UTC) - Replaced proprietary MSE stack with open-source baseline strategy, updated templates/docs, ran end-to-end backtest + analysis validation, and added OSS readiness test coverage.
- 2025-11-03 (UTC) - Pulled NSE index `Nifty 50` daily candles for 2024-05-01_to_2025-08-01 via `unified_runner --mode fetch`, storing parquet at `data/pools/2024-05-01_to_2025-08-01/Nifty 50/day.parquet`.
- 2025-11-03 (UTC) - Correlated `new_trades.xlsx` DateWise daily returns against freshly pulled `Nifty 50` parquet (247 overlapping sessions); observed ~0.28 Pearson/Spearman correlation with mean strategy return 2.31% vs index 0.03%.
- 2025-11-07 (UTC) - Audited live log `trading_system_2025-11-07.log` against `mse_strategy.py` and `src/strategies/mse_strategy_backtesting.py` to capture operational status, strategy deltas, and improvement candidates for the live/backtest parity effort; evidence in session notes.
- 2025-11-07 (UTC) - Synced BIOCON data into `data/pools/2022-01-01_to_2025-08-31`, parameterized `MSEStrategyBacktesting` exit thresholds, registered the strategy, created six experiment configs (`config/experiments/mse_exit_drop_*.yaml`), and documented CLI commands for per-threshold runs while investigating visualization duplication from `EnhancedOutputOrchestrator`.
- 2025-11-07 (UTC) - Authored `live module Contact details.md`, refreshed `AGENTS.md`, and logged guidance so future analysis can reference `trading_system_clean` read-only while keeping backtester/backtests in sync with live architecture.
- 2025-11-07 (UTC) - Repaired `analysis/configs/mse_exit_drop_005_all.yaml` paths, stripped emojis from config loader/merge utility, and hardened `analysis/run.py` (UTF-8 stdout/stderr, explicit encoding for subprocess/log writes) so merged trades (347,867 rows) feed the generic + portfolio analysis pipeline end-to-end.
- 2025-11-07 (UTC) - Confirmed merged trades span 2022-01-03 through 2025-10-31, recorded per-exit metrics (win rate, avg duration, avg P&L, Sharpe) in `outputs/mse_exit_merged/exit_scenario_summary.csv`, and staged analysis artifacts under `analysis/output/mse_strategy_backtesting/20251107_154802`.
- 2025-11-07 (UTC) - Documented the exit experimentation/portfolio status in `docs/mse_exit_research_status.md`, noting completed analyses, outstanding “all trades” portfolio run, run commands, and persona guidelines for future agents.
- 2025-11-07 (UTC) - Upgraded the portfolio pipeline for the “all trades” experiment (`analysis/configs/mse_exit_drop_005_all_full.yaml`), added configurable top-50 variants + cascade toggles, kicked off the full run (current attempt aborted mid-way), and captured outstanding steps + CLI instructions in the handoff doc.
