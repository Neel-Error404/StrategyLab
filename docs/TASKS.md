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
