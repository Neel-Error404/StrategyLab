# Task Journal

Authoritative task journal for epics, stories, issues, decisions, and work logs.

If this file is moved or renamed, update the path in `AGENTS.md` under "Task Protocol & Agent Persona" in the same change set.

## Overview
- Purpose: Provide resilient, continuously updated context to recover from interruptions and ensure continuity.
- Scope: All ongoing work related to this repository (features, fixes, experiments, and docs).

## Active Epics
- EPIC: Task Protocol Adoption
  - Goal: Establish clear process, journal, and output style for agents and contributors.
  - Success Criteria: `AGENTS.md` updated, journal in place and used, outputs follow style.

## Stories
- STORY: EPIC-Task-Protocol - Define and publish protocol in `AGENTS.md`
  - Acceptance Criteria:
    - Protocol documented with journaling rules, persona, and output style.
    - Journal path specified and verified.
  - Status: Done ✅

- STORY: EPIC-Task-Protocol - Create and initialize task journal
  - Acceptance Criteria:
    - `docs/TASKS.md` created with sections for epics, stories, issues, decision log, work log.
    - Initial entries added.
  - Status: Done ✅

- STORY: EPIC-Task-Protocol - Align backtester with live entry/exit semantics
  - Acceptance Criteria:
    - Backtester uses next-bar OPEN for entries/exits when using live-matching strategy.
    - No overlapping trades per ticker; entries occur only when flat.
    - Documented in Work Log; minimal, strategy-scoped change.
  - Status: In Progress 🚧

## Issues
- ISSUE: STORY-Protocol-Doc - Update `AGENTS.md` with protocol, persona, output style
  - Status: Done ✅
  - Evidence: See diff of `AGENTS.md` in this change.

- ISSUE: STORY-Journal-Init - Add `docs/TASKS.md` with initial structure
  - Status: Done ✅

- ISSUE: STORY-LiveAlign - Shift entry/exit to next-bar OPEN in `mse_80pct_no_cascade_live_matching`
  - Status: Done ✅
  - Files: `src/strategies/mse_80pct_no_cascade_live_matching.py`
  - Notes: Adds `use_open_for_entry/exit` flags and moves entry signals to next bar; clears detection-bar entries.

## Decision Log
- 2025-09-08 (UTC) — Journal Location
  - Decision: Use `docs/TASKS.md` as the authoritative journal path.
  - Rationale: Visible to contributors, aligns with existing `docs/` structure, versioned for history.
  - Alternatives considered: `logs/` (not versioned for process), `.codex/` (hidden from casual contributors).

- 2025-09-08 (UTC) — Curated backup scope
  - Decision: Limit curated backups to only `.py` files under `src/` and `config/`, plus `.csv` files within those directories; exclude credentials (e.g., `config/access_tokens/**`) and all generated data elsewhere.
  - Rationale: Create a lean “code-only” artifact for the database repo, avoid large datasets and secret material.
  - Alternatives considered: Include docs and root reports; rejected to keep scope minimal and avoid noise.

- 2025-09-08 (UTC) — Docs inclusion for curated backup
  - Decision: Include `docs/**`, `AGENTS.md`, `README.md`, `RELEASE_NOTES.md`, and `requirements.txt` in the final curated backup after validating they reflect current `src/` and `config/`.
  - Rationale: Provide self-contained, truthful documentation with the code package.
  - Notes: `RELEASE_NOTES.md` limited to code/config changes; no analysis items per policy.

## Work Log
- 2025-09-08 (UTC) — Initialize protocol and journal
  - Changes: Updated `AGENTS.md` with Task Protocol, persona, output style; created `docs/TASKS.md`.
  - Files: `AGENTS.md`, `docs/TASKS.md`
  - Notes: Begin logging future work and decisions here. Use UTC timestamps.

- 2025-09-08 (UTC) — Backtester live-semantics alignment
  - Changes: Updated `mse_80pct_no_cascade_live_matching` to set entry/exit at next-bar OPEN and flag extractor via `use_open_for_entry/exit`.
  - Files: `src/strategies/mse_80pct_no_cascade_live_matching.py`
  - Evidence: Plan to compare against live via existing scripts (`create_live_matching_backtest.py`, `extract_live_matching_results.py`).

- 2025-09-08 (UTC) — Run backtest for BIOCON, JSWSTEEL, KOTAKBANK
  - Range: 2025-05-29_to_2025-09-06; Strategy: `mse_80pct_no_cascade_live_matching`; Parallel: disabled (sandbox limitation).
  - Output Dir: `outputs/20250908_052243/mse_80pct_no_cascade_live_matching/2025-05-29_to_2025-09-06`
  - Artifacts: strategy_trades and risk_approved_trades CSV/JSON per ticker; executive summary JSON.
  - Next: Validate entries/exits occur at next-bar OPEN; compare trade counts vs live; check for overlapping trades.

- 2025-09-08 (UTC) — Overlap audit + window analytics
  - Created: `analysis_reports/overlap_audit.json` (no overlaps across BIOCON/JSWSTEEL/KOTAKBANK).
  - Created: `analysis_reports/backtest_window_summary.json` (window: 2025-08-26..2025-09-04).
  - Created: `audits/strategy_trades_window/*_StrategyTrades_2025-08-26_to_2025-09-04.csv`.

- 2025-09-08 (UTC) — Broker trades pairing (Aug 26–Sep 4)
  - Source: `real_trades_excel_converted.csv` (Excel parsing blocked; openpyxl not available in sandbox).
  - Output: `audits/broker_trades/BROKER_Trades_2025-08-26_to_2025-09-04.csv` + summary JSON.
  - Note: Simple first-opposite pairing per (ticker, day); quantity mismatches counted and logged.

- 2025-09-08 (UTC) — Create merged file (strategy vs broker)
  - Output: `audits/merged/MERGED_Strategy_Broker_Trades_2025-08-26_to_2025-09-04.csv`
  - Summary: `audits/merged/MERGED_summary_2025-08-26_to_2025-09-04.csv`
  - Format: unified columns [source,ticker,symbol,trade_type,entry_time,entry_price,exit_time,exit_price,quantity,profit_currency,profit_pct,trade_duration_min], times normalized to naive IST.

- 2025-09-08 (UTC) — Build code-only curated backup
  - Changes: Generated curated manifest and zip containing only code `.py` under `src/` and `config/`, and config CSVs; excluded tokens and generated data.
  - Files: `backup_curated_manifest_codeonly_20250908_100936.txt`, `backup_curated_codeonly_20250908_100936.zip`
  - Counts: 116 Python files, 3 CSV files, 119 total entries.

- 2025-09-08 (UTC) — Align docs and release notes
  - Changes: Updated `README.md` to Python 3.10+ and venv setup; rewrote `RELEASE_NOTES.md` to only include code/config changes (runners, strategies, config) with no analysis items.
  - Files: `README.md`, `RELEASE_NOTES.md`
  - Validation: Verified presence of `src/runners/unified_runner.py`, CLI modules, strategy registry and implementations, config templates.
