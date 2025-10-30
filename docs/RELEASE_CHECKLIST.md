# Release Checklist – StrategyLab Equities V2

## Pre-flight
- [x] Ensure working branch `release/strategylab-v2`.
- [x] Confirm options modules/data removed (`rg "src.core.options"`).
- [x] Verify UTF-8 encoding for new modules.

## Tests
- [x] Activate repo venv: `.venv\Scripts\activate`.
- [x] Run parity + precision suites: `python -m pytest tests/test_backtest_live_parity.py tests/test_precision_validation.py -q`.
- [ ] (Optional) Run broader smoke tests once data pools available.

## Docs & Packaging
- [x] README updated for V2 release.
- [x] `RELEASE_NOTES.md` prepared.
- [x] Update audit trail (`docs/strategylab_v2_phase0_audit.md`, `docs/TASKS.md`).
- [ ] Attach minimal sample manifest or instructions for generating pools.

## Publish
- [ ] Tag release (e.g., `v2.0.0-equities`).
- [ ] Push branch & tag to origin.
- [ ] Create GitHub release with notes.
- [ ] Share incremental update workflow in announcement.
