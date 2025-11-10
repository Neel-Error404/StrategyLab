# StrategyLabs v0.2.0 Merge Plan

**Date**: 2025-10-12
**From**: strategylabs_updated.zip (strategylabs-v2/master)
**To**: unified_trading_setup/backtester (master)
**Status**: Planning Complete, Ready for Execution

---

## 📊 Diff Analysis Summary

- **Files Changed**: 651
- **Lines Changed**: ~368,547
- **Backup Branch**: `backup-before-strategylabs-merge-20251012-182956`
- **Source Remote**: `strategylabs-v2` → `../../strategylabs_updated_extracted`

---

## 🎯 5-Feature Incremental Merge Strategy

### Feature Branch Structure

```
master (current: 1fb28a7)
├── feature/options-trading-infrastructure  ← PR #1
├── feature/analysis-framework              ← PR #2
├── feature/indian-equities-pipeline        ← PR #3
├── feature/test-infrastructure             ← PR #4
└── feature/phase4-backtest-system          ← PR #5
```

---

## 📦 PR #1: Options Trading Infrastructure

### Branch: `feature/options-trading-infrastructure`

### Scope
Complete options trading system with replay engine, pricing models, and validation.

### Files to Merge
```
src/core/options/                          # NEW directory
├── __init__.py
├── options_engine.py                      # Existing file (may need review)
├── config/
│   └── options_config.yaml
├── data/
│   ├── cache/
│   ├── validation_results/
│   └── lot_sizes.csv
├── planning/
│   ├── implementation_plan.md
│   └── decisions.md
├── pricing/
│   ├── synthetic_engine.py
│   ├── actual_engine.py
│   ├── hybrid_engine.py
│   └── volatility_models.py
├── replay/
│   ├── config.py
│   ├── engine.py
│   ├── trade_mapper.py
│   ├── position_tracker.py
│   └── metrics_calculator.py
├── validation/
│   ├── pricing_validator.py
│   ├── data_fetcher.py
│   └── validation_runner.py
├── PHASE1_STATUS.md
├── PHASE2_READINESS_ASSESSMENT.md
├── PHASE3_COMPLETE.md
├── PHASE4_COMPLETE.md
├── README.md
├── VALIDATION_REPORT.md
└── VETERAN_ASSESSMENT.md

data/pools/options/                        # NEW directory
└── 2025-04-01_to_2025-10-08/
    ├── BANKNIFTY/
    ├── INFY/
    ├── NIFTY/
    ├── RELIANCE/
    └── TCS/
```

### Merge Steps
```powershell
cd unified_trading_setup/backtester

# 1. Checkout feature branch
git checkout -b feature/options-trading-infrastructure master

# 2. Cherry-pick or selective merge
# Option A: Selective file copy (recommended for first PR)
mkdir -p src/core/options
cp -r ../../strategylabs_updated_extracted/src/core/options/* src/core/options/
mkdir -p data/pools/options
cp -r ../../strategylabs_updated_extracted/data/pools/options/* data/pools/options/

# 3. Stage and commit
git add src/core/options data/pools/options
git commit -m "feat: add options trading infrastructure

- Complete options replay engine with 4 phases validated
- Dual pricing: Black-Scholes + Actual historical
- Strike/expiry selection & Greeks calculation
- Comprehensive validation reports (PHASE1-4 complete)
- Support for BANKNIFTY, NIFTY, RELIANCE, TCS, INFY

Files: 50+ new files in src/core/options/
Data: Options data for 5 tickers (6 months)
"

# 4. Test
python -m pytest tests/options/ -v
python src/core/options/replay/engine.py --help

# 5. Push and create PR
git push origin feature/options-trading-infrastructure
```

### Testing Checklist
- [ ] Options module imports without errors
- [ ] Config loads successfully
- [ ] Sample replay executes
- [ ] Data files accessible
- [ ] Documentation renders correctly

### Conflicts Expected
- `src/core/options/options_engine.py` may exist in master (check and merge carefully)

---

## 📦 PR #2: Analysis Framework

### Branch: `feature/analysis-framework`

### Scope
Advanced analysis scripts, portfolio construction, and strategy optimization tools.

### Files to Merge
```
analysis/                                   # NEW directory
├── ANALYSIS_PROTOCOL.md
├── DIRECTORY_STRUCTURE.md
├── DOCUMENTATION_INDEX.md
├── IMPLEMENTATION_STATUS.md
├── METHODOLOGY.md
├── MSE_STRATEGY_GUIDE.md
├── WORKFLOW_SOP.md
├── config_template.yaml
├── run.py
├── generic/
│   ├── modules/
│   │   ├── config_loader.py
│   │   └── data_loader.py
│   └── scripts/
│       ├── 01_basic_eda.py
│       ├── 02_trade_type_analysis.py
│       ├── 03_cascade_analysis.py
│       ├── 04_stop_loss_simulation.py
│       ├── 05_ticker_ranking.py
│       ├── 06_risk_adjusted_patterns.py
│       ├── 07_top50_vs_overall.py
│       ├── 08_top50_pattern_breakdown.py
│       └── 09_validation_check.py
├── portfolio_construction/
│   ├── scripts/ (6 scripts)
│   ├── data/
│   ├── docs/
│   └── requirements.txt
├── portfolio_learning/
├── strategy_optimization/
│   ├── modules/ (7 modules)
│   ├── scripts/ (12 scripts)
│   ├── checkpoints/
│   └── docs/
├── integration/
│   └── core/trade_enhancer.py
└── reports/
```

### Merge Steps
```powershell
git checkout -b feature/analysis-framework master

# Copy analysis framework
cp -r ../../strategylabs_updated_extracted/analysis .

# Stage and commit
git add analysis/
git commit -m "feat: add advanced analysis framework

- 9 generic analysis scripts for comprehensive insights
- Portfolio construction & optimization (PyPortfolioOpt)
- Strategy optimization with walk-forward validation
- Cascade/anti-cascade pattern analysis
- Trade enhancement utilities

Modules: 20+ analysis scripts
Features: EDA, stop-loss sim, ticker ranking, portfolio optimization
"

# Test
python analysis/run.py --help
python analysis/generic/scripts/01_basic_eda.py --help

git push origin feature/analysis-framework
```

### Testing Checklist
- [ ] All analysis scripts execute
- [ ] Portfolio construction runs
- [ ] Config templates load
- [ ] Dependencies install (check requirements.txt conflicts)

### Conflicts Expected
- None expected (new directory)

---

## 📦 PR #3: Indian Equities Pipeline

### Branch: `feature/indian-equities-pipeline`

### Scope
Automated Indian equities master dataset pipeline with Yahoo Finance integration.

### Files to Merge
```
src/data_tools/                             # NEW directory
├── __init__.py
└── indian_equities_master/
    ├── __init__.py
    ├── cli.py
    ├── fetcher.py
    ├── processor.py
    └── validator.py

config/indian_equities_master.yaml          # NEW file

data/indian_equities_master.csv            # NEW file
data/indian_equities_master.csv.metadata.json  # NEW file

docs/DATA_DICTIONARY.md                     # NEW file (if exists)
```

### Merge Steps
```powershell
git checkout -b feature/indian-equities-pipeline master

# Copy files
mkdir -p src/data_tools
cp -r ../../strategylabs_updated_extracted/src/data_tools/* src/data_tools/
cp ../../strategylabs_updated_extracted/config/indian_equities_master.yaml config/
cp ../../strategylabs_updated_extracted/data/indian_equities_master.csv* data/

# Stage and commit
git add src/data_tools config/indian_equities_master.yaml data/indian_equities_master.csv*
git commit -m "feat: add Indian equities master pipeline

- Yahoo Finance screener integration
- Daily-refreshable security master
- CLI: python -m src.data_tools.indian_equities_master.cli update
- Automated data validation & metadata generation

Output: data/indian_equities_master.csv (authoritative)
"

# Test
python -m src.data_tools.indian_equities_master.cli --help
python -m src.data_tools.indian_equities_master.cli update --dry-run

git push origin feature/indian-equities-pipeline
```

### Testing Checklist
- [ ] CLI executes
- [ ] Data fetching works (requires internet)
- [ ] Output CSV validates
- [ ] Config loads successfully

### Conflicts Expected
- Possible conflict with existing `data/` directory (should be additive)

---

## 📦 PR #4: Test Infrastructure

### Branch: `feature/test-infrastructure`

### Scope
Enhanced test suites for options and Indian equities modules.

### Files to Merge
```
tests/                                      # Existing directory
├── conftest.py                             # UPDATE
├── options/                                # NEW
│   ├── __init__.py
│   ├── test_replay_engine.py
│   ├── test_pricing.py
│   ├── test_validation.py
│   └── fixtures/
└── indian_equities_master/                 # NEW
    ├── __init__.py
    ├── test_fetcher.py
    ├── test_processor.py
    └── test_validator.py
```

### Merge Steps
```powershell
git checkout -b feature/test-infrastructure master

# Copy test directories
cp -r ../../strategylabs_updated_extracted/tests/options tests/
cp -r ../../strategylabs_updated_extracted/tests/indian_equities_master tests/

# Update conftest.py (merge carefully)
# Review existing conftest.py first
# Then merge changes from extracted version

# Stage and commit
git add tests/
git commit -m "feat: add test infrastructure for new modules

- Options module test suite (replay, pricing, validation)
- Indian equities master tests (fetcher, processor, validator)
- Updated pytest configuration
- Test fixtures for integration tests

Coverage: Options (80%+), Indian Equities (90%+)
"

# Test
python -m pytest tests/options/ -v
python -m pytest tests/indian_equities_master/ -v

git push origin feature/test-infrastructure
```

### Testing Checklist
- [ ] All tests discover and run
- [ ] No test conflicts
- [ ] Fixtures work correctly
- [ ] Coverage reports generate

### Conflicts Expected
- `tests/conftest.py` will need manual merge

---

## 📦 PR #5: Phase 4 Backtest System

### Branch: `feature/phase4-backtest-system`

### Scope
Production-scale backtest runner for multi-ticker options validation.

### Files to Merge
```
PHASE4_FULL_BACKTEST_PROMPT.md              # NEW file (44K)
run_phase4_backtest.py                      # NEW file (11K)
test_phase3_integration.py                  # NEW file (13K)
test_data_loading.py                        # NEW file (3.4K)
merge_trade_files.py                        # NEW file (2.7K)

outputs/phase4_backtest/                    # NEW directory
└── equity_trades_full.csv                  # Sample data
```

### Merge Steps
```powershell
git checkout -b feature/phase4-backtest-system master

# Copy root files
cp ../../strategylabs_updated_extracted/PHASE4_FULL_BACKTEST_PROMPT.md .
cp ../../strategylabs_updated_extracted/run_phase4_backtest.py .
cp ../../strategylabs_updated_extracted/test_phase3_integration.py .
cp ../../strategylabs_updated_extracted/test_data_loading.py .
cp ../../strategylabs_updated_extracted/merge_trade_files.py .

# Copy sample outputs
mkdir -p outputs/phase4_backtest
cp -r ../../strategylabs_updated_extracted/outputs/phase4_backtest/* outputs/phase4_backtest/ 2>/dev/null || true

# Stage and commit
git add PHASE4_FULL_BACKTEST_PROMPT.md run_phase4_backtest.py test_phase3_integration.py test_data_loading.py merge_trade_files.py outputs/phase4_backtest/
git commit -m "feat: add Phase 4 production backtest system

- Multi-ticker production runner (5 tickers, 6 months)
- Comprehensive validation framework
- Integration test suite for phase 3
- Trade file merging utilities
- Complete audit trail with SHA256 hashing

Runner: run_phase4_backtest.py
Documentation: PHASE4_FULL_BACKTEST_PROMPT.md (44K guide)
"

# Test
python run_phase4_backtest.py --help
python test_phase3_integration.py
python test_data_loading.py

git push origin feature/phase4-backtest-system
```

### Testing Checklist
- [ ] Phase 4 runner executes
- [ ] Integration tests pass
- [ ] Data loading validates
- [ ] Documentation renders correctly

### Conflicts Expected
- None expected (new files at root)

---

## 🔧 Additional Updates to Merge

### AGENTS.md Update
- Current: 7.4K
- New: 11K (252 additional lines)
- Action: Review and merge (likely additive)

### README.md Update
- Changes: Indian Equities Master section added (lines 44-59)
- Action: Merge sections carefully

### Requirements.txt
- Verify: Both versions are nearly identical
- Action: Check for any new dependencies in analysis/portfolio_construction/requirements.txt

---

## 📋 Merge Execution Order

### Phase 1: Foundation (Week 1)
1. **PR #3** - Indian Equities Pipeline (lowest risk, no dependencies)
2. **PR #4** - Test Infrastructure (enables testing of subsequent PRs)

### Phase 2: Core Features (Week 2-3)
3. **PR #1** - Options Trading Infrastructure (major feature)
4. **PR #2** - Analysis Framework (depends on stable data)

### Phase 3: Validation (Week 4)
5. **PR #5** - Phase 4 Backtest System (integration of all features)

---

## 🔍 Pre-Merge Checklist (Each PR)

### Before Creating PR
- [ ] Feature branch created from latest master
- [ ] Files copied and staged
- [ ] Commit message follows convention
- [ ] Local tests pass
- [ ] No unintended files included
- [ ] .gitignore rules checked

### After Creating PR
- [ ] CI/CD passes (if configured)
- [ ] Code review requested
- [ ] Documentation reviewed
- [ ] Integration tests pass
- [ ] No merge conflicts with master

### Before Merging
- [ ] All review comments addressed
- [ ] Squash commits if needed
- [ ] Update CHANGELOG.md
- [ ] Tag release if applicable

---

## 🚨 Conflict Resolution Strategy

### Expected Conflicts

1. **src/core/options/options_engine.py**
   - Exists in both repos
   - Resolution: Compare implementations, keep best or merge

2. **tests/conftest.py**
   - Will need manual merge
   - Resolution: Combine pytest fixtures from both

3. **README.md**
   - Minor updates
   - Resolution: Accept both changes, combine sections

### Conflict Resolution Process
```powershell
# If conflict occurs during merge
git status  # Identify conflicted files
git diff --ours --theirs <file>  # Review differences
# Manually edit file to resolve
git add <file>
git commit
```

---

## 📊 Post-Merge Validation

### After Each PR Merge
```powershell
# 1. Full test suite
python -m pytest tests/ -v --cov=src

# 2. Validate runner
python src/runners/unified_runner.py --mode validate

# 3. Run sample backtest
python src/runners/unified_runner.py --mode backtest --date-ranges 2025-06-06_to_2025-06-07 --tickers RELIANCE

# 4. Check for regressions
git diff HEAD~1 config/  # Verify no unintended config changes
```

### After All PRs Merged
```powershell
# 1. Full system validation
python src/runners/unified_runner.py --mode validate --date-ranges 2024-01-01_to_2024-01-31

# 2. Options system test
python run_phase4_backtest.py

# 3. Indian equities update
python -m src.data_tools.indian_equities_master.cli update

# 4. Analysis framework test
python analysis/run.py --strategy mse --mode generic

# 5. Create release tag
git tag -a v0.2.0 -m "StrategyLab v0.2.0: Options Trading & Advanced Analytics"
git push origin v0.2.0
```

---

## 📝 Communication Plan

### Team Communication
1. **Before Starting**: Announce merge plan to team
2. **Per PR**: Send PR link with summary
3. **After Each Merge**: Brief status update
4. **Final Merge**: Comprehensive release notes

### Documentation Updates
1. Update main README.md with new features
2. Create CHANGELOG.md entry for v0.2.0
3. Update wiki/docs with new module documentation
4. Record any breaking changes or migration notes

---

## 🎯 Success Metrics

### Definition of Done (Per PR)
- [ ] All tests pass
- [ ] Documentation complete
- [ ] Code reviewed and approved
- [ ] No regressions in existing functionality
- [ ] New features validated

### Overall Migration Success
- [ ] All 5 PRs merged
- [ ] Zero critical bugs
- [ ] System performance maintained or improved
- [ ] Team trained on new features
- [ ] Production deployment successful

---

## 🔄 Rollback Plan

### If Issues Arise
```powershell
# Rollback to backup branch
git checkout master
git reset --hard backup-before-strategylabs-merge-20251012-182956
git push origin master --force  # Use with extreme caution!

# Or revert specific PR
git revert <merge-commit-sha>
git push origin master
```

### Backup Strategy
- Backup branch: `backup-before-strategylabs-merge-20251012-182956`
- Local backup: Create zip of working directory
- Remote backup: Verify GitHub has latest state

---

## 📞 Support & Escalation

### Merge Issues
- Git conflicts: Review MERGE_PLAN.md conflict resolution section
- Test failures: Check test logs in `tests/`
- Runtime errors: Review logs in `logs/`

### Escalation Path
1. Check documentation and README files
2. Review git commit history for context
3. Consult backup branch for comparison
4. Create issue with detailed error logs

---

## 🎓 Lessons Learned Template

### Post-Merge Review (After Each PR)
- What went well?
- What could be improved?
- Unexpected issues encountered?
- Time vs. estimate accuracy?
- Process improvements for next PR?

---

**Next Steps**: Execute PR #3 (Indian Equities Pipeline) first as lowest-risk foundation.
