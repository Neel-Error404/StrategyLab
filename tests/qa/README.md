# QA Testing Suite

This directory contains comprehensive integration tests for the StrategyLab backtesting system.

## Directory Structure

```
tests/qa/
├── README.md                            # This file
├── __init__.py                          # Package init
├── phase0_environment_setup.py          # Phase 0.1: Environment validation
├── phase0_data_baseline.py              # Phase 0.2: Fresh data pull
├── phase1_core_backtester_single.py     # Phase 1.1: Single ticker/strategy
├── phase1_core_backtester_multi.py      # Phase 1.2: Multi-ticker/strategy
├── phase1_known_truth_validation.py     # Phase 1.3: Manual verification
├── phase2_etl_gap_detection.py          # Phase 2.1: Gap calculation
├── phase2_etl_incremental_update.py     # Phase 2.2: Incremental update
├── phase3_options_single_trade.py       # Phase 3.1: Single options trade
├── phase3_options_multi_ticker.py       # Phase 3.2: Multi-ticker options
├── phase3_options_known_truth.py        # Phase 3.3: Options validation
├── phase4_generic_analysis.py           # Phase 4.1: Analysis modules
├── phase4_portfolio_construction.py     # Phase 4.2: Portfolio optimization
└── phase4_full_pipeline.py              # Phase 4.3: End-to-end pipeline
```

## Test Execution Order

1. **Phase 0**: Environment & Data Baseline (Day 1)
2. **Phase 1**: Core Backtester (Day 2)
3. **Phase 2**: ETL Update Tool (Day 3)
4. **Phase 3**: Options Tester (Day 4-5)
5. **Phase 4**: Analysis & Portfolio (Day 6)

## Running Tests

### Individual Test
```powershell
python tests/qa/phase0_environment_setup.py
```

### All Tests (Sequential)
```powershell
# Run all QA tests
pytest tests/qa/ -v --tb=short
```

### Specific Phase
```powershell
# Run all Phase 1 tests
pytest tests/qa/phase1_*.py -v
```

## Documentation

- **Testing Plan**: `QA_INTEGRATION_TESTING_PLAN.md`
- **Testing Journal**: `QA_TESTING_JOURNAL.md`
- **System Assessment**: `COMPREHENSIVE_SYSTEM_ASSESSMENT.md`

## Success Criteria

All tests must pass with:
- Numerical accuracy: ±0.01 (1 paisa)
- Reproducibility: SHA256 hash match
- Performance: Within time targets
- Data integrity: Zero corruption/duplicates

## Notes

- All tests assume fresh data baseline in `data/pools/qa_testing_baseline/`
- Tests document results in `QA_TESTING_JOURNAL.md`
- Critical tests validate against known truth (manual calculation)
