# Architecture Decision Records (ADR)

**Purpose**: Document key architectural and design decisions made during options backtesting implementation

**Format**: Each decision follows the template below

---

## ADR Template

```
## ADR-XXX: [Decision Title]

**Date**: YYYY-MM-DD
**Status**: Proposed | Accepted | Deprecated | Superseded
**Deciders**: [Names/roles]
**Context**: [What is the issue we're trying to solve?]

### Decision
[What is the change we're proposing/have made?]

### Rationale
[Why did we choose this option?]

### Alternatives Considered
1. **Option A**: [Description]
   - Pros: ...
   - Cons: ...
   - Rejected because: ...

2. **Option B**: [Description]
   - Pros: ...
   - Cons: ...
   - Rejected because: ...

### Consequences
**Positive**:
- [Benefit 1]
- [Benefit 2]

**Negative**:
- [Tradeoff 1]
- [Tradeoff 2]

**Risks**:
- [Risk 1 and mitigation]

### Implementation Notes
[Technical details, code references, related ADRs]

### Review Date
[When should we revisit this decision?]
```

---

## ADR-001: Decoupled Replay Architecture

**Date**: 2025-10-08
**Status**: Accepted
**Deciders**: StrategyLab Team

### Context
Need to add options backtesting without modifying existing equity backtesting system. Want to maintain separation of concerns and enable independent evolution of both systems.

### Decision
Build options backtesting as a **replay engine** that consumes equity backtester outputs (`trades.csv`, `base_data.csv`) rather than integrating into the equity execution engine.

### Rationale
- **No Risk to Existing System**: Equity backtester remains untouched, no regression risk
- **Clean Separation**: Options logic is isolated, easier to test and maintain
- **Apples-to-Apples Comparison**: Same underlying signals executed in equity vs options
- **Rapid Prototyping**: Can iterate on options logic without equity code changes
- **Reversibility**: Can disable options module without affecting equity system

### Alternatives Considered

1. **Integrated Execution Engine**
   - Pros: Single codebase, shared infrastructure
   - Cons: High coupling, complex branching logic, regression risk
   - Rejected because: Too risky to modify proven equity system

2. **Separate Strategy Implementation**
   - Pros: Complete independence
   - Cons: Duplicate signal generation logic, hard to compare
   - Rejected because: Want to test same signals on different instruments

### Consequences

**Positive**:
- Zero impact on existing equity backtests
- Clear data flow (equity → options)
- Easy to explain and audit
- Can run both in parallel for comparison

**Negative**:
- Slight data duplication (must read `base_data.csv` again)
- Two separate execution paths (equity and options)
- Can't easily implement "hybrid" strategies (equity + options)

**Risks**:
- If equity backtester output format changes, options module breaks
  - **Mitigation**: Define strict data contract, version control schemas

### Implementation Notes
- Options module in `src/core/options/`
- Main entry point: `replay/replay_runner.py`
- Consumes: `trades.csv`, `base_data.csv`
- Produces: `options_trades.csv`, `options_base_data.csv`, etc.

### Review Date
After Phase 2 (MVP) completion - verify architecture scales

---

## ADR-002: Hybrid Pricing Mode as Default

**Date**: 2025-10-08
**Status**: Accepted
**Deciders**: StrategyLab Team

### Context
Need to balance pricing accuracy (actual market data) with backtest coverage (synthetic for data gaps). Upstox provides only 6 months of expired options data, but we have 3+ years of underlying equity data.

### Decision
Implement **three pricing modes** (synthetic, actual, hybrid) with **hybrid as the recommended default** after validation.

### Rationale
- **Hybrid Mode**:
  - Uses actual market prices when available (high accuracy)
  - Falls back to synthetic for data gaps (extended coverage)
  - Tracks which mode was used per trade (transparency)
  - Best of both worlds for most use cases

- **Keep All 3 Modes**:
  - Synthetic: Rapid prototyping, unlimited time periods
  - Actual: Final validation, most conservative estimates
  - Hybrid: Production backtesting

### Alternatives Considered

1. **Synthetic Only**
   - Pros: Works for any time period, fast, no data fetching
   - Cons: Inaccurate, overstates P&L, not tradeable
   - Rejected as sole mode because: Too many assumptions, can't trust results

2. **Actual Only**
   - Pros: Most accurate, reflects real market
   - Cons: Limited to 6 months, can't compare to long equity backtests
   - Rejected as sole mode because: Too restrictive, limits analysis

3. **Always Prefer Newer Data** (hybrid variant)
   - Pros: Most accurate for recent periods
   - Cons: Discontinuity in pricing quality over time
   - Rejected because: Introduces time-based bias in results

### Consequences

**Positive**:
- Flexibility to choose mode based on use case
- Can extend backtests beyond 6 months with documented assumptions
- Empirical validation (Phase 1) will quantify synthetic accuracy

**Negative**:
- Three code paths to maintain (complexity)
- Hybrid results are "blended reality" (harder to explain)
- Need to track and report which mode was used

**Risks**:
- Synthetic model is too inaccurate, hybrid becomes "garbage in"
  - **Mitigation**: Phase 1 validation with strict acceptance criteria (<10% error)
- Users misinterpret hybrid results as fully accurate
  - **Mitigation**: Clearly document pricing_mode in all outputs, generate data quality reports

### Implementation Notes
- `pricing/synthetic_engine.py` - Black-Scholes models
- `pricing/actual_engine.py` - Historical data queries
- `pricing/hybrid_engine.py` - Orchestrates fallback logic
- Config: `options_config.yaml → pricing.mode`
- Output: `options_trades.csv → pricing_mode` column

### Review Date
After Phase 3 (Actual Data Integration) - verify hybrid accuracy

---

## ADR-003: Fixed 1-Lot Strategy for MVP

**Date**: 2025-10-08
**Status**: Accepted
**Deciders**: StrategyLab Team

### Context
Options lot sizes are fixed (e.g., 505 contracts for RELIANCE), making it impossible to exactly replicate equity position sizes. Need a capital allocation strategy.

### Decision
**MVP (Phase 2)**: Fixed 1-lot per trade, regardless of equity position size.

**Future (Phase 4)**: Add configurable lot sizing (capital_match, delta_match).

### Rationale
- **Start Simple**: 1-lot is easiest to implement, test, and reason about
- **Clean Comparisons**: Every option trade uses same lot count, isolates signal quality
- **Accept Mismatch**: Capital deployed will differ from equity, but that's okay for initial validation
- **Iterate**: Can add sophisticated lot sizing once foundation is proven

### Alternatives Considered

1. **Capital Matching from Day 1**
   - Pros: More realistic, better comparison to equity
   - Cons: Complex (fractional lots, rounding), hides signal quality in allocation logic
   - Rejected for MVP because: Premature optimization, adds complexity

2. **Delta Matching**
   - Pros: Most accurate exposure replication
   - Cons: Requires Greeks calculation before entry, dynamic adjustment
   - Rejected for MVP because: Too complex for initial validation

3. **Equity Quantity / Lot Size** (fractional)
   - Pros: Direct mapping
   - Cons: Always results in fractional lots (usually <1), not tradeable
   - Rejected because: Not executable in real market

### Consequences

**Positive**:
- Simplest possible implementation
- Easy to understand results
- Fast MVP delivery
- Can measure raw signal quality (without capital allocation noise)

**Negative**:
- Capital deployed will be much less than equity (options cost 2-5% of stock)
- Absolute P&L not directly comparable to equity
- Can't answer "what if I deployed same capital?" question in MVP

**Risks**:
- Stakeholders misinterpret results ("options only made ₹10K vs equity ₹100K")
  - **Mitigation**: Clear documentation, focus on P&L% and Sharpe, not absolute ₹
  - Generate "capital efficiency" metric (P&L per ₹ deployed)

### Implementation Notes
- `config/options_config.yaml → lot_sizing.method = "fixed"`
- `lot_sizing.fixed.lots_per_trade = 1`
- Phase 4: Add `capital_match` and `delta_match` methods
- Comparison report: Include "capital_deployed" column

### Review Date
After Phase 2 (MVP) - evaluate if fixed 1-lot is sufficient or if capital matching is critical

---

## ADR-004: Follow Equity Exit Signals (No Independent Stops)

**Date**: 2025-10-08
**Status**: Accepted (for MVP)
**Deciders**: StrategyLab Team

### Context
Options can move faster than underlying (leverage, theta decay). Should we add independent stop-loss/take-profit for options, or follow equity exit signals exactly?

### Decision
**MVP (Phase 2)**: Follow equity exit signals exactly. No independent options exits.

**Future (Phase 4)**: Add optional options-specific risk management (stop-loss, take-profit).

### Rationale
- **Pure Replay**: Testing "what if I used options instead of equity?" with identical signal logic
- **Apples-to-Apples**: Any performance difference is due to instrument leverage, not exit timing
- **Simplicity**: One exit trigger (equity signal), easier to implement and debug
- **Validate Foundation First**: Prove signal quality before adding risk overlays

**Exception**: Force close 24 hours before expiry (operational risk, not strategy)

### Alternatives Considered

1. **Independent Options Stops from Day 1**
   - Pros: More realistic, protects against option-specific risks (theta decay eating into position)
   - Cons: Introduces new variable (stop level), hard to isolate signal quality vs risk management
   - Rejected for MVP because: Conflates two different questions

2. **Hybrid (Both Equity and Options Exits)**
   - Pros: "Whichever comes first" is most conservative
   - Cons: Complex logic, results depend heavily on stop levels
   - Rejected for MVP because: Too many parameters

### Consequences

**Positive**:
- Clean isolation of signal quality
- Easy to explain (same signal, different instrument)
- Fast implementation
- Results are deterministic (one exit rule)

**Negative**:
- Options may hit large unrealized losses that equity didn't (theta decay)
- Can't answer "should I add options-specific risk management?" yet
- May overstate options performance if equity exits are too late

**Risks**:
- Options blow past reasonable stop levels while waiting for equity signal
  - **Mitigation**: Track unrealized P&L drawdown in `options_base_data.csv`, analyze post-hoc
  - Phase 4 can add stops based on Phase 2 learnings

### Implementation Notes
- `config/options_config.yaml → position_management.exit.follow_equity_signal = true`
- `position_management.exit.stop_loss.enabled = false` (MVP)
- `position_management.exit.take_profit.enabled = false` (MVP)
- **Always enabled**: `force_close_before_expiry` (operational risk)
- Phase 4: Make stops configurable, run sensitivity analysis

### Review Date
After Phase 2 - analyze unrealized drawdowns to inform Phase 4 stop levels

---

## ADR-005: ATM Strike Selection for MVP

**Date**: 2025-10-08
**Status**: Accepted (for MVP)
**Deciders**: StrategyLab Team

### Context
Multiple strike selection strategies are possible (ATM, delta-based, moneyness, premium %). Which to use for MVP?

### Decision
**MVP (Phase 2)**: ATM (At-The-Money) strikes only - strike closest to underlying price.

**Future (Phase 4)**: Add delta-based, moneyness, premium % methods.

### Rationale
- **Most Liquid**: ATM options have tightest spreads, highest volume/OI
- **Balanced Greeks**: ~0.50 delta (moderate leverage), meaningful theta, good gamma
- **Deterministic**: No ambiguity (always pick closest strike)
- **Data Availability**: ATM options most likely to exist in historical data
- **Simplest**: No need to calculate delta or fit vol surface

### Alternatives Considered

1. **Delta-Based (e.g., 30-delta)**
   - Pros: More OTM (higher leverage), consistent delta across tickers
   - Cons: Requires delta calculation before entry, strike availability varies
   - Deferred to Phase 4 because: Adds complexity, ATM is sufficient for validation

2. **Fixed Moneyness (e.g., 5% OTM)**
   - Pros: Consistent distance from price
   - Cons: Delta varies with vol regime, may select illiquid strikes
   - Deferred to Phase 4

3. **Premium % (e.g., 2% of underlying)**
   - Pros: Fixed capital per contract
   - Cons: Strike changes with vol (unstable), may not exist
   - Deferred to Phase 4

### Consequences

**Positive**:
- Simplest implementation (no vol calculations needed)
- Best data availability (ATM always exists)
- Most liquid (real-world executable)
- Easy to explain

**Negative**:
- Can't test OTM strategies (higher leverage, lower cost)
- Delta is not constant across trades (varies with vol)
- May not match how traders actually select strikes

**Risks**:
- ATM performance doesn't generalize to OTM
  - **Mitigation**: Phase 4 sensitivity analysis will test other strike methods

### Implementation Notes
- `config/options_config.yaml → strike_selection.method = "atm"`
- `trade_mapper.py → round_to_nearest_strike(underlying_price)`
- Strike interval: 50 for NIFTY, 100 for BANKNIFTY, varies by ticker
- Phase 4: Implement `DeltaStrikeSelector`, `MoneynessStrikeSelector`

### Review Date
After Phase 2 - evaluate if ATM results are promising enough to justify OTM testing

---

## ADR-006: Nearest Weekly Expiry for MVP

**Date**: 2025-10-08
**Status**: Accepted (for MVP)
**Deciders**: StrategyLab Team

### Context
Options have multiple expiries (weekly, monthly, quarterly). Which to use?

### Decision
**MVP (Phase 2)**: Nearest weekly expiry (for indices), nearest monthly (for equities if no weekly).

**Future (Phase 4)**: Add fixed DTE targeting (e.g., always 7 DTE).

### Rationale
- **High Theta**: Weekly options decay faster (more theta capture for short hold times)
- **Liquidity**: Weekly expiries are very liquid for NIFTY/BANKNIFTY
- **Match Equity Hold Times**: Equity trades avg 12-24 hours, weekly expiries appropriate
- **Operational Simplicity**: Always pick next expiry, no DTE calculation needed

### Alternatives Considered

1. **Monthly Expiries**
   - Pros: More time value, less expiry management
   - Cons: Lower theta, more capital (higher premium), less leverage
   - Rejected for MVP because: Equity signals are short-term, don't need long expiries

2. **Fixed DTE (e.g., always 7 days)**
   - Pros: Consistent time decay, easier to compare across trades
   - Cons: Requires DTE calculation, may not have exact match
   - Deferred to Phase 4 because: More complex, nearest weekly is close enough

3. **Quarterly Expiries**
   - Pros: Very far-dated, less time decay
   - Cons: Act more like equity (low leverage), defeats purpose of options
   - Rejected because: Not aligned with short-term signals

### Consequences

**Positive**:
- High theta capture (if signals are fast)
- Simple logic (no calculations)
- Best liquidity (weekly NIFTY/BANKNIFTY)

**Negative**:
- Frequent expiry events (rollover or close every week)
- High theta decay works against slow signals
- Time to expiry varies (3-7 days typically)

**Risks**:
- Signals are too slow, theta eats all profits
  - **Mitigation**: Phase 2 will reveal this quickly, can switch to monthly in Phase 4

### Implementation Notes
- `config/options_config.yaml → expiry_selection.method = "nearest_weekly"`
- `trade_mapper.py → get_nearest_weekly_expiry(entry_time)`
- Constraint: Min 3 DTE to enter (avoid <72 hours)
- Force close 24 hours before expiry

### Review Date
After Phase 2 - analyze avg hold time vs expiry timing

---

## Future ADRs (To Be Written)

- **ADR-007**: Pre-fetch vs On-Demand Data Fetching
- **ADR-008**: Parquet vs CSV for Data Storage
- **ADR-009**: Synthetic Volatility Model Selection (after Phase 1)
- **ADR-010**: Liquidity Filter Thresholds
- **ADR-011**: Fractional Lot Rounding Strategy
- **ADR-012**: Greeks Calculation Frequency
- **ADR-013**: [TBD based on implementation findings]

---

**Document Status**: Living document, updated as decisions are made

**Last Updated**: 2025-10-08
