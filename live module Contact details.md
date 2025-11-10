# Live Module Contact Details

Reference sheet for bridging the `backtester` repo with the live stack in `D:\Balcony\Trading\unified_trading_setup\trading_system_clean`. The live project **must remain read-only**; use it for context only.

## Location & Access
- Root: `D:\Balcony\Trading\unified_trading_setup\trading_system_clean`
- Python sources: `live_module/src`
- Configs: `config/`
- Logs: `logs/daily`, `logs/live`
- Data/artifacts: `live_module/data`

## Critical Modules To Mirror in Backtests

| Capability | Live Module Source | Why It Matters |
|------------|-------------------|----------------|
| Position persistence & recovery | `live_module/src/position/unified_position_manager.py` | Writes `live_module/data/positions/positions.json` after every mutation (atomic save). Backtests should source canonical position state from here when validating mode logic or MACD peak tracking. |
| Strategy signal implementation | `live_module/src/strategies/mse_strategy.py` | Ensures indicator math, entry/exit thresholds, and mode switching match the production “MSE_Strategy.” |
| Global risk limits | `live_module/src/risk/global_risk_manager.py` plus `config/ticker_config.csv` | Documents exposure caps (₹100k), quantity caps (≤500), and conflict checks that run before orders are submitted. |
| Position sizing & leverage | `live_module/src/central_ops/position_sizer.py`, `live_module/src/central_ops/leverage_calculator.py`, `live_module/src/central_ops/simple_broker_reconciliation.py` | Allocation model (TATASTEEL 30 %, JSWSTEEL 40 %, BIOCON 30 %), MIS leverage (5×) and 10 % capital safety buffer drive order quantities; needed to reproduce realistic fills during backtests. |
| Data freshness gating | `live_module/src/data/strategy_aware_freshness_tracker.py`, `live_module/src/data/polling_data_orchestrator.py` | Mixed-timeframe allowance (fresh 5 min even if 15 min stale) explains signal timing differences vs. strict backtests. |
| Scheduler cadence | `live_module/src/data/polling_scheduler.py` | Polls every five minutes; ties directly to candle alignment assumptions in the backtester. |
| Broker orchestration | `live_module/src/central_ops/unified_order_executor.py`, `live_module/src/brokers/*` | Needed when verifying rejection patterns (e.g., margin shortfall) or post-order sync routines. |

## Validation Checklist When Comparing Backtests vs Live
1. **Indicator parity** – Confirm EMA/MACD helpers in backtests replay the same lag/rounding as `live_module/src/strategies/mse_strategy.py`.
2. **Position state** – Use `live_module/data/positions/positions.json` snapshots to reproduce entry/exit mode, peak MACD, and open quantity assumptions.
3. **Capital & sizing** – Match live allocations (30/40/30) and 5× MIS leverage. Verify `SimpleBrokerReconciliation.get_available_capital()` output (Flattrade `cash`) used in logs.
4. **Risk gates** – Respect exposure, quantity, and “single strategy per symbol” constraints defined in `GlobalRiskManager`.
5. **Freshness rules** – Mirror `StrategyAwareFreshnessTracker`’s smart readiness logic (shortest timeframe freshness, 09:15‑09:30 leniency, mixed-timeframe allowance).
6. **Scheduler timing** – Align backtest signal timestamps with the 5‑minute polling schedule logged by `polling_scheduler`.
7. **Broker errors** – When backtests show fills but live logs show rejects, cross-check `logs/daily/trading_system_YYYY-MM-DD.log` for capital or circuit-breaker causes before flagging logic gaps.

## Usage Guidelines
- Treat every path above as **read-only**. Copy snippets or configs into the backtester only after confirming they are safe to expose.
- When documentation drifts, update this file rather than editing anything under `trading_system_clean`.
- Cite exact modules/lines (using `path:line`) in analyses so reviewers can trace back to live implementations quickly.
