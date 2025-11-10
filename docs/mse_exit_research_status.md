# MSE Exit Experimentation – Research Status
_Updated: 2025-11-07_

## 1. Objective
Establish parity between the live `mse_strategy.py` execution (0.05 peak-drop exit) and the backtester so we can study alternative exit percentages, quantify performance deltas across ~116 tickers (2022-01-01 → 2025-10-31 data span), and feed the results into portfolio construction experiments (anti-cascading vs all-trade universes, 5–6 ticker bundles).

---

## 2. Work Completed
| # | Work Item | Evidence / Path |
|---|-----------|-----------------|
| 1 | Pulled BIOCON + other missing tickers into `data/pools/2022-01-01_to_2025-08-31` and verified timestamps extend to 2025-10-31. | `scripts/check_actual_data_ranges.py` |
| 2 | Parameterized exit threshold CLI configs (`config/experiments/mse_exit_drop_*.yaml`) and ran six backtests (drop from peak: 1, 5, 20, 55, 80, 95%). | Raw outputs under `outputs/mse_exit_drop_XXX_*` |
| 3 | Merged scenario trades and computed summary stats (win-rate, avg duration, avg ₹ P&L, Sharpe/trade). | `outputs/mse_exit_merged/exit_scenario_summary.csv` |
| 4 | Ran full analysis stack for 5% exit (validation, trade-type, cascade, ticker ranking, anti-cascade portfolio pipeline) with emojis stripped to keep Windows CP1252 happy. | `analysis/output/mse_strategy_backtesting/20251107_154802/...` |
| 5 | Created “all trades” analysis config (`analysis/configs/mse_exit_drop_005_all_full.yaml`) to replay the portfolio pipeline without anti-cascade filtering and to compare 5- vs 6-ticker bundles. | config file |
| 6 | Implemented configurable top-50 variant + cascade skipping inside `analysis/portfolio_construction/scripts/01_corrected_anti_cascading_subset.py` (can now pick `top50_variant: all` + `exclude_same_direction: false`). | script |
| 7 | Standardized logs + scripts to ASCII (config loader, run orchestrator, merge utility) to stop encoding crashes. | `analysis/run.py`, `analysis/generic/...`, `utils/merge_trades.py` |

---

## 3. Current Outputs & Metrics
- **Scenario Sharpe sweep** (per-trade Sharpe from `exit_scenario_summary.csv`):
  - 1% drop: Sharpe 19.60, avg duration 27.9 min (12 293 trades).
  - 5% drop: Sharpe 19.09, avg duration 33.1 min (10 745 trades). ← baseline.
  - 95% drop: Sharpe 10.95, avg duration 127 min (5 207 trades).
- **5% exit generic analysis** (`analysis/reports/mse_strategy_backtesting/20251107_154802/generic/basic_eda/basic_eda_report.md`):
  - Win-rate 50.1 %, PF 1.87, avg trade duration 32.2 min.
  - Sell legs PF 1.92 vs Buy PF 1.82.
  - Time-of-day leader 15:00 bucket with 57.3 % WR.
- **Anti-cascade filtered universe (price < ₹2 000)**:
  - 44 656 trades, 30 tickers, heavy PSU/defensive mix (PNB, NTPC, AXISBANK, UBL, INDHOTEL recurring).
  - Top 5 Sharpe portfolios (5 tickers) hit Sharpe ≈1.29, annualized return 5.9–7.6 %, max DD 3.8–9.6 %.
- **Cascade vs Anti-cascade overlap** (`analysis/output/.../portfolio/ticker_ranking/cascade_comparison_summary.md`):
  - Only 38/50 overlap between all-trades and anti-cascading leaderboards → filtering materially changes constituents even if WR impact is small.

---

## 4. Current State
| Area | Status | Notes |
|-------|--------|-------|
| Scenario backtests (1–95 % exits) | ✅ Done | Merge + summary CSV complete. |
| Generic analysis for 5 % exit | ✅ Done | Under `analysis/output/.../20251107_154802/generic`. |
| Anti-cascade portfolio pipeline (5 tickers) | ✅ Done | `analysis/output/.../20251107_154802/portfolio` contains optimizer + PyPFO. |
| “All trades” portfolio pipeline (5/6 tickers) | 🚧 In-progress | Config + data ready; run timed out mid-flight while sector module was dumping logs. Need rerun (see §5). |
| Persona/runbook doc | ✅ This file |

---

## 5. Remaining Work / Next Steps
1. **Re-run full portfolio stack using all trades**  
   ```powershell
   .\.venv\Scripts\python.exe analysis\run.py --config analysis\configs\mse_exit_drop_005_all_full.yaml --targets portfolio --skip-merge
   ```  
   Let it finish (~8–10 min). Expect outputs under `analysis/output/mse_strategy_backtesting/20251107_154802_full/portfolio`.

2. **Compare universes**  
   - Pull top portfolios from both runs (`portfolio_optimizer/portfolio_performance_top50.csv`, `pypfopt_weights/*.md`, `equity_curves/portfolio_summary_stats.csv`).  
   - Record Sharpe/return/DD deltas, overlapping tickers, sector spread.

3. **Decide on base dataset for portfolio experimentation**  
   - If “all trades” curves outperform without extra volatility, adopt them.  
   - Otherwise keep anti-cascade as the conservative baseline and note rationale.

4. **Optional polish**  
   - Remove remaining emojis from `analysis/portfolio_construction/scripts/0*` to tighten Windows compatibility.  
   - Parameterize price-threshold + top50 variant via CLI flags if we plan more sweeps.

5. **Future experiments**  
   - Once best exit + universe confirmed, layer capital policy simulations (risk parity, exposure caps) mirroring GlobalRiskManager’s ~₹9.7k cash constraints.  
   - Validate specific live-problem tickers (BIOCON, JSWSTEEL, TATASTEEL), ensuring their trade counts exist in `data/pools/...`.

---

## 6. Key Paths / Artifacts
| Purpose | Path |
|---------|------|
| Scenario summary | `outputs/mse_exit_merged/exit_scenario_summary.csv` |
| Baseline merged trades | `analysis/output/mse_strategy_backtesting/20251107_154802/data/all_trades_merged.csv` |
| Baseline portfolio results | `analysis/output/mse_strategy_backtesting/20251107_154802/portfolio/...` |
| All-trades merged file | `analysis/output/mse_strategy_backtesting/20251107_154802_full/data/all_trades_merged.csv` |
| All-trades config | `analysis/configs/mse_exit_drop_005_all_full.yaml` |
| Live vs backtest logs | `trading_system_2025-11-07.log`, `src/strategies/mse_strategy_backtesting.py` |

---

## 7. Agent Persona & Collaboration Guidelines
- **Role**: Senior Full-Stack Quant Engineer (15+ yrs), systems mindset.  
  - Prioritize reproducibility, minimal diffs, and evidence-first decisions.
  - Treat live repo (`D:\Balcony\Trading\unified_trading_setup\trading_system_clean`) as read-only reference.  
- **Working agreements**:
  1. Keep `docs/TASKS.md` Work Log & Decision Log updated each session.
  2. Run `analysis/run.py` via `.venv\Scripts\python.exe` (Windows PowerShell).
  3. Prefer ASCII-only logs (no emojis) to avoid CP1252 errors.
  4. Document CLI commands + paths for every significant run; stash artifacts under `analysis/output/...` or `outputs/...`.
  5. Validate any assumption about data coverage using `scripts/check_actual_data_ranges.py` before adjusting configs.
- **Handoff expectations**:
  - Note whether all-trades portfolio run completed (if not, state partial outputs).
  - Highlight blockers explicitly (encoding, missing files, etc.).
  - Always attach run logs (`analysis/run_logs/.../*.md`) when handing over.

---

## 8. Quick Reference – Commands
```powershell
# Baseline analysis (merged + generic + portfolio anti-cascade)
.\.venv\Scripts\python.exe analysis\run.py --config analysis\configs\mse_exit_drop_005_all.yaml

# Portfolio-only rerun (anti-cascade baseline) after tweaking scripts
.\.venv\Scripts\python.exe analysis\run.py --config analysis\configs\mse_exit_drop_005_all.yaml --targets portfolio --skip-merge

# All-trades portfolio experiment (5 & 6 tickers)
.\.venv\Scripts\python.exe analysis\run.py --config analysis\configs\mse_exit_drop_005_all_full.yaml --targets portfolio --skip-merge

# Merge trades once per config
.\.venv\Scripts\python.exe utils\merge_trades.py --config analysis\configs\mse_exit_drop_005_all_full.yaml
```

---

## 9. Open Questions
1. Does the all-trades universe deliver materially better Sharpe/DD than the anti-cascade subset?
2. Do we need per-ticker stop-loss/target logic before matching live logs, or is exit-threshold tuning sufficient?
3. Should JSWSTEEL margin rejections be modeled via capital policy (since live cash ≈ ₹9.7k)?

Document owner: current agent (update timestamp when modifying). Feel free to append findings under new dated headings. 
