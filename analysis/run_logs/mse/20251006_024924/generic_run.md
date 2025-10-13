# Generic Run Report

- **Timestamp**: 2025-10-07 20:21:47
- **Strategy**: mse
- **Run ID**: 20251006_024924
- **Date Range**: 2022-01-01_to_2025-08-31
- **Label**: mse_baseline_analysis

| Module | Status | Duration | Outputs | Notes |
|---|---|---|---|---|
| `validation_check` | ✅ | 2.1s | — | Completed successfully. |
| `ticker_ranking` | ✅ | 2.3s | — | Completed successfully. |
| `cascade_analysis` | ✅ | 8.4s | — | Completed successfully. |
| `basic_eda` | ✅ | 2.0s | — | Completed successfully. |

## Diagnostics
### validation_check stdout
```text
✅ Loaded config from: /mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/StrategyLab-master/analysis/configs/example_mse_config.yaml
   Run ID: 20251006_024924
   Strategy: mse
   Date Range: 2022-01-01_to_2025-08-31
   Strategy trades dir: outputs/20251006_024924/mse/2022-01-01_to_2025-08-31/data/strategy_trades
   Base data dir: outputs/20251006_024924/mse/2022-01-01_to_2025-08-31/data/base_data
📊 Loading trade data from: all_trades_merged.csv
✅ Loaded 60,629 trades
   Date range: 2022-01-03 09:35:00+05:30 to 2025-08-29 15:10:00+05:30
   Tickers: 24
✅ All required columns present.

📈 Overall stats: Trades=60,629, WinRate=49.19%, P&L=₹101,403.46

📝 Validation report saved → analysis/reports/mse/20251006_024924/generic/validation_check/validation_check_validation_report.md

✅ Validation check complete.
```
### ticker_ranking stdout
```text
✅ Loaded config from: /mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/StrategyLab-master/analysis/configs/example_mse_config.yaml
   Run ID: 20251006_024924
   Strategy: mse
   Date Range: 2022-01-01_to_2025-08-31
   Strategy trades dir: outputs/20251006_024924/mse/2022-01-01_to_2025-08-31/data/strategy_trades
   Base data dir: outputs/20251006_024924/mse/2022-01-01_to_2025-08-31/data/base_data
📊 Loading trade data for ticker ranking analysis...
📊 Loading trade data from: all_trades_merged.csv
✅ Loaded 60,629 trades
   Date range: 2022-01-03 09:35:00+05:30 to 2025-08-29 15:10:00+05:30
   Tickers: 24
✅ Loaded 60,629 trades covering 24 tickers

📊 Calculating ticker-level metrics...
✅ Metrics computed for 24 tickers.

🧮 Applied composite score weights:
   • Profitability: 30%
   • Risk Management: 25%
   • Consistency: 20%
   • Efficiency: 15%
   • Frequency: 10%

🏆 Tier distribution:
   • Tier 1 - Excellent: 1 tickers (4.2%)
   • Tier 2 - Good: 10 tickers (41.7%)
   • Tier 3 - Average: 8 tickers (33.3%)
   • Tier 4 - Poor: 5 tickers (20.8%)

🥇 Top 50 tickers preview:
    1. ULTRACEMCO | score 82.0 | P&L ₹21,648
    2. MARUTI | score 69.6 | P&L ₹20,998
    3. SBIN | score 66.9 | P&L ₹1,755
    4. RELIANCE | score 66.8 | P&L ₹2,738
    5. LT | score 66.6 | P&L ₹6,652
    6. TCS | score 65.3 | P&L ₹6,486
    7. KOTAKBANK | score 58.4 | P&L ₹3,360
    8. TATASTEEL | score 57.3 | P&L ₹357
    9. ITC | score 57.1 | P&L ₹689
   10. INFY | score 56.3 | P&L ₹2,940

🔻 Bottom 50 tickers preview:
    1. ULTRACEMCO | score 82.0 | P&L ₹21,648
    2. MARUTI | score 69.6 | P&L ₹20,998
    3. SBIN | score 66.9 | P&L ₹1,755
    4. RELIANCE | score 66.8 | P&L ₹2,738
    5. LT | score 66.6 | P&L ₹6,652
    6. TCS | score 65.3 | P&L ₹6,486
    7. KOTAKBANK | score 58.4 | P&L ₹3,360
    8. TATASTEEL | score 57.3 | P&L ₹357
    9. ITC | score 57.1 | P&L ₹689
   10. INFY | score 56.3 | P&L ₹2,940

📊 Tier summary:
                   total_pnl                     avg_profit_per_trade  ... total_trades composite_score              
                       count      mean       sum                 mean  ...         mean            mean    min    max
tier                                                                   ...                                           
Tier 1 - Excellent         1  21647.65  21647.65                 8.78  ...      2465.00           81.95  81.95  81.95
Tier 2 - Good             10   5267.98  52679.82                 2.13  ...      2482.80           62.01  55.76  69.61
Tier 3 - Average           8   2095.47  16763.72                 0.81  ...      2562.88           52.65  48.24  54.15
Tier 4 - Poor              5   2062.46  10312.28                 0.80  ...      2566.60           40.10  32.11  47.60

[4 rows x 12 columns]

💧 Liquidity analysis:
                    Ticker_Count  Avg_Total_PnL  Avg_Profit_Per_Trade  Avg_Win_Rate  Avg_Composite_Score
liquidity_category                                                                                      
Very Low                       0            NaN                   NaN           NaN                  NaN
Low                            0            NaN                   NaN           NaN                  NaN
Medium                        24        4225.14                  1.69         49.21                55.16
High                           0            NaN                   NaN           NaN                  NaN
Very High                      0            NaN                   NaN           NaN                  NaN

💾 Saved ranking table → analysis/output/mse/20251006_024924/generic/ticker_ranking/ticker_ranking_ticker_scores.csv
💾 Saved summary JSON → analysis/output/mse/20251006_024924/generic/ticker_ranking/ticker_analysis_summary.json

✅ TICKER RANKING ANALYSIS COMPLETE
   Total tickers analysed: 24
   Top performer: ULTRACEMCO (score 82.0)
   Results directory: analysis/output/mse/20251006_024924/generic/ticker_ranking
```
### cascade_analysis stdout
```text
============================================================
CASCADE ANALYSIS - YAML Config Driven
============================================================
✅ Loaded config from: /mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/StrategyLab-master/analysis/configs/example_mse_config.yaml
   Run ID: 20251006_024924
   Strategy: mse
   Date Range: 2022-01-01_to_2025-08-31
   Strategy trades dir: outputs/20251006_024924/mse/2022-01-01_to_2025-08-31/data/strategy_trades
   Base data dir: outputs/20251006_024924/mse/2022-01-01_to_2025-08-31/data/base_data
📊 Loading trade data from: all_trades_merged.csv
✅ Loaded 60,629 trades
   Date range: 2022-01-03 09:35:00+05:30 to 2025-08-29 15:10:00+05:30
   Tickers: 24

🔍 Validating trade data...

🏷️  TAGGING TRADES WITH CASCADE PATTERNS
============================================================
📊 Processing 60,629 trades...
✅ Tagging complete!

📊 CASCADE TAG DISTRIBUTION:
FIRST_TRADE_OF_DAY          21402
LOSING_CASCADE_SAME_DIR     13728
WINNING_CASCADE_SAME_DIR    12337
LOSING_CASCADE_OPP_DIR       9231
WINNING_CASCADE_OPP_DIR      3907
FIRST_TRADE_FOR_TICKER         23
FIRST_TRADE_OVERALL             1

📊 CASCADE PERFORMANCE ANALYSIS
============================================================

📈 PERFORMANCE BY CASCADE TAG:
                          count  total_profit  avg_profit  profit_std  win_rate
cascade_tag                                                                    
FIRST_TRADE_FOR_TICKER       23     -8.050000      -0.350       5.462      47.8
FIRST_TRADE_OF_DAY        21402  38301.601562       1.790      19.433      48.3
FIRST_TRADE_OVERALL           1     -0.250000      -0.250         NaN       0.0
LOSING_CASCADE_OPP_DIR     9231  18191.500000       1.971      14.961      50.6
LOSING_CASCADE_SAME_DIR   13728  20019.589844       1.458      13.287      47.6
WINNING_CASCADE_OPP_DIR    3907   7807.040039       1.998      15.239      52.3
WINNING_CASCADE_SAME_DIR  12337  17092.039062       1.385      11.708      50.4

🔍 CASCADE vs NON-CASCADE:
   Cascade trades: 39,203 (49.7% WR)
   Non-cascade trades: 21,426 (48.3% WR)
   Difference: +1.4%

🏆 TRADES AFTER WINNING:
   Count: 29,821
   Win Rate: 49.5%
   Avg Profit: ₹1.58

❌ TRADES AFTER LOSING:
   Count: 29,844
   Win Rate: 49.0%
   Avg Profit: ₹1.80

⏰ TIME GAP ANALYSIS:
                   count  avg_profit  win_rate
time_gap_category                             
1-2_HOURS          10426       1.888      49.5
15-30_MIN           3700       1.622      48.7
2-4_HOURS           5630       2.121      52.9
30-60_MIN           8627       1.516      47.8
4+_HOURS             810       2.079      56.3
5-15_MIN           10010       1.071      49.5

💾 Saved tagged trades to: analysis/output/mse/20251006_024924/generic/cascade_analysis/cascade_analysis_cascade_tags.csv
💾 Saved statistics to: analysis/output/mse/20251006_024924/generic/cascade_analysis/cascade_analysis_cascade_metrics.json

============================================================
✅ CASCADE ANALYSIS COMPLETE!
============================================================

Outputs:
  - Tagged Trades: analysis/output/mse/20251006_024924/generic/cascade_analysis/cascade_analysis_cascade_tags.csv
  - Statistics: analysis/output/mse/20251006_024924/generic/cascade_analysis/cascade_analysis_cascade_metrics.json
  - Report: analysis/reports/mse/20251006_024924/generic/cascade_analysis/cascade_analysis_cascade_insights.md
```
### basic_eda stdout
```text
============================================================
BASIC EDA - Exploratory Data Analysis
============================================================
✅ Loaded config from: /mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/StrategyLab-master/analysis/configs/example_mse_config.yaml
   Run ID: 20251006_024924
   Strategy: mse
   Date Range: 2022-01-01_to_2025-08-31
   Strategy trades dir: outputs/20251006_024924/mse/2022-01-01_to_2025-08-31/data/strategy_trades
   Base data dir: outputs/20251006_024924/mse/2022-01-01_to_2025-08-31/data/base_data
📊 Loading trade data from: all_trades_merged.csv
✅ Loaded 60,629 trades
   Date range: 2022-01-03 09:35:00+05:30 to 2025-08-29 15:10:00+05:30
   Tickers: 24

🔍 Validating trade data...

📊 OVERALL STATISTICS
============================================================
Total Trades: 60,629
  Winning: 29,822 (49.2%)
  Losing: 29,844 (50.8%)
  Breakeven: 963

Profitability:
  Total P&L: ₹101,403.46
  Average P&L: ₹1.67
  Profit Factor: 1.75
  Avg Win: ₹7.93
  Avg Loss: ₹-4.53

Duration:
  Average: 51.1 minutes (0.9 hours)

📊 TRADE TYPE DISTRIBUTION
============================================================
Buy Trades:
  Count: 30,327 (50.0%)
  Win Rate: 48.6%
  Total Profit: ₹53,164.24
  Avg Profit: ₹1.75

Sell Trades:
  Count: 30,302 (50.0%)
  Win Rate: 49.7%
  Total Profit: ₹48,239.23
  Avg Profit: ₹1.59

📊 TICKER-LEVEL PERFORMANCE
============================================================

Top 10 Tickers by Total Profit:
            trade_count  total_profit  avg_profit  avg_duration   win_rate
ticker                                                                    
ULTRACEMCO         2465  21647.650391        8.78     51.660000  51.724138
MARUTI             2479  20997.900391        8.47     51.220001  49.576442
TITAN              2436   6705.100098        2.75     51.610001  49.712644
LT                 2455   6652.399902        2.71     53.459999  51.853360
TCS                2490   6485.549805        2.60     50.340000  51.044177
HINDUNILVR         2582   4428.569824        1.72     51.900002  49.457785
ASIANPAINT         2531   4333.950195        1.71     49.560001  48.241802
ADANIPORTS         2591   4038.100098        1.56     50.869999  50.328059
KOTAKBANK          2592   3359.800049        1.30     49.240002  49.768519
INFY               2419   2940.250000        1.22     53.549999  50.434064

Bottom 5 Tickers by Total Profit:
           trade_count  total_profit  avg_profit  avg_duration   win_rate
ticker                                                                   
ITC               2558    688.700012        0.27     51.020000  48.514464
POWERGRID         2558    587.969971        0.23     50.840000  48.279906
ONGC              2498    559.030029        0.22     50.570000  49.119295
WIPRO             2409    445.529999        0.18     53.910000  48.152760
TATASTEEL         2381    357.130005        0.15     53.169998  49.433011

📅 TIME-OF-DAY PERFORMANCE
============================================================
Top 5 Hours by Win Rate:
  15:00 → 58.7% win rate (₹0.77 avg)
  14:00 → 50.9% win rate (₹1.33 avg)
  12:00 → 49.4% win rate (₹1.92 avg)
  13:00 → 48.8% win rate (₹1.85 avg)
  09:00 → 48.2% win rate (₹1.67 avg)

💾 Saved statistics to: analysis/output/mse/20251006_024924/generic/basic_eda/basic_eda_summary.json
💾 Saved ticker performance to: analysis/output/mse/20251006_024924/generic/basic_eda/basic_eda_ticker_performance.csv

📄 Report saved to: analysis/reports/mse/20251006_024924/generic/basic_eda/basic_eda_report.md

============================================================
✅ BASIC EDA COMPLETE!
============================================================
```