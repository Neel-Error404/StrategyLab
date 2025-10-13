# Portfolio Run Report

- **Timestamp**: 2025-10-07 20:32:06
- **Strategy**: mse
- **Run ID**: 20251006_024924
- **Date Range**: 2022-01-01_to_2025-08-31
- **Label**: mse_baseline_analysis

| Module | Status | Duration | Outputs | Notes |
|---|---|---|---|---|
| `equity_curves` | ✅ | 7.7s | — | Completed successfully. |
| `pypfopt_weights` | ✅ | 2.5s | — | Completed successfully. |
| `portfolio_optimizer` | ✅ | 79.5s | — | Completed successfully. |
| `combination_generator` | ✅ | 17.7s | — | Completed successfully. |
| `sector_classification` | ✅ | 1.4s | — | Completed successfully. |
| `anti_cascade_filter` | ✅ | 4.1s | — | Completed successfully. |
| `ticker_ranking` | ✅ | 5.8s | — | Completed successfully. |

## Diagnostics
### equity_curves stdout
```text
✅ Loaded config from: /mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/StrategyLab-master/analysis/configs/example_mse_config.yaml
   Run ID: 20251006_024924
   Strategy: mse
   Date Range: 2022-01-01_to_2025-08-31
   Strategy trades dir: outputs/20251006_024924/mse/2022-01-01_to_2025-08-31/data/strategy_trades
   Base data dir: outputs/20251006_024924/mse/2022-01-01_to_2025-08-31/data/base_data
🚀 STARTING EQUITY CURVE & VISUALIZATION GENERATOR
================================================================================
📁 Config: /mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/StrategyLab-master/analysis/configs/example_mse_config.yaml
📊 Strategy: mse
📅 Date Range: 2022-01-01_to_2025-08-31
================================================================================
📊 Portfolio sizes: [5]
📊 Top N per size: 5
📁 Output directory: analysis/output/mse/20251006_024924/portfolio/equity_curves

📂 Loading data...
✅ Loaded 24,546 trades
✅ Loaded sector mapping for 17 tickers

================================================================================
📊 PROCESSING 5-TICKER PORTFOLIOS
================================================================================
📈 Processing Top 5 portfolios from 5-ticker results...

Processing: 5T-#1 (Sharpe 0.84)
  Tickers: AXISBANK, HCLTECH, INFY, SUNPHARMA, KOTAKBANK
✅ Saved: monthly_returns_5T_top1.png
✅ Saved: rolling_sharpe_5T_top1.png
✅ Saved: sector_allocation_5T-1_(Sharpe_0.84).png

Processing: 5T-#2 (Sharpe 0.83)
  Tickers: NTPC, AXISBANK, HCLTECH, SUNPHARMA, KOTAKBANK

Processing: 5T-#3 (Sharpe 0.82)
  Tickers: NTPC, AXISBANK, NESTLEIND, HCLTECH, KOTAKBANK

Processing: 5T-#4 (Sharpe 0.80)
  Tickers: AXISBANK, NESTLEIND, HCLTECH, INFY, KOTAKBANK

Processing: 5T-#5 (Sharpe 0.78)
  Tickers: NTPC, AXISBANK, INFY, SUNPHARMA, KOTAKBANK
✅ Saved: equity_curves_5ticker.png

📊 Generating summary statistics...

✅ Summary Statistics:
          Portfolio  Total Return (%)  Annualized Return (%)  Annualized Volatility (%)  Sharpe Ratio  Max Drawdown (%)  Win Rate (%)  Final Value (₹)
5T-#1 (Sharpe 0.84)             12.65                   3.37                       4.08         0.826             -4.88         51.77         112746.0
5T-#2 (Sharpe 0.83)             13.04                   3.47                       4.26         0.816             -8.47         52.32         113186.0
5T-#3 (Sharpe 0.82)             12.81                   3.41                       4.25         0.804             -7.93         52.99         112899.0
5T-#4 (Sharpe 0.80)             12.03                   3.21                       4.10         0.785             -4.76         52.10         112082.0
5T-#5 (Sharpe 0.78)             11.94                   3.19                       4.20         0.760             -7.17         52.32         112087.0

✅ Saved: portfolio_summary_stats.csv


================================================================================
✅ VISUALIZATION COMPLETE
📁 All visualizations saved to: /mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/StrategyLab-master/analysis/output/mse/20251006_024924/portfolio/equity_curves
================================================================================
```
### pypfopt_weights stdout
```text
✅ Loaded config from: /mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/StrategyLab-master/analysis/configs/example_mse_config.yaml
   Run ID: 20251006_024924
   Strategy: mse
   Date Range: 2022-01-01_to_2025-08-31
   Strategy trades dir: outputs/20251006_024924/mse/2022-01-01_to_2025-08-31/data/strategy_trades
   Base data dir: outputs/20251006_024924/mse/2022-01-01_to_2025-08-31/data/base_data
🚀 STARTING PYPFOPT OPTIMAL WEIGHT ALLOCATION
================================================================================
📁 Config: /mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/StrategyLab-master/analysis/configs/example_mse_config.yaml
📊 Strategy: mse
📅 Date Range: 2022-01-01_to_2025-08-31
================================================================================
📊 Portfolio sizes to optimize: [5]
📊 Top N portfolios per size: 10

📂 Loading anti-cascading trade data...
✅ Loaded 24,546 anti-cascading trades

================================================================================
Processing 5-ticker portfolios...
================================================================================
✅ Loaded 10 portfolios for optimization

================================================================================
📊 PROCESSING 5-TICKER PORTFOLIOS
================================================================================
Total portfolios to optimize: 10

Portfolio_1: AXISBANK, HCLTECH, INFY, SUNPHARMA, KOTAKBANK
ERROR in LDL_factor: Error in KKT matrix LDL factorization when computing the nonzero elements. The problem seems to be non-convex
ERROR in osqp_setup: KKT matrix factorization.
The problem seems to be non-convex.
⚠️  Max Sharpe failed: 4
ERROR in LDL_factor: Error in KKT matrix LDL factorization when computing the nonzero elements. The problem seems to be non-convex
ERROR in osqp_setup: KKT matrix factorization.
The problem seems to be non-convex.
⚠️  Min Volatility failed: 4
  Equal Weight (1/N)             → Sharpe: 1.0246, Return: 4.86%, Vol: 4.74%

Portfolio_2: NTPC, AXISBANK, HCLTECH, SUNPHARMA, KOTAKBANK
ERROR in LDL_factor: Error in KKT matrix LDL factorization when computing the nonzero elements. The problem seems to be non-convex
ERROR in osqp_setup: KKT matrix factorization.
The problem seems to be non-convex.
⚠️  Max Sharpe failed: 4
ERROR in LDL_factor: Error in KKT matrix LDL factorization when computing the nonzero elements. The problem seems to be non-convex
ERROR in osqp_setup: KKT matrix factorization.
The problem seems to be non-convex.
⚠️  Min Volatility failed: 4
  Equal Weight (1/N)             → Sharpe: 0.8032, Return: 4.09%, Vol: 5.09%

Portfolio_3: NTPC, AXISBANK, NESTLEIND, HCLTECH, KOTAKBANK
ERROR in LDL_factor: Error in KKT matrix LDL factorization when computing the nonzero elements. The problem seems to be non-convex
ERROR in osqp_setup: KKT matrix factorization.
The problem seems to be non-convex.
⚠️  Max Sharpe failed: 4
ERROR in LDL_factor: Error in KKT matrix LDL factorization when computing the nonzero elements. The problem seems to be non-convex
ERROR in osqp_setup: KKT matrix factorization.
The problem seems to be non-convex.
⚠️  Min Volatility failed: 4
  Equal Weight (1/N)             → Sharpe: 0.6917, Return: 3.50%, Vol: 5.05%

Portfolio_4: AXISBANK, NESTLEIND, HCLTECH, INFY, KOTAKBANK
ERROR in LDL_factor: Error in KKT matrix LDL factorization when computing the nonzero elements. The problem seems to be non-convex
ERROR in osqp_setup: KKT matrix factorization.
The problem seems to be non-convex.
⚠️  Max Sharpe failed: 4
ERROR in LDL_factor: Error in KKT matrix LDL factorization when computing the nonzero elements. The problem seems to be non-convex
ERROR in osqp_setup: KKT matrix factorization.
The problem seems to be non-convex.
⚠️  Min Volatility failed: 4
  Equal Weight (1/N)             → Sharpe: 0.8541, Return: 3.99%, Vol: 4.67%

Portfolio_5: NTPC, AXISBANK, INFY, SUNPHARMA, KOTAKBANK
ERROR in LDL_factor: Error in KKT matrix LDL factorization when computing the nonzero elements. The problem seems to be non-convex
ERROR in osqp_setup: KKT matrix factorization.
The problem seems to be non-convex.
⚠️  Max Sharpe failed: 4
ERROR in LDL_factor: Error in KKT matrix LDL factorization when computing the nonzero elements. The problem seems to be non-convex
ERROR in osqp_setup: KKT matrix factorization.
The problem seems to be non-convex.
⚠️  Min Volatility failed: 4
  Equal Weight (1/N)             → Sharpe: 0.9159, Return: 4.56%, Vol: 4.98%

Portfolio_6: AXISBANK, RELIANCE, HCLTECH, INFY, KOTAKBANK
ERROR in LDL_factor: Error in KKT matrix LDL factorization when computing the nonzero elements. The problem seems to be non-convex
ERROR in osqp_setup: KKT matrix factorization.
The problem seems to be non-convex.
⚠️  Max Sharpe failed: 4
ERROR in LDL_factor: Error in KKT matrix LDL factorization when computing the nonzero elements. The problem seems to be non-convex
ERROR in osqp_setup: KKT matrix factorization.
The problem seems to be non-convex.
⚠️  Min Volatility failed: 4
  Equal Weight (1/N)             → Sharpe: 0.7869, Return: 3.87%, Vol: 4.92%

Portfolio_7: ITC, AXISBANK, HCLTECH, INFY, KOTAKBANK
ERROR in LDL_factor: Error in KKT matrix LDL factorization when computing the nonzero elements. The problem seems to be non-convex
ERROR in osqp_setup: KKT matrix factorization.
The problem seems to be non-convex.
⚠️  Max Sharpe failed: 4
  Equal Weight (1/N)             → Sharpe: 0.9714, Return: 4.58%, Vol: 4.72%
  Min Volatility                 → Sharpe: nan, Return: nan%, Vol: 33494.60%

Portfolio_8: NTPC, AXISBANK, HCLTECH, INFY, KOTAKBANK
ERROR in LDL_factor: Error in KKT matrix LDL factorization when computing the nonzero elements. The problem seems to be non-convex
ERROR in osqp_setup: KKT matrix factorization.
The problem seems to be non-convex.
⚠️  Max Sharpe failed: 4
ERROR in LDL_factor: Error in KKT matrix LDL factorization when computing the nonzero elements. The problem seems to be non-convex
ERROR in osqp_setup: KKT matrix factorization.
The problem seems to be non-convex.
⚠️  Min Volatility failed: 4
  Equal Weight (1/N)             → Sharpe: 0.8445, Return: 4.42%, Vol: 5.23%

Portfolio_9: WIPRO, NTPC, AXISBANK, HCLTECH, KOTAKBANK
ERROR in LDL_factor: Error in KKT matrix LDL factorization when computing the nonzero elements. The problem seems to be non-convex
ERROR in osqp_setup: KKT matrix factorization.
The problem seems to be non-convex.
⚠️  Max Sharpe failed: 4
  Equal Weight (1/N)             → Sharpe: 0.6626, Return: 3.56%, Vol: 5.37%
  Min Volatility                 → Sharpe: nan, Return: nan%, Vol: 33708.00%

Portfolio_10: NTPC, AXISBANK, NESTLEIND, INFY, KOTAKBANK
ERROR in LDL_factor: Error in KKT matrix LDL factorization when computing the nonzero elements. The problem seems to be non-convex
ERROR in osqp_setup: KKT matrix factorization.
The problem seems to be non-convex.
⚠️  Max Sharpe failed: 4
ERROR in LDL_factor: Error in KKT matrix LDL factorization when computing the nonzero elements. The problem seems to be non-convex
ERROR in osqp_setup: KKT matrix factorization.
The problem seems to be non-convex.
⚠️  Min Volatility failed: 4
  Equal Weight (1/N)             → Sharpe: 0.6989, Return: 3.44%, Vol: 4.92%

================================================================================
📈 OPTIMIZATION SUMMARY (5 tickers)
================================================================================

Equal Weight (1/N):
  Avg Sharpe: 0.8254
  Avg Return: 4.09%
  Avg Volatility: 4.97%

Min Volatility:
  Avg Sharpe: nan
  Avg Return: nan%
  Avg Volatility: 33601.30%

================================================================================
💡 IMPROVEMENT OVER EQUAL WEIGHT
================================================================================
Min Volatility                 → +nan% Sharpe improvement

💾 SAVING OPTIMIZATION RESULTS
============================================================
✅ Optimal weights saved: optimal_weights_5ticker.csv
✅ Summary report saved: pypfopt_summary_5ticker.md
📁 Location: analysis/output/mse/20251006_024924/portfolio/pypfopt_weights

🎉 PYPFOPT OPTIMIZATION COMPLETED!
📊 Processed 1 portfolio sizes
```
### portfolio_optimizer stdout
```text
✅ Loaded config from: /mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/StrategyLab-master/analysis/configs/example_mse_config.yaml
   Run ID: 20251006_024924
   Strategy: mse
   Date Range: 2022-01-01_to_2025-08-31
   Strategy trades dir: outputs/20251006_024924/mse/2022-01-01_to_2025-08-31/data/strategy_trades
   Base data dir: outputs/20251006_024924/mse/2022-01-01_to_2025-08-31/data/base_data
🚀 STARTING PORTFOLIO OPTIMIZATION ENGINE
================================================================================
📁 Config: /mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/StrategyLab-master/analysis/configs/example_mse_config.yaml
📊 Strategy: mse
📅 Date Range: 2022-01-01_to_2025-08-31
================================================================================
🚀 PORTFOLIO OPTIMIZATION ENGINE
================================================================================
📊 Loading valid combinations and trade data...
✅ Loaded 5,054 valid 5-ticker combinations

✅ Total combinations to evaluate: 5,054
✅ Portfolio sizes: [5]
✅ Loaded 24,546 anti-cascading trades
✅ Date range: 2022-01-03 to 2025-08-29

🎯 PORTFOLIO OPTIMIZATION IN PROGRESS
================================================================================
💡 Evaluating portfolio-level performance for all combinations

📊 Processing 5,054 portfolio combinations...
   Processed: 1,000/5,054 (19.8%) | Valid: 1,000
   Processed: 2,000/5,054 (39.6%) | Valid: 2,000
   Processed: 3,000/5,054 (59.4%) | Valid: 3,000
   Processed: 4,000/5,054 (79.1%) | Valid: 4,000
   Processed: 5,000/5,054 (98.9%) | Valid: 5,000

✅ Portfolio optimization complete!
   Total combinations processed: 5,054
   Valid portfolios with performance data: 5,054

🏆 TOP 50 PORTFOLIO PERFORMERS
================================================================================
📊 Ranked by Portfolio-Level Sharpe Ratio

📈 TOP 50 PORTFOLIOS BY SHARPE RATIO:
----------------------------------------------------------------------------------------------------------------------------------
Rank   Size   Sharpe   PF     WinRate   Ann.Ret    Ann.Vol    MaxDD    Tickers                                           
----------------------------------------------------------------------------------------------------------------------------------
1      5        0.839  1.16     51.8%      3.42%      4.08%    -4.9% AXISBANK, HCLTECH, INFY, SUNPHARMA, KOTAKBANK     
2      5        0.832  1.16     52.4%      3.54%      4.25%    -8.5% NTPC, AXISBANK, HCLTECH, SUNPHARMA, KOTAKBANK     
3      5        0.817  1.16     53.0%      3.47%      4.24%    -7.9% NTPC, AXISBANK, NESTLEIND, HCLTECH, KOTAKBANK     
4      5        0.796  1.15     52.2%      3.26%      4.09%    -4.8% AXISBANK, NESTLEIND, HCLTECH, INFY, KOTAKBANK     
5      5        0.779  1.15     52.4%      3.27%      4.19%    -7.2% NTPC, AXISBANK, INFY, SUNPHARMA, KOTAKBANK        
6      5        0.776  1.15     52.7%      3.32%      4.29%    -4.6% AXISBANK, RELIANCE, HCLTECH, INFY, KOTAKBANK      
7      5        0.767  1.15     50.9%      3.15%      4.10%    -6.2% ITC, AXISBANK, HCLTECH, INFY, KOTAKBANK           
8      5        0.758  1.15     51.9%      3.50%      4.63%    -8.6% NTPC, AXISBANK, HCLTECH, INFY, KOTAKBANK          
9      5        0.754  1.15     51.4%      3.49%      4.63%    -6.1% WIPRO, NTPC, AXISBANK, HCLTECH, KOTAKBANK         
10     5        0.747  1.14     53.1%      3.12%      4.17%    -6.9% NTPC, AXISBANK, NESTLEIND, INFY, KOTAKBANK        
11     5        0.733  1.14     51.7%      3.39%      4.63%    -4.1% WIPRO, AXISBANK, HCLTECH, INFY, KOTAKBANK         
12     5        0.731  1.14     50.8%      3.14%      4.29%    -7.3% NTPC, AXISBANK, HCLTECH, INFY, SUNPHARMA          
13     5        0.728  1.14     52.4%      2.79%      3.83%    -4.2% ITC, AXISBANK, HCLTECH, SUNPHARMA, KOTAKBANK      
14     5        0.725  1.14     53.5%      2.73%      3.77%    -4.2% AXISBANK, NESTLEIND, INFY, SUNPHARMA, KOTAKBANK   
15     5        0.717  1.14     51.5%      3.20%      4.47%    -5.0% JSWSTEEL, AXISBANK, HCLTECH, INFY, KOTAKBANK      
16     5        0.716  1.13     51.4%      3.05%      4.26%    -7.2% NTPC, AXISBANK, NESTLEIND, HCLTECH, INFY          
17     5        0.707  1.14     52.9%      2.93%      4.14%    -7.4% NTPC, AXISBANK, NESTLEIND, SUNPHARMA, KOTAKBANK   
18     5        0.703  1.13     52.7%      2.71%      3.85%    -5.6% AXISBANK, NESTLEIND, HCLTECH, SUNPHARMA, KOTAKBANK
19     5        0.700  1.13     52.4%      2.89%      4.13%    -4.2% WIPRO, AXISBANK, NESTLEIND, HCLTECH, KOTAKBANK    
20     5        0.696  1.14     52.4%      3.11%      4.46%    -8.0% NTPC, ITC, AXISBANK, HCLTECH, KOTAKBANK           
21     5        0.690  1.13     50.4%      2.74%      3.97%    -5.3% ITC, AXISBANK, HCLTECH, INFY, SUNPHARMA           
22     5        0.688  1.13     51.3%      2.87%      4.18%    -4.8% WIPRO, AXISBANK, HCLTECH, SUNPHARMA, KOTAKBANK    
23     5        0.685  1.13     51.8%      2.97%      4.33%    -6.1% WIPRO, NTPC, AXISBANK, SUNPHARMA, KOTAKBANK       
24     5        0.684  1.13     52.7%      2.92%      4.27%    -5.2% WIPRO, NTPC, AXISBANK, NESTLEIND, KOTAKBANK       
25     5        0.682  1.13     51.4%      2.84%      4.17%    -4.6% JSWSTEEL, AXISBANK, HCLTECH, SUNPHARMA, KOTAKBANK 
26     5        0.681  1.13     50.8%      3.07%      4.51%    -4.3% WIPRO, JSWSTEEL, AXISBANK, HCLTECH, KOTAKBANK     
27     5        0.679  1.13     52.2%      2.73%      4.02%    -5.7% AXISBANK, NESTLEIND, RELIANCE, HCLTECH, KOTAKBANK 
28     5        0.678  1.13     51.5%      2.71%      4.00%    -6.0% AXISBANK, RELIANCE, HCLTECH, SUNPHARMA, KOTAKBANK 
29     5        0.673  1.13     52.2%      2.78%      4.14%    -4.8% WIPRO, ITC, AXISBANK, HCLTECH, KOTAKBANK          
30     5        0.670  1.13     51.7%      2.54%      3.79%    -4.4% ITC, AXISBANK, INFY, SUNPHARMA, KOTAKBANK         
31     5        0.650  1.13     49.9%      3.20%      4.92%    -6.5% WIPRO, NTPC, AXISBANK, HCLTECH, INFY              
32     5        0.647  1.12     51.9%      2.70%      4.17%    -4.8% JSWSTEEL, AXISBANK, NESTLEIND, HCLTECH, KOTAKBANK 
33     5        0.646  1.12     53.8%      2.53%      3.92%    -6.8% NTPC, AXISBANK, NESTLEIND, INFY, SUNPHARMA        
34     5        0.645  1.12     53.9%      2.55%      3.95%    -4.0% AXISBANK, RELIANCE, INFY, SUNPHARMA, KOTAKBANK    
35     5        0.644  1.12     51.8%      2.79%      4.33%    -4.7% WIPRO, AXISBANK, RELIANCE, HCLTECH, KOTAKBANK     
36     5        0.644  1.12     50.1%      2.80%      4.35%    -6.3% WIPRO, NTPC, AXISBANK, NESTLEIND, HCLTECH         
37     5        0.644  1.12     50.7%      2.69%      4.18%    -5.2% WIPRO, AXISBANK, INFY, SUNPHARMA, KOTAKBANK       
38     5        0.643  1.12     49.9%      2.55%      3.96%    -5.7% AXISBANK, NESTLEIND, HCLTECH, INFY, SUNPHARMA     
39     5        0.638  1.12     52.2%      2.69%      4.21%    -6.5% NTPC, ITC, AXISBANK, SUNPHARMA, KOTAKBANK         
40     5        0.636  1.12     49.3%      2.75%      4.32%    -5.3% JSWSTEEL, AXISBANK, HCLTECH, INFY, SUNPHARMA      
41     5        0.636  1.12     51.5%      2.62%      4.12%    -6.6% NTPC, ITC, AXISBANK, HCLTECH, SUNPHARMA           
42     5        0.636  1.12     51.7%      2.64%      4.15%    -5.5% WIPRO, AXISBANK, NESTLEIND, INFY, KOTAKBANK       
43     5        0.635  1.12     50.6%      2.61%      4.11%    -4.2% JSWSTEEL, AXISBANK, INFY, SUNPHARMA, KOTAKBANK    
44     5        0.633  1.12     51.3%      2.69%      4.25%    -6.4% ITC, JSWSTEEL, AXISBANK, HCLTECH, KOTAKBANK       
45     5        0.633  1.12     49.7%      2.61%      4.12%    -5.0% AXISBANK, NESTLEIND, RELIANCE, HCLTECH, INFY      
46     5        0.632  1.12     51.6%      2.46%      3.90%    -6.5% ITC, AXISBANK, NESTLEIND, HCLTECH, KOTAKBANK      
47     5        0.632  1.12     53.0%      2.54%      4.02%    -4.9% AXISBANK, NESTLEIND, RELIANCE, INFY, KOTAKBANK    
48     5        0.630  1.12     52.7%      2.58%      4.10%    -5.6% AXISBANK, RELIANCE, HCLTECH, INFY, SUNPHARMA      
49     5        0.629  1.13     54.3%      3.01%      4.78%    -9.8% NTPC, AXISBANK, RELIANCE, HCLTECH, KOTAKBANK      
50     5        0.629  1.12     52.5%      2.92%      4.65%    -5.9% WIPRO, NTPC, AXISBANK, INFY, KOTAKBANK            

📊 PERFORMANCE DISTRIBUTION SUMMARY:
   Sharpe Ratio   - Mean: -0.007 | Median: -0.013 | Max: 0.839
   Annual Return  - Mean: -0.10% | Median: -0.06% | Max: 3.54%
   Annual Vol     - Mean: 5.02% | Median: 4.89%
   Max Drawdown   - Mean: -9.48% | Median: -9.17%

🎯 BEST PORTFOLIO BY SIZE:
   5-ticker: Sharpe=0.839 | AXISBANK, HCLTECH, INFY, SUNPHARMA, KOTAKBANK

💾 SAVING OPTIMIZATION RESULTS
============================================================
✅ All portfolio results saved: portfolio_performance_all.csv
✅ Top 50 portfolios saved: portfolio_performance_top50.csv
✅ Optimization summary saved: portfolio_optimization_summary.md
📁 Location: analysis/output/mse/20251006_024924/portfolio/portfolio_optimizer

🎉 PORTFOLIO OPTIMIZATION COMPLETED!
📊 5,054 portfolios evaluated
🏆 Top 50 best performers identified

🎯 Next: PyPortfolioOpt Weight Optimization & Equity Curve Generation
```
### combination_generator stdout
```text
✅ Loaded config from: /mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/StrategyLab-master/analysis/configs/example_mse_config.yaml
   Run ID: 20251006_024924
   Strategy: mse
   Date Range: 2022-01-01_to_2025-08-31
   Strategy trades dir: outputs/20251006_024924/mse/2022-01-01_to_2025-08-31/data/strategy_trades
   Base data dir: outputs/20251006_024924/mse/2022-01-01_to_2025-08-31/data/base_data
🚀 STARTING INTELLIGENT COMBINATION GENERATION
================================================================================
📁 Config: /mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/StrategyLab-master/analysis/configs/example_mse_config.yaml
📊 Strategy: mse
📅 Date Range: 2022-01-01_to_2025-08-31
================================================================================
🔧 INTELLIGENT COMBINATION GENERATION
================================================================================
📊 Loading Phase 2 results...
✅ Loaded 17 tickers with sector mapping
✅ Loaded 17x17 correlation matrix
✅ Loaded 24,546 anti-cascading trades

🎯 STEP 1: INDIVIDUAL TICKER PERFORMANCE ANALYSIS
======================================================================
   TATASTEEL    | Trades: 1374 | Sharpe: -0.017 | PF:  0.95 | WR:  46.9%
   ONGC         | Trades: 1413 | Sharpe: -0.015 | PF:  0.95 | WR:  45.5%
   WIPRO        | Trades: 1413 | Sharpe: -0.013 | PF:  0.96 | WR:  46.5%
   POWERGRID    | Trades: 1484 | Sharpe: -0.012 | PF:  0.96 | WR:  50.4%
   NTPC         | Trades: 1417 | Sharpe:  0.005 | PF:  1.02 | WR:  47.3%
   ITC          | Trades: 1457 | Sharpe: -0.002 | PF:  0.99 | WR:  49.1%
   SBIN         | Trades: 1395 | Sharpe: -0.045 | PF:  0.86 | WR:  47.2%
   JSWSTEEL     | Trades: 1444 | Sharpe: -0.010 | PF:  0.97 | WR:  46.7%
   AXISBANK     | Trades: 1464 | Sharpe:  0.044 | PF:  1.14 | WR:  50.5%
   NESTLEIND    | Trades: 1537 | Sharpe: -0.012 | PF:  0.96 | WR:  48.5%
   ADANIPORTS   | Trades: 1446 | Sharpe: -0.004 | PF:  0.98 | WR:  49.3%
   RELIANCE     | Trades: 1401 | Sharpe: -0.016 | PF:  0.95 | WR:  47.3%
   HCLTECH      | Trades: 1484 | Sharpe:  0.025 | PF:  1.08 | WR:  49.2%
   INFY         | Trades: 1401 | Sharpe:  0.018 | PF:  1.06 | WR:  49.8%
   TECHM        | Trades: 1449 | Sharpe: -0.014 | PF:  0.96 | WR:  47.3%
   SUNPHARMA    | Trades: 1522 | Sharpe:  0.008 | PF:  1.02 | WR:  48.8%
   KOTAKBANK    | Trades: 1445 | Sharpe:  0.047 | PF:  1.15 | WR:  51.3%

✅ Calculated metrics for 17 tickers

🔍 STEP 2: PRE-FILTERING TICKER UNIVERSE
======================================================================

📊 FILTER: MINIMUM TRADE THRESHOLD (200 trades)
   TATASTEEL    | ✅ 1,374 trades
   ONGC         | ✅ 1,413 trades
   WIPRO        | ✅ 1,413 trades
   POWERGRID    | ✅ 1,484 trades
   NTPC         | ✅ 1,417 trades
   ITC          | ✅ 1,457 trades
   SBIN         | ✅ 1,395 trades
   JSWSTEEL     | ✅ 1,444 trades
   AXISBANK     | ✅ 1,464 trades
   NESTLEIND    | ✅ 1,537 trades
   ADANIPORTS   | ✅ 1,446 trades
   RELIANCE     | ✅ 1,401 trades
   HCLTECH      | ✅ 1,484 trades
   INFY         | ✅ 1,401 trades
   TECHM        | ✅ 1,449 trades
   SUNPHARMA    | ✅ 1,522 trades
   KOTAKBANK    | ✅ 1,445 trades

   Result: 17/17 tickers pass minimum trade threshold

📊 FINAL VALID TICKER UNIVERSE: 17 tickers
   TATASTEEL    | PF:  0.95 | Acc:  46.9% | Sharpe: -0.017
   ONGC         | PF:  0.95 | Acc:  45.5% | Sharpe: -0.015
   WIPRO        | PF:  0.96 | Acc:  46.5% | Sharpe: -0.013
   POWERGRID    | PF:  0.96 | Acc:  50.4% | Sharpe: -0.012
   NTPC         | PF:  1.02 | Acc:  47.3% | Sharpe:  0.005
   ITC          | PF:  0.99 | Acc:  49.1% | Sharpe: -0.002
   SBIN         | PF:  0.86 | Acc:  47.2% | Sharpe: -0.045
   JSWSTEEL     | PF:  0.97 | Acc:  46.7% | Sharpe: -0.010
   AXISBANK     | PF:  1.14 | Acc:  50.5% | Sharpe:  0.044
   NESTLEIND    | PF:  0.96 | Acc:  48.5% | Sharpe: -0.012
   ADANIPORTS   | PF:  0.98 | Acc:  49.3% | Sharpe: -0.004
   RELIANCE     | PF:  0.95 | Acc:  47.3% | Sharpe: -0.016
   HCLTECH      | PF:  1.08 | Acc:  49.2% | Sharpe:  0.025
   INFY         | PF:  1.06 | Acc:  49.8% | Sharpe:  0.018
   TECHM        | PF:  0.96 | Acc:  47.3% | Sharpe: -0.014
   SUNPHARMA    | PF:  1.02 | Acc:  48.8% | Sharpe:  0.008
   KOTAKBANK    | PF:  1.15 | Acc:  51.3% | Sharpe:  0.047

📈 SHARPE RATIO DISTRIBUTION:
   Positive Sharpe: 6/17 tickers (35.3%)
   Range: -0.045 to 0.047

================================================================================
PROCESSING 5-TICKER PORTFOLIOS
================================================================================

🎯 STEP 3: GENERATING 5-TICKER COMBINATIONS
======================================================================
📊 Total possible 5-ticker combinations: 6,188

🔄 Generating and filtering combinations...
   Diversification filters:
   - Max sector concentration: 60%
   - Max average correlation: 0.75

📊 GENERATION RESULTS:
   Total tested: 6,188
   Passed sector filter: 5,054 (81.7%)
   Passed correlation filter: 5,054 (100.0% of sector-valid)
   Final valid combinations: 5,054 (81.7%)

💾 STEP 4: SAVING VALID COMBINATIONS
============================================================
✅ Valid combinations saved: valid_combinations_5ticker.csv
✅ Summary saved: combination_generation_summary_5ticker.md
📁 Location: analysis/output/mse/20251006_024924/portfolio/combination_generator

📊 Ready for portfolio optimization with 5,054 combinations

🏆 COMBINATION GENERATION COMPLETED SUCCESSFULLY!

📊 GENERATION SUMMARY:
   5-ticker portfolios: 5,054 valid combinations

🎯 Next: Portfolio Optimization
```
### sector_classification stdout
```text
✅ Loaded config from: /mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/StrategyLab-master/analysis/configs/example_mse_config.yaml
   Run ID: 20251006_024924
   Strategy: mse
   Date Range: 2022-01-01_to_2025-08-31
   Strategy trades dir: outputs/20251006_024924/mse/2022-01-01_to_2025-08-31/data/strategy_trades
   Base data dir: outputs/20251006_024924/mse/2022-01-01_to_2025-08-31/data/base_data
🚀 STARTING SECTOR CLASSIFICATION & CORRELATION ANALYSIS
================================================================================
📁 Config: /mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/StrategyLab-master/analysis/configs/example_mse_config.yaml
📊 Strategy: mse
📅 Date Range: 2022-01-01_to_2025-08-31
================================================================================
🏗️ SECTOR CLASSIFICATION & CORRELATION ANALYSIS
================================================================================
📊 Loading anti-cascading trades dataset...
✅ Loaded 24,546 anti-cascading trades
✅ Working with 17 affordable tickers
📁 Source: analysis/output/mse/20251006_024924/portfolio/anti_cascade_filter

🎯 STEP 1: SECTOR CLASSIFICATION MAPPING
============================================================
📊 SECTOR DISTRIBUTION ANALYSIS:
   Total sectors identified: 6
   Total tickers classified: 17

📋 SECTOR BREAKDOWN:
   Unclassified                        |  9 tickers (52.9%) | Avg Sharpe: -0.008
      → TATASTEEL, ONGC, WIPRO, POWERGRID, NTPC, NESTLEIND, ADANIPORTS, TECHM, SUNPHARMA
   Banking & Financial Services        |  3 tickers (17.6%) | Avg Sharpe: 0.016
      → SBIN, AXISBANK, KOTAKBANK
   Information Technology              |  2 tickers (11.8%) | Avg Sharpe: 0.022
      → HCLTECH, INFY
   Consumer Goods & FMCG               |  1 tickers ( 5.9%) | Avg Sharpe: -0.002
      → ITC
   Infrastructure & Construction       |  1 tickers ( 5.9%) | Avg Sharpe: -0.010
      → JSWSTEEL
   Energy & Power                      |  1 tickers ( 5.9%) | Avg Sharpe: -0.016
      → RELIANCE

🎯 DIVERSIFICATION ANALYSIS:
   Maximum sector concentration: 52.9%
   Minimum sectors for diversified portfolio: 3
   Available for sector-balanced portfolios: ✅ YES

⭐ SECTOR PERFORMANCE ANALYSIS:
   Banking & Financial Services        | Sharpe: 0.016 | PF: 1.05 | Score: 0.6
   Consumer Goods & FMCG               | Sharpe: -0.002 | PF: 0.99 | Score: 0.5
   Energy & Power                      | Sharpe: -0.016 | PF: 0.95 | Score: 0.5
   Information Technology              | Sharpe: 0.022 | PF: 1.06 | Score: 0.6
   Infrastructure & Construction       | Sharpe: -0.010 | PF: 0.97 | Score: 0.5
   Unclassified                        | Sharpe: -0.008 | PF: 0.97 | Score: 0.5

📊 STEP 2: CORRELATION MATRIX CALCULATION
============================================================
🔍 Calculating correlations from actual daily trade returns...
   Processing daily returns per ticker...
   TATASTEEL    → ✅ 881 days | Avg: -0.013% | Std: 0.743%
   ONGC         → ✅ 894 days | Avg: -0.016% | Std: 0.832%
   WIPRO        → ✅ 894 days | Avg: -0.006% | Std: 0.608%
   POWERGRID    → ✅ 899 days | Avg: 0.006% | Std: 0.742%
   NTPC         → ✅ 894 days | Avg: 0.005% | Std: 0.800%
   ITC          → ✅ 899 days | Avg: -0.004% | Std: 0.521%
   SBIN         → ✅ 894 days | Avg: -0.023% | Std: 0.658%
   JSWSTEEL     → ✅ 897 days | Avg: 0.006% | Std: 0.670%
   AXISBANK     → ✅ 894 days | Avg: 0.034% | Std: 0.582%
   NESTLEIND    → ✅ 899 days | Avg: -0.010% | Std: 0.503%
   ADANIPORTS   → ✅ 894 days | Avg: 0.001% | Std: 1.156%
   RELIANCE     → ✅ 885 days | Avg: -0.007% | Std: 0.570%
   HCLTECH      → ✅ 891 days | Avg: 0.014% | Std: 0.538%
   INFY         → ✅ 891 days | Avg: 0.011% | Std: 0.535%
   TECHM        → ✅ 891 days | Avg: -0.019% | Std: 0.675%
   SUNPHARMA    → ✅ 898 days | Avg: 0.014% | Std: 0.522%
   KOTAKBANK    → ✅ 895 days | Avg: 0.021% | Std: 0.482%

   Building correlation matrix...
   Correlation matrix dimensions: (905, 17)
   Total trading days analyzed: 905
   Date range: 2022-01-03 to 2025-08-29

📊 CORRELATION STATISTICS:
   Average correlation: 0.135
   Maximum correlation: 0.511
   Minimum correlation: 0.000
   Median correlation: 0.087

🔍 STEP 3: SECTOR CORRELATION ANALYSIS
============================================================
📊 DIVERSIFICATION INSIGHTS:
   Average within-sector correlation: 0.151
   Average cross-sector correlation:  0.128
   Diversification benefit: 0.023
   ⚠️  Limited diversification benefit across sectors

💾 STEP 4: SAVING SECTOR & CORRELATION DATA
============================================================
✅ Sector mapping saved: sector_mapping.csv
✅ Correlation matrix saved: correlation_matrix.csv
✅ Daily returns data saved: daily_returns_data.csv
✅ Analysis summary saved: sector_correlation_summary.md
📁 Location: analysis/output/mse/20251006_024924/portfolio/sector_classification

🎉 SECTOR CLASSIFICATION & CORRELATION ANALYSIS COMPLETED!
📊 Ready for next step: Intelligent Combination Generation

🏆 SECTOR ANALYSIS COMPLETED SUCCESSFULLY!
🎯 Next: Intelligent Combination Generation
```
### anti_cascade_filter stdout
```text
✅ Loaded config from: /mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/StrategyLab-master/analysis/configs/example_mse_config.yaml
   Run ID: 20251006_024924
   Strategy: mse
   Date Range: 2022-01-01_to_2025-08-31
   Strategy trades dir: outputs/20251006_024924/mse/2022-01-01_to_2025-08-31/data/strategy_trades
   Base data dir: outputs/20251006_024924/mse/2022-01-01_to_2025-08-31/data/base_data
🚀 STARTING ANTI-CASCADING SUBSET CREATION
================================================================================
📁 Config: /mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/StrategyLab-master/analysis/configs/example_mse_config.yaml
📊 Strategy: mse
📅 Date Range: 2022-01-01_to_2025-08-31
================================================================================
🔧 LOADING ANTI-CASCADING TOP 50 FROM FOUNDATION ANALYSIS
===========================================================================
✅ Loaded Anti-Cascading Top 50 list
✅ Top 10 performers: KOTAKBANK, AXISBANK, HINDUNILVR, MARUTI, ASIANPAINT, HCLTECH, INFY, TITAN, SUNPHARMA, TCS
📁 Source: analysis/output/mse/20251006_024924/portfolio/ticker_ranking/TOP50_ANTICASCADING_TRADES.csv
📊 Loading trade data from: all_trades_merged.csv
✅ Loaded 60,629 trades
   Date range: 2022-01-03 09:35:00+05:30 to 2025-08-29 15:10:00+05:30
   Tickers: 24

📊 FILTERING TRADES FOR TOP 50 TICKERS
==================================================
✅ Original dataset: 60,629 total trades
✅ Date range: 2022-01-03 09:35:00+05:30 to 2025-08-29 15:15:00+05:30
✅ Filtered to 60,629 trades from Top 50 tickers

🔍 IDENTIFYING AFFORDABLE TICKERS (Under ₹2,000)
============================================================
📊 PRICE ANALYSIS RESULTS:
   Total Top 50 tickers: 24
   Tickers under ₹2,000: 17 (70.8%)

📋 AFFORDABLE TICKERS:
   Ticker       | Price    | Rank | PF    | Sharpe | Category       
   ----------------------------------------------------------------------
   TATASTEEL    | ₹154.93 |   21 |  0.95 | -0.017 | Under ₹500
   ONGC         | ₹233.46 |   22 |  0.95 | -0.015 | Under ₹500
   WIPRO        | ₹249.19 |   19 |  0.96 | -0.013 | Under ₹500
   POWERGRID    | ₹274.60 |   15 |  0.96 | -0.012 | Under ₹500
   NTPC         | ₹328.50 |   12 |  1.02 |  0.005 | Under ₹500
   ITC          | ₹409.65 |   13 |  0.99 | -0.002 | Under ₹500
   SBIN         | ₹801.80 |   24 |  0.86 | -0.045 | Under ₹1000
   JSWSTEEL     | ₹1024.10 |   17 |  0.97 | -0.010 | Under ₹2000
   AXISBANK     | ₹1044.40 |    2 |  1.14 |  0.044 | Under ₹2000
   NESTLEIND    | ₹1156.60 |   16 |  0.96 | -0.012 | Under ₹2000
   ADANIPORTS   | ₹1313.40 |   14 |  0.98 | -0.004 | Under ₹2000
   RELIANCE     | ₹1352.70 |   20 |  0.95 | -0.016 | Under ₹2000
   HCLTECH      | ₹1452.60 |    6 |  1.08 |  0.025 | Under ₹2000
   INFY         | ₹1469.20 |    7 |  1.06 |  0.018 | Under ₹2000
   TECHM        | ₹1483.70 |   18 |  0.96 | -0.014 | Under ₹2000
   SUNPHARMA    | ₹1592.40 |    9 |  1.02 |  0.008 | Under ₹2000
   KOTAKBANK    | ₹1958.20 |    1 |  1.15 |  0.047 | Under ₹2000

🎯 APPLYING ANTI-CASCADING FILTER
============================================================
📊 Working with 43,191 trades from 17 affordable tickers
🔍 Categorizing trades for cascade detection...

📊 TRADE CATEGORIZATION RESULTS:
   CONSECUTIVE_SAME_DIRECTION     |   18,645 ( 43.2%) | ❌ EXCLUDE
   FIRST_TRADE_OF_DAY             |   15,173 ( 35.1%) | ✅ INCLUDE
   CONSECUTIVE_OPPOSITE_DIRECTION |    9,356 ( 21.7%) | ✅ INCLUDE
   FIRST_TRADE_FOR_TICKER         |       16 (  0.0%) | ✅ INCLUDE
   FIRST_TRADE_OVERALL            |        1 (  0.0%) | ✅ INCLUDE

🎯 ANTI-CASCADING FILTER RESULTS:
   Original trades: 43,191
   Excluded (cascading): 18,645 (43.2%)
   Remaining (anti-cascading): 24,546 (56.8%)

💾 SAVING FILTERED DATASET
==================================================
✅ Filtered trades saved: anti_cascading_trades_filtered.csv
✅ Ticker metadata saved: affordable_tickers_metadata.csv
✅ Summary report saved: anti_cascade_filter_summary.md
📁 Location: analysis/output/mse/20251006_024924/portfolio/anti_cascade_filter

🔍 VERIFICATION:
   File size: 24,546 trades
   Memory usage: 20.8 MB

🏆 ANTI-CASCADING SUBSET CREATION COMPLETED!
📊 Filtered dataset: 24,546 trades
🎯 Next step: Sector classification and correlation analysis
```
### ticker_ranking stdout
```text
✅ Loaded config from: /mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/StrategyLab-master/analysis/configs/example_mse_config.yaml
   Run ID: 20251006_024924
   Strategy: mse
   Date Range: 2022-01-01_to_2025-08-31
   Strategy trades dir: outputs/20251006_024924/mse/2022-01-01_to_2025-08-31/data/strategy_trades
   Base data dir: outputs/20251006_024924/mse/2022-01-01_to_2025-08-31/data/base_data
🚀 STARTING COMPREHENSIVE CASCADE vs ANTI-CASCADE ANALYSIS
================================================================================
📁 Config: /mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/StrategyLab-master/analysis/configs/example_mse_config.yaml
📊 Strategy: mse
📅 Date Range: 2022-01-01_to_2025-08-31
================================================================================
📊 Loading trade data from: all_trades_merged.csv
✅ Loaded 60,629 trades
   Date range: 2022-01-03 09:35:00+05:30 to 2025-08-29 15:10:00+05:30
   Tickers: 24
🔄 COMPREHENSIVE CASCADE vs ANTI-CASCADE ANALYSIS
======================================================================
🎯 Objective: Compare Top 50 performers across trade categories
======================================================================

📊 STEP 1: TAGGING TRADES WITH CASCADE PATTERNS
==================================================
✅ Loaded 60,629 total trades
✅ Date range: 2022-01-03 to 2025-08-29
✅ Unique tickers: 24
🏷️  Tagging trades with cascade characteristics...

📊 TRADE TAGGING RESULTS:
   CONSECUTIVE_SAME_DIRECTION     |   26,065 ( 43.0%) | 🔄 CASCADING
   FIRST_TRADE_OF_DAY             |   21,402 ( 35.3%) | ✅ ANTI-CASCADING
   CONSECUTIVE_OPPOSITE_DIRECTION |   13,138 ( 21.7%) | ✅ ANTI-CASCADING
   FIRST_TRADE_FOR_TICKER         |       23 (  0.0%) | ✅ ANTI-CASCADING
   FIRST_TRADE_OVERALL            |        1 (  0.0%) | ✅ ANTI-CASCADING

📈 STEP 2A: CALCULATING PERFORMANCE - ALL TRADES
============================================================
📊 Analyzing 60,629 trades in ALL category
✅ Calculated performance for 24 tickers
   Top 3: KOTAKBANK, MARUTI, AXISBANK

📈 STEP 2B: CALCULATING PERFORMANCE - CASCADING TRADES
============================================================
📊 Analyzing 26,065 trades in CASCADING category
✅ Calculated performance for 24 tickers
   Top 3: TITAN, TECHM, MARUTI

📈 STEP 2C: CALCULATING PERFORMANCE - ANTI_CASCADING TRADES
============================================================
📊 Analyzing 34,564 trades in ANTI_CASCADING category
✅ Calculated performance for 24 tickers
   Top 3: KOTAKBANK, AXISBANK, HINDUNILVR

🔍 STEP 3: COMPARING TOP 50 PERFORMERS ACROSS CATEGORIES
============================================================
📊 OVERLAP ANALYSIS:
   All vs Cascading: 24/50 (48.0%)
   All vs Anti-Cascading: 24/50 (48.0%)
   Cascading vs Anti-Cascading: 24/50 (48.0%)

   Tickers unique to Anti-Cascading Top 50: 0
   Tickers unique to All Trades Top 50: 0

💾 STEP 4: SAVING COMPREHENSIVE RESULTS
==================================================
✅ All performance rankings saved to: analysis/output/mse/20251006_024924/portfolio/ticker_ranking
✅ Top 50 lists saved for each category
✅ Comparison summary saved: cascade_comparison_summary.md

🎯 CRITICAL QUESTION ANSWERED:
❌ Current Top 50 basis is FLAWED! Only 48.0% overlap with Anti-Cascading Top 50
✅ MUST use Anti-Cascading Top 50 for portfolio construction

🏆 COMPREHENSIVE ANALYSIS COMPLETED!
📂 Results saved to: analysis/output/mse/20251006_024924/portfolio/ticker_ranking
```