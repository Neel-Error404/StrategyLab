# Nifty 100 Scaling - Complete CLI Command Guide

## Phase 1: Data Fetching (CURRENTLY IN PROGRESS)

### Fetch Command Executed:
```powershell
cd d:\Balcony\Trading\unified_trading_setup\backtester
.\.venv\Scripts\python.exe scripts\fetch_nifty100_missing_tickers.py `
  --pool-path "data/pools/2022-01-01_to_2025-08-31/" `
  --start-date "2022-01-03" `
  --end-date "2025-11-01"
```

**Status**: In Progress (started 13:46 IST)
**Progress**: ADANIPORTS completed, 92 remaining
**Estimated Duration**: 2-3 hours from start time
**Log File**: `logs/nifty100_fetch.log` (optional)

### Monitor Fetch Progress:
```powershell
# Check how many tickers have been fetched
Get-ChildItem -Path "data/pools/2022-01-01_to_2025-08-31/" -Directory | Measure-Object

# View latest log entries
Get-Content -Path "logs/nifty100_fetch.log" -Tail 50

# Check data availability for specific ticker
Get-ChildItem -Path "data/pools/2022-01-01_to_2025-08-31/ADANIPORTS/"
```

---

## Phase 2: Folder Rename (AFTER DATA FETCH COMPLETE)

### Rename Pool Folder:
```powershell
cd d:\Balcony\Trading\unified_trading_setup\backtester
Rename-Item -Path "data/pools/2022-01-01_to_2025-08-31" `
            -NewName "2022-01-03_to_2025-11-01"
```

**Why**: Folder name "2022-01-01_to_2025-08-31" is inaccurate
- Start: 2022-01-03 (markets were closed 2022-01-01)
- End: 2025-11-01 (latest data, not 2025-08-31)

---

## Phase 3: Backtest Execution (AFTER DATA FETCH + RENAME COMPLETE)

### Backtest Command for Nifty 100 (103 tickers):
```powershell
cd d:\Balcony\Trading\unified_trading_setup\backtester
.\.venv\Scripts\python.exe -m src.runners.unified_runner `
  --mode backtest `
  --template minimal `
  --date-ranges 2022-01-03_to_2025-11-01 `
  --tickers ADANIPORTS,ADANIGREEN,APOLLOHOSP,ASIANPAINT,AXISBANK,BAJAJFINSV,BAJAJ-AUTO,BPCL,BHEL,BOSCHLTD,BRITANNIA,CHOLAFIN,CIPLA,COLPAL,DLF,DABUR,DELHIVERY,DIVISLAB,EICHERMOT,ESCORTS,FEDERALBNK,GAIL,GLAXO,GMRINFRA,GODREJCP,GODREJPROP,GRANULES,GRAPHITE,GRASIM,HAVELLS,HERCULES,HONEYWELL,IBULHSGFIN,ICICIBANK,IDBI,IDFCBANK,INDHOTEL,INDIGO,INDUSIND,IOC,IPCALAB,IRCTC,IRFC,ITC,ITI,JKCEMENT,JSWSTEEL,JSL,JINDALSTEL,KOTAKBANK,LT,LALPATHLAB,LAURUSLABS,LTIM,LTTS,LUPIN,MANAPPURAM,MRF,MARUTI,MINDTREE,MAXHEALTH,MCX,MOTHERSUMI,MPHASIS,MSUMI,NAVNETEDUL,NBCC,NDRAUTO,NESTLEIND,NMDC,NTPC,ONGC,PAYTM,PERSISTENT,PETRONET,PFC,PIDILITIND,PNB,POCL,POLYCAB,POWERGRID,PSB,PSUBANK,PVBANK,RAMCOCEM,RECL,SBICARD,SBILIFE,SBIN,SHREECEM,SHYAMMETL,SIEMENS,SONACOMS,SPAREINDS,STLTECH,SUNPHARMA,SUNTV,SUMMITSEC,SYNGENE,TATACHEM,TATASTEEL,TATAGLOBAL,TATAMOTORS,TATAPOWER,TITAGARH,TORNTPHARM,TITAN,TRIVENI,TVS,TVSMOTOR,UBL,UNIONBANK,UPL,VGUARD,VINATIORGA,VIMTALABS,WIPRO,YESBANK,ZEEJENT,ZEEL,RELIANCE,TCS,INFY,TECHM,ABFRL,BANKNIFTY `
  --parallel `
  --max-workers 8 `
  --skip-visualization `
  --strategies mse
```

**Expected Results**:
- Runtime: 45-90 minutes (with --skip-visualization optimization)
- Output folder: `outputs/backtest_results_{timestamp}/`
- Files generated: base_data.csv, strategy_trades.csv, risk_approved_trades.csv

---

## Quick Reference Summary

| Phase | Task | Command | Status | Duration |
|-------|------|---------|--------|----------|
| 1 | Fetch 30 existing tickers | `data_fetcher --mode update` |  Complete | 10 min |
| 1 | Fetch 93 missing tickers | `fetch_nifty100_missing_tickers.py` |  In Progress | 2-3 hr |
| 2 | Rename folder | `Rename-Item` |  Ready | 1 sec |
| 3 | Run Nifty 100 backtest | `unified_runner --mode backtest` |  Ready | 45-90 min |

---

## Performance Optimizations Enabled
1.  --skip-visualization: 70-80% speedup
2.  --parallel: Multi-threaded execution
3.  --max-workers 8: Optimal for 103 tickers
4.  Minimal risk template: Faster evaluation

---

## Expected Final Stats
- Total tickers in backtest: 103 Nifty 100 constituents
- Data points per ticker: ~70,000 candles (15m) + ~210,000 candles (5m)
- Strategy: MSE (Mean Square Error reversal)
- Total data volume: ~20 GB processed

---

## Troubleshooting

If fetch script hangs:
1. Check network connectivity to Upstox API
2. Verify access token validity
3. Check rate limits (Upstox typically allows 100 requests/sec)

If backtest fails:
1. Ensure all 93 tickers have both 15m and 5m data
2. Check for duplicate timestamps (data integrity issue)
3. Verify output folder has write permissions

