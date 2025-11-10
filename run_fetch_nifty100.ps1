# PowerShell script to run Nifty 100 fetch in background with monitoring
cd d:\Balcony\Trading\unified_trading_setup\backtester
. .\.venv\Scripts\Activate.ps1

Write-Host "Starting Nifty 100 fetch with improved rate limit handling..." -ForegroundColor Green
Write-Host "Batches: 57 (2 tickers per batch)" -ForegroundColor Cyan
Write-Host "Rate limit waits: 5s -> 10s -> 20s -> 40s for 429 errors" -ForegroundColor Cyan
Write-Host "Batch pauses: 30s between batches" -ForegroundColor Cyan
Write-Host ""
Write-Host "Estimated time: 3-4 hours for all 114 tickers" -ForegroundColor Yellow
Write-Host ""
Write-Host "Starting fetch at: $(Get-Date)" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green

# Run the fetch
python scripts\fetch_nifty100_missing_tickers.py --pool-path "data/pools/2022-01-01_to_2025-08-31/" --start-date "2022-01-03" --end-date "2025-11-01"

Write-Host "================================" -ForegroundColor Green
Write-Host "Fetch completed at: $(Get-Date)" -ForegroundColor Green

# Check final count
$tickerCount = (Get-ChildItem -Path "data/pools/2022-01-01_to_2025-08-31/" -Directory | Measure-Object).Count
Write-Host "Final ticker count: $tickerCount" -ForegroundColor Cyan
