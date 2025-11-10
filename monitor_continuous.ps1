# Continuous fetch progress monitoring
Write-Host "=== NIFTY 100 FETCH PROGRESS MONITOR ===" -ForegroundColor Green
Write-Host "Started: 2025-11-01 14:08:55" -ForegroundColor Cyan
Write-Host "Batches: 57 total (2 tickers each, 30s pause between)" -ForegroundColor Cyan
Write-Host ""

$baseTime = Get-Date -Date "2025-11-01 14:08:55"
$basePath = "d:\Balcony\Trading\unified_trading_setup\backtester\data\pools\2022-01-01_to_2025-08-31\"

while ($true) {
    $elapsed = ((Get-Date) - $baseTime).TotalMinutes
    $count = (Get-ChildItem -Path $basePath -Directory -ErrorAction SilentlyContinue).Count
    $new = if ($count -gt 31) { $count - 31 } else { 0 }
    $batchesComplete = [math]::Floor($new / 2)
    $progress = [math]::Round(($count / 145) * 100, 1)
    
    $timeStr = Get-Date -Format "HH:mm:ss"
    Write-Host "[$timeStr] Elapsed: $([math]::Round($elapsed, 1)) mins | Tickers: $count/145 | New: $new | Batches: $batchesComplete/57 | Progress: $progress%" -ForegroundColor Cyan
    
    Start-Sleep -Seconds 60  # Update every 60 seconds
}
