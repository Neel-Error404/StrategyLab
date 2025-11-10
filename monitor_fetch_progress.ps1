# Monitor script - check progress every 5 minutes
$startTime = Get-Date
while($true) {
    $elapsed = ((Get-Date) - $startTime).TotalMinutes
    $count = (Get-ChildItem -Path "data/pools/2022-01-01_to_2025-08-31/" -Directory | Measure-Object).Count
    Write-Host "[$([datetime]::now.ToString('HH:mm:ss'))] Elapsed: $([math]::Round($elapsed, 1)) mins | Tickers: $count/145 | Progress: $([math]::Round(($count/145)*100, 1))%" -ForegroundColor Cyan
    Start-Sleep -Seconds 300  # Check every 5 minutes
}
