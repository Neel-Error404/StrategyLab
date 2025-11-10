#!/usr/bin/env pwsh
<#
.SYNOPSIS
Run data fetcher with proper environment setup
#>

# Get the script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Write-Host "Script directory: $scriptDir"

# Change to the script directory
Push-Location $scriptDir

# Activate venv
Write-Host "Activating virtual environment..."
& ".\.venv\Scripts\Activate.ps1"

# Run the data fetcher
Write-Host "Running data fetcher update..."
python src\core\etl\data_fetcher.py --mode update --pool-path data/pools/2022-01-01_to_2025-08-31/ @args

# Restore original directory
Pop-Location
