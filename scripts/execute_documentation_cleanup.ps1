# Documentation Cleanup Execution Script
# PowerShell script to execute the documentation reorganization plan
#
# Usage: .\scripts\execute_documentation_cleanup.ps1
#
# This script will:
# 1. Create archive directory structure
# 2. Move development files to archive/
# 3. Delete redundant files
# 4. Stage new user-facing files for commit
# 5. Verify the cleanup was successful

param(
    [switch]$DryRun = $false
)

$ErrorActionPreference = "Stop"

# Colors for output
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Info { Write-Host $args -ForegroundColor Cyan }
function Write-Warning { Write-Host $args -ForegroundColor Yellow }
function Write-Error { Write-Host $args -ForegroundColor Red }

# Get repository root
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
Set-Location $repoRoot

Write-Info "=========================================================================="
Write-Info "DOCUMENTATION CLEANUP EXECUTION"
Write-Info "=========================================================================="
Write-Info ""
Write-Info "Repository: $repoRoot"

if ($DryRun) {
    Write-Warning "DRY RUN MODE - No changes will be made"
}

Write-Info ""

# Phase 1: Create archive structure
Write-Info "Phase 1: Creating archive directory structure..."

$archiveDirs = @(
    "archive/development/journals",
    "archive/development/phases",
    "archive/development/features",
    "archive/development/status",
    "archive/development/internal"
)

foreach ($dir in $archiveDirs) {
    $fullPath = Join-Path $repoRoot $dir
    if (-not (Test-Path $fullPath)) {
        if (-not $DryRun) {
            New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
            Write-Success "  Created: $dir"
        } else {
            Write-Info "  [DRY RUN] Would create: $dir"
        }
    } else {
        Write-Info "  Already exists: $dir"
    }
}

Write-Info ""

# Phase 2: Move development files to archive
Write-Info "Phase 2: Archiving development files..."

$filesToArchive = @{
    'journals' = @(
        'IMPLEMENTATION_JOURNAL.md',
        'QA_TESTING_JOURNAL.md',
        'QA_ANALYSIS_SYSTEM_JOURNAL.md',
        'QA_PROGRESS_SUMMARY.md',
        'IMPLEMENTATION_GUIDE.md'
    )
    'phases' = @(
        'COMPREHENSIVE_PHASES_5_10_UNDERSTANDING.md',
        'PHASES_5_10_VISUAL_ROADMAP.md',
        'PHASE_5_10_RECONCILIATION_REPORT.md',
        'PHASE_4.2_PORTFOLIO_VALIDATION_COMPLETE.md',
        'YOUR_UNDERSTANDING_SUMMARY.md',
        'QA_INTEGRATION_TESTING_PLAN.md',
        'COMPREHENSIVE_SYSTEM_ASSESSMENT.md'
    )
    'features' = @(
        'NIFTY100_SCALING_GUIDE.md',
        'NIFTY100_SCALING_STATUS.md',
        'NIFTY100_CLI_COMMANDS.md',
        'DATA_POOL_CORRECTION_PLAN.md'
    )
    'status' = @(
        'FETCH_STATUS_REPORT.md',
        'USER_REVIEW_ANALYSIS_COMPLETE.md',
        'live module Contact details.md'
    )
    'internal' = @(
        'AGENTS.md',
        'log_prompt.md'
    )
}

$movedCount = 0
$skippedCount = 0

foreach ($category in $filesToArchive.Keys) {
    Write-Info "  Category: $category"

    foreach ($file in $filesToArchive[$category]) {
        $sourcePath = Join-Path $repoRoot $file
        $destPath = Join-Path $repoRoot "archive/development/$category/$file"

        if (Test-Path $sourcePath) {
            if (-not $DryRun) {
                # Check if file is tracked by git
                $gitStatus = git ls-files $file 2>$null
                if ($gitStatus) {
                    # Use git mv for tracked files
                    git mv $file "archive/development/$category/$file" 2>&1 | Out-Null
                } else {
                    # Use regular move for untracked files
                    Move-Item -Path $sourcePath -Destination $destPath -Force
                }
                Write-Success "    Moved: $file"
                $movedCount++
            } else {
                Write-Info "    [DRY RUN] Would move: $file"
            }
        } else {
            Write-Warning "    Not found: $file"
            $skippedCount++
        }
    }
}

Write-Info ""
Write-Info "  Total moved: $movedCount files"
if ($skippedCount -gt 0) {
    Write-Warning "  Total skipped: $skippedCount files (not found)"
}
Write-Info ""

# Phase 3: Delete redundant files
Write-Info "Phase 3: Deleting redundant files..."

$filesToDelete = @(
    'README_START_HERE.md',
    'START_HERE_READING_GUIDE.md'
)

$deletedCount = 0

foreach ($file in $filesToDelete) {
    $filePath = Join-Path $repoRoot $file

    if (Test-Path $filePath) {
        if (-not $DryRun) {
            Remove-Item -Path $filePath -Force
            Write-Success "  Deleted: $file"
            $deletedCount++
        } else {
            Write-Info "  [DRY RUN] Would delete: $file"
        }
    } else {
        Write-Info "  Already removed: $file"
    }
}

Write-Info ""
Write-Info "  Total deleted: $deletedCount files"
Write-Info ""

# Phase 4: Stage new user-facing files
Write-Info "Phase 4: Staging new user-facing files..."

$filesToAdd = @(
    'QUICKSTART.md',
    'docs/CRITICAL_STRATEGY_IMPLEMENTATION_GUIDE.md',
    'README.md'
)

foreach ($file in $filesToAdd) {
    $filePath = Join-Path $repoRoot $file

    if (Test-Path $filePath) {
        if (-not $DryRun) {
            git add $file 2>&1 | Out-Null
            Write-Success "  Staged: $file"
        } else {
            Write-Info "  [DRY RUN] Would stage: $file"
        }
    } else {
        Write-Warning "  Not found: $file"
    }
}

Write-Info ""

# Phase 5: Verify cleanup
Write-Info "Phase 5: Verifying cleanup..."
Write-Info ""

if (-not $DryRun) {
    # Run verification script
    python scripts/verify_documentation_cleanup.py
} else {
    Write-Info "[DRY RUN] Verification skipped in dry run mode"
}

Write-Info ""

# Summary
Write-Info "=========================================================================="
if ($DryRun) {
    Write-Warning "DRY RUN COMPLETE - No changes were made"
    Write-Info ""
    Write-Info "To execute the cleanup, run:"
    Write-Info "  .\scripts\execute_documentation_cleanup.ps1"
} else {
    Write-Success "CLEANUP COMPLETE"
    Write-Info ""
    Write-Info "Next steps:"
    Write-Info "  1. Review git status: git status"
    Write-Info "  2. Create missing docs:"
    Write-Info "     - docs/ERROR_REFERENCE.md"
    Write-Info "     - CONTRIBUTING.md"
    Write-Info "  3. Commit changes:"
    Write-Info "     git commit -m 'docs: reorganize documentation structure for V2 release'"
    Write-Info "  4. Test user flow: README -> QUICKSTART -> first backtest"
}
Write-Info "=========================================================================="
