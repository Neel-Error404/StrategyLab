#!/usr/bin/env python3
"""
Documentation Cleanup Verification Script

Verifies that the documentation reorganization was successful:
- Checks root directory has only expected .md files
- Validates archive structure exists
- Confirms critical user-facing docs are present
- Reports any unexpected files

Usage:
    python scripts/verify_documentation_cleanup.py
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

# Expected files after cleanup
EXPECTED_ROOT_MD = {
    'README.md',
    'QUICKSTART.md',
    'CONTRIBUTING.md',  # To be created
    'ARCHITECTURE.md',
    'FEATURES.md',
    'RELEASES.md',
    'RELEASE_NOTES.md',
    'DOCUMENTATION_AUDIT_REPORT.md',  # This audit report
}

EXPECTED_DOCS_MD = {
    # User Guides
    'SETUP_GUIDE.md',
    'BROKER_SETUP.md',
    'STRATEGY_GUIDE.md',
    'TEMPLATE_GUIDE.md',
    'CLI_REFERENCE.md',
    'OUTPUT_GUIDE.md',
    'TROUBLESHOOTING.md',
    'ERROR_REFERENCE.md',  # To be created
    # Technical Guides
    'CRITICAL_STRATEGY_IMPLEMENTATION_GUIDE.md',
    'DATA_VALIDATION_CRITERIA.md',
    'SIGNAL_HANDLING_AND_VALIDATION_FIXES.md',
    # Project Management
    'CHANGELOG.md',
    'TASKS.md',
    'RELEASE_CHECKLIST.md',
    'OSS_RELEASE_REPORT.md',
    'strategylab_v2_phase0_audit.md',
}

EXPECTED_ARCHIVE_STRUCTURE = {
    'archive/development/journals',
    'archive/development/phases',
    'archive/development/features',
    'archive/development/status',
    'archive/development/internal',
}

# Files that should have been archived
ARCHIVED_FILES = {
    'journals': [
        'IMPLEMENTATION_JOURNAL.md',
        'QA_TESTING_JOURNAL.md',
        'QA_ANALYSIS_SYSTEM_JOURNAL.md',
        'QA_PROGRESS_SUMMARY.md',
        'IMPLEMENTATION_GUIDE.md',
    ],
    'phases': [
        'COMPREHENSIVE_PHASES_5_10_UNDERSTANDING.md',
        'PHASES_5_10_VISUAL_ROADMAP.md',
        'PHASE_5_10_RECONCILIATION_REPORT.md',
        'PHASE_4.2_PORTFOLIO_VALIDATION_COMPLETE.md',
        'YOUR_UNDERSTANDING_SUMMARY.md',
        'QA_INTEGRATION_TESTING_PLAN.md',
        'COMPREHENSIVE_SYSTEM_ASSESSMENT.md',
    ],
    'features': [
        'NIFTY100_SCALING_GUIDE.md',
        'NIFTY100_SCALING_STATUS.md',
        'NIFTY100_CLI_COMMANDS.md',
        'DATA_POOL_CORRECTION_PLAN.md',
    ],
    'status': [
        'FETCH_STATUS_REPORT.md',
        'USER_REVIEW_ANALYSIS_COMPLETE.md',
        'live module Contact details.md',
    ],
    'internal': [
        'AGENTS.md',
        'log_prompt.md',
    ],
}

# Files that should have been deleted
DELETED_FILES = {
    'README_START_HERE.md',
    'START_HERE_READING_GUIDE.md',
}


def get_repo_root() -> Path:
    """Get repository root directory."""
    script_dir = Path(__file__).parent
    return script_dir.parent


def check_root_markdown_files(repo_root: Path) -> Tuple[bool, List[str]]:
    """Check root directory for expected markdown files."""
    issues = []

    # Get actual .md files in root
    actual_md_files = {f.name for f in repo_root.glob('*.md')}

    # Check for missing expected files (excluding to-be-created)
    to_be_created = {'CONTRIBUTING.md', 'ERROR_REFERENCE.md'}
    required_files = EXPECTED_ROOT_MD - to_be_created
    missing = required_files - actual_md_files

    if missing:
        issues.append(f"Missing expected root files: {', '.join(missing)}")

    # Check for unexpected files (should have been archived/deleted)
    unexpected = actual_md_files - EXPECTED_ROOT_MD
    if unexpected:
        issues.append(f"Unexpected root files (should be archived/deleted): {', '.join(unexpected)}")

    # Check for deleted files (should not exist)
    still_present = actual_md_files & DELETED_FILES
    if still_present:
        issues.append(f"Files should have been deleted: {', '.join(still_present)}")

    return len(issues) == 0, issues


def check_docs_folder(repo_root: Path) -> Tuple[bool, List[str]]:
    """Check docs/ folder for expected markdown files."""
    issues = []
    docs_dir = repo_root / 'docs'

    if not docs_dir.exists():
        return False, ["docs/ folder does not exist"]

    # Get actual .md files in docs/
    actual_md_files = {f.name for f in docs_dir.glob('*.md')}

    # Check for missing expected files (excluding to-be-created)
    to_be_created = {'ERROR_REFERENCE.md'}
    required_files = EXPECTED_DOCS_MD - to_be_created
    missing = required_files - actual_md_files

    if missing:
        issues.append(f"Missing expected docs/ files: {', '.join(missing)}")

    return len(issues) == 0, issues


def check_archive_structure(repo_root: Path) -> Tuple[bool, List[str]]:
    """Check archive/ structure exists."""
    issues = []

    for archive_path in EXPECTED_ARCHIVE_STRUCTURE:
        full_path = repo_root / archive_path
        if not full_path.exists():
            issues.append(f"Archive directory missing: {archive_path}")

    return len(issues) == 0, issues


def check_archived_files(repo_root: Path) -> Tuple[bool, List[str]]:
    """Check that files were properly archived."""
    issues = []

    for category, files in ARCHIVED_FILES.items():
        archive_dir = repo_root / 'archive' / 'development' / category

        for filename in files:
            # Check if file is in archive
            archived_path = archive_dir / filename
            root_path = repo_root / filename

            if archived_path.exists():
                # Good - file is archived
                if root_path.exists():
                    issues.append(f"File exists in both root and archive: {filename}")
            else:
                # Not in archive - should it still be in root?
                if not root_path.exists():
                    issues.append(f"File missing from both root and archive/{category}/: {filename}")

    return len(issues) == 0, issues


def generate_report(repo_root: Path) -> Dict[str, any]:
    """Generate complete verification report."""
    report = {
        'root_md_files': check_root_markdown_files(repo_root),
        'docs_folder': check_docs_folder(repo_root),
        'archive_structure': check_archive_structure(repo_root),
        'archived_files': check_archived_files(repo_root),
    }

    return report


def print_report(report: Dict[str, Tuple[bool, List[str]]]) -> bool:
    """Print verification report and return overall success status."""
    print("=" * 80)
    print("DOCUMENTATION CLEANUP VERIFICATION REPORT")
    print("=" * 80)
    print()

    all_passed = True

    # Root Markdown Files
    passed, issues = report['root_md_files']
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} Root Markdown Files")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
        all_passed = False
    else:
        print("  - All expected root files present")
        print("  - No unexpected files in root")
    print()

    # Docs Folder
    passed, issues = report['docs_folder']
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} docs/ Folder")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
        all_passed = False
    else:
        print("  - All expected docs/ files present")
    print()

    # Archive Structure
    passed, issues = report['archive_structure']
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} Archive Structure")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
        all_passed = False
    else:
        print("  - All archive directories created")
    print()

    # Archived Files
    passed, issues = report['archived_files']
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} Archived Files")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
        all_passed = False
    else:
        print("  - All files properly archived")
        print("  - No duplicates between root and archive")
    print()

    # Overall Status
    print("=" * 80)
    if all_passed:
        print("[SUCCESS] Documentation cleanup verified successfully!")
        print()
        print("Next steps:")
        print("  1. Create docs/ERROR_REFERENCE.md")
        print("  2. Create CONTRIBUTING.md")
        print("  3. Commit changes")
        print("  4. Test new user flow: README -> QUICKSTART -> backtest")
    else:
        print("[FAILED] Documentation cleanup has issues (see above)")
        print()
        print("Review the DOCUMENTATION_AUDIT_REPORT.md and re-run cleanup steps.")
    print("=" * 80)

    return all_passed


def main():
    """Main entry point."""
    repo_root = get_repo_root()

    print(f"Repository root: {repo_root}")
    print()

    report = generate_report(repo_root)
    success = print_report(report)

    # Exit with appropriate code
    exit(0 if success else 1)


if __name__ == '__main__':
    main()
