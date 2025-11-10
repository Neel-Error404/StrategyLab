# Documentation Cleanup - Quick Reference Card

**One-page guide to execute the documentation reorganization**

---

## TL;DR

**Problem**: 29 markdown files in root directory, users confused about where to start

**Solution**: Archive 21 development files, delete 3 redundant files, keep 7 user-facing files

**Time**: 10 minutes automated + 1 hour for new docs

**Impact**: 76% reduction in root clutter, clear user journey

---

## Execute Cleanup (Choose One)

### Option A: Automated Script (Recommended)
```powershell
# Run cleanup
.\scripts\execute_documentation_cleanup.ps1

# Verify
python scripts\verify_documentation_cleanup.py

# If all passes, commit
git status
git commit -m "docs: reorganize documentation structure for V2 release"
```

### Option B: Manual (See DOCUMENTATION_AUDIT_REPORT.md Phase 1-7)

---

## What Happens

| Action | Count | Files Go To |
|--------|-------|-------------|
| Archive | 21 | `archive/development/{journals,phases,features,status,internal}/` |
| Delete | 3 | Removed (redundant entry points) |
| Keep | 7 | Root directory |
| Commit | 2 | QUICKSTART.md, docs/CRITICAL_STRATEGY_IMPLEMENTATION_GUIDE.md |

---

## After Cleanup

### Root Directory (7 files)
```
README.md                 # Entry point
QUICKSTART.md             # 15-min setup
CONTRIBUTING.md           # To create
ARCHITECTURE.md           # System design
FEATURES.md               # Feature list
RELEASES.md               # History
RELEASE_NOTES.md          # Current release
```

### User Journey
```
README.md
    ↓
QUICKSTART.md
    ↓
setup.py → verify_setup.py → quickstart.py
    ↓
First successful backtest (15 minutes)
```

---

## Next Steps

1. **Execute cleanup** (10 min)
   - Run automated script or manual commands
   - Verify with verification script

2. **Create missing docs** (1 hour)
   - `docs/ERROR_REFERENCE.md` - Error catalog
   - `CONTRIBUTING.md` - Contribution guide

3. **Commit and test** (30 min)
   - Review git status
   - Commit with descriptive message
   - Test new user flow

---

## Verify Success

```powershell
# Should show only 7 .md files
ls *.md

# Should pass all checks
python scripts\verify_documentation_cleanup.py

# Should show clean staging
git status
```

---

## Rollback (If Needed)

```powershell
# Reset everything
git reset --hard HEAD

# Or restore specific file
cp archive/development/journals/IMPLEMENTATION_JOURNAL.md .
```

---

## Read More

- **Full audit**: DOCUMENTATION_AUDIT_REPORT.md (24 KB)
- **Visual summary**: DOCUMENTATION_CLEANUP_SUMMARY.md (11 KB)
- **Verification script**: scripts/verify_documentation_cleanup.py
- **Execution script**: scripts/execute_documentation_cleanup.ps1

---

**Created**: 2025-11-07
**Status**: Ready to execute
**Risk**: Low (all changes reversible)
