---
name: hidden-file-count-mismatch
description: Fix inconsistent file counts when hidden files (e.g. .DS_Store) are counted in some places but skipped in others, and related scan progress/reporting bugs
source: auto-skill
extracted_at: '2026-06-11T08:46:39.586Z'
---

## Problem
Python's `fnmatch.fnmatch(".DS_Store", "*")` returns `True` — unlike shell globbing, Python's fnmatch does **not** treat dotfiles specially. This causes a cascade of inconsistencies: hidden files get counted in file-count functions, then silently skipped during ingestion, and the scan progress UI shows nonsensical numbers like "processing 2 / 1" and "2 Files, 0 processed".

## Symptoms
1. File count reports N files, but only N-1 are actually ingested
2. During scan: progress shows "processing X / 1" where X > 1 (scanned > total)
3. After scan: displays "X Files, 0 processed" instead of the correct counts
4. Logs show `file_count=X` on creation, `total_files=X` on scan init, but only one file is processed

## Root Cause
Hidden file handling is inconsistent across **three** code paths:

1. **Counting functions** (`count_files_in_folder`, `_count_files_in_folder`): use Python's `rglob`/`glob` which includes dotfiles, and **do not filter them out**
2. **Pattern matching** (`_matches_pattern`): uses `fnmatch.fnmatch()` which matches dotfiles with `*` patterns, and **does not filter them out**
3. **Ingestion handlers** (`_handle_file_in_scan`): **do** skip hidden files via `if fname.startswith(".")`

**Additionally**, the scan loop increments `summary["scanned"]` **before** the pattern check, so hidden files are counted in `scanned` even though they're later skipped.

**Finally**, the scan endpoint sets `last_scan_total_files = result.get("scanned", 0)` which conflates "files that matched the pattern" with "total files in the folder", and `last_scan_processed = result.get("new", 0) + result.get("modified", 0)` which only counts newly ingested files, not all scanned ones.

## Fix

### 1. File counting — add `and not f.name.startswith(".")` to the filter
```python
# Before
all_files.update(f for f in matched if f.is_file())

# After
all_files.update(f for f in matched if f.is_file() and not f.name.startswith("."))
```

### 2. Pattern matching — add early return for dotfiles
```python
# Before
def _matches_pattern(self, filepath: str, pattern: str = "*") -> bool:
    if pattern == "*":
        return True
    # ... pattern matching logic

# After
def _matches_pattern(self, filepath: str, pattern: str = "*") -> bool:
    fname = os.path.basename(filepath)
    # Exclude hidden files regardless of pattern
    if fname.startswith("."):
        return False
    if pattern == "*":
        return True
    # ... pattern matching logic
```

### 3. Scan loop — move `summary["scanned"]` increment to AFTER the pattern check
```python
# Before
for fname in files:
    fpath = os.path.join(root, fname)
    summary["scanned"] += 1                    # ← incremented for ALL files
    if not self._matches_pattern(fpath, ds.scan_pattern):
        summary["skipped"] += 1
        continue
    # ... process file

# After
for fname in files:
    fpath = os.path.join(root, fname)
    # ← no counted here
    if not self._matches_pattern(fpath, ds.scan_pattern):
        summary["skipped"] += 1
        continue
    summary["scanned"] += 1                    # ← only for files that match
    # ... process file
```

### 4. Scan endpoint — use correct values for total_files and processed
```python
# Before
ds.last_scan_total_files = result.get("scanned", 0)       # ← wrong: only files matching pattern
ds.last_scan_processed = result.get("new", 0) + result.get("modified", 0)  # ← wrong: only new/modified

# After
ds.last_scan_total_files = latest_file_count              # ← total files (from count_files_in_folder before scan)
ds.last_scan_processed = result.get("scanned", 0)         # ← all files that were actually scanned (matching pattern)
```

## Key files to check
- `backend/app/api/api_v1/datastores.py` — `count_files_in_folder()` function, `trigger_datastore_scan()` endpoint
- `backend/app/services/datastore_watcher.py` — `_count_files_in_folder()` (DataStoreWatcher class), `_matches_pattern()` (may have multiple classes in different classes), `scan_single_datastore()` scan loop

## General principle
When using Python's `glob`/`rglob`/`fnmatch`, remember they don't exclude dotfiles. If your codebase treats dotfiles as hidden/special (e.g. skipping them during processing), apply the exclusion **consistently** in counting, matching, and processing steps. Also verify that progress counters are incremented in the correct order relative to filtering logic, and that UI-facing fields map to semantically correct values.
