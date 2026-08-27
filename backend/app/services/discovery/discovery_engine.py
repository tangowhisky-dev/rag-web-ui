"""Discovery engine — walks datastore folders, hashes files, compares against manifest.

Walks all registered datastore folders, computes SHA-256 hashes concurrently,
compares against the DataStoreFileManifest table, and returns structured
results (new, modified, deleted file lists).

All log messages use the ``[DISCOVERY]`` prefix for diagnostic inspection.
"""

from __future__ import annotations

import hashlib
import fnmatch
import logging
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from multiprocessing import cpu_count
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.db.session import SessionLocal
from app.models.datastore import DataStore, DataStoreFileManifest

_FLUSH_LOCK = threading.Lock()

logger = logging.getLogger(__name__)

# ── File hashing ──────────────────────────────────────────────────────────────


def hash_file(file_path: str) -> str:
    """Compute SHA-256 hash of a file using 8192-byte chunks.

    Returns the hex digest string, or empty string on I/O error or if the
    file's size changes while being read.
    """
    try:
        size_before = os.path.getsize(file_path)
    except OSError:
        logger.warning("[DISCOVERY] failed_to_get_size path=%s", file_path)
        return ""

    h = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    except OSError:
        logger.warning("[DISCOVERY] failed_to_compute_hash path=%s", file_path)
        return ""

    try:
        size_after = os.path.getsize(file_path)
    except OSError:
        logger.warning("[DISCOVERY] failed_to_get_size path=%s", file_path)
        return ""

    if size_before != size_after:
        logger.warning("[DISCOVERY] file size changed during hashing: %s", file_path)
        return ""

    return h.hexdigest()


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class DiscoveryConfig:
    """Configuration for discovery scans.

    Attributes:
        max_workers:  Maximum concurrent hash workers.  Defaults to
                      ``min(cpu_count(), 32)`` so resource usage stays
                      bounded even on large servers.
        scan_pattern: File name pattern passed to ``fnmatch`` (default
                      ``"*"`` to match everything).
        skip_hidden:  When ``True`` files whose basename starts with
                      ``"."`` are excluded.
    """

    max_workers: int = field(default=None)  # type: ignore[assignment]
    scan_pattern: str = "*"
    skip_hidden: bool = True

    def __post_init__(self) -> None:
        if self.max_workers is None:
            self.max_workers = min(cpu_count(), 16)


# ── Pattern matching ──────────────────────────────────────────────────────────


def _matches_pattern(file_path: str, pattern: str, skip_hidden: bool = True) -> bool:
    """Return ``True`` when *file_path* should be scanned.

    Checks *fnmatch* against the file name and, when *skip_hidden* is
    ``True``, excludes files whose basename starts with ``"."``.
    Also excludes temp/lock files from common editors and office suites.
    """
    base = os.path.basename(file_path)

    if skip_hidden and base.startswith("."):
        return False
    # Skip temp/lock files: ~$file.docx (MS Office), .~file (Emacs/gedit),
    # file.tmp, file.swp, file.bak
    if base.startswith("~$") or base.startswith(".~"):
        return False
    ext = os.path.splitext(base)[1].lower()
    if ext in (".tmp", ".swp", ".swo", ".bak", ".lock"):
        return False

    return fnmatch.fnmatch(base, pattern)


# ── Single-file collection ────────────────────────────────────────────────────


def hash_and_collect(file_path: str, config: DiscoveryConfig) -> dict[str, Any] | None:
    """Hash a single file and return metadata, or ``None`` if skipped.

    Returns a dict with keys:
        ``file_path``, ``file_hash``, ``file_size``, ``file_mtime``

    Returns ``None`` when the file is skipped by pattern or hidden filters.
    """
    if not _matches_pattern(file_path, config.scan_pattern, config.skip_hidden):
        return None

    file_hash = hash_file(file_path)
    if not file_hash:
        return None

    try:
        st = os.stat(file_path)
    except OSError:
        logger.warning("[DISCOVERY] failed_to_stat path=%s", file_path)
        return None

    return {
        "file_path": file_path,
        "file_hash": file_hash,
        "file_size": st.st_size,
        "file_mtime": st.st_mtime_ns,
    }


# ── Worker ────────────────────────────────────────────────────────────────────


def _hash_worker(args: tuple[str, "DiscoveryConfig"]) -> dict[str, Any] | None:
    """Worker function for :class:`concurrent.futures.ThreadPoolExecutor`.

    Accepts a ``(file_path, config)`` tuple and delegates to
    :func:`hash_and_collect`.
    """
    file_path, config = args
    return hash_and_collect(file_path, config)


def stat_and_collect(file_path: str, config: DiscoveryConfig) -> dict[str, Any] | None:
    """Stat a single file and return metadata without hashing.

    Returns a dict with keys:
        ``file_path``, ``file_size``, ``file_mtime``

    Returns ``None`` when the file is skipped by pattern/hidden filters
    or is inaccessible.
    """
    if not _matches_pattern(file_path, config.scan_pattern, config.skip_hidden):
        return None
    try:
        st = os.stat(file_path)
    except OSError:
        logger.warning("[DISCOVERY] failed_to_stat path=%s", file_path)
        return None
    return {
        "file_path": file_path,
        "file_size": st.st_size,
        "file_mtime": st.st_mtime_ns,
    }


def _stat_worker(args: tuple[str, "DiscoveryConfig"]) -> dict[str, Any] | None:
    """Worker function for stat-only concurrent collection."""
    file_path, config = args
    return stat_and_collect(file_path, config)


# ── DataStore discovery ──────────────────────────────────────────────────────


@dataclass
class DiscoveryResult:
    """Result of scanning a single datastore.

    Attributes:
        datastore_id:            DataStore database ID.
        datastore_name:          Human-readable DataStore name.
        folder_path:             Absolute folder path that was scanned.
        new_files:               List of dicts for files not in manifest.
        modified_files:          List of dicts for files whose hash changed.
        deleted_files:           List of dicts for manifest entries missing
                                 from the filesystem.
        skipped_files:           Count of files skipped by pattern/hidden.
        total_files_discovered:  Total number of files actually scanned.
        elapsed_ms:              Wall-clock time in milliseconds.

    ``to_dict()`` returns a serialisable ``dict`` for SSE / API responses.
    """

    datastore_id: int
    datastore_name: str
    folder_path: str
    new_files: list[dict[str, Any]] = field(default_factory=list)
    modified_files: list[dict[str, Any]] = field(default_factory=list)
    deleted_files: list[dict[str, Any]] = field(default_factory=list)
    skipped_files: int = 0
    total_files_discovered: int = 0
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable dict for SSE / API consumers."""
        return {
            "datastore_id": self.datastore_id,
            "datastore_name": self.datastore_name,
            "folder_path": self.folder_path,
            "new_files": self.new_files,
            "modified_files": self.modified_files,
            "deleted_files": self.deleted_files,
            "skipped_files": self.skipped_files,
            "total_files_discovered": self.total_files_discovered,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


def _walk_files(folder_path: str) -> list[str]:
    """Walk *folder_path* and return absolute file paths (no symlinks)."""
    paths: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(folder_path, followlinks=False):
        for fname in filenames:
            paths.append(os.path.join(dirpath, fname))
    return paths


def _classify_files(
    manifest_map: dict[str, DataStoreFileManifest],
    collected: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare collected metadata against the manifest.

    Returns ``(new_files, modified_files, deleted_files)``.
    """
    collected_map: dict[str, dict[str, Any]] = {}
    for entry in collected:
        rel = entry["file_path"]
        collected_map[rel] = entry

    new_files: list[dict[str, Any]] = []
    modified_files: list[dict[str, Any]] = []

    for rel, meta in collected_map.items():
        existing = manifest_map.get(rel)
        if existing is None:
            new_files.append(meta)
        elif existing.file_hash != meta["file_hash"]:
            modified_files.append(meta)

    deleted: list[dict[str, Any]] = []
    for rel, manifest_entry in manifest_map.items():
        if rel not in collected_map:
            deleted.append({
                "file_path": rel,
                "old_hash": manifest_entry.file_hash,
                "old_size": manifest_entry.file_size,
            })

    return new_files, modified_files, deleted


def _upsert_manifest(
    db: Session,
    datastore_id: int,
    new_files: list[dict[str, Any]],
    modified_files: list[dict[str, Any]],
    manifest_map: dict[str, DataStoreFileManifest],
) -> None:
    """Persist new and updated manifest entries.

    Updates existing entries in-place (for modified files) and
    inserts new entries via :meth:`Session.add`.  A lock serialises
    the flush so :func:`discover_all` can safely call this from
    multiple threads.
    """
    now = datetime.now(timezone.utc)

    # Update modified entries in place
    for entry in modified_files:
        existing = manifest_map.get(entry["file_path"])
        if existing:
            existing.file_hash = entry["file_hash"]
            existing.file_size = entry["file_size"]
            existing.file_mtime = entry.get("file_mtime")

    # Add new entries
    for entry in new_files:
        db.add(
            DataStoreFileManifest(
                datastore_id=datastore_id,
                file_path=entry["file_path"],
                file_hash=entry["file_hash"],
                file_size=entry["file_size"],
                file_mtime=entry.get("file_mtime"),
                discovered_at=now,
                updated_at=now,
            )
        )

    if modified_files or new_files:
        with _FLUSH_LOCK:
            db.flush()
        db.commit()


def discover_datastore(
    datastore_id: int,
    force_full_hash: bool = False,
) -> DiscoveryResult:
    """Walk a datastore's folder, stat files, and compare against the manifest.

    Uses a two-phase stat-first approach for incremental scanning:

    Phase 1 — stat every file concurrently (cheap: one syscall per file,
    no content read).  Compare (mtime, size) against the manifest.
    Files where both match are **unchanged** — reuse the manifest hash
    and skip hashing entirely.

    Phase 2 — hash only the candidates (new files + files where mtime or
    size changed).  This is the expensive part (full file read), but it
    only runs for files that actually changed.

    When *force_full_hash* is ``True``, skip the stat comparison and hash
    every file.  Use this for periodic safety-net scans to catch the edge
    case where content changed but mtime/size didn't (rare, but possible
    with some file sync tools that preserve mtime).

    Returns a :class:`DiscoveryResult` classifying files as *new*,
    *modified*, or *deleted*.  New and updated entries (including
    ``file_mtime``) are persisted back to the manifest table.

    Creates its own session — safe to call from multiple threads concurrently.
    """
    start = time.monotonic()

    db = SessionLocal()
    try:
        # Load the DataStore and its manifest entries in a single query.
        ds_stmt = (
            select(DataStore)
            .options(selectinload(DataStore.manifest_entries))
            .where(DataStore.id == datastore_id)
        )
        ds = db.scalars(ds_stmt).first()

        if ds is None:
            logger.warning("[DISCOVERY] datastore_not_found id=%d", datastore_id)
            return DiscoveryResult(
                datastore_id=datastore_id,
                datastore_name="unknown",
                folder_path="",
            )

        if not ds.is_active:
            logger.info("[DISCOVERY] datastore_inactive id=%d", datastore_id)
            return DiscoveryResult(
                datastore_id=datastore_id,
                datastore_name=ds.name,
                folder_path=ds.folder_path,
            )

        if not ds.folder_path or not os.path.isdir(ds.folder_path):
            logger.warning(
                "[DISCOVERY] datastore_folder_missing id=%d path=%s",
                datastore_id,
                ds.folder_path,
            )
            return DiscoveryResult(
                datastore_id=datastore_id,
                datastore_name=ds.name,
                folder_path=ds.folder_path or "",
            )

        # Build manifest lookup keyed by file_path.
        manifest_map: dict[str, DataStoreFileManifest] = {
            m.file_path: m for m in ds.manifest_entries
        }

        logger.info(
            "[DISCOVERY] scanning_start datastore_id=%d folder=%s force_full_hash=%s",
            datastore_id,
            ds.folder_path,
            force_full_hash,
        )

        # Walk the folder.
        file_paths = _walk_files(ds.folder_path)
        logger.info(
            "[DISCOVERY] files_walking datastore_id=%d count=%d",
            datastore_id,
            len(file_paths),
        )

        config = DiscoveryConfig()
        skipped = 0
        collected: list[dict[str, Any]] = []

        if force_full_hash:
            # Hash every file — safety-net path.
            with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
                futures = {
                    executor.submit(_hash_worker, (fp, config)): fp
                    for fp in file_paths
                }
                for future in as_completed(futures):
                    fp = futures[future]
                    try:
                        result = future.result()
                    except Exception:
                        logger.exception(
                            "[DISCOVERY] worker_exception path=%s", fp
                        )
                        skipped += 1
                        continue
                    if result is None:
                        skipped += 1
                    else:
                        collected.append(result)

            logger.info(
                "[DISCOVERY] hashing_done datastore_id=%d collected=%d skipped=%d (force_full_hash=True)",
                datastore_id,
                len(collected),
                skipped,
            )
        else:
            # ── Phase 1: stat all files concurrently ───────────────────
            stat_results: dict[str, dict[str, Any]] = {}
            with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
                futures = {
                    executor.submit(_stat_worker, (fp, config)): fp
                    for fp in file_paths
                }
                for future in as_completed(futures):
                    fp = futures[future]
                    try:
                        result = future.result()
                    except Exception:
                        logger.exception(
                            "[DISCOVERY] stat_worker_exception path=%s", fp
                        )
                        skipped += 1
                        continue
                    if result is None:
                        skipped += 1
                    else:
                        stat_results[result["file_path"]] = result

            # ── First-scan fast path ────────────────────────────────────
            # When the manifest is empty (first scan of a datastore), every
            # file is new by definition. Skip the hash phase entirely —
            # downstream consumers (watcher, recovery) hash each file lazily
            # during ingestion, interleaved with the conversion read at
            # 4-way concurrency instead of a 16-way batch that saturates
            # network mounts. The manifest is populated incrementally as
            # each file is processed.
            if not manifest_map:
                logger.info(
                    "[DISCOVERY] first_scan_skip_hashing datastore_id=%d total=%d skipped=%d",
                    datastore_id,
                    len(stat_results),
                    skipped,
                )
                collected = [
                    {
                        "file_path": fp,
                        "file_hash": "",  # placeholder — downstream hashes lazily
                        "file_size": meta["file_size"],
                        "file_mtime": meta["file_mtime"],
                    }
                    for fp, meta in stat_results.items()
                ]
            else:
                # ── Compare stats against manifest ──────────────────────
                # Files where (mtime, size) match the manifest are unchanged.
                # Reuse the manifest hash and skip hashing.
                unchanged: list[dict[str, Any]] = []
                candidates: list[str] = []  # file_paths that need hashing

                for fp, stat_meta in stat_results.items():
                    existing = manifest_map.get(fp)
                    if (
                        existing is not None
                        and existing.file_mtime is not None
                        and existing.file_mtime == stat_meta["file_mtime"]
                        and existing.file_size == stat_meta["file_size"]
                    ):
                        # Unchanged — reuse manifest hash.
                        unchanged.append({
                            "file_path": fp,
                            "file_hash": existing.file_hash,
                            "file_size": stat_meta["file_size"],
                            "file_mtime": stat_meta["file_mtime"],
                        })
                    else:
                        # New or modified — needs hashing.
                        candidates.append(fp)

                logger.info(
                    "[DISCOVERY] stat_done datastore_id=%d total=%d unchanged=%d candidates=%d skipped=%d",
                    datastore_id,
                    len(stat_results),
                    len(unchanged),
                    len(candidates),
                    skipped,
                )

                # ── Phase 2: hash only candidates ──────────────────────
                hashed: list[dict[str, Any]] = []
                if candidates:
                    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
                        futures = {
                            executor.submit(_hash_worker, (fp, config)): fp
                            for fp in candidates
                        }
                        for future in as_completed(futures):
                            fp = futures[future]
                            try:
                                result = future.result()
                            except Exception:
                                logger.exception(
                                    "[DISCOVERY] worker_exception path=%s", fp
                                )
                                skipped += 1
                                continue
                            if result is None:
                                skipped += 1
                            else:
                                hashed.append(result)

                    logger.info(
                        "[DISCOVERY] hashing_done datastore_id=%d hashed=%d skipped=%d",
                        datastore_id,
                        len(hashed),
                        skipped - (len(stat_results) - len(unchanged) - len(candidates)),
                    )

                collected = unchanged + hashed

        # Classify.
        new_files, modified_files, deleted_files = _classify_files(
            manifest_map, collected
        )

        logger.info(
            "[DISCOVERY] classification_done datastore_id=%d new=%d modified=%d deleted=%d",
            datastore_id,
            len(new_files),
            len(modified_files),
            len(deleted_files),
        )

        # Persist new/updated entries.
        # Skip manifest upsert for first-scan entries with empty hashes —
        # the manifest will be populated incrementally by downstream
        # consumers (watcher/recovery) as each file is hashed during ingestion.
        has_real_hashes = any(f.get("file_hash") for f in new_files)
        if has_real_hashes or modified_files:
            _upsert_manifest(db, datastore_id, new_files, modified_files, manifest_map)

        elapsed = (time.monotonic() - start) * 1000

        result = DiscoveryResult(
            datastore_id=ds.id,
            datastore_name=ds.name,
            folder_path=ds.folder_path,
            new_files=new_files,
            modified_files=modified_files,
            deleted_files=deleted_files,
            skipped_files=skipped,
            total_files_discovered=len(collected),
            elapsed_ms=elapsed,
        )

        logger.info(
            "[DISCOVERY] scan_complete datastore_id=%d elapsed_ms=%.1f new=%d modified=%d deleted=%d",
            result.datastore_id,
            result.elapsed_ms,
            len(result.new_files),
            len(result.modified_files),
            len(result.deleted_files),
        )

        return result
    finally:
        db.close()


# ── All datastores ────────────────────────────────────────────────────────────


def discover_all(db: Session) -> list[DiscoveryResult]:
    """Run :func:`discover_datastore` for every active DataStore concurrently.

    Returns a list of :class:`DiscoveryResult` — one per active datastore.
    Inactive datastores are skipped silently.
    """
    ids_stmt = select(DataStore.id).where(DataStore.is_active == True)  # noqa: E712
    active_ids = db.scalars(ids_stmt).all()

    if not active_ids:
        logger.info("[DISCOVERY] no_active_datastores")
        return []

    logger.info("[DISCOVERY] discovering_all count=%d", len(active_ids))

    # Discover each datastore concurrently (each gets its own session).
    results: list[DiscoveryResult] = []
    with ThreadPoolExecutor(max_workers=min(len(active_ids), 8)) as executor:
        futures = {
            executor.submit(discover_datastore, did): did
            for did in active_ids
        }
        for future in as_completed(futures):
            did = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception:
                logger.exception("[DISCOVERY] datastore_exception id=%d", did)
                results.append(
                    DiscoveryResult(
                        datastore_id=did,
                        datastore_name="error",
                        folder_path="",
                        elapsed_ms=0.0,
                    )
                )

    logger.info(
        "[DISCOVERY] all_done total=%d", len(results)
    )
    return results
