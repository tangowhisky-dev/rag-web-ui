import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import AsyncIterator

from app.core.config import settings

logger = logging.getLogger(__name__)


def _base() -> Path:
    return Path(settings.UPLOAD_DIR)


def init_storage() -> None:
    """Ensure the uploads base directory exists."""
    _base().mkdir(parents=True, exist_ok=True)
    logger.info(f"Storage initialised at {_base()}")


def get_abs_path(object_path: str) -> str:
    """Return the absolute filesystem path for a relative object_path."""
    return str(_base() / object_path)


def save_file(object_path: str, content: bytes) -> None:
    """Write *content* to the given relative path, creating directories as needed."""
    abs_path = _base() / object_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_bytes(content)
    logger.info(f"Saved file: {abs_path}")


async def save_file_stream(
    object_path: str,
    source: AsyncIterator[bytes],
    chunk_size: int = 1024 * 1024,
) -> tuple[str, int]:
    """Stream *source* to disk, computing SHA-256 and counting bytes.

    Avoids loading the entire file into memory.  Returns
    ``(sha256_hexdigest, total_bytes)``.
    """
    abs_path = _base() / object_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    total = 0
    with open(abs_path, "wb") as f:
        async for chunk in source:
            f.write(chunk)
            h.update(chunk)
            total += len(chunk)
    logger.info(f"Saved file (streamed): {abs_path} ({total} bytes)")
    return h.hexdigest(), total


def move_file(src_path: str, dst_path: str) -> None:
    """Move a file from *src_path* to *dst_path* (both relative to UPLOAD_DIR)."""
    src = _base() / src_path
    dst = _base() / dst_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    logger.info(f"Moved file: {src} -> {dst}")


def delete_file(object_path: str) -> None:
    """Delete a single file. Silently ignores missing files."""
    abs_path = _base() / object_path
    try:
        abs_path.unlink()
        logger.info(f"Deleted file: {abs_path}")
    except FileNotFoundError:
        logger.warning(f"File not found (skip delete): {abs_path}")


def kb_path(user_id: int, kb_id: int) -> str:
    """Return the relative path prefix for a user's knowledge base."""
    return f"user_{user_id}/kb_{kb_id}"


def delete_kb_files(user_id: int, kb_id: int) -> None:
    """Remove the entire user_{user_id}/kb_{kb_id}/ directory tree."""
    kb_dir = _base() / f"user_{user_id}" / f"kb_{kb_id}"
    if kb_dir.exists():
        shutil.rmtree(kb_dir)
        logger.info(f"Deleted KB directory: {kb_dir}")
    else:
        logger.info(f"KB directory not found (nothing to delete): {kb_dir}")


def list_files(prefix: str) -> list[str]:
    """Return relative paths of all files whose path starts with *prefix*."""
    base = _base()
    prefix_path = base / prefix
    if not prefix_path.exists():
        return []
    results = []
    for p in prefix_path.rglob("*"):
        if p.is_file():
            results.append(str(p.relative_to(base)))
    return results


# ── Ephemeral chat-file uploads ───────────────────────────────────────────────

def ephemeral_chat_dir(chat_id: int) -> Path:
    """Return (and create) the ephemeral upload dir for a chat."""
    d = _base() / "ephemeral" / str(chat_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_ephemeral_file(chat_id: int, filename: str, content: bytes) -> str:
    """Write file bytes to uploads/ephemeral/{chat_id}/{filename}.

    If a file with the same name already exists in the directory, appends _1, _2, …
    before the extension until a free slot is found.
    Returns the absolute path so markitdown can read it.
    """
    d = ephemeral_chat_dir(chat_id)
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    candidate = d / filename
    counter = 1
    while candidate.exists():
        candidate = d / f"{stem}_{counter}{suffix}"
        counter += 1
    candidate.write_bytes(content)
    logger.info("[chat_files] saved ephemeral file: %s", candidate)
    return str(candidate)


def delete_ephemeral_chat_files(chat_id: int) -> None:
    """Remove the entire uploads/ephemeral/{chat_id}/ directory tree.
    Called when the parent chat is deleted."""
    d = _base() / "ephemeral" / str(chat_id)
    if d.exists():
        shutil.rmtree(d)
        logger.info("[chat_files] deleted ephemeral dir: %s", d)
    else:
        logger.debug("[chat_files] ephemeral dir not found (skip): %s", d)
