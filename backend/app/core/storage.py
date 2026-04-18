import logging
import os
import shutil
from pathlib import Path

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
