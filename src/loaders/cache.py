"""On-disk cache for slow / metered loader steps (transcription, captioning).

Transcribing a 43 MB video or captioning an image goes through a metered API;
without a cache every command that loads the data folder pays for it again, and
an API error (e.g. 402 exhausted credits) would abort the whole load. Results
are keyed by file name + size + modification time, so editing or replacing the
file invalidates its entry.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

CACHE_DIR = Path(os.environ.get("LOADER_CACHE_DIR", Path(__file__).resolve().parents[2] / ".cache" / "loaders"))


def cached_text(path: Path, kind: str, produce: Callable[[], str | None]) -> str | None:
    """Return the cached text for `path`, else `produce()` it (and cache it).

    A failing `produce` (network error, exhausted credits, ...) is logged and
    yields None so the caller can skip the file instead of crashing.
    """
    stat = path.stat()
    key = hashlib.sha256(f"{kind}|{path.name}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()[:24]
    entry = CACHE_DIR / f"{kind}-{key}.txt"
    if entry.exists():
        return entry.read_text(encoding="utf-8")
    try:
        text = produce()
    except Exception as exc:  # noqa: BLE001 - any provider failure means "skip this file"
        logger.warning("%s failed for %s: %s", kind, path.name, str(exc).splitlines()[0][:200])
        return None
    if text:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        entry.write_text(text, encoding="utf-8")
    return text
