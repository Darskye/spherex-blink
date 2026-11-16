"""A size limit for the on-disk cache, for hosts with a small disk.

With SPX_DISK_CACHE_MB set, the large cache files (cutouts and frame payloads)
stop being written once the cache folder reaches that size. The viewer keeps
working from memory; a position opened after that is simply downloaded again on
its next visit. Nothing is deleted.

Developed with AI assistance (Claude, Anthropic). See README "Use of AI".
"""
from __future__ import annotations

import logging
import os
import time

from . import config

log = logging.getLogger("spx.disk")

_checked_at = 0.0
_has_room = True


def _folder_bytes(folder) -> int:
    total = 0
    for root, _dirs, files in os.walk(folder):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def has_room() -> bool:
    """True while the cache folder is under its limit. Measured at most once per check interval."""
    global _checked_at, _has_room
    if not config.DISK_CACHE_MAX_BYTES:
        return True
    now = time.time()
    if now - _checked_at >= config.DISK_CHECK_INTERVAL_S:
        _checked_at = now
        used = _folder_bytes(config.CACHE_DIR)
        room = used < config.DISK_CACHE_MAX_BYTES
        if room != _has_room:
            log.warning("disk cache is %s its limit (%d MB used of %d MB)", "under" if room else "at",
                        used // 2**20, config.DISK_CACHE_MAX_BYTES // 2**20)
        _has_room = room
    return _has_room
