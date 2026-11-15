"""Build, and remember, the binary frame payloads the browser asks for.

Aligning and coadding a few hundred exposures takes several seconds, so finished
payloads are kept in memory and on disk. The disk copy is keyed by exactly which
exposures went into it: when new data arrives for a position the key changes and
the frames are rebuilt, and a server restart costs nothing.

Developed with AI assistance (Claude, Anthropic). See README "Use of AI".
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import OrderedDict

from . import config, process
from .stack import ExposureData, StackJob

log = logging.getLogger("spx.payloads")

# Bump when the image-building recipe changes, so stale files are not served.
RECIPE_VERSION = "2"

_memory: OrderedDict[str, bytes] = OrderedDict()
_locks: dict[str, asyncio.Lock] = {}


def signature(exposures: list[ExposureData]) -> str:
    """Short fingerprint of a set of exposures and the image recipe."""
    digest = hashlib.sha1(RECIPE_VERSION.encode())
    for key in sorted(ed.exp.key for ed in exposures):
        digest.update(key.encode())
    return digest.hexdigest()[:14]


def _build(job: StackJob, band: str, kind: str, mask: bool, exposures: list[ExposureData], complete: bool) -> bytes:
    if band == "colour":
        header, data, flags = process.build_colour_frames(job, mask, exposures)
    else:
        detector = None if band == "all" else int(band)
        header, data, flags = process.build_frames(job, detector, kind, mask, exposures)
    header["complete"] = complete
    return process.pack_frames(header, data, flags)


async def frames(job: StackJob, band: str, kind: str, mask: bool, exposures: list[ExposureData], complete: bool) -> bytes:
    """Payload for one view. ``exposures`` is a snapshot taken on the event loop."""
    name_signature = signature(exposures)
    name = f"{job.key}_{band}_{kind}_{int(mask)}_{name_signature}"
    cached = _memory.get(name)
    if cached is not None:
        _memory.move_to_end(name)
        return cached

    lock = _locks.setdefault(name, asyncio.Lock())
    async with lock:
        cached = _memory.get(name)
        if cached is None:
            path = config.CACHE_DIR / "frames" / f"{name}.bin"
            if complete and path.exists():
                try:
                    cached = await asyncio.to_thread(path.read_bytes)
                except OSError as err:
                    log.warning("unreadable frame cache %s: %s", path, err)
            if cached is None:
                cached = await asyncio.to_thread(_build, job, band, kind, mask, exposures, complete)
                if complete:
                    await asyncio.to_thread(_store, path, cached)
            _memory[name] = cached
            while len(_memory) > config.FRAME_CACHE_MAX_ITEMS:
                _memory.popitem(last=False)
    _locks.pop(name, None)
    return cached


def _store(path, payload: bytes) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(payload)
        tmp.replace(path)
    except OSError as err:              # a full or read-only disk must not break the request
        log.warning("could not store frame cache %s: %s", path, err)
