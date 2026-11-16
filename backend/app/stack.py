"""Load, cache and track the set of cutouts ("stack") for one sky position.

A stack is every SPHEREx exposure covering a position, cut out around it. Jobs
run in the background so the browser can poll progress; cutouts are cached on
disk per exposure, so a later visit only downloads exposures that are new in the
archive. That is what keeps the tool current as data releases arrive.

Developed with AI assistance (Claude, Anthropic). See README "Use of AI".
"""
from __future__ import annotations

import asyncio
import collections
import contextlib
import json
import logging
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from . import config, discovery, diskbudget, fetch, wavecal
from .discovery import Exposure

log = logging.getLogger("spx.stack")

_DEFAULT_FWHM = {1: 6.3, 2: 6.3, 3: 6.6, 4: 7.5, 5: 8.5, 6: 9.0}   # arcsec, used if header lacks PSF_FWHM


@dataclass
class ExposureData:
    exp: Exposure
    cut: fetch.RawCutout
    epoch: int = 0
    _wcs: WCS | None = field(default=None, repr=False)

    @property
    def wcs(self) -> WCS:
        if self._wcs is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._wcs = WCS(self.cut.header)
        return self._wcs

    @property
    def mjd(self) -> float:
        return float(self.cut.header.get("MJD-AVG", self.exp.mjd))

    @property
    def psf_fwhm_arcsec(self) -> float:
        value = float(self.cut.header.get("PSF_FWHM", 0.0) or 0.0)
        return value if 3.0 < value < 20.0 else _DEFAULT_FWHM[self.exp.detector]

    @property
    def omega_arcsec2(self) -> float:
        value = float(self.cut.header.get("OMEGA_MEDIAN", 0.0) or 0.0)
        return value if 20.0 < value < 60.0 else config.NATIVE_PIXEL_ARCSEC ** 2


def _download_order(pending: list[Exposure], epoch_of: dict[str, int], priority_band: int | None) -> list[Exposure]:
    """Priority band first; within each band, round-robin across epochs."""
    bands = [d for d in config.DETECTORS if d == priority_band] + [d for d in config.DETECTORS if d != priority_band]
    ordered: list[Exposure] = []
    for det in bands:
        by_epoch: dict[int, list[Exposure]] = {}
        for exp in sorted((e for e in pending if e.detector == det), key=lambda e: e.mjd):
            by_epoch.setdefault(epoch_of.get(exp.key, 0), []).append(exp)
        queues = [by_epoch[k] for k in sorted(by_epoch)]
        while any(queues):
            for q in queues:
                if q:
                    ordered.append(q.pop(0))
    return ordered


def snap_fov(fov_arcmin: float) -> float:
    return min(config.FOV_CHOICES_ARCMIN, key=lambda f: abs(f - fov_arcmin))


def stack_key(ra: float, dec: float, fov_arcmin: float) -> str:
    return f"{ra:.5f}_{dec:+.5f}_f{fov_arcmin:g}"


# --------------------------------------------------------------------------- disk cache


def _cutout_dir(key: str) -> Path:
    return config.CACHE_DIR / "cutouts" / key


def _save_cutout(key: str, exp: Exposure, cut: fetch.RawCutout) -> None:
    """Keep a cutout for next time. Best effort: a full or read-only disk must not lose the exposure."""
    if not diskbudget.has_room():
        return
    folder = _cutout_dir(key)
    path = folder / f"{exp.key}.npz"
    tmp = folder / f"{exp.key}.tmp.npz"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            tmp,
            sci=cut.sci,
            variance=cut.variance,
            flags=cut.flags,
            header=np.array(cut.header.tostring()),
            meta=np.array(json.dumps({
                "zodi_median": cut.zodi_median,
                "calib_id": cut.calib_id,
                "flag_bits": cut.flag_bits,
                "flag_descriptions": cut.flag_descriptions,
            })),
        )
        tmp.replace(path)
    except OSError as err:
        log.warning("could not store cutout %s: %s", path, err)
        tmp.unlink(missing_ok=True)


def _load_cutout(key: str, exp: Exposure) -> fetch.RawCutout | None:
    path = _cutout_dir(key) / f"{exp.key}.npz"
    if not path.exists():
        return None
    try:
        with np.load(path) as z:
            meta = json.loads(str(z["meta"]))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                header = fits.Header.fromstring(str(z["header"]))
            return fetch.RawCutout(
                sci=z["sci"], variance=z["variance"], flags=z["flags"], header=header,
                zodi_median=float(meta["zodi_median"]), calib_id=meta["calib_id"],
                flag_bits={k: int(v) for k, v in meta["flag_bits"].items()},
                flag_descriptions=meta["flag_descriptions"],
            )
    except Exception as err:  # a corrupt cache file must never break a request
        log.warning("dropping unreadable cache file %s: %s", path, err)
        path.unlink(missing_ok=True)
        return None


def _miss_path(key: str, exp: Exposure) -> Path:
    return _cutout_dir(key) / f"{exp.key}.miss"


def _read_json(path: Path, ttl_s: float | None):
    try:
        if ttl_s is not None and time.time() - path.stat().st_mtime > ttl_s:
            return None
        return json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None


def _write_json(path: Path, payload) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), "utf-8")
        tmp.replace(path)
    except OSError as err:              # a full or read-only disk must not break the request
        log.warning("could not store %s: %s", path, err)


# --------------------------------------------------------------------------- jobs


class StackJob:
    def __init__(self, key: str, ra: float, dec: float, fov_arcmin: float):
        self.key = key
        self.ra = ra
        self.dec = dec
        self.fov_arcmin = fov_arcmin
        self.cutout_deg = fov_arcmin / 60.0 * config.CUTOUT_PAD_FACTOR
        self.stage = "searching"            # searching | fetching | done | error
        self.error: str | None = None
        self.n_available = 0                # exposures in the archive at this position
        self.exposures: list[Exposure] = []
        self.epoch_of: dict[str, int] = {}
        self.total = 0
        self.done = 0
        self.failed = 0
        self.from_cache = 0
        self.data: dict[str, ExposureData] = {}
        self.wave: dict[int, wavecal.WaveTable] = {}
        self.wave_tried: set[int] = set()
        self.bands_ready: set[int] = set()
        self.remaining: dict[int, int] = {}
        self.flag_bits: dict[str, int] = {}
        self.flag_descriptions: dict[str, str] = {}
        self.version = 0
        self.last_used = time.time()
        self.task: asyncio.Task | None = None
        self._changed = asyncio.Event()

    # -- state helpers
    def bump(self) -> None:
        self.version += 1
        self._changed.set()
        self._changed = asyncio.Event()

    @property
    def finished(self) -> bool:
        return self.stage in ("done", "error")

    @property
    def nbytes(self) -> int:
        return sum(ed.cut.nbytes for ed in self.data.values())

    async def wait_for(self, predicate, timeout: float = 600.0) -> None:
        deadline = time.time() + timeout
        while not predicate():
            if self.stage == "error":
                raise RuntimeError(self.error or "load failed")
            if self.finished:
                return
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for SPHEREx data")
            try:
                await asyncio.wait_for(self._changed.wait(), timeout=min(remaining, 5.0))
            except asyncio.TimeoutError:
                pass

    def band_exposures(self, detector: int | None) -> list[ExposureData]:
        items = [ed for ed in self.data.values() if detector is None or ed.exp.detector == detector]
        return sorted(items, key=lambda ed: ed.mjd)

    @property
    def bad_mask(self) -> int:
        bits = self.flag_bits or config.DEFAULT_FLAG_BITS
        mask = 0
        for name in config.BAD_FLAG_NAMES:
            if name in bits:
                mask |= 1 << bits[name]
        return mask

    @property
    def saturation_mask(self) -> int:
        bits = self.flag_bits or config.DEFAULT_FLAG_BITS
        return sum(1 << bits[name] for name in ("OVERFLOW", "BLOOM") if name in bits)

    def flag_names(self, value: int) -> list[str]:
        bits = self.flag_bits or config.DEFAULT_FLAG_BITS
        return [name for name, bit in sorted(bits.items(), key=lambda kv: kv[1]) if value & (1 << bit)]

    # -- summaries sent to the browser
    def epochs_summary(self) -> list[dict]:
        epochs: dict[int, dict] = {}
        for exp in self.exposures:
            ep = self.epoch_of.get(exp.key, 0)
            slot = epochs.setdefault(ep, {"index": ep, "mjd_min": exp.mjd, "mjd_max": exp.mjd, "n": 0, "per_band": {}})
            slot["mjd_min"] = min(slot["mjd_min"], exp.mjd)
            slot["mjd_max"] = max(slot["mjd_max"], exp.mjd)
            slot["n"] += 1
            slot["per_band"][str(exp.detector)] = slot["per_band"].get(str(exp.detector), 0) + 1
        out = []
        for ep in sorted(epochs):
            slot = epochs[ep]
            slot["date_min"] = discovery.mjd_to_iso_date(slot["mjd_min"])
            slot["date_max"] = discovery.mjd_to_iso_date(slot["mjd_max"])
            slot["year"] = round(discovery.mjd_to_decimal_year(0.5 * (slot["mjd_min"] + slot["mjd_max"])), 3)
            out.append(slot)
        return out

    def snapshot(self) -> dict:
        per_band = {str(d): sum(1 for e in self.exposures if e.detector == d) for d in config.DETECTORS}
        loaded = {str(d): 0 for d in config.DETECTORS}
        for ed in list(self.data.values()):
            loaded[str(ed.exp.detector)] += 1
        return {
            "loaded_per_band": loaded,
            "loaded": len(self.data),
            "key": self.key,
            "stage": self.stage,
            "error": self.error,
            "ra": self.ra,
            "dec": self.dec,
            "fov_arcmin": self.fov_arcmin,
            "n_available": self.n_available,
            "total": self.total,
            "done": self.done,
            "failed": self.failed,
            "from_cache": self.from_cache,
            "bands_ready": sorted(self.bands_ready),
            "per_band": per_band,
            "epochs": self.epochs_summary() if self.exposures else [],
            "version": self.version,
        }


class StackManager:
    def __init__(self) -> None:
        self.jobs: dict[str, StackJob] = {}
        self.client: httpx.AsyncClient | None = None
        self.semaphore = asyncio.Semaphore(config.FETCH_CONCURRENCY)
        self.throttle = fetch.Throttle()
        self._wave_locks: dict[int, asyncio.Lock] = {}
        self._archive_failed_at = 0.0           # when the archive last failed to give a complete exposure list

    async def start(self) -> None:
        limits = httpx.Limits(max_connections=config.FETCH_CONCURRENCY + 6, max_keepalive_connections=config.FETCH_CONCURRENCY)
        self.client = httpx.AsyncClient(
            headers={"User-Agent": config.USER_AGENT}, limits=limits, follow_redirects=True, timeout=config.FETCH_TIMEOUT_S
        )

    async def stop(self) -> None:
        for job in self.jobs.values():
            if job.task and not job.task.done():
                job.task.cancel()
        if self.client:
            await self.client.aclose()

    # -- public API
    def get_job(self, ra: float, dec: float, fov_arcmin: float, priority_band: int | None = None) -> StackJob:
        ra = round(ra % 360.0, 5)
        dec = round(dec, 5)
        fov_arcmin = snap_fov(fov_arcmin)
        key = stack_key(ra, dec, fov_arcmin)
        job = self.jobs.get(key)
        if job is None or (job.stage == "error" and time.time() - job.last_used > 10):
            job = StackJob(key, ra, dec, fov_arcmin)
            self.jobs[key] = job
            job.task = asyncio.create_task(self._run(job, priority_band))
        job.last_used = time.time()
        self._evict()
        return job

    def _evict(self) -> None:
        total = sum(job.nbytes for job in self.jobs.values())
        if total <= config.STACK_CACHE_MAX_BYTES:
            return
        for job in sorted(self.jobs.values(), key=lambda j: j.last_used):
            if total <= config.STACK_CACHE_MAX_BYTES:
                break
            if job.finished and time.time() - job.last_used > 30:
                total -= job.nbytes
                del self.jobs[job.key]

    # -- internals
    async def _exposure_list(self, ra: float, dec: float) -> list[Exposure]:
        path = config.CACHE_DIR / "exposures" / f"{ra:.5f}_{dec:+.5f}.json"
        cached = _read_json(path, config.EXPOSURE_LIST_TTL_S)
        if cached is not None:
            return [Exposure.from_dict(d) for d in cached]
        stale = _read_json(path, None)          # older than the refresh interval, but still what we knew
        known = [Exposure.from_dict(d) for d in stale] if stale is not None else None
        # The archive failed a moment ago and this position is on record: do not make the visitor wait again.
        if known is not None and time.time() - self._archive_failed_at < config.ARCHIVE_RETRY_AFTER_S:
            return known
        try:
            search = discovery.search_exposures(self.client, ra, dec)
            # With a list on record, a slow archive must not hold up a view that can already be drawn.
            exposures = await (asyncio.wait_for(search, config.SEARCH_PATIENCE_S) if known is not None else search)
        except discovery.PartialSearch as partial:
            # One collection answered and another did not. That must never shrink the list:
            # keep every exposure on record, add any new ones, and store nothing, so the
            # archive is asked again next time.
            self._archive_failed_at = time.time()
            merged = {e.key: e for e in (known or [])}
            for exp in partial.exposures:
                merged.setdefault(exp.key, exp)
            log.warning("archive search answered only in part for %s %s; keeping the %d exposures on record",
                        ra, dec, len(known or []))
            return sorted(merged.values(), key=lambda e: (e.mjd, e.detector))
        except Exception:                       # archive unreachable or too slow: fall back to what we knew
            self._archive_failed_at = time.time()
            if known is not None:
                log.warning("archive search failed; using stale exposure list for %s %s", ra, dec)
                return known
            raise
        _write_json(path, [e.to_dict() for e in exposures])
        return exposures

    async def _run(self, job: StackJob, priority_band: int | None) -> None:
        try:
            exposures = await self._exposure_list(job.ra, job.dec)
            job.n_available = len(exposures)
            exposures = discovery.cap_exposures(exposures, config.MAX_EXPOSURES_PER_BAND)
            labels = discovery.assign_epochs([e.mjd for e in exposures])
            job.exposures = exposures
            job.epoch_of = {e.key: int(lab) for e, lab in zip(exposures, labels)}
            job.total = len(exposures)
            job.stage = "fetching"
            job.bump()

            job.remaining = {d: sum(1 for e in exposures if e.detector == d) for d in config.DETECTORS}
            for det, count in job.remaining.items():
                if count == 0:
                    job.bands_ready.add(det)
            job.stage = "fetching"
            job.bump()

            # Pass 1: everything already on disk (instant for a position seen before).
            cached = await asyncio.gather(*(asyncio.to_thread(_load_cutout, job.key, exp) for exp in exposures))
            pending: list[Exposure] = []
            for exp, cut in zip(exposures, cached):
                if cut is not None:
                    job.from_cache += 1
                    await self._register(job, exp, cut)
                    self._finish_one(job, exp)
                else:
                    pending.append(exp)
            job.bump()

            # Pass 2: download the rest. The band the user is looking at goes first,
            # and within a band exposures alternate between epochs so that an early
            # partial view already contains every epoch.
            queue = collections.deque(_download_order(pending, job.epoch_of, priority_band))
            workers = [asyncio.create_task(self._worker(job, queue)) for _ in range(min(config.FETCH_CONCURRENCY, len(queue)))]
            if workers:
                await asyncio.gather(*workers)
            job.bands_ready.update(config.DETECTORS)
            job.stage = "done"
            job.bump()
        except asyncio.CancelledError:
            raise
        except Exception as err:
            log.exception("stack job %s failed", job.key)
            job.stage = "error"
            job.error = f"{type(err).__name__}: {err}"
            job.bump()

    def _finish_one(self, job: StackJob, exp: Exposure) -> None:
        job.done += 1
        job.remaining[exp.detector] -= 1
        if job.remaining[exp.detector] <= 0:
            job.bands_ready.add(exp.detector)

    async def _register(self, job: StackJob, exp: Exposure, cut: fetch.RawCutout) -> None:
        if not job.flag_bits:
            job.flag_bits = cut.flag_bits
            job.flag_descriptions = cut.flag_descriptions
        await self._ensure_wave(job, exp, cut.calib_id)
        job.data[exp.key] = ExposureData(exp=exp, cut=cut, epoch=job.epoch_of.get(exp.key, 0))

    async def _worker(self, job: StackJob, queue: "collections.deque[Exposure]") -> None:
        while queue:
            exp = queue.popleft()
            try:
                if _miss_path(job.key, exp).exists():
                    raise fetch.CutoutUnavailable("previously unavailable")
                try:
                    cut = await fetch.fetch_cutout(
                        self.client, self.semaphore, self.throttle, exp.url, job.ra, job.dec, job.cutout_deg
                    )
                except fetch.CutoutUnavailable as err:
                    if "gave up" not in str(err):          # permanent (e.g. off-detector): remember it
                        path = _miss_path(job.key, exp)
                        with contextlib.suppress(OSError):
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_text(str(err), "utf-8")
                    raise
                await asyncio.to_thread(_save_cutout, job.key, exp, cut)
                await self._register(job, exp, cut)
            except fetch.CutoutUnavailable as err:
                job.failed += 1
                log.info("no cutout for %s: %s", exp.key, err)
            except Exception:
                job.failed += 1
                log.exception("unexpected failure loading %s", exp.key)
            finally:
                self._finish_one(job, exp)
                job.bump()

    async def _ensure_wave(self, job: StackJob, exp: Exposure, calib_id: str) -> None:
        det = exp.detector
        if det in job.wave or det in job.wave_tried:
            return
        lock = self._wave_locks.setdefault(det, asyncio.Lock())
        async with lock:
            if det in job.wave or det in job.wave_tried:
                return
            job.wave_tried.add(det)          # one attempt per detector per job, even if it fails
            table = await asyncio.to_thread(wavecal.load_table, det, calib_id)
            if table is None:
                try:
                    async with self.semaphore:
                        data = await fetch.fetch_full_cutout(self.client, self.throttle, exp.url, job.ra, job.dec)
                    table = await asyncio.to_thread(wavecal.table_from_fits, data, det, calib_id)
                    await asyncio.to_thread(wavecal.save_table, table)
                except Exception as err:
                    log.warning("could not fetch wavelength table for D%s (%s); trying a cached one", det, err)
                    table = await asyncio.to_thread(wavecal.load_any_table, det)
            if table is not None:
                job.wave[det] = table
