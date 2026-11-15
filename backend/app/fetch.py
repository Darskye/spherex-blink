"""Download SPHEREx spectral-image cutouts from IRSA, efficiently.

IRSA's cutout service returns a multi-extension FITS file laid out as
PRIMARY, IMAGE, FLAGS, VARIANCE, ZODI, EPSF, WCS-WAVE. The EPSF table is about
3.9 MB regardless of cutout size and is not needed for viewing, so the download
is stopped as soon as ZODI has arrived (roughly the first 130 KB). The
wavelength lookup table (WCS-WAVE) is identical for every exposure of a
detector, so it is fetched once per detector and cached (see ``wavecal``).

Developed with AI assistance (Claude, Anthropic). See README "Use of AI".
"""
from __future__ import annotations

import asyncio
import logging
import random
import re
import time
import warnings
from dataclasses import dataclass

import httpx
import numpy as np
from astropy.io import fits

from . import config

log = logging.getLogger("spx.fetch")

_BLOCK = 2880
_NEEDED_HDUS = 5        # PRIMARY + IMAGE + FLAGS + VARIANCE + ZODI
_CALIB_RE = re.compile(r"\[CALIB\]\s+spectral_wcs\s+(\S+)")
_DROP_CARDS = ("HISTORY", "COMMENT", "")


class CutoutUnavailable(Exception):
    """The archive could not make a cutout for this exposure (e.g. off the detector)."""


class FitsPrefixReader:
    """Incrementally find where the first ``n_hdus`` HDUs of a FITS stream end."""

    def __init__(self, n_hdus: int = _NEEDED_HDUS):
        self.buf = bytearray()
        self.n_hdus = n_hdus
        self._hdus_done = 0
        self._pos = 0              # start of the HDU currently being parsed
        self._scan = 0             # next header block to inspect
        self.end_offset: int | None = None

    def feed(self, chunk: bytes) -> bool:
        """Add bytes; return True once the needed HDUs are completely buffered."""
        self.buf.extend(chunk)
        while self.end_offset is None:
            # Look for the END card, one 2880-byte header block at a time.
            found_end = False
            while self._scan + _BLOCK <= len(self.buf):
                block = bytes(self.buf[self._scan:self._scan + _BLOCK])
                self._scan += _BLOCK
                if any(block[i:i + 80].rstrip() == b"END" for i in range(0, _BLOCK, 80)):
                    found_end = True
                    break
            if not found_end:
                return False
            header = fits.Header.fromstring(bytes(self.buf[self._pos:self._scan]).decode("ascii", "replace"))
            naxis = int(header.get("NAXIS", 0))
            if naxis == 0:
                data_bytes = 0
            else:
                npix = 1
                for i in range(1, naxis + 1):
                    npix *= int(header[f"NAXIS{i}"])
                data_bytes = (abs(int(header["BITPIX"])) // 8) * int(header.get("GCOUNT", 1)) * (
                    int(header.get("PCOUNT", 0)) + npix
                )
            padded = ((data_bytes + _BLOCK - 1) // _BLOCK) * _BLOCK
            self._pos = self._scan + padded
            self._scan = self._pos
            self._hdus_done += 1
            if self._hdus_done >= self.n_hdus:
                self.end_offset = self._pos
        return len(self.buf) >= self.end_offset

    def result(self) -> bytes:
        assert self.end_offset is not None
        return bytes(self.buf[: self.end_offset])


@dataclass
class RawCutout:
    """Native-pixel cutout of one exposure, as delivered by the archive."""

    sci: np.ndarray          # IMAGE - ZODI, MJy/sr, float32
    variance: np.ndarray     # MJy^2/sr^2, float32
    flags: np.ndarray        # int32 bitmask
    header: fits.Header      # IMAGE header (WCS, times, PSF, calibration) without HISTORY
    zodi_median: float       # median modelled zodiacal light that was subtracted
    calib_id: str            # spectral-WCS calibration identifier
    flag_bits: dict[str, int]
    flag_descriptions: dict[str, str]

    @property
    def nbytes(self) -> int:
        return self.sci.nbytes + self.variance.nbytes + self.flags.nbytes


def cutout_url(url: str, ra: float, dec: float, size_deg: float) -> str:
    return f"{url}?center={ra:.6f},{dec:.6f}d&size={size_deg:.5f}"


def _flag_definitions(flag_header: fits.Header) -> tuple[dict[str, int], dict[str, str]]:
    bits: dict[str, int] = {}
    descriptions: dict[str, str] = {}
    for key in flag_header:
        if key.startswith("MSKN"):
            plane = key[4:]
            name = str(flag_header[key]).strip()
            mask = int(flag_header.get(f"MSKM{plane}", 1 << int(plane)))
            bits[name] = mask.bit_length() - 1
            descriptions[name] = str(flag_header.get(f"MSKD{plane}", "")).strip()
    if not bits:
        bits = dict(config.DEFAULT_FLAG_BITS)
        descriptions = dict(config.DEFAULT_FLAG_DESCRIPTIONS)
    return bits, descriptions


def parse_cutout(data: bytes) -> RawCutout:
    """Turn the leading HDUs of a cutout file into arrays."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hdul = fits.HDUList.fromstring(data)
        names = {h.name for h in hdul}
        missing = {"IMAGE", "FLAGS", "VARIANCE", "ZODI"} - names
        if missing:
            raise CutoutUnavailable(f"cutout is missing extensions: {sorted(missing)}")
        image = np.asarray(hdul["IMAGE"].data, dtype=np.float32)
        zodi = np.asarray(hdul["ZODI"].data, dtype=np.float32)
        variance = np.asarray(hdul["VARIANCE"].data, dtype=np.float32)
        flags = np.asarray(hdul["FLAGS"].data, dtype=np.int32)
        full_header = hdul["IMAGE"].header
        flag_bits, flag_desc = _flag_definitions(hdul["FLAGS"].header)
    if image.ndim != 2 or image.size == 0:
        raise CutoutUnavailable("empty cutout")
    calib_id = ""
    for line in full_header.get("HISTORY", []):
        match = _CALIB_RE.search(str(line))
        if match:
            calib_id = match.group(1)
    header = fits.Header([card for card in full_header.cards if card.keyword not in _DROP_CARDS])
    finite_zodi = zodi[np.isfinite(zodi)]
    return RawCutout(
        sci=image - zodi,
        variance=variance,
        flags=flags,
        header=header,
        zodi_median=float(np.median(finite_zodi)) if finite_zodi.size else 0.0,
        calib_id=calib_id,
        flag_bits=flag_bits,
        flag_descriptions=flag_desc,
    )


class ArchiveBusy(Exception):
    """The archive answered 'Server busy, retry later' (HTTP 503/429)."""


class Throttle:
    """Shared politeness control for the archive.

    IRSA answers HTTP 503 "Server busy" on a fraction of cutout requests even
    at low request rates (measured: about one in five, sustaining roughly four
    successful cutouts per second). When that happens every worker pauses
    briefly, backing off further if it keeps happening, instead of hammering
    a shared public service.
    """

    def __init__(self) -> None:
        self._until = 0.0
        self._streak = 0
        self.busy_count = 0

    async def wait(self) -> None:
        delay = self._until - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay + random.uniform(0.0, 0.25))

    def busy(self) -> None:
        self.busy_count += 1
        self._streak = min(self._streak + 1, 5)
        pause = min(4.0, 0.35 * (2 ** (self._streak - 1)))
        self._until = max(self._until, time.monotonic() + pause)

    def ok(self) -> None:
        self._streak = 0


async def _download_prefix(client: httpx.AsyncClient, url: str) -> bytes:
    """Stream a cutout and stop once the science extensions have arrived."""
    reader = FitsPrefixReader()
    async with client.stream("GET", url, timeout=config.FETCH_TIMEOUT_S) as resp:
        if resp.status_code in (429, 503):
            raise ArchiveBusy(f"HTTP {resp.status_code}")
        if resp.status_code in (500, 502, 504):
            raise httpx.HTTPStatusError(f"HTTP {resp.status_code}", request=resp.request, response=resp)
        if resp.status_code != 200:
            raise CutoutUnavailable(f"HTTP {resp.status_code}")
        if "fits" not in resp.headers.get("content-type", "").lower():
            raise CutoutUnavailable("archive did not return a FITS file")
        async for chunk in resp.aiter_bytes(32768):
            if reader.feed(chunk):
                break
    if reader.end_offset is None or len(reader.buf) < reader.end_offset:
        raise httpx.ReadError("cutout stream ended early")
    return reader.result()


async def fetch_cutout(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    throttle: Throttle,
    url: str,
    ra: float,
    dec: float,
    size_deg: float,
) -> RawCutout:
    """Fetch and parse one cutout, waiting out 'server busy' answers and retrying network errors."""
    target = cutout_url(url, ra, dec, size_deg)
    last_err: Exception | None = None
    busy = errors = 0
    while busy < config.FETCH_BUSY_RETRIES and errors < config.FETCH_RETRIES:
        await throttle.wait()
        try:
            async with semaphore:
                data = await _download_prefix(client, target)
            throttle.ok()
            return await asyncio.to_thread(parse_cutout, data)
        except CutoutUnavailable:
            raise
        except ArchiveBusy as err:
            busy += 1
            last_err = err
            throttle.busy()
        except (httpx.HTTPError, OSError, ValueError) as err:
            errors += 1
            last_err = err
            await asyncio.sleep(min(6.0, 0.5 * (2 ** errors)) + random.uniform(0.0, 0.4))
    raise CutoutUnavailable(f"gave up after {busy} busy answers and {errors} errors: {last_err!r}")


async def fetch_full_cutout(client: httpx.AsyncClient, throttle: Throttle, url: str, ra: float, dec: float) -> bytes:
    """Download a complete (small) cutout file, including the WCS-WAVE table."""
    target = cutout_url(url, ra, dec, 0.01)
    last_err: Exception | None = None
    for attempt in range(config.FETCH_BUSY_RETRIES):
        await throttle.wait()
        try:
            resp = await client.get(target, timeout=config.FETCH_TIMEOUT_S * 2)
            if resp.status_code == 200:
                throttle.ok()
                return resp.content
            if resp.status_code in (429, 503):
                throttle.busy()
            last_err = RuntimeError(f"HTTP {resp.status_code}")
        except httpx.HTTPError as err:
            last_err = err
            await asyncio.sleep(min(6.0, 0.5 * (2 ** attempt)) + random.uniform(0.0, 0.4))
    raise CutoutUnavailable(f"could not download calibration cutout: {last_err!r}")
