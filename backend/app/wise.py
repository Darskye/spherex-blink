"""A "then" frame from NASA's WISE mission (2010), aligned to the SPHEREx frames.

SPHEREx has watched the sky for about a year, so even the fastest stars have
moved only a pixel or two. WISE mapped the same sky at 3.4 and 4.6 micron in
2010; blinking its image against today's SPHEREx frames gives a 15-year baseline
on which nearby objects move by arcminutes.

Images are AllWISE Atlas coadds from IRSA, found with the same Simple Image
Access service used for SPHEREx and cut out with IRSA's cutout service.

Developed with AI assistance (Claude, Anthropic). See README "Use of AI".
"""
from __future__ import annotations

import asyncio
import gzip
import io
import json
import logging
import math
import warnings

import httpx
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from . import config, discovery, process

log = logging.getLogger("spx.wise")

COLLECTION = "wise_allwise"
BANDS = {"W1": 3.368, "W2": 4.618}          # micron
_in_flight: dict[str, asyncio.Task] = {}


def _cache_path(ra: float, dec: float, fov_arcmin: float, band: str):
    return config.CACHE_DIR / "wise" / f"{ra:.5f}_{dec:+.5f}_f{fov_arcmin:g}_{band}.npz"


def _parse_sia(content: bytes, band: str, ra: float, dec: float) -> list[str]:
    """Science images of the tiles covering the position, nearest tile centre first."""
    from astropy.io import votable

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = votable.parse_single_table(io.BytesIO(content)).to_table(use_names_over_ids=True)
    found: dict[str, float] = {}
    for row in table:
        url = str(row["access_url"]).strip()
        if str(row["energy_bandpassname"]).strip() != band or not url.endswith("-int-3.fits"):
            continue
        d_ra = ((float(row["s_ra"]) - ra + 540.0) % 360.0 - 180.0) * math.cos(math.radians(dec))
        found[url] = math.hypot(d_ra, float(row["s_dec"]) - dec)
    return sorted(found, key=found.get)


def _resample(data: bytes, grid: process.Grid) -> tuple[np.ndarray, float]:
    """WISE cutout (FITS bytes) -> background-subtracted image on the viewer grid, plus its MJD."""
    from scipy import ndimage

    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fits.open(io.BytesIO(data)) as hdul:
            header = hdul[0].header
            image = np.asarray(hdul[0].data, dtype=np.float32)
        wcs = WCS(header)
        ra, dec = grid.world
        x, y = wcs.all_world2pix(ra, dec, 0)
    background, _ = process.robust_background(image, np.isfinite(image))
    image = image - background
    # WISE pixels (1.375 arcsec) are finer than the viewer grid for larger fields,
    # so smooth a little first to avoid aliasing when sampling.
    ratio = grid.scale_arcsec / 1.375
    if ratio > 1.2:
        image = ndimage.gaussian_filter(np.nan_to_num(image, nan=0.0), sigma=0.42 * ratio)
    out = ndimage.map_coordinates(image, np.array([y, x]), order=1, mode="constant", cval=np.nan, prefilter=False)
    mjd = 55400.0                                              # mid-2010 if the header has no date
    midobs = header.get("MIDOBS")
    if midobs:
        try:
            from astropy.time import Time

            mjd = float(Time(str(midobs), format="isot", scale="utc").mjd)
        except ValueError:
            pass
    return out.astype(np.float32), mjd


async def _load(client: httpx.AsyncClient, ra: float, dec: float, fov_arcmin: float, band: str) -> dict:
    path = _cache_path(ra, dec, fov_arcmin, band)
    if path.exists():
        try:
            with np.load(path) as z:
                return {"image": z["image"], "meta": json.loads(str(z["meta"]))}
        except Exception as err:
            log.warning("dropping unreadable WISE cache %s: %s", path, err)
            path.unlink(missing_ok=True)

    # Ask for every tile that touches the field, not just the one under its centre.
    reach_deg = fov_arcmin / 60.0 * 0.75
    resp = await client.get(
        config.IRSA_SIA_URL,
        params={"COLLECTION": COLLECTION, "POS": f"circle {ra:.6f} {dec:.6f} {reach_deg:.5f}"},
        timeout=config.FETCH_TIMEOUT_S,
    )
    resp.raise_for_status()
    urls = await asyncio.to_thread(_parse_sia, resp.content, band, ra, dec)
    if not urls:
        raise LookupError("no WISE image covers this position")

    # A field near the edge of a WISE tile is only partly covered by it, so the
    # neighbouring tiles (which overlap) fill in what is missing.
    grid = process.Grid(ra, dec, fov_arcmin)
    size_deg = fov_arcmin / 60.0 * 1.5
    image, mjd, last_err = None, None, None
    for url in urls[:4]:
        data = None
        target = f"{url}?center={ra:.6f},{dec:.6f}&size={size_deg:.5f}&gzip=false"
        for attempt in range(5):
            try:
                cut = await client.get(target, timeout=120.0)
                if cut.status_code == 200:
                    data = cut.content
                    break
                last_err = RuntimeError(f"HTTP {cut.status_code}")
                if cut.status_code not in (429, 500, 502, 503, 504):
                    break
            except httpx.HTTPError as err:
                last_err = err
            await asyncio.sleep(1.0 + attempt)
        if data is None:
            continue
        try:
            tile, tile_mjd = await asyncio.to_thread(_resample, data, grid)
        except Exception as err:                       # an unreadable tile must not sink the others
            last_err = err
            continue
        if image is None:
            image, mjd = tile, tile_mjd
        else:
            image = np.where(np.isfinite(image), image, tile)
        if np.isfinite(image).mean() > 0.995:
            break
    if image is None:
        raise RuntimeError(f"WISE cutout failed: {last_err!r}")
    meta = {
        "source": "WISE", "band": band, "wl": BANDS[band],
        "mjd": mjd, "date": discovery.mjd_to_iso_date(mjd), "year": discovery.mjd_to_decimal_year(mjd),
        "credit": "AllWISE Atlas image, NASA/IPAC Infrared Science Archive",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, image=image, meta=np.array(json.dumps(meta)))
    tmp.replace(path)
    return {"image": image, "meta": meta}


async def frame(client: httpx.AsyncClient, ra: float, dec: float, fov_arcmin: float, band: str) -> dict:
    """WISE image on the viewer grid for (ra, dec, fov). Concurrent requests share one download."""
    key = f"{ra:.5f}_{dec:+.5f}_{fov_arcmin:g}_{band}"
    task = _in_flight.get(key)
    if task is None:
        task = asyncio.create_task(_load(client, ra, dec, fov_arcmin, band))
        _in_flight[key] = task
        task.add_done_callback(lambda _t: _in_flight.pop(key, None))
    return await asyncio.shield(task)


def pack(result: dict, ra: float, dec: float, fov_arcmin: float) -> bytes:
    """Same binary layout as the SPHEREx frames, with one frame."""
    image = result["image"]
    meta = result["meta"]
    n = image.shape[0]
    finite = image[np.isfinite(image)]
    noise = float(1.4826 * np.median(np.abs(finite - np.median(finite)))) if finite.size else 0.0
    header = {
        "n": n, "nframes": 1, "ra": ra, "dec": dec, "fov_arcmin": fov_arcmin,
        "scale_arcsec": fov_arcmin * 60.0 / n,
        "band": meta["band"], "kind": "wise", "mask": False, "complete": True, "n_exposures": 1,
        "unit": "DN (background subtracted)", "credit": meta["credit"],
        "frames": [{
            "epoch": -1, "detector": None, "source": "WISE", "wise_band": meta["band"],
            "mjd": meta["mjd"], "mjd_min": meta["mjd"], "mjd_max": meta["mjd"],
            "date": meta["date"], "year": meta["year"], "nexp": 1,
            "wl": meta["wl"], "wl_min": meta["wl"], "wl_max": meta["wl"], "noise": noise,
        }],
    }
    return process.pack_frames(header, image[None, :, :].astype("<f4"), np.zeros((1, n, n), dtype=np.uint8))
