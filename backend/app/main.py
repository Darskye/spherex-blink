"""HTTP API and static site for the SPHEREx blink viewer.

Developed with AI assistance (Claude, Anthropic). See README "Use of AI".
"""
from __future__ import annotations

import asyncio
import logging
import math
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import catalogs, config, finder, gaia, payloads, photometry, resolver, wise
from .stack import StackJob, StackManager, snap_fov

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("spx.api")

manager = StackManager()

EXAMPLES = [
    {
        "id": "epsindb", "name": "Epsilon Indi B", "query": "eps Ind B", "band": "colour", "fov": 6, "mode": "blink",
        "tag": "Brown dwarf pair", "featured": True, "wise": False,
        "blurb": "A pair of brown dwarfs 12 light-years away.",
        "look_for": "It is the red dot at the centre (red means cold), creeping south-east from pass to pass.",
    },
    {
        "id": "barnard", "name": "Barnard's Star", "query": "Barnard's Star", "band": "colour", "fov": 6, "mode": "blink",
        "tag": "Fastest-moving star", "wise": False,
        "blurb": "The fastest-moving star in the sky, about 10.4 arcseconds per year.",
        "look_for": "The bright star at the centre hops north between survey passes.",
    },
    {
        "id": "wise0855", "name": "WISE 0855-0714", "query": "WISE 0855", "band": "colour", "fov": 6, "mode": "blink",
        "tag": "Coldest known brown dwarf", "wise": False,
        "blurb": "The coldest known brown dwarf, moving about 8 arcseconds per year.",
        "look_for": "A faint red dot at the centre, drifting west into a blue background star.",
        # The coordinates printed in the challenge statement are its 2010 position;
        # cache that field too, because people will paste them.
        "also_warm": ["08h55m10.831s -07d14m42.53s"],
    },
    {
        "id": "rwaur", "name": "RW Aurigae", "query": "RW Aur", "band": "colour", "fov": 6, "mode": "blink",
        "tag": "Young variable star", "wise": False,
        "blurb": "A young variable star that dims when dust crosses in front of it.",
        "look_for": "Its brightness and colour change between passes; the light curve below measures how much.",
    },
    {
        "id": "rdra", "name": "R Draconis", "query": "R Dra", "band": "colour", "fov": 6, "mode": "blink",
        "tag": "Pulsating red giant", "wise": False,
        "blurb": "A pulsating red giant with a period of about 245 days.",
        "look_for": "It lies near the ecliptic pole, so SPHEREx sees it almost continuously. Watch it swell and fade.",
    },
]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    await manager.start()
    yield
    await manager.stop()


app = FastAPI(title="SPHEREx Blink", version="1.0", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=2000)
# Read-only public API: any site may embed or query it.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])


def _job(ra: float, dec: float, fov: float, band: str | None = None) -> StackJob:
    if not (math.isfinite(ra) and math.isfinite(dec)) or not (-90.0 <= dec <= 90.0) or not (0.0 <= ra <= 360.0):
        raise HTTPException(400, "ra must be 0-360 and dec -90 to 90 (decimal degrees)")
    priority = int(band) if band and band.isdigit() and int(band) in config.DETECTORS else None
    return manager.get_job(ra, dec, fov, priority)


def _parse_band(band: str) -> int | None:
    if band.lower() == "all":
        return None
    if band.isdigit() and int(band) in config.DETECTORS:
        return int(band)
    raise HTTPException(400, "band must be 1-6 or 'all'")


@app.get("/api/config")
async def get_config():
    return {
        "fov_choices_arcmin": config.FOV_CHOICES_ARCMIN,
        "default_fov_arcmin": config.DEFAULT_FOV_ARCMIN,
        "detectors": {str(d): {"min_um": lo, "max_um": hi} for d, (lo, hi) in config.DETECTOR_RANGES_UM.items()},
        "grid_n": config.GRID_N,
        "max_exposures_per_band": config.MAX_EXPOSURES_PER_BAND,
        "bad_flags": config.BAD_FLAG_NAMES,
        "flag_descriptions": config.DEFAULT_FLAG_DESCRIPTIONS,
        "view_epoch": config.VIEW_EPOCH_YEAR,
        "examples": EXAMPLES,
    }


@app.get("/api/resolve")
async def api_resolve(q: str = Query(..., min_length=1, max_length=120)):
    result = await resolver.resolve(manager.client, q)
    if result is None:
        raise HTTPException(404, f"Could not understand '{q}'. Enter RA and Dec (e.g. '269.45 4.77' or '17h57m48s +04d41m36s') or an object name.")
    return result


@app.get("/api/status")
async def api_status(ra: float, dec: float, fov: float = config.DEFAULT_FOV_ARCMIN, band: str | None = None):
    """Start (or check on) loading the SPHEREx data for a position."""
    return _job(ra, dec, fov, band).snapshot()


@app.get("/api/frames")
async def api_frames(
    ra: float, dec: float, fov: float = config.DEFAULT_FOV_ARCMIN,
    band: str = "2", kind: str = "epochs", mask: int = 1, partial: int = 0,
):
    """Aligned frames as a compact binary payload (see process.pack_frames).

    With partial=1 the frames are built from whatever has been downloaded so far,
    so the browser can show a first image within seconds and refine it.
    """
    if kind not in ("epochs", "exposures"):
        raise HTTPException(400, "kind must be 'epochs' or 'exposures'")
    colour = band.lower() in ("colour", "color")
    detector = None if colour else _parse_band(band)
    if detector is None and kind == "exposures":
        raise HTTPException(400, "choose a single band to view every exposure")
    job = _job(ra, dec, fov, band)

    def band_ready() -> bool:
        return job.finished if detector is None else detector in job.bands_ready

    def has_data() -> bool:
        return any(detector is None or ed.exp.detector == detector for ed in list(job.data.values()))

    try:
        await job.wait_for((lambda: band_ready() or has_data()) if partial else band_ready)
    except TimeoutError:
        raise HTTPException(504, "Timed out waiting for the archive")
    except RuntimeError as err:
        raise HTTPException(502, str(err))
    if job.stage == "error":
        raise HTTPException(502, job.error or "load failed")

    complete = band_ready()
    exposures = job.band_exposures(detector)          # snapshot taken on the event loop
    name = "colour" if colour else ("all" if detector is None else str(detector))
    cached = await payloads.frames(job, name, kind, bool(mask), exposures, complete)
    cache_control = "public, max-age=300" if complete else "no-store"
    return Response(content=cached, media_type="application/octet-stream", headers={"Cache-Control": cache_control})


@app.get("/api/spectrum")
async def api_spectrum(
    ra: float, dec: float, fov: float = config.DEFAULT_FOV_ARCMIN,
    sra: float | None = None, sdec: float | None = None, partial: int = 0,
    pmra: float | None = None, pmdec: float | None = None, epoch: float | None = None,
):
    """Quick-look spectrum and light curve at (sra, sdec), default the field centre.

    For a moving target give its proper motion (pmra, pmdec in mas/yr) and the
    decimal year at which (sra, sdec) applies; the measurement then follows it.
    """
    job = _job(ra, dec, fov)
    try:
        await job.wait_for((lambda: job.finished or len(job.data) > 0) if partial else (lambda: job.finished))
    except TimeoutError:
        raise HTTPException(504, "Timed out waiting for the archive")
    except RuntimeError as err:
        raise HTTPException(502, str(err))
    if job.stage == "error":
        raise HTTPException(502, job.error or "load failed")
    sra = job.ra if sra is None else sra
    sdec = job.dec if sdec is None else sdec
    sep = math.hypot((sra - job.ra) * math.cos(math.radians(job.dec)), sdec - job.dec) * 60.0
    if sep > job.fov_arcmin:
        raise HTTPException(400, "position is outside the loaded field")
    motion = None
    if pmra is not None and pmdec is not None and (pmra or pmdec):
        if abs(pmra) > 20000 or abs(pmdec) > 20000:
            raise HTTPException(400, "proper motion is out of range (mas/yr)")
        motion = {"pmra": pmra, "pmdec": pmdec, "epoch": epoch if epoch is not None else config.VIEW_EPOCH_YEAR}
    # Gaia stars let the measurement say when a neighbour is blended in. Optional:
    # if the catalogue is not to hand within a few seconds the spectrum is returned
    # without the check (blend_checked = false) and the browser asks again later.
    stars = None
    try:
        catalogue = await asyncio.wait_for(gaia.cone(manager.client, round(job.ra, 4), round(job.dec, 4), job.fov_arcmin), timeout=3.0)
        stars = catalogue["stars"]
    except Exception as err:
        log.info("no Gaia stars for blend check: %s", err.__class__.__name__)
    complete = job.finished
    exposures = job.band_exposures(None)              # snapshot taken on the event loop
    if motion is None and stars:
        # A click on a fast-moving catalogued star: follow the star, not the spot it was clicked at.
        followed = photometry.track_for_click(stars, sra, sdec, [ed.mjd for ed in exposures])
        if followed:
            sra, sdec, motion = followed
    result = await asyncio.to_thread(photometry.extract, job, sra, sdec, exposures, motion, stars)
    result["blend_checked"] = stars is not None
    result["complete"] = complete
    result["n_exposures"] = len(exposures)
    return JSONResponse(_clean(result))


@app.get("/api/wise")
async def api_wise(ra: float, dec: float, fov: float = config.DEFAULT_FOV_ARCMIN, band: str = "W2"):
    """The same field as NASA's WISE mission saw it in 2010, on the viewer grid."""
    band = band.upper()
    if band not in wise.BANDS:
        raise HTTPException(400, "band must be W1 or W2")
    if not (0.0 <= ra <= 360.0 and -90.0 <= dec <= 90.0):
        raise HTTPException(400, "ra must be 0-360 and dec -90 to 90 (decimal degrees)")
    ra, dec, fov = round(ra % 360.0, 5), round(dec, 5), snap_fov(fov)
    try:
        result = await wise.frame(manager.client, ra, dec, fov, band)
    except LookupError as err:
        raise HTTPException(404, str(err))
    except Exception as err:
        log.warning("WISE frame failed: %r", err)
        raise HTTPException(502, "The archive did not return a WISE image just now.")
    return Response(content=wise.pack(result, ra, dec, fov), media_type="application/octet-stream",
                    headers={"Cache-Control": "public, max-age=300"})


@app.get("/api/catalogue")
async def api_catalogue(ra: float, dec: float, fov: float = config.DEFAULT_FOV_ARCMIN, source: str = "simbad"):
    """Named, catalogued objects in the field from SIMBAD, 2MASS or AllWISE."""
    if source not in catalogs.SOURCES:
        raise HTTPException(400, f"source must be one of {', '.join(catalogs.SOURCES)}")
    if not (0.0 <= ra <= 360.0 and -90.0 <= dec <= 90.0):
        raise HTTPException(400, "ra must be 0-360 and dec -90 to 90 (decimal degrees)")
    try:
        return await catalogs.lookup(manager.client, source, round(ra % 360.0, 4), round(dec, 4), snap_fov(fov))
    except Exception as err:
        log.warning("catalogue %s failed: %r", source, err)
        raise HTTPException(502, f"The {catalogs.CREDITS[source]} service did not answer just now.")


@app.get("/api/candidates")
async def api_candidates(ra: float, dec: float, fov: float = config.DEFAULT_FOV_ARCMIN):
    """Search the field for sources that moved, changed brightness, are in no
    catalogue, or were catalogued in 2010 and are not there now (see finder.py).

    Waits for the field to finish loading. The first search of a field can take
    a minute or more, because it also waits for four catalogue services.
    """
    job = _job(ra, dec, fov)
    try:
        await job.wait_for(lambda: job.finished)
    except TimeoutError:
        raise HTTPException(504, "Timed out waiting for the archive")
    except RuntimeError as err:
        raise HTTPException(502, str(err))
    if job.stage == "error":
        raise HTTPException(502, job.error or "load failed")
    if not job.data:
        raise HTTPException(404, "SPHEREx has no images of this position yet.")
    try:
        report = await finder.run(manager.client, job)
    except ValueError as err:
        raise HTTPException(422, f"This field cannot be searched yet: {err}")
    return JSONResponse(_clean(report))


@app.get("/api/gaia")
async def api_gaia(ra: float, dec: float, fov: float = config.DEFAULT_FOV_ARCMIN):
    try:
        return await gaia.cone(manager.client, round(ra, 4), round(dec, 4), min(max(fov, 1.0), 20.0))
    except Exception as err:
        log.warning("Gaia query failed: %s", err)
        raise HTTPException(502, "The Gaia archive did not answer; the overlay is unavailable right now.")


@app.get("/api/health")
async def api_health():
    return {"ok": True, "jobs": len(manager.jobs)}


def _clean(value):
    """Replace NaN/inf (not valid JSON) with None, recursively."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


class RevalidatedStatic(StaticFiles):
    """Static files that browsers re-check on every load, so a new deploy is seen at once."""

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


if config.FRONTEND_DIR.is_dir():
    app.mount("/", RevalidatedStatic(directory=config.FRONTEND_DIR, html=True), name="frontend")
