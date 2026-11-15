"""Gaia DR3 stars around a position, for the known-object overlay.

Gaia positions are for epoch 2016.0; the browser moves each star along its
proper motion to the date of the frame being shown, so a known high-motion star
is marked where it should be, and anything that moves without a marker stands out.

Two services are tried: the ESA Gaia Archive (fast, but it rate-limits) and the
copy of Gaia DR3 hosted by NASA/IPAC IRSA (slower, same catalogue).

Developed with AI assistance (Claude, Anthropic). See README "Use of AI".
"""
from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import math
import time

import httpx

from . import config

log = logging.getLogger("spx.gaia")
IRSA_TAP_URL = "https://irsa.ipac.caltech.edu/TAP/sync"
COLUMNS = "source_id, ra, dec, pmra, pmdec, parallax, phot_g_mean_mag, ref_epoch, phot_variable_flag"


def _cache_path(ra: float, dec: float, radius_deg: float):
    return config.CACHE_DIR / "gaia" / f"{ra:.4f}_{dec:+.4f}_r{radius_deg:.4f}_v2.json"


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _star(rec: dict) -> dict | None:
    ra, dec = _number(rec.get("ra")), _number(rec.get("dec"))
    if ra is None or dec is None:
        return None
    return {
        "id": str(rec.get("source_id")),
        "ra": ra, "dec": dec,
        "pmra": _number(rec.get("pmra")), "pmdec": _number(rec.get("pmdec")),
        "plx": _number(rec.get("parallax")), "g": _number(rec.get("phot_g_mean_mag")),
        "epoch": _number(rec.get("ref_epoch")) or 2016.0,
        # Gaia DR3 itself found this star's brightness to vary.
        "var": str(rec.get("phot_variable_flag") or "").strip().upper() == "VARIABLE",
    }


async def _from_esa(client: httpx.AsyncClient, ra: float, dec: float, radius_deg: float) -> list[dict]:
    query = (
        f"SELECT TOP {config.GAIA_MAX_ROWS} {COLUMNS} FROM gaiadr3.gaia_source "
        f"WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {radius_deg:.5f})) "
        f"AND phot_g_mean_mag < {config.GAIA_MAG_LIMIT} ORDER BY phot_g_mean_mag"
    )
    resp = await client.get(
        config.GAIA_TAP_URL,
        params={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json", "QUERY": query},
        timeout=30.0,
    )
    resp.raise_for_status()
    payload = resp.json()
    names = [m["name"] for m in payload["metadata"]]
    return [dict(zip(names, row)) for row in payload["data"]]


async def _from_irsa(client: httpx.AsyncClient, ra: float, dec: float, radius_deg: float) -> list[dict]:
    query = (
        f"SELECT TOP {config.GAIA_MAX_ROWS} {COLUMNS} FROM gaia_dr3_source "
        f"WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {radius_deg:.5f})) = 1 "
        f"AND phot_g_mean_mag < {config.GAIA_MAG_LIMIT} ORDER BY phot_g_mean_mag"
    )
    resp = await client.get(IRSA_TAP_URL, params={"QUERY": query, "FORMAT": "CSV"}, timeout=90.0)
    resp.raise_for_status()
    return list(csv.DictReader(io.StringIO(resp.text)))


_in_flight: dict[str, asyncio.Task] = {}


async def cone(client: httpx.AsyncClient, ra: float, dec: float, fov_arcmin: float) -> dict:
    """Gaia stars covering a field. Concurrent calls for the same field share one query.

    The query runs in its own task, so a caller that gives up waiting (see
    main.api_spectrum) does not cancel it for the others.
    """
    key = f"{ra:.4f}_{dec:+.4f}_{fov_arcmin:g}"
    task = _in_flight.get(key)
    if task is None:
        task = asyncio.create_task(_cone(client, ra, dec, fov_arcmin))
        _in_flight[key] = task
        task.add_done_callback(lambda _t: _in_flight.pop(key, None))
    return await asyncio.shield(task)


async def _cone(client: httpx.AsyncClient, ra: float, dec: float, fov_arcmin: float) -> dict:
    # Cover the square's diagonal, plus margin for stars that have moved into the
    # field since 2016 (the fastest star, Barnard's, moves about 104 arcsec per decade).
    radius_deg = (fov_arcmin / 2.0 * math.sqrt(2.0) + 2.0) / 60.0
    path = _cache_path(ra, dec, radius_deg)
    try:
        if time.time() - path.stat().st_mtime < 30 * 86400:
            return json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        pass

    records, source, last_err = None, None, None
    attempts = (
        ("Gaia DR3 (ESA Gaia Archive)", _from_esa),
        ("Gaia DR3 (hosted by NASA/IPAC IRSA)", _from_irsa),
        ("Gaia DR3 (ESA Gaia Archive)", _from_esa),
    )
    for index, (label, fetcher) in enumerate(attempts):
        try:
            records = await fetcher(client, ra, dec, radius_deg)
            source = label
            break
        except (httpx.HTTPError, ValueError, KeyError) as err:
            last_err = err
            log.info("Gaia source %s failed (%s)", label, err.__class__.__name__)
            if index == 1:
                await asyncio.sleep(2.0)
    if records is None:
        raise RuntimeError(f"no Gaia service answered: {last_err!r}")

    stars = [s for s in (_star(rec) for rec in records) if s is not None]
    result = {"source": source, "radius_deg": radius_deg, "mag_limit": config.GAIA_MAG_LIMIT, "stars": stars}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result), "utf-8")
    return result
