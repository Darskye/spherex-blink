"""What is already known in a field: names and types from published catalogues.

Three sources, each fetched once per field and cached:

* SIMBAD (CDS; the Harvard mirror is tried first because the Strasbourg service
  was unavailable while this was built): named objects with a type, such as
  "eps Ind B, Brown Dwarf". Positions are for epoch J2000 with proper motions.
* 2MASS Point Source Catalog (IRSA): near-infrared sources, positions from about 1998-2001.
* AllWISE Source Catalog (IRSA): mid-infrared sources, positions from about 2010.

Gaia DR3 is handled in ``gaia.py``. Together these let the viewer say whether a
dot is catalogued, and under what name, or whether nothing is known there.

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

log = logging.getLogger("spx.catalogs")

SIMBAD_TAP_URLS = (
    "https://simbad.harvard.edu/simbad/sim-tap/sync",
    "https://simbad.cds.unistra.fr/simbad/sim-tap/sync",
)
IRSA_TAP_URL = "https://irsa.ipac.caltech.edu/TAP/sync"
SOURCES = ("simbad", "twomass", "allwise")
CREDITS = {
    "simbad": "SIMBAD, CDS Strasbourg",
    "twomass": "2MASS Point Source Catalog, NASA/IPAC IRSA",
    "allwise": "AllWISE Source Catalog, NASA/IPAC IRSA",
}
CACHE_DAYS = 30
_CACHE_TAG = {"allwise": "-v2"}      # bump when a catalogue's columns change
_in_flight: dict[str, asyncio.Task] = {}


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _field_radius_deg(fov_arcmin: float) -> float:
    """Half the field's diagonal, with a little margin."""
    return (fov_arcmin / 2.0 * math.sqrt(2.0) + 0.5) / 60.0


async def _simbad(client: httpx.AsyncClient, ra: float, dec: float, fov_arcmin: float) -> list[dict]:
    # SIMBAD gives positions for epoch 2000. The fastest star has moved about
    # 4.5 arcminutes since then, so search that much wider and let the viewer
    # move each object to the date of the frame.
    radius = _field_radius_deg(fov_arcmin) + 5.0 / 60.0
    query = (
        "SELECT TOP 400 b.main_id, b.ra, b.dec, b.pmra, b.pmdec, b.otype, o.description, b.sp_type "
        "FROM basic AS b LEFT JOIN otypedef AS o ON o.otype = b.otype "
        f"WHERE CONTAINS(POINT('ICRS', b.ra, b.dec), CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {radius:.5f})) = 1"
    )
    last: Exception | None = None
    for url in SIMBAD_TAP_URLS:
        try:
            resp = await client.post(
                url, data={"request": "doQuery", "lang": "adql", "format": "json", "query": query}, timeout=30.0
            )
            if resp.status_code != 200 or "json" not in resp.headers.get("content-type", ""):
                last = RuntimeError(f"HTTP {resp.status_code} from {url}")
                continue
            payload = resp.json()
            names = [m["name"] for m in payload["metadata"]]
            objects = []
            for row in payload["data"]:
                rec = dict(zip(names, row))
                o_ra, o_dec = _number(rec.get("ra")), _number(rec.get("dec"))
                if o_ra is None or o_dec is None or not rec.get("main_id"):
                    continue
                objects.append({
                    "name": str(rec["main_id"]).strip(),
                    "ra": o_ra, "dec": o_dec,
                    "pmra": _number(rec.get("pmra")), "pmdec": _number(rec.get("pmdec")),
                    "otype": (rec.get("otype") or "").strip(),
                    "kind": (rec.get("description") or "").strip(),
                    "sptype": (rec.get("sp_type") or "").strip() or None,
                    "epoch": 2000.0,
                })
            return objects
        except (httpx.HTTPError, ValueError, KeyError) as err:
            last = err
    raise RuntimeError(f"SIMBAD did not answer: {last!r}")


async def _irsa(client: httpx.AsyncClient, query: str) -> list[dict]:
    last: object = None
    for attempt in range(3):
        try:
            resp = await client.get(IRSA_TAP_URL, params={"QUERY": query, "FORMAT": "CSV"}, timeout=240.0)
            if resp.status_code == 200:
                return list(csv.DictReader(io.StringIO(resp.text)))
            last = f"HTTP {resp.status_code}"
        except httpx.HTTPError as err:
            last = err.__class__.__name__
        await asyncio.sleep(2.0 + 3 * attempt)
    raise RuntimeError(f"IRSA catalogue query failed: {last}")


async def _twomass(client: httpx.AsyncClient, ra: float, dec: float, fov_arcmin: float) -> list[dict]:
    radius = _field_radius_deg(fov_arcmin)
    rows = await _irsa(
        client,
        "SELECT designation, ra, dec, j_m, h_m, k_m, jdate FROM fp_psc "
        f"WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {radius:.5f})) = 1",
    )
    objects = []
    for row in rows:
        o_ra, o_dec = _number(row.get("ra")), _number(row.get("dec"))
        if o_ra is None or o_dec is None:
            continue
        jdate = _number(row.get("jdate"))
        objects.append({
            "name": f"2MASS J{row['designation'].strip()}",
            "ra": o_ra, "dec": o_dec,
            "j": _number(row.get("j_m")), "h": _number(row.get("h_m")), "k": _number(row.get("k_m")),
            "epoch": 2000.0 + (jdate - 2451545.0) / 365.25 if jdate else 1999.5,
        })
    return objects


async def _allwise(client: httpx.AsyncClient, ra: float, dec: float, fov_arcmin: float) -> list[dict]:
    radius = _field_radius_deg(fov_arcmin)
    rows = await _irsa(
        client,
        "SELECT designation, ra, dec, w1mpro, w2mpro, cc_flags FROM allwise_p3as_psd "
        f"WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {radius:.5f})) = 1",
    )
    objects = []
    for row in rows:
        o_ra, o_dec = _number(row.get("ra")), _number(row.get("dec"))
        if o_ra is None or o_dec is None:
            continue
        objects.append({
            "name": f"WISEA {row['designation'].strip()}",
            "ra": o_ra, "dec": o_dec,
            "w1": _number(row.get("w1mpro")), "w2": _number(row.get("w2mpro")),
            # Contamination flags per band ("0000" is clean; letters mark halos, spikes, ghosts, persistence).
            "cc": (row.get("cc_flags") or "").strip() or None,
            "epoch": 2010.5,
        })
    return objects


_FETCHERS = {"simbad": _simbad, "twomass": _twomass, "allwise": _allwise}


async def _load(client: httpx.AsyncClient, source: str, ra: float, dec: float, fov_arcmin: float) -> dict:
    path = config.CACHE_DIR / "catalogues" / f"{source}{_CACHE_TAG.get(source, '')}_{ra:.4f}_{dec:+.4f}_f{fov_arcmin:g}.json"
    try:
        if time.time() - path.stat().st_mtime < CACHE_DAYS * 86400:
            return json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        pass
    objects = await _FETCHERS[source](client, ra, dec, fov_arcmin)
    result = {"source": source, "credit": CREDITS[source], "objects": objects}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result), "utf-8")
    tmp.replace(path)
    return result


async def lookup(client: httpx.AsyncClient, source: str, ra: float, dec: float, fov_arcmin: float) -> dict:
    """Catalogued objects covering a field. Concurrent calls for the same field share one query."""
    if source not in _FETCHERS:
        raise ValueError(f"unknown catalogue {source!r}")
    key = f"{source}_{ra:.4f}_{dec:+.4f}_{fov_arcmin:g}"
    task = _in_flight.get(key)
    if task is None:
        task = asyncio.create_task(_load(client, source, ra, dec, fov_arcmin))
        _in_flight[key] = task
        task.add_done_callback(lambda _t: _in_flight.pop(key, None))
    return await asyncio.shield(task)
