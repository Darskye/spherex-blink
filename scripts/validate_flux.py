"""Check the tool's brightness measurements against two independent catalogues.

For ordinary, isolated field stars around a target, compare the flux this tool
measures from SPHEREx with the flux published by 2MASS (J, H, Ks) and by WISE
(W1, W2) at the same wavelengths. Stars are steady, so the two should agree.

    python scripts/validate_flux.py "eps Ind B" "WISE 0855"

The field must already be cached (run prewarm.py or open it in the app first).

Developed with AI assistance (Claude, Anthropic). See README "Use of AI".
"""
from __future__ import annotations

import asyncio
import csv
import io
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app import gaia, photometry, resolver  # noqa: E402
from app.stack import StackManager  # noqa: E402

TAP = "https://irsa.ipac.caltech.edu/TAP/sync"
# name: (catalogue column, zero-point flux in Jy, SPHEREx wavelength window in micron)
# Zero points: Cohen et al. 2003 (2MASS) and Jarrett et al. 2011 (WISE).
FILTERS = {
    "J": ("j_m", 1594.0, (1.17, 1.33)),
    "H": ("h_m", 1024.0, (1.51, 1.80)),
    "Ks": ("k_m", 666.7, (2.00, 2.32)),
    "W1": ("w1mpro", 309.54, (3.05, 3.75)),
    "W2": ("w2mpro", 171.787, (4.25, 4.95)),
}
MATCH_ARCSEC = 2.0
ISOLATION_ARCSEC = 18.0        # no comparably bright neighbour within three SPHEREx pixels


async def tap(client, query: str) -> list[dict]:
    """Run a catalogue query at IRSA. The service is slow when busy, so be patient (and cache the answer)."""
    import hashlib
    import json

    import httpx

    from app import config

    cache = config.CACHE_DIR / "catalogues" / (hashlib.sha1(query.encode()).hexdigest()[:16] + ".json")
    if cache.exists():
        return json.loads(cache.read_text("utf-8"))
    last = None
    for attempt in range(4):
        try:
            resp = await client.get(TAP, params={"QUERY": query, "FORMAT": "CSV"}, timeout=300.0)
            if resp.status_code == 200:
                rows = list(csv.DictReader(io.StringIO(resp.text)))
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(rows), "utf-8")
                return rows
            last = f"HTTP {resp.status_code}"
        except httpx.HTTPError as err:
            last = err.__class__.__name__
        await asyncio.sleep(3.0 + 3 * attempt)
    raise RuntimeError(f"catalogue query failed: {last}")


def separation(ra1, dec1, ra2, dec2) -> float:
    d_ra = ((ra1 - ra2 + 540.0) % 360.0 - 180.0) * math.cos(math.radians(dec1))
    return math.hypot(d_ra, dec1 - dec2) * 3600.0


def nearest(rows: list[dict], ra: float, dec: float) -> dict | None:
    best, best_sep = None, MATCH_ARCSEC
    for row in rows:
        sep = separation(ra, dec, float(row["ra"]), float(row["dec"]))
        if sep < best_sep:
            best, best_sep = row, sep
    return best


async def field(manager: StackManager, name: str) -> list[dict]:
    target = await resolver.resolve(manager.client, name)
    job = manager.get_job(round(target["ra"], 5), round(target["dec"], 5), 6.0, None)
    await job.wait_for(lambda: job.finished, timeout=900)
    radius = 3.0 / 60.0
    cone = f"CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {job.ra:.5f}, {job.dec:.5f}, {radius:.4f})) = 1"
    stars = (await gaia.cone(manager.client, round(job.ra, 4), round(job.dec, 4), job.fov_arcmin))["stars"]
    twomass = await tap(manager.client, f"SELECT ra, dec, j_m, h_m, k_m, ph_qual FROM fp_psc WHERE {cone}")
    allwise = await tap(manager.client, f"SELECT ra, dec, w1mpro, w2mpro, cc_flags, ext_flg FROM allwise_p3as_psd WHERE {cone}")
    print(f"  {name}: {len(twomass)} 2MASS and {len(allwise)} AllWISE sources in the field", flush=True)
    exposures = job.band_exposures(None)
    rows = []
    for star in stars:
        if star["g"] is None or not (10.5 <= star["g"] <= 15.5):
            continue
        # Move the star to the middle of the SPHEREx data (Gaia positions are for 2016).
        ra, dec = resolver.propagate(star["ra"], star["dec"], star["pmra"] or 0.0, star["pmdec"] or 0.0, star["epoch"], 2026.0)
        if separation(ra, dec, job.ra, job.dec) > 150.0 or separation(ra, dec, job.ra, job.dec) < 20.0:
            continue                                   # inside the cutouts, and not the target itself
        crowded = any(
            other is not star and other["g"] is not None and other["g"] < star["g"] + 2.5
            and separation(ra, dec, other["ra"], other["dec"]) < ISOLATION_ARCSEC
            for other in stars
        )
        if crowded:
            continue
        points = [p for p in (photometry.measure(ed, job, ra, dec) for ed in exposures) if p and not p["flagged"]]
        if len(points) < 60:
            continue
        m2, mw = nearest(twomass, ra, dec), nearest(allwise, ra, dec)
        for band, (column, zero_point, (lo, hi)) in FILTERS.items():
            source = m2 if column.endswith("_m") else mw
            if source is None or source.get(column) in (None, "", "null"):
                continue
            if band in ("J", "H", "Ks") and source["ph_qual"]["JHKs".index(band[0]) if band != "Ks" else 2] != "A":
                continue
            if band in ("W1", "W2") and (source.get("cc_flags", "0000")[int(band[1]) - 1] != "0" or source.get("ext_flg") not in ("0", 0)):
                continue
            inside = [p["flux"] for p in points if lo <= p["wl"] <= hi]
            if len(inside) < 5:
                continue
            catalogue = zero_point * 1000.0 * 10 ** (-0.4 * float(source[column]))       # mJy
            measured = float(np.median(inside))
            if catalogue < 0.5:
                continue                                   # too faint for a clean comparison
            rows.append({"field": name, "g": star["g"], "band": band, "catalogue": catalogue, "measured": measured,
                         "ratio": measured / catalogue, "n": len(inside)})
    return rows


async def main() -> int:
    names = sys.argv[1:] or ["eps Ind B", "WISE 0855"]
    manager = StackManager()
    await manager.start()
    rows: list[dict] = []
    try:
        for name in names:
            got = await field(manager, name)
            stars = len({(r["field"], r["g"]) for r in got})
            print(f"{name}: {stars} isolated stars, {len(got)} star-filter comparisons")
            rows.extend(got)
    finally:
        await manager.stop()
    if not rows:
        print("no comparisons possible")
        return 1
    print(f"\n{'filter':7}{'n':>4}{'median ratio':>14}{'scatter':>10}   (tool flux / catalogue flux; 1.00 is perfect)")
    for band in FILTERS:
        ratios = np.array([r["ratio"] for r in rows if r["band"] == band])
        if ratios.size == 0:
            continue
        med = np.median(ratios)
        scatter = 1.4826 * np.median(np.abs(ratios - med))
        print(f"{band:7}{ratios.size:>4}{med:>14.3f}{scatter:>10.3f}")
    ratios = np.array([r["ratio"] for r in rows])
    med = np.median(ratios)
    print(f"{'all':7}{ratios.size:>4}{med:>14.3f}{1.4826 * np.median(np.abs(ratios - med)):>10.3f}")
    for name in names:
        sub = np.array([r["ratio"] for r in rows if r["field"] == name])
        if sub.size:
            print(f"  {name}: median {np.median(sub):.3f} over {sub.size} comparisons")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
