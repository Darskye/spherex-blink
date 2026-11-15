"""Download and cache everything the example targets need, so they open instantly.

Run once after deploying (and again whenever you want the examples refreshed):

    python scripts/prewarm.py                 # the five challenge targets
    python scripts/prewarm.py "Wolf 359"      # any other names or coordinates

Developed with AI assistance (Claude, Anthropic). See README "Use of AI".
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app import catalogs, config, finder, gaia, payloads, resolver, wise  # noqa: E402
from app.main import EXAMPLES  # noqa: E402
from app.stack import StackManager  # noqa: E402


async def warm(manager: StackManager, query: str, fov: float, band: int | None) -> bool:
    target = await resolver.resolve(manager.client, query)
    if target is None:
        print(f"  could not resolve {query!r}")
        return False
    ra, dec = round(target["ra"], 5), round(target["dec"], 5)
    started = time.time()
    job = manager.get_job(ra, dec, fov, band)
    last = -1
    while not job.finished:
        await asyncio.sleep(2.0)
        if job.done != last:
            last = job.done
            print(f"  {query}: {job.done}/{job.total} exposures", flush=True)
    if job.stage == "error":
        print(f"  {query}: FAILED: {job.error}")
        return False
    try:
        stars = await gaia.cone(manager.client, round(ra, 4), round(dec, 4), job.fov_arcmin)
        n_stars = len(stars["stars"])
    except Exception as err:
        n_stars = f"unavailable ({err.__class__.__name__})"
    # Names of what is already catalogued here (SIMBAD is quick; the IRSA catalogues can take a minute).
    named = []
    for source in catalogs.SOURCES:
        try:
            found = await catalogs.lookup(manager.client, source, round(ra, 4), round(dec, 4), job.fov_arcmin)
            named.append(f"{source} {len(found['objects'])}")
        except Exception as err:
            named.append(f"{source} unavailable ({err.__class__.__name__})")
    print(f"  {query}: catalogues: {', '.join(named)}", flush=True)
    # Build and store the views people open first, so they appear at once.
    for view in ("colour", "2"):
        detector = None if view == "colour" else int(view)
        await payloads.frames(job, view, "epochs", True, job.band_exposures(detector), True)
    # The candidate search (it reuses the colour view and the catalogues fetched above).
    try:
        report = await finder.run(manager.client, job)
        counts = report["counts"]
        print(f"  {query}: candidates: {counts['unexplained']} unexplained, {counts['known']} already known, "
              f"{counts['doubtful']} doubtful, from {counts['sources']} sources", flush=True)
    except Exception as err:
        print(f"  {query}: candidate search failed ({err.__class__.__name__}: {err})")
    # The 2010 comparison frames (both WISE bands the viewer may ask for).
    wise_ok = []
    for wise_band in wise.BANDS:
        try:
            await wise.frame(manager.client, ra, dec, job.fov_arcmin, wise_band)
            wise_ok.append(wise_band)
        except Exception as err:
            print(f"  {query}: WISE {wise_band} unavailable ({err.__class__.__name__})")
    print(f"  {query}: {len(job.data)}/{job.total} exposures cached ({job.failed} failed), "
          f"{len(job.epochs_summary())} passes, Gaia stars: {n_stars}, WISE: {'+'.join(wise_ok) or 'none'}, "
          f"{time.time() - started:.0f}s")
    return job.failed == 0


async def main() -> int:
    queries = sys.argv[1:]
    manager = StackManager()
    await manager.start()
    ok = True
    try:
        if queries:
            for query in queries:
                ok &= await warm(manager, query, config.DEFAULT_FOV_ARCMIN, None)
        else:
            for example in EXAMPLES:
                band = int(example["band"]) if example["band"].isdigit() else None
                for query in [example["query"], *example.get("also_warm", [])]:
                    ok &= await warm(manager, query, float(example["fov"]), band)
    finally:
        await manager.stop()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
