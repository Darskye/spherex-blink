"""Run the whole pipeline for one target from the command line.

Useful for checking a target, pre-warming the cache, and making a quick PNG
contact sheet of the epoch frames.

    python scripts/try_target.py "Barnard's Star" --band 2 --png out.png

Developed with AI assistance (Claude, Anthropic). See README "Use of AI".
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app import config, gaia, photometry, process, resolver  # noqa: E402
from app.stack import StackManager  # noqa: E402


def stretch(image: np.ndarray, lo: float, hi: float) -> np.ndarray:
    scaled = np.clip((image - lo) / max(hi - lo, 1e-9), 0, 1)
    out = np.arcsinh(scaled * 10.0) / np.arcsinh(10.0)
    return np.nan_to_num(out * 255.0).astype(np.uint8)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target")
    parser.add_argument("--band", default="2")
    parser.add_argument("--fov", type=float, default=config.DEFAULT_FOV_ARCMIN)
    parser.add_argument("--png", default=None)
    parser.add_argument("--no-spectrum", action="store_true")
    args = parser.parse_args()

    manager = StackManager()
    await manager.start()
    try:
        target = await resolver.resolve(manager.client, args.target)
        if target is None:
            print("could not resolve", args.target)
            return 2
        print(f"target: {target['name'] or target['query']}  centre RA={target['ra']:.5f} Dec={target['dec']:+.5f}"
              f"  (moved {target['moved_arcsec']:.0f} arcsec since J2000)")
        band = None if args.band == "all" else int(args.band)

        t0 = time.time()
        job = manager.get_job(target["ra"], target["dec"], args.fov, band)
        last = -1
        while not job.finished:
            await asyncio.sleep(0.5)
            if job.done != last:
                last = job.done
                print(f"  {job.stage}: {job.done}/{job.total} (failed {job.failed}, cached {job.from_cache}) "
                      f"bands ready {sorted(job.bands_ready)}  t={time.time() - t0:.1f}s", flush=True)
        if job.stage == "error":
            print("ERROR:", job.error)
            return 1
        print(f"loaded {len(job.data)}/{job.total} exposures of {job.n_available} available in {time.time() - t0:.1f}s; "
              f"memory {job.nbytes / 1e6:.1f} MB")
        for ep in job.epochs_summary():
            print(f"  epoch {ep['index']}: {ep['date_min']} .. {ep['date_max']}  n={ep['n']}  per band {ep['per_band']}")

        t1 = time.time()
        header, data, flags = process.build_frames(job, band, "epochs", True)
        print(f"frames: {data.shape} built in {time.time() - t1:.2f}s; payload {len(process.pack_frames(header, data, flags)) / 1e3:.0f} kB")
        for fm in header["frames"]:
            wl = f"{fm['wl_min']:.2f}-{fm['wl_max']:.2f} um" if fm["wl"] is not None else "wavelength n/a"
            print(f"  frame epoch {fm['epoch']}: {fm['date']}  nexp={fm['nexp']}  {wl}  noise={fm['noise']:.4f} MJy/sr")

        if args.png and data.shape[0]:
            from PIL import Image

            finite = data[np.isfinite(data)]
            lo, hi = np.percentile(finite, [20, 99.7])
            tiles = [stretch(frame, lo, hi) for frame in data]
            if data.shape[0] >= 2:
                diff = data[-1] - data[0]
                span = max(np.nanpercentile(np.abs(diff), 99.5), 1e-9)
                tiles.append(np.nan_to_num(np.clip(diff / span * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8))
            sheet = np.concatenate(tiles, axis=1)
            Image.fromarray(sheet).resize((sheet.shape[1] * 2, sheet.shape[0] * 2), Image.NEAREST).save(args.png)
            print("wrote", args.png, "(epoch frames left to right; last tile = last minus first)")

        if not args.no_spectrum:
            t2 = time.time()
            # Same rules as the web app: follow targets moving faster than 0.1 arcsec/yr,
            # and use Gaia stars to flag blended measurements.
            motion = None
            if np.hypot(target.get("pmra") or 0.0, target.get("pmdec") or 0.0) >= 100.0:
                motion = {"pmra": target["pmra"] or 0.0, "pmdec": target["pmdec"] or 0.0, "epoch": target["epoch"]}
            stars = None
            try:
                stars = (await gaia.cone(manager.client, round(job.ra, 4), round(job.dec, 4), job.fov_arcmin))["stars"]
            except Exception as err:
                print(f"  (Gaia unavailable for blend check: {err.__class__.__name__})")
            spec = photometry.extract(job, job.ra, job.dec, None, motion, stars)
            if spec["blend"]:
                b = spec["blend"]
                print(f"  blended: {b['n']} of {spec['n_points']} exposures; a Gaia star (G {b['g']}) comes within {b['closest_arcsec']:.1f} arcsec")
            print(f"spectrum: {spec['n_points']} points ({spec['n_flagged']} flagged, {spec['n_saturated']} saturated, "
                  f"{spec['n_detected']} detected >3 sigma), median SNR {spec['median_snr']:.1f}, "
                  f"tracked={spec['tracked']}, in {time.time() - t2:.2f}s")
            for det in config.DETECTORS:
                band = [p for p in spec["points"] if p["detector"] == det and not p["flagged"]]
                if band:
                    flux = np.array([p["flux"] for p in band])
                    print(f"  D{det} spectrum: median {np.median(flux):.3f} mJy, median SNR {np.median([p['snr'] for p in band]):.1f} (n={len(band)})")
                series = [p for p in spec["light_curve"] if p["detector"] == det]
                if series:
                    text = "  ".join(f"{p['date'][:7]}:{p['rel']:.2f}+/-{p['rel_err']:.2f}(n={p['n']})" for p in series)
                    print(f"  D{det} light curve [{series[0]['method']}]: {text}")
            if spec["motion"]:
                mo = spec["motion"]
                print(f"  measured motion: {mo['total']:.2f} +/- {mo['total_err']:.2f} arcsec/yr at PA {mo['pa_deg']:.0f} deg "
                      f"(east {mo['pm_east']:+.2f}, north {mo['pm_north']:+.2f}) over {mo['baseline_yr']:.2f} yr, {len(mo['passes'])} passes"
                      + (f"; catalogue {np.hypot(target['pmra'], target['pmdec']) / 1000:.2f}" if target.get("pmra") else ""))
            if spec["variability"]:
                v = spec["variability"]
                print(f"  variability: D{v['detector']} changed {v['fraction'] * 100:.0f}% ({v['significance']:.1f} sigma) "
                      f"between {v['faint_date']} and {v['bright_date']} [{v['method']}]")
        return 0
    finally:
        await manager.stop()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
