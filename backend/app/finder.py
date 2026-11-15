"""Automatic search of one field for things worth a second look.

The viewer lets a person blink a patch of sky and spot what moved or changed.
This module does the same job by measurement, for every source in the field:

1. find the sources in the survey-pass images;
2. measure where each one is, and how bright, in every pass;
3. flag what moved and what changed brightness (no catalogue is consulted yet,
   so a known object that gets flagged is a fair test of the method);
4. confirm each flag with the slower exposure-by-exposure measurement;
5. only then compare with the catalogues: which flagged sources are already
   known, which sources no catalogue lists, and which catalogued 2010 sources
   are no longer there;
6. measure anything that differs from the 2010 AllWISE catalogue again, in the
   wavelength range WISE itself used, so that colour is not mistaken for change.

Everything it returns is a candidate: a place for a person to look, with the
measurements and the checks it passed. It is not a discovery list.

Developed with AI assistance (Claude, Anthropic). See README "Use of AI".
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import ndimage

from . import catalogs, config, gaia, payloads, photometry, process, resolver

log = logging.getLogger("spx.finder")

RECIPE_VERSION = "2"
NATIVE = config.NATIVE_PIXEL_ARCSEC

DETECT_SNR = 6.0              # in the image of all passes combined
PASS_DETECT_SNR = 8.0         # in one pass alone (catches what is there only once)
MAX_SOURCES = 300
APERTURE_NATIVE = 1.5         # aperture radius, detector pixels
ANNULUS_NATIVE = (3.0, 5.0)   # sky annulus, detector pixels
PSF_FWHM_ARCSEC = 7.5
POSITION_FLOOR_ARCSEC = 0.3   # same floor the per-exposure motion fit uses
MATCH_ARCSEC = 7.5            # a catalogue entry this close is the same object
BLEND_ARCSEC = 9.0

SCREEN_MOTION_ARCSEC_YR = 0.8
SCREEN_MOTION_SIGMA = 4.0
SCREEN_CHANGE_FRACTION = 0.15
SCREEN_CHANGE_SIGMA = 3.5
CONFIRM_CHANGE_FRACTION = 0.15
CONFIRM_CHANGE_SIGMA = 6.0
DRIFT_MIN_ARCSEC = 3.0        # offset from a 2010 or 1999 catalogue position worth reporting
GONE_MIN_SNR = 10.0           # how clearly a catalogued source should have been seen
GONE_MAX_FRACTION = 0.4       # and how little of it is there now
MAX_CONFIRM = 14
MAX_REPORTED = 24

# The SPHEREx wavelength range that matches each WISE band (micron), and the band's zero point (Jy, Vega).
WISE_LIKE = {"w1": (2.8, 3.8, 309.54), "w2": (4.1, 5.0, 171.787)}
FADED_MAX_FRACTION = 0.75     # "faded": less than this is left, like for like, with three sigma to spare
COLD_COLOUR_MAG = 2.0         # W1 - W2 of an object that shines almost only in W2 (late T and Y dwarfs)
MAX_YARDSTICK = 6             # ordinary stars used to tie this tool's brightness scale to AllWISE's

CHANNELS = ("3.8–5.0 µm", "1.6–3.8 µm", "0.75–1.6 µm")

# SIMBAD object types whose brightness is expected to change.
_VARIABLE_OTYPES = {
    "V*", "Ir*", "Er*", "Ro*", "Pu*", "LP*", "Mi*", "RR*", "Ce*", "dS*", "RV*", "WV*", "bC*", "cC*", "gD*", "SX*",
    "TT*", "Or*", "Y*O", "Ae*", "EB*", "El*", "CV*", "No*", "SN*", "BY*", "RS*", "Sy*", "XB*", "LXB", "HXB",
    "AGN", "QSO", "BLL", "Bla", "Sy1", "Sy2", "SyG", "C*", "S*", "AB*", "OH*", "pA*", "WR*", "Be*", "s*r", "s*b",
}
_VARIABLE_WORDS = (
    "variable", "mira", "t tauri", "young stellar", "eclipsing", "nova", "blazar", "quasar", "active gal",
    "bl lac", "cepheid", "pulsating", "eruptive", "asymptotic", "carbon star", "symbiotic", "orion", "seyfert", "herbig",
)
_EXTENDED_WORDS = ("galaxy", "galaxies", "nebula", "cluster", "hii", "cloud", "region")


# --------------------------------------------------------------------------- small helpers


def _sigma(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    med = np.median(values)
    return float(1.4826 * np.median(np.abs(values - med)))


def _snr_image(image: np.ndarray, smooth_px: float) -> np.ndarray:
    """The image, optionally smoothed, in units of its own noise (0 where there is no data)."""
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros(image.shape, dtype=np.float32)
    filled = np.where(finite, image - np.median(image[finite]), 0.0)
    if smooth_px > 0:
        filled = ndimage.gaussian_filter(filled, smooth_px, mode="nearest")
    filled = filled - np.median(filled[finite])
    sigma = _sigma(filled[finite])
    if sigma <= 0:
        return np.zeros(image.shape, dtype=np.float32)
    return np.where(finite, filled / sigma, 0.0).astype(np.float32)


def _centroid(image: np.ndarray, x: float, y: float, native: float, reach: float = 1.5) -> tuple[float, float] | None:
    """Gaussian-weighted centre of light near (x, y); None if it wanders off or finds nothing."""
    n = image.shape[0]
    sigma, radius = 0.8 * native, 2.0 * native
    x0, y0 = x, y
    for _ in range(10):
        c0, c1 = max(0, int(x - radius)), min(n - 1, int(x + radius) + 1)
        r0, r1 = max(0, int(y - radius)), min(n - 1, int(y + radius) + 1)
        if c1 <= c0 or r1 <= r0:
            return None
        yy, xx = np.mgrid[r0:r1 + 1, c0:c1 + 1]
        weight = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma * sigma)) * np.clip(image[r0:r1 + 1, c0:c1 + 1], 0, None)
        total = float(weight.sum())
        if total <= 0:
            return None
        nx, ny = float((weight * xx).sum() / total), float((weight * yy).sum() / total)
        moved = math.hypot(nx - x, ny - y)
        x, y = nx, ny
        if moved < 0.01:
            break
    if math.hypot(x - x0, y - y0) > reach * native:
        return None
    return x, y


class _Apertures:
    """Background-subtracted aperture sums on a stack of planes shaped (passes, channels, n, n)."""

    def __init__(self, cube: np.ndarray, native: float):
        self.radius = int(math.ceil(ANNULUS_NATIVE[1] * native)) + 1
        dy, dx = np.mgrid[-self.radius:self.radius + 1, -self.radius:self.radius + 1]
        rr = np.hypot(dx, dy)
        self.inside = rr <= APERTURE_NATIVE * native
        self.annulus = (rr >= ANNULUS_NATIVE[0] * native) & (rr <= ANNULUS_NATIVE[1] * native)
        self.n_inside = int(self.inside.sum())
        pad = self.radius
        self.padded = np.pad(cube, ((0, 0), (0, 0), (pad, pad), (pad, pad)), constant_values=np.nan)

    def at(self, x: int, y: int, passes: slice = slice(None)) -> np.ndarray:
        """Flux at integer pixel (x, y) for the chosen passes: shape (passes, channels), NaN where not covered."""
        size = 2 * self.radius + 1
        sub = self.padded[passes, :, y:y + size, x:x + size]
        inside = sub[..., self.inside]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            background = np.nanmedian(sub[..., self.annulus], axis=-1)
        covered = np.isfinite(inside).sum(axis=-1) / self.n_inside
        flux = np.nansum(inside - background[..., None], axis=-1)
        return np.where((covered >= 0.8) & np.isfinite(background), flux, np.nan)


def _aperture_noise(apertures: _Apertures, cube: np.ndarray, native: float, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Scatter of aperture sums on empty sky, per pass and channel (correlated noise included)."""
    n = cube.shape[-1]
    rng = np.random.default_rng(20261114)
    margin = int(math.ceil(APERTURE_NATIVE * native)) + 1
    for clearance in (2.5, 1.8):
        blank = []
        for x, y in rng.integers(margin, n - margin, size=(2500, 2)):
            if xs.size and np.min(np.hypot(xs - x, ys - y)) < clearance * native:
                continue
            blank.append((int(x), int(y)))
            if len(blank) >= 250:
                break
        if len(blank) >= 25:
            break
    if len(blank) >= 25:
        fluxes = np.stack([apertures.at(x, y) for x, y in blank])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            med = np.nanmedian(fluxes, axis=0)
            noise = 1.4826 * np.nanmedian(np.abs(fluxes - med), axis=0)
    else:
        noise = np.full(cube.shape[:2], np.nan)
    # Where empty sky could not be sampled (a crowded field), fall back on the pixel noise.
    for p in range(cube.shape[0]):
        for c in range(cube.shape[1]):
            if not (np.isfinite(noise[p, c]) and noise[p, c] > 0):
                noise[p, c] = _sigma(cube[p, c]) * math.sqrt(apertures.n_inside) * 2.0
    return noise


def _saturation_radius(flagged: np.ndarray, x: float, y: float, native: float) -> float:
    """How far (grid pixels) the flagged core around (x, y) reaches; 0 if the centre is not flagged."""
    n = flagged.shape[0]
    reach = int(math.ceil(12 * native))
    c0, c1 = max(0, int(round(x)) - reach), min(n - 1, int(round(x)) + reach)
    r0, r1 = max(0, int(round(y)) - reach), min(n - 1, int(round(y)) + reach)
    yy, xx = np.mgrid[r0:r1 + 1, c0:c1 + 1]
    ring = np.rint(np.hypot(xx - x, yy - y)).astype(int)
    sub = flagged[r0:r1 + 1, c0:c1 + 1]
    radius = 0
    for b in range(reach + 1):
        members = ring == b
        if not members.any():
            continue
        if sub[members].mean() < 0.15:
            break
        radius = b + 1
    return float(radius)


def _wings(cube: np.ndarray, flagf: np.ndarray, x: float, y: float, r_sat: float, native: float) -> tuple[np.ndarray, np.ndarray]:
    """Brightness of a saturated star's unsaturated outer wings, per pass and channel."""
    passes, channels, n, _ = cube.shape
    r1, r2 = r_sat + 0.5 * native, r_sat + 2.5 * native
    b1, b2 = r_sat + 5.0 * native, r_sat + 7.0 * native
    reach = int(math.ceil(b2)) + 1
    c0, c1 = max(0, int(round(x)) - reach), min(n - 1, int(round(x)) + reach)
    r0, r1_ = max(0, int(round(y)) - reach), min(n - 1, int(round(y)) + reach)
    yy, xx = np.mgrid[r0:r1_ + 1, c0:c1 + 1]
    rr = np.hypot(xx - x, yy - y)
    ring, sky = (rr >= r1) & (rr <= r2), (rr >= b1) & (rr <= b2)
    flux = np.full((passes, channels), np.nan)
    err = np.full((passes, channels), np.nan)
    for p in range(passes):
        clean = flagf[p, r0:r1_ + 1, c0:c1 + 1] < 0.5
        for c in range(channels):
            sub = cube[p, c, r0:r1_ + 1, c0:c1 + 1]
            good = clean & np.isfinite(sub)
            values, back = sub[ring & good], sub[sky & good]
            if values.size < 30 or back.size < 30:
                continue
            flux[p, c] = float(np.median(values) - np.median(back))
            err[p, c] = 1.2533 * _sigma(values) / math.sqrt(values.size)
    return flux, err


def _fit_motion(years: np.ndarray, east: np.ndarray, north: np.ndarray, err: np.ndarray) -> dict | None:
    """Straight-line motion through per-pass positions (arcsec, arcsec per year)."""
    ok = np.isfinite(east) & np.isfinite(north) & np.isfinite(err) & (err > 0)
    if ok.sum() < 2:
        return None
    t, e, nn, s = years[ok], east[ok], north[ok], err[ok]
    if t.max() - t.min() < 0.25:
        return None
    w = 1.0 / s ** 2
    t0 = float((w * t).sum() / w.sum())
    sxx = float((w * (t - t0) ** 2).sum())
    e0, n0 = float((w * e).sum() / w.sum()), float((w * nn).sum() / w.sum())
    pm_e = float((w * (t - t0) * (e - e0)).sum() / sxx)
    pm_n = float((w * (t - t0) * (nn - n0)).sum() / sxx)
    resid = w * ((e - e0 - pm_e * (t - t0)) ** 2 + (nn - n0 - pm_n * (t - t0)) ** 2)
    dof = 2 * (int(ok.sum()) - 2)
    return {
        "pm_east": pm_e, "pm_north": pm_n, "total": math.hypot(pm_e, pm_n), "err": 1.0 / math.sqrt(sxx),
        "chi2": float(resid.sum() / dof) if dof > 0 else None,
        "baseline": float(t.max() - t.min()), "passes": int(ok.sum()),
    }


# --------------------------------------------------------------------------- step 1-3: measure, then screen


def measure_field(header: dict, data: np.ndarray, flags: np.ndarray) -> dict:
    """Find the sources and measure each one in every pass.

    header, data, flags: the colour frames of a field (process.build_colour_frames):
    three planes per pass (3.8-5.0, 1.6-3.8 and 0.75-1.6 micron) and one flag plane.
    """
    n = int(header["n"])
    meta = header["frames"]
    passes = len(meta)
    if passes == 0 or data.shape[0] != 3 * passes:
        raise ValueError("the finder needs the colour frames of at least one pass")
    scale = float(header["scale_arcsec"])
    native = NATIVE / scale
    cube = np.asarray(data, dtype=np.float32).reshape(passes, 3, n, n)
    flagf = np.asarray(flags, dtype=np.float32).reshape(passes, n, n) / 255.0
    years = np.array([m["year"] for m in meta], dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean = np.nanmean(cube, axis=0)                           # (3, n, n)
    smooth = 0.5 * native

    def significance(planes: np.ndarray) -> np.ndarray:
        """Best of: any one channel, or the three together. A cold object shows in one channel only."""
        snr = np.stack([_snr_image(plane, smooth) for plane in planes])
        return np.maximum(snr.max(axis=0), snr.sum(axis=0) / math.sqrt(len(planes)))

    def peaks(det: np.ndarray, threshold: float) -> list[tuple[float, float, float]]:
        size = int(2 * round(native)) + 1
        local = (det == ndimage.maximum_filter(det, size=size, mode="nearest")) & (det >= threshold)
        rows, cols = np.nonzero(local)
        return [(float(c), float(r), float(det[r, c])) for r, c in zip(rows, cols)]

    found = peaks(significance(mean), DETECT_SNR)
    per_pass_det = [significance(cube[p]) for p in range(passes)]
    if passes > 1:
        for p in range(passes):
            for x, y, snr in peaks(per_pass_det[p], PASS_DETECT_SNR):
                if all(math.hypot(x - fx, y - fy) > 1.5 * native for fx, fy, _ in found):
                    found.append((x, y, snr))
    found.sort(key=lambda item: -item[2])

    white_mean = np.sum([_snr_image(plane, 0) for plane in mean], axis=0)
    white = [np.sum([_snr_image(cube[p, c], 0) for c in range(3)], axis=0) for p in range(passes)]
    flagged_any = (flagf >= 0.5).any(axis=0)

    border = APERTURE_NATIVE * native
    sources: list[dict] = []
    for x, y, snr in found:                                           # brightest first
        centre = _centroid(white_mean, x, y, native) or (x, y)
        x, y = centre
        if not (border <= x <= n - 1 - border and border <= y <= n - 1 - border):
            continue
        if any(math.hypot(x - s["x"], y - s["y"]) < 0.7 * native for s in sources):
            continue
        # Inside a brighter star's saturated core there are only its spikes and glare.
        if any(s["r_sat"] > 0 and math.hypot(x - s["x"], y - s["y"]) < s["r_sat"] + 2.0 * native for s in sources):
            continue
        xi, yi = int(round(x)), int(round(y))
        core = flagged_any[max(0, yi - 1):yi + 2, max(0, xi - 1):xi + 2]
        r_sat = _saturation_radius(flagged_any, x, y, native) if core.any() else 0.0
        sources.append({"x": x, "y": y, "snr": snr, "r_sat": r_sat, "saturated": r_sat >= 0.8 * native})
        if len(sources) >= MAX_SOURCES:
            break

    xs = np.array([s["x"] for s in sources])
    ys = np.array([s["y"] for s in sources])
    apertures = _Apertures(cube, native)
    noise = _aperture_noise(apertures, cube, native, xs, ys)      # (passes, 3)
    mean_apertures = _Apertures(mean[None], native)
    mean_noise = _aperture_noise(mean_apertures, mean[None], native, xs, ys)[0]

    count = len(sources)
    flux = np.full((count, passes, 3), np.nan)
    err = np.full((count, passes, 3), np.nan)
    for i, s in enumerate(sources):
        xi, yi = int(round(s["x"])), int(round(s["y"]))
        s["edge"] = not (2.5 * native <= s["x"] <= n - 1 - 2.5 * native and 2.5 * native <= s["y"] <= n - 1 - 2.5 * native)

        # Where it is in each pass (grid pixels), and how well that is known (arcsec).
        positions = np.full((passes, 2), np.nan)
        for p in range(passes):
            centre = _centroid(white[p], s["x"], s["y"], native, reach=1.5)
            if centre is not None:
                positions[p] = centre

        if s["saturated"]:
            flux[i], err[i] = _wings(cube, flagf, s["x"], s["y"], s["r_sat"], native)
        else:
            for p in range(passes):
                px = int(round(positions[p, 0])) if np.isfinite(positions[p, 0]) else xi
                py = int(round(positions[p, 1])) if np.isfinite(positions[p, 1]) else yi
                px, py = min(max(px, 0), n - 1), min(max(py, 0), n - 1)
                flux[i, p] = apertures.at(px, py, slice(p, p + 1))[0]
                err[i, p] = noise[p]
                # A measurement resting on flagged pixels is not trusted.
                if flagf[p, max(0, py - 2):py + 3, max(0, px - 2):px + 3].max() >= 0.5:
                    flux[i, p] = np.nan

        with np.errstate(invalid="ignore", divide="ignore"):
            pass_snr = np.sqrt(np.nansum(np.clip(flux[i] / err[i], 0, None) ** 2, axis=1))
        floor = 1.2 if s["saturated"] else POSITION_FLOOR_ARCSEC
        with np.errstate(invalid="ignore", divide="ignore"):
            pos_err = np.hypot(floor, PSF_FWHM_ARCSEC / (2.355 * np.maximum(pass_snr, 1e-3)))
        seen = np.isfinite(positions[:, 0]) & (pass_snr >= 5.0)
        s["positions"] = positions
        s["pass_snr"] = pass_snr
        s["motion"] = _fit_motion(
            years,
            np.where(seen, -(positions[:, 0] - s["x"]) * scale, np.nan),      # east is towards column 0
            np.where(seen, -(positions[:, 1] - s["y"]) * scale, np.nan),      # north is towards row 0
            pos_err,
        )
        s["mean_flux"] = mean_apertures.at(xi, yi)[0]

    for i, s in enumerate(sources):
        others = [math.hypot(s["x"] - o["x"], s["y"] - o["y"]) for j, o in enumerate(sources) if j != i]
        s["neighbour_arcsec"] = min(others) * scale if others else None
        s["near_saturated"] = any(
            o["saturated"] and o is not s and o["snr"] > s["snr"]
            and math.hypot(s["x"] - o["x"], s["y"] - o["y"]) < o["r_sat"] + 4 * native
            for o in sources
        )

    relative, relative_err, floors = _relative(flux, err, np.array([s["saturated"] for s in sources], dtype=bool))
    for i, s in enumerate(sources):
        s["flux"], s["flux_err"] = flux[i], err[i]
        s["change"] = _change(relative[i], relative_err[i], flux[i], err[i], floors, meta)

    return {
        "n": n, "scale": scale, "native": native, "passes": passes, "years": years, "meta": meta,
        "ra": float(header["ra"]), "dec": float(header["dec"]), "fov": float(header["fov_arcmin"]),
        "sources": sources, "noise": noise, "mean_noise": mean_noise, "floors": floors,
        "mean": mean, "mean_apertures": mean_apertures, "flagged_any": flagged_any,
    }


def _relative(flux: np.ndarray, err: np.ndarray, saturated: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Brightness of each source in each pass relative to its own average, with the
    field's ordinary stars used to take out pass-to-pass differences common to all.

    Each pass samples slightly different wavelengths, which makes every star a
    few percent brighter or fainter. The spread that leaves among ordinary stars
    is returned as the floor below which a change means nothing.
    """
    count, passes, channels = flux.shape
    floors = np.full(channels, 0.10)
    if count == 0:
        return flux.copy(), err.copy(), floors
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean = np.nanmean(flux, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = flux / mean
        ratio_err = err / np.abs(mean)
        snr = np.nanmean(flux / err, axis=1)                                    # (count, channels)
    norm = np.ones((passes, channels))
    for c in range(channels):
        reference = ~saturated & (snr[:, c] >= 25.0) & np.isfinite(ratio[:, :, c]).all(axis=1)
        if reference.sum() < 5:
            continue
        norm[:, c] = np.median(ratio[reference, :, c], axis=0)
        floors[c] = float(np.clip(_sigma((ratio[reference, :, c] / norm[:, c] - 1.0).ravel()), 0.03, 0.25))
    return ratio / norm, ratio_err / norm, floors


def _change(rel: np.ndarray, rel_err: np.ndarray, flux: np.ndarray, err: np.ndarray, floors: np.ndarray, meta: list[dict]) -> dict | None:
    """Largest brightness change between passes, in the channel where it is most significant."""
    best = None
    for c in range(rel.shape[1]):
        ok = np.isfinite(rel[:, c]) & np.isfinite(rel_err[:, c])
        if ok.sum() < 2:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            if np.nanmax(flux[ok, c] / err[ok, c]) < 10.0:
                continue
        index = np.nonzero(ok)[0]
        hi = index[np.argmax(rel[index, c])]
        lo = index[np.argmin(rel[index, c])]
        top, bottom = float(rel[hi, c]), float(rel[lo, c])
        if top <= 0:
            continue
        middle = 0.5 * (top + max(bottom, 0.0))
        sigma = math.sqrt(rel_err[hi, c] ** 2 + rel_err[lo, c] ** 2 + 2.0 * (floors[c] * middle) ** 2)
        candidate = {
            "channel": c, "fraction": float(min(1.0, (top - bottom) / top)), "significance": float((top - bottom) / sigma),
            "bright": meta[hi]["date"], "faint": meta[lo]["date"], "passes": int(ok.sum()),
        }
        if best is None or candidate["significance"] > best["significance"]:
            best = candidate
    return best


def screen(field: dict) -> list[dict]:
    """First pass, with no catalogue in sight: what looks as if it moved or changed?"""
    hits = []
    for index, s in enumerate(field["sources"]):
        m = s["motion"]
        if (m and m["total"] >= SCREEN_MOTION_ARCSEC_YR and m["total"] >= SCREEN_MOTION_SIGMA * m["err"]
                and (m["chi2"] is None or m["chi2"] < 8.0) and not s["edge"]):
            hits.append({"kind": "moved", "source": index, "strength": m["total"] / m["err"]})
        ch = s["change"]
        if (ch and ch["fraction"] >= SCREEN_CHANGE_FRACTION and ch["significance"] >= SCREEN_CHANGE_SIGMA
                and not s["edge"] and not s["near_saturated"]):
            hits.append({"kind": "changed", "source": index, "strength": ch["significance"]})
    hits.sort(key=lambda h: -h["strength"])
    return hits


# --------------------------------------------------------------------------- step 4: confirm


def confirm(field: dict, hits: list[dict], job, exposures) -> tuple[list[dict], int]:
    """Re-measure each flagged source exposure by exposure (the method behind the
    spectrum panel), which compares like wavelength with like. Returns the hits
    that survive, and how many did not.

    A motion counts only if the two methods agree on it: the pass images (first
    look) and the single exposures (this step) are measured independently, and
    an artefact rarely fools both in the same way. No catalogue is used here
    either; whether a neighbour shares the pixel is judged afterwards, in explain().
    """
    grid = process.Grid(field["ra"], field["dec"], field["fov"])
    year = float(np.mean(field["years"]))
    kept, dropped, measured = [], 0, {}
    for hit in hits[:MAX_CONFIRM]:
        s = field["sources"][hit["source"]]
        if hit["source"] not in measured:
            ra, dec = (float(v) for v in grid.wcs.all_pix2world(s["x"], s["y"], 0))
            result = photometry.extract(job, ra, dec, exposures, None, None)
            motion, first = result.get("motion"), s["motion"]
            agreed = None
            if motion and motion["significant"] and first and first["total"] >= SCREEN_MOTION_ARCSEC_YR:
                gap = math.hypot(motion["pm_east"] - first["pm_east"], motion["pm_north"] - first["pm_north"])
                if gap <= max(3.0 * math.hypot(first["err"], motion["total_err"]), 0.3 * first["total"]):
                    agreed = motion
            if agreed:
                # It moves: measure its brightness again, following it.
                tracked = photometry.extract(job, ra, dec, exposures, {
                    "pmra": agreed["pm_east"] * 1000.0, "pmdec": agreed["pm_north"] * 1000.0, "epoch": year,
                }, None)
                result["variability"] = tracked.get("variability")
            measured[hit["source"]] = (result, agreed)
        result, agreed = measured[hit["source"]]
        if hit["kind"] == "moved":
            if agreed:
                hit["confirmed"] = {
                    "total": agreed["total"], "err": agreed["total_err"], "pa_deg": agreed["pa_deg"],
                    "pm_east": agreed["pm_east"], "pm_north": agreed["pm_north"], "baseline": agreed["baseline_yr"],
                }
        else:
            v = result.get("variability")
            if v and v["fraction"] >= CONFIRM_CHANGE_FRACTION and v["significance"] >= CONFIRM_CHANGE_SIGMA:
                hit["confirmed"] = {
                    "fraction": v["fraction"], "significance": v["significance"], "detector": v["detector"],
                    "bright": v["bright_date"], "faint": v["faint_date"], "method": v["method"],
                }
        if "confirmed" in hit:
            kept.append(hit)
        else:
            dropped += 1
    return kept, dropped + max(0, len(hits) - MAX_CONFIRM)


# --------------------------------------------------------------------------- step 5: what do the catalogues say?


def _clean_name(name: str) -> str:
    for prefix in ("NAME ", "V* ", "* ", "** ", "EM* "):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return " ".join(name.split())


def _is_variable_type(obj: dict) -> bool:
    otype = (obj.get("otype") or "").rstrip("?")
    kind = (obj.get("kind") or "").lower()
    return otype in _VARIABLE_OTYPES or any(word in kind for word in _VARIABLE_WORDS)


def _is_extended(obj: dict) -> bool:
    kind = (obj.get("kind") or "").lower()
    return any(word in kind for word in _EXTENDED_WORDS)


def _month(date: str | None) -> str:
    try:
        return datetime.strptime(date[:10], "%Y-%m-%d").strftime("%b %Y")
    except (TypeError, ValueError):
        return date or "?"


def _compass(pa_deg: float) -> str:
    names = ("north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west")
    return names[int(((pa_deg % 360.0) + 22.5) // 45.0) % 8]


class _Catalogue:
    """Catalogue entries placed on the field's pixel grid at a given date."""

    def __init__(self, objects: list[dict] | None, grid: process.Grid, year: float):
        self.available = objects is not None
        self.objects = objects or []
        ras, decs = [], []
        for o in self.objects:
            ra, dec = o["ra"], o["dec"]
            if o.get("pmra") or o.get("pmdec"):
                ra, dec = resolver.propagate(ra, dec, o.get("pmra") or 0.0, o.get("pmdec") or 0.0, o.get("epoch") or 2000.0, year)
            ras.append(ra)
            decs.append(dec)
        if ras:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x, y = grid.wcs.all_world2pix(np.array(ras), np.array(decs), 0)
            self.x, self.y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        else:
            self.x, self.y = np.zeros(0), np.zeros(0)

    def nearest(self, x: float, y: float, radius_px: float) -> tuple[dict | None, float | None]:
        if self.x.size == 0:
            return None, None
        d = np.hypot(self.x - x, self.y - y)
        i = int(np.argmin(d))
        return (self.objects[i], float(d[i])) if d[i] <= radius_px else (None, None)

    def within(self, x: float, y: float, radius_px: float) -> list[dict]:
        """Entries within the radius, nearest first."""
        if self.x.size == 0:
            return []
        d = np.hypot(self.x - x, self.y - y)
        return [self.objects[i] for i in np.argsort(d) if d[i] <= radius_px]

    def position(self, obj: dict) -> tuple[float, float]:
        i = next(k for k, o in enumerate(self.objects) if o is obj)
        return float(self.x[i]), float(self.y[i])


def _principal(found: list[dict], catalogue: "_Catalogue", x: float, y: float, scale: float) -> dict | None:
    """Of several SIMBAD entries at one spot (a star, its planets, its components), the one to name."""
    stars = [o for o in found if (o.get("otype") or "").rstrip("?") != "Pl" and "planet" not in (o.get("kind") or "").lower()]
    found = stars or found
    if not found:
        return None

    def distance(o: dict) -> float:
        ox, oy = catalogue.position(o)
        return math.hypot(ox - x, oy - y) * scale

    nearest = distance(found[0])
    close = [o for o in found if distance(o) <= nearest + 2.0]
    return min(close, key=lambda o: len(_clean_name(o["name"])))


def _motion_of(obj: dict | None) -> float | None:
    if not obj or obj.get("pmra") is None or obj.get("pmdec") is None:
        return None
    return math.hypot(obj["pmra"], obj["pmdec"]) / 1000.0


def explain(field: dict, hits: list[dict], gaia_stars: list[dict] | None, cats: dict[str, list[dict] | None]) -> dict:
    """Compare with the catalogues and write up the candidates."""
    grid = process.Grid(field["ra"], field["dec"], field["fov"])
    scale, native, n = field["scale"], field["native"], field["n"]
    year = float(np.mean(field["years"]))
    sources = field["sources"]
    match_px, blend_px = MATCH_ARCSEC / scale, BLEND_ARCSEC / scale

    gaia = _Catalogue(gaia_stars, grid, year)
    simbad = _Catalogue(cats.get("simbad"), grid, year)
    twomass = _Catalogue(cats.get("twomass"), grid, year)
    allwise = _Catalogue(cats.get("allwise"), grid, year)
    simbad_2010 = _Catalogue(cats.get("simbad"), grid, 2010.5)

    # Small shift between this tool's positions and Gaia's, taken out before comparing.
    offsets = []
    for s in sources:
        if s["saturated"] or s["snr"] < 15 or len(gaia.within(s["x"], s["y"], blend_px)) != 1:
            continue
        star, dist = gaia.nearest(s["x"], s["y"], 5.0 / scale)
        if star is not None:
            i = gaia.objects.index(star)
            offsets.append((s["x"] - gaia.x[i], s["y"] - gaia.y[i]))
    shift = np.median(np.array(offsets), axis=0) if len(offsets) >= 5 else np.zeros(2)

    def sky(x: float, y: float) -> tuple[float, float]:
        ra, dec = grid.wcs.all_pix2world(x, y, 0)
        return float(ra), float(dec)

    for s in sources:
        s["cx"], s["cy"] = s["x"] - shift[0], s["y"] - shift[1]
        s["ra"], s["dec"] = sky(s["cx"], s["cy"])
        s["gaia"], _ = gaia.nearest(s["cx"], s["cy"], match_px)
        s["simbad"] = _principal(simbad.within(s["cx"], s["cy"], match_px), simbad, s["cx"], s["cy"], scale)
        s["twomass"], s["twomass_px"] = twomass.nearest(s["cx"], s["cy"], match_px)
        s["allwise"], s["allwise_px"] = allwise.nearest(s["cx"], s["cy"], match_px)
        s["crowd"] = len(gaia.within(s["cx"], s["cy"], blend_px))

    density = len([1 for x, y in zip(gaia.x, gaia.y) if 0 <= x < n and 0 <= y < n]) / max(field["fov"] ** 2, 1e-6)
    chance = 1.0 - math.exp(-density * math.pi * (MATCH_ARCSEC / 60.0) ** 2)      # odds a random spot has a Gaia star this close
    crowded = chance > 0.25
    notes: list[str] = []
    candidates: list[dict] = []

    def name_of(s: dict) -> str | None:
        if s["simbad"]:
            return _clean_name(s["simbad"]["name"])
        if s["gaia"]:
            return f"Gaia DR3 {s['gaia']['id']}"
        if s["allwise"]:
            return s["allwise"]["name"]
        if s["twomass"]:
            return s["twomass"]["name"]
        return None

    def base(s: dict, kind: str) -> dict:
        caveats = []
        if s["saturated"]:
            caveats.append("The star is too bright for SPHEREx (saturated), so it is measured through its outer glow.")
        if s["crowd"] >= 2 or (s["neighbour_arcsec"] is not None and s["neighbour_arcsec"] < 2.5 * NATIVE):
            caveats.append("Another star lies within one or two pixels, so their light is mixed.")
        if s["near_saturated"]:
            caveats.append("It sits in the glare of a much brighter star.")
        if s["edge"]:
            caveats.append("It is close to the edge of the field.")
        return {
            "kind": kind, "ra": s["ra"], "dec": s["dec"], "snr": round(float(s["snr"]), 1),
            "name": name_of(s), "facts": [], "caveats": caveats, "known": None,
        }

    # ---- what moved or changed between passes
    for hit in hits:
        s = sources[hit["source"]]
        c = base(s, hit["kind"])
        measured = hit["confirmed"]
        if hit["kind"] == "moved":
            c["title"] = f"Moved {measured['total']:.1f}″ per year"
            c["facts"].append(
                f"Its position shifted {measured['total']:.2f} ± {measured['err']:.2f}″ per year towards the "
                f"{_compass(measured['pa_deg'])}, measured over {measured['baseline']:.1f} years of SPHEREx passes."
            )
            c["measured"] = {k: round(float(v), 3) for k, v in measured.items()}
            c["strength"] = measured["total"] / max(measured["err"], 1e-6)
            listed = [(o, _motion_of(o)) for o in (s["simbad"], s["gaia"]) if _motion_of(o) is not None]
            agreeing = [
                (o, pm) for o, pm in listed
                if math.hypot(o["pmra"] / 1000.0 - measured["pm_east"], o["pmdec"] / 1000.0 - measured["pm_north"])
                <= max(3.0 * measured["err"], 0.35 * measured["total"])
            ]
            if agreeing:
                o, pm = agreeing[0]
                kind = (s["simbad"] or {}).get("kind")
                c["verdict"] = "known"
                c["known"] = {"name": c["name"], "type": kind}
                c["explanation"] = (
                    f"Already known: {c['name']}" + (f" ({kind})" if kind else "") +
                    f". Its catalogued motion is {pm:.2f}″ per year, which matches."
                )
            elif listed:
                o, pm = listed[0]
                c["verdict"] = "doubtful"
                c["explanation"] = (
                    f"Probably not real. {c['name']} is catalogued here with a motion of only {pm:.2f}″ per year, "
                    "so the shift is more likely two blended stars or an image artefact."
                )
            else:
                c["verdict"] = "doubtful" if c["caveats"] else "unexplained"
                c["explanation"] = (
                    "No catalogue lists a moving object here."
                    + (" The measurement has the weaknesses listed below, so treat it with caution." if c["caveats"] else "")
                )
        else:
            band = config.DETECTOR_RANGES_UM[int(measured["detector"])]
            c["title"] = f"Changed brightness by {measured['fraction'] * 100:.0f}%"
            c["facts"].append(
                f"At {band[0]:.2f}–{band[1]:.2f} µm it was {measured['fraction'] * 100:.0f}% fainter in "
                f"{_month(measured['faint'])} than in {_month(measured['bright'])} "
                f"({measured['significance']:.0f} times the measurement error)."
            )
            c["measured"] = {
                "fraction": round(float(measured["fraction"]), 3), "significance": round(float(measured["significance"]), 1),
                "detector": int(measured["detector"]), "bright": measured["bright"], "faint": measured["faint"],
            }
            c["strength"] = measured["significance"]
            variable_in_gaia = bool(s["gaia"] and s["gaia"].get("var"))
            if s["simbad"] and _is_variable_type(s["simbad"]):
                c["verdict"] = "known"
                c["known"] = {"name": c["name"], "type": s["simbad"].get("kind")}
                c["explanation"] = f"Already known: SIMBAD lists {c['name']} as \"{s['simbad'].get('kind') or 'variable object'}\"."
            elif variable_in_gaia:
                c["verdict"] = "known"
                c["known"] = {"name": c["name"], "type": "Variable (Gaia DR3)"}
                c["explanation"] = f"Already known: Gaia DR3 flags {c['name']} as variable."
            else:
                c["verdict"] = "doubtful" if c["caveats"] else "unexplained"
                what = f"{c['name']} is catalogued, but not as a variable" if c["name"] else "Nothing is catalogued at this position"
                c["explanation"] = (
                    f"{what} in SIMBAD or Gaia DR3. Other variable-star lists (AAVSO VSX, ZTF) were not checked."
                )
        candidates.append(c)

    # ---- catalogue comparisons (need the infrared catalogues)
    have_ir = twomass.available and allwise.available
    if not have_ir:
        notes.append("The 2MASS or AllWISE catalogue did not answer, so sources missing from, or gone since, those catalogues were not searched for.")
    if not gaia.available:
        notes.append("The Gaia catalogue did not answer, so known stars could not be told apart from new ones.")
    if not simbad.available:
        notes.append("SIMBAD did not answer, so known objects are not named.")

    already = {(round(c["ra"], 4), round(c["dec"], 4)) for c in candidates}
    interior = lambda x, y: 2.5 * native <= x <= n - 1 - 2.5 * native and 2.5 * native <= y <= n - 1 - 2.5 * native  # noqa: E731

    if have_ir and gaia.available:
        for s in sources:
            if (round(s["ra"], 4), round(s["dec"], 4)) in already:
                continue
            seen_often = int((s["pass_snr"] >= 4.0).sum())
            solid = s["snr"] >= 8.0 and seen_often >= min(2, field["passes"]) and not s["edge"] and not s["near_saturated"] and not s["saturated"]
            if not solid:
                continue
            # (a) in no catalogue at all. Nothing catalogued may lie just beyond the matching radius either:
            # two catalogued neighbours merge into one source here, centred between their catalogue positions.
            wide = 12.0 / scale
            merged = bool(
                gaia.within(s["cx"], s["cy"], wide) or twomass.within(s["cx"], s["cy"], wide) or allwise.within(s["cx"], s["cy"], wide)
            )
            if not (s["gaia"] or s["simbad"] or s["twomass"] or s["allwise"]) and merged:
                continue
            if not (s["gaia"] or s["simbad"] or s["twomass"] or s["allwise"]):
                c = base(s, "uncatalogued")
                c["title"] = "Not in any catalogue"
                red = _redness(s)
                c["facts"].append(
                    f"Seen in {seen_often} of {field['passes']} passes at {s['snr']:.0f} times the noise, with no SIMBAD entry within "
                    f"{MATCH_ARCSEC:.1f}″ and no Gaia, 2MASS or AllWISE entry within 12″."
                )
                if red:
                    c["facts"].append(red)
                c["strength"] = float(s["snr"])
                c["verdict"] = "doubtful" if (c["caveats"] or crowded) else "unexplained"
                c["explanation"] = "No catalogue explains this source." + (
                    " The field is crowded, so a faint uncatalogued star is the likeliest answer." if crowded else ""
                )
                candidates.append(c)
                continue
            # (b) in the infrared catalogues, not in Gaia, and not where those catalogues put it
            # Only for a source that stands alone: two faint catalogued sources a few arcseconds apart
            # merge into one here, and its centre then lies between them, not on either.
            alone = (
                s["crowd"] == 0 and len(allwise.within(s["cx"], s["cy"], 12.0 / scale)) == 1
                and (s["neighbour_arcsec"] is None or s["neighbour_arcsec"] >= 15.0)
            )
            if s["gaia"] is None and s["allwise"] is not None and s["snr"] >= 15.0 and alone:
                offset = s["allwise_px"] * scale
                consistent = True
                if s["twomass"] is not None:
                    # Real motion carries on in a straight line: the 1999 position must lie further back along it.
                    ax, ay = allwise.position(s["allwise"])
                    tx, ty = twomass.position(s["twomass"])
                    stretch = (year - (s["twomass"].get("epoch") or 1999.5)) / max(year - 2010.5, 1.0)
                    px, py = s["cx"] + (ax - s["cx"]) * stretch, s["cy"] + (ay - s["cy"]) * stretch
                    consistent = math.hypot(tx - px, ty - py) * scale <= 2.5
                c = base(s, "drifted")
                c["_wise"] = {key: s["allwise"].get(key) for key in ("ra", "dec", "w1", "w2")}
                if offset >= DRIFT_MIN_ARCSEC and consistent and not c["caveats"]:
                    rate = offset / max(year - 2010.5, 1.0)
                    c["title"] = f"{offset:.1f}″ from its 2010 position"
                    c["facts"].append(
                        f"It is {offset:.1f}″ from where the AllWISE catalogue put it in 2010. If that is motion, it is about "
                        f"{rate:.2f}″ per year. Gaia has no entry for it."
                        + (" Its 1999 position in 2MASS lies further back along the same line." if s["twomass"] is not None else "")
                    )
                    c["measured"] = {"offset_arcsec": round(offset, 2), "rate": round(rate, 3)}
                    c["strength"] = offset / 0.6
                    catalogue_pm = _motion_of(s["simbad"])
                    if catalogue_pm is not None and catalogue_pm >= 0.5 * rate:
                        c["verdict"] = "known"
                        c["known"] = {"name": c["name"], "type": s["simbad"].get("kind")}
                        c["explanation"] = f"Already known: {c['name']} has a catalogued motion of {catalogue_pm:.2f}″ per year."
                    elif s["simbad"] and _is_extended(s["simbad"]):
                        c["verdict"] = "doubtful"
                        c["explanation"] = f"Probably not motion: SIMBAD lists {c['name']} as \"{s['simbad']['kind']}\", an extended object whose centre each telescope measures differently."
                    else:
                        c["verdict"] = "doubtful" if c["caveats"] else "unexplained"
                        c["explanation"] = "No catalogued motion explains the offset. Check it with \"Compare with 2010\"."
                    candidates.append(c)

        candidates.extend(_gone(field, allwise, simbad_2010, simbad, gaia, shift, sky, interior, already))

    candidates, tally = _ordered(candidates)
    counts = {"sources": len(sources), **tally}
    if crowded:
        notes.append(
            f"This is a crowded field: a random spot has a {chance * 100:.0f}% chance of lying within {MATCH_ARCSEC:.1f}″ of a Gaia star, "
            "so \"not in any catalogue\" is a weak test here."
        )
    if field["passes"] < 2:
        notes.append("Only one survey pass covers this field so far, so nothing can be said about motion or change.")
    return {
        "candidates": candidates, "counts": counts, "notes": notes, "crowded": crowded,
        "position_shift_arcsec": round(float(np.hypot(*shift) * scale), 2),
        "catalogues": {
            "gaia": len(gaia.objects) if gaia.available else None, "simbad": len(simbad.objects) if simbad.available else None,
            "twomass": len(twomass.objects) if twomass.available else None, "allwise": len(allwise.objects) if allwise.available else None,
        },
    }


def _ordered(candidates: list[dict]) -> tuple[list[dict], dict]:
    """Most interesting first, numbered, with the tally by verdict."""
    order = {"unexplained": 0, "known": 1, "doubtful": 2}
    candidates = sorted(candidates, key=lambda c: (order[c["verdict"]], -float(c.get("strength", 0.0))))
    for index, c in enumerate(candidates, start=1):
        c["id"] = index
        c["strength"] = round(float(c.get("strength", 0.0)), 1)
    return candidates, {verdict: sum(1 for c in candidates if c["verdict"] == verdict) for verdict in order}


def _redness(s: dict) -> str | None:
    """Say so when a source shows only at the longest wavelengths (the mark of a cold object)."""
    red, green, blue = (float(v) if np.isfinite(v) else 0.0 for v in s["mean_flux"])
    if red > 0 and red > 3.0 * max(green, 0.0) and red > 3.0 * max(blue, 0.0):
        return "It shows almost only at 3.8–5.0 µm, the way a cold object does."
    return None


def _gone(field: dict, allwise: _Catalogue, simbad_2010: _Catalogue, simbad_now: _Catalogue, gaia: _Catalogue,
          shift: np.ndarray, sky, interior, already: set) -> list[dict]:
    """AllWISE sources of 2010 that should be plain to see, and are missing or far fainter now."""
    sources, scale, native = field["sources"], field["scale"], field["native"]
    match_px = MATCH_ARCSEC / scale
    # How bright a source of a given WISE magnitude appears here: calibrated on the field's own stars.
    pairs: dict[int, list[float]] = {0: [], 1: []}       # channel 0 (3.8-5.0 um) against W2, channel 1 (1.6-3.8 um) against W1
    for s in sources:
        a = s["allwise"]
        if a is None or s["saturated"] or s["crowd"] > 1:
            continue
        for channel, key in ((0, "w2"), (1, "w1")):
            value = s["mean_flux"][channel]
            if a.get(key) is not None and np.isfinite(value) and value > 10.0 * field["mean_noise"][channel]:
                pairs[channel].append(a[key] + 2.5 * math.log10(value))
    zero = {channel: float(np.median(v)) for channel, v in pairs.items() if len(v) >= 6}
    spread = {channel: max(0.12, _sigma(np.array(pairs[channel]))) for channel in zero}      # magnitudes
    if not zero:
        return []

    saturated = [s for s in sources if s["saturated"]]
    out = []
    for i, a in enumerate(allwise.objects):
        x, y = allwise.x[i] + shift[0], allwise.y[i] + shift[1]            # where it would be in this tool's frame
        if not interior(x, y):
            continue
        if any(math.hypot(x - s["x"], y - s["y"]) < s["r_sat"] + 4 * native for s in saturated):
            continue
        xi, yi = int(round(x)), int(round(y))
        if field["flagged_any"][max(0, yi - 3):yi + 4, max(0, xi - 3):xi + 4].any():
            continue
        if a.get("cc") and a["cc"][:2] != "00":
            continue                                                       # AllWISE itself marks it as a likely artefact
        now = field["mean_apertures"].at(xi, yi)[0]
        best = None
        for channel, key in ((0, "w2"), (1, "w1")):
            if channel not in zero or a.get(key) is None or not np.isfinite(now[channel]):
                continue
            expected = 10 ** ((zero[channel] - a[key]) / 2.5)
            clarity = expected / field["mean_noise"][channel]
            left = max(0.0, float(now[channel])) / expected
            deficit = -2.5 * math.log10(max(left, 0.01))                   # magnitudes fainter than expected
            if clarity >= GONE_MIN_SNR and left < GONE_MAX_FRACTION and deficit > 4.0 * spread[channel]:
                if best is None or clarity > best["clarity"]:
                    best = {"clarity": clarity, "left": left, "key": key, "mag": a[key],
                            "absent": bool(now[channel] < 3.0 * field["mean_noise"][channel])}
        if best is None:
            continue
        # The other WISE band must not contradict it: a source as bright as expected there has not gone.
        contradicted = False
        for channel, key in ((0, "w2"), (1, "w1")):
            if key == best["key"] or channel not in zero or a.get(key) is None or not np.isfinite(now[channel]):
                continue
            expected = 10 ** ((zero[channel] - a[key]) / 2.5)
            if expected / field["mean_noise"][channel] >= GONE_MIN_SNR and now[channel] > 0.6 * expected:
                contradicted = True
        if contradicted:
            continue
        ra, dec = sky(allwise.x[i], allwise.y[i])
        if (round(ra, 4), round(dec, 4)) in already:
            continue
        percent = max(1, round(best["left"] * 100))
        c = {
            "kind": "gone", "ra": ra, "dec": dec, "snr": round(float(best["clarity"]), 1), "name": a["name"],
            "title": "Here in 2010, not here now" if best["absent"] else f"Only {percent}% as bright as in 2010",
            "caveats": [], "known": None, "strength": float(best["clarity"]),
            "facts": [
                f"AllWISE recorded a source here in 2010 ({best['key'].upper()} = {best['mag']:.1f} mag). It should stand out at about "
                f"{best['clarity']:.0f} times the noise; " + (
                    "nothing is detected there now." if best["absent"] else f"what is there now is about {percent}% as bright."
                )
            ],
            "measured": {"expected_snr": round(float(best["clarity"]), 1), "fraction_left": round(float(best["left"]), 2)},
            "_wise": {key: a.get(key) for key in ("ra", "dec", "w1", "w2")},
        }
        mover = _principal(simbad_2010.within(allwise.x[i], allwise.y[i], 5.0 / scale), simbad_2010, allwise.x[i], allwise.y[i], scale)
        pm = _motion_of(mover)
        if mover is not None and pm is not None and pm >= 0.15:
            now_ra, now_dec = sky(*simbad_now.position(mover))
            c["verdict"] = "known"
            c["known"] = {"name": _clean_name(mover["name"]), "type": mover.get("kind"), "now_ra": now_ra, "now_dec": now_dec}
            c["explanation"] = (
                f"Already known: this was {_clean_name(mover['name'])}" + (f" ({mover['kind']})" if mover.get("kind") else "") +
                f", which moves {pm:.2f}″ per year and has since travelled {pm * (float(np.mean(field['years'])) - 2010.5):.0f}″."
            )
        else:
            here = _principal(simbad_now.within(allwise.x[i], allwise.y[i], match_px), simbad_now, allwise.x[i], allwise.y[i], scale)
            g, _ = gaia.nearest(allwise.x[i], allwise.y[i], match_px)
            if here is not None and _is_variable_type(here):
                c["verdict"] = "known"
                c["known"] = {"name": _clean_name(here["name"]), "type": here.get("kind")}
                c["explanation"] = (
                    f"Already known: SIMBAD lists {_clean_name(here['name'])} as \"{here.get('kind') or 'variable object'}\", "
                    "so it has simply faded."
                )
            elif g is not None or here is not None:
                c["verdict"] = "doubtful"
                c["explanation"] = (
                    "A catalogued object is still listed at this spot, so it has more likely faded than moved, or the "
                    "catalogues measure it differently. Worth a look in \"Compare with 2010\"."
                )
            else:
                c["verdict"] = "unexplained"
                c["explanation"] = (
                    "Nothing catalogued explains where it went: it has moved away, faded, or was never real. "
                    "Check it with \"Compare with 2010\"."
                )
        out.append(c)
    return out


# --------------------------------------------------------------------------- step 6: like for like with WISE
#
# Step 5 compares broad SPHEREx images with a catalogue made at 3.4 and 4.6 micron. Two things fool it:
#   - colour: a red object (a quasar, a galaxy) is fainter at 1.6-3.8 micron than a star of the same W1
#     brightness, so it looks as if it had faded;
#   - neighbours: two objects a pixel apart, one bluer than the other. This tool centres on the blue one,
#     WISE centred on the red one, and the gap looks like motion.
# Both go away when the measurement is made in WISE's own wavelength range, exposure by exposure.


def _band_flux(points: list[dict], lo: float, hi: float) -> tuple[float, float] | None:
    """Mean brightness (mJy) and its error from the unflagged measurements between two wavelengths."""
    use = [(p["flux"], p["err"]) for p in points
           if lo <= p["wl"] <= hi and not p["flagged"] and p["err"] > 0 and math.isfinite(p["flux"])]
    if len(use) < 6:
        return None
    flux, err = (np.array(v, dtype=float) for v in zip(*use))
    keep = np.abs(flux - np.median(flux)) <= 4.0 * np.maximum(err, _sigma(flux))       # one bad exposure must not decide it
    flux, err = flux[keep], err[keep]
    if len(flux) < 6:
        return None
    weight = 1.0 / err ** 2
    mean = float((weight * flux).sum() / weight.sum())
    scatter = float((weight * (flux - mean) ** 2).sum() / (len(flux) - 1))             # 1 when the errors are honest
    return mean, float(math.sqrt(max(1.0, scatter) / weight.sum()))


def _wise_like(job, exposures, ra: float, dec: float) -> dict[str, tuple[float, float]]:
    """Brightness at one position in the SPHEREx ranges that match WISE W1 and W2: {band: (mJy, error)}."""
    points = []
    for ed in exposures:
        if ed.exp.detector < 4:
            continue                                               # both ranges lie on detectors 4 to 6
        point = photometry.measure(ed, job, ra, dec)
        if point is not None:
            points.append(point)
    out = {}
    for key, (lo, hi, _) in WISE_LIKE.items():
        measured = _band_flux(points, lo, hi)
        if measured is not None:
            out[key] = measured
    return out


def _catalogue_mjy(mag: float, key: str) -> float:
    return WISE_LIKE[key][2] * 1000.0 * 10 ** (-mag / 2.5)


def _yardstick(field: dict, job, exposures) -> tuple[list[dict], dict[str, float]]:
    """Well-measured, isolated stars that Gaia and AllWISE both list, and how this tool's brightness
    compares with AllWISE's for them in each range (close to 1)."""
    scale = field["scale"]
    stars = [
        s for s in field["sources"]
        if s.get("gaia") and s.get("allwise") and not s["saturated"] and not s["near_saturated"] and not s["edge"]
        and s["crowd"] <= 1 and s["snr"] >= 20.0 and s["allwise_px"] * scale <= 2.5
        and (s["neighbour_arcsec"] is None or s["neighbour_arcsec"] >= 15.0)
    ]
    stars = sorted(stars, key=lambda s: -s["snr"])[:MAX_YARDSTICK]
    ratios: dict[str, list[float]] = {key: [] for key in WISE_LIKE}
    for s in stars:
        s["_wise_like"] = _wise_like(job, exposures, s["ra"], s["dec"])
        for key, (flux, err) in s["_wise_like"].items():
            if s["allwise"].get(key) is not None and flux >= 10.0 * err:
                ratios[key].append(flux / _catalogue_mjy(s["allwise"][key], key))
    return stars, {key: float(np.median(v)) for key, v in ratios.items() if len(v) >= 3}


def _fractions(now: dict, wise: dict, scale_by: dict) -> dict[str, tuple[float, float, float]]:
    """Per WISE band: the fraction of the 2010 brightness there now, its error, and how
    clearly the 2010 brightness would be seen (in units of the measurement error)."""
    out = {}
    for key, (flux, err) in now.items():
        if wise.get(key) is None:
            continue
        expected = _catalogue_mjy(wise[key], key) * scale_by.get(key, 1.0)
        out[key] = (flux / expected, err / expected, expected / err)
    return out


def _has_faded(fractions: dict[str, tuple[float, float, float]], cold: bool = False) -> str | None:
    """The band that shows the source is fainter than in 2010, or None if that is not established.

    cold: in 2010 nearly all of its light was in W2 (a brown dwarf's colour). Light that is
    still there at W1 then belongs to something else, and does not show that it stayed.
    """
    usable = {key: v for key, v in fractions.items() if v[2] >= 6.0}
    if not usable:
        return None
    key = max(usable, key=lambda k: usable[k][2])
    left, err, _ = usable[key]
    if left + 3.0 * err > FADED_MAX_FRACTION:
        return None
    if cold and key == "w2":
        return key
    for other, (left2, err2, clarity) in usable.items():
        if other != key and clarity >= 8.0 and left2 - 2.0 * err2 > FADED_MAX_FRACTION:
            return None                                            # as bright as ever in the other band: it has not gone
    return key


def _has_moved(old: tuple[float, float], new: tuple[float, float], leak: float) -> bool:
    """True when the light now sits at the present position and not at the 2010 one.

    old, new: (brightness, error) at the two positions, same wavelength range.
    leak: the share of a star's light still measured that far from its centre.
    """
    (f_old, e_old), (f_new, e_new) = old, new
    if f_new < 8.0 * e_new:
        return False
    if f_new - f_old < 3.0 * math.hypot(e_old, e_new):
        return False
    return f_old / f_new <= leak + 0.35 * (1.0 - leak)


def _leak(job, exposures, stars: list[dict], key: str, east: float, north: float) -> tuple[float, int]:
    """Share of an ordinary star's light still measured (east, north) arcseconds from its centre,
    and how many stars that rests on. Falls back to a Gaussian of the nominal width."""
    ratios = []
    for s in stars:
        centre = s.get("_wise_like", {}).get(key)
        if centre is None or centre[0] < 15.0 * centre[1]:
            continue
        ra = s["ra"] + east / 3600.0 / max(math.cos(math.radians(s["dec"])), 1e-6)
        aside = _wise_like(job, exposures, ra, s["dec"] + north / 3600.0).get(key)
        if aside is not None:
            ratios.append(aside[0] / centre[0])
    if len(ratios) >= 3:
        return float(np.median(ratios)), len(ratios)
    sigma = PSF_FWHM_ARCSEC / 2.355
    return float(math.exp(-(east ** 2 + north ** 2) / (4.0 * sigma ** 2))), 0


def check_against_wise(candidates: list[dict], field: dict, job, exposures) -> tuple[list[dict], int, int]:
    """Re-measure every candidate that rests on the 2010 AllWISE catalogue, in WISE's own
    wavelength ranges. Returns the candidates that stand, how many were checked, and how many fell."""
    pending = sorted((c for c in candidates if c.get("_wise")), key=lambda c: -float(c.get("strength", 0.0)))
    if not pending:
        return candidates, 0, 0
    stars, scale_by = _yardstick(field, job, exposures)
    scaled = (
        f" The comparison is scaled by {len(stars)} ordinary stars in this field." if scale_by
        else " Too few ordinary stars were available to check the brightness scale in this field."
    )
    fallen = set()
    for index, c in enumerate(pending):
        wise = c["_wise"]
        if index >= MAX_CONFIRM:
            fallen.add(id(c))
            continue
        if c["kind"] == "gone":
            now = _wise_like(job, exposures, wise["ra"], wise["dec"])
            fractions = _fractions(now, wise, scale_by)
            colour = wise["w1"] - wise["w2"] if wise.get("w1") is not None and wise.get("w2") is not None else None
            cold = colour is not None and colour >= COLD_COLOUR_MAG
            key = _has_faded(fractions, cold)
            if key is None:
                fallen.add(id(c))
                continue
            left, err, clarity = fractions[key]
            lo, hi, _ = WISE_LIKE[key]
            absent = now[key][0] < 3.0 * now[key][1]
            percent = max(1, round(left * 100))
            c["title"] = "Here in 2010, not here now" if absent else f"Only {percent}% as bright as in 2010"
            c["facts"] = [
                f"AllWISE recorded a source here in 2010 ({key.upper()} = {wise[key]:.1f} mag). Measured now at the same "
                f"wavelengths ({lo:.1f}–{hi:.1f} µm), " + (
                    "nothing is detected there." if absent
                    else f"it is about {percent}% as bright (give or take {max(1, round(err * 100))}%)."
                ) + scaled
            ]
            stayed = fractions.get("w1")
            if cold and key == "w2" and stayed and stayed[2] >= 6.0 and stayed[0] - 2.0 * stayed[1] > FADED_MAX_FRACTION:
                c["facts"].append(
                    f"In 2010 it was {colour:.1f} magnitudes brighter at 4.6 µm than at 3.4 µm, the colour of a cold object. "
                    "The light still seen here at 2.8–3.8 µm therefore belongs to something else, such as a background star."
                )
            c["measured"] = {
                "band": key, "fraction_left": round(float(left), 2), "fraction_err": round(float(err), 2),
                "expected_snr": round(float(clarity), 1),
            }
            c["snr"] = round(float(clarity), 1)
            c["strength"] = (1.0 - left) / max(err, 1e-6)
        else:
            new = _wise_like(job, exposures, c["ra"], c["dec"])
            old = _wise_like(job, exposures, wise["ra"], wise["dec"])
            keys = [key for key in new if key in old and wise.get(key) is not None]
            if not keys:
                fallen.add(id(c))
                continue
            key = max(keys, key=lambda k: new[k][0] / new[k][1])
            east = (wise["ra"] - c["ra"]) * math.cos(math.radians(c["dec"])) * 3600.0
            north = (wise["dec"] - c["dec"]) * 3600.0
            leak, n_stars = _leak(job, exposures, stars, key, east, north)
            same = 0.4 <= new[key][0] / (_catalogue_mjy(wise[key], key) * scale_by.get(key, 1.0)) <= 2.5
            if not (same and _has_moved(old[key], new[key], leak)):
                fallen.add(id(c))
                continue
            lo, hi, _ = WISE_LIKE[key]
            share = max(0, round(100 * old[key][0] / new[key][0]))
            c["facts"].append(
                f"Checked at WISE's own wavelengths ({lo:.1f}–{hi:.1f} µm): the 2010 position now holds {share}% of the light "
                f"measured at the present position. " + (
                    f"The same step away from {n_stars} ordinary stars in this field leaves {round(100 * leak)}%."
                    if n_stars else f"A single star would leave about {round(100 * leak)}% there."
                )
            )
            c["measured"].update({"band": key, "old_share": round(old[key][0] / new[key][0], 2), "star_share": round(leak, 2)})
    for s in stars:
        s.pop("_wise_like", None)
    kept = [c for c in candidates if id(c) not in fallen]
    for c in kept:
        c.pop("_wise", None)
    return kept, len(pending), len(fallen)


# --------------------------------------------------------------------------- the whole search


def search(job, exposures, header: dict, data: np.ndarray, flags: np.ndarray,
           gaia_stars: list[dict] | None, cats: dict[str, list[dict] | None]) -> dict:
    """Run every step on one field. Blocking: call it from a worker thread."""
    field = measure_field(header, data, flags)
    hits = screen(field)
    confirmed, dropped = confirm(field, hits, job, exposures)
    report = explain(field, confirmed, gaia_stars, cats)
    kept, checked, fell = check_against_wise(report["candidates"], field, job, exposures)
    kept, tally = _ordered(kept)
    report["candidates"] = kept[:MAX_REPORTED]
    report["counts"].update(tally)
    report["counts"]["first_pass_flags"] = len(hits) + checked
    report["counts"]["not_confirmed"] = dropped + fell
    report["field"] = {
        "ra": field["ra"], "dec": field["dec"], "fov_arcmin": field["fov"], "passes": field["passes"],
        "dates": [m["date"] for m in field["meta"]],
        "baseline_yr": round(float(field["years"].max() - field["years"].min()), 2),
    }
    report["method"] = (
        "Sources are found in the combined image, then measured in every survey pass. Anything that appears to move "
        f"(at least {SCREEN_MOTION_ARCSEC_YR}″ per year) or change brightness (at least {int(SCREEN_CHANGE_FRACTION * 100)}%) is "
        "measured again exposure by exposure, and kept only if that confirms it. The catalogues (Gaia DR3, SIMBAD, 2MASS, "
        "AllWISE) are consulted last, to say what is already known; anything that differs from the 2010 AllWISE catalogue is "
        "then measured again at WISE's own wavelengths, so that colour is not mistaken for change. These are candidates to "
        "look at, not discoveries."
    )
    return report


# --------------------------------------------------------------------------- with the archive and the cache

_in_flight: dict[str, asyncio.Task] = {}


async def _quietly(awaitable, timeout: float):
    """The result, or None if the service is slow or down (the report then says what was not checked)."""
    try:
        return await asyncio.wait_for(awaitable, timeout)
    except Exception as err:                    # noqa: BLE001 - any failure means "not available"
        log.info("catalogue unavailable for the candidate search: %s", err.__class__.__name__)
        return None


async def _run(client, job, path: Path) -> dict:
    exposures = job.band_exposures(None)
    payload = await payloads.frames(job, "colour", "epochs", True, exposures, True)
    header, data, flags = process.unpack_frames(payload)
    ra, dec, fov = round(job.ra, 4), round(job.dec, 4), job.fov_arcmin
    stars, simbad, twomass, allwise = await asyncio.gather(
        _quietly(gaia.cone(client, ra, dec, fov), 150.0),
        _quietly(catalogs.lookup(client, "simbad", ra, dec, fov), 60.0),
        _quietly(catalogs.lookup(client, "twomass", ra, dec, fov), 300.0),
        _quietly(catalogs.lookup(client, "allwise", ra, dec, fov), 300.0),
    )
    cats = {
        "simbad": simbad["objects"] if simbad else None,
        "twomass": twomass["objects"] if twomass else None,
        "allwise": allwise["objects"] if allwise else None,
    }
    started = time.time()
    report = await asyncio.to_thread(search, job, exposures, header, data, flags, stars["stars"] if stars else None, cats)
    report["seconds"] = round(time.time() - started, 1)
    report["n_exposures"] = len(exposures)
    # Keep the result only if every catalogue answered; otherwise try again next time.
    if all(v is not None for v in report["catalogues"].values()):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(report, allow_nan=False, default=float), "utf-8")
            tmp.replace(path)
        except (OSError, ValueError) as err:
            log.warning("could not store candidates %s: %s", path, err)
    return report


async def run(client, job, fresh: bool = False) -> dict:
    """Candidates for a fully loaded field. Stored on disk; concurrent callers share one search."""
    exposures = job.band_exposures(None)
    name = f"{job.key}_{payloads.signature(exposures)}_v{RECIPE_VERSION}"
    path = config.CACHE_DIR / "candidates" / f"{name}.json"
    if not fresh:
        try:
            return json.loads(path.read_text("utf-8"))
        except (OSError, ValueError):
            pass
    task = _in_flight.get(name)
    if task is None:
        task = asyncio.create_task(_run(client, job, path))
        _in_flight[name] = task
        task.add_done_callback(lambda _t: _in_flight.pop(name, None))
    return await asyncio.shield(task)
