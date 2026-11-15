"""Quick-look spectrum and light curve of one sky position.

Every SPHEREx exposure samples a source at a different wavelength, so measuring
the source's flux in each exposure gives its spectrum directly. Comparing the
same wavelengths between survey passes gives a light curve per detector band.

Spectrum: PSF-weighted forced photometry (Gaussian of the per-exposure FWHM in
the file header) with a local annulus sky. If the target has a known proper
motion the aperture follows it from exposure to exposure.

Light curve: each measurement is divided by the typical brightness of that
source at the same wavelength (a running median over all passes), so a change of
sampled wavelength between passes is not mistaken for variability. Stars bright
enough to saturate the detector core are followed through their unsaturated
outer wings instead, which scale with the star's brightness.

These are quick-look values. They are not the SPHEREx project's calibrated
spectrophotometry (IRSA's Spectrophotometry Tool should be used for science).

Developed with AI assistance (Claude, Anthropic). See README "Use of AI".
"""
from __future__ import annotations

import math
import warnings

import numpy as np

from . import config, discovery
from .process import robust_background
from .resolver import propagate
from .stack import ExposureData, StackJob

BOX_HALF = 3                  # 7x7 native pixels around the source
ANNULUS_IN, ANNULUS_OUT = 4.0, 8.0
MJY_PER_MJYSR_ARCSEC2 = config.ARCSEC2_TO_SR * 1.0e9      # (MJy/sr) * arcsec^2 -> mJy
MIN_POINTS_PER_PASS = 3       # a pass needs this many measurements in a band to be plotted
# Pass-to-pass position systematics that averaging cannot remove: parallax, unresolved
# companions, saturation, and astrometric differences between data releases. Measured
# on RW Aurigae (a star with no real motion at this precision), which otherwise shows
# a spurious 0.5 arcsec/yr.
POSITION_FLOOR_ARCSEC = 0.3
BLEND_RADIUS_ARCSEC = 7.0     # a catalogued neighbour this close shares the target's PSF (pixels are 6.15 arcsec)
BLEND_RADIUS_FAINT_ARCSEC = 12.5   # used when the target is too faint or red to be a Gaia star itself
SELF_MATCH_ARCSEC = 2.5       # a Gaia star that stays this close in every exposure is the target itself
WING_WIDTH = 3.0              # native pixels
METHOD_TEXT = (
    "PSF-weighted forced photometry (Gaussian, per-exposure width), local annulus sky. "
    "Light curves compare like wavelengths between passes; saturated stars are followed "
    "through their unsaturated outer wings. Quick look only."
)


def _psf_weights(x: float, y: float, sigma: float, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Fraction of a point source's flux that lands in each pixel.

    The width in the file header (PSF_FWHM) describes the effective PSF: the
    optical PSF already smoothed by the pixel. Pixel values are therefore that
    profile sampled at pixel centres. Integrating it over each pixel as well would
    count the pixel smoothing twice; an earlier version did, and read stars about
    14% too bright against 2MASS and WISE (see scripts/validate_flux.py).
    """
    gx = np.exp(-0.5 * ((xs - x) / sigma) ** 2)
    gy = np.exp(-0.5 * ((ys - y) / sigma) ** 2)
    return np.outer(gy, gx) / (2.0 * math.pi * sigma * sigma)


def _saturation_radius(bad: np.ndarray, rr: np.ndarray, max_radius: int = 24) -> float:
    """Radius (native pixels) out to which the source's core is flagged; 0 if the core is clean."""
    limit = int(min(max_radius, rr.max()))
    fractions = []
    for k in range(limit):
        ring = (rr >= k) & (rr < k + 1)
        fractions.append(float(bad[ring].mean()) if ring.any() else 0.0)
    if not fractions or max(fractions[:2]) < 0.5:
        return 0.0
    for k in range(1, len(fractions) - 1):
        if fractions[k] < 0.25 and fractions[k + 1] < 0.25:
            return float(k)
    return float(limit)


def _centroid_offset(
    ed: ExposureData, data: np.ndarray, x: float, y: float, sky_sigma: float, ra_ref: float, dec_ref: float
) -> tuple[float | None, float | None]:
    """Offset (east, north; arcsec) of the source's measured centre from a fixed reference position.

    The centre is the weighted mean of the positive pixels round the brightest
    peak within two pixels of the expected position. It must be a peak in its own
    right: a pixel on the slope of a brighter neighbour just outside the search
    window is not this source, and pixels are weighted down away from the peak,
    so a neighbour cannot drag the centre towards itself. Single-exposure centres
    are rough because SPHEREx undersamples its PSF; the motion fit uses the
    median of many of them per pass.
    """
    ny, nx = data.shape
    xi, yi = int(round(x)), int(round(y))
    x0, x1 = max(0, xi - 2), min(nx, xi + 3)
    y0, y1 = max(0, yi - 2), min(ny, yi + 3)
    window = np.where(np.isfinite(data[y0:y1, x0:x1]), data[y0:y1, x0:x1], -np.inf)
    if window.size == 0 or not np.isfinite(window.max()) or window.max() < 5.0 * sky_sigma:
        return None, None
    peak = None
    for flat in np.argsort(window, axis=None)[::-1]:
        wy, wx = np.unravel_index(int(flat), window.shape)
        value = window[wy, wx]
        if value < 5.0 * sky_sigma:
            break
        cx_, cy_ = wx + x0, wy + y0
        hood = data[max(0, cy_ - 1):min(ny, cy_ + 2), max(0, cx_ - 1):min(nx, cx_ + 2)]
        if value >= np.nanmax(hood):
            peak = (cx_, cy_)
            break
    if peak is None:
        return None, None
    px, py = peak
    bx0, bx1 = max(0, px - 2), min(nx, px + 3)
    by0, by1 = max(0, py - 2), min(ny, py + 3)
    box = data[by0:by1, bx0:bx1]
    ys, xs = np.mgrid[by0:by1, bx0:bx1]
    weight = np.where(np.isfinite(box) & (box > 0), box, 0.0) * np.exp(-((xs - px) ** 2 + (ys - py) ** 2) / 2.0)
    total = float(weight.sum())
    if total <= 0:
        return None, None
    cx, cy = float((weight * xs).sum() / total), float((weight * ys).sum() / total)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ra_c, dec_c = ed.wcs.all_pix2world(cx, cy, 0)
    d_ra = ((float(ra_c) - ra_ref + 540.0) % 360.0) - 180.0
    east = d_ra * math.cos(math.radians(dec_ref)) * 3600.0
    north = (float(dec_c) - dec_ref) * 3600.0
    return east, north


def _measured_motion(points: list[dict]) -> dict | None:
    """Fit a straight-line motion to the per-pass median positions (arcsec per year)."""
    usable = [p for p in points if p.get("east") is not None and p["snr"] >= 8.0]
    passes = []
    for ep in sorted({p["epoch"] for p in usable}):
        group = [p for p in usable if p["epoch"] == ep]
        if len(group) < 5:
            continue
        east = np.array([p["east"] for p in group])
        north = np.array([p["north"] for p in group])
        me, mn = float(np.median(east)), float(np.median(north))
        scale = 1.2533 * 1.4826 / math.sqrt(len(group))
        passes.append({
            "epoch": int(ep), "n": len(group),
            "year": float(np.mean([discovery.mjd_to_decimal_year(p["mjd"]) for p in group])),
            "east": me, "north": mn,
            "east_err": math.hypot(POSITION_FLOOR_ARCSEC, scale * float(np.median(np.abs(east - me)))),
            "north_err": math.hypot(POSITION_FLOOR_ARCSEC, scale * float(np.median(np.abs(north - mn)))),
        })
    if len(passes) < 2:
        return None
    t = np.array([p["year"] for p in passes])
    baseline = float(t.max() - t.min())
    if baseline < 0.25:
        return None

    def slope(values: np.ndarray, errors: np.ndarray) -> tuple[float, float]:
        w = 1.0 / errors ** 2
        t0 = float((w * t).sum() / w.sum())
        sxx = float((w * (t - t0) ** 2).sum())
        rate = float((w * (t - t0) * values).sum() / sxx)
        return rate, 1.0 / math.sqrt(sxx)

    pm_e, err_e = slope(np.array([p["east"] for p in passes]), np.array([p["east_err"] for p in passes]))
    pm_n, err_n = slope(np.array([p["north"] for p in passes]), np.array([p["north_err"] for p in passes]))
    total = math.hypot(pm_e, pm_n)
    total_err = math.hypot(pm_e * err_e, pm_n * err_n) / total if total > 0 else math.hypot(err_e, err_n)
    return {
        "pm_east": pm_e, "pm_north": pm_n, "err_east": err_e, "err_north": err_n,
        "total": total, "total_err": total_err,
        "significant": bool(total >= 4.0 * total_err and total >= 1.0),
        "pa_deg": (math.degrees(math.atan2(pm_e, pm_n)) + 360.0) % 360.0,
        "baseline_yr": baseline, "passes": passes,
    }


def measure(ed: ExposureData, job: StackJob, ra: float, dec: float, motion: dict | None = None) -> dict | None:
    """Flux of the source at (ra, dec) in one exposure, or None if it is off the cutout.

    motion: optional {"pmra", "pmdec", "epoch"}; (ra, dec) is then the position at
    ``epoch`` (decimal year) and is moved to the date of this exposure.
    """
    ra_ref, dec_ref = ra, dec
    if motion:
        ra, dec = propagate(ra, dec, motion["pmra"], motion["pmdec"], motion["epoch"], discovery.mjd_to_decimal_year(ed.mjd))
    sci = ed.cut.sci
    ny, nx = sci.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x, y = ed.wcs.all_world2pix(ra, dec, 0, quiet=True)
    x, y = float(x), float(y)
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    if x < 1.5 or y < 1.5 or x > nx - 2.5 or y > ny - 2.5:
        return None

    bad = ((ed.cut.flags & job.bad_mask) != 0) | ~np.isfinite(sci)

    # Local sky from an annulus around the source.
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.hypot(xx - x, yy - y)
    annulus = (rr >= ANNULUS_IN) & (rr <= ANNULUS_OUT)
    sky, sky_sigma = robust_background(sci, annulus & ~bad)
    if sky_sigma <= 0:
        _, sky_sigma = robust_background(sci, ~bad)

    x0, x1 = max(0, int(round(x)) - BOX_HALF), min(nx, int(round(x)) + BOX_HALF + 1)
    y0, y1 = max(0, int(round(y)) - BOX_HALF), min(ny, int(round(y)) + BOX_HALF + 1)
    sigma = float(np.clip(ed.psf_fwhm_arcsec / 2.3548 / config.NATIVE_PIXEL_ARCSEC, 0.40, 1.3))
    psf = _psf_weights(x, y, sigma, np.arange(x0, x1), np.arange(y0, y1))
    data = sci[y0:y1, x0:x1] - sky
    box_bad = bad[y0:y1, x0:x1]
    box_finite = np.isfinite(sci[y0:y1, x0:x1])

    good = ~box_bad
    reliable = float(psf[good].sum()) >= 0.5          # most of the PSF falls on trustworthy pixels
    use = good if reliable else box_finite
    denom = float((psf[use] ** 2).sum())
    if denom <= 0:
        return None
    flux_pix = float((psf[use] * data[use]).sum() / denom)              # MJy/sr * pixel
    var = ed.cut.variance[y0:y1, x0:x1]
    var = np.where(np.isfinite(var) & (var > 0), var, sky_sigma ** 2)
    var = np.maximum(var, sky_sigma ** 2)
    err_pix = float(math.sqrt((psf[use] ** 2 * var[use]).sum()) / denom)

    to_mjy = ed.omega_arcsec2 * MJY_PER_MJYSR_ARCSEC2
    core = psf > 0.05
    flags_or = int(np.bitwise_or.reduce(ed.cut.flags[y0:y1, x0:x1][core])) if core.any() else 0

    table = job.wave.get(ed.exp.detector)
    if table is not None:
        wl, bw = table.evaluate(ed.cut.header, x, y)
        wl, bw = float(wl), float(bw)
    else:
        lo, hi = config.DETECTOR_RANGES_UM[ed.exp.detector]
        wl, bw = 0.5 * (lo + hi), None

    flux = flux_pix * to_mjy
    err = err_pix * to_mjy
    east, north = _centroid_offset(ed, sci - sky, x, y, sky_sigma, ra_ref, dec_ref)
    return {
        "_ra": ra, "_dec": dec,   # where the aperture was for this exposure (differs per exposure when tracking)
        "blend_sep": None, "blend_g": None, "blended": False,
        "east": east,            # where the source actually is, relative to the reference position (arcsec)
        "north": north,
        "obs_id": ed.exp.obs_id,
        "detector": ed.exp.detector,
        "epoch": ed.epoch,
        "mjd": ed.mjd,
        "date": discovery.mjd_to_iso_date(ed.mjd),
        "wl": wl,
        "bw": bw,
        "flux": flux,
        "err": err,
        "snr": flux / err if err > 0 else 0.0,
        "flagged": not reliable,
        "flags": flags_or,
        "flag_names": job.flag_names(flags_or & job.bad_mask),
        "fwhm": ed.psf_fwhm_arcsec,
        # Kept for the wing measurement of saturated stars; removed before the reply is sent.
        "_sat_radius": 0.0 if reliable else _saturation_radius(bad, rr),
        "_geometry": (sci, bad, rr),
    }


def _wing_brightness(sci: np.ndarray, bad: np.ndarray, rr: np.ndarray, r_in: float, r_out: float) -> float | None:
    """Median surface brightness in an annulus beyond the saturated core, above the far sky."""
    ring = (rr >= r_in) & (rr <= r_out)
    use = ring & ~bad
    if use.sum() < 12 or use.sum() < 0.5 * ring.sum():
        return None
    far = (rr >= r_out + 6.0) & ~bad
    sky = float(np.median(sci[far])) if far.sum() >= 40 else robust_background(sci, ~bad)[0]
    value = float(np.median(sci[use])) - sky
    return value if value > 0 else None


def _relative_series(wl: np.ndarray, value: np.ndarray, epoch: np.ndarray, mjd: np.ndarray) -> list[dict]:
    """Brightness of each pass relative to the all-pass average at the same wavelengths."""
    order = np.argsort(wl)
    wl, value, epoch, mjd = wl[order], value[order], epoch[order], mjd[order]
    half = max(2, len(value) // 12)
    template = np.array([np.median(value[max(0, i - half): i + half + 1]) for i in range(len(value))])
    ok = template > 0
    ratio = np.where(ok, value / np.where(ok, template, 1.0), np.nan)
    series = []
    for ep in np.unique(epoch):
        member = (epoch == ep) & np.isfinite(ratio)
        n = int(member.sum())
        if n < MIN_POINTS_PER_PASS:
            continue
        rel = float(np.median(ratio[member]))
        scatter = 1.4826 * float(np.median(np.abs(ratio[member] - rel)))
        series.append({
            "epoch": int(ep), "n": n,
            "mjd": float(np.mean(mjd[member])),
            "rel": rel,
            "rel_err": max(0.01, 1.2533 * scatter / math.sqrt(n)),
            "wl": float(np.mean(wl[member])),
        })
    # Normalise so the plotted passes average to 1.
    if series:
        mean = float(np.mean([s["rel"] for s in series]))
        if mean > 0:
            for s in series:
                s["rel"] /= mean
                s["rel_err"] /= mean
    return series


def _mark_blends(points: list[dict], stars: list[dict]) -> dict | None:
    """Flag measurements that have another catalogued star inside the PSF.

    SPHEREx pixels are 6 arcsec, so a neighbour a few arcseconds away is measured
    together with the target. Gaia stars are moved to the date of each exposure. A
    star that stays on the aperture in every exposure is taken to be the target
    itself and is ignored.
    """
    if not points or not stars:
        return None
    s_ra = np.array([s["ra"] for s in stars], dtype=float)
    s_dec = np.array([s["dec"] for s in stars], dtype=float)
    s_pmra = np.array([s["pmra"] or 0.0 for s in stars], dtype=float)
    s_pmdec = np.array([s["pmdec"] or 0.0 for s in stars], dtype=float)
    s_epoch = np.array([s["epoch"] or 2016.0 for s in stars], dtype=float)
    s_g = np.array([s["g"] if s["g"] is not None else np.nan for s in stars], dtype=float)

    years = np.array([discovery.mjd_to_decimal_year(p["mjd"]) for p in points])
    p_ra = np.array([p["_ra"] for p in points])
    p_dec = np.array([p["_dec"] for p in points])
    dt = years[:, None] - s_epoch[None, :]
    star_dec = s_dec[None, :] + s_pmdec[None, :] * dt / 3.6e6
    star_ra = s_ra[None, :] + s_pmra[None, :] * dt / 3.6e6 / np.maximum(np.cos(np.radians(s_dec[None, :])), 1e-6)
    d_ra = ((star_ra - p_ra[:, None] + 540.0) % 360.0 - 180.0) * np.cos(np.radians(p_dec[:, None]))
    sep = np.hypot(d_ra, star_dec - p_dec[:, None]) * 3600.0          # (points, stars), arcsec

    is_target = sep.max(axis=0) < SELF_MATCH_ARCSEC
    sep[:, is_target] = np.inf
    # If the target is itself a Gaia star, neighbours more than 3 magnitudes fainter
    # (under about 6% of its light) are not worth a warning.
    target_g = s_g[is_target]
    target_g = target_g[np.isfinite(target_g)]
    if target_g.size:
        sep[:, np.isfinite(s_g) & (s_g > float(target_g.min()) + 3.0)] = np.inf
        radius = BLEND_RADIUS_ARCSEC
    else:
        # Not a Gaia star itself, so it is faint or very red: even the wings of a
        # neighbour two pixels away can outshine it.
        radius = BLEND_RADIUS_FAINT_ARCSEC
    nearest = np.argmin(sep, axis=1)
    count = 0
    closest = None
    for i, p in enumerate(points):
        distance = float(sep[i, nearest[i]])
        if not math.isfinite(distance) or distance > radius:
            continue
        p["blend_sep"] = distance
        p["blend_g"] = None if not math.isfinite(s_g[nearest[i]]) else float(s_g[nearest[i]])
        p["blended"] = True
        count += 1
        if closest is None or distance < closest["sep"]:
            closest = {"sep": distance, "g": p["blend_g"]}
    if count == 0:
        return None
    return {"n": count, "closest_arcsec": closest["sep"], "g": closest["g"], "radius_arcsec": radius}


def _light_curve(points: list[dict]) -> list[dict]:
    light_curve: list[dict] = []
    for det in config.DETECTORS:
        band = [p for p in points if p["detector"] == det and not p["blended"]]
        if len(band) < 2 * MIN_POINTS_PER_PASS:
            continue
        clean = [p for p in band if not p["flagged"]]
        method = None
        if len(clean) >= max(2 * MIN_POINTS_PER_PASS, 0.5 * len(band)):
            if float(np.median([p["snr"] for p in clean])) < 3.0:
                continue                                     # too faint in this band to follow
            used = clean
            values = np.array([p["flux"] for p in clean])
            method = "psf"
        else:
            # Mostly flagged. If that is because the core saturates, use the wings.
            radii = np.array([p["_sat_radius"] for p in band if p["flagged"]])
            if radii.size == 0 or float(np.median(radii)) <= 0:
                continue
            r_in = float(np.ceil(np.percentile(radii, 85)) + 1.0)
            r_out = r_in + WING_WIDTH
            used, wing_values = [], []
            for p in band:
                wing = _wing_brightness(*p["_geometry"], r_in, r_out)
                if wing is not None:
                    used.append(p)
                    wing_values.append(wing)
            if len(used) < 2 * MIN_POINTS_PER_PASS:
                continue
            values = np.array(wing_values)
            method = "wings"
        series = _relative_series(
            np.array([p["wl"] for p in used]), values,
            np.array([p["epoch"] for p in used]), np.array([p["mjd"] for p in used]),
        )
        if len(series) < 2:
            continue
        for s in series:
            members = [p for p in used if p["epoch"] == s["epoch"]]
            s.update({
                "detector": det, "method": method,
                "date": discovery.mjd_to_iso_date(s["mjd"]),
                "flux": float(np.median([p["flux"] for p in members])) if method == "psf" else None,
            })
            light_curve.append(s)
    return light_curve


def _variability(light_curve: list[dict]) -> dict | None:
    best = None
    for det in config.DETECTORS:
        series = [p for p in light_curve if p["detector"] == det]
        if len(series) < 2:
            continue
        lo = min(series, key=lambda p: p["rel"])
        hi = max(series, key=lambda p: p["rel"])
        sigma = math.hypot(lo["rel_err"], hi["rel_err"])
        if sigma <= 0 or hi["rel"] <= 0:
            continue
        candidate = {
            "detector": det,
            "significance": (hi["rel"] - lo["rel"]) / sigma,
            "fraction": (hi["rel"] - lo["rel"]) / hi["rel"],
            "bright_date": hi["date"], "faint_date": lo["date"],
            "method": hi["method"],
        }
        if best is None or candidate["significance"] > best["significance"]:
            best = candidate
    return best


def extract(
    job: StackJob, ra: float, dec: float,
    exposures: list[ExposureData] | None = None, motion: dict | None = None,
    stars: list[dict] | None = None,
) -> dict:
    """Spectrum points plus a per-band, per-pass light curve for one position.

    stars: Gaia stars in the field (see gaia.cone), used to flag blended measurements.
    """
    if exposures is None:
        exposures = job.band_exposures(None)
    points = []
    for ed in exposures:
        point = measure(ed, job, ra, dec, motion)
        if point is not None:
            points.append(point)
    points.sort(key=lambda p: p["wl"])

    blend = _mark_blends(points, stars or [])
    light_curve = _light_curve(points)
    saturated = sum(1 for p in points if p["flagged"] and p["_sat_radius"] > 0)
    for p in points:
        for key in ("_geometry", "_sat_radius", "_ra", "_dec"):
            p.pop(key, None)

    good = [p for p in points if not p["flagged"]]
    detected = [p for p in good if p["snr"] >= 3.0]
    return {
        "ra": ra, "dec": dec,
        "tracked": bool(motion),
        "n_points": len(points),
        "n_flagged": len(points) - len(good),
        "n_saturated": saturated,
        "n_detected": len(detected),
        "median_snr": float(np.median([p["snr"] for p in good])) if good else 0.0,
        "points": points,
        "light_curve": light_curve,
        "variability": _variability(light_curve),
        "motion": _measured_motion([p for p in points if not p["blended"]]),
        "blend": blend,
        "flag_descriptions": job.flag_descriptions or config.DEFAULT_FLAG_DESCRIPTIONS,
        "method": METHOD_TEXT,
    }
