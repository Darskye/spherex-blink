"""Turn native SPHEREx cutouts into aligned, north-up frames for blinking.

Every exposure has its own roll angle and distortion, so each is resampled onto
one common tangent-plane grid (north up, east left). Frames are then either the
individual exposures or a coadd per survey pass ("epoch").

Developed with AI assistance (Claude, Anthropic). See README "Use of AI".
"""
from __future__ import annotations

import json
import struct
import warnings
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from astropy.wcs import WCS
from scipy import ndimage

from . import config, discovery
from .stack import ExposureData, StackJob

MAGIC = b"SPXB"
BAD_TOLERANCE = 1.0e-3       # largest flagged-pixel contribution an output pixel may have and still be trusted
MIN_EXPOSURES_PER_PASS = 3   # thinner passes are left out of the pass view when better ones exist


@dataclass(frozen=True)
class Grid:
    """Output pixel grid: GRID_N x GRID_N, row 0 = north edge, column 0 = east edge."""

    ra0: float
    dec0: float
    fov_arcmin: float
    n: int = config.GRID_N

    @property
    def scale_arcsec(self) -> float:
        return self.fov_arcmin * 60.0 / self.n

    @cached_property
    def wcs(self) -> WCS:
        w = WCS(naxis=2)
        centre = (self.n - 1) / 2.0 + 1.0          # FITS 1-based reference pixel
        w.wcs.crpix = [centre, centre]
        w.wcs.cdelt = [-self.scale_arcsec / 3600.0, -self.scale_arcsec / 3600.0]
        w.wcs.crval = [self.ra0, self.dec0]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        return w

    @cached_property
    def world(self) -> tuple[np.ndarray, np.ndarray]:
        cols, rows = np.meshgrid(np.arange(self.n), np.arange(self.n))
        ra, dec = self.wcs.all_pix2world(cols, rows, 0)
        return ra, dec


def robust_background(sci: np.ndarray, good: np.ndarray) -> tuple[float, float]:
    """Sigma-clipped median and scatter of the good pixels of a cutout."""
    values = sci[good]
    if values.size < 8:
        values = sci[np.isfinite(sci)]
    if values.size == 0:
        return 0.0, 0.0
    med = float(np.median(values))
    sig = float(1.4826 * np.median(np.abs(values - med)))
    for _ in range(4):
        if sig <= 0:
            break
        keep = np.abs(values - med) < 3.0 * sig
        if keep.sum() < 8:
            break
        values = values[keep]
        med = float(np.median(values))
        sig = float(1.4826 * np.median(np.abs(values - med)))
    return med, sig


def fill_saturated(sci: np.ndarray, flags: np.ndarray, saturation_mask: int) -> np.ndarray:
    """Give saturated star cores a value so they draw as bright, not as holes.

    The pipeline stores NaN where a pixel overflowed. For display only, such
    pixels take the brightest neighbouring value. They stay flagged, and
    photometry never uses this.
    """
    hole = ~np.isfinite(sci) & ((flags & saturation_mask) != 0)
    if not hole.any():
        return sci
    # Only holes inside a genuinely bright source are filled. The same flags mark
    # bloomed cosmic-ray hits in empty sky, and those must stay empty.
    _, noise = robust_background(sci, np.isfinite(sci))
    bright = 50.0 * noise if noise > 0 else np.inf
    filled = sci.copy()
    for _ in range(4):
        todo = hole & ~np.isfinite(filled)
        if not todo.any():
            break
        neighbour = ndimage.maximum_filter(np.where(np.isfinite(filled), filled, -np.inf), size=3, mode="nearest")
        fillable = todo & np.isfinite(neighbour) & (neighbour > bright)
        if not fillable.any():
            break
        filled[fillable] = neighbour[fillable]
    return filled


def reproject_exposure(
    ed: ExposureData, grid: Grid, bad_mask: int, saturation_mask: int = 0
) -> tuple[np.ndarray, np.ndarray, float]:
    """Resample one exposure onto the grid.

    Returns (image, bad_fraction, weight): ``image`` is background-subtracted
    surface brightness in MJy/sr with NaN outside the cutout; ``bad_fraction``
    is how much of each output pixel comes from flagged detector pixels;
    ``weight`` is the exposure's inverse median variance.
    """
    sci = np.array(ed.cut.sci, dtype=np.float32)
    finite = np.isfinite(sci)
    bad = ((ed.cut.flags & bad_mask) != 0) | ~finite
    background, _ = robust_background(sci, ~bad)
    sci = sci - background
    sci[~finite] = np.nan
    if saturation_mask:
        sci = fill_saturated(sci, ed.cut.flags, saturation_mask)

    ra, dec = grid.world
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x, y = ed.wcs.all_world2pix(ra, dec, 0, quiet=True)
    coords = np.array([y, x])
    image = ndimage.map_coordinates(sci, coords, order=1, mode="constant", cval=np.nan, prefilter=False)
    bad_fraction = ndimage.map_coordinates(bad.astype(np.float32), coords, order=1, mode="constant", cval=1.0, prefilter=False)

    variance = ed.cut.variance[np.isfinite(ed.cut.variance) & (ed.cut.variance > 0)]
    weight = 1.0 / float(np.median(variance)) if variance.size else 0.0
    return image.astype(np.float32), bad_fraction.astype(np.float32), weight


def coadd(
    images: list[np.ndarray], bad_fractions: list[np.ndarray], weights: list[float], use_mask: bool,
    native_pixels: float = 3.0,
):
    """Weighted mean of aligned exposures.

    native_pixels: how many output pixels span one detector pixel.

    Flagged pixels are left out. Where every exposure is flagged (typically the
    saturated core of a bright star) the unmasked mean is shown instead, so the
    star still appears; the returned flag-fraction plane records that.
    """
    cube = np.stack(images)                                   # (k, n, n)
    badf = np.stack(bad_fractions)
    w = np.asarray(weights, dtype=np.float64)
    if not np.any(w > 0):
        w = np.ones_like(w)
    positive = w[w > 0]
    w = np.clip(w, None, 5.0 * np.median(positive))           # no single exposure dominates
    w = w / w.max()

    finite = np.isfinite(cube)
    # Bilinear resampling spreads a flagged detector pixel into its neighbours, so an
    # output pixel is trusted only if no flagged pixel contributed to it at all.
    valid = finite & (badf <= BAD_TOLERANCE) if use_mask else finite

    # Reject what only one exposure sees (unflagged cosmic rays, satellite glints,
    # fast asteroids): a pixel is dropped if it differs from the median of the
    # exposures by more than 5 sigma of that exposure's noise plus the brightest
    # median value within about one detector pixel. The second term protects real
    # sources: SPHEREx undersamples its PSF, so a star's peak pixel legitimately
    # changes a lot from exposure to exposure (and with wavelength). Fast movers
    # are therefore absent from pass coadds by design; "Every exposure" mode is
    # the place to see them.
    if use_mask and cube.shape[0] >= 4:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")           # all-NaN columns are expected
            med = np.nanmedian(np.where(valid, cube, np.nan), axis=0)
        raw = np.asarray(weights, dtype=np.float64)               # 1 / median variance of each exposure
        sigma = np.where(raw > 0, 1.0 / np.sqrt(np.where(raw > 0, raw, 1.0)), np.inf)
        footprint = max(3, int(round(1.5 * native_pixels)) | 1)   # odd window ~1.5 detector pixels wide
        local_peak = ndimage.maximum_filter(np.nan_to_num(np.abs(med), nan=0.0), size=footprint, mode="nearest")
        deviation = np.abs(cube - med[None, :, :])
        limit = 5.0 * sigma[:, None, None] + 1.5 * local_peak[None, :, :]
        valid = valid & ~(deviation > limit)
    wcube = w[:, None, None] * valid
    wsum = wcube.sum(axis=0)
    w3 = w[:, None, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (np.where(valid, cube, 0.0) * w3).sum(axis=0) / wsum
        fallback = (np.where(finite, cube, 0.0) * w3).sum(axis=0) / (w3 * finite).sum(axis=0)
        out = np.where(wsum > 0, mean, fallback)
        covered = finite.sum(axis=0)
        flagged = (finite & (badf > BAD_TOLERANCE)).sum(axis=0)
        flag_fraction = np.where(covered > 0, flagged / np.maximum(covered, 1), 0.0)
    return out.astype(np.float32), flag_fraction.astype(np.float32), valid.sum(axis=0)


def _noise(image: np.ndarray) -> float:
    values = image[np.isfinite(image)]
    if values.size == 0:
        return 0.0
    med = np.median(values)
    return float(1.4826 * np.median(np.abs(values - med)))


def centre_wavelength(job: StackJob, ed: ExposureData) -> tuple[float | None, float | None]:
    """Wavelength and bandwidth (micron) this exposure recorded at the field centre."""
    table = job.wave.get(ed.exp.detector)
    if table is None:
        return None, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x, y = ed.wcs.all_world2pix(job.ra, job.dec, 0, quiet=True)
    wl, bw = table.evaluate(ed.cut.header, float(x), float(y))
    return float(wl), float(bw)


def _resample_all(job: StackJob, grid: Grid, exposures: list[ExposureData]) -> list[tuple]:
    """Resample every exposure onto the grid: (exposure, image, bad fraction, weight, wavelength, bandwidth)."""
    bad_mask = job.bad_mask
    saturation_mask = job.saturation_mask
    resampled = []
    for ed in exposures:
        image, badf, weight = reproject_exposure(ed, grid, bad_mask, saturation_mask)
        if not np.isfinite(image).any():
            continue
        wl, bw = centre_wavelength(job, ed)
        resampled.append((ed, image, badf, weight, wl, bw))
    return resampled


# Colour composites: three broad channels, each the sum of two detectors.
COLOUR_CHANNELS = (
    ("red", (5, 6), "3.8–5.0 µm"),
    ("green", (3, 4), "1.6–3.8 µm"),
    ("blue", (1, 2), "0.75–1.6 µm"),
)


def build_colour_frames(
    job: StackJob, use_mask: bool, exposures: list[ExposureData] | None = None
) -> tuple[dict, np.ndarray, np.ndarray]:
    """One three-channel frame per survey pass, for false-colour display.

    Long wavelengths are shown red, short wavelengths blue, so an object that
    is cold (bright only near 4-5 micron) stands out red among ordinary stars.
    Planes are stored pass by pass as red, green, blue.
    """
    grid = Grid(job.ra, job.dec, job.fov_arcmin)
    if exposures is None:
        exposures = job.band_exposures(None)
    resampled = _resample_all(job, grid, exposures)
    native = config.NATIVE_PIXEL_ARCSEC / grid.scale_arcsec

    def members(epoch: int, detectors: tuple[int, ...]) -> list[tuple]:
        return [item for item in resampled if item[0].epoch == epoch and item[0].exp.detector in detectors]

    epochs = sorted({item[0].epoch for item in resampled})
    # Prefer passes where every channel is well covered; fall back to any coverage at all.
    for minimum in (MIN_EXPOSURES_PER_PASS, 1):
        usable = [ep for ep in epochs if all(len(members(ep, dets)) >= minimum for _n, dets, _r in COLOUR_CHANNELS)]
        if usable:
            break

    planes: list[np.ndarray] = []
    flag_planes: list[np.ndarray] = []
    meta: list[dict] = []
    for epoch in usable:
        mjds, nexp, flag_stack = [], 0, []
        for _name, detectors, _range in COLOUR_CHANNELS:
            group = members(epoch, detectors)
            image, flag_fraction, _count = coadd(
                [g[1] for g in group], [g[2] for g in group], [g[3] for g in group], use_mask, native_pixels=native
            )
            planes.append(image)
            flag_stack.append(flag_fraction)
            mjds.extend(g[0].mjd for g in group)
            nexp += len(group)
        flag_planes.append(np.max(np.stack(flag_stack), axis=0))
        meta.append({
            "epoch": epoch, "detector": None,
            "mjd": float(np.mean(mjds)), "mjd_min": float(min(mjds)), "mjd_max": float(max(mjds)),
            "date": discovery.mjd_to_iso_date(float(np.mean(mjds))),
            "date_min": discovery.mjd_to_iso_date(min(mjds)), "date_max": discovery.mjd_to_iso_date(max(mjds)),
            "year": discovery.mjd_to_decimal_year(float(np.mean(mjds))),
            "nexp": nexp, "wl": None, "wl_min": 0.75, "wl_max": 5.0, "noise": _noise(planes[-3]),
        })

    n = grid.n
    data = np.stack(planes) if planes else np.zeros((0, n, n), dtype=np.float32)
    flags = np.stack(flag_planes) if flag_planes else np.zeros((0, n, n), dtype=np.float32)
    header = {
        "n": n, "nframes": int(data.shape[0]), "channels": 3, "nflags": int(flags.shape[0]),
        "channel_ranges": [r for _n, _d, r in COLOUR_CHANNELS],
        "ra": job.ra, "dec": job.dec, "fov_arcmin": job.fov_arcmin, "scale_arcsec": grid.scale_arcsec,
        "band": "colour", "kind": "epochs", "mask": use_mask,
        "n_exposures": len(resampled), "skipped_passes": len(epochs) - len(usable),
        "unit": "MJy/sr", "frames": meta,
    }
    return header, data.astype("<f4"), np.clip(np.round(flags * 255.0), 0, 255).astype(np.uint8)


def build_frames(
    job: StackJob, band: int | None, kind: str, use_mask: bool, exposures: list[ExposureData] | None = None
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Make the frames for one view.

    band: detector number, or None for all detectors combined.
    kind: "epochs" (one coadd per survey pass) or "exposures" (every exposure).
    exposures: the exposures to use (defaults to everything loaded for the band).
    """
    grid = Grid(job.ra, job.dec, job.fov_arcmin)
    if exposures is None:
        exposures = job.band_exposures(band)
    resampled = _resample_all(job, grid, exposures)

    frames: list[np.ndarray] = []
    flag_planes: list[np.ndarray] = []
    meta: list[dict] = []
    skipped: list[int] = []

    if kind == "exposures":
        # Single exposures are shown as observed; the flag plane says which pixels to distrust.
        for ed, image, badf, _weight, wl, bw in resampled:
            frames.append(image)
            flag_planes.append(np.where(np.isfinite(image) & (badf > BAD_TOLERANCE), 1.0, 0.0))
            meta.append({
                "epoch": ed.epoch, "detector": ed.exp.detector, "obs_id": ed.exp.obs_id,
                "mjd": ed.mjd, "mjd_min": ed.mjd, "mjd_max": ed.mjd,
                "date": discovery.mjd_to_iso_date(ed.mjd), "year": discovery.mjd_to_decimal_year(ed.mjd),
                "nexp": 1, "wl": wl, "wl_min": wl, "wl_max": wl, "bw": bw,
                "noise": _noise(image),
            })
    else:
        epochs = sorted({item[0].epoch for item in resampled})
        counts = {ep: sum(1 for item in resampled if item[0].epoch == ep) for ep in epochs}
        # A "pass" made of one or two exposures cannot reject cosmic rays or fill in
        # flagged pixels, and blinks as a burst of artefacts. Where at least two
        # passes are better covered, the thin ones are left out (they remain
        # available under "Every exposure").
        solid = [ep for ep in epochs if counts[ep] >= MIN_EXPOSURES_PER_PASS]
        skipped = [ep for ep in epochs if counts[ep] < MIN_EXPOSURES_PER_PASS] if len(solid) >= 2 else []
        for epoch in epochs:
            if epoch in skipped:
                continue
            group = [item for item in resampled if item[0].epoch == epoch]
            image, flag_fraction, _count = coadd(
                [g[1] for g in group], [g[2] for g in group], [g[3] for g in group], use_mask,
                native_pixels=config.NATIVE_PIXEL_ARCSEC / grid.scale_arcsec,
            )
            mjds = [g[0].mjd for g in group]
            wls = [g[4] for g in group if g[4] is not None]
            frames.append(image)
            flag_planes.append(flag_fraction)
            meta.append({
                "epoch": epoch, "detector": band,
                "mjd": float(np.mean(mjds)), "mjd_min": float(min(mjds)), "mjd_max": float(max(mjds)),
                "date": discovery.mjd_to_iso_date(float(np.mean(mjds))),
                "date_min": discovery.mjd_to_iso_date(min(mjds)), "date_max": discovery.mjd_to_iso_date(max(mjds)),
                "year": discovery.mjd_to_decimal_year(float(np.mean(mjds))),
                "nexp": len(group),
                "wl": float(np.mean(wls)) if wls else None,
                "wl_min": float(min(wls)) if wls else None, "wl_max": float(max(wls)) if wls else None,
                "noise": _noise(image),
            })

    n = grid.n
    data = np.stack(frames) if frames else np.zeros((0, n, n), dtype=np.float32)
    flags = np.stack(flag_planes) if flag_planes else np.zeros((0, n, n), dtype=np.float32)
    header = {
        "n": n,
        "nframes": int(data.shape[0]),
        "ra": job.ra, "dec": job.dec,
        "fov_arcmin": job.fov_arcmin,
        "scale_arcsec": grid.scale_arcsec,
        "band": band, "kind": kind, "mask": use_mask,
        "n_exposures": len(resampled),
        "skipped_passes": len(skipped),
        "unit": "MJy/sr",
        "frames": meta,
    }
    return header, data.astype("<f4"), np.clip(np.round(flags * 255.0), 0, 255).astype(np.uint8)


def pack_frames(header: dict, data: np.ndarray, flags: np.ndarray) -> bytes:
    """Binary payload: 'SPXB' | uint32 header length | JSON | padding | float32 frames | uint8 flag planes."""
    blob = json.dumps(header, allow_nan=False, default=_json_default).encode("utf-8")
    blob += b" " * ((-len(blob)) % 4)
    return MAGIC + struct.pack("<I", len(blob)) + blob + data.tobytes() + flags.tobytes()


def unpack_frames(payload: bytes) -> tuple[dict, np.ndarray, np.ndarray]:
    """Inverse of pack_frames: (header, float32 planes, uint8 flag planes)."""
    if payload[:4] != MAGIC:
        raise ValueError("not a frame payload")
    (length,) = struct.unpack("<I", payload[4:8])
    header = json.loads(payload[8:8 + length].decode("utf-8"))
    n, nframes = int(header["n"]), int(header["nframes"])
    nflags = int(header.get("nflags", nframes))
    start = 8 + length
    data = np.frombuffer(payload, dtype="<f4", count=nframes * n * n, offset=start).reshape(nframes, n, n)
    flags = np.frombuffer(payload, dtype=np.uint8, count=nflags * n * n, offset=start + 4 * nframes * n * n).reshape(nflags, n, n)
    return header, data, flags


def _json_default(value):
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    raise TypeError(type(value))
