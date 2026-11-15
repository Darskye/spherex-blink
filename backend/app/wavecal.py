"""Per-pixel wavelength for SPHEREx spectral images.

SPHEREx has no filter wheel: a linear variable filter sits over each detector,
so the wavelength a source is observed at depends on where it lands on the
detector. Each file carries a 9x9 lookup table (the WCS-WAVE extension) giving
central wavelength and bandwidth across the detector; the table is the same for
every exposure of a detector that shares a calibration identifier. It is
downloaded once per detector, cached, and bilinearly interpolated here. The
interpolation was checked against ``astropy.wcs.WCS(header, fobj, key="W")`` and
agrees to the printed precision.

Developed with AI assistance (Claude, Anthropic). See README "Use of AI".
"""
from __future__ import annotations

import io
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

from . import config

log = logging.getLogger("spx.wavecal")


@dataclass
class WaveTable:
    detector: int
    calib_id: str
    x: np.ndarray        # control points along detector X (1-based FITS pixels)
    y: np.ndarray        # control points along detector Y
    values: np.ndarray   # shape (len(y), len(x), 2): [..., 0]=wavelength um, [..., 1]=bandwidth um

    def evaluate(self, header: fits.Header, x0, y0) -> tuple[np.ndarray, np.ndarray]:
        """Wavelength and bandwidth (micron) at 0-based cutout pixel(s) (x0, y0)."""
        x0 = np.asarray(x0, dtype=float)
        y0 = np.asarray(y0, dtype=float)
        # Cutout pixel -> raw detector pixel, using the file's "W" alternate WCS.
        xd = (x0 + 1.0 - float(header["CRPIX1W"])) * float(header.get("CDELT1W", 1.0)) + float(header.get("CRVAL1W", 1.0))
        yd = (y0 + 1.0 - float(header["CRPIX2W"])) * float(header.get("CDELT2W", 1.0)) + float(header.get("CRVAL2W", 1.0))
        xd = np.clip(xd, self.x[0], self.x[-1])
        yd = np.clip(yd, self.y[0], self.y[-1])
        ix = np.clip(np.searchsorted(self.x, xd, side="right") - 1, 0, len(self.x) - 2)
        iy = np.clip(np.searchsorted(self.y, yd, side="right") - 1, 0, len(self.y) - 2)
        fx = (xd - self.x[ix]) / (self.x[ix + 1] - self.x[ix])
        fy = (yd - self.y[iy]) / (self.y[iy + 1] - self.y[iy])
        v = self.values
        out = (
            v[iy, ix] * ((1 - fx) * (1 - fy))[..., None]
            + v[iy, ix + 1] * (fx * (1 - fy))[..., None]
            + v[iy + 1, ix] * ((1 - fx) * fy)[..., None]
            + v[iy + 1, ix + 1] * (fx * fy)[..., None]
        )
        return out[..., 0], out[..., 1]


def _safe(calib_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in (calib_id or "unknown"))


def _paths(detector: int, calib_id: str) -> list[Path]:
    name = f"D{detector}_{_safe(calib_id)}.npz"
    return [config.CACHE_DIR / "wavecal" / name, config.BUNDLED_DATA_DIR / "wavecal" / name]


def load_table(detector: int, calib_id: str) -> WaveTable | None:
    for path in _paths(detector, calib_id):
        if path.exists():
            try:
                with np.load(path) as z:
                    return WaveTable(detector, calib_id, z["x"].astype(float), z["y"].astype(float), z["values"].astype(float))
            except (OSError, ValueError, KeyError) as err:
                log.warning("unreadable wavelength table %s: %s", path, err)
    return None


def load_any_table(detector: int) -> WaveTable | None:
    """Fallback: any cached or bundled table for this detector, whatever its calibration id."""
    for folder in (config.CACHE_DIR / "wavecal", config.BUNDLED_DATA_DIR / "wavecal"):
        if folder.is_dir():
            for path in sorted(folder.glob(f"D{detector}_*.npz")):
                try:
                    with np.load(path) as z:
                        calib = path.stem.split("_", 1)[1]
                        return WaveTable(detector, calib, z["x"].astype(float), z["y"].astype(float), z["values"].astype(float))
                except (OSError, ValueError, KeyError):
                    continue
    return None


def table_from_fits(data: bytes, detector: int, calib_id: str) -> WaveTable:
    """Read the WCS-WAVE extension out of a complete cutout file."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fits.open(io.BytesIO(data)) as hdul:
            tab = hdul["WCS-WAVE"].data
            x = np.asarray(tab["X"][0], dtype=float)
            y = np.asarray(tab["Y"][0], dtype=float)
            values = np.asarray(tab["VALUES"][0], dtype=float).reshape(len(y), len(x), 2)
    return WaveTable(detector, calib_id, x, y, values)


def save_table(table: WaveTable) -> None:
    path = _paths(table.detector, table.calib_id)[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, x=table.x, y=table.y, values=table.values)
    tmp.replace(path)
