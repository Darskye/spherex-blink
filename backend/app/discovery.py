"""Find SPHEREx spectral images that cover a sky position, and group them in time.

Discovery uses IRSA's Simple Image Access (SIA2) service, which was measured at
about one second per query (the TAP route took about fifteen).

Developed with AI assistance (Claude, Anthropic). See README "Use of AI".
"""
from __future__ import annotations

import asyncio
import io
import logging
import math
import warnings
from dataclasses import asdict, dataclass

import httpx
import numpy as np

from . import config

log = logging.getLogger("spx.discovery")


@dataclass(frozen=True)
class Exposure:
    """One SPHEREx detector exposure that covers the requested position."""

    obs_id: str
    detector: int
    url: str
    mjd: float           # mid-exposure, MJD (UTC)
    exptime: float       # seconds
    collection: str
    release: str

    @property
    def key(self) -> str:
        return f"{self.obs_id}_D{self.detector}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Exposure":
        return cls(**d)


def _parse_votable(content: bytes, collection: str) -> list[Exposure]:
    from astropy.io import votable

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = votable.parse_single_table(io.BytesIO(content)).to_table(use_names_over_ids=True)
    out: list[Exposure] = []
    if len(table) == 0:
        return out
    for row in table:
        band = str(row["energy_bandpassname"])           # e.g. "SPHEREx-D6"
        try:
            detector = int(band.strip()[-1])
        except ValueError:
            continue
        if detector not in config.DETECTORS:
            continue
        t_min, t_max = float(row["t_min"]), float(row["t_max"])
        if not (math.isfinite(t_min) and math.isfinite(t_max)):
            continue
        out.append(
            Exposure(
                obs_id=str(row["obs_id"]).strip(),
                detector=detector,
                url=str(row["access_url"]).strip(),
                mjd=0.5 * (t_min + t_max),
                exptime=float(row["t_exptime"]) if "t_exptime" in table.colnames else float("nan"),
                collection=collection,
                release=str(row["obs_release_date"]).strip() if "obs_release_date" in table.colnames else "",
            )
        )
    return out


async def _query_collection(client: httpx.AsyncClient, collection: str, ra: float, dec: float) -> list[Exposure]:
    params = {"COLLECTION": collection, "POS": f"circle {ra:.6f} {dec:.6f} {config.SIA_RADIUS_DEG}"}
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = await client.get(config.IRSA_SIA_URL, params=params, timeout=config.FETCH_TIMEOUT_S)
            if resp.status_code == 200:
                return await asyncio.to_thread(_parse_votable, resp.content, collection)
            # An unknown collection name is not fatal: the others still answer.
            if resp.status_code in (400, 404):
                log.info("SIA collection %s unavailable (HTTP %s)", collection, resp.status_code)
                return []
            last_err = RuntimeError(f"SIA HTTP {resp.status_code}")
        except (httpx.HTTPError, ValueError) as err:  # network trouble or malformed VOTable
            last_err = err
        await asyncio.sleep(0.6 * (attempt + 1))
    raise RuntimeError(f"IRSA image search failed for {collection}: {last_err}")


async def search_exposures(client: httpx.AsyncClient, ra: float, dec: float) -> list[Exposure]:
    """Return every public SPHEREx exposure whose footprint contains (ra, dec)."""
    results = await asyncio.gather(
        *(_query_collection(client, c, ra, dec) for c in config.SIA_COLLECTIONS),
        return_exceptions=True,
    )
    merged: dict[str, Exposure] = {}
    errors = []
    for res in results:
        if isinstance(res, Exception):
            errors.append(res)
            continue
        for exp in res:
            merged.setdefault(exp.key, exp)   # collections are listed in preference order
    if not merged and errors:
        raise errors[0]
    return sorted(merged.values(), key=lambda e: (e.mjd, e.detector))


def cap_exposures(exposures: list[Exposure], max_per_band: int) -> list[Exposure]:
    """Limit the number of exposures per detector, sampling evenly in time."""
    kept: list[Exposure] = []
    for det in config.DETECTORS:
        band = sorted((e for e in exposures if e.detector == det), key=lambda e: e.mjd)
        if len(band) > max_per_band:
            idx = np.unique(np.round(np.linspace(0, len(band) - 1, max_per_band)).astype(int))
            band = [band[i] for i in idx]
        kept.extend(band)
    return sorted(kept, key=lambda e: (e.mjd, e.detector))


def assign_epochs(
    mjds,
    gap_days: float = config.EPOCH_GAP_DAYS,
    max_span_days: float = config.EPOCH_MAX_SPAN_DAYS,
    split_days: float = config.EPOCH_SPLIT_DAYS,
) -> np.ndarray:
    """Label each observation time with an epoch index (0, 1, 2 ...).

    SPHEREx sweeps a typical position for a couple of weeks every six months, so
    times are first split wherever there is a long gap. Near the ecliptic poles
    coverage is nearly continuous; passes longer than ``max_span_days`` are cut
    into equal windows of roughly ``split_days`` so variability stays visible.
    """
    t = np.asarray(mjds, dtype=float)
    labels = np.zeros(t.shape, dtype=int)
    if t.size == 0:
        return labels
    order = np.argsort(t, kind="stable")
    ts = t[order]
    breaks = np.where(np.diff(ts) > gap_days)[0] + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [ts.size]))
    sorted_labels = np.zeros(ts.size, dtype=int)
    next_label = 0
    for a, b in zip(starts, ends):
        span = ts[b - 1] - ts[a]
        if span > max_span_days:
            nbins = int(math.ceil(span / split_days))
            width = span / nbins
            bins = np.minimum(((ts[a:b] - ts[a]) / width).astype(int), nbins - 1)
            for value in np.unique(bins):          # skip empty windows
                sorted_labels[a:b][bins == value] = next_label
                next_label += 1
        else:
            sorted_labels[a:b] = next_label
            next_label += 1
    labels[order] = sorted_labels
    return labels


def mjd_to_iso_date(mjd: float) -> str:
    """MJD -> 'YYYY-MM-DD' (UTC) without importing astropy.time."""
    import datetime as _dt

    return (_dt.datetime(1858, 11, 17) + _dt.timedelta(days=float(mjd))).strftime("%Y-%m-%d")


def mjd_to_decimal_year(mjd: float) -> float:
    return 2000.0 + (float(mjd) - 51544.5) / 365.25
