"""Turn what the user typed into sky coordinates.

Accepts decimal degrees, sexagesimal coordinates, or an object name. Names are
resolved with the CDS Sesame service (SIMBAD), which returns the J2000 position
and the proper motion. Fast-moving nearby objects have drifted arcminutes since
J2000, so resolved positions are propagated to the present before centring the
view.

Developed with AI assistance (Claude, Anthropic). See README "Use of AI".
"""
from __future__ import annotations

import math
import re
import urllib.parse
import xml.etree.ElementTree as ET

import httpx

from . import config

_DECIMAL_PAIR = re.compile(r"^\s*([+-]?\d+(?:\.\d*)?)\s*[,\s]\s*([+-]?\d+(?:\.\d*)?)\s*$")
_COORD_CHARS = re.compile(r"^[\d\s:hmsd°'\"+\-.,′″]+$", re.IGNORECASE)

# Astrometry for the five challenge targets, retrieved from SIMBAD (via Sesame)
# on 2026-11-13: ICRS position at epoch J2000 and proper motion in mas/yr. Kept
# here so the examples work even if the name service is unreachable.
BUILTIN_TARGETS = {
    "barnard's star": {
        "name": "Barnard's Star", "ra": 269.4520769586187, "dec": 4.69336496657667,
        "pmra": -801.551, "pmdec": 10362.394, "plx": 546.9759, "otype": "High proper-motion star",
        "ref": "Gaia DR3 via SIMBAD",
    },
    "eps ind b": {
        "name": "Epsilon Indi B", "ra": 331.04414658852124, "dec": -56.78284419511611,
        "pmra": 3981.977, "pmdec": -2466.832, "plx": 270.658, "otype": "Brown dwarf binary",
        "ref": "Gaia DR3 via SIMBAD",
    },
    "wise 0855": {
        "name": "WISE 0855-0714", "ra": 133.81890825, "dec": -7.247048,
        "pmra": -8123.7, "pmdec": 673.2, "plx": 439.0, "otype": "Brown dwarf",
        "ref": "Kirkpatrick et al. 2021 via SIMBAD",
    },
    "rw aur": {
        "name": "RW Aurigae", "ra": 76.95652603458, "dec": 30.40143817389,
        "pmra": 7.23, "pmdec": -20.33, "plx": 15.42, "otype": "T Tauri variable star",
        "ref": "Hipparcos (van Leeuwen 2007) via SIMBAD",
    },
    "r dra": {
        "name": "R Draconis", "ra": 248.16761083295, "dec": 66.75497941722,
        "pmra": -20.643, "pmdec": 6.149, "plx": 1.2509, "otype": "Mira variable star",
        "ref": "Gaia DR3 via SIMBAD",
    },
}
_ALIASES = {
    "barnards star": "barnard's star", "barnard star": "barnard's star", "barnard": "barnard's star",
    "name barnard's star": "barnard's star", "gj 699": "barnard's star",
    "epsilon indi b": "eps ind b", "eps indi b": "eps ind b", "* eps ind b": "eps ind b",
    "epsilon indi ba": "eps ind b", "eps ind ba": "eps ind b",
    "wise 0855-0714": "wise 0855", "wise 0855−0714": "wise 0855", "w0855": "wise 0855",
    "wise j085510.83-071442.5": "wise 0855", "wisea j085510.74-071442.5": "wise 0855",
    "wise j0855-0714": "wise 0855",
    "rw aurigae": "rw aur", "v* rw aur": "rw aur",
    "r draconis": "r dra", "v* r dra": "r dra",
}


def parse_coordinates(text: str) -> tuple[float, float] | None:
    """Parse 'RA Dec' in decimal degrees or sexagesimal (RA in hours). None if not coordinates."""
    text = text.strip()
    match = _DECIMAL_PAIR.match(text)
    if match:
        ra, dec = float(match.group(1)), float(match.group(2))
        if 0.0 <= ra <= 360.0 and -90.0 <= dec <= 90.0:
            return ra % 360.0, dec
        return None
    if not _COORD_CHARS.match(text) or len(re.findall(r"\d+", text)) < 2:
        return None
    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord

        coord = SkyCoord(text.replace("′", "'").replace("″", '"').replace("−", "-"), unit=(u.hourangle, u.deg), frame="icrs")
        return float(coord.ra.deg), float(coord.dec.deg)
    except Exception:
        return None


def propagate(ra: float, dec: float, pmra_masyr: float, pmdec_masyr: float, from_year: float, to_year: float) -> tuple[float, float]:
    """Linear proper-motion propagation (pmra is mu_alpha * cos(dec), as catalogues give it)."""
    dt = to_year - from_year
    new_dec = dec + pmdec_masyr * dt / 3.6e6
    new_ra = ra + pmra_masyr * dt / 3.6e6 / max(math.cos(math.radians(dec)), 1e-6)
    return new_ra % 360.0, max(-90.0, min(90.0, new_dec))


def _text(node, path: str) -> str | None:
    found = node.find(path)
    return found.text.strip() if found is not None and found.text else None


def parse_sesame(xml_text: str) -> dict | None:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None
    for resolver in root.iter("Resolver"):
        ra, dec = _text(resolver, "jradeg"), _text(resolver, "jdedeg")
        if ra is None or dec is None:
            continue

        def number(path: str) -> float | None:
            value = _text(resolver, path)
            try:
                return float(value) if value is not None else None
            except ValueError:
                return None

        return {
            "name": _text(resolver, "oname") or "",
            "ra": float(ra), "dec": float(dec),
            "pmra": number("pm/pmRA"), "pmdec": number("pm/pmDE"),
            "plx": number("plx/v"), "otype": _text(resolver, "otype") or "",
            "ref": "SIMBAD via CDS Sesame",
        }
    return None


async def resolve_name(client: httpx.AsyncClient, name: str) -> dict | None:
    key = re.sub(r"\s+", " ", name.strip().lower())
    key = _ALIASES.get(key, key)
    if key in BUILTIN_TARGETS:
        return dict(BUILTIN_TARGETS[key])
    url = config.SESAME_URL + "?" + urllib.parse.quote(name.strip())
    try:
        resp = await client.get(url, timeout=20.0)
    except httpx.HTTPError:
        return None
    if resp.status_code != 200:
        return None
    return parse_sesame(resp.text)


def moved_target_near(ra: float, dec: float) -> dict | None:
    """Do these coordinates match an old position of a fast-moving target we know?

    Published coordinates of nearby objects are for a past date. The challenge
    statement gives WISE 0855's position as WISE measured it around 2010; the
    object is now about two arcminutes away. If the coordinates lie on a known
    target's past track but not at its present position, say so.
    """
    best = None
    for key, target in BUILTIN_TARGETS.items():
        pm = math.hypot(target["pmra"], target["pmdec"])
        if pm < 500.0:
            continue
        closest, when = math.inf, None
        for tenth in range(19900, 20301, 5):                   # 1990.0 to 2030.0 in half-year steps
            year = tenth / 10.0
            t_ra, t_dec = propagate(target["ra"], target["dec"], target["pmra"], target["pmdec"], 2000.0, year)
            d_ra = ((t_ra - ra + 540.0) % 360.0 - 180.0) * math.cos(math.radians(dec))
            sep = math.hypot(d_ra, t_dec - dec) * 3600.0
            if sep < closest:
                closest, when = sep, year
        now_ra, now_dec = propagate(target["ra"], target["dec"], target["pmra"], target["pmdec"], 2000.0, config.VIEW_EPOCH_YEAR)
        d_ra = ((now_ra - ra + 540.0) % 360.0 - 180.0) * math.cos(math.radians(dec))
        now_sep = math.hypot(d_ra, now_dec - dec) * 3600.0
        if closest < 20.0 and now_sep > 20.0 and (best is None or closest < best["match_arcsec"]):
            best = {
                "name": target["name"], "query": key, "match_year": round(when),
                "match_arcsec": closest, "moved_arcsec": now_sep, "ra": now_ra, "dec": now_dec,
            }
    return best


async def resolve(client: httpx.AsyncClient, query: str) -> dict | None:
    """Resolve a query to a view centre. Returns None if it cannot be understood."""
    coords = parse_coordinates(query)
    if coords is not None:
        return {
            "kind": "coordinates", "query": query, "name": None,
            "ra": coords[0], "dec": coords[1],
            "ra_j2000": coords[0], "dec_j2000": coords[1],
            "pmra": None, "pmdec": None, "plx": None, "otype": None, "ref": None,
            "epoch": None, "moved_arcsec": 0.0,
            "suggest": moved_target_near(coords[0], coords[1]),
        }
    target = await resolve_name(client, query)
    if target is None:
        return None
    pmra, pmdec = target.get("pmra") or 0.0, target.get("pmdec") or 0.0
    ra, dec = propagate(target["ra"], target["dec"], pmra, pmdec, 2000.0, config.VIEW_EPOCH_YEAR)
    moved = math.hypot(pmra, pmdec) * (config.VIEW_EPOCH_YEAR - 2000.0) / 1000.0
    return {
        "kind": "name", "query": query, "name": target.get("name") or query,
        "ra": ra, "dec": dec,
        "ra_j2000": target["ra"], "dec_j2000": target["dec"],
        "pmra": target.get("pmra"), "pmdec": target.get("pmdec"), "plx": target.get("plx"),
        "otype": target.get("otype"), "ref": target.get("ref"),
        "epoch": config.VIEW_EPOCH_YEAR, "moved_arcsec": moved,
    }
