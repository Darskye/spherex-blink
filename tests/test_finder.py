"""Tests for the candidate finder, on made-up fields where the right answer is known.

Developed with AI assistance (Claude, Anthropic). See README "Use of AI".
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from app import config, finder, process

N = config.GRID_N
FOV = 6.0
SCALE = FOV * 60.0 / N                        # arcsec per grid pixel
PSF_PX = 7.5 / 2.3548 / SCALE                 # star width (sigma) in grid pixels
YEARS = [2025.4, 2025.9, 2026.4]
MEAN_YEAR = float(np.mean(YEARS))
COLOUR = (0.8, 1.0, 1.2)                      # an ordinary star: red, green, blue channel

# Twelve steady stars on a loose grid, well apart, plus the special cases added by each test.
STEADY = [
    {"x": 30 + 44 * (i % 4) + 5 * (i // 4), "y": 34 + 50 * (i // 4) + 7 * (i % 4), "flux": 60.0 + 25.0 * i}
    for i in range(12)
]
MOVER = {"x": 96.0, "y": 60.0, "flux": 150.0, "vx": 4.0 / SCALE, "vy": 0.0}          # 4 arcsec per year, due west
VARIABLE = {"x": 52.0, "y": 110.0, "flux": 180.0, "factors": [1.0, 1.0, 0.5]}       # half as bright in the last pass
UNLISTED = {"x": 150.0, "y": 100.0, "flux": 90.0}                                   # steady, but in no catalogue


def _blob(x: float, y: float, flux: float) -> np.ndarray:
    yy, xx = np.mgrid[0:N, 0:N]
    return flux * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * PSF_PX ** 2)) / (2 * math.pi * PSF_PX ** 2)


def _frames(stars: list[dict], noise: float = 0.01, seed: int = 11):
    """Colour frames as process.build_colour_frames makes them: three planes per pass, one flag plane."""
    rng = np.random.default_rng(seed)
    planes = []
    for p, year in enumerate(YEARS):
        dt = year - MEAN_YEAR
        for share in COLOUR:
            image = rng.normal(0.0, noise, (N, N))
            for s in stars:
                factor = s.get("factors", [1.0] * len(YEARS))[p]
                image += _blob(s["x"] + s.get("vx", 0.0) * dt, s["y"] + s.get("vy", 0.0) * dt, s["flux"] * share * factor)
            planes.append(image.astype(np.float32))
    header = {
        "n": N, "nframes": len(planes), "channels": 3, "nflags": len(YEARS), "scale_arcsec": SCALE,
        "ra": 150.0, "dec": 20.0, "fov_arcmin": FOV,
        "frames": [{"year": y, "date": f"{int(y)}-{1 + int((y % 1) * 12):02d}-15", "nexp": 30} for y in YEARS],
    }
    return header, np.stack(planes), np.zeros((len(YEARS), N, N), dtype=np.uint8)


def _nearest(field: dict, star: dict) -> int:
    return min(range(len(field["sources"])),
               key=lambda i: math.hypot(field["sources"][i]["x"] - star["x"], field["sources"][i]["y"] - star["y"]))


def _sky(x: float, y: float) -> tuple[float, float]:
    ra, dec = process.Grid(150.0, 20.0, FOV).wcs.all_pix2world(x, y, 0)
    return float(ra), float(dec)


def _catalogues(listed: list[dict], mover_known: bool, variable_flag: bool):
    """Catalogue entries for the listed stars, as the catalogue services return them."""
    gaia, twomass, allwise, simbad = [], [], [], []
    for i, s in enumerate(listed):
        ra, dec = _sky(s["x"], s["y"])
        moving = s is MOVER
        if moving and not mover_known:
            continue
        gaia.append({
            "id": str(1000 + i), "ra": ra, "dec": dec, "g": 15.0, "epoch": MEAN_YEAR,
            "pmra": -4000.0 if moving else 0.0, "pmdec": 0.0, "var": bool(variable_flag and s is VARIABLE),
        })
        magnitude = 15.0 - 2.5 * math.log10(s["flux"] / 100.0)
        twomass.append({"name": f"2MASS J{i:04d}", "ra": ra, "dec": dec, "epoch": 1999.5})
        allwise.append({"name": f"WISEA J{i:04d}", "ra": ra, "dec": dec, "w1": magnitude, "w2": magnitude, "epoch": 2010.5, "cc": "0000"})
        if moving:
            simbad.append({"name": "NAME Test Dwarf", "ra": ra, "dec": dec, "pmra": -4000.0, "pmdec": 0.0,
                           "otype": "BD*", "kind": "Brown Dwarf", "sptype": "T6", "epoch": MEAN_YEAR})
    return gaia, {"simbad": simbad, "twomass": twomass, "allwise": allwise}


def test_frame_payload_survives_a_round_trip():
    header, data, flags = _frames(STEADY[:2])
    flags[1, 5, 7] = 200
    again, data2, flags2 = process.unpack_frames(process.pack_frames(header, data.astype("<f4"), flags))
    assert again["frames"] == header["frames"] and again["scale_arcsec"] == header["scale_arcsec"]
    assert np.array_equal(data2, data) and np.array_equal(flags2, flags)


def test_straight_line_fit_recovers_a_motion_and_its_uncertainty():
    years = np.array(YEARS)
    east = -4.0 * (years - MEAN_YEAR) + np.array([0.05, -0.1, 0.05])
    north = np.array([0.02, -0.03, 0.01])
    fit = finder._fit_motion(years, east, north, np.full(3, 0.3))
    assert fit["pm_east"] == pytest.approx(-4.0, abs=0.1) and fit["pm_north"] == pytest.approx(0.0, abs=0.1)
    assert fit["err"] == pytest.approx(0.3 / math.sqrt(0.5), rel=0.01)       # sum of (t - mean)^2 is 0.5
    assert finder._fit_motion(years[:1], east[:1], north[:1], np.full(1, 0.3)) is None


def test_first_look_flags_the_mover_and_the_variable_and_nothing_else():
    header, data, flags = _frames(STEADY + [MOVER, VARIABLE])
    field = finder.measure_field(header, data, flags)
    assert len(field["sources"]) == len(STEADY) + 2
    hits = {(h["kind"], h["source"]) for h in finder.screen(field)}
    mover, variable = _nearest(field, MOVER), _nearest(field, VARIABLE)
    assert hits == {("moved", mover), ("changed", variable)}

    motion = field["sources"][mover]["motion"]
    assert motion["pm_east"] == pytest.approx(-4.0, abs=0.5)                 # west is negative east
    assert abs(motion["pm_north"]) < 0.5
    change = field["sources"][variable]["change"]
    assert change["fraction"] == pytest.approx(0.5, abs=0.06)
    assert change["faint"] == header["frames"][2]["date"]
    for i, s in enumerate(field["sources"]):
        if i not in (mover, variable):
            assert s["motion"]["total"] < 0.5 and s["change"]["fraction"] < 0.1


def _confirmed(field: dict) -> list[dict]:
    """Stand-in for the exposure-by-exposure check, which needs the single exposures."""
    hits = finder.screen(field)
    for hit in hits:
        s = field["sources"][hit["source"]]
        if hit["kind"] == "moved":
            m = s["motion"]
            hit["confirmed"] = {"total": m["total"], "err": m["err"], "pm_east": m["pm_east"], "pm_north": m["pm_north"],
                                "pa_deg": (math.degrees(math.atan2(m["pm_east"], m["pm_north"])) + 360.0) % 360.0, "baseline": m["baseline"]}
        else:
            c = s["change"]
            hit["confirmed"] = {"fraction": c["fraction"], "significance": c["significance"], "detector": 2,
                                "bright": c["bright"], "faint": c["faint"], "method": "psf"}
    return hits


def test_catalogues_separate_the_known_from_the_unexplained():
    header, data, flags = _frames(STEADY + [MOVER, VARIABLE, UNLISTED])
    field = finder.measure_field(header, data, flags)
    gaia, cats = _catalogues(STEADY + [MOVER, VARIABLE], mover_known=True, variable_flag=False)
    # A source AllWISE saw in 2010, as bright as the field's stars, where there is nothing now.
    ra, dec = _sky(140.0, 40.0)
    cats["allwise"].append({"name": "WISEA Jgone", "ra": ra, "dec": dec, "w1": 15.0, "w2": 15.0, "epoch": 2010.5, "cc": "0000"})
    report = finder.explain(field, _confirmed(field), gaia, cats)

    found = {c["kind"]: c for c in report["candidates"]}
    assert set(found) == {"moved", "changed", "uncatalogued", "gone"}
    assert found["moved"]["verdict"] == "known" and found["moved"]["known"]["name"] == "Test Dwarf"
    assert "4.00″ per year, which matches" in found["moved"]["explanation"]
    assert "west" in found["moved"]["facts"][0]
    assert found["changed"]["verdict"] == "unexplained" and "not as a variable" in found["changed"]["explanation"]
    assert found["uncatalogued"]["verdict"] == "unexplained"
    assert found["gone"]["verdict"] == "unexplained" and found["gone"]["title"] == "Here in 2010, not here now"
    assert report["counts"] == {"sources": len(STEADY) + 3, "known": 1, "unexplained": 3, "doubtful": 0}
    assert [c["verdict"] for c in report["candidates"]][:3] == ["unexplained"] * 3      # the unexplained come first
    assert not report["crowded"] and report["position_shift_arcsec"] < 0.5

    # Positions come back on the sky where the sources were put.
    ra, dec = _sky(UNLISTED["x"], UNLISTED["y"])
    assert found["uncatalogued"]["ra"] == pytest.approx(ra, abs=1 / 3600) and found["uncatalogued"]["dec"] == pytest.approx(dec, abs=1 / 3600)


def test_a_catalogued_variable_is_reported_as_already_known():
    header, data, flags = _frames(STEADY + [MOVER, VARIABLE])
    field = finder.measure_field(header, data, flags)
    gaia, cats = _catalogues(STEADY + [MOVER, VARIABLE], mover_known=True, variable_flag=True)
    report = finder.explain(field, _confirmed(field), gaia, cats)
    changed = next(c for c in report["candidates"] if c["kind"] == "changed")
    assert changed["verdict"] == "known" and "Gaia DR3 flags" in changed["explanation"]
    assert report["counts"]["unexplained"] == 0


def test_a_mover_no_catalogue_knows_is_left_unexplained():
    header, data, flags = _frames(STEADY + [MOVER])
    field = finder.measure_field(header, data, flags)
    gaia, cats = _catalogues(STEADY + [MOVER], mover_known=False, variable_flag=False)
    report = finder.explain(field, _confirmed(field), gaia, cats)
    moved = next(c for c in report["candidates"] if c["kind"] == "moved")
    assert moved["verdict"] == "unexplained" and moved["known"] is None
    assert moved["explanation"].startswith("No catalogue lists a moving object here.")


def test_missing_catalogues_are_reported_not_guessed_around():
    header, data, flags = _frames(STEADY + [UNLISTED])
    field = finder.measure_field(header, data, flags)
    report = finder.explain(field, [], None, {"simbad": None, "twomass": None, "allwise": None})
    assert report["candidates"] == []                         # nothing is called "uncatalogued" without the catalogues
    assert len(report["notes"]) == 3
    assert report["catalogues"] == {"gaia": None, "simbad": None, "twomass": None, "allwise": None}


def test_several_entries_at_one_spot_are_named_after_the_star():
    grid = process.Grid(150.0, 20.0, FOV)
    ra, dec = _sky(96.0, 96.0)
    entries = [
        {"name": "NAME Barnard's Star c", "ra": ra, "dec": dec, "otype": "Pl", "kind": "Extra-solar Planet"},
        {"name": "NAME Barnard's Star b", "ra": ra, "dec": dec, "otype": "Pl?", "kind": "Extra-solar Planet Candidate"},
        {"name": "NAME Barnard's star", "ra": ra, "dec": dec, "otype": "BY*", "kind": "BY Dra Variable"},
    ]
    catalogue = finder._Catalogue(entries, grid, MEAN_YEAR)
    chosen = finder._principal(catalogue.within(96.0, 96.0, 4.0), catalogue, 96.0, 96.0, SCALE)
    assert chosen["kind"] == "BY Dra Variable"
    assert finder._is_variable_type(chosen) and not finder._is_variable_type({"otype": "*", "kind": "Star"})


# ---- step 6: differences from the 2010 AllWISE catalogue, measured again at WISE's own wavelengths.
# The numbers are the ones this tool measured for real objects during the first sky scan (see README).


def test_brightness_in_a_wavelength_range_ignores_one_bad_exposure():
    points = [{"wl": 2.9 + 0.07 * i, "flux": 0.100 + 0.004 * (-1) ** i, "err": 0.005, "flagged": False} for i in range(12)]
    points.append({"wl": 3.3, "flux": 1.0, "err": 0.005, "flagged": False})        # a cosmic ray
    points.append({"wl": 3.3, "flux": 5.0, "err": 0.005, "flagged": True})         # flagged: never used
    points.append({"wl": 4.5, "flux": 9.0, "err": 0.005, "flagged": False})        # outside the range
    flux, err = finder._band_flux(points, 2.8, 3.8)
    assert flux == pytest.approx(0.100, abs=0.002)
    assert err == pytest.approx(0.005 / math.sqrt(12), rel=0.05)
    assert finder._band_flux(points[:4], 2.8, 3.8) is None                         # too few to trust


def test_a_red_object_is_not_called_faded_but_a_dimmed_one_is():
    # A quasar of W1 = 16.125 now measured at 57.9 micro-jansky: half of what AllWISE recorded.
    fractions = finder._fractions(
        {"w1": (0.0579, 0.0058), "w2": (0.1252, 0.0114)}, {"w1": 16.125, "w2": 14.836}, {"w1": 1.06, "w2": 1.09}
    )
    assert fractions["w1"][0] == pytest.approx(0.50, abs=0.01)
    assert finder._has_faded(fractions) == "w1"
    # A second quasar, as bright as ever at W1. The broad 1.6-3.8 micron image made it look 68% fainter.
    assert finder._has_faded({"w1": (0.88, 0.04, 26.0), "w2": (0.63, 0.06, 16.0)}) is None
    # Half as bright, but too noisy to be sure; and too faint in 2010 to say anything.
    assert finder._has_faded({"w1": (0.50, 0.12, 8.0)}) is None
    assert finder._has_faded({"w1": (0.10, 0.25, 4.0)}) is None
    # WISE 0855: the W2 light has left, a background star still shines at W1. Only a cold object may do that.
    left_behind = {"w1": (1.23, 0.06, 17.5), "w2": (0.23, 0.02, 57.5)}
    assert finder._has_faded(left_behind) is None
    assert finder._has_faded(left_behind, cold=True) == "w2"


def test_two_neighbours_of_different_colour_are_not_called_motion():
    # 6.3 arcsec apart: the 2010 position still holds as much W1 light as the present one. Nothing moved.
    assert not finder._has_moved((0.1209, 0.0045), (0.1308, 0.0045), 0.25)
    # A real mover leaves only the spill of its own light behind.
    assert finder._has_moved((0.030, 0.0045), (0.1308, 0.0045), 0.25)
    # A 3 arcsec step is a smaller difference, but still a measurable one for a bright enough source.
    assert finder._has_moved((0.100, 0.003), (0.131, 0.003), 0.75)
    assert not finder._has_moved((0.126, 0.003), (0.131, 0.003), 0.75)
    # Too faint at the present position to say.
    assert not finder._has_moved((0.002, 0.0045), (0.020, 0.0045), 0.25)


def test_catalogue_differences_are_checked_like_for_like(monkeypatch):
    step = 6.0 / 3600.0
    measured = {
        (10.0, 5.0): {"w1": (0.0579, 0.0058), "w2": (0.1252, 0.0114)},             # quasar that really faded
        (11.0, 5.0): {"w1": (0.1172, 0.0051), "w2": (0.1262, 0.0123)},             # red quasar, unchanged at W1
        (12.0, 5.0): {"w1": (0.1308, 0.0045)},                                     # blue neighbour ...
        (12.0, round(5.0 - step, 6)): {"w1": (0.1209, 0.0045)},                    # ... of a red source that stayed put
        (13.0, 5.0): {"w1": (0.280, 0.006), "w2": (1.000, 0.012)},                 # brown dwarf, where it is now ...
        (13.0, round(5.0 - step, 6)): {"w1": (0.075, 0.006), "w2": (0.260, 0.012)},    # ... and where it was
    }
    monkeypatch.setattr(finder, "_wise_like", lambda job, exposures, ra, dec: measured[(round(ra, 6), round(dec, 6))])
    monkeypatch.setattr(finder, "_yardstick", lambda field, job, exposures: ([], {}))

    def candidate(kind, ra, wise, **more):
        return {"kind": kind, "verdict": "unexplained", "ra": ra, "dec": 5.0, "strength": 10.0, "title": "first look",
                "facts": ["first look"], "caveats": [], "measured": {}, "_wise": wise, **more}

    faded = candidate("gone", 10.0, {"ra": 10.0, "dec": 5.0, "w1": 16.125, "w2": 14.836})
    red = candidate("gone", 11.0, {"ra": 11.0, "dec": 5.0, "w1": 15.92, "w2": 14.86})
    pair = candidate("drifted", 12.0, {"ra": 12.0, "dec": round(5.0 - step, 6), "w1": 16.104, "w2": 15.573})
    dwarf = candidate("drifted", 13.0, {"ra": 13.0, "dec": round(5.0 - step, 6), "w1": 15.1, "w2": 13.1})
    moved = {"kind": "moved", "verdict": "known", "ra": 14.0, "dec": 5.0, "strength": 9.0, "title": "Moved", "facts": [], "caveats": []}

    kept, checked, fell = finder.check_against_wise([faded, red, pair, dwarf, moved], {}, None, None)
    assert (checked, fell) == (4, 2)
    assert kept == [faded, dwarf, moved]
    assert all("_wise" not in c for c in kept)
    assert faded["title"] == "Only 53% as bright as in 2010" and "2.8–3.8 µm" in faded["facts"][0]
    assert faded["measured"]["band"] == "w1"
    assert dwarf["measured"]["band"] == "w2" and dwarf["measured"]["old_share"] == pytest.approx(0.26)
    assert "4.1–5.0 µm" in dwarf["facts"][-1]
